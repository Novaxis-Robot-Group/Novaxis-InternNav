import argparse
import html
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from flask import Flask, abort, jsonify, redirect, request, send_file, send_from_directory, url_for
from PIL import Image, ImageDraw
from demo_agent import (
    activate_demo_library,
    control_demo_agent,
    default_demo_library_path,
    delete_demo_library,
    get_demo_library,
    get_demo_state,
    load_demo_libraries,
    parse_navigation_steps,
    upsert_demo_library,
)
from experiment_analyzer_agent import (
    answer_question as answer_experiment_question,
    build_experiment_index,
    index_status as experiment_index_status,
    is_visual_question as is_experiment_visual_question,
    load_qa_history as load_experiment_qa_history,
)
from experiment_instance_analyzer import (
    build_instance_index,
    instance_index_status,
    load_instance_index,
)
from local_qwen_service import LOCAL_QWEN_MODEL, LocalQwenServiceLauncher
from realworld_service_launcher import RealworldServiceLauncher
from runtime_config import default_runtime_config_path, load_runtime_config, sanitize_runtime_config, save_runtime_config
from speech_to_text import MAX_AUDIO_BYTES, SpeechTranscriber, speech_backend_status
from upper_agent import (
    DEFAULT_UPPER_AGENT_CONFIG,
    classify_spoken_navigation_command,
    evaluate_latest as evaluate_upper_agent_latest,
    get_upper_agent_config,
    load_latest_event as load_latest_upper_agent_event,
    long_term_memory_status,
    rewrite_spoken_navigation_instruction,
    set_upper_agent_config,
)


ACTION_LABELS = {
    0: "STOP",
    1: "FWD",
    2: "LEFT",
    3: "RIGHT",
    5: "LOOK_DOWN",
}

RUN_REVIEW_FILENAME = "run_review.json"
RUN_OUTCOMES = {"", "success", "failed"}
RUN_OUTCOME_LABELS = {
    "": "未标记",
    "success": "成功",
    "failed": "失败",
}
ANALYSIS_PROGRESS_FILENAME = "experiment_analysis_progress.json"


def _analysis_progress_path(run_dir):
    return Path(run_dir).resolve() / ANALYSIS_PROGRESS_FILENAME


def write_analysis_progress(run_dir, phase, percent, message, current=0, total=0):
    """Persist lightweight progress so another Flask request can poll it."""
    payload = {
        "phase": str(phase),
        "percent": max(0, min(100, int(percent))),
        "message": str(message),
        "current": int(current or 0),
        "total": int(total or 0),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    path = _analysis_progress_path(run_dir)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False))
    os.replace(temporary, path)
    return payload


def load_analysis_progress(run_dir):
    try:
        return json.loads(_analysis_progress_path(run_dir).read_text())
    except (OSError, json.JSONDecodeError):
        return {
            "phase": "idle",
            "percent": 0,
            "message": "等待分析任务",
            "current": 0,
            "total": 0,
        }


def ensure_full_frame_instance_index(run_dir, force_detector=False):
    """Build the local full-frame detector/index stage before visual QA."""
    run_dir = Path(run_dir).resolve()
    project_root = Path(__file__).resolve().parents[2]
    detection_path = run_dir / "experiment_instance_detections.jsonl"
    detector_meta_path = run_dir / "experiment_instance_detector_meta.json"
    index_path = run_dir / "experiment_instance_index.json"
    rgb_count = sum(1 for _ in run_dir.glob("frame_*_rgb.jpg"))
    detector_meta = {}
    try:
        detector_meta = json.loads(detector_meta_path.read_text())
    except (OSError, json.JSONDecodeError):
        pass
    detector_current = (
        detection_path.is_file()
        and int(detector_meta.get("processed_frame_count", -1)) == rgb_count
        and bool(detector_meta.get("all_classes", False))
    )
    detector_ran = False
    if force_detector or not detector_current:
        write_analysis_progress(run_dir, "detecting", 3, "正在启动全帧目标检测", 0, rgb_count)
        model = Path(
            os.environ.get("INTERNNAV_INSTANCE_MODEL", str(project_root / "yolo11s.pt"))
        ).expanduser()
        command = [
            sys.executable,
            str(Path(__file__).with_name("yolo_instance_worker.py")),
            "--run-dir",
            str(run_dir),
            "--model",
            str(model),
            "--classes",
            "",
            "--confidence",
            "0.05",
            "--image-size",
            "1280",
            "--batch-size",
            "32",
            "--device",
            os.environ.get("INTERNNAV_INSTANCE_GPU", "0"),
            "--half",
        ]
        process = subprocess.Popen(
            command,
            cwd=str(project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        output_lines = []
        progress_pattern = re.compile(r"processed\s+(\d+)/(\d+)\s+frames")
        for line in process.stdout or ():
            output_lines.append(line)
            match = progress_pattern.search(line)
            if match:
                current, total = (int(value) for value in match.groups())
                percent = 5 + int(65 * current / max(1, total))
                write_analysis_progress(
                    run_dir,
                    "detecting",
                    percent,
                    f"全帧目标检测 {current}/{total}",
                    current,
                    total,
                )
        return_code = process.wait(timeout=1800)
        if return_code != 0:
            detail = ("".join(output_lines) or "Detector failed").strip()
            raise RuntimeError(detail[-2000:])
        detector_ran = True
    else:
        write_analysis_progress(run_dir, "cached", 72, "全帧检测索引已缓存，跳过重复扫描", rgb_count, rgb_count)
    index_stale = (
        not index_path.is_file()
        or index_path.stat().st_mtime_ns < detection_path.stat().st_mtime_ns
    )
    if detector_ran or index_stale:
        write_analysis_progress(run_dir, "indexing", 76, "正在构建目标实例索引", rgb_count, rgb_count)
        index = build_instance_index(
            run_dir,
            minimum_score=0.05,
            iou_threshold=0.20,
            max_frame_gap=12,
        )
    else:
        index = None
    write_analysis_progress(run_dir, "retrieving", 84, "实例索引就绪，正在检索相关证据", rgb_count, rgb_count)
    return {"detector_ran": detector_ran, "index": index or load_instance_index(run_dir)}

# experiment_visualizer.py 同时承担两类职责：
# 1. Server 侧实验记录：ExperimentLogger/save_experiment_frame 把每帧 RGB、Depth、输出、debug 保存成 run 包。
# 2. Web Viewer：create_viewer_app 提供网页/API，用于实时看图、调 runtime config、启动服务、配置 Upper Agent。
# 其中 Upper Agent 的配置和决策通过 runtime_config.json 与 upper_agent.py 互通。

def make_json_safe(value):
    # 将 numpy / Path 等对象转换成 json 可以直接保存的 Python 原生类型。
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(v) for v in value]
    return value


def frame_sort_key(path):
    # 从 frame_000001_xxx 这样的文件名中提取帧号，便于排序。
    match = re.search(r"frame_(\d+)", Path(path).name)
    return int(match.group(1)) if match else -1


def list_runs(log_dir):
    # 返回实验根目录下所有日期时间 run 包。
    root = Path(log_dir).expanduser().resolve()
    if not root.exists():
        return []
    return sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name, reverse=True)


def list_frame_metadata(run_dir):
    # 返回某次实验内所有 waypoint json 文件。
    return sorted(Path(run_dir).glob("frame_*_waypoint.json"), key=frame_sort_key)


def load_run_review(run_dir):
    """Load user-managed run metadata without touching frame-level logs."""
    review_path = Path(run_dir) / RUN_REVIEW_FILENAME
    default = {"pinned": False, "outcome": "", "updated_at": ""}
    if not review_path.exists():
        return default
    try:
        with open(review_path) as f:
            review = json.load(f)
    except (OSError, json.JSONDecodeError):
        return default
    if not isinstance(review, dict):
        return default
    outcome = str(review.get("outcome") or "").strip().lower()
    return {
        "pinned": bool(review.get("pinned", False)),
        "outcome": outcome if outcome in RUN_OUTCOMES else "",
        "updated_at": str(review.get("updated_at") or ""),
    }


def save_run_review(run_dir, review):
    """Atomically persist the optional pin/outcome annotation for one run."""
    run_dir = Path(run_dir)
    outcome = str(review.get("outcome") or "").strip().lower()
    if outcome not in RUN_OUTCOMES:
        raise ValueError("outcome must be one of: success, failed, or empty")
    saved = {
        "pinned": bool(review.get("pinned", False)),
        "outcome": outcome,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    review_path = run_dir / RUN_REVIEW_FILENAME
    temporary_path = review_path.with_suffix(".json.tmp")
    with open(temporary_path, "w") as f:
        json.dump(saved, f, indent=2, ensure_ascii=False)
    os.replace(temporary_path, review_path)
    return saved


def load_metadata(path):
    with open(path) as f:
        return json.load(f)


def run_instruction_summary(run_dir):
    """Return the latest instruction using the frame's actual agent state.

    Upper Agent priority is used only when it was enabled for that frame. Older
    runs may not contain runtime_config or agent_task_instruction, so the
    fallback chain intentionally ends at the frame's low-level instruction.
    """
    metadata_paths = list_frame_metadata(run_dir)
    if not metadata_paths:
        return {"text": "No instruction recorded", "source": "none"}
    metadata = load_metadata(metadata_paths[-1])
    runtime_config = metadata.get("runtime_config") if isinstance(metadata, dict) else {}
    runtime_config = runtime_config if isinstance(runtime_config, dict) else {}
    upper = runtime_config.get("upper_agent")
    upper = upper if isinstance(upper, dict) else {}

    if bool(upper.get("enabled", False)):
        upper_instruction = str(
            metadata.get("agent_task_instruction")
            or upper.get("task_instruction")
            or ""
        ).strip()
        if upper_instruction:
            return {"text": upper_instruction, "source": "Upper Agent"}

    instruction = str(metadata.get("instruction") or "").strip()
    if instruction:
        return {"text": instruction, "source": "Instruction"}
    return {"text": "No instruction recorded", "source": "none"}


def run_agent_mode(run_dir):
    """Return the agent participation mode captured by this experiment itself."""
    metadata_paths = list_frame_metadata(run_dir)
    if not metadata_paths:
        return {"label": "未知", "class": "agentModeUnknown", "title": "该实验没有保存帧元数据。"}
    metadata = load_metadata(metadata_paths[-1])
    config = metadata_runtime_config(metadata) if isinstance(metadata, dict) else {}
    upper = config.get("upper_agent") if isinstance(config, dict) else None
    if isinstance(upper, dict):
        if bool(upper.get("enabled", False)):
            return {"label": "Upper Agent", "class": "agentModeUpper", "title": "该实验启用了上层智能体，由其拆分子目标。"}
        return {"label": "Low-level only", "class": "agentModeLow", "title": "该实验未启用上层智能体，仅由 InternVLA-N1 低层导航。"}
    return {"label": "未知", "class": "agentModeUnknown", "title": "旧实验未保存 Upper Agent 配置快照。"}


def run_date_label(run_name):
    """Format the YYYYMMDD prefix used by run directories for the index."""
    match = re.match(r"^(\d{4})(\d{2})(\d{2})(?:_|$)", str(run_name))
    if not match:
        return "Other dates"
    year, month, day = match.groups()
    return f"{year}-{month}-{day}"


def format_agent_dialogue(metadata):
    # 将 agent_debug 中的 conversation_history 格式化成适合页面展示的纯文本。
    agent_debug = metadata.get("agent_debug") or {}
    lines = []
    llm_output = agent_debug.get("llm_output")
    if llm_output:
        lines.append(f"LLM output: {llm_output}")
        lines.append("")
    lines.append(f"episode_idx: {agent_debug.get('episode_idx', 'N/A')}")
    lines.append(f"last_s2_idx: {agent_debug.get('last_s2_idx', 'N/A')}")
    lines.append(f"last_instruction: {agent_debug.get('last_instruction', 'N/A')}")
    lines.append(f"num_rgb_history: {agent_debug.get('num_rgb_history', 'N/A')}")
    lines.append(f"num_input_images: {agent_debug.get('num_input_images', 'N/A')}")
    if agent_debug.get("save_dir"):
        lines.append(f"agent_save_dir: {agent_debug.get('save_dir')}")
    lines.append("")

    history = agent_debug.get("conversation_history") or []
    if not history:
        lines.append("No agent dialogue captured for this frame.")
        lines.append("New frames saved after this update will include agent_debug.conversation_history.")
        return "\n".join(lines)

    for idx, message in enumerate(history):
        role = message.get("role", "unknown")
        lines.append(f"[{idx}] {role}")
        content = message.get("content", [])
        if isinstance(content, list):
            for item in content:
                if item.get("type") == "text":
                    text = item.get("text", "").strip()
                    if text:
                        lines.append(text)
                elif item.get("type") == "image":
                    lines.append(item.get("image", "<image>"))
                else:
                    lines.append(json.dumps(item, ensure_ascii=False))
        else:
            lines.append(str(content))
        lines.append("")
    return "\n".join(lines).strip()


def format_timing_metrics(metadata):
    # 从响应里的 _timing 中提取各阶段耗时，便于在页面展示。
    response = metadata.get("response") or {}
    timing = response.get("_timing") or {}
    if not timing:
        return []

    display_order = [
        ("transport", "Transport"),
        ("server_total_time", "Server total"),
        ("server_core_time", "Server core"),
        ("inference_time", "Inference"),
        ("request_read_time", "HTTP read"),
        ("payload_to_bytes_time", "Zenoh payload->bytes"),
        ("payload_decode_time", "Zenoh payload decode"),
        ("image_depth_decode_time", "Image/depth decode"),
        ("request_payload_bytes", "Request payload bytes"),
        ("request_image_bytes", "Request image bytes"),
        ("request_depth_bytes", "Request depth bytes"),
    ]
    rows = []
    for key, label in display_order:
        value = timing.get(key)
        if value is None:
            continue
        if key.endswith("_time") and isinstance(value, (int, float)):
            value_str = f"{value * 1000:.2f} ms"
        else:
            value_str = str(value)
        rows.append((label, value_str))
    return rows


def render_timing_panel(metadata):
    rows = format_timing_metrics(metadata)
    if not rows:
        return "<p class='status'>No timing metrics were captured for this frame.</p>"
    cells = "".join(
        f"<tr><td>{html.escape(label)}</td><td>{html.escape(value)}</td></tr>"
        for label, value in rows
    )
    return f"""
    <div class='timingPanel'>
      <table class='timingTable'>
        <thead><tr><th>Stage</th><th>Value</th></tr></thead>
        <tbody>{cells}</tbody>
      </table>
    </div>
    """


def latest_frame_mtime(run_dir):
    metadata_paths = list_frame_metadata(run_dir)
    if not metadata_paths:
        return 0.0
    return metadata_paths[-1].stat().st_mtime


def apply_voice_interaction_command(runtime_config_path, instruction, command_type="navigate"):
    """Atomically publish one confirmed command and a one-shot model reset."""
    instruction = re.sub(r"\s+", " ", str(instruction or "")).strip()
    if not instruction:
        raise ValueError("确认后的语音指令为空。")
    current = load_runtime_config(runtime_config_path)

    upper = current.get("upper_agent") if isinstance(current.get("upper_agent"), dict) else {}
    upper = dict(upper)
    command_id = datetime.now().strftime("voice-%Y%m%d-%H%M%S-%f")
    explicit_stop = str(command_type or "").strip().lower() == "stop"
    was_stopped = bool(upper.get("voice_stop_active")) or not bool(current.get("service_enabled", True))
    if explicit_stop:
        for key in (
            "hard_reset_requested",
            "hard_reset_reason",
            "hard_reset_requested_at",
            "hard_reset_subgoal",
            "hard_reset_command_id",
        ):
            upper.pop(key, None)
        upper["voice_stop_active"] = True
        upper["replan_requested"] = False
        current["instruction"] = ""
        current["service_enabled"] = False
    else:
        upper["hard_reset_requested"] = True
        upper["hard_reset_reason"] = "confirmed_voice_interaction_command"
        upper["hard_reset_requested_at"] = datetime.now().isoformat(timespec="seconds")
        upper["hard_reset_subgoal"] = instruction
        upper["hard_reset_command_id"] = command_id
        upper["voice_stop_active"] = False
        current["instruction"] = instruction
        current["service_enabled"] = True
    if upper.get("enabled") and not explicit_stop:
        # A direct voice command becomes the active Upper-Agent task too, so a
        # later automatic evaluation cannot revive the previous spoken task.
        upper["task_instruction"] = instruction
        upper["last_task_status"] = "running"
        upper["last_subgoal"] = instruction
        upper["last_decision_at"] = datetime.now().isoformat(timespec="seconds")
        upper["replan_requested"] = False
    current["upper_agent"] = upper
    current.pop("_upper_agent_pause", None)
    current["voice_interaction"] = {
        "last_command_id": command_id,
        "last_applied_at": datetime.now().isoformat(timespec="seconds"),
        "last_command_type": "stop" if explicit_stop else "navigate",
        "last_instruction": instruction,
        "resumed_after_stop": bool(not explicit_stop and was_stopped),
    }
    reason = "explicit_stop_applied" if explicit_stop else "new_instruction_applied"
    return save_runtime_config(runtime_config_path, current), True, reason


def render_voice_input(target_id):
    """Render a microphone recorder that writes transcription into one textarea."""
    interaction_button = ""
    if target_id == "lowLevelInstruction":
        interaction_button = (
            '<button type="button" class="voiceInteractionButton" '
            'onclick="toggleVoiceInteraction(this)">进入交互模式</button>'
        )
    return f"""
      <div class="voiceInput" data-voice-target="{html.escape(target_id, quote=True)}">
        <select class="voiceDeviceSelect" aria-label="语音输入设备" title="选择电脑上的麦克风">
          <option value="">默认麦克风</option>
        </select>
        <select class="voiceLanguageSelect" aria-label="语音处理模式" title="选择语音识别与英文优化方式">
          <option value="zh_refine">中文转英文并优化</option>
          <option value="auto_refine">自动识别并优化英文</option>
          <option value="zh">仅转写中文</option>
          <option value="en">仅转写英文</option>
        </select>
        <button type="button" class="voiceRecordButton" onclick="toggleVoiceRecording(this)">开始录音</button>
        {interaction_button}
        <span class="voiceStatus">单次录音会自动应用；交互模式只执行确认后的明确指令，并持续倾听。</span>
      </div>
    """


def render_runtime_config_panel(config):
# Runtime Config 面板控制“低层大脑/服务运行时参数”。
# 这里的 instruction 是 InternVLA-N1 当前执行的短指令；
# 如果 Upper Agent 开启，它会把 current_subgoal 自动写到这个字段。
    enabled = bool(config.get("service_enabled", True))
    state_label = "RUNNING" if enabled else "STOPPED"
    state_class = "stateRun" if enabled else "stateStop"
    device = config.get("device") or "unknown"
    runtime_config_path = config.get("runtime_config_path") or ""
    selected_stt_backend = str(config.get("speech_to_text_backend", "faster-whisper"))
    selected_stt_model = str(config.get("speech_to_text_model", "small"))
    speech_status = speech_backend_status(
        selected_stt_model,
        backend=selected_stt_backend,
        device=config.get("speech_to_text_device"),
        funasr_model_path=config.get("funasr_model_path"),
        sensevoice_model_path=config.get("sensevoice_model_path"),
    )
    cached_stt_models = set(speech_status.get("cached_models") or [])
    backend_ready = speech_status.get("backend_ready") or {}
    stt_backend_labels = {
        "faster-whisper": "faster-whisper",
        "funasr-nano": "Fun-ASR-Nano (四川话/地域口音)",
        "sensevoice": "SenseVoiceSmall (快速中文)",
    }
    stt_backend_options = "".join(
        f'<option value="{name}" {"selected" if selected_stt_backend == name else ""}>'
        f'{label}{" - ready" if backend_ready.get(name) else " - setup required"}</option>'
        for name, label in stt_backend_labels.items()
    )
    stt_model_options = "".join(
        f'<option value="{name}" '
        f'{"selected" if selected_stt_model == name else ""} '
        f'{"" if name in cached_stt_models else "disabled"}>'
        f'{name}{" (cached)" if name in cached_stt_models else " (download required)"}</option>'
        for name in ("tiny", "base", "small", "medium", "large-v3")
    )
    return f"""
    <form id="runtimeConfigForm" class="configGrid">
      <input name="service_enabled" type="hidden" value="{'true' if enabled else 'false'}">
      <section class="configSection">
        <div class="configSectionHead"><span>01</span><div><strong>运行控制与指令</strong><small>控制低层策略门，并设置 InternVLA-N1 当前真正执行的导航指令。</small></div></div>
        <div class="configSectionGrid">
          <div class="serviceControls">
            <div>
              <span class="stateDot {state_class}"></span>
              <span class="stateText">{state_label}</span>
              <span class="status">policy gate, model process stays alive</span>
            </div>
            <div>
              <button type="button" class="primary" onclick="setServiceEnabled(true)">Start Policy</button>
              <button type="button" class="danger" onclick="setServiceEnabled(false)">Stop / E-Stop</button>
            </div>
          </div>
          <div class="runtimeMeta">
            <div><span>GPU</span><strong>{html.escape(str(device))}</strong></div>
            <div><span>Config file</span><code>{html.escape(str(runtime_config_path) or "not written by model server yet")}</code></div>
          </div>
          <label class="instructionField">Low-level brain instruction
            <textarea id="lowLevelInstruction" name="instruction" rows="3">{html.escape(config.get('instruction', ''))}</textarea>
            {render_voice_input('lowLevelInstruction')}
            <span class="fieldHint">低层 InternVLA-N1 当前真正执行的指令。开启 Upper Agent 后通常由 current_subgoal 自动写入；手动改这里会直接影响大脑。</span>
          </label>
        </div>
      </section>
      <section class="configSection">
        <div class="configSectionHead"><span>02</span><div><strong>导航推理与视觉输入</strong><small>调节 System2 历史上下文、规划频率、返回轨迹长度和模型输入尺寸。</small></div></div>
        <div class="configSectionGrid">
          <label>History
            <input name="num_history" type="number" min="0" max="16" value="{int(config.get('num_history', 8))}">
            <span class="fieldHint">System2 VLM 使用的历史图像数量。越大越有上下文但更慢；室内导航建议 4-8，迷路或长走廊可适当加大。</span>
          </label>
          <label>Plan gap
            <input name="plan_step_gap" type="number" min="1" max="32" value="{int(config.get('plan_step_gap', 4))}">
            <span class="fieldHint">低层大脑重新调用慢速 System2 的帧间隔。越小反应越快但推理更频繁；实机建议 3-6，转弯多可调小。</span>
          </label>
          <label>Trajectory points
            <input name="return_traj_points" type="number" min="1" max="33" value="{int(config.get('return_traj_points', 10))}">
            <span class="fieldHint">返回给机器人 client 的轨迹点数量。越多看得远但控制更复杂；MPC/连续轨迹可 10-32，保守测试可 5-10。</span>
          </label>
          <label>Resize W
            <input name="resize_w" type="number" min="224" max="768" step="32" value="{int(config.get('resize_w', 384))}">
            <span class="fieldHint">送入 System2 的图像宽度。越大细节越多但更慢；默认 384，目标较小或远距离可试 448/512。</span>
          </label>
          <label>Resize H
            <input name="resize_h" type="number" min="224" max="768" step="32" value="{int(config.get('resize_h', 384))}">
            <span class="fieldHint">送入 System2 的图像高度。通常和 Resize W 保持一致；过大增加显存和延迟。</span>
          </label>
        </div>
      </section>
      <section class="configSection">
        <div class="configSectionHead"><span>03</span><div><strong>实验记录与重规划</strong><small>控制实验帧保存密度，以及低层连续停止后触发高层重规划的条件。</small></div></div>
        <div class="configSectionGrid">
          <label>Save every N frames
            <input name="save_frame_interval" type="number" min="0" max="1000" value="{int(config.get('save_frame_interval', 1))}">
            <span class="fieldHint">实验日志保存频率。0 表示不保存，1 表示每帧保存；保存越频繁越利于可视化和 Upper Agent，但更占磁盘。</span>
          </label>
          <label>Consecutive STOP threshold
            <input name="low_level_stop_replan_threshold" type="number" min="1" max="20" value="{int(config.get('low_level_stop_replan_threshold', 3))}">
            <span class="fieldHint">低层大脑连续输出多少次 STOP 后，才触发 Upper Agent 重规划和低层硬重置。1 表示一次 STOP 就重规划；建议从 3 开始，数值越大越能过滤短暂停顿，但原地等待也会更久。</span>
          </label>
        </div>
      </section>
      <section class="configSection configSectionTask">
        <div class="configSectionHead"><span>04</span><div><strong>语音交互</strong><small>配置自动断句、指令语义门控、语音识别后端及语言优化模型。</small></div></div>
        <div class="configSectionGrid">
          <label>Voice silence seconds
            <input name="voice_silence_seconds" type="number" min="0.4" max="5" step="0.1" value="{float(config.get('voice_silence_seconds', 1.4)):.1f}">
            <span class="fieldHint">连续静音多久后判定一句话结束并自动转写。调小响应更快但可能截断停顿中的句子；调大更稳但等待更久。短指令建议 0.8-1.4 秒，长句可用 1.5-2.0 秒。</span>
          </label>
          <label>Voice command confidence
            <input name="voice_command_confidence_threshold" type="number" min="0.5" max="1" step="0.01" value="{float(config.get('voice_command_confidence_threshold', 0.78)):.2f}">
            <span class="fieldHint">语音被判定为明确机器人指令所需的最低置信度。调高可减少闲聊和噪声误触发，调低则更容易接纳指令；安静环境建议 0.75-0.82，嘈杂环境建议 0.82-0.90。该分数是语义门控分数，不是模型准确率。</span>
          </label>
          <label>Speech backend
            <select name="speech_to_text_backend">{stt_backend_options}</select>
            <span class="fieldHint">语音识别引擎。四川话优先选择 Fun-ASR-Nano；普通话低延迟可使用 SenseVoiceSmall；faster-whisper 保留为通用兼容后端。setup required 表示依赖或本地模型尚未准备好。</span>
          </label>
          <label>Speech device
            <select name="speech_to_text_device">
              <option value="cpu" {'selected' if str(config.get('speech_to_text_device', 'cpu')) == 'cpu' else ''}>cpu</option>
              <option value="cuda:0" {'selected' if str(config.get('speech_to_text_device')) == 'cuda:0' else ''}>cuda:0</option>
              <option value="cuda:1" {'selected' if str(config.get('speech_to_text_device')) == 'cuda:1' else ''}>cuda:1</option>
            </select>
            <span class="fieldHint">FunASR/SenseVoice 的运行设备。InternVLA 当前在 cuda:1，语音模型优先用空闲的 cuda:0；不确定显卡占用时使用 cpu。Whisper 仍遵循其原有环境变量设备设置。</span>
          </label>
          <label>Whisper model
            <select name="speech_to_text_model">
              {stt_model_options}
            </select>
            <span class="fieldHint">本地 faster-whisper 语音转文字模型。tiny/base 最快但识别能力较弱，small 是速度和中文准确率的折中。网页只加载已完整缓存的模型；未下载完整会立即报错，不会在交互期间联网等待。</span>
          </label>
          <label>Fun-ASR-Nano model path
            <input name="funasr_model_path" value="{html.escape(str(config.get('funasr_model_path') or 'checkpoints/Fun-ASR-Nano-2512'))}">
            <span class="fieldHint">支持四川等地域口音的本地模型目录。必须预先下载完整，网页不会在交互请求中临时下载。</span>
          </label>
          <label>SenseVoice model path
            <input name="sensevoice_model_path" value="{html.escape(str(config.get('sensevoice_model_path') or 'checkpoints/SenseVoiceSmall'))}">
            <span class="fieldHint">SenseVoiceSmall 本地模型目录。适合低延迟中文识别，但四川话专项能力弱于 Fun-ASR-Nano。</span>
          </label>
          <label>Voice language model
            <input name="voice_language_model" list="voiceLanguageModels" value="{html.escape(str(config.get('voice_language_model') or ''))}" placeholder="留空则使用 Upper Agent Model">
            <datalist id="voiceLanguageModels"><option value="local-qwen3.6-vl"><option value="qwen3-vl-flash"><option value="qwen3-vl-plus"><option value="qwen-vl-plus"></datalist>
            <span class="fieldHint">负责判断是否为明确指令、中文转英文和表达优化的视觉语言模型。选择 local-qwen3.6-vl 时使用本机 GPU 服务，不会传出 DashScope API Key。</span>
          </label>
        </div>
      </section>
      <div class="configActions">
        <button class="primary" type="submit">Apply Config</button>
        <span id="configStatus" class="status">Loaded {html.escape(config.get('updated_at', '') or 'defaults')}</span>
      </div>
    </form>
    """


def render_service_launcher_panel(status):
# 模型服务启动面板：从网页启动/停止 HTTP 或 Zenoh 服务进程。
# 注意它控制的是进程生命周期；Runtime Config 的 Start/Stop 控制的是 policy gate。
    config = status.get("config") or {}
    running = bool(status.get("running"))
    state_label = "RUNNING" if running else "STOPPED"
    state_class = "stateRun" if running else "stateStop"
    transport = config.get("transport", "http")
    mode = config.get("zenoh_mode", "")
    zenoh_listen = str(config.get("zenoh_listen", ""))
    transport_preset = "http"
    if transport == "zenoh" and zenoh_listen.startswith("udp/"):
        transport_preset = "zenoh_udp"
    elif transport == "zenoh":
        transport_preset = "zenoh_custom"
    checked_no_multicast = "checked" if config.get("zenoh_no_multicast_scouting") else ""
    checked_no_warmup = "checked" if config.get("no_warmup") else ""
    return f"""
    <form id="serviceLauncherForm" class="configGrid">
      <section class="configSection">
        <div class="configSectionHead"><span>01</span><div><strong>服务进程</strong><small>查看模型服务进程状态，并控制 HTTP 或 Zenoh 推理服务的启动与停止。</small></div></div>
        <div class="configSectionGrid">
          <div class="serviceControls">
            <div>
              <span class="stateDot {state_class}"></span>
              <span class="stateText">{state_label}</span>
              <span class="status">{html.escape(str(status.get('transport') or transport).upper())} service</span>
            </div>
            <div>
              <button type="submit" class="primary">Start Service</button>
              <button type="button" class="danger" onclick="stopModelService()">Stop Service</button>
            </div>
          </div>
          <div class="runtimeMeta">
            <div><span>PID</span><strong>{html.escape(str(status.get('pid') or 'none'))}</strong></div>
            <div><span>Log file</span><code>{html.escape(str(status.get('log_path') or 'not started'))}</code></div>
          </div>
        </div>
      </section>
      <section class="configSection">
        <div class="configSectionHead"><span>02</span><div><strong>模型与硬件</strong><small>指定推理模型目录、使用的 GPU，以及服务启动时是否执行模型预热。</small></div></div>
        <div class="configSectionGrid">
          <label>GPU device
            <input name="device" value="{html.escape(str(config.get('device', 'cuda:0')))}">
          </label>
          <label>Model path
            <input name="model_path" value="{html.escape(str(config.get('model_path', 'checkpoints/InternVLA-N1-DualVLN')))}">
          </label>
          <label class="checkField">
            <input name="no_warmup" type="checkbox" {checked_no_warmup}>
            Skip startup warmup
          </label>
        </div>
      </section>
      <section class="configSection">
        <div class="configSectionHead"><span>03</span><div><strong>传输方式与 HTTP</strong><small>选择 HTTP 或 Zenoh 传输；使用 HTTP 时在这里设置监听地址和端口。</small></div></div>
        <div class="configSectionGrid">
          <label>Transport
            <select name="transport_preset" onchange="applyTransportPreset(this.value)">
              <option value="http" {'selected' if transport_preset == 'http' else ''}>HTTP</option>
              <option value="zenoh_udp" {'selected' if transport_preset == 'zenoh_udp' else ''}>Zenoh UDP</option>
              <option value="zenoh_custom" {'selected' if transport_preset == 'zenoh_custom' else ''}>Zenoh custom</option>
            </select>
            <input name="transport" type="hidden" value="{html.escape(str(transport))}">
          </label>
          <label>HTTP host
            <input name="host" value="{html.escape(str(config.get('host', '0.0.0.0')))}">
          </label>
          <label>HTTP port
            <input name="http_port" type="number" min="1" max="65535" value="{int(config.get('http_port', 8848))}">
          </label>
        </div>
      </section>
      <section class="configSection configSectionTask">
        <div class="configSectionHead"><span>04</span><div><strong>Zenoh 通信</strong><small>配置 Zenoh RPC key、会话模式、连接端点和监听端点；仅在使用 Zenoh 时生效。</small></div></div>
        <div class="configSectionGrid">
          <label>Zenoh key
            <input name="zenoh_key" value="{html.escape(str(config.get('zenoh_key', 'internvla/eval_dual')))}">
          </label>
          <label>Zenoh mode
            <select name="zenoh_mode">
              <option value="" {'selected' if mode == '' else ''}>default</option>
              <option value="peer" {'selected' if mode == 'peer' else ''}>peer</option>
              <option value="client" {'selected' if mode == 'client' else ''}>client</option>
            </select>
          </label>
          <label class="checkField">
            <input name="zenoh_no_multicast_scouting" type="checkbox" {checked_no_multicast}>
            Disable multicast scouting
          </label>
          <label>Zenoh connect
            <textarea name="zenoh_connect" rows="2">{html.escape(str(config.get('zenoh_connect', '')))}</textarea>
          </label>
          <label>Zenoh listen
            <textarea name="zenoh_listen" rows="2">{html.escape(str(config.get('zenoh_listen', '')))}</textarea>
          </label>
        </div>
      </section>
      <div class="configActions">
        <span id="serviceStatus" class="status">Started {html.escape(str(status.get('started_at') or 'never'))}</span>
      </div>
    </form>
    """


def render_local_qwen_panel(status):
    """Controls for the separate local VLM; it never controls the robot policy."""
    config = status.get("config") or {}
    model = status.get("model") or {}
    running = bool(status.get("running"))
    ready = bool(status.get("ready"))
    label = "READY" if ready else ("LOADING" if running else "STOPPED")
    state_class = "stateRun" if ready else ("stateWait" if running else "stateStop")
    download_label = (
        f"{model.get('present_shards', 0)}/{model.get('expected_shards', 0)} shards · {model.get('size_gib', 0)} GiB"
        if model.get("expected_shards") else f"{model.get('size_gib', 0)} GiB downloaded"
    )
    return f"""
    <form id="localQwenForm" class="configGrid" onsubmit="startLocalQwen(event); return false;">
      <div class="serviceControls">
        <div>
          <span class="stateDot {state_class}"></span>
          <span class="stateText">{label}</span>
          <span class="status">Qwen3.6-VL local service · GPU {html.escape(str(config.get('gpu', '1')))}</span>
        </div>
        <div>
          <button type="submit" class="primary">Start Local Qwen</button>
          <button type="button" class="danger" onclick="stopLocalQwen()">Stop Local Qwen</button>
          <button type="button" onclick="useLocalQwenForAgents()">Use in Agents</button>
        </div>
      </div>
      <div class="runtimeMeta">
        <div><span>Weights</span><strong>{html.escape(download_label)}</strong></div>
        <div><span>PID</span><strong>{html.escape(str(status.get('pid') or 'none'))}</strong></div>
        <div><span>Log file</span><code>{html.escape(str(status.get('log_path') or 'not started'))}</code></div>
      </div>
      <label>GPU index
        <input name="gpu" value="{html.escape(str(config.get('gpu', '1')))}">
        <span class="fieldHint">物理 GPU 编号。当前固定为 1，会与 InternVLA 同卡共享显存。</span>
      </label>
      <label>Local Qwen model path
        <input name="model_path" value="{html.escape(str(config.get('model_path', '')))}">
        <span class="fieldHint">权重未完整下载前不能启动；当前只读取此目录，不修改 InternVLA checkpoints。</span>
      </label>
      <label>Port
        <input name="port" type="number" min="1" max="65535" value="{int(config.get('port', 8000))}">
      </label>
      <label>GPU memory fraction
        <input name="gpu_memory_utilization" type="number" min="0.2" max="0.85" step="0.01" value="{float(config.get('gpu_memory_utilization', 0.60))}">
        <span class="fieldHint">同卡运行 InternVLA 时建议 0.55-0.65；过高可能使导航大脑显存不足。</span>
      </label>
      <label>Context length
        <input name="max_model_len" type="number" min="1024" max="32768" step="1024" value="{int(config.get('max_model_len', 8192))}">
        <span class="fieldHint">上层导航和 QA 通常 8K 足够；更高上下文会占用更多 KV 显存。</span>
      </label>
      <label>Concurrent requests
        <input name="max_num_seqs" type="number" min="1" max="16" value="{int(config.get('max_num_seqs', 2))}">
      </label>
      <div class="configActions"><span id="localQwenStatus" class="status">Local model alias: {LOCAL_QWEN_MODEL}</span></div>
    </form>
    """


def render_task_feedback_block(event):
    output = (event or {}).get("output") or {}
    feedback = output.get("task_feedback") if isinstance(output.get("task_feedback"), dict) else {}
    report = (event or {}).get("task_report") if isinstance((event or {}).get("task_report"), dict) else {}
    summary = str(feedback.get("summary") or report.get("summary") or "").strip()
    failure_reason = str(feedback.get("failure_reason") or report.get("failure_reason") or "").strip()
    recommendation = str(feedback.get("recommendation") or report.get("recommendation") or "").strip()
    count = feedback.get("count") if feedback.get("count") is not None else report.get("count")
    count_label = str(feedback.get("count_label") or report.get("count_label") or "").strip()
    findings = report.get("findings") or feedback.get("findings") or []
    findings_html = "".join(
        f"""
        <div class="taskFinding severity-{html.escape(str(item.get('severity') or 'info'))}">
          <strong>{html.escape(str(item.get('description') or ''))}</strong>
          <span>{html.escape(str(item.get('type') or 'observation'))}{' · ' + html.escape(str(item.get('location'))) if item.get('location') else ''}</span>
          {f'<small>{html.escape(str(item.get("evidence")))}</small>' if item.get('evidence') else ''}
        </div>
        """
        for item in findings
        if isinstance(item, dict) and item.get("description")
    )
    if not any([summary, failure_reason, recommendation, count is not None, findings_html]):
        return '<div class="taskFeedbackReport"><p class="status">尚无任务反馈；智能体会在观察、计数、巡检或任务结束时更新这里。</p></div>'
    spoken_text = summary or failure_reason
    status = str(output.get("task_status") or report.get("task_status") or "running")
    count_html = ""
    if count is not None:
        count_html = f'<div class="taskFeedbackCount"><strong>{html.escape(str(count))}</strong><span>{html.escape(count_label or "count")}</span></div>'
    return f"""
      <section id="taskFeedbackReport" class="taskFeedbackReport status-{html.escape(status)}"
               data-status="{html.escape(status, quote=True)}"
               data-frame="{html.escape(str((event or {}).get('frame_idx', '')), quote=True)}"
               data-feedback="{html.escape(spoken_text, quote=True)}">
        <div class="taskFeedbackHeader">
          <div><span>Task Feedback</span><strong>{html.escape(status.upper())}</strong></div>
          <button type="button" onclick="speakTaskFeedback(this.closest('.taskFeedbackReport').dataset.feedback)">播报反馈</button>
        </div>
        {f'<p class="taskFeedbackSummary">{html.escape(summary)}</p>' if summary else ''}
        {count_html}
        {f'<div class="taskFeedbackFailure"><strong>失败原因</strong><span>{html.escape(failure_reason)}</span></div>' if failure_reason else ''}
        {f'<div class="taskFeedbackRecommendation"><strong>建议</strong><span>{html.escape(recommendation)}</span></div>' if recommendation else ''}
        {f'<div class="taskFindingGrid">{findings_html}</div>' if findings_html else ''}
      </section>
    """


def render_upper_agent_panel(config, run_name=None, latest_event=None):
# Upper Agent 面板控制“上层智能体”：
# - task_instruction：用户给智能体的完整任务。
# - auto_apply_instruction：是否把智能体输出写给低层大脑。
# - pause_policy_while_thinking：智能体调用 Qwen 时是否先让低层暂停。
# - read_every_n_frames / min_seconds_between_calls：控制智能体思考频率。
    upper = get_upper_agent_config(config)
    enabled = bool(upper.get("enabled"))
    auto_apply = bool(upper.get("auto_apply_instruction"))
    pause_while_thinking = bool(upper.get("pause_policy_while_thinking"))
    enable_route_memory = bool(upper.get("enable_route_memory"))
    enable_long_term_memory = bool(upper.get("enable_long_term_memory"))
    enable_graph_memory_capture = bool(upper.get("enable_graph_memory_capture"))
    safety_mode = bool(upper.get("safety_mode"))
    state_label = "ACTIVE" if enabled else "OFF"
    state_class = "stateRun" if enabled else "stateStop"
    checked_enabled = "checked" if enabled else ""
    checked_auto = "checked" if auto_apply else ""
    checked_pause = "checked" if pause_while_thinking else ""
    checked_memory = "checked" if enable_route_memory else ""
    checked_long_memory = "checked" if enable_long_term_memory else ""
    checked_graph_memory = "checked" if enable_graph_memory_capture else ""
    checked_safety = "checked" if safety_mode else ""
    checked_auto_speak = "checked" if upper.get("auto_speak_task_feedback", True) else ""
    guidance_style = str(upper.get("guidance_style") or "directional")
    feedback_language = str(upper.get("feedback_language") or "zh")
    key_status = "configured" if upper.get("api_key") else f"env: {upper.get('api_key_env') or 'unset'}"
    last_error = str(upper.get("last_error") or "").strip()
    memory_status = long_term_memory_status()
    memory_state = str(memory_status.get("state") or "idle")
    memory_error = str(memory_status.get("error") or "")
    latest_html = "<p class='status'>No upper-agent decision has been recorded for this run.</p>"
    if latest_event:
        output = latest_event.get("output") or {}
        memory = latest_event.get("route_memory") or {}
        demo = output.get("demo_agent") if isinstance(output.get("demo_agent"), dict) else {}
        decision = str(demo.get("decision") or output.get("demo_step_decision") or "-")
        observed_file = str(latest_event.get("image_file") or "")
        reference_file = str(latest_event.get("demo_reference_image_file") or "")
        motion_frames = latest_event.get("motion_context_frames") or []
        execution_evidence = latest_event.get("demo_execution_evidence") or {}
        assessment = output.get("execution_assessment") or {}

        def frame_file_link(filename):
            if not filename:
                return "-"
            label = html.escape(filename)
            if not run_name:
                return label
            href = url_for("file_view", run_name=run_name, filename=filename)
            return f'<a href="{html.escape(href, quote=True)}" target="_blank" rel="noopener">{label}</a>'

        demo_rows = ""
        if demo:
            demo_rows = f"""
            <tr><td>Demo library</td><td>{html.escape(str(demo.get('library_name') or ''))}</td></tr>
            <tr><td>Demo progress</td><td>{html.escape(str(demo.get('step_number') or '-'))} / {html.escape(str(demo.get('total_steps') or '-'))}</td></tr>
            """
        motion_links = " → ".join(
            f"{html.escape(str(item.get('frame_idx', '?')))}"
            for item in motion_frames
            if isinstance(item, dict)
        ) or "-"
        trajectory_evidence = "-"
        if execution_evidence:
            sampled_motion = ", ".join(
                f"{item.get('frame_idx', '?')}:{item.get('planned_motion', 'unknown')}"
                for item in execution_evidence.get("sampled_frame_outputs", [])
                if isinstance(item, dict)
            )
            trajectory_evidence = (
                f"required={execution_evidence.get('required_turn', 'none')}; "
                f"matching outputs={execution_evidence.get('matching_turn_output_count', 0)}; "
                f"turn segments={execution_evidence.get('matching_turn_segment_count', 0)}; "
                f"visual change={float(execution_evidence.get('endpoint_visual_change_score') or 0.0):.2f}; "
                f"mode={execution_evidence.get('transition_mode', 'balanced')}; "
                f"sampled=[{sampled_motion}]"
            )
        latest_html = f"""
        <table class="kvTable">
          <tbody>
            <tr><td>Agent observed frame</td><td>{html.escape(str(latest_event.get('frame_idx', 'N/A')))} · {frame_file_link(observed_file)}</td></tr>
            <tr><td>Step-start reference frame</td><td>{frame_file_link(reference_file)}</td></tr>
            <tr><td>Motion sequence frames</td><td>{motion_links}</td></tr>
            <tr><td>Trajectory evidence</td><td>{html.escape(trajectory_evidence)}</td></tr>
            <tr><td>Execution assessment</td><td>{html.escape(str(assessment.get('reason') or '-'))}</td></tr>
            <tr><td>Completion confidence</td><td>{float(assessment.get('completion_confidence') or 0.0):.2f} · completed={html.escape(str(bool(assessment.get('subgoal_completed'))).lower())} · turn={html.escape(str(assessment.get('observed_turn_direction') or 'uncertain'))} · turn_completed={html.escape(str(bool(assessment.get('turn_completed'))).lower())}</td></tr>
            <tr><td>Decision</td><td><strong>{html.escape(decision)}</strong></td></tr>
            <tr><td>Status</td><td>{html.escape(str(output.get('task_status', '')))}</td></tr>
            <tr><td>Agent task</td><td>{html.escape(str(upper.get('task_instruction', '')))}</td></tr>
            <tr><td>Phase</td><td>{html.escape(str(output.get('navigation_phase', '')))}</td></tr>
            <tr><td>Executable subgoal</td><td>{html.escape(str(output.get('current_subgoal', '')))}</td></tr>
            {demo_rows}
            <tr><td>Evidence</td><td>{html.escape(str(output.get('visual_evidence', '')))}</td></tr>
            <tr><td>Memory place</td><td>{html.escape(str(memory.get('current_place', '')))}</td></tr>
            <tr><td>Memory hint</td><td>{html.escape(str(memory.get('next_direction_hint', '')))}</td></tr>
            <tr><td>Visited</td><td>{html.escape(', '.join(str(x) for x in memory.get('visited_landmarks', [])[-5:]))}</td></tr>
            <tr><td>Long-term memory</td><td>{html.escape(str((latest_event.get('long_term_memory') or {}).get('status', {}).get('state') or memory_state))} · retrieved={len((latest_event.get('long_term_memory') or {}).get('retrieved') or [])} · retrieval={float((latest_event.get('long_term_memory') or {}).get('retrieval_ms') or 0.0):.1f}ms · write_queued={html.escape(str(bool((latest_event.get('long_term_memory') or {}).get('write_queued'))).lower())}</td></tr>
            <tr><td>Call time</td><td>{float(latest_event.get('call_time') or 0.0):.2f}s</td></tr>
          </tbody>
        </table>
        {render_task_feedback_block(latest_event)}
        """
    run_value = html.escape(str(run_name or ""))
    return f"""
    <form id="upperAgentForm" class="configGrid" onsubmit="submitUpperAgentConfig(event); return false;">
      <input name="run_name" type="hidden" value="{run_value}">
      <div class="serviceControls">
        <div>
          <span class="stateDot {state_class}"></span>
          <span class="stateText">{state_label}</span>
          <span class="status">upper agent observes saved frames, then refines low-level instructions</span>
        </div>
        <div>
          <button type="submit" class="primary">Apply Agent Config</button>
          <button type="button" onclick="runUpperAgentNow('{run_value}')">Run Once</button>
        </div>
      </div>
      <div class="runtimeMeta">
        <div><span>API key</span><strong>{html.escape(key_status)}</strong></div>
        <div><span>Latest</span><code id="upperAgentLatestLine">{html.escape(str((latest_event or {}).get('created_at') or 'not evaluated yet'))}</code>{f"<span class='upperAgentError'>Last error: {html.escape(last_error)}</span>" if last_error else ""}</div>
        <div><span>Long-term memory</span><strong>{html.escape(memory_state.upper())}</strong><code>Mem0 · CPU embedding · pending {int(memory_status.get('pending_writes') or 0)}</code>{f"<span class='upperAgentError'>{html.escape(memory_error)}</span>" if memory_error else ""}</div>
      </div>
      <section class="configSection">
        <div class="configSectionHead"><span>01</span><div><strong>运行控制与安全</strong><small>决定智能体是否接管、何时暂停低层，以及输出采用什么控制风格。</small></div></div>
        <div class="configSectionGrid">
          <label class="checkField">
            <input name="enabled" type="checkbox" {checked_enabled}>
            <span>Enable upper-level agent<span class="fieldHint">开启后，上层智能体会定期读取最新实验帧，分析任务进度并生成新的低层子目标。</span></span>
          </label>
          <label class="checkField">
            <input name="auto_apply_instruction" type="checkbox" {checked_auto}>
            <span>Auto write current_subgoal<span class="fieldHint">智能体输出的 current_subgoal 直接写入低层大脑 instruction；实机联调建议开启。</span></span>
          </label>
          <label class="checkField">
            <input name="pause_policy_while_thinking" type="checkbox" {checked_pause}>
            <span>Pause policy while thinking<span class="fieldHint">思考期间暂停旧指令，更安全但会产生停顿感。</span></span>
          </label>
          <label class="checkField">
            <input name="safety_mode" type="checkbox" {checked_safety}>
            <span>Safety mode<span class="fieldHint">转弯、墙边和盲区时使用更保守的控制提示；实机建议开启。</span></span>
          </label>
          <label class="checkField">
            <input name="auto_speak_task_feedback" type="checkbox" {checked_auto_speak}>
            <span>Auto speak final feedback<span class="fieldHint">任务完成或失败后自动播报计数、巡检结论或失败原因。</span></span>
          </label>
          <label>Guidance style
            <select name="guidance_style">
              <option value="directional" {'selected' if guidance_style == 'directional' else ''}>directional</option>
              <option value="subgoal" {'selected' if guidance_style == 'subgoal' else ''}>subgoal</option>
              <option value="cautious" {'selected' if guidance_style == 'cautious' else ''}>cautious</option>
            </select>
            <span class="fieldHint">directional 偏方向动作，subgoal 偏阶段目标，cautious 更保守。</span>
          </label>
          <label>Feedback language
            <select name="feedback_language">
              <option value="zh" {'selected' if feedback_language == 'zh' else ''}>中文</option>
              <option value="en" {'selected' if feedback_language == 'en' else ''}>English</option>
              <option value="auto" {'selected' if feedback_language == 'auto' else ''}>跟随任务语言</option>
            </select>
            <span class="fieldHint">只控制任务反馈语言，不影响发给低层大脑的英文导航指令。</span>
          </label>
        </div>
      </section>

      <section class="configSection">
        <div class="configSectionHead"><span>02</span><div><strong>模型与接口</strong><small>选择 Upper Agent 模型，并配置本地或云端 OpenAI-compatible 接口。</small></div></div>
        <div class="configSectionGrid">
          <label>Model
            <input name="model" list="upperAgentModels" value="{html.escape(str(upper.get('model') or DEFAULT_UPPER_AGENT_CONFIG['model']))}">
            <datalist id="upperAgentModels"><option value="local-qwen3.6-vl"><option value="qwen3-vl-flash"><option value="qwen3-vl-plus"><option value="qwen-vl-plus"></datalist>
            <span class="fieldHint">local-qwen3.6-vl 使用本机 GPU 1；flash/plus 为 DashScope API。</span>
          </label>
          <label>API URL
            <input name="api_url" value="{html.escape(str(upper.get('api_url') or DEFAULT_UPPER_AGENT_CONFIG['api_url']))}">
            <span class="fieldHint">模型的 OpenAI-compatible 接口地址，通常不用改。</span>
          </label>
          <label>API key env
            <input name="api_key_env" value="{html.escape(str(upper.get('api_key_env') or DEFAULT_UPPER_AGENT_CONFIG['api_key_env']))}">
            <span class="fieldHint">读取 API key 的环境变量名称。</span>
          </label>
          <label>API key
            <input name="api_key" type="password" placeholder="leave blank to keep existing key">
            <span class="fieldHint">留空保留已有 key；真实值不会回传前端或写入实验日志。</span>
          </label>
          <label>Max tokens
            <input name="max_tokens" type="number" min="64" max="4096" value="{int(upper.get('max_tokens', 512))}">
            <span class="fieldHint">简化控制 JSON 通常 256-512 足够；越大输出越慢。</span>
          </label>
          <label>Temperature
            <input name="temperature" type="number" min="0" max="2" step="0.1" value="{float(upper.get('temperature', 0.2))}">
            <span class="fieldHint">导航建议 0.0-0.3；越低越稳定。</span>
          </label>
        </div>
      </section>

      <section class="configSection">
        <div class="configSectionHead"><span>03</span><div><strong>调用节奏与重规划</strong><small>控制智能体观察频率、指令执行窗口和重新规划时机。</small></div></div>
        <div class="configSectionGrid">
          <label>Read every N saved frames
            <input name="read_every_n_frames" type="number" min="1" max="1000" value="{int(upper.get('read_every_n_frames', 5))}">
            <span class="fieldHint">越小反应越快但调用更频繁；实机可从 5-15 开始。</span>
          </label>
          <label>Min call interval seconds
            <input name="min_seconds_between_calls" type="number" min="0" max="3600" step="0.5" value="{float(upper.get('min_seconds_between_calls', 2.0))}">
            <span class="fieldHint">新 subgoal 的最短连续执行窗口；建议 8-15 秒。</span>
          </label>
          <label>Fresh-frame wait seconds
            <input name="replan_settle_seconds" type="number" min="0" max="10" step="0.1" value="{float(upper.get('replan_settle_seconds', 1.0))}">
            <span class="fieldHint">重规划前等待最新相机帧写入；建议 0.8-1.5 秒。</span>
          </label>
          <label>Max subgoal age frames
            <input name="max_subgoal_age_frames" type="number" min="10" max="10000" value="{int(upper.get('max_subgoal_age_frames', 120))}">
            <span class="fieldHint">同一子目标最长持续帧数；超时后推进阶段并触发同实验硬重置。</span>
          </label>
        </div>
      </section>

      <section class="configSection">
        <div class="configSectionHead"><span>04</span><div><strong>记忆系统</strong><small>区分单次实验工作记忆、跨实验 Mem0 长期记忆和未来图记忆数据采集。</small></div></div>
        <div class="configSectionGrid">
          <label class="checkField">
            <input name="enable_route_memory" type="checkbox" {checked_memory}>
            <span>Working route memory<span class="fieldHint">当前实验的实时工作记忆，保存为 upper_agent_memory.json。</span></span>
          </label>
          <label class="checkField">
            <input name="enable_long_term_memory" type="checkbox" {checked_long_memory}>
            <span>Cross-experiment Mem0<span class="fieldHint">检索相似场景和历史结果；超时自动跳过，不阻塞导航。</span></span>
          </label>
          <label class="checkField">
            <input name="enable_graph_memory_capture" type="checkbox" {checked_graph_memory}>
            <span>Graph-memory events<span class="fieldHint">记录“场景—地点—地标—动作—结果”，暂不参与实时控制。</span></span>
          </label>
          <label>History events
            <input name="history_events" type="number" min="0" max="30" value="{int(upper.get('history_events', 6))}">
            <span class="fieldHint">带入最近决策条数；建议 4-8。</span>
          </label>
          <label>Working memory items
            <input name="max_memory_items" type="number" min="1" max="50" value="{int(upper.get('max_memory_items', 12))}">
            <span class="fieldHint">单次实验每类路线记忆的保留上限；建议 12-20。</span>
          </label>
          <label>Long-term memory Top K
            <input name="long_term_memory_top_k" type="number" min="1" max="5" value="{int(upper.get('long_term_memory_top_k', 3))}">
            <span class="fieldHint">每次送入模型的跨实验记忆条数；建议 2-3。</span>
          </label>
          <label>Long-term character budget
            <input name="long_term_memory_char_budget" type="number" min="160" max="1200" step="40" value="{int(upper.get('long_term_memory_char_budget', 600))}">
            <span class="fieldHint">长期记忆总字符上限；建议 400-600。</span>
          </label>
          <label>Long-term timeout ms
            <input name="long_term_memory_timeout_ms" type="number" min="20" max="1000" step="10" value="{int(upper.get('long_term_memory_timeout_ms', 180))}">
            <span class="fieldHint">超时本轮跳过记忆；建议 120-200ms。</span>
          </label>
        </div>
      </section>

      <section class="configSection">
        <div class="configSectionHead"><span>05</span><div><strong>视觉输入与 Demo 判断</strong><small>调节图像尺寸、时序证据数量以及 Demo 子任务切换严格程度。</small></div></div>
        <div class="configSectionGrid">
          <label>Max image width
            <input name="max_image_width" type="number" min="224" max="1600" value="{int(upper.get('max_image_width', 768))}">
            <span class="fieldHint">越大细节越多但更慢；实机建议 512-768。</span>
          </label>
          <label>Motion context frames
            <input name="motion_context_frames" type="number" min="4" max="12" value="{int(upper.get('motion_context_frames', 8))}">
            <span class="fieldHint">Demo 执行片段的时序关键帧数；建议 8，复杂转弯可 10-12。</span>
          </label>
          <label>Demo completion confidence
            <input name="demo_completion_confidence_threshold" type="number" min="0.5" max="1" step="0.05" value="{float(upper.get('demo_completion_confidence_threshold', 0.75))}">
            <span class="fieldHint">子任务切换置信度门槛；误跳步时提高到 0.85。</span>
          </label>
          <label>Demo transition mode
            <select name="demo_transition_mode">
              <option value="balanced" {'selected' if upper.get('demo_transition_mode', 'balanced') == 'balanced' else ''}>Balanced（推荐）</option>
              <option value="strict" {'selected' if upper.get('demo_transition_mode') == 'strict' else ''}>Strict</option>
            </select>
            <span class="fieldHint">Balanced 融合时序和动作证据；Strict 要求更明确的视觉完成证据。</span>
          </label>
        </div>
      </section>

      <section class="configSection configSectionTask">
        <div class="configSectionHead"><span>06</span><div><strong>任务输入</strong><small>用户给 Upper Agent 的完整任务，智能体会拆分成低层可执行子目标。</small></div></div>
        <div class="configSectionGrid">
          <label class="instructionField">Upper Agent Task Instruction
            <textarea id="upperAgentTaskInstruction" name="task_instruction" rows="4" placeholder="Describe the full task for the upper agent. It will decompose this into short InternVLA commands.">{html.escape(str(upper.get('task_instruction') or ''))}</textarea>
            {render_voice_input('upperAgentTaskInstruction')}
            <span class="fieldHint">支持文字或语音输入；应用后由 current_subgoal 写入低层大脑。</span>
          </label>
        </div>
      </section>
      <div class="configActions">
        <span id="upperAgentStatus" class="status">Loaded {html.escape(upper.get('updated_at', '') or 'defaults')}</span>
      </div>
      <div class="instructionField" id="upperAgentLatestPanel">
        {latest_html}
      </div>
    </form>
    """


def render_demo_agent_panel(config):
    """Render deterministic, reusable navigation libraries below Upper Agent."""
    state = get_demo_state(config)
    commands = list(state.get("commands") or [])
    index = int(state.get("current_step_index") or 0) if commands else 0
    current = str(state.get("current_command") or (commands[index] if commands else ""))
    status = str(state.get("status") or "idle")
    enabled = bool(state.get("enabled"))
    step_label = f"{index + 1} / {len(commands)}" if commands else "- / -"
    return f"""
    <section class="demoAgent" id="demoAgentPanel">
      <div class="demoAgentHeader">
        <div>
          <span class="demoKicker">DETERMINISTIC SEQUENCE</span>
          <h4>Demo Agent</h4>
          <p class="status">上层模型只判断保持、前进一步、完成或失败；发给导航大脑的文字始终来自所选指令库。</p>
        </div>
        <span class="demoState {'isActive' if enabled else ''}">{html.escape(status.upper())}</span>
      </div>
      <div class="demoRuntime">
        <div><span>场景</span><strong>{html.escape(str(state.get('scene') or '未选择'))}</strong></div>
        <div><span>指令库</span><strong>{html.escape(str(state.get('library_name') or '未启动'))}</strong></div>
        <div><span>进度</span><strong>{html.escape(step_label)}</strong></div>
        <div class="demoCurrent"><span>当前低层命令</span><code>{html.escape(current or '等待启动后首帧判断')}</code></div>
      </div>
      <div class="demoToolbar">
        <label>场景筛选<select id="demoSceneFilter" onchange="filterDemoLibraries()"><option value="">全部场景</option></select></label>
        <label>已保存指令库<select id="demoLibrarySelect" onchange="selectDemoLibrary(this.value)"><option value="">选择记录</option></select></label>
        <div class="demoControlButtons">
          <button type="button" class="primary" onclick="activateSelectedDemoLibrary()">启动所选库</button>
          <button type="button" onclick="controlDemoAgent('pause')">暂停</button>
          <button type="button" onclick="controlDemoAgent('resume')">继续</button>
          <button type="button" onclick="controlDemoAgent('reset')">从第 1 条重跑</button>
          <button type="button" class="danger" onclick="controlDemoAgent('stop')">停止</button>
        </div>
      </div>
      <div id="demoLibraryRecords" class="demoLibraryRecords"><p class="status">正在读取已保存记录...</p></div>
      <form id="demoLibraryForm" class="demoEditor" onsubmit="saveDemoLibrary(event); return false;">
        <input id="demoLibraryId" name="id" type="hidden">
        <label>指令库名称<input id="demoLibraryName" name="name" maxlength="120" required placeholder="例如：A 区沙发巡航"></label>
        <label>场景<input id="demoLibraryScene" name="scene" maxlength="120" required list="demoSceneOptions" placeholder="例如：办公区 A"><datalist id="demoSceneOptions"></datalist></label>
        <label class="instructionField">备注 / 起点信息<textarea id="demoLibraryNotes" name="notes" rows="2" placeholder="例如：起点在电梯口，机器人朝向玻璃门"></textarea></label>
        <label class="instructionField">顺序导航指令库
          <textarea id="demoCommandsText" name="commands_text" rows="5" required oninput="scheduleDemoPreview()" placeholder="Go straight to the glass doors; Turn right; Continue to the sofa; Stop near the sofa"></textarea>
          <span class="fieldHint">推荐用分号、逗号或换行分隔。纯原子命令也可用空格，如“前进 左转 直行 停止”；包含空格的英文长指令请用分号、逗号或换行。</span>
        </label>
        <div class="demoPreview instructionField"><strong>解析预览</strong><ol id="demoCommandPreview"><li class="status">输入后显示执行顺序</li></ol></div>
        <div class="configActions instructionField">
          <button type="submit" class="primary">保存记录</button>
          <button type="button" onclick="newDemoLibrary()">新建</button>
          <button type="button" class="danger" onclick="deleteSelectedDemoLibrary()">删除记录</button>
          <span id="demoAgentStatus" class="status">可保存、修改并在其他实验中复用。</span>
        </div>
      </form>
    </section>
    """


def metadata_runtime_config(metadata):
# 兼容新旧实验日志：新版 metadata 顶层有 runtime_config；
# 老日志可能只在 response._timing.runtime_config 里保存了快照。
    config = metadata.get("runtime_config")
    if config:
        return config
    response = metadata.get("response") or {}
    timing = response.get("_timing") or {}
    return timing.get("runtime_config") or {}


def sanitize_metadata_for_client(metadata):
# 返回给浏览器前做脱敏，避免 api_key/token 等敏感字段进入前端或网络响应。
    metadata = make_json_safe(metadata or {})
    if isinstance(metadata, dict):
        if "runtime_config" in metadata:
            metadata["runtime_config"] = sanitize_runtime_config(metadata.get("runtime_config"))
        response = metadata.get("response")
        if isinstance(response, dict):
            timing = response.get("_timing")
            if isinstance(timing, dict) and "runtime_config" in timing:
                timing["runtime_config"] = sanitize_runtime_config(timing.get("runtime_config"))
    return metadata


def render_config_snapshot_panel(metadata):
    config = metadata_runtime_config(metadata)
    if not config:
        return "<p class='status'>No runtime config snapshot was captured for this frame.</p>"

    ordered = [
        ("service_enabled", "Policy enabled"),
        ("device", "GPU device"),
        ("runtime_config_path", "Config file"),
        ("instruction", "Low-level brain instruction"),
        ("num_history", "History"),
        ("plan_step_gap", "Plan gap"),
        ("return_traj_points", "Trajectory points"),
        ("resize_w", "Resize W"),
        ("resize_h", "Resize H"),
        ("save_frame_interval", "Save every N frames"),
        ("low_level_stop_replan_threshold", "Consecutive STOP threshold"),
        ("voice_silence_seconds", "Voice silence seconds"),
        ("voice_command_confidence_threshold", "Voice command confidence"),
        ("speech_to_text_backend", "Speech backend"),
        ("speech_to_text_model", "Speech-to-text model"),
        ("speech_to_text_device", "Speech device"),
        ("funasr_model_path", "Fun-ASR-Nano model path"),
        ("sensevoice_model_path", "SenseVoice model path"),
        ("voice_language_model", "Voice language model"),
        ("updated_at", "Config updated at"),
    ]
    rows = []
    for key, label in ordered:
        if key not in config:
            continue
        rows.append(f"<tr><td>{html.escape(label)}</td><td>{html.escape(str(config[key]))}</td></tr>")
    return f"""
    <table class="kvTable">
      <tbody>{''.join(rows)}</tbody>
    </table>
    """


def render_common_styles():
    return """
      :root { color-scheme: dark; }
      body { font-family: system-ui, sans-serif; margin: 0; background: #07110f; color: #dbe7e2; }
      body:before { content: ""; position: fixed; inset: 0; pointer-events: none; background: linear-gradient(rgba(255,255,255,.025) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,.02) 1px, transparent 1px); background-size: 28px 28px; opacity: .7; }
      .shell { max-width: 1280px; margin: 0 auto; padding: 24px; position: relative; }
      .wrap { padding: 18px; }
      .topbar { background: rgba(10, 26, 23, .92); color: white; padding: 14px 20px; display: flex; align-items: center; justify-content: space-between; gap: 14px; border-bottom: 1px solid rgba(98, 255, 194, .22); backdrop-filter: blur(12px); }
      .topbar a { color: #83f2c2; }
      .hero { background: linear-gradient(120deg, rgba(21, 52, 45, .98), rgba(6, 19, 18, .92)); color: white; padding: 22px 24px; border-radius: 8px; margin-bottom: 18px; border: 1px solid rgba(118, 255, 200, .25); box-shadow: 0 20px 60px rgba(0,0,0,.28); position: relative; overflow: hidden; min-height: 132px; }
      .hero:after { content: ""; position: absolute; inset: auto 0 0 0; height: 2px; background: linear-gradient(90deg, transparent, #5df0bd, transparent); opacity: .8; }
      .hero p { color: #b9d7cd; margin: 6px 0 0; }
      .heroGrid { display: grid; grid-template-columns: minmax(320px, 1fr) 260px; gap: 18px; align-items: center; }
      .heroMedia { height: 132px; border-radius: 8px; overflow: hidden; border: 1px solid rgba(118,255,200,.28); background: #020806; box-shadow: inset 0 0 24px rgba(98,240,173,.12); }
      .heroMedia img { width: 100%; height: 100%; object-fit: cover; opacity: .95; filter: saturate(1.16) contrast(1.08); border: 0 !important; }
      .panel { background: rgba(12, 25, 22, .86); border: 1px solid rgba(129, 255, 204, .18); padding: 14px; border-radius: 8px; box-shadow: 0 16px 48px rgba(0,0,0,.22); }
      .layout { display: grid; grid-template-columns: minmax(560px, 1fr) 450px; gap: 16px; align-items: start; }
      .runDetail .layout { grid-template-columns: minmax(0, 1.45fr) minmax(390px, 440px); }
      .stack { display: grid; gap: 16px; }
      table { border-collapse: collapse; background: rgba(10, 22, 19, .92); width: 100%; border-radius: 8px; overflow: hidden; }
      td, th { padding: 12px 14px; border-bottom: 1px solid rgba(176, 255, 220, .12); text-align: left; }
      th { background: rgba(30, 66, 56, .65); color: #bce9d5; }
      a { color: #69e8bb; text-decoration: none; }
      button, a.button { display: inline-block; padding: 8px 12px; margin: 0 8px 8px 0; border: 1px solid rgba(117, 255, 200, .32); background: rgba(12, 30, 26, .88); border-radius: 6px; color: #8bf1c8; text-decoration: none; cursor: pointer; }
      button:hover, a.button:hover { border-color: rgba(117,255,200,.75); }
      .primary { background: rgba(32, 92, 70, .95) !important; border-color: #65e1ad !important; color: #eafff6 !important; }
      .danger { background: rgba(125, 28, 39, .92) !important; border-color: #ff6578 !important; color: #fff2f4 !important; }
      .status { color: #8aa39a; font-size: 13px; }
      img { max-width: 100%; border: 1px solid #d1d5db; background: #111; }
      .mediaControls { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin-bottom: 10px; }
      .mediaScale { display: flex; align-items: center; gap: 7px; margin-left: auto; color: #a7cfbf; font-size: 12px; }
      .mediaScale input { width: 148px; accent-color: #69e8bb; }
      .mediaScale output { min-width: 34px; color: #e7fff5; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
      .mediaStage { width: var(--media-width, 100%); max-width: 100%; transition: width .16s ease; }
      #mainImage { display: block; width: 100%; height: auto; max-width: none; box-sizing: border-box; border-color: rgba(151,255,213,.35); }
      pre { white-space: pre-wrap; word-break: break-word; overflow: auto; font-size: 12px; line-height: 1.42; color: #d9eee6; }
      #dialogueText, #metadataJson { max-height: 34vh; }
      .timingTable { width: 100%; border-collapse: collapse; margin-top: 6px; }
      .timingTable th, .timingTable td { padding: 6px 8px; border-bottom: 1px solid rgba(176,255,220,.12); text-align: left; }
      .timingTable th { background: rgba(30, 66, 56, .65); }
      .kvTable { width: 100%; border-collapse: collapse; }
      .kvTable td { padding: 7px 8px; border-bottom: 1px solid rgba(176,255,220,.12); vertical-align: top; }
      .kvTable td:first-child { width: 150px; color: #9edbc2; font-weight: 600; }
      .configGrid { display: grid; grid-template-columns: repeat(3, minmax(120px, 1fr)); gap: 12px; align-items: end; }
      .configGrid label { display: grid; gap: 5px; color: #a9cfc1; font-size: 13px; }
      .configSection { grid-column: 1 / -1; padding: 18px 0 6px; border-top: 1px solid rgba(105, 232, 187, .2); }
      .configSection:first-of-type { padding-top: 8px; border-top: 0; }
      .configSectionHead { display: flex; align-items: flex-start; gap: 11px; margin-bottom: 14px; }
      .configSectionHead > span { flex: 0 0 30px; width: 30px; height: 30px; display: grid; place-items: center; border: 1px solid rgba(105, 232, 187, .36); color: #69e8bb; font: 11px ui-monospace, SFMono-Regular, Menlo, monospace; }
      .configSectionHead strong { display: block; color: #e7fff5; font-size: 15px; }
      .configSectionHead small { display: block; margin-top: 3px; color: #6f978a; font-size: 11px; line-height: 1.4; }
      .configSectionGrid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px 18px; align-items: start; }
      .configSectionGrid .instructionField { grid-column: 1 / -1; }
      .configSectionTask { padding-bottom: 0; }
      .fieldHint { display: block; color: #6f978a; font-size: 11px; line-height: 1.35; margin-top: 2px; }
      .configGrid textarea, .configGrid input, .configGrid select { width: 100%; box-sizing: border-box; border: 1px solid rgba(130, 255, 205, .28); border-radius: 6px; padding: 8px; font: inherit; background: rgba(2, 10, 8, .72); color: #e7fff5; }
      .instructionField { grid-column: 1 / -1; }
      .voiceInput { display: grid; grid-template-columns: minmax(180px, 1fr) 120px auto auto minmax(180px, 1fr); gap: 8px; align-items: center; margin-top: 3px; padding: 8px; border: 1px solid rgba(105,232,187,.16); border-radius: 6px; background: rgba(5,25,18,.45); }
      .voiceInput select { min-width: 0; padding: 7px 8px; }
      .voiceRecordButton { min-width: 94px; padding: 7px 10px; white-space: nowrap; }
      .voiceRecordButton.isRecording { color: #fff; border-color: #ff6679; background: #9c2638; box-shadow: 0 0 15px rgba(255,82,104,.25); }
      .voiceInteractionButton { min-width: 108px; padding: 7px 10px; white-space: nowrap; }
      .voiceInteractionButton.isActive { color: #06140f; border-color: #7fffd0; background: #68e7b7; box-shadow: 0 0 18px rgba(89,238,184,.3); }
      .voiceRecordButton:disabled { opacity: .58; cursor: wait; }
      .voiceStatus { min-width: 0; color: #779d8f; font-size: 11px; line-height: 1.35; overflow-wrap: anywhere; }
      .voiceStatus.isError { color: #ff9aaa; }
      .voiceStatus.isSuccess { color: #85eabd; }
      .cameraMonitorShell { min-height: 100vh; padding: 14px; box-sizing: border-box; display: grid; grid-template-rows: auto minmax(0, 1fr); gap: 12px; background: #020806; }
      .cameraMonitorBar { display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap; padding: 10px 12px; border: 1px solid rgba(105,232,187,.26); background: rgba(5,25,18,.86); }
      .cameraMonitorMeta { display: flex; align-items: center; gap: 18px; flex-wrap: wrap; color: #8fb8a8; font: 12px ui-monospace, SFMono-Regular, Menlo, monospace; }
      .cameraMonitorMeta strong { color: #e8fff6; }
      .cameraMonitorZoom { display: flex; align-items: center; gap: 6px; padding-left: 8px; border-left: 1px solid rgba(105,232,187,.2); }
      .cameraMonitorZoom button { width: 34px; height: 32px; margin: 0; padding: 0; font-size: 18px; line-height: 1; }
      .cameraMonitorZoom input { width: 130px; accent-color: #69e8bb; }
      .cameraMonitorZoom output { min-width: 42px; text-align: right; color: #dffff2; }
      .cameraMonitorViewport { position: relative; min-height: 0; display: grid; place-items: center; overflow: hidden; border: 1px solid rgba(105,232,187,.3); background: #000; }
      .cameraMonitorViewport img { width: 100%; height: 100%; object-fit: contain; border: 0; background: #000; transform: scale(var(--camera-zoom, 1)); transform-origin: center; transition: transform .16s ease; }
      .cameraMonitorWaiting { position: absolute; inset: 0; display: grid; place-items: center; pointer-events: none; color: #6f978a; font: 13px ui-monospace, SFMono-Regular, Menlo, monospace; background: radial-gradient(circle at center, rgba(17,56,43,.26), transparent 55%); }
      .cameraMonitorViewport.hasFrame .cameraMonitorWaiting { display: none; }
      .monitorOnline { color: #71f4bf !important; }
      .monitorOffline { color: #ff8191 !important; }
      .checkField { display: flex !important; grid-template-columns: none !important; align-items: center; gap: 8px !important; min-height: 38px; }
      .checkField input { width: auto !important; }
      .configActions { grid-column: 1 / -1; display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
      .serviceControls { grid-column: 1 / -1; display: flex; justify-content: space-between; align-items: center; gap: 12px; padding: 10px 12px; border: 1px solid rgba(129,255,204,.18); border-radius: 8px; background: rgba(2,10,8,.38); }
      .playbackControls { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin-top: 10px; }
      .playbackControls label { display: flex; align-items: center; gap: 6px; color: #a9cfc1; font-size: 13px; }
      .playbackControls select { width: auto; border: 1px solid rgba(130,255,205,.28); border-radius: 6px; padding: 7px; background: rgba(2,10,8,.72); color: #e7fff5; }
      .experimentQa { display: grid; gap: 12px; }
      .experimentQaHead { display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap; }
      .experimentQaStatus { display: flex; gap: 8px; flex-wrap: wrap; color: #8fb8a8; font-size: 12px; }
      .experimentQaStatus span, .qaSource { padding: 4px 7px; border: 1px solid rgba(105,232,187,.2); border-radius: 4px; background: rgba(4,24,18,.55); }
      .experimentQa textarea { width: 100%; min-height: 74px; box-sizing: border-box; resize: vertical; padding: 10px; border: 1px solid rgba(130,255,205,.3); border-radius: 6px; background: rgba(2,10,8,.72); color: #e7fff5; font: inherit; }
      .homeQaSelector { display: grid; grid-template-columns: auto minmax(240px, 1fr) auto; gap: 10px; align-items: center; }
      .homeQaSelector label { color: #9fd3bf; font-size: 13px; }
      .homeQaSelector select { width: 100%; min-width: 0; box-sizing: border-box; padding: 9px 10px; border: 1px solid rgba(130,255,205,.3); border-radius: 6px; background: rgba(2,10,8,.72); color: #e7fff5; }
      .homeQaSelector button { margin: 0; }
      .homeQaQuestionRow { display: grid; grid-template-columns: minmax(0, 1fr) 210px auto; gap: 10px; align-items: end; }
      .homeQaQuestionRow label { display: grid; gap: 5px; color: #9fd3bf; font-size: 12px; }
      .homeQaQuestionRow input { width: 100%; box-sizing: border-box; padding: 9px 10px; border: 1px solid rgba(130,255,205,.3); border-radius: 6px; background: rgba(2,10,8,.72); color: #e7fff5; }
      .homeQaQuestionRow textarea { min-height: 82px; }
      .qaAnswer { display: none; padding-top: 12px; border-top: 1px solid rgba(105,232,187,.16); }
      .qaAnswer.isVisible { display: block; }
      .qaAnswerText { margin: 5px 0 10px; color: #eafff6; font-size: 16px; line-height: 1.6; }
      .qaMeta { display: flex; gap: 12px; flex-wrap: wrap; color: #91b9aa; font-size: 12px; }
      .qaProgress { display: grid; grid-template-columns: minmax(160px, 1fr) auto; gap: 9px; align-items: center; padding: 9px 10px; border: 1px solid rgba(105,232,187,.18); border-radius: 6px; background: rgba(2,14,10,.62); }
      .qaProgress[hidden] { display: none; }
      .qaProgressTrack { height: 7px; overflow: hidden; border-radius: 4px; background: rgba(123,177,156,.16); }
      .qaProgressFill { display: block; width: 0; height: 100%; border-radius: inherit; background: #58e8b3; box-shadow: 0 0 12px rgba(88,232,179,.5); transition: width .25s ease; }
      .qaProgressText { color: #9ccabb; font: 11px ui-monospace, SFMono-Regular, Menlo, monospace; white-space: nowrap; }
      .qaEvidenceGrid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 10px; margin-top: 12px; }
      .qaEvidenceItem { display: block; color: #c8eee0; border: 1px solid rgba(105,232,187,.18); border-radius: 6px; overflow: hidden; background: rgba(2,12,9,.64); }
      .qaEvidenceItem img { display: block; width: 100%; aspect-ratio: 4 / 3; object-fit: cover; border: 0; }
      .qaEvidenceItem span { display: block; padding: 8px; font-size: 11px; line-height: 1.4; }
      .qaHistory { display: grid; gap: 8px; max-height: 320px; overflow: auto; }
      .qaHistoryItem { padding: 9px 10px; border-left: 2px solid rgba(105,232,187,.45); background: rgba(3,17,13,.46); }
      .qaHistoryItem b { display: block; color: #aef3d3; margin-bottom: 4px; }
      .qaHistoryItem p { margin: 0; color: #bad5ca; line-height: 1.45; }
      .runtimeMeta { grid-column: 1 / -1; display: grid; grid-template-columns: 120px minmax(0, 1fr); gap: 10px; padding: 10px 12px; border: 1px solid rgba(129,255,204,.13); border-radius: 8px; background: rgba(2,10,8,.25); }
      .runtimeMeta div { min-width: 0; }
      .runtimeMeta span { display: block; color: #7fac9b; font-size: 12px; margin-bottom: 3px; }
      .runtimeMeta strong { color: #eafff6; font-size: 15px; }
      .runtimeMeta code { display: block; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: #bce9d5; }
      .agentDecision { display: grid; gap: 8px; }
      .agentDecision strong { color: #eafff6; }
      .agentDecision .instruction { color: #75ffd0; font-weight: 700; }
      .upperAgentError { display: block; margin-top: 5px; color: #ff9aa8; font-size: 12px; white-space: normal; overflow-wrap: anywhere; }
      .taskFeedbackReport { grid-column: 1 / -1; margin-top: 14px; padding: 14px; border: 1px solid rgba(98,240,189,.28); background: rgba(3,19,15,.78); border-radius: 8px; }
      .taskFeedbackReport.status-completed { border-color: rgba(91,255,181,.6); box-shadow: inset 3px 0 #35e7a1; }
      .taskFeedbackReport.status-failed { border-color: rgba(255,94,119,.62); box-shadow: inset 3px 0 #ff5e77; }
      .taskFeedbackHeader { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
      .taskFeedbackHeader > div { display: flex; align-items: baseline; gap: 10px; }
      .taskFeedbackHeader span { color: #8fd9bd; font-size: 12px; text-transform: uppercase; }
      .taskFeedbackHeader strong { color: #eafff7; letter-spacing: 0; }
      .taskFeedbackSummary { margin: 12px 0; font-size: 17px; line-height: 1.5; color: #effff9; }
      .taskFeedbackCount { display: inline-flex; align-items: baseline; gap: 9px; margin: 4px 0 12px; padding: 8px 12px; border: 1px solid rgba(83,242,185,.32); background: rgba(25,91,69,.24); border-radius: 6px; }
      .taskFeedbackCount strong { font-size: 28px; color: #61f0ba; }
      .taskFeedbackCount span { color: #b8dacf; }
      .taskFeedbackFailure, .taskFeedbackRecommendation { display: grid; grid-template-columns: 72px 1fr; gap: 10px; padding: 9px 0; border-top: 1px solid rgba(176,255,220,.1); }
      .taskFeedbackFailure strong { color: #ff91a3; }
      .taskFeedbackRecommendation strong { color: #82dfbd; }
      .taskFindingGrid { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 8px; margin-top: 10px; }
      .taskFinding { display: flex; flex-direction: column; gap: 4px; padding: 10px; background: rgba(10,32,26,.86); border: 1px solid rgba(139,231,197,.16); border-radius: 6px; }
      .taskFinding.severity-warning { border-color: rgba(255,196,85,.5); }
      .taskFinding.severity-critical { border-color: rgba(255,89,116,.64); }
      .taskFinding span, .taskFinding small { color: #8eb4a7; line-height: 1.4; }
      .demoAgent { margin-top: 18px; padding-top: 18px; border-top: 1px solid rgba(105,232,187,.22); }
      .demoAgentHeader { display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; margin-bottom: 14px; }
      .demoAgentHeader h4 { margin: 4px 0; font-size: 20px; color: #effff8; }
      .demoKicker { color: #69e8bb; font: 10px ui-monospace, SFMono-Regular, Menlo, monospace; letter-spacing: .14em; }
      .demoState { padding: 5px 9px; border: 1px solid rgba(145,171,161,.35); border-radius: 999px; color: #91a9a0; font: 700 11px ui-monospace, SFMono-Regular, Menlo, monospace; }
      .demoState.isActive { color: #70f1bd; border-color: rgba(85,237,179,.55); background: rgba(21,94,68,.24); box-shadow: 0 0 16px rgba(68,231,170,.12); }
      .demoRuntime { display: grid; grid-template-columns: 1fr 1fr 90px; gap: 10px; padding: 11px 12px; border: 1px solid rgba(129,255,204,.14); background: rgba(2,10,8,.38); }
      .demoRuntime div { min-width: 0; }
      .demoRuntime span { display: block; margin-bottom: 4px; color: #759e8f; font-size: 11px; }
      .demoRuntime strong { display: block; color: #e8fff6; overflow-wrap: anywhere; }
      .demoRuntime .demoCurrent { grid-column: 1 / -1; border-top: 1px solid rgba(129,255,204,.11); padding-top: 9px; }
      .demoRuntime code { color: #7ff0c4; white-space: normal; overflow-wrap: anywhere; }
      .demoToolbar { display: grid; grid-template-columns: minmax(160px, .65fr) minmax(220px, 1fr); gap: 10px; align-items: end; margin: 12px 0; }
      .demoToolbar label, .demoEditor label { display: grid; gap: 5px; color: #a9cfc1; font-size: 13px; }
      .demoToolbar select, .demoEditor input, .demoEditor textarea { width: 100%; box-sizing: border-box; border: 1px solid rgba(130,255,205,.28); border-radius: 6px; padding: 8px; font: inherit; background: rgba(2,10,8,.72); color: #e7fff5; }
      .demoControlButtons { grid-column: 1 / -1; display: flex; flex-wrap: wrap; gap: 6px; }
      .demoLibraryRecords { display: grid; gap: 7px; margin: 10px 0 12px; }
      .demoLibraryRow { display: grid; grid-template-columns: minmax(130px,.8fr) minmax(120px,.6fr) 70px minmax(180px,1.3fr) auto; gap: 9px; align-items: center; padding: 8px 10px; border-bottom: 1px solid rgba(129,255,204,.12); background: rgba(4,17,14,.4); }
      .demoLibraryRow strong { color: #e7fff5; }
      .demoLibraryRow span, .demoLibraryRow small { color: #83aa9b; overflow-wrap: anywhere; }
      .demoLibraryRow button { margin: 0; padding: 6px 9px; }
      .demoEditor { display: grid; grid-template-columns: 1fr 1fr; gap: 11px; padding: 12px; border-left: 2px solid rgba(105,232,187,.55); background: rgba(5,21,17,.48); }
      .demoPreview { padding: 10px; border: 1px dashed rgba(105,232,187,.22); }
      .demoPreview ol { margin: 8px 0 0; padding-left: 24px; color: #d7eee5; }
      .demoPreview li { margin: 5px 0; line-height: 1.4; }
      .runInstruction { max-width: 520px; color: #bce9d5; }
      .runInstruction summary { max-width: 520px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; cursor: pointer; }
      .runInstruction summary::marker { color: #69e8bb; }
      .runInstructionFull { margin-top: 8px; padding: 8px 10px; border-left: 2px solid #69e8bb; color: #e7fff5; line-height: 1.45; white-space: normal; overflow-wrap: anywhere; background: rgba(2, 10, 8, .5); }
      .runInstructionSource { display: block; color: #6f978a; font-size: 11px; margin-top: 3px; }
      .deleteButton { padding: 6px 9px !important; margin: 0 !important; color: #ffb7c0 !important; border-color: rgba(255,101,120,.5) !important; }
      .pinButton { padding: 6px 9px !important; margin: 0 !important; }
      .reviewSelect { width: 100%; min-width: 92px; box-sizing: border-box; border: 1px solid rgba(130,255,205,.28); border-radius: 6px; padding: 6px 8px; font: inherit; background: rgba(2,10,8,.72); color: #e7fff5; }
      .outcomeBadge { display: inline-block; margin-bottom: 6px; padding: 3px 7px; border-radius: 999px; border: 1px solid rgba(176,255,220,.2); font-size: 11px; font-weight: 700; }
      .outcomeSuccess { color: #9ff5c9; border-color: rgba(87,232,154,.55); background: rgba(30,115,74,.26); }
      .outcomeFailed { color: #ffc2ca; border-color: rgba(255,101,120,.55); background: rgba(125,28,39,.3); }
      .outcomeUnset { color: #9ab4aa; }
      .agentMode { display: inline-block; padding: 4px 7px; border-radius: 999px; border: 1px solid rgba(176,255,220,.2); font-size: 11px; font-weight: 700; white-space: nowrap; }
      .agentModeUpper { color: #7feaff; border-color: rgba(88,216,255,.55); background: rgba(14,86,112,.35); }
      .agentModeLow { color: #b4e7be; border-color: rgba(112,207,134,.48); background: rgba(27,92,45,.30); }
      .agentModeUnknown { color: #b7c4be; border-color: rgba(174,187,180,.32); background: rgba(83,93,87,.25); }
      .pinnedRow { background: rgba(38, 90, 70, .16); }
      .pinnedDivider td { padding: 10px 14px 7px; border-bottom: 1px solid rgba(255,210,103,.38); background: rgba(92,74,24,.42); color: #ffe39a; font-size: 13px; font-weight: 700; letter-spacing: .08em; }
      .runSelect { width: 16px; height: 16px; accent-color: #69e8bb; }
      .tableActions { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin: 0 0 10px; }
      .dateDivider td { padding: 10px 14px 7px; border-bottom: 1px solid rgba(117,255,200,.28); background: rgba(20, 67, 54, .58); color: #8bf1c8; font-size: 13px; font-weight: 700; letter-spacing: .08em; }
      .stateDot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; margin-right: 8px; box-shadow: 0 0 14px currentColor; }
      .stateRun { color: #62f0ad; background: #62f0ad; }
      .stateStop { color: #ff5268; background: #ff5268; }
      .stateText { font-weight: 700; letter-spacing: .08em; margin-right: 10px; }
      input[type=range] { width: 100%; }
      .emptyState { min-height: 52vh; display: grid; place-items: center; text-align: center; }
      .pulse { width: 12px; height: 12px; border-radius: 50%; background: #2f9d6a; display: inline-block; margin-right: 8px; animation: pulse 1.3s infinite; }
      @keyframes pulse { 0% { opacity: .35; transform: scale(.9); } 50% { opacity: 1; transform: scale(1.12); } 100% { opacity: .35; transform: scale(.9); } }
      .homeIntro { position: fixed; inset: 0; z-index: 1000; display: grid; place-items: center; background: #030b08; transition: opacity .42s ease, visibility .42s ease; }
      .homeIntro.isLeaving { opacity: 0; visibility: hidden; pointer-events: none; }
      .homeIntroFrame { width: min(790px, calc(100% - 40px)); padding: 30px; border-top: 1px solid #69e8bb; border-bottom: 1px solid rgba(105,232,187,.38); position: relative; overflow: hidden; }
      .homeIntroFrame:before, .homeIntroFrame:after { content: ""; position: absolute; width: 12px; height: 12px; border-color: #8ef6ca; border-style: solid; }
      .homeIntroFrame:before { top: 0; left: 0; border-width: 2px 0 0 2px; } .homeIntroFrame:after { right: 0; bottom: 0; border-width: 0 2px 2px 0; }
      .homeIntroGrid { display: grid; grid-template-columns: minmax(0, 1fr) 250px; align-items: center; gap: 34px; }
      .homeIntroKicker { color: #78dab5; font: 12px ui-monospace, SFMono-Regular, Menlo, monospace; letter-spacing: .16em; }
      .homeIntroTitle { margin: 10px 0; color: #effff7; font-size: clamp(34px, 6vw, 62px); line-height: 1; letter-spacing: 0; }
      .homeIntroText { margin: 0; color: #9fc6b5; font-size: 15px; }
      .homeIntroProgress { height: 3px; margin-top: 24px; background: rgba(112,233,187,.18); overflow: hidden; }
      .homeIntroProgress i { display: block; height: 100%; width: 100%; background: #65e7b6; transform-origin: left; animation: homeIntroProgress 1.45s ease-out both; }
      @keyframes homeIntroProgress { from { transform: scaleX(0); } to { transform: scaleX(1); } }
      .homeIntroVisual { height: 190px; border: 1px solid rgba(120,235,187,.38); position: relative; overflow: hidden; background-color: rgba(4,19,14,.82); }
      .homeIntroRoute { position: absolute; left: 20%; top: 60%; width: 62%; height: 2px; background: #73edbd; transform: rotate(-24deg); transform-origin: left; animation: homeIntroRoute 1.1s .18s both; }
      .homeIntroNode { position: absolute; width: 10px; height: 10px; border: 2px solid #b2ffe0; background: #0e7f5a; animation: homeIntroNode 1.3s infinite; }
      .homeIntroNode.a { left: 18%; top: 62%; } .homeIntroNode.b { left: 49%; top: 47%; animation-delay: .28s; } .homeIntroNode.c { left: 76%; top: 31%; animation-delay: .55s; }
      .homeIntroScan { position: absolute; left: 0; right: 0; height: 1px; background: #8effcc; animation: homeIntroScan 1.25s linear infinite; opacity: .75; }
      @keyframes homeIntroRoute { from { transform: rotate(-24deg) scaleX(0); } to { transform: rotate(-24deg) scaleX(1); } }
      @keyframes homeIntroNode { 0%,100% { box-shadow: 0 0 0 rgba(112,246,190,0); } 50% { box-shadow: 0 0 18px rgba(112,246,190,.85); } }
      @keyframes homeIntroScan { from { top: 0; } to { top: 100%; } }
      .homeIntroSkip { margin-top: 16px; border: 0; padding: 4px 0; background: transparent; color: #78c9a9; font: 12px ui-monospace, SFMono-Regular, Menlo, monospace; }
      button, .button { transition: transform .16s ease, box-shadow .16s ease, border-color .16s ease, background-color .16s ease; will-change: transform; }
      button:hover, .button:hover { transform: translateY(-2px); box-shadow: 0 8px 20px rgba(0,0,0,.22); }
      button:active, .button:active { transform: translateY(1px) scale(.96); box-shadow: none; }
      .runLink { display: inline-block; color: #7ff3c2; font-weight: 700; transition: transform .18s ease, color .18s ease, text-shadow .18s ease; }
      .runLink:hover { transform: scale(1.035); color: #d4ffee; text-shadow: 0 0 14px rgba(111,242,189,.62); }
      .runLink.isLaunching { transform: scale(1.08); color: #edfff7; text-shadow: 0 0 22px rgba(111,242,189,.9); }
      .runNavigateOverlay { position: fixed; inset: 0; z-index: 900; display: grid; place-items: center; pointer-events: none; opacity: 0; background: rgba(2,14,10,.18); transition: opacity .16s ease; }
      .runNavigateOverlay.isActive { opacity: 1; }
      .runNavigateOverlay span { color: #baffdf; font: 13px ui-monospace, SFMono-Regular, Menlo, monospace; letter-spacing: .14em; opacity: 0; transform: scale(.9); }
      .runNavigateOverlay.isActive span { animation: runNavigatePulse .22s ease both; }
      @keyframes runNavigatePulse { from { opacity: 0; transform: scale(.9); } to { opacity: 1; transform: scale(1); } }
      .runDetail .wrap > .panel, .runDetail .layout > .panel, .runDetail .stack .panel { animation: detailPanelEnter .48s ease both; }
      .runDetail .layout > .panel { animation-delay: .08s; }
      .runDetail .stack .panel:nth-child(1) { animation-delay: .14s; } .runDetail .stack .panel:nth-child(2) { animation-delay: .20s; }
      .runDetail .stack .panel:nth-child(3) { animation-delay: .26s; } .runDetail .stack .panel:nth-child(4) { animation-delay: .32s; }
      .runDetail .stack .panel:nth-child(5) { animation-delay: .38s; } .runDetail .stack .panel:nth-child(6) { animation-delay: .44s; }
      .runDetail .stack .panel:nth-child(7) { animation-delay: .50s; }
      @keyframes detailPanelEnter { from { opacity: 0; transform: translateY(12px) scale(.988); } to { opacity: 1; transform: translateY(0) scale(1); } }
      .roverDock { position: fixed; right: 18px; bottom: 18px; z-index: 120; display: flex; align-items: end; gap: 10px; }
      .runDetail .roverDock { left: 18px; right: auto; }
      .roverRobot { width: 98px; height: 82px; position: relative; border: 0; padding: 0; background: transparent; filter: drop-shadow(0 10px 14px rgba(0,0,0,.34)); transition: transform .28s ease; }
      .roverRobot:hover { transform: translateY(-4px) scale(1.04); box-shadow: none; } .roverRobot:active { transform: scale(.95); }
      .roverBody { position: absolute; left: 22px; top: 29px; width: 54px; height: 29px; border: 2px solid #8ef6ca; border-radius: 7px; background: #123c31; }
      .roverHead { position: absolute; left: 3px; top: 22px; width: 28px; height: 24px; border: 2px solid #baffdf; border-radius: 6px; background: #164b3d; }
      .roverEye { position: absolute; left: 8px; top: 8px; width: 5px; height: 5px; background: #71f4bf; box-shadow: 10px 0 #71f4bf; animation: roverBlink 3.4s infinite; }
      .roverLeg { position: absolute; top: 56px; width: 8px; height: 20px; border: 2px solid #85dfbb; border-top: 0; background: #0d2a22; transform-origin: top center; }
      .roverLeg.a { left: 29px; } .roverLeg.b { left: 48px; } .roverLeg.c { left: 66px; } .roverLeg.d { left: 80px; }
      .roverTail { position: absolute; left: 75px; top: 31px; width: 17px; height: 14px; border-top: 2px solid #83e8bd; transform: rotate(-25deg); transform-origin: left; }
      .roverHeart { position: absolute; left: 41px; top: -8px; width: 12px; height: 12px; background: #ff7188; opacity: 0; transform: rotate(45deg) scale(.4); }
      .roverHeart:before, .roverHeart:after { content: ""; position: absolute; width: 12px; height: 12px; border-radius: 50%; background: #ff7188; } .roverHeart:before { left: -6px; } .roverHeart:after { top: -6px; }
      .roverChargeLine { position: absolute; left: 42px; top: -18px; width: 12px; height: 18px; border-left: 2px solid #ffd86a; border-right: 2px solid #ffd86a; opacity: 0; }
      .roverListenWave { position: absolute; left: -13px; top: 17px; width: 26px; height: 34px; border: 2px solid #80f4ff; border-left: 0; border-top-color: transparent; border-bottom-color: transparent; border-radius: 0 50% 50% 0; opacity: 0; pointer-events: none; }
      .roverRobot.isListening { transform: translateY(-3px); filter: drop-shadow(0 0 14px rgba(91,235,255,.42)); }
      .roverRobot.isListening .roverHead { transform: translateY(-5px) rotate(-7deg); border-color: #9af7ff; background: #15505a; transition: transform .24s ease; }
      .roverRobot.isListening .roverEye { background: #b9fbff; box-shadow: 10px 0 #b9fbff; animation: roverListenEye .7s infinite alternate; }
      .roverRobot.isListening .roverListenWave { opacity: 1; animation: roverListenWave 1.05s ease-out infinite; }
      .roverRobot.isListening .roverTail { animation: roverWag .65s infinite alternate; }
      .roverRobot.isProcessing .roverHead { border-color: #ffe28a; animation: roverThinkHead .8s ease-in-out infinite alternate; }
      .roverRobot.isProcessing .roverEye { background: #ffe28a; box-shadow: 10px 0 #ffe28a; animation: roverProcessingEye .85s linear infinite; }
      .roverRobot.isProcessing .roverListenWave { opacity: .75; border-color: #ffe28a; border-top-color: transparent; border-bottom-color: transparent; animation: roverListenWave .7s ease-out infinite; }
      .roverRobot.isCharging .roverChargeLine { opacity: 1; animation: roverCharge .65s infinite; } .roverRobot.isCharging .roverBody { border-color: #ffe07b; box-shadow: inset 0 0 calc(8px + 26px * var(--charge-level, .12)) rgba(255,216,106,calc(.2 + .7 * var(--charge-level, .12))); }
      .roverRobot.isBone { animation: roverBonePop .34s ease-in-out infinite alternate; filter: drop-shadow(0 0 13px rgba(245,255,251,.88)); }
      .roverRobot.isBone > i { opacity: 0; }
      .roverRobot.isBone:before { content: ""; position: absolute; left: 20px; top: 34px; width: 58px; height: 14px; border-radius: 999px; background: #f5fff9; box-shadow: inset 0 -2px 0 rgba(80,138,112,.24); }
      .roverRobot.isBone:after { content: ""; position: absolute; left: 18px; top: 27px; width: 18px; height: 18px; border-radius: 50%; background: #fff; box-shadow: 42px 0 #fff, 0 13px #fff, 42px 13px #fff; }
      .roverBlastRing { position: absolute; left: 50%; bottom: 2px; width: 28px; height: 28px; border: 2px solid rgba(227,255,241,.95); border-radius: 50%; transform: translateX(-50%) scale(.25); opacity: 0; pointer-events: none; }
      .roverBlastRing.isActive { animation: roverBlastRange .8s ease-out 3; }
      .roverSearchPanel { position: fixed; right: 18px; bottom: 116px; z-index: 121; width: min(340px, calc(100vw - 28px)); padding: 14px; border: 1px solid rgba(112,237,187,.54); border-radius: 8px; background: rgba(4,22,16,.97); box-shadow: 0 18px 48px rgba(0,0,0,.48), 0 0 30px rgba(85,235,177,.12); display: none; }
      .runDetail .roverSearchPanel { left: 18px; right: auto; }
      .roverSearchPanel.isOpen { display: block; animation: roverSearchOpen .24s ease both; }
      .roverSearchHead { display: flex; align-items: center; justify-content: space-between; gap: 8px; margin-bottom: 10px; }
      .roverSearchHead strong { color: #d9ffea; font-size: 14px; } .roverSearchClose { min-width: 28px; padding: 3px 7px; font-size: 18px; line-height: 1; }
      .roverSearchForm { display: flex; gap: 7px; } .roverSearchForm input { min-width: 0; flex: 1; padding: 7px 8px; font-size: 12px; }
      .roverSearchFilters { display: flex; flex-wrap: wrap; gap: 6px; margin: 9px 0; } .roverSearchFilters button { padding: 4px 7px; font-size: 11px; }
      .roverSearchResults { max-height: 210px; overflow: auto; border-top: 1px solid rgba(112,237,187,.16); }
      .roverSearchResult { display: block; padding: 8px 2px; border-bottom: 1px solid rgba(112,237,187,.12); color: #b8efd4; text-decoration: none; }
      .roverSearchResult:hover { color: #f0fff7; } .roverSearchResult b { display: block; font: 12px ui-monospace, SFMono-Regular, Menlo, monospace; } .roverSearchResult small { display: block; margin-top: 3px; color: #7da996; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
      .roverSearchEmpty { padding: 12px 2px; color: #84aa99; font-size: 12px; }
      .roverRobot.pose-sit { transform: translateY(8px); } .roverRobot.pose-sit .roverLeg.c, .roverRobot.pose-sit .roverLeg.d { transform: rotate(52deg); }
      .roverRobot.pose-lie { transform: scaleX(1.16) scaleY(.72) translateY(17px); } .roverRobot.pose-lie .roverLeg { opacity: .3; }
      .roverRobot.pose-paw .roverLeg.a { transform: rotate(-64deg) translateY(-10px); } .roverRobot.pose-paw .roverTail { animation: roverWag .36s 5 alternate; }
      .roverRobot.pose-heart .roverHeart { opacity: 1; animation: roverHeart 1s infinite; } .roverRobot.pose-heart .roverTail { animation: roverWag .32s 6 alternate; }
      @keyframes roverBlink { 0%, 45%, 48%, 100% { transform: scaleY(1); } 46%, 47% { transform: scaleY(.12); } } @keyframes roverWag { to { transform: rotate(24deg); } } @keyframes roverHeart { 50% { transform: rotate(45deg) translate(-4px,-5px) scale(1.15); } } @keyframes roverCharge { 50% { filter: drop-shadow(0 0 8px #ffdc69); transform: translateY(2px); } } @keyframes roverBonePop { to { transform: translateY(-7px) rotate(4deg) scale(1.13); } } @keyframes roverBlastRange { 0% { opacity: .98; transform: translateX(-50%) scale(.3); } 100% { opacity: 0; transform: translateX(-50%) scale(5); } } @keyframes roverSearchOpen { from { opacity: 0; transform: translateY(10px) scale(.96); } to { opacity: 1; transform: translateY(0) scale(1); } }
      @keyframes roverListenWave { from { opacity: .9; transform: scale(.55); } to { opacity: 0; transform: scale(1.35); } }
      @keyframes roverListenEye { to { box-shadow: 10px 0 #b9fbff, 0 0 12px #73efff, 10px 0 12px #73efff; } }
      @keyframes roverThinkHead { from { transform: translateX(-2px) rotate(-5deg); } to { transform: translateX(3px) rotate(5deg); } }
      @keyframes roverProcessingEye { 0%,100% { opacity: .45; } 50% { opacity: 1; box-shadow: 10px 0 #ffe28a, 0 0 10px #ffe28a; } }
      .roverTop { padding: 7px 9px; font-size: 12px; white-space: nowrap; }
      @media (prefers-reduced-motion: reduce) { *, *:before, *:after { animation-duration: .01ms !important; animation-iteration-count: 1 !important; transition-duration: .01ms !important; } }
      @media (max-width: 1250px) and (min-width: 1001px) { .configSectionGrid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
      @media (max-width: 1000px) { .layout, .runDetail .layout, .configGrid, .configSectionGrid, .heroGrid { grid-template-columns: 1fr; } .heroMedia { display: none; } .voiceInput { grid-template-columns: minmax(0, 1fr) minmax(110px, .45fr) auto; } .voiceStatus { grid-column: 1 / -1; } }
      @media (max-width: 620px) { .homeIntroGrid, .homeQaSelector, .homeQaQuestionRow { grid-template-columns: 1fr; gap: 9px; } .homeIntroVisual { height: 130px; } .roverDock { right: 10px; bottom: 10px; } .runDetail .roverDock { left: 10px; right: auto; } .roverSearchPanel, .runDetail .roverSearchPanel { left: 10px; right: auto; bottom: 106px; } .roverRobot { width: 78px; transform: scale(.84); transform-origin: right bottom; } }
    """


def render_config_script():
    return """
      function configEscapeHtml(value) {
        return String(value)
          .replace(/&/g, '&amp;')
          .replace(/</g, '&lt;')
          .replace(/>/g, '&gt;')
          .replace(/\"/g, '&quot;');
      }
      let activeVoiceRecording = null;
      let voiceInteractionMode = false;
      let voiceInteractionContainer = null;
      let voiceInteractionRestartTimer = null;
      const voiceDeviceStorageKey = 'internnavVoiceDeviceId';

      function setVoiceStatus(container, message, kind='') {
        const status = container && container.querySelector('.voiceStatus');
        if (!status) return;
        status.textContent = message;
        status.className = 'voiceStatus' + (kind ? ' is' + kind : '');
      }

      function syncRuntimePolicyState(enabled) {
        const form = document.getElementById('runtimeConfigForm');
        const hidden = form && form.querySelector('input[name="service_enabled"]');
        if (hidden) hidden.value = enabled ? 'true' : 'false';
        const stateText = form && form.querySelector('.stateText');
        const stateDot = form && form.querySelector('.stateDot');
        if (stateText) stateText.textContent = enabled ? 'RUNNING' : 'STOPPED';
        if (stateDot) {
          stateDot.classList.toggle('stateRun', enabled);
          stateDot.classList.toggle('stateStop', !enabled);
        }
      }

      function syncVoiceDevice(deviceId) {
        document.querySelectorAll('.voiceDeviceSelect').forEach((select) => {
          if (Array.from(select.options).some((option) => option.value === deviceId)) select.value = deviceId;
        });
        try { localStorage.setItem(voiceDeviceStorageKey, deviceId || ''); } catch (_) {}
      }

      async function refreshVoiceDevices() {
        const containers = document.querySelectorAll('.voiceInput');
        if (!containers.length || !navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
        try {
          const devices = (await navigator.mediaDevices.enumerateDevices()).filter((device) => device.kind === 'audioinput');
          let saved = '';
          try { saved = localStorage.getItem(voiceDeviceStorageKey) || ''; } catch (_) {}
          document.querySelectorAll('.voiceDeviceSelect').forEach((select) => {
            const current = select.value || saved;
            select.innerHTML = '<option value="">默认麦克风</option>' + devices.map((device, index) =>
              `<option value="${configEscapeHtml(device.deviceId)}">${configEscapeHtml(device.label || '麦克风 ' + (index + 1))}</option>`
            ).join('');
            if (Array.from(select.options).some((option) => option.value === current)) select.value = current;
            select.onchange = () => syncVoiceDevice(select.value);
          });
        } catch (error) {
          containers.forEach((container) => setVoiceStatus(container, '无法读取音频设备：' + error.message, 'Error'));
        }
      }

      function insertVoiceTranscript(target, transcript) {
        // A voice command represents a new task, so never append it to a stale
        // instruction left by the previous experiment.
        target.value = transcript.trim();
        target.dispatchEvent(new Event('input', {bubbles: true}));
        target.focus();
      }

      function updateVoiceInteractionButton(container, active) {
        const button = container && container.querySelector('.voiceInteractionButton');
        if (!button) return;
        button.classList.toggle('isActive', active);
        button.textContent = active ? '退出交互模式' : '进入交互模式';
      }

      function scheduleVoiceInteractionRestart(container, recordButton) {
        clearTimeout(voiceInteractionRestartTimer);
        if (!voiceInteractionMode || voiceInteractionContainer !== container) return;
        voiceInteractionRestartTimer = window.setTimeout(() => {
          if (voiceInteractionMode && voiceInteractionContainer === container && !activeVoiceRecording) {
            setVoiceStatus(container, '交互模式正在倾听；请说一条明确的导航指令。');
            toggleVoiceRecording(recordButton);
          }
        }, 1200);
      }

      function toggleVoiceInteraction(button) {
        const container = button.closest('.voiceInput');
        if (!container || container.dataset.voiceTarget !== 'lowLevelInstruction') return;
        clearTimeout(voiceInteractionRestartTimer);
        if (voiceInteractionMode && voiceInteractionContainer === container) {
          voiceInteractionMode = false;
          voiceInteractionContainer = null;
          updateVoiceInteractionButton(container, false);
          if (activeVoiceRecording && activeVoiceRecording.container === container) {
            activeVoiceRecording.discard = true;
            stopVoiceRecording(activeVoiceRecording, '正在退出交互模式...');
          } else {
            setVoiceStatus(container, '已退出交互模式。');
            if (typeof setRoverInteractionState === 'function') setRoverInteractionState('idle');
          }
          return;
        }
        if (voiceInteractionMode || activeVoiceRecording) {
          setVoiceStatus(container, '请先结束当前录音或另一个交互会话。', 'Error');
          return;
        }
        voiceInteractionMode = true;
        voiceInteractionContainer = container;
        updateVoiceInteractionButton(container, true);
        setVoiceStatus(container, '交互模式已开启；闲聊和噪声会被忽略。');
        toggleVoiceRecording(container.querySelector('.voiceRecordButton'));
      }

      function cleanupVoiceMonitoring(recording) {
        clearTimeout(recording.timeout);
        if (recording.vadTimer) clearInterval(recording.vadTimer);
        if (recording.audioContext && recording.audioContext.state !== 'closed') recording.audioContext.close().catch(() => {});
      }

      function stopVoiceRecording(recording, message='正在结束录音...') {
        if (!recording || recording.recorder.state !== 'recording') return;
        cleanupVoiceMonitoring(recording);
        setVoiceStatus(recording.container, message);
        if (typeof setRoverInteractionState === 'function') setRoverInteractionState('processing');
        recording.recorder.stop();
      }

      function startVoiceActivityDetection(recording) {
        const AudioContextClass = window.AudioContext || window.webkitAudioContext;
        if (!AudioContextClass) return;
        const audioContext = new AudioContextClass();
        const analyser = audioContext.createAnalyser();
        analyser.fftSize = 2048;
        audioContext.createMediaStreamSource(recording.stream).connect(analyser);
        const samples = new Uint8Array(analyser.fftSize);
        recording.audioContext = audioContext;
        recording.heardSpeech = false;
        recording.speechFrames = 0;
        recording.noiseFloor = 0.006;
        recording.lastSpeechAt = 0;
        recording.vadTimer = setInterval(() => {
          if (recording.recorder.state !== 'recording') return;
          analyser.getByteTimeDomainData(samples);
          let energy = 0;
          for (let index = 0; index < samples.length; index += 1) {
            const sample = (samples[index] - 128) / 128;
            energy += sample * sample;
          }
          const rms = Math.sqrt(energy / samples.length);
          if (!recording.heardSpeech) recording.noiseFloor = recording.noiseFloor * .92 + Math.min(rms, .03) * .08;
          const threshold = Math.max(.015, recording.noiseFloor * 2.8);
          if (rms > threshold) {
            recording.speechFrames += 1;
            if (recording.speechFrames >= 3) {
              recording.heardSpeech = true;
              recording.lastSpeechAt = performance.now();
              setVoiceStatus(recording.container, '已检测到语音，讲话结束后将自动转写。');
            }
          } else {
            recording.speechFrames = 0;
          }
          if (
            recording.heardSpeech
            && performance.now() - recording.lastSpeechAt >= recording.silenceSeconds * 1000
          ) {
            stopVoiceRecording(recording, '检测到讲话结束，正在自动转写...');
          }
        }, 50);
      }

      async function uploadVoiceRecording(recording) {
        const {container, button, chunks, mimeType, language, target, stream, refine, targetKind} = recording;
        cleanupVoiceMonitoring(recording);
        button.disabled = true;
        button.classList.remove('isRecording');
        button.textContent = '正在转写';
        setVoiceStatus(container, '本地语音模型正在识别，请稍候...');
        if (typeof setRoverInteractionState === 'function') setRoverInteractionState('processing');
        stream.getTracks().forEach((track) => track.stop());
        if (recording.discard) {
          button.disabled = false;
          button.classList.remove('isRecording');
          button.textContent = '开始录音';
          if (activeVoiceRecording === recording) activeVoiceRecording = null;
          setVoiceStatus(container, '已退出交互模式。');
          if (typeof setRoverInteractionState === 'function') setRoverInteractionState('idle');
          return;
        }
        try {
          const blob = new Blob(chunks, {type: mimeType || 'audio/webm'});
          const form = new FormData();
          form.append('audio', blob, mimeType.includes('ogg') ? 'recording.ogg' : 'recording.webm');
          form.append('language', language);
          form.append('refine', refine ? 'true' : 'false');
          form.append('target', targetKind);
          form.append('interaction_mode', recording.interactionMode ? 'true' : 'false');
          const requestController = new AbortController();
          const requestTimeout = window.setTimeout(() => requestController.abort(), 60000);
          let response;
          try {
            response = await fetch('/api/speech/transcribe', {
              method: 'POST',
              body: form,
              signal: requestController.signal
            });
          } catch (requestError) {
            if (requestError && requestError.name === 'AbortError') {
              throw new Error('语音处理超过 60 秒，已停止等待。请检查模型缓存或重启 Viewer。');
            }
            throw requestError;
          } finally {
            window.clearTimeout(requestTimeout);
          }
          const data = await response.json().catch(() => ({}));
          if (!response.ok || !data.ok) throw new Error(data.error || '语音转写失败');
          const timingParts = [];
          if (Number.isFinite(Number(data.transcription_seconds))) timingParts.push(`转写 ${Number(data.transcription_seconds).toFixed(2)}s`);
          if (Number.isFinite(Number(data.semantic_seconds))) timingParts.push(`语义 ${Number(data.semantic_seconds).toFixed(2)}s`);
          const timingSuffix = timingParts.length ? `（${timingParts.join('，')}）` : '';
          if (recording.interactionMode && !data.accepted) {
            const heard = String(data.transcript || '').trim();
            const confidence = Math.round(Number(data.confidence || 0) * 100);
            setVoiceStatus(container, `已忽略非指令语音（${confidence}%）：${heard || data.reason || '未识别到有效内容'}${timingSuffix}`);
          } else if (recording.interactionMode) {
            insertVoiceTranscript(target, data.instruction || data.text);
            if (data.applied) {
              syncRuntimePolicyState(Boolean(data.service_enabled));
              if (data.command_type === 'stop') {
                setVoiceStatus(container, `已确认 STOP，服务器 Policy Gate 已关闭。${timingSuffix}`, 'Success');
              } else {
                setVoiceStatus(container, `新指令已确认并应用，低层大脑将在下一帧重置。${timingSuffix}`, 'Success');
              }
            } else {
              setVoiceStatus(container, '指令已确认，但服务器未能应用或重置。', 'Error');
            }
          } else if (data.refine_error) {
            insertVoiceTranscript(target, data.text);
            setVoiceStatus(container, `语音已识别，但英文优化失败，未自动启动：${data.refine_error}`, 'Error');
          } else {
            insertVoiceTranscript(target, data.text);
            setVoiceStatus(container, '指令已生成，正在自动应用配置并启动 Policy...');
            await autoApplyVoiceInstruction(recording);
            setVoiceStatus(container, `指令已应用，Policy 已启动。${timingSuffix}`, 'Success');
          }
        } catch (error) {
          setVoiceStatus(container, error.message || String(error), 'Error');
        } finally {
          button.disabled = false;
          button.textContent = '开始录音';
          if (activeVoiceRecording === recording) activeVoiceRecording = null;
          if (typeof setRoverInteractionState === 'function') setRoverInteractionState('idle');
          scheduleVoiceInteractionRestart(container, button);
        }
      }

      async function toggleVoiceRecording(button) {
        const container = button.closest('.voiceInput');
        if (!container) return;
        if (activeVoiceRecording) {
          if (activeVoiceRecording.button !== button) {
            setVoiceStatus(container, '请先结束另一个输入框的录音。', 'Error');
            return;
          }
          stopVoiceRecording(activeVoiceRecording);
          return;
        }
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia || !window.MediaRecorder) {
          setVoiceStatus(container, '浏览器不支持录音。请通过 SSH 隧道访问 http://127.0.0.1:8899。', 'Error');
          return;
        }
        const target = document.getElementById(container.dataset.voiceTarget);
        const deviceId = container.querySelector('.voiceDeviceSelect').value;
        const mode = container.querySelector('.voiceLanguageSelect').value;
        const language = mode.startsWith('zh') ? 'zh' : (mode.startsWith('en') ? 'en' : '');
        const refine = mode.endsWith('_refine');
        const targetKind = container.dataset.voiceTarget === 'lowLevelInstruction' ? 'low_level' : 'upper_task';
        try {
          setVoiceStatus(container, '正在申请麦克风权限...');
          if (typeof setRoverInteractionState === 'function') setRoverInteractionState('listening');
          const audio = deviceId ? {deviceId: {exact: deviceId}} : true;
          const stream = await navigator.mediaDevices.getUserMedia({audio});
          await refreshVoiceDevices();
          const candidates = ['audio/webm;codecs=opus', 'audio/ogg;codecs=opus', 'audio/webm'];
          const mimeType = candidates.find((type) => MediaRecorder.isTypeSupported(type)) || '';
          const recorder = new MediaRecorder(stream, mimeType ? {mimeType} : undefined);
          const silenceInput = document.querySelector('[name="voice_silence_seconds"]');
          const configuredSilence = Number(silenceInput?.value ?? 1.4);
          const silenceSeconds = Number.isFinite(configuredSilence)
            ? Math.max(0.4, Math.min(5, configuredSilence))
            : 1.4;
          const recording = {container, button, target, language, refine, targetKind, stream, recorder, mimeType: recorder.mimeType, chunks: [], timeout: null, vadTimer: null, audioContext: null, interactionMode: voiceInteractionMode && voiceInteractionContainer === container, discard: false, silenceSeconds};
          recorder.ondataavailable = (event) => { if (event.data.size) recording.chunks.push(event.data); };
          recorder.onstop = () => uploadVoiceRecording(recording);
          recorder.onerror = (event) => {
            stream.getTracks().forEach((track) => track.stop());
            activeVoiceRecording = null;
            button.classList.remove('isRecording');
            button.textContent = '开始录音';
            setVoiceStatus(container, '录音失败：' + (event.error?.message || 'unknown error'), 'Error');
            if (typeof setRoverInteractionState === 'function') setRoverInteractionState('idle');
            if (voiceInteractionMode && voiceInteractionContainer === container) {
              voiceInteractionMode = false;
              voiceInteractionContainer = null;
              updateVoiceInteractionButton(container, false);
            }
          };
          activeVoiceRecording = recording;
          recorder.start(250);
          startVoiceActivityDetection(recording);
          button.classList.add('isRecording');
          button.textContent = '停止并转写';
          setVoiceStatus(container, `等待说话；讲话结束并静音约 ${silenceSeconds.toFixed(1)} 秒后自动转写。`);
          recording.timeout = setTimeout(() => stopVoiceRecording(recording, '已达到 60 秒上限，正在转写...'), 60000);
        } catch (error) {
          setVoiceStatus(container, '无法使用麦克风：' + error.message, 'Error');
          if (typeof setRoverInteractionState === 'function') setRoverInteractionState('idle');
          if (voiceInteractionMode && voiceInteractionContainer === container) {
            voiceInteractionMode = false;
            voiceInteractionContainer = null;
            updateVoiceInteractionButton(container, false);
          }
        }
      }

      if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', refreshVoiceDevices);
      else refreshVoiceDevices();
      if (navigator.mediaDevices && navigator.mediaDevices.addEventListener) navigator.mediaDevices.addEventListener('devicechange', refreshVoiceDevices);

      async function deleteRun(runName) {
        const confirmed = window.confirm(
          '确定删除实验 ' + runName + ' 吗？\\n\\n该实验目录中的图片、深度图、GIF、航点 JSON 和 Upper Agent 记录都会永久删除。'
        );
        if (!confirmed) return;
        try {
          const response = await fetch('/api/run/' + encodeURIComponent(runName) + '/delete', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({confirm: true})
          });
          const data = await response.json().catch(() => ({}));
          if (!response.ok || !data.ok) {
            window.alert('删除失败：' + (data.error || response.statusText));
            return;
          }
          window.location.reload();
        } catch (error) {
          window.alert('删除失败：' + error);
        }
      }
      async function updateRunReview(runName, payload) {
        try {
          const response = await fetch('/api/run/' + encodeURIComponent(runName) + '/review', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload)
          });
          const data = await response.json().catch(() => ({}));
          if (!response.ok || !data.ok) {
            window.alert('保存实验标记失败：' + (data.error || response.statusText));
            return;
          }
          window.location.reload();
        } catch (error) {
          window.alert('保存实验标记失败：' + error);
        }
      }
      function toggleRunPin(button) {
        updateRunReview(button.dataset.run, {pinned: button.dataset.pinned !== 'true'});
      }
      function setRunOutcome(select) {
        updateRunReview(select.dataset.run, {outcome: select.value});
      }
      function toggleAllRuns(checked) {
        document.querySelectorAll('.runSelect').forEach((box) => { box.checked = checked; });
      }
      async function deleteSelectedRuns() {
        const names = Array.from(document.querySelectorAll('.runSelect:checked')).map((box) => box.value);
        if (!names.length) {
          window.alert('请先选择要删除的实验。');
          return;
        }
        const confirmed = window.confirm(
          '确定删除选中的 ' + names.length + ' 个实验吗？\\n\\n所有图片、深度图、GIF、航点 JSON 和 Upper Agent 记录都会永久删除。'
        );
        if (!confirmed) return;
        try {
          const response = await fetch('/api/runs/delete', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({confirm: true, names: names})
          });
          const data = await response.json().catch(() => ({}));
          if (!response.ok || !data.ok) {
            window.alert('批量删除失败：' + (data.error || response.statusText));
            return;
          }
          window.location.reload();
        } catch (error) {
          window.alert('批量删除失败：' + error);
        }
      }
      function runtimeConfigPayload(form) {
        const payload = Object.fromEntries(new FormData(form).entries());
        ['resize_w', 'resize_h', 'num_history', 'plan_step_gap', 'return_traj_points', 'save_frame_interval', 'low_level_stop_replan_threshold', 'voice_silence_seconds', 'voice_command_confidence_threshold'].forEach((key) => {
          payload[key] = Number(payload[key]);
        });
        payload.service_enabled = payload.service_enabled === 'true';
        return payload;
      }
      async function saveRuntimeConfigForm(reload=true) {
        const form = document.getElementById('runtimeConfigForm');
        const status = document.getElementById('configStatus');
        if (!form) throw new Error('Runtime Config 表单不存在。');
        if (status) status.innerText = 'Applying...';
        const response = await fetch('/api/runtime-config', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(runtimeConfigPayload(form))
        });
        const data = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(data.error || 'Runtime Config apply failed');
        if (status) status.innerText = 'Applied ' + (data.updated_at || new Date().toLocaleTimeString());
        if (reload) setTimeout(() => window.location.reload(), 300);
        return data;
      }
      async function submitRuntimeConfig(event) {
        event.preventDefault();
        const status = document.getElementById('configStatus');
        try { await saveRuntimeConfigForm(true); }
        catch (error) { if (status) status.innerText = error.message; }
      }
      async function setServiceEnabled(enabled, reload=true) {
        const status = document.getElementById('configStatus');
        if (status) status.innerText = enabled ? 'Starting policy...' : 'Stopping policy...';
        const response = await fetch('/api/runtime-config/control', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({service_enabled: enabled})
        });
        const data = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(data.error || 'Policy control failed');
        if (status) status.innerText = enabled ? 'Policy start requested' : 'Policy stop requested';
        if (reload) setTimeout(() => window.location.reload(), 300);
        return data;
      }
      async function submitServiceLauncher(event) {
        event.preventDefault();
        const form = event.target;
        applyTransportPreset(form.transport_preset.value, false);
        const payload = Object.fromEntries(new FormData(form).entries());
        delete payload.transport_preset;
        payload.http_port = Number(payload.http_port || 8848);
        payload.zenoh_no_multicast_scouting = form.zenoh_no_multicast_scouting.checked;
        payload.no_warmup = form.no_warmup.checked;
        const status = document.getElementById('serviceStatus');
        if (status) status.innerText = 'Starting service...';
        const response = await fetch('/api/model-service', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(payload)
        });
        const data = await response.json().catch(() => ({}));
        if (!response.ok) {
          if (status) status.innerText = data.error || 'Start failed';
          return;
        }
        if (status) status.innerText = 'Started pid=' + data.pid;
        setTimeout(() => window.location.reload(), 600);
      }
      async function stopModelService() {
        const status = document.getElementById('serviceStatus');
        if (status) status.innerText = 'Stopping service...';
        const response = await fetch('/api/model-service', { method: 'DELETE' });
        const data = await response.json().catch(() => ({}));
        if (!response.ok) {
          if (status) status.innerText = data.error || 'Stop failed';
          return;
        }
        if (status) status.innerText = 'Stopped';
        setTimeout(() => window.location.reload(), 600);
      }
      async function startLocalQwen(event) {
        event.preventDefault();
        const form = event.target;
        const payload = Object.fromEntries(new FormData(form).entries());
        payload.port = Number(payload.port || 8000);
        payload.gpu_memory_utilization = Number(payload.gpu_memory_utilization || 0.60);
        payload.max_model_len = Number(payload.max_model_len || 8192);
        payload.max_num_seqs = Number(payload.max_num_seqs || 2);
        const status = document.getElementById('localQwenStatus');
        if (status) status.innerText = 'Starting local Qwen; model loading can take several minutes...';
        try {
          const response = await fetch('/api/local-qwen', {
            method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)
          });
          const data = await response.json().catch(() => ({}));
          if (!response.ok) throw new Error(data.error || 'Local Qwen start failed');
          if (status) status.innerText = 'Local Qwen loading, pid=' + data.pid + '. Refresh to check READY.';
          setTimeout(() => window.location.reload(), 1000);
        } catch (error) { if (status) status.innerText = error.message; }
      }
      async function stopLocalQwen() {
        const status = document.getElementById('localQwenStatus');
        if (status) status.innerText = 'Stopping local Qwen...';
        try {
          const response = await fetch('/api/local-qwen', {method:'DELETE'});
          const data = await response.json().catch(() => ({}));
          if (!response.ok) throw new Error(data.error || 'Local Qwen stop failed');
          if (status) status.innerText = 'Local Qwen stopped';
          setTimeout(() => window.location.reload(), 500);
        } catch (error) { if (status) status.innerText = error.message; }
      }
      async function useLocalQwenForAgents() {
        const status = document.getElementById('localQwenStatus');
        if (status) status.innerText = 'Applying local Qwen to agent configuration...';
        try {
          const response = await fetch('/api/local-qwen/use', {method:'POST'});
          const data = await response.json().catch(() => ({}));
          if (!response.ok) throw new Error(data.error || 'Could not configure local Qwen');
          if (status) status.innerText = 'Local Qwen selected for Upper Agent, voice refinement, and QA.';
          setTimeout(() => window.location.reload(), 500);
        } catch (error) { if (status) status.innerText = error.message; }
      }
      let demoLibraries = [];
      let demoPreviewTimer = null;
      function demoStatus(message, isError=false) {
        const node = document.getElementById('demoAgentStatus');
        if (!node) return;
        node.innerText = message;
        node.style.color = isError ? '#ff9aaa' : '';
      }
      function currentDemoLibrary() {
        const id = (document.getElementById('demoLibrarySelect') || {}).value || '';
        return demoLibraries.find((item) => String(item.id) === String(id));
      }
      function populateDemoSelectors(selectedId='') {
        const sceneFilter = document.getElementById('demoSceneFilter');
        const librarySelect = document.getElementById('demoLibrarySelect');
        const sceneOptions = document.getElementById('demoSceneOptions');
        if (!sceneFilter || !librarySelect) return;
        const previousScene = sceneFilter.value;
        const scenes = Array.from(new Set(demoLibraries.map((item) => item.scene).filter(Boolean))).sort();
        sceneFilter.innerHTML = '<option value="">全部场景</option>' + scenes.map((scene) =>
          '<option value="' + configEscapeHtml(scene) + '">' + configEscapeHtml(scene) + '</option>').join('');
        if (scenes.includes(previousScene)) sceneFilter.value = previousScene;
        if (sceneOptions) sceneOptions.innerHTML = scenes.map((scene) =>
          '<option value="' + configEscapeHtml(scene) + '"></option>').join('');
        const visible = demoLibraries.filter((item) => !sceneFilter.value || item.scene === sceneFilter.value);
        librarySelect.innerHTML = '<option value="">选择记录</option>' + visible.map((item) =>
          '<option value="' + configEscapeHtml(item.id) + '">' + configEscapeHtml(item.name) +
          ' · ' + configEscapeHtml(item.commands.length) + ' 条</option>').join('');
        if (visible.some((item) => String(item.id) === String(selectedId))) librarySelect.value = selectedId;
        renderDemoLibraryRecords(visible);
      }
      function renderDemoLibraryRecords(libraries) {
        const target = document.getElementById('demoLibraryRecords');
        if (!target) return;
        if (!libraries.length) { target.innerHTML = '<p class="status">这个场景还没有保存记录。</p>'; return; }
        target.innerHTML = libraries.map((item) =>
          '<div class="demoLibraryRow"><strong>' + configEscapeHtml(item.name) + '</strong>' +
          '<span>' + configEscapeHtml(item.scene) + '</span>' +
          '<span>' + configEscapeHtml(item.commands.length) + ' 条</span>' +
          '<small title="' + configEscapeHtml(item.notes || '') + '">' + configEscapeHtml(item.notes || '无备注') + '</small>' +
          '<button type="button" data-demo-id="' + configEscapeHtml(item.id) + '">编辑</button></div>'
        ).join('');
        target.querySelectorAll('[data-demo-id]').forEach((button) => button.addEventListener('click', () => selectDemoLibrary(button.dataset.demoId)));
      }
      async function loadDemoLibraries(selectedId='') {
        if (!document.getElementById('demoAgentPanel')) return;
        try {
          const response = await fetch('/api/demo-agent/libraries', {cache: 'no-store'});
          const data = await response.json().catch(() => ({}));
          if (!response.ok || !data.ok) throw new Error(data.error || '加载失败');
          demoLibraries = data.libraries || [];
          populateDemoSelectors(selectedId || ((data.active || {}).library_id || ''));
          const chosen = selectedId || ((data.active || {}).library_id || '');
          if (chosen) selectDemoLibrary(chosen);
        } catch (error) { demoStatus('加载指令库失败：' + error.message, true); }
      }
      function filterDemoLibraries() { populateDemoSelectors(); }
      function selectDemoLibrary(id) {
        const library = demoLibraries.find((item) => String(item.id) === String(id));
        const select = document.getElementById('demoLibrarySelect');
        if (select && library) select.value = library.id;
        if (!library) return;
        document.getElementById('demoLibraryId').value = library.id || '';
        document.getElementById('demoLibraryName').value = library.name || '';
        document.getElementById('demoLibraryScene').value = library.scene || '';
        document.getElementById('demoLibraryNotes').value = library.notes || '';
        document.getElementById('demoCommandsText').value = (library.commands || []).join('\\n');
        renderDemoPreview(library.commands || []);
        demoStatus('已载入：' + library.name + '，修改后点击“保存记录”。');
      }
      function newDemoLibrary() {
        const form = document.getElementById('demoLibraryForm');
        if (form) form.reset();
        document.getElementById('demoLibraryId').value = '';
        const select = document.getElementById('demoLibrarySelect');
        if (select) select.value = '';
        renderDemoPreview([]);
        demoStatus('正在新建指令库。');
      }
      function renderDemoPreview(commands, error='') {
        const list = document.getElementById('demoCommandPreview');
        if (!list) return;
        if (error) { list.innerHTML = '<li style="color:#ff9aaa">' + configEscapeHtml(error) + '</li>'; return; }
        if (!commands.length) { list.innerHTML = '<li class="status">输入后显示执行顺序</li>'; return; }
        list.innerHTML = commands.map((command) => '<li>' + configEscapeHtml(command) + '</li>').join('');
      }
      function scheduleDemoPreview() {
        clearTimeout(demoPreviewTimer);
        demoPreviewTimer = setTimeout(previewDemoCommands, 220);
      }
      async function previewDemoCommands() {
        const text = (document.getElementById('demoCommandsText') || {}).value || '';
        try {
          const response = await fetch('/api/demo-agent/parse', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({commands_text: text})
          });
          const data = await response.json().catch(() => ({}));
          if (!response.ok || !data.ok) throw new Error(data.error || '解析失败');
          renderDemoPreview(data.commands || []);
        } catch (error) { renderDemoPreview([], error.message); }
      }
      async function saveDemoLibrary(event) {
        event.preventDefault();
        const form = event.target;
        const payload = Object.fromEntries(new FormData(form).entries());
        demoStatus('正在保存...');
        try {
          const response = await fetch('/api/demo-agent/libraries', {
            method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)
          });
          const data = await response.json().catch(() => ({}));
          if (!response.ok || !data.ok) throw new Error(data.error || '保存失败');
          await loadDemoLibraries(data.library.id);
          demoStatus('已保存：' + data.library.name);
        } catch (error) { demoStatus('保存失败：' + error.message, true); }
      }
      async function activateSelectedDemoLibrary() {
        const library = currentDemoLibrary();
        if (!library) { demoStatus('请先选择一个已保存的指令库。', true); return; }
        demoStatus('正在启动 ' + library.name + '...');
        try {
          const response = await fetch('/api/demo-agent/activate/' + encodeURIComponent(library.id), {method: 'POST'});
          const data = await response.json().catch(() => ({}));
          if (!response.ok || !data.ok) throw new Error(data.error || '启动失败');
          demoStatus('已启动，等待最新相机帧后下发第 1 条指令。');
          setTimeout(() => window.location.reload(), 450);
        } catch (error) { demoStatus('启动失败：' + error.message, true); }
      }
      async function controlDemoAgent(action) {
        demoStatus('正在执行：' + action + '...');
        try {
          const response = await fetch('/api/demo-agent/control', {
            method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({action})
          });
          const data = await response.json().catch(() => ({}));
          if (!response.ok || !data.ok) throw new Error(data.error || '控制失败');
          setTimeout(() => window.location.reload(), 350);
        } catch (error) { demoStatus('控制失败：' + error.message, true); }
      }
      async function deleteSelectedDemoLibrary() {
        const library = currentDemoLibrary();
        if (!library) { demoStatus('请先选择要删除的记录。', true); return; }
        if (!window.confirm('确定删除指令库“' + library.name + '”吗？')) return;
        try {
          const response = await fetch('/api/demo-agent/libraries/' + encodeURIComponent(library.id), {method: 'DELETE'});
          const data = await response.json().catch(() => ({}));
          if (!response.ok || !data.ok) throw new Error(data.error || '删除失败');
          newDemoLibrary();
          await loadDemoLibraries();
          demoStatus('记录已删除。');
        } catch (error) { demoStatus('删除失败：' + error.message, true); }
      }
      document.addEventListener('DOMContentLoaded', () => loadDemoLibraries());
      function upperAgentConfigPayload(form) {
        const payload = Object.fromEntries(new FormData(form).entries());
        delete payload.run_name;
        payload.enabled = form.enabled.checked;
        payload.auto_apply_instruction = form.auto_apply_instruction.checked;
        payload.pause_policy_while_thinking = form.pause_policy_while_thinking.checked;
        payload.enable_route_memory = form.enable_route_memory.checked;
        payload.enable_long_term_memory = form.enable_long_term_memory.checked;
        payload.enable_graph_memory_capture = form.enable_graph_memory_capture.checked;
        payload.safety_mode = form.safety_mode.checked;
        payload.auto_speak_task_feedback = form.auto_speak_task_feedback.checked;
        ['read_every_n_frames', 'history_events', 'max_memory_items', 'max_subgoal_age_frames', 'max_tokens', 'max_image_width', 'long_term_memory_top_k', 'long_term_memory_char_budget', 'long_term_memory_timeout_ms'].forEach((key) => {
          payload[key] = Number(payload[key]);
        });
        ['min_seconds_between_calls', 'replan_settle_seconds', 'temperature'].forEach((key) => {
          payload[key] = Number(payload[key]);
        });
        return payload;
      }
      async function saveUpperAgentConfigForm(reload=true) {
        const form = document.getElementById('upperAgentForm');
        const status = document.getElementById('upperAgentStatus');
        if (!form) throw new Error('Upper Agent 配置表单不存在。');
        if (status) status.innerText = 'Applying upper agent config...';
        const response = await fetch('/api/upper-agent/config', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(upperAgentConfigPayload(form))
        });
        const data = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(data.error || 'Upper Agent config failed');
        if (status) status.innerText = 'Applied ' + (data.updated_at || new Date().toLocaleTimeString());
        if (reload) setTimeout(() => window.location.reload(), 400);
        return data;
      }
      async function autoApplyVoiceInstruction(recording) {
        if (recording.targetKind === 'upper_task') {
          const upperForm = document.getElementById('upperAgentForm');
          if (!upperForm) throw new Error('Upper Agent 配置表单不存在。');
          upperForm.enabled.checked = true;
          upperForm.auto_apply_instruction.checked = true;
          await saveUpperAgentConfigForm(false);
        } else {
          await saveRuntimeConfigForm(false);
        }
        await setServiceEnabled(true, false);
        setTimeout(() => window.location.reload(), 650);
      }
      async function submitUpperAgentConfig(event) {
        event.preventDefault();
        const status = document.getElementById('upperAgentStatus');
        try { await saveUpperAgentConfigForm(true); }
        catch (error) { if (status) status.innerText = error.message; }
      }
      async function runUpperAgentNow(runName, force=true) {
        if (window.upperAgentBusy) return;
        window.upperAgentBusy = true;
        const status = document.getElementById('upperAgentStatus');
        try {
          if (!runName) {
            if (status) status.innerText = 'Open a run before evaluating upper agent';
            return;
          }
          if (status) status.innerText = 'Calling upper agent...';
          const response = await fetch('/api/upper-agent/evaluate/' + encodeURIComponent(runName), {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({force, settle: true})
          });
          const data = await response.json().catch(() => ({}));
          if (!response.ok || !data.ok) {
            if (status) status.innerText = data.error || 'Upper agent failed';
            return;
          }
          if (data.skipped) {
            if (status) status.innerText = 'Skipped: ' + data.reason;
            return;
          }
          if (status) status.innerText = 'Upper agent updated frame ' + data.event.frame_idx;
          renderUpperAgentDecision(data.event);
        } finally {
          window.upperAgentBusy = false;
        }
      }
      function renderUpperAgentDecision(event) {
        const panel = document.getElementById('upperAgentLatestPanel');
        const line = document.getElementById('upperAgentLatestLine');
        if (!panel || !event) return;
        const output = event.output || {};
        const memory = event.route_memory || {};
        const demo = output.demo_agent || null;
        const decision = (demo && demo.decision) || output.demo_step_decision || '-';
        const observedFile = event.image_file || '';
        const referenceFile = event.demo_reference_image_file || '';
        const motionFrames = event.motion_context_frames || [];
        const executionEvidence = event.demo_execution_evidence || null;
        const assessment = output.execution_assessment || {};
        const runName = document.querySelector('#upperAgentForm input[name="run_name"]')?.value || '';
        const frameLink = (filename) => {
          if (!filename) return '-';
          const label = configEscapeHtml(filename);
          if (!runName) return label;
          return `<a href="/file/${encodeURIComponent(runName)}/${encodeURIComponent(filename)}" target="_blank" rel="noopener">${label}</a>`;
        };
        const trajectoryEvidence = executionEvidence ?
          `required=${executionEvidence.required_turn || 'none'}; matching outputs=${executionEvidence.matching_turn_output_count || 0}; turn segments=${executionEvidence.matching_turn_segment_count || 0}; visual change=${Number(executionEvidence.endpoint_visual_change_score || 0).toFixed(2)}; mode=${executionEvidence.transition_mode || 'balanced'}; sampled=[${(executionEvidence.sampled_frame_outputs || []).map(item => `${item.frame_idx ?? '?'}:${item.planned_motion || 'unknown'}`).join(', ')}]` : '-';
        const demoRows = demo ? `
            <tr><td>Demo library</td><td>${configEscapeHtml(demo.library_name || '')}</td></tr>
            <tr><td>Demo progress</td><td>${configEscapeHtml(demo.step_number || '-')} / ${configEscapeHtml(demo.total_steps || '-')}</td></tr>` : '';
        if (line) line.innerText = event.created_at || 'updated';
        panel.innerHTML = `
          <table class="kvTable"><tbody>
            <tr><td>Agent observed frame</td><td>${configEscapeHtml(event.frame_idx ?? 'N/A')} · ${frameLink(observedFile)}</td></tr>
            <tr><td>Step-start reference frame</td><td>${frameLink(referenceFile)}</td></tr>
            <tr><td>Motion sequence frames</td><td>${configEscapeHtml(motionFrames.map(item => item.frame_idx ?? '?').join(' → ') || '-')}</td></tr>
            <tr><td>Trajectory evidence</td><td>${configEscapeHtml(trajectoryEvidence)}</td></tr>
            <tr><td>Execution assessment</td><td>${configEscapeHtml(assessment.reason || '-')}</td></tr>
            <tr><td>Completion confidence</td><td>${Number(assessment.completion_confidence || 0).toFixed(2)} · completed=${Boolean(assessment.subgoal_completed)} · turn=${configEscapeHtml(assessment.observed_turn_direction || 'uncertain')} · turn_completed=${Boolean(assessment.turn_completed)}</td></tr>
            <tr><td>Decision</td><td><strong>${configEscapeHtml(decision)}</strong></td></tr>
            <tr><td>Status</td><td>${configEscapeHtml(output.task_status || '')}</td></tr>
            <tr><td>Agent task</td><td>${configEscapeHtml(event.task_instruction || '')}</td></tr>
            <tr><td>Phase</td><td>${configEscapeHtml(output.navigation_phase || '')}</td></tr>
            <tr><td>Executable subgoal</td><td>${configEscapeHtml(output.current_subgoal || '')}</td></tr>
            ${demoRows}
            <tr><td>Evidence</td><td>${configEscapeHtml(output.visual_evidence || '')}</td></tr>
            <tr><td>Memory place</td><td>${configEscapeHtml(memory.current_place || '')}</td></tr>
            <tr><td>Memory hint</td><td>${configEscapeHtml(memory.next_direction_hint || '')}</td></tr>
            <tr><td>Visited</td><td>${configEscapeHtml((memory.visited_landmarks || []).slice(-5).join(', '))}</td></tr>
            <tr><td>Call time</td><td>${Number(event.call_time || 0).toFixed(2)}s</td></tr>
          </tbody></table>
          ${renderTaskFeedback(event)}`;
        maybeAutoSpeakTaskFeedback(
          output.task_status || '',
          event.frame_idx ?? '',
          ((output.task_feedback || {}).summary || (output.task_feedback || {}).failure_reason || '')
        );
      }
      function renderTaskFeedback(event) {
        const output = event.output || {};
        const feedback = output.task_feedback || {};
        const report = event.task_report || {};
        const summary = feedback.summary || report.summary || '';
        const failure = feedback.failure_reason || report.failure_reason || '';
        const recommendation = feedback.recommendation || report.recommendation || '';
        const count = feedback.count ?? report.count;
        const countLabel = feedback.count_label || report.count_label || 'count';
        const findings = report.findings || feedback.findings || [];
        if (!summary && !failure && !recommendation && count == null && !findings.length) {
          return '<div class="taskFeedbackReport"><p class="status">尚无任务反馈；智能体会在观察、计数、巡检或任务结束时更新这里。</p></div>';
        }
        const status = output.task_status || report.task_status || 'running';
        const spoken = summary || failure;
        const findingHtml = findings.map((item) => `
          <div class="taskFinding severity-${configEscapeHtml(item.severity || 'info')}">
            <strong>${configEscapeHtml(item.description || '')}</strong>
            <span>${configEscapeHtml(item.type || 'observation')}${item.location ? ' · ' + configEscapeHtml(item.location) : ''}</span>
            ${item.evidence ? '<small>' + configEscapeHtml(item.evidence) + '</small>' : ''}
          </div>`).join('');
        return `<section id="taskFeedbackReport" class="taskFeedbackReport status-${configEscapeHtml(status)}"
                         data-status="${configEscapeHtml(status)}" data-frame="${configEscapeHtml(event.frame_idx ?? '')}"
                         data-feedback="${configEscapeHtml(spoken)}">
          <div class="taskFeedbackHeader"><div><span>Task Feedback</span><strong>${configEscapeHtml(status.toUpperCase())}</strong></div>
            <button type="button" onclick="speakTaskFeedback(this.closest('.taskFeedbackReport').dataset.feedback)">播报反馈</button></div>
          ${summary ? '<p class="taskFeedbackSummary">' + configEscapeHtml(summary) + '</p>' : ''}
          ${count != null ? '<div class="taskFeedbackCount"><strong>' + configEscapeHtml(count) + '</strong><span>' + configEscapeHtml(countLabel) + '</span></div>' : ''}
          ${failure ? '<div class="taskFeedbackFailure"><strong>失败原因</strong><span>' + configEscapeHtml(failure) + '</span></div>' : ''}
          ${recommendation ? '<div class="taskFeedbackRecommendation"><strong>建议</strong><span>' + configEscapeHtml(recommendation) + '</span></div>' : ''}
          ${findingHtml ? '<div class="taskFindingGrid">' + findingHtml + '</div>' : ''}
        </section>`;
      }
      function speakTaskFeedback(text) {
        text = String(text || '').trim();
        if (!text || !window.speechSynthesis) return;
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = /[\u3400-\u9fff]/.test(text) ? 'zh-CN' : 'en-US';
        utterance.rate = 1.0;
        window.speechSynthesis.speak(utterance);
      }
      function maybeAutoSpeakTaskFeedback(status, frame, text) {
        const form = document.getElementById('upperAgentForm');
        const enabled = form && form.auto_speak_task_feedback && form.auto_speak_task_feedback.checked;
        if (!enabled || !['completed', 'failed'].includes(String(status)) || !String(text || '').trim()) return;
        const key = `internnav-task-feedback:${location.pathname}:${frame}:${status}`;
        try {
          if (localStorage.getItem(key)) return;
          localStorage.setItem(key, 'spoken');
        } catch (_) {}
        speakTaskFeedback(text);
      }
      document.addEventListener('DOMContentLoaded', () => {
        const report = document.getElementById('taskFeedbackReport');
        if (report) maybeAutoSpeakTaskFeedback(report.dataset.status, report.dataset.frame, report.dataset.feedback);
      });
      function applyTransportPreset(value, updateStatus=true) {
        const form = document.getElementById('serviceLauncherForm');
        if (!form) return;
        if (value === 'http') {
          form.transport.value = 'http';
        } else {
          form.transport.value = 'zenoh';
          if (value === 'zenoh_udp') {
            form.zenoh_mode.value = 'peer';
            form.zenoh_connect.value = '';
            form.zenoh_listen.value = 'udp/0.0.0.0:7447';
            form.zenoh_no_multicast_scouting.checked = true;
          }
        }
        const status = document.getElementById('serviceStatus');
        if (updateStatus && status) {
          status.innerText = value === 'http'
            ? 'HTTP selected'
            : (value === 'zenoh_udp' ? 'Zenoh UDP selected' : 'Zenoh custom selected');
        }
      }
      document.addEventListener('submit', function(event) {
        if (event.target && event.target.id === 'runtimeConfigForm') submitRuntimeConfig(event);
        if (event.target && event.target.id === 'serviceLauncherForm') submitServiceLauncher(event);
        if (event.target && event.target.id === 'upperAgentForm') submitUpperAgentConfig(event);
      });
    """


def render_robot_companion():
    """Minimal floating visual companion shared by index and run pages."""
    return """
      <aside class="roverDock" id="roverDock" aria-label="实验记录助手">
        <button class="roverRobot" id="roverRobot" type="button" onclick="roverCharge()" title="连续点击为 Rover 充电">
          <i class="roverHead"><i class="roverEye"></i></i><i class="roverBody"></i><i class="roverTail"></i>
          <i class="roverLeg a"></i><i class="roverLeg b"></i><i class="roverLeg c"></i><i class="roverLeg d"></i><i class="roverHeart"></i><i class="roverChargeLine"></i><i class="roverListenWave"></i>
        </button>
        <i class="roverBlastRing" id="roverBlastRing" aria-hidden="true"></i>
        <button class="roverTop" type="button" onclick="window.scrollTo({top: 0, behavior: 'smooth'})">回到顶部</button>
      </aside>
      <section class="roverSearchPanel" id="roverSearchPanel" aria-label="实验记录查询" aria-hidden="true">
        <div class="roverSearchHead"><strong>实验记录查询</strong><button class="roverSearchClose" type="button" onclick="closeRoverSearch()" title="关闭查询面板" aria-label="关闭查询面板">&times;</button></div>
        <form class="roverSearchForm" onsubmit="searchRoverRuns(event)"><input id="roverSearchInput" type="search" placeholder="任务、指令或实验编号"><button type="submit">查询</button></form>
        <div class="roverSearchFilters"><button type="button" onclick="searchRoverRuns(null, 'latest')">最新</button><button type="button" onclick="searchRoverRuns(null, 'success')">成功</button><button type="button" onclick="searchRoverRuns(null, 'failed')">失败</button></div>
        <div class="roverSearchResults" id="roverSearchResults"><div class="roverSearchEmpty">输入关键词，或选择一个快速筛选条件。</div></div>
      </section>
    """


def render_robot_companion_script():
    return """
      let roverPoseTimer = null, roverRandomTimer = null, roverChargeResetTimer = null, roverBoneTimer = null;
      let roverCharging = false, roverChargeClicks = 0, roverExploding = false;
      let roverInteractionState = 'idle';
      function roverEscape(value) { const node = document.createElement('span'); node.innerText = String(value || ''); return node.innerHTML; }
      function closeRoverSearch() {
        const panel = document.getElementById('roverSearchPanel');
        if (!panel) return; panel.classList.remove('isOpen'); panel.setAttribute('aria-hidden', 'true');
      }
      async function searchRoverRuns(event, filter='') {
        if (event) event.preventDefault();
        const input = document.getElementById('roverSearchInput'); const results = document.getElementById('roverSearchResults');
        if (!results) return; results.innerHTML = '<div class="roverSearchEmpty">正在查询实验记录...</div>';
        try {
          const query = new URLSearchParams(); if (input && input.value.trim()) query.set('q', input.value.trim()); if (filter) query.set('filter', filter);
          const response = await fetch('/api/runs/search?' + query.toString()); const payload = await response.json();
          if (!response.ok || !payload.ok) throw new Error(payload.error || '查询失败');
          const items = payload.results || [];
          results.innerHTML = items.length ? items.map((item) => `<a class="roverSearchResult" href="${roverEscape(item.url)}"><b>${roverEscape(item.name)} · ${roverEscape(item.outcome)}</b><small>${roverEscape(item.mode)} | ${roverEscape(item.instruction)}</small></a>`).join('') : '<div class="roverSearchEmpty">没有找到匹配的实验记录。</div>';
        } catch (error) { results.innerHTML = `<div class="roverSearchEmpty">查询失败：${roverEscape(error.message)}</div>`; }
      }
      function openRoverSearch() {
        const panel = document.getElementById('roverSearchPanel');
        if (!panel) return; panel.classList.add('isOpen'); panel.setAttribute('aria-hidden', 'false'); searchRoverRuns(null, 'latest');
      }
      function roverCharge() {
        const robot = document.getElementById('roverRobot'); if (!robot || roverExploding || roverInteractionState !== 'idle') return;
        clearTimeout(roverPoseTimer); clearTimeout(roverChargeResetTimer); roverCharging = true; roverChargeClicks += 1;
        robot.className = 'roverRobot isCharging'; robot.style.setProperty('--charge-level', String(Math.min(1, roverChargeClicks / 8)));
        if (roverChargeClicks >= 8) {
          roverExploding = true; roverCharging = false; robot.className = 'roverRobot isBone'; robot.style.removeProperty('--charge-level');
          const blastRing = document.getElementById('roverBlastRing'); if (blastRing) { blastRing.classList.remove('isActive'); void blastRing.offsetWidth; blastRing.classList.add('isActive'); }
          openRoverSearch();
          roverBoneTimer = window.setTimeout(() => { roverExploding = false; roverChargeClicks = 0; robot.className = 'roverRobot'; if (blastRing) blastRing.classList.remove('isActive'); }, 5000);
          return;
        }
        roverChargeResetTimer = window.setTimeout(() => { roverCharging = false; roverChargeClicks = 0; robot.className = 'roverRobot'; robot.style.removeProperty('--charge-level'); }, 2000);
      }
      function roverPose(name) {
        const robot = document.getElementById('roverRobot'); if (!robot || roverInteractionState !== 'idle') return;
        clearTimeout(roverPoseTimer); robot.className = 'roverRobot pose-' + name;
        roverPoseTimer = window.setTimeout(() => { if (!roverCharging && !roverExploding) robot.className = 'roverRobot'; }, 5000);
      }
      function setRoverInteractionState(state) {
        const robot = document.getElementById('roverRobot');
        if (!robot || roverExploding) return;
        roverInteractionState = state || 'idle';
        clearTimeout(roverPoseTimer);
        if (roverInteractionState === 'listening') {
          roverCharging = false; roverChargeClicks = 0; clearTimeout(roverChargeResetTimer);
          robot.style.removeProperty('--charge-level');
          robot.className = 'roverRobot isListening';
          robot.title = 'Rover 正在倾听你的指令';
          robot.setAttribute('aria-label', 'Rover 正在倾听');
        } else if (roverInteractionState === 'processing') {
          robot.className = 'roverRobot isProcessing';
          robot.title = 'Rover 正在处理语音指令';
          robot.setAttribute('aria-label', 'Rover 正在处理语音指令');
        } else {
          roverInteractionState = 'idle';
          robot.className = 'roverRobot';
          robot.title = '连续点击为 Rover 充电';
          robot.setAttribute('aria-label', '实验记录助手');
        }
      }
      roverRandomTimer = window.setInterval(() => { if (!roverCharging && !roverExploding && roverInteractionState === 'idle') roverPose(['sit', 'lie', 'paw', 'heart'][Math.floor(Math.random() * 4)]); }, 20000);
    """


def draw_text_panel(draw, xy, lines, fill=(0, 0, 0), text_fill=(255, 255, 255), padding=8, line_h=18):
    # 简单文本面板，用于在图像上叠加 frame、instruction、动作等信息。
    x, y = xy
    width = max(220, max((len(line) for line in lines), default=0) * 7 + padding * 2)
    height = padding * 2 + line_h * len(lines)
    draw.rectangle([x, y, x + width, y + height], fill=fill)
    for i, line in enumerate(lines):
        draw.text((x + padding, y + padding + i * line_h), line, fill=text_fill)


def trajectory_to_points(trajectory):
    # 将 json 中的 trajectory 安全转换成 [N, 2] numpy 数组。
    if trajectory is None:
        return None
    points = np.asarray(trajectory, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 2:
        return None
    return points[:, :2]


def project_local_trajectory_to_image(trajectory, image_size):
    # 调试用近似投影：局部 x 视为前方，局部 y 视为左右，画在第一视角图像上。
    points = trajectory_to_points(trajectory)
    if points is None:
        return []

    width, height = image_size
    xy = points[:, :2]
    forward = xy[:, 0]
    lateral = xy[:, 1]
    max_forward = max(float(np.max(np.abs(forward))), 0.35)
    max_lateral = max(float(np.max(np.abs(lateral))), 0.25)
    scale_forward = height * 0.52 / max_forward
    scale_lateral = width * 0.34 / max_lateral
    scale = min(scale_forward, scale_lateral)

    origin = np.array([width * 0.5, height * 0.88], dtype=np.float32)
    projected = []
    for x, y in xy:
        px = origin[0] - y * scale
        py = origin[1] - x * scale
        projected.append((float(np.clip(px, 0, width - 1)), float(np.clip(py, 0, height - 1))))
    return projected


def draw_image_trajectory(draw, trajectory, image_size):
    # 将局部轨迹近似画到 RGB 图像上；这是调试可视化，不是严格几何投影。
    projected = project_local_trajectory_to_image(trajectory, image_size)
    if len(projected) < 2:
        return
    draw.line(projected, fill=(0, 0, 0), width=8, joint="curve")
    draw.line(projected, fill=(0, 220, 255), width=5, joint="curve")
    start = projected[0]
    end = projected[-1]
    draw.ellipse([start[0] - 6, start[1] - 6, start[0] + 6, start[1] + 6], fill=(0, 210, 80), outline=(0, 0, 0), width=2)
    draw.ellipse([end[0] - 8, end[1] - 8, end[0] + 8, end[1] + 8], fill=(255, 70, 40), outline=(0, 0, 0), width=2)


def draw_trajectory_inset(draw, trajectory, image_size):
    # 右上角 top-down 局部轨迹视图，x 为前方，y 为左右方向。
    points = trajectory_to_points(trajectory)
    if points is None:
        return

    width, _ = image_size
    inset_w, inset_h = 250, 205
    margin = 18
    left = max(margin, width - inset_w - margin)
    top = margin
    right = left + inset_w
    bottom = top + inset_h
    draw.rectangle([left, top, right, bottom], fill=(250, 250, 250), outline=(20, 20, 20), width=2)
    draw.text((left + 10, top + 8), "local trajectory", fill=(20, 20, 20))
    draw.text((left + 10, bottom - 20), "x forward, y lateral", fill=(80, 80, 80))

    plot_left, plot_top = left + 22, top + 34
    plot_right, plot_bottom = right - 18, bottom - 30
    for i in range(1, 4):
        gx = plot_left + (plot_right - plot_left) * i / 4
        gy = plot_top + (plot_bottom - plot_top) * i / 4
        draw.line([gx, plot_top, gx, plot_bottom], fill=(220, 220, 220))
        draw.line([plot_left, gy, plot_right, gy], fill=(220, 220, 220))

    min_xy = points.min(axis=0)
    max_xy = points.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1e-3)
    scale = min((plot_right - plot_left) / span[0], (plot_bottom - plot_top) / span[1])
    center = (min_xy + max_xy) / 2.0
    canvas_center = np.array([(plot_left + plot_right) / 2.0, (plot_top + plot_bottom) / 2.0])

    projected = []
    for x, y in points:
        px = canvas_center[0] + (x - center[0]) * scale
        py = canvas_center[1] - (y - center[1]) * scale
        projected.append((float(px), float(py)))
    draw.line(projected, fill=(40, 110, 230), width=3, joint="curve")
    draw.ellipse([projected[0][0] - 4, projected[0][1] - 4, projected[0][0] + 4, projected[0][1] + 4], fill=(0, 180, 0))
    draw.ellipse([projected[-1][0] - 5, projected[-1][1] - 5, projected[-1][0] + 5, projected[-1][1] + 5], fill=(230, 40, 40))


def draw_pixel_goal(draw, pixel_goal):
    # 在 RGB 图上标出 System 2 预测的图像像素目标。
    if pixel_goal is None or len(pixel_goal) < 2:
        return
    x, y = int(pixel_goal[0]), int(pixel_goal[1])
    radius = 11
    draw.ellipse([x - radius, y - radius, x + radius, y + radius], outline=(255, 45, 45), width=4)
    draw.line([x - 18, y, x + 18, y], fill=(255, 45, 45), width=3)
    draw.line([x, y - 18, x, y + 18], fill=(255, 45, 45), width=3)
    draw.text((x + 14, max(0, y - 14)), f"pixel_goal {pixel_goal}", fill=(255, 45, 45))


def draw_action_bar(draw, actions, image_size):
    # 将离散动作序列画成底部状态条。
    if not actions:
        return
    width, height = image_size
    labels = [ACTION_LABELS.get(int(action), str(action)) for action in actions]
    text = "  ".join(labels)
    draw_text_panel(draw, (12, height - 44), [f"discrete_action: {text}"], fill=(20, 20, 20), text_fill=(255, 235, 90))


def draw_frame_visualization(rgb, json_output, frame_idx, instruction):
    # 生成增强可视化图：RGB 上叠加近似轨迹、pixel goal、动作条和 top-down 小地图。
    vis = Image.fromarray(rgb).convert("RGB")
    draw = ImageDraw.Draw(vis)
    width, height = vis.size
    instruction_line = instruction[:140] if instruction else ""
    draw_text_panel(
        draw,
        (0, 0),
        [f"frame={frame_idx}", instruction_line],
        fill=(0, 0, 0),
        text_fill=(240, 240, 240),
        padding=10,
    )

    trajectory = json_output.get("trajectory")
    draw_image_trajectory(draw, trajectory, vis.size)
    draw_pixel_goal(draw, json_output.get("pixel_goal"))
    draw_action_bar(draw, json_output.get("discrete_action"), vis.size)

    if trajectory is not None:
        draw.text((12, height - 68), "cyan path: approximate local trajectory overlay", fill=(0, 240, 255))
    return vis


def save_experiment_frame(
    run_dir,
    frame_idx,
    rgb,
    depth,
    request_data,
    instruction,
    json_output,
    generate_time,
    agent_debug=None,
    runtime_config=None,
):
    # 保存当前帧的原始输入、模型输出和增强可视化结果。
    # 一个 frame 会生成：
    # - frame_xxxxxx_rgb.jpg：狗传来的 RGB。
    # - frame_xxxxxx_depth.png：深度图，按 10000 倍存成 uint16。
    # - frame_xxxxxx_vis.jpg：叠加动作/轨迹/文字后的可视化。
    # - frame_xxxxxx_waypoint.json：本帧完整元数据，供网页和 Upper Agent 读取。
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    frame_stem = f"frame_{frame_idx:06d}"

    rgb_file = f"{frame_stem}_rgb.jpg"
    depth_file = f"{frame_stem}_depth.png"
    vis_file = f"{frame_stem}_vis.jpg"
    waypoint_file = f"{frame_stem}_waypoint.json"

    Image.fromarray(rgb).save(run_dir / rgb_file, quality=95)
    depth_uint16 = np.clip(depth * 10000.0, 0, 65535).astype(np.uint16)
    Image.fromarray(depth_uint16).save(run_dir / depth_file)
    draw_frame_visualization(rgb, json_output, frame_idx, instruction).save(run_dir / vis_file, quality=95)

    metadata = {
        "frame_idx": frame_idx,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "instruction": instruction,
# instruction 是低层大脑本帧实际收到的短指令。
        "agent_task_instruction": (runtime_config or {}).get("upper_agent", {}).get("task_instruction", ""),
# agent_task_instruction 是上层智能体的总任务，和低层短指令分开保存，便于复盘。
        "request_json": request_data,
        "response": json_output,
        "generate_time": generate_time,
        "runtime_config": make_json_safe(runtime_config or {}),
        "rgb_file": rgb_file,
        "depth_file": depth_file,
        "vis_file": vis_file,
        "agent_debug": agent_debug or {},
    }
    with open(run_dir / waypoint_file, "w") as f:
        json.dump(make_json_safe(metadata), f, indent=2, ensure_ascii=False)
    return run_dir / waypoint_file


class ExperimentLogger:
    # Server 侧使用的轻量实验记录器。
    def __init__(self, log_dir="output/realworld_experiments", save_frame_interval=0):
        self.log_dir = Path(log_dir).expanduser().resolve()
        self.save_frame_interval = int(save_frame_interval)
        self.run_dir = None

    @property
    def enabled(self):
        return self.save_frame_interval > 0

    def new_run(self):
        if not self.enabled:
            return None
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.log_dir / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        return self.run_dir

    def ensure_run(self):
        if not self.enabled:
            return None
        if self.run_dir is None:
            return self.new_run()
        return self.run_dir

    def should_save_frame(self, frame_idx):
# save_frame_interval 控制保存频率；0 表示不保存，1 表示每帧保存。
        return self.enabled and (frame_idx - 1) % self.save_frame_interval == 0

    def save_frame(
        self, frame_idx, rgb, depth, request_data, instruction, json_output, generate_time, agent_debug=None, runtime_config=None
    ):
        if not self.should_save_frame(frame_idx):
            return None
        run_dir = self.ensure_run()
        if run_dir is None:
            return None
        return save_experiment_frame(
            run_dir, frame_idx, rgb, depth, request_data, instruction, json_output, generate_time, agent_debug, runtime_config
        )


def refresh_run_visualizations(run_dir):
    # 对已有实验包重画增强可视化图，适合给旧数据补新版可视化。
    run_dir = Path(run_dir).expanduser().resolve()
    updated = 0
    for metadata_path in list_frame_metadata(run_dir):
        metadata = load_metadata(metadata_path)
        rgb_path = run_dir / metadata.get("rgb_file", metadata_path.name.replace("_waypoint.json", "_rgb.jpg"))
        if not rgb_path.exists():
            continue
        rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
        vis = draw_frame_visualization(
            rgb,
            metadata.get("response", {}),
            metadata.get("frame_idx", frame_sort_key(metadata_path)),
            metadata.get("instruction", ""),
        )
        vis_file = metadata.get("vis_file", metadata_path.name.replace("_waypoint.json", "_vis.jpg"))
        vis.save(run_dir / vis_file, quality=95)
        updated += 1
    return updated


def create_gif_from_run(run_dir, source="vis", output_path=None, duration_ms=250, max_width=960):
    # 将某次实验的连续 jpg/png 帧合成 GIF。
    run_dir = Path(run_dir).expanduser().resolve()
    pattern = f"frame_*_{source}.jpg" if source in {"rgb", "vis"} else f"frame_*_{source}.png"
    frame_paths = sorted(run_dir.glob(pattern), key=frame_sort_key)
    if not frame_paths:
        raise FileNotFoundError(f"No frames found for source='{source}' in {run_dir}")
    if output_path is None:
        output_path = run_dir / f"{source}_feedback.gif"
    else:
        output_path = Path(output_path).expanduser().resolve()

    frames = []
    for path in frame_paths:
        image = Image.open(path).convert("RGB")
        if max_width and image.width > max_width:
            new_height = int(image.height * max_width / image.width)
            image = image.resize((max_width, new_height))
        frames.append(image)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    return output_path


def safe_run_dir(log_dir, run_name):
    # 限制 Web 路由只能访问实验根目录下的 run 包。
    root = Path(log_dir).expanduser().resolve()
    run_dir = (root / run_name).resolve()
    if root not in run_dir.parents and run_dir != root:
        abort(404)
    if not run_dir.is_dir():
        abort(404)
    return run_dir


def gif_is_stale(output_path, frame_paths):
    # 如果 GIF 不存在，或有新帧比 GIF 更新，则需要重新生成。
    output_path = Path(output_path)
    if not output_path.exists():
        return True
    if not frame_paths:
        return False
    latest_frame_mtime = max(path.stat().st_mtime for path in frame_paths)
    return output_path.stat().st_mtime < latest_frame_mtime


def create_viewer_app(log_dir, runtime_config_path=None):
    # 创建可交互查看实验包的 Flask Web app。
    # 这个 app 不直接跑模型推理，而是：
    # - 读写 runtime_config.json；
    # - 调 realworld_service_launcher 启停 HTTP/Zenoh 模型服务；
    # - 读取 output/realworld_experiments 下的实验帧；
    # - 调 upper_agent.evaluate_latest 做上层智能体决策。
    app = Flask(__name__)
    root = Path(log_dir).expanduser().resolve()
    config_path = Path(runtime_config_path or default_runtime_config_path(root)).expanduser().resolve()
    demo_library_path = default_demo_library_path(config_path)
    asset_root = Path(__file__).resolve().parents[2] / "assets"
    repo_root = Path(__file__).resolve().parents[2]
    service_launcher = RealworldServiceLauncher(
        repo_root=repo_root,
        runtime_config_path=config_path,
        experiment_log_dir=root,
    )
    local_qwen_launcher = LocalQwenServiceLauncher(
        repo_root=repo_root,
        state_path=root.parent / "local_qwen_service_state.json",
    )
    speech_transcriber = SpeechTranscriber()
    hero_asset = "InternNav.gif"

    def load_viewer_config():
# 每次请求都从磁盘读取最新 runtime_config，这样网页和模型服务可异步共享配置。
        config = load_runtime_config(config_path)
        config.setdefault("runtime_config_path", str(config_path))
        return config

    def public_upper_agent_config(config):
# 给浏览器看的 Upper Agent 配置，必须删除真实 api_key，只保留是否已配置的布尔信息。
        upper = get_upper_agent_config(config)
        upper["credential_configured"] = bool(get_upper_agent_config(config).get("api_key"))
        upper.pop("api_key", None)
        return upper

    def render_waiting_page(after_run=""):
        config = load_viewer_config()
        service_status = service_launcher.status()
        return f"""
        <html>
        <head>
          <title>InternNav Live</title>
          <style>{render_common_styles()}</style>
        </head>
        <body>
          <div class="topbar">
            <div><a href="{url_for('index')}">Runs</a> / <b>Live</b> / <a href="{url_for('camera_monitor')}">Camera Monitor</a></div>
            <div class="status" id="waitStatus">Waiting</div>
          </div>
          <div class="shell">
            <div class="hero">
              <div class="heroGrid">
                <div>
                  <h2>InternNav Live Control</h2>
                  <p>Log dir: <code>{html.escape(str(root))}</code></p>
                </div>
                <div class="heroMedia"><img src="{url_for('asset_view', filename=hero_asset)}" alt="InternNav" loading="lazy" decoding="async"></div>
              </div>
            </div>
            <div class="panel emptyState">
              <div>
                <h2><span class="pulse"></span>Waiting for a new experiment</h2>
                <p class="status" id="waitHint">Latest completed run will stay untouched until a new frame arrives.</p>
              </div>
            </div>
            <div class="panel" style="margin-top:16px">
              <h3>Model Service</h3>
              {render_service_launcher_panel(service_status)}
              {render_local_qwen_panel(local_qwen_launcher.status())}
            </div>
            <div class="panel" style="margin-top:16px">
              <h3>Runtime Config</h3>
              {render_runtime_config_panel(config)}
            </div>
            <div class="panel" style="margin-top:16px">
              <h3>Upper-Level Agent</h3>
              {render_upper_agent_panel(config)}
              {render_demo_agent_panel(config)}
            </div>
          </div>
          <script>
            const afterRun = "{html.escape(after_run)}";
            async function pollForNewRun() {{
              try {{
                const response = await fetch("{url_for('api_runs')}", {{ cache: 'no-store' }});
                if (!response.ok) return;
                const runs = await response.json();
                if (runs.length && runs[0].frames > 0 && runs[0].name !== afterRun) {{
                  window.location = "/run/" + runs[0].name + "?live=1";
                  return;
                }}
                document.getElementById('waitStatus').innerText = 'Waiting ' + new Date().toLocaleTimeString();
              }} catch (err) {{
                document.getElementById('waitStatus').innerText = 'Polling error';
              }}
            }}
            setInterval(pollForNewRun, 1500);
            {render_config_script()}
          </script>
        </body>
        </html>
        """

    @app.route("/asset/<path:filename>")
    def asset_view(filename):
        return send_from_directory(asset_root, filename)

    @app.route("/api/ping")
    def api_ping():
        return jsonify({"ok": True, "time": datetime.now().isoformat(timespec="seconds")})

    @app.route("/api/speech/status")
    def api_speech_status():
        config = load_viewer_config()
        return jsonify(
            {
                "ok": True,
                **speech_backend_status(
                    config.get("speech_to_text_model"),
                    backend=config.get("speech_to_text_backend"),
                    device=config.get("speech_to_text_device"),
                    funasr_model_path=config.get("funasr_model_path"),
                    sensevoice_model_path=config.get("sensevoice_model_path"),
                ),
                **speech_transcriber.status(),
            }
        )

    @app.route("/api/speech/transcribe", methods=["POST"])
    def api_speech_transcribe():
        audio = request.files.get("audio")
        if audio is None:
            return jsonify({"ok": False, "error": "请求中缺少 audio 录音文件。"}), 400
        if request.content_length and request.content_length > MAX_AUDIO_BYTES + 1024 * 1024:
            return jsonify({"ok": False, "error": "录音文件过大，请将单次录音控制在 60 秒以内。"}), 413
        try:
            request_started_at = time.perf_counter()
            viewer_config = load_viewer_config()
            transcription_started_at = time.perf_counter()
            audio_bytes = audio.stream.read(MAX_AUDIO_BYTES + 1)
            result = speech_transcriber.transcribe(
                audio_bytes,
                filename=audio.filename or "recording.webm",
                language=request.form.get("language"),
                model_name=viewer_config.get("speech_to_text_model"),
                backend=viewer_config.get("speech_to_text_backend"),
                device=viewer_config.get("speech_to_text_device"),
                funasr_model_path=viewer_config.get("funasr_model_path"),
                sensevoice_model_path=viewer_config.get("sensevoice_model_path"),
            )
            result["transcription_seconds"] = time.perf_counter() - transcription_started_at
            result["transcript"] = result["text"]
            result["refined"] = False
            interaction_mode = str(request.form.get("interaction_mode") or "").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            if interaction_mode:
                upper_config = get_upper_agent_config(viewer_config)
                voice_language_model = str(viewer_config.get("voice_language_model") or "").strip()
                if voice_language_model:
                    upper_config["model"] = voice_language_model
                confidence_threshold = viewer_config.get("voice_command_confidence_threshold", 0.78)
                semantic_started_at = time.perf_counter()
                decision = classify_spoken_navigation_command(
                    upper_config,
                    result["transcript"],
                    target=str(request.form.get("target") or "low_level"),
                    confidence_threshold=confidence_threshold,
                )
                result["semantic_seconds"] = time.perf_counter() - semantic_started_at
                result.update(decision)
                result["interaction_mode"] = True
                result["applied"] = False
                if decision["accepted"]:
                    saved, applied, apply_reason = apply_voice_interaction_command(
                        config_path,
                        decision["instruction"],
                        command_type=decision.get("command_type"),
                    )
                    result["text"] = decision["instruction"]
                    result["instruction"] = decision["instruction"]
                    result["applied"] = applied
                    result["apply_reason"] = apply_reason
                    result["service_enabled"] = bool(saved.get("service_enabled"))
                result["total_seconds"] = time.perf_counter() - request_started_at
                return jsonify({"ok": True, **result})
            should_refine = str(request.form.get("refine") or "").strip().lower() in {"1", "true", "yes", "on"}
            if should_refine:
                try:
                    upper_config = get_upper_agent_config(load_viewer_config())
                    voice_language_model = str(viewer_config.get("voice_language_model") or "").strip()
                    if voice_language_model:
                        upper_config["model"] = voice_language_model
                    semantic_started_at = time.perf_counter()
                    result["text"] = rewrite_spoken_navigation_instruction(
                        upper_config,
                        result["transcript"],
                        target=str(request.form.get("target") or "upper_task"),
                    )
                    result["semantic_seconds"] = time.perf_counter() - semantic_started_at
                    result["refined"] = True
                except Exception as refine_exc:
                    # English refinement is an explicit user-selected contract.
                    # Never silently apply the raw Chinese transcript when it
                    # fails, because that can reset and start the robot with the
                    # wrong instruction language.
                    result["refine_error"] = str(refine_exc)
                    result["text"] = ""
                    result["total_seconds"] = time.perf_counter() - request_started_at
                    return jsonify(
                        {
                            "ok": False,
                            **result,
                            "error": f"中文转英文失败，未写入指令：{refine_exc}",
                        }
                    ), 502
            result["total_seconds"] = time.perf_counter() - request_started_at
            return jsonify({"ok": True, **result})
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        except RuntimeError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 503
        except Exception as exc:
            print(f"speech transcription failed: {type(exc).__name__}: {exc}")
            return jsonify({"ok": False, "error": f"语音转写失败：{exc}"}), 500

    def latest_saved_camera_payload():
        for run_dir in list_runs(root):
            metadata_paths = list_frame_metadata(run_dir)
            if not metadata_paths:
                continue
            metadata_path = metadata_paths[-1]
            metadata = sanitize_metadata_for_client(load_metadata(metadata_path))
            age_seconds = max(0.0, datetime.now().timestamp() - metadata_path.stat().st_mtime)
            return {
                "ok": True,
                "available": True,
                "online": age_seconds < 3.0,
                "age_seconds": age_seconds,
                "run_name": run_dir.name,
                "frame_idx": metadata.get("frame_idx"),
                "saved_at": metadata.get("saved_at"),
                "trajectory_points": len((metadata.get("response") or {}).get("trajectory") or []),
                "has_pixel_goal": (metadata.get("response") or {}).get("pixel_goal") is not None,
                "vis_url": url_for("file_view", run_name=run_dir.name, filename=metadata.get("vis_file")),
                "rgb_url": url_for("file_view", run_name=run_dir.name, filename=metadata.get("rgb_file")),
                "depth_url": url_for("file_view", run_name=run_dir.name, filename=metadata.get("depth_file")),
            }
        return {"ok": True, "available": False, "online": False, "age_seconds": None}

    @app.route("/api/camera-monitor/latest")
    def api_camera_monitor_latest():
        return jsonify(latest_saved_camera_payload())

    @app.route("/camera-monitor")
    def camera_monitor():
        return f"""
        <html>
        <head>
          <title>InternNav Camera Monitor</title>
          <style>{render_common_styles()}</style>
        </head>
        <body>
          <main class="cameraMonitorShell">
            <header class="cameraMonitorBar">
              <div><a class="button" href="{url_for('index')}">Back</a> <strong>Latest Camera & Trajectory Monitor</strong></div>
              <div class="cameraMonitorMeta">
                <span id="cameraMonitorState" class="monitorOffline">WAITING</span>
                <span>Run <strong id="cameraMonitorRun">--</strong></span>
                <span>Frame <strong id="cameraMonitorFrame">--</strong></span>
                <span>Trajectory <strong id="cameraMonitorTrajectory">0 points</strong></span>
                <span>Latest <strong id="cameraMonitorAge">--</strong></span>
                <button type="button" class="primary" onclick="setCameraMode('vis')">Trajectory Overlay</button>
                <button type="button" onclick="setCameraMode('rgb')">RGB</button>
                <button type="button" onclick="setCameraMode('depth')">Depth</button>
                <span class="cameraMonitorZoom">
                  <button type="button" title="缩小画面" aria-label="缩小画面" onclick="adjustCameraZoom(-25)">−</button>
                  <input id="cameraMonitorZoom" type="range" min="25" max="300" step="5" value="100" aria-label="画面缩放比例" oninput="setCameraZoom(this.value)">
                  <button type="button" title="放大画面" aria-label="放大画面" onclick="adjustCameraZoom(25)">+</button>
                  <button type="button" title="恢复 100%" onclick="setCameraZoom(100)">1:1</button>
                  <output id="cameraMonitorZoomValue">100%</output>
                </span>
              </div>
            </header>
            <section class="cameraMonitorViewport" id="cameraMonitorViewport">
              <img id="cameraMonitorImage" alt="Latest Go2 camera frame with trajectory overlay">
              <div class="cameraMonitorWaiting">WAITING FOR THE FIRST SAVED CAMERA FRAME</div>
            </section>
          </main>
          <script>
            const monitorImage = document.getElementById('cameraMonitorImage');
            const viewport = document.getElementById('cameraMonitorViewport');
            let cameraMode = 'vis';
            let cameraZoom = 100;
            let latestCameraPayload = null;
            let displayedFrameToken = '';
            function setCameraZoom(value) {{
              cameraZoom = Math.max(25, Math.min(300, Number(value) || 100));
              viewport.style.setProperty('--camera-zoom', String(cameraZoom / 100));
              document.getElementById('cameraMonitorZoom').value = cameraZoom;
              document.getElementById('cameraMonitorZoomValue').value = Math.round(cameraZoom) + '%';
              try {{ localStorage.setItem('internnav-camera-monitor-zoom', String(cameraZoom)); }} catch (_) {{}}
            }}
            function adjustCameraZoom(delta) {{ setCameraZoom(cameraZoom + Number(delta || 0)); }}
            function setCameraMode(mode) {{
              cameraMode = mode;
              displayedFrameToken = '';
              renderLatestCameraFrame();
            }}
            function renderLatestCameraFrame() {{
              if (!latestCameraPayload || !latestCameraPayload.available) return;
              const frameToken = latestCameraPayload.run_name + ':' + latestCameraPayload.frame_idx + ':' + cameraMode;
              if (frameToken === displayedFrameToken) return;
              displayedFrameToken = frameToken;
              const url = latestCameraPayload[cameraMode + '_url'];
              if (url) monitorImage.src = url + '?t=' + Date.now();
            }}
            monitorImage.onload = () => viewport.classList.add('hasFrame');
            viewport.addEventListener('wheel', (event) => {{
              event.preventDefault();
              adjustCameraZoom(event.deltaY < 0 ? 10 : -10);
            }}, {{passive:false}});
            viewport.addEventListener('dblclick', () => setCameraZoom(100));
            async function refreshCameraStatus() {{
              try {{
                const response = await fetch('/api/camera-monitor/latest', {{cache: 'no-store'}});
                const data = await response.json();
                latestCameraPayload = data;
                const state = document.getElementById('cameraMonitorState');
                state.textContent = data.online ? 'LIVE' : (data.available ? 'STALE / LAST FRAME' : 'WAITING');
                state.className = data.online ? 'monitorOnline' : 'monitorOffline';
                document.getElementById('cameraMonitorRun').textContent = data.run_name || '--';
                document.getElementById('cameraMonitorFrame').textContent = data.frame_idx ?? '--';
                document.getElementById('cameraMonitorTrajectory').textContent = Number(data.trajectory_points || 0) + ' points';
                document.getElementById('cameraMonitorAge').textContent = data.age_seconds == null ? '--' : data.age_seconds.toFixed(1) + 's ago';
                renderLatestCameraFrame();
              }} catch (_) {{
                const state = document.getElementById('cameraMonitorState');
                state.textContent = 'VIEWER OFFLINE'; state.className = 'monitorOffline';
              }}
            }}
            try {{ setCameraZoom(localStorage.getItem('internnav-camera-monitor-zoom') || 100); }} catch (_) {{ setCameraZoom(100); }}
            refreshCameraStatus(); setInterval(refreshCameraStatus, 250);
          </script>
        </body>
        </html>
        """

    @app.route("/")
    def index():
        runs = list_runs(root)
        config = load_viewer_config()
        service_status = service_launcher.status()
        latest_run = runs[0] if runs else None
        latest_upper_event = load_latest_upper_agent_event(latest_run) if latest_run else None
        rows = []
        reviewed_runs = [(run, load_run_review(run)) for run in runs]
        pinned_runs = [(run, review) for run, review in reviewed_runs if review["pinned"]]
        normal_runs = [(run, review) for run, review in reviewed_runs if not review["pinned"]]

        def append_run_row(run, review):
            frame_count = len(list_frame_metadata(run))
            latest_badge = "Latest" if runs and run == runs[0] else ""
            instruction = run_instruction_summary(run)
            agent_mode = run_agent_mode(run)
            instruction_text = html.escape(str(instruction.get("text") or ""))
            instruction_source = html.escape(str(instruction.get("source") or ""))
            run_value = html.escape(run.name, quote=True)
            outcome = review["outcome"]
            outcome_label = RUN_OUTCOME_LABELS[outcome]
            outcome_class = {
                "success": "outcomeSuccess",
                "failed": "outcomeFailed",
            }.get(outcome, "outcomeUnset")
            options = "".join(
                f"<option value='{value}'{' selected' if value == outcome else ''}>{label}</option>"
                for value, label in RUN_OUTCOME_LABELS.items()
            )
            pin_label = "Unpin" if review["pinned"] else "Pin"
            row_class = "pinnedRow" if review["pinned"] else ""
            rows.append(
                f"<tr class='{row_class}'><td><input class='runSelect' type='checkbox' value='{run_value}' aria-label='选择实验 {run_value}'></td>"
                f"<td><span class='agentMode {agent_mode['class']}' title='{html.escape(agent_mode['title'], quote=True)}'>{html.escape(agent_mode['label'])}</span></td>"
                f"<td><a class='runLink' href='{url_for('run_view', run_name=run.name, live=1)}'>{html.escape(run.name)}</a></td>"
                f"<td>{frame_count}</td>"
                f"<td>{latest_badge}</td>"
                f"<td class='runInstruction'><details><summary title='点击展开完整指令'>{instruction_text}</summary><div class='runInstructionFull'>{instruction_text}</div></details><span class='runInstructionSource'>{instruction_source}</span></td>"
                f"<td><span class='outcomeBadge {outcome_class}'>{outcome_label}</span><select class='reviewSelect' data-run='{run_value}' onchange='setRunOutcome(this)' aria-label='实验结果'>{options}</select></td>"
                f"<td><a href='{url_for('gif_view', run_name=run.name)}'>GIF</a> · <a href='{url_for('run_view', run_name=run.name, live=0)}#experimentQaPanel'>QA</a><br><button type='button' class='button pinButton' data-run='{run_value}' data-pinned='{str(review['pinned']).lower()}' onclick='toggleRunPin(this)'>{pin_label}</button><button type='button' class='button deleteButton' onclick=\"deleteRun('{run_value}')\">Delete</button></td></tr>"
            )

        if pinned_runs:
            rows.append("<tr class='pinnedDivider'><td colspan='8'>PINNED EXPERIMENTS</td></tr>")
            for run, review in pinned_runs:
                append_run_row(run, review)

        last_date = None
        for run, review in normal_runs:
            date_label = run_date_label(run.name)
            if date_label != last_date:
                rows.append(f"<tr class='dateDivider'><td colspan='8'>{html.escape(date_label)}</td></tr>")
                last_date = date_label
            append_run_row(run, review)
        latest_link = url_for('live_view')
        qa_options = "".join(
            f"<option value='{html.escape(run.name, quote=True)}'>{'★ LATEST · ' if index == 0 else ''}{html.escape(run.name)} · {len(list_frame_metadata(run))} frames</option>"
            for index, run in enumerate(runs)
        )
        qa_model = html.escape(str(get_upper_agent_config(config).get("model") or "qwen3-vl-flash"), quote=True)
        latest_qa_link = (
            f"{url_for('run_view', run_name=latest_run.name, live=0)}#experimentQaPanel"
            if latest_run
            else "#homeExperimentQa"
        )
        return f"""
        <html>
        <head>
          <title>InternNav Experiments</title>
          <style>{render_common_styles()}</style>
        </head>
        <body>
          <div class="homeIntro" id="homeIntro" aria-label="InternNav startup">
            <div class="homeIntroFrame">
              <div class="homeIntroGrid">
                <div>
                  <div class="homeIntroKicker">NAVIGATION CONTROL / INITIALIZING</div>
                  <h1 class="homeIntroTitle">INTERNNAV</h1>
                  <p class="homeIntroText">Real-world navigation experiment console</p>
                  <div class="homeIntroProgress"><i></i></div>
                  <button class="homeIntroSkip" type="button" onclick="dismissHomeIntro()">SKIP INITIALIZATION</button>
                </div>
                <div class="homeIntroVisual" aria-hidden="true">
                  <i class="homeIntroRoute"></i><i class="homeIntroNode a"></i><i class="homeIntroNode b"></i><i class="homeIntroNode c"></i><i class="homeIntroScan"></i>
                </div>
              </div>
            </div>
          </div>
          <div class="runNavigateOverlay" id="runNavigateOverlay" aria-hidden="true"><span>LOADING EXPERIMENT</span></div>
          {render_robot_companion()}
          <div class="shell">
            <div class="hero">
              <div class="heroGrid">
                <div>
                  <h2>InternNav Experiment Viewer</h2>
                  <p>Log dir: <code>{html.escape(str(root))}</code></p>
                  <a class="button" href="{latest_link}">Open Latest Live Run</a>
                  <a class="button primary" href="{url_for('camera_monitor')}">Camera Monitor</a>
                  <a class="button primary" href="{latest_qa_link}">Open Experiment QA</a>
                </div>
                <div class="heroMedia"><img src="{url_for('asset_view', filename=hero_asset)}" alt="InternNav" loading="lazy" decoding="async"></div>
              </div>
            </div>
            <div class="panel experimentQa" id="homeExperimentQa" style="margin-bottom:18px">
              <div class="experimentQaHead">
                <div>
                  <h3 style="margin:0 0 5px">Experiment QA</h3>
                  <div class="status">选择一次实验，进入离线问答、关键帧检索和证据回溯界面。</div>
                </div>
                <span class="experimentQaStatus"><span>{len(runs)} experiments available</span>{f'<span>Latest: {html.escape(latest_run.name)}</span>' if latest_run else ''}</span>
              </div>
              <div class="homeQaSelector">
                <label for="homeQaRun">Experiment</label>
                <select id="homeQaRun" {'disabled' if not runs else ''}>{qa_options or '<option>No experiments available</option>'}</select>
                <button class="primary" type="button" onclick="openSelectedExperimentQa()" {'disabled' if not runs else ''}>Open Analysis</button>
              </div>
              <div class="homeQaQuestionRow">
                <label>Question
                  <textarea id="homeQaQuestion" placeholder="例如：这次巡检过程中看到了灰色沙发吗？在哪些帧？"></textarea>
                </label>
                <label>Analysis model
                  <input id="homeQaModel" list="experimentQaModels" value="{qa_model}" placeholder="qwen3-vl-flash">
                </label>
                <button class="primary" id="homeQaAskButton" type="button" onclick="askExperimentFromHome()" {'disabled' if not runs else ''}>Ask</button>
              </div>
              <datalist id="experimentQaModels">
                <option value="local-qwen3.6-vl"><option value="qwen3-vl-flash"><option value="qwen3-vl-plus"><option value="qwen-vl-max"><option value="qwen-vl-plus">
              </datalist>
              <div class="status" id="homeQaMessage"></div>
              <div class="qaProgress" id="homeQaProgress" hidden>
                <div class="qaProgressTrack"><i class="qaProgressFill" id="homeQaProgressFill"></i></div>
                <span class="qaProgressText" id="homeQaProgressText">0%</span>
              </div>
              <div class="qaAnswer" id="homeQaAnswer"></div>
            </div>
            <div class="panel" style="margin-bottom:18px">
              <h3>Model Service</h3>
              {render_service_launcher_panel(service_status)}
              {render_local_qwen_panel(local_qwen_launcher.status())}
            </div>
            <div class="panel" style="margin-bottom:18px">
              <h3>Runtime Config</h3>
              {render_runtime_config_panel(config)}
            </div>
            <div class="panel" style="margin-bottom:18px">
              <h3>Upper-Level Agent</h3>
              {render_upper_agent_panel(
                  config,
                  run_name=latest_run.name if latest_run else None,
                  latest_event=latest_upper_event,
              )}
              {render_demo_agent_panel(config)}
            </div>
            <div class="tableActions">
              <button type="button" onclick="toggleAllRuns(true)">Select all</button>
              <button type="button" onclick="toggleAllRuns(false)">Clear selection</button>
              <button type="button" class="danger" onclick="deleteSelectedRuns()">Delete selected</button>
            </div>
            <table><tr><th><input class='runSelect' type='checkbox' onclick='toggleAllRuns(this.checked)' aria-label='全选实验'></th><th>Mode</th><th>Run</th><th>Frames</th><th>Status</th><th>Instruction</th><th>Review</th><th>Actions</th></tr>{''.join(rows)}</table>
          </div>
          <script>
            function openSelectedExperimentQa() {{
              const selector = document.getElementById('homeQaRun');
              if (!selector || !selector.value) return;
              window.location.href = '/run/' + encodeURIComponent(selector.value) + '#experimentQaPanel';
            }}
            function homeQaEscape(value) {{
              return String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
            }}
            function renderHomeQaAnswer(record) {{
              const target = document.getElementById('homeQaAnswer');
              const plan = record.evidence_plan || {{}};
              const evidencePlan = Number(plan.frame_count || (record.evidence || []).length || 0);
              const evidence = (record.evidence || []).map((item) => `
                <a class="qaEvidenceItem" href="${{item.rgb_url}}" target="_blank">
                  <img src="${{item.rgb_url}}" alt="Evidence frame ${{homeQaEscape(item.frame_idx)}}" loading="lazy">
                  <span><b>Frame ${{homeQaEscape(item.frame_idx)}}</b><br>${{homeQaEscape(item.reason || '')}}</span>
                </a>`).join('');
              target.innerHTML = `
                <div class="status">ANSWER · ${{homeQaEscape(record.model || '')}} · ${{homeQaEscape(record.search_strategy || '')}}</div>
                <div class="qaAnswerText">${{homeQaEscape(record.answer || '')}}</div>
                <div class="qaMeta"><span>Confidence: ${{(Number(record.confidence || 0) * 100).toFixed(0)}}%</span><span>Call: ${{Number(record.call_time || 0).toFixed(2)}}s</span><span>Evidence plan: ${{evidencePlan}} frames (${{homeQaEscape(plan.source || 'legacy')}})</span></div>
                ${{record.uncertainty ? `<p class="status"><b>Uncertainty:</b> ${{homeQaEscape(record.uncertainty)}}</p>` : ''}}
                <div class="qaEvidenceGrid">${{evidence}}</div>`;
              target.classList.add('isVisible');
            }}
            function setHomeQaProgress(progress, visible=true) {{
              const panel = document.getElementById('homeQaProgress');
              const fill = document.getElementById('homeQaProgressFill');
              const text = document.getElementById('homeQaProgressText');
              if (!panel || !fill || !text) return;
              panel.hidden = !visible;
              const percent = Math.max(0, Math.min(100, Number(progress && progress.percent || 0)));
              fill.style.width = percent + '%';
              text.innerText = `${{percent.toFixed(0)}}% · ${{progress && progress.message || '准备分析'}}`;
            }}
            function pollHomeQaProgress(runName) {{
              let stopped = false;
              const update = async () => {{
                if (stopped) return;
                try {{
                  const response = await fetch('/api/experiment-analysis/' + encodeURIComponent(runName) + '/progress', {{cache:'no-store'}});
                  const data = await response.json();
                  if (data.ok) setHomeQaProgress(data.progress, true);
                }} catch (error) {{}}
              }};
              update();
              const timer = window.setInterval(update, 500);
              return () => {{ stopped = true; window.clearInterval(timer); }};
            }}
            async function askExperimentFromHome() {{
              const runName = document.getElementById('homeQaRun').value;
              const question = document.getElementById('homeQaQuestion').value.trim();
              const model = document.getElementById('homeQaModel').value.trim();
              const button = document.getElementById('homeQaAskButton');
              const message = document.getElementById('homeQaMessage');
              if (!question) {{ message.innerText = '请先输入问题。'; return; }}
              button.disabled = true;
              message.innerText = '正在准备全帧索引并检索证据...';
              setHomeQaProgress({{percent: 1, message: '请求已提交'}}, true);
              const stopProgress = pollHomeQaProgress(runName);
              try {{
                const response = await fetch('/api/experiment-analysis/' + encodeURIComponent(runName) + '/ask', {{
                  method: 'POST', headers: {{'Content-Type': 'application/json'}}, body: JSON.stringify({{question, model}})
                }});
                const data = await response.json().catch(() => ({{}}));
                if (!response.ok || !data.ok) throw new Error(data.error || 'Experiment analysis failed');
                renderHomeQaAnswer(data.record);
                message.innerText = '分析完成；点击证据帧可查看原图。';
                setHomeQaProgress({{percent:100, message:'实验分析完成'}}, true);
              }} catch (error) {{
                message.innerText = error.message;
                setHomeQaProgress({{percent:100, message:'实验分析失败'}} , true);
              }}
              finally {{ stopProgress(); button.disabled = false; }}
            }}
            function dismissHomeIntro() {{
              const intro = document.getElementById('homeIntro');
              if (intro) intro.classList.add('isLeaving');
            }}
            (() => {{
              const key = 'internnav-home-intro-seen';
              try {{
                if (sessionStorage.getItem(key)) {{ dismissHomeIntro(); return; }}
                sessionStorage.setItem(key, '1');
              }} catch (error) {{}}
              window.setTimeout(dismissHomeIntro, 1550);
              document.addEventListener('keydown', (event) => {{ if (event.key === 'Escape') dismissHomeIntro(); }});
            }})();
            document.addEventListener('click', (event) => {{
              const link = event.target.closest('a.runLink');
              if (!link || event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
              link.classList.add('isLaunching');
              const overlay = document.getElementById('runNavigateOverlay');
              if (overlay) overlay.classList.add('isActive');
              // Keep native anchor navigation. Intercepting it with a delayed
              // window.location call could leave the loading overlay stuck when
              // the SSH tunnel or Viewer process reconnects at that moment.
              window.setTimeout(() => {{
                link.classList.remove('isLaunching');
                if (overlay) overlay.classList.remove('isActive');
              }}, 2500);
            }});
            window.addEventListener('pageshow', () => {{
              const overlay = document.getElementById('runNavigateOverlay');
              if (overlay) overlay.classList.remove('isActive');
              document.querySelectorAll('.runLink.isLaunching').forEach((link) => link.classList.remove('isLaunching'));
            }});
          </script>
          <script>{render_config_script()}</script>
          <script>{render_robot_companion_script()}</script>
        </body>
        </html>
        """

    @app.route("/live")
    def live_view():
        runs = list_runs(root)
        if not runs:
            return render_waiting_page("")
        latest = runs[0]
        recent_age = datetime.now().timestamp() - latest_frame_mtime(latest)
        if len(list_frame_metadata(latest)) > 0 and recent_age < 12:
            return redirect(url_for("run_view", run_name=latest.name, live=1))
        return render_waiting_page(latest.name)

    @app.route("/run/<run_name>")
    def run_view(run_name):
        run_dir = safe_run_dir(root, run_name)
        metadata_paths = list_frame_metadata(run_dir)
        if not metadata_paths:
            return f"No waypoint json files found in {html.escape(str(run_dir))}", 404
        live = request.args.get("live", "0") == "1"
        index_arg = int(request.args.get("i", len(metadata_paths) - 1 if live else 0))
        index_arg = max(0, min(index_arg, len(metadata_paths) - 1))
        metadata = sanitize_metadata_for_client(load_metadata(metadata_paths[index_arg]))
        config = load_viewer_config()
        service_status = service_launcher.status()
        latest_upper_event = load_latest_upper_agent_event(run_dir)
        response = metadata.get("response", {})
        vis_file = metadata.get("vis_file")
        rgb_file = metadata.get("rgb_file")
        depth_file = metadata.get("depth_file")
        prev_i = max(0, index_arg - 1)
        next_i = min(len(metadata_paths) - 1, index_arg + 1)
        pretty_json = html.escape(json.dumps(metadata, indent=2, ensure_ascii=False))
        dialogue_text = html.escape(format_agent_dialogue(metadata))
        timing_html = render_timing_panel(metadata)
        config_snapshot_html = render_config_snapshot_panel(metadata)
        analysis_status = experiment_index_status(run_dir)
        analysis_status_json = html.escape(json.dumps(analysis_status, ensure_ascii=False))
        analysis_model = html.escape(str(get_upper_agent_config(config).get("model") or "qwen3-vl-flash"), quote=True)

        return f"""
        <html>
        <head>
          <title>{html.escape(run_name)}</title>
          <style>{render_common_styles()}</style>
        </head>
        <body class="runDetail">
          {render_robot_companion()}
          <div class="topbar">
            <div><a href='{url_for('index')}'>Runs</a> / <b>{html.escape(run_name)}</b> / <a href="{url_for('camera_monitor')}">Camera Monitor</a></div>
            <div class="status" id="liveStatus">{'Live polling enabled' if live else 'Manual browsing'}</div>
          </div>
          <div class="wrap">
            <div class="panel">
              <a class="button" href='{url_for('run_view', run_name=run_name, i=prev_i, live=0)}'>Prev</a>
              <a class="button" href='{url_for('run_view', run_name=run_name, i=next_i, live=0)}'>Next</a>
              <a class="button primary" href='{url_for('run_view', run_name=run_name, live=1)}'>Live Latest</a>
              <a class="button" href='{url_for('gif_view', run_name=run_name, refresh=1)}' target="_blank">Open / Update GIF</a>
              <a class="button primary" href="#experimentQaPanel" onclick="window.setTimeout(() => document.getElementById('experimentQaQuestion').focus(), 350)">Experiment QA</a>
              <span id="frameLabel">Frame {index_arg + 1} / {len(metadata_paths)} | saved frame_idx={metadata.get('frame_idx')}</span>
              <input id="frameSlider" type="range" min="0" max="{len(metadata_paths) - 1}" value="{index_arg}"
                onchange="loadFrame(Number(this.value), false)">
              <div class="playbackControls">
                <button onclick="stepFrame(-1)">Prev Frame</button>
                <button onclick="stepFrame(1)">Next Frame</button>
                <button class="primary" onclick="startPlayback(1)">Play</button>
                <button class="primary" onclick="startPlayback(-1)">Reverse</button>
                <button onclick="pausePlayback()">Pause</button>
                <label>Speed
                  <select id="playbackSpeed">
                    <option value="1200">Slow</option>
                    <option value="650" selected>Normal</option>
                    <option value="250">Fast</option>
                  </select>
                </label>
              </div>
            </div>
            <div class="layout">
              <div class="panel">
                <div class="mediaControls">
                  <button onclick="setImage(currentFrame.vis_url)">Visualization</button>
                  <button onclick="setImage(currentFrame.rgb_url)">RGB</button>
                  <button onclick="setImage(currentFrame.depth_url)">Depth</button>
                  <label class="mediaScale">画面宽度 <input id="mediaWidth" type="range" min="50" max="100" value="100" oninput="setMediaWidth(this.value)"><output id="mediaWidthValue">100%</output></label>
                  <button type="button" onclick="setMediaWidth(100)">Fill</button>
                </div>
                <div class="mediaStage" id="mediaStage"><img id="mainImage" src="{url_for('file_view', run_name=run_name, filename=vis_file)}"></div>
              </div>
              <div class="stack">
                <div class="panel">
                  <h3>Model Service</h3>
                  {render_service_launcher_panel(service_status)}
                  {render_local_qwen_panel(local_qwen_launcher.status())}
                </div>
                <div class="panel">
                  <h3>Runtime Config</h3>
                  {render_runtime_config_panel(config)}
                </div>
                <div class="panel">
                  <h3>Upper-Level Agent</h3>
                  {render_upper_agent_panel(config, run_name=run_name, latest_event=latest_upper_event)}
                  {render_demo_agent_panel(config)}
                </div>
                <div class="panel experimentQa" id="experimentQaPanel">
                  <div class="experimentQaHead">
                    <div>
                      <h3 style="margin:0 0 5px">Experiment QA</h3>
                      <div class="status">离线复盘智能体：检索实验关键帧、动作和日志，不参与机器人控制。</div>
                    </div>
                    <div>
                      <button type="button" onclick="buildExperimentIndex(true)">Build / Rebuild Index</button>
                      <button type="button" onclick="buildObjectInstanceIndex()">Detect Full RGB + Build Instances</button>
                    </div>
                  </div>
                  <div class="experimentQaStatus" id="experimentQaStatus" data-initial="{analysis_status_json}"></div>
                  <label class="status">Analysis model
                    <input id="experimentQaModel" list="runExperimentQaModels" value="{analysis_model}" style="display:block;width:100%;box-sizing:border-box;margin-top:5px;padding:9px;border:1px solid rgba(130,255,205,.3);border-radius:6px;background:rgba(2,10,8,.72);color:#e7fff5">
                  </label>
                  <datalist id="runExperimentQaModels"><option value="local-qwen3.6-vl"><option value="qwen3-vl-flash"><option value="qwen3-vl-plus"><option value="qwen-vl-max"><option value="qwen-vl-plus"></datalist>
                  <textarea id="experimentQaQuestion" placeholder="例如：这次巡检中是否看到灰色沙发？机器人为什么在中途停顿？"></textarea>
                  <div>
                    <button class="primary" type="button" id="experimentQaAskButton" onclick="askExperimentQuestion()">Analyze Experiment</button>
                    <span class="status" id="experimentQaMessage"></span>
                  </div>
                  <div class="qaProgress" id="experimentQaProgress" hidden>
                    <div class="qaProgressTrack"><i class="qaProgressFill" id="experimentQaProgressFill"></i></div>
                    <span class="qaProgressText" id="experimentQaProgressText">0%</span>
                  </div>
                  <div class="qaAnswer" id="experimentQaAnswer"></div>
                  <div>
                    <h4>QA History</h4>
                    <div class="qaHistory" id="experimentQaHistory"><span class="status">Loading history...</span></div>
                  </div>
                </div>
                <div class="panel">
                  <h3>Applied Config Snapshot</h3>
                  <p class="status">Read-only config captured with this frame for experiment records.</p>
                  <div id="configSnapshotPanel">{config_snapshot_html}</div>
                </div>
                <div class="panel">
                  <h3>Timing Breakdown</h3>
                  <div id="timingPanel">{timing_html}</div>
                </div>
                <div class="panel">
                  <h3>Agent Dialogue</h3>
                  <pre id="dialogueText">{dialogue_text}</pre>
                </div>
                <div class="panel">
                  <h3>Waypoint JSON</h3>
                  <p id="outputKeys"><b>Output keys:</b> {html.escape(', '.join(response.keys()))}</p>
                  <pre id="metadataJson">{pretty_json}</pre>
                </div>
              </div>
            </div>
          </div>
          <script>
            let currentFrame = {{
              index: {index_arg},
              count: {len(metadata_paths)},
              vis_url: "{url_for('file_view', run_name=run_name, filename=vis_file)}",
              rgb_url: "{url_for('file_view', run_name=run_name, filename=rgb_file)}",
              depth_url: "{url_for('file_view', run_name=run_name, filename=depth_file)}"
            }};
            let liveMode = {str(live).lower()};
            let playbackTimer = null;
            let playbackDirection = 1;
            const currentRunName = "{html.escape(run_name)}";
            const frameApiBase = "{url_for('api_frame_base', run_name=run_name)}";
            function renderExperimentQaStatus(status, objectInstances=null) {{
              const target = document.getElementById('experimentQaStatus');
              if (!target) return;
              if (!status || !status.exists) {{
                target.innerHTML = '<span>Index not built</span><span>点击 Build 后生成关键帧与事件索引</span>';
                return;
              }}
              const summary = status.summary || {{}};
              const source = status.source || {{}};
              const objectStatus = objectInstances && objectInstances.exists
                ? `${{Number(objectInstances.instance_count || 0)}} tracked instances · ${{objectInstances.capabilities && objectInstances.capabilities.cross_loop_unique_count ? '3D unique counting' : '2D tracks; no cross-loop unique count'}}`
                : 'Full-frame instance index not built';
              target.innerHTML =
                `<span>${{status.stale ? 'Index stale' : 'Index ready'}}</span>` +
                `<span>${{Number(summary.keyframe_count || 0)}} keyframes / ${{Number(source.frame_count || 0)}} frames</span>` +
                `<span>${{escapeHtml(status.backend || 'unknown backend')}}</span>` +
                `<span>${{escapeHtml(objectStatus)}}</span>`;
            }}
            async function refreshExperimentAnalysisStatus() {{
              try {{
                const response = await fetch('/api/experiment-analysis/' + encodeURIComponent(currentRunName) + '/status', {{cache:'no-store'}});
                const data = await response.json();
                if (data.ok) renderExperimentQaStatus(data.status, data.object_instances);
              }} catch (error) {{}}
            }}
            function setExperimentQaProgress(progress, visible=true) {{
              const panel = document.getElementById('experimentQaProgress');
              const fill = document.getElementById('experimentQaProgressFill');
              const text = document.getElementById('experimentQaProgressText');
              if (!panel || !fill || !text) return;
              panel.hidden = !visible;
              const percent = Math.max(0, Math.min(100, Number(progress && progress.percent || 0)));
              fill.style.width = percent + '%';
              text.innerText = `${{percent.toFixed(0)}}% · ${{progress && progress.message || '准备分析'}}`;
            }}
            function pollExperimentQaProgress() {{
              let stopped = false;
              const update = async () => {{
                if (stopped) return;
                try {{
                  const response = await fetch('/api/experiment-analysis/' + encodeURIComponent(currentRunName) + '/progress', {{cache:'no-store'}});
                  const data = await response.json();
                  if (data.ok) setExperimentQaProgress(data.progress, true);
                }} catch (error) {{}}
              }};
              update();
              const timer = window.setInterval(update, 500);
              return () => {{ stopped = true; window.clearInterval(timer); }};
            }}
            async function buildExperimentIndex(force=false) {{
              const message = document.getElementById('experimentQaMessage');
              message.innerText = '正在构建实验索引...';
              try {{
                const response = await fetch('/api/experiment-analysis/' + encodeURIComponent(currentRunName) + '/index', {{
                  method: 'POST', headers: {{'Content-Type': 'application/json'}}, body: JSON.stringify({{force}})
                }});
                const data = await response.json().catch(() => ({{}}));
                if (!response.ok || !data.ok) throw new Error(data.error || 'Index build failed');
                renderExperimentQaStatus(data.status);
                message.innerText = '索引已更新。';
              }} catch (error) {{
                message.innerText = error.message;
                setExperimentQaProgress({{percent:100, message:'实验分析失败'}}, true);
              }}
            }}
            async function buildObjectInstanceIndex() {{
              const message = document.getElementById('experimentQaMessage');
              message.innerText = '正在对全部 RGB 帧检测目标并构建 RGB-D 实例索引，请稍候...';
              setExperimentQaProgress({{percent:1, message:'请求已提交'}}, true);
              const stopProgress = pollExperimentQaProgress();
              try {{
                const response = await fetch('/api/experiment-analysis/' + encodeURIComponent(currentRunName) + '/instances/index', {{
                  method:'POST', headers:{{'Content-Type':'application/json'}},
                  body:JSON.stringify({{run_detector:true, force_detector:true, detector_confidence:0.05, image_size:1280, minimum_score:0.05, iou_threshold:0.20, max_frame_gap:12}})
                }});
                const data = await response.json().catch(() => ({{}}));
                if (!response.ok || !data.ok) throw new Error(data.error || 'Object instance index failed');
                message.innerText = `实例索引完成：${{Number(data.instance_count || 0)}} 条轨迹。`;
                setExperimentQaProgress({{percent:100, message:'全帧目标检测与实例索引已完成'}}, true);
                await refreshExperimentAnalysisStatus();
              }} catch (error) {{ message.innerText = error.message; }}
              finally {{ stopProgress(); }}
            }}
            function renderExperimentQaResult(record) {{
              const target = document.getElementById('experimentQaAnswer');
              const ranges = (record.time_ranges || []).map((item) =>
                `frames ${{escapeHtml(item.start_frame)}}-${{escapeHtml(item.end_frame)}}: ${{escapeHtml(item.reason || '')}}`
              ).join('<br>');
              const sources = (record.data_sources || []).map((item) => `<span class="qaSource">${{escapeHtml(item)}}</span>`).join(' ');
              const evidence = (record.evidence || []).map((item) => `
                <a class="qaEvidenceItem" href="${{item.rgb_url}}" target="_blank">
                  <img src="${{item.rgb_url}}" alt="Evidence frame ${{escapeHtml(item.frame_idx)}}" loading="lazy">
                  <span><b>Frame ${{escapeHtml(item.frame_idx)}}</b><br>${{escapeHtml(item.reason || '')}}</span>
                </a>`).join('');
              target.innerHTML = `
                <div class="status">ANSWER</div>
                <div class="qaAnswerText">${{escapeHtml(record.answer || '')}}</div>
                <div class="qaMeta"><span>Confidence: ${{(Number(record.confidence || 0) * 100).toFixed(0)}}%</span><span>Call: ${{Number(record.call_time || 0).toFixed(2)}}s</span></div>
                ${{record.uncertainty ? `<p class="status"><b>Uncertainty:</b> ${{escapeHtml(record.uncertainty)}}</p>` : ''}}
                ${{ranges ? `<p class="status"><b>Relevant ranges:</b><br>${{ranges}}</p>` : ''}}
                <div>${{sources}}</div><div class="qaEvidenceGrid">${{evidence}}</div>`;
              target.classList.add('isVisible');
            }}
            async function askExperimentQuestion() {{
              const question = document.getElementById('experimentQaQuestion').value.trim();
              const model = document.getElementById('experimentQaModel').value.trim();
              const button = document.getElementById('experimentQaAskButton');
              const message = document.getElementById('experimentQaMessage');
              if (!question) {{ message.innerText = '请先输入问题。'; return; }}
              button.disabled = true; message.innerText = '正在准备全帧索引并检索证据...';
              setExperimentQaProgress({{percent:1, message:'请求已提交'}}, true);
              const stopProgress = pollExperimentQaProgress();
              try {{
                const response = await fetch('/api/experiment-analysis/' + encodeURIComponent(currentRunName) + '/ask', {{
                  method: 'POST', headers: {{'Content-Type': 'application/json'}}, body: JSON.stringify({{question, model}})
                }});
                const data = await response.json().catch(() => ({{}}));
                if (!response.ok || !data.ok) throw new Error(data.error || 'Experiment analysis failed');
                renderExperimentQaResult(data.record);
                message.innerText = '分析完成。';
                setExperimentQaProgress({{percent:100, message:'实验分析完成'}}, true);
                loadExperimentQaHistory();
                renderExperimentQaStatus(data.status);
              }} catch (error) {{ message.innerText = error.message; }}
              finally {{ stopProgress(); button.disabled = false; }}
            }}
            async function loadExperimentQaHistory() {{
              const target = document.getElementById('experimentQaHistory');
              try {{
                const response = await fetch('/api/experiment-analysis/' + encodeURIComponent(currentRunName) + '/history', {{cache: 'no-store'}});
                const data = await response.json();
                if (!data.records || !data.records.length) {{ target.innerHTML = '<span class="status">No Experiment QA questions yet.</span>'; return; }}
                target.innerHTML = data.records.map((record) => `
                  <div class="qaHistoryItem"><b>${{escapeHtml(record.question || '')}}</b><p>${{escapeHtml(record.answer || '')}}</p><small>${{escapeHtml(record.created_at || '')}} · confidence ${{(Number(record.confidence || 0) * 100).toFixed(0)}}%</small></div>`).join('');
              }} catch (error) {{ target.innerHTML = '<span class="status">History load failed.</span>'; }}
            }}
            (() => {{
              const statusNode = document.getElementById('experimentQaStatus');
              try {{ renderExperimentQaStatus(JSON.parse(statusNode.dataset.initial || '{{}}')); }} catch (error) {{ renderExperimentQaStatus(null); }}
              refreshExperimentAnalysisStatus();
              loadExperimentQaHistory();
            }})();
            function setMediaWidth(value) {{
              const width = Math.max(50, Math.min(100, Number(value) || 100));
              document.getElementById('mediaStage').style.setProperty('--media-width', width + '%');
              document.getElementById('mediaWidth').value = width;
              document.getElementById('mediaWidthValue').innerText = width + '%';
              try {{ localStorage.setItem('internnav-media-width', String(width)); }} catch (error) {{}}
            }}
            try {{ setMediaWidth(localStorage.getItem('internnav-media-width') || 100); }} catch (error) {{ setMediaWidth(100); }}
            function escapeHtml(value) {{
              return String(value)
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                .replace(/\"/g, '&quot;');
            }}
            function buildTimingHtml(metadata) {{
              const timing = (metadata && metadata.response && metadata.response._timing) || {{}};
              const rows = [];
              const order = [
                ['transport', 'Transport'],
                ['server_total_time', 'Server total'],
                ['server_core_time', 'Server core'],
                ['inference_time', 'Inference'],
                ['request_read_time', 'HTTP read'],
                ['payload_to_bytes_time', 'Zenoh payload->bytes'],
                ['payload_decode_time', 'Zenoh payload decode'],
                ['image_depth_decode_time', 'Image/depth decode'],
                ['request_payload_bytes', 'Request payload bytes'],
                ['request_image_bytes', 'Request image bytes'],
                ['request_depth_bytes', 'Request depth bytes']
              ];
              order.forEach(([key, label]) => {{
                const value = timing[key];
                if (value === undefined || value === null) return;
                const display = key.endsWith('_time') && typeof value === 'number'
                  ? (value * 1000).toFixed(2) + ' ms'
                  : String(value);
                rows.push(`<tr><td>${{escapeHtml(label)}}</td><td>${{escapeHtml(display)}}</td></tr>`);
              }});
              if (!rows.length) {{
                return "<p class='status'>No timing metrics were captured for this frame.</p>";
              }}
              return `<div class='timingPanel'><table class='timingTable'><thead><tr><th>Stage</th><th>Value</th></tr></thead><tbody>${{rows.join('')}}</tbody></table></div>`;
            }}
            function buildConfigSnapshotHtml(metadata) {{
              const timingConfig = metadata && metadata.response && metadata.response._timing
                ? metadata.response._timing.runtime_config
                : null;
              const config = (metadata && metadata.runtime_config) || timingConfig || {{}};
              const order = [
                ['service_enabled', 'Policy enabled'],
                ['device', 'GPU device'],
                ['runtime_config_path', 'Config file'],
                ['instruction', 'Low-level brain instruction'],
                ['num_history', 'History'],
                ['plan_step_gap', 'Plan gap'],
                ['return_traj_points', 'Trajectory points'],
                ['resize_w', 'Resize W'],
                ['resize_h', 'Resize H'],
                ['save_frame_interval', 'Save every N frames'],
                ['low_level_stop_replan_threshold', 'Consecutive STOP threshold'],
                ['voice_silence_seconds', 'Voice silence seconds'],
                ['voice_command_confidence_threshold', 'Voice command confidence'],
                ['speech_to_text_backend', 'Speech backend'],
                ['speech_to_text_model', 'Speech-to-text model'],
                ['speech_to_text_device', 'Speech device'],
                ['funasr_model_path', 'Fun-ASR-Nano model path'],
                ['sensevoice_model_path', 'SenseVoice model path'],
                ['voice_language_model', 'Voice language model'],
                ['updated_at', 'Config updated at']
              ];
              const rows = [];
              order.forEach(([key, label]) => {{
                if (config[key] === undefined || config[key] === null) return;
                rows.push(`<tr><td>${{escapeHtml(label)}}</td><td>${{escapeHtml(config[key])}}</td></tr>`);
              }});
              const upper = config.upper_agent || {{}};
              if (upper.task_instruction) {{
                rows.push(`<tr><td>Upper Agent task</td><td>${{escapeHtml(upper.task_instruction)}}</td></tr>`);
              }}
              if (!rows.length) return "<p class='status'>No runtime config snapshot was captured for this frame.</p>";
              return `<table class='kvTable'><tbody>${{rows.join('')}}</tbody></table>`;
            }}
            function setImage(src) {{
              if (!src) return;
              document.getElementById('mainImage').src = src + (src.includes('?') ? '&' : '?') + 't=' + Date.now();
            }}
            function renderFrame(payload) {{
              if (!payload) return;
              currentFrame = payload;
              document.getElementById('frameLabel').innerText =
                `Frame ${{payload.index + 1}} / ${{payload.count}} | saved frame_idx=${{payload.metadata.frame_idx}}`;
              const slider = document.getElementById('frameSlider');
              slider.max = Math.max(0, payload.count - 1);
              slider.value = payload.index;
              document.getElementById('metadataJson').innerText = JSON.stringify(payload.metadata, null, 2);
              document.getElementById('dialogueText').innerText = formatDialogue(payload.metadata);
              document.getElementById('timingPanel').innerHTML = buildTimingHtml(payload.metadata);
              document.getElementById('configSnapshotPanel').innerHTML = buildConfigSnapshotHtml(payload.metadata);
              document.getElementById('outputKeys').innerHTML =
                '<b>Output keys:</b> ' + Object.keys(payload.metadata.response || {{}}).join(', ');
              setImage(payload.vis_url);
              document.getElementById('liveStatus').innerText = 'Live updated ' + new Date().toLocaleTimeString();
            }}
            async function loadFrame(index, keepLive, keepPlayback=false) {{
              if (!keepLive) liveMode = false;
              if (!keepPlayback) pausePlayback(false);
              const bounded = Math.max(0, Math.min(index, Math.max(0, currentFrame.count - 1)));
              try {{
                const response = await fetch(frameApiBase + '/' + bounded, {{ cache: 'no-store' }});
                if (response.ok) {{
                  const payload = await response.json();
                  renderFrame(payload);
                  document.getElementById('liveStatus').innerText = 'Frame loaded ' + new Date().toLocaleTimeString();
                }}
              }} catch (err) {{
                document.getElementById('liveStatus').innerText = 'Frame load error';
              }}
            }}
            function stepFrame(delta) {{
              loadFrame(currentFrame.index + delta, false);
            }}
            function pausePlayback(updateStatus=true) {{
              if (playbackTimer) {{
                clearInterval(playbackTimer);
                playbackTimer = null;
              }}
              if (updateStatus) document.getElementById('liveStatus').innerText = 'Playback paused';
            }}
            function startPlayback(direction) {{
              liveMode = false;
              pausePlayback(false);
              playbackDirection = direction >= 0 ? 1 : -1;
              const speed = Number(document.getElementById('playbackSpeed').value || 650);
              document.getElementById('liveStatus').innerText = playbackDirection > 0 ? 'Playing forward' : 'Playing reverse';
              playbackTimer = setInterval(async () => {{
                let nextIndex = currentFrame.index + playbackDirection;
                if (nextIndex >= currentFrame.count) nextIndex = 0;
                if (nextIndex < 0) nextIndex = currentFrame.count - 1;
                await loadFrame(nextIndex, false, true);
              }}, speed);
            }}
            async function pollLatest() {{
              if (!liveMode) return;
              try {{
                const response = await fetch("{url_for('api_latest', run_name=run_name)}", {{ cache: 'no-store' }});
                if (response.ok) {{
                  renderFrame(await response.json());
                  maybeRunUpperAgent();
                }}
              }} catch (err) {{
                document.getElementById('liveStatus').innerText = 'Live polling error';
              }}
            }}
            async function maybeRunUpperAgent() {{
              if (window.upperAgentBusy) return;
              window.upperAgentBusy = true;
              try {{
                const response = await fetch('/api/upper-agent/evaluate/' + encodeURIComponent(currentRunName), {{
                  method: 'POST',
                  headers: {{'Content-Type': 'application/json'}},
                  body: JSON.stringify({{force: false, settle: true}})
                }});
                const data = await response.json().catch(() => ({{}}));
                if (response.ok && data.ok && !data.skipped) {{
                  renderUpperAgentDecision(data.event);
                }}
              }} catch (err) {{
                // Keep live visualization independent from upper-agent API/network failures.
              }} finally {{
                window.upperAgentBusy = false;
              }}
            }}
            if (liveMode) setInterval(pollLatest, 1500);
            function formatDialogue(metadata) {{
              const debug = metadata.agent_debug || {{}};
              const lines = [];
              if (debug.llm_output) {{
                lines.push('LLM output: ' + debug.llm_output);
                lines.push('');
              }}
              lines.push('episode_idx: ' + (debug.episode_idx ?? 'N/A'));
              lines.push('last_s2_idx: ' + (debug.last_s2_idx ?? 'N/A'));
              lines.push('last_instruction: ' + (debug.last_instruction ?? 'N/A'));
              lines.push('num_rgb_history: ' + (debug.num_rgb_history ?? 'N/A'));
              lines.push('num_input_images: ' + (debug.num_input_images ?? 'N/A'));
              if (debug.save_dir) lines.push('agent_save_dir: ' + debug.save_dir);
              lines.push('');
              const history = debug.conversation_history || [];
              if (!history.length) {{
                lines.push('No agent dialogue captured for this frame.');
                lines.push('New frames saved after this update will include agent_debug.conversation_history.');
                return lines.join('\\n');
              }}
              history.forEach((message, idx) => {{
                lines.push('[' + idx + '] ' + (message.role || 'unknown'));
                const content = message.content || [];
                if (Array.isArray(content)) {{
                  content.forEach((item) => {{
                    if (item.type === 'text' && item.text) lines.push(item.text.trim());
                    else if (item.type === 'image') lines.push(item.image || '<image>');
                    else lines.push(JSON.stringify(item));
                  }});
                }} else {{
                  lines.push(String(content));
                }}
                lines.push('');
              }});
              return lines.join('\\n').trim();
            }}
            document.addEventListener('keydown', function(e) {{
              if (e.key === 'ArrowLeft') window.location = '{url_for('run_view', run_name=run_name, i=prev_i, live=0)}';
              if (e.key === 'ArrowRight') window.location = '{url_for('run_view', run_name=run_name, i=next_i, live=0)}';
            }});
            {render_config_script()}
            {render_robot_companion_script()}
          </script>
        </body>
        </html>
        """

    @app.route("/file/<run_name>/<path:filename>")
    def file_view(run_name, filename):
# 提供单帧 RGB/Depth/Vis 文件访问，safe_run_dir 防止路径越界。
        run_dir = safe_run_dir(root, run_name)
        file_path = (run_dir / filename).resolve()
        if run_dir not in file_path.parents and file_path != run_dir:
            abort(404)
        if not file_path.exists():
            abort(404)
        return send_from_directory(run_dir, filename)

    @app.route("/gif/<run_name>")
    def gif_view(run_name):
# 页面点击 GIF 时走这里；如果 run 里有新帧，会自动重新合成 GIF。
        run_dir = safe_run_dir(root, run_name)
        source = request.args.get("source", "vis")
        duration_ms = int(request.args.get("duration_ms", 250))
        output_path = run_dir / f"{source}_feedback.gif"
        pattern = f"frame_*_{source}.jpg" if source in {"rgb", "vis"} else f"frame_*_{source}.png"
        frame_paths = sorted(run_dir.glob(pattern), key=frame_sort_key)
        if request.args.get("refresh", "0") == "1" or gif_is_stale(output_path, frame_paths):
            create_gif_from_run(run_dir, source=source, output_path=output_path, duration_ms=duration_ms)
        return send_file(output_path, mimetype="image/gif")

    @app.route("/api/runs")
    def api_runs():
        return jsonify(
            [
                {
                    "name": run_dir.name,
                    "frames": len(list_frame_metadata(run_dir)),
                    "review": load_run_review(run_dir),
                }
                for run_dir in list_runs(root)
            ]
        )

    @app.route("/api/runs/search")
    def api_search_runs():
        """Read-only run search used by the floating Rover companion."""
        query = str(request.args.get("q") or "").strip().casefold()
        filter_name = str(request.args.get("filter") or "").strip().lower()
        if filter_name not in {"", "latest", "success", "failed"}:
            return jsonify({"ok": False, "error": "Unsupported filter."}), 400

        matches = []
        runs = list_runs(root)
        for run_dir in runs:
            review = load_run_review(run_dir)
            if filter_name == "latest" and run_dir != runs[0]:
                continue
            if filter_name in {"success", "failed"} and review["outcome"] != filter_name:
                continue
            summary = run_instruction_summary(run_dir)
            mode = run_agent_mode(run_dir)
            metadata_paths = list_frame_metadata(run_dir)
            task_instruction = ""
            if metadata_paths:
                metadata = load_metadata(metadata_paths[-1])
                config = metadata_runtime_config(metadata) if isinstance(metadata, dict) else {}
                upper = config.get("upper_agent") if isinstance(config, dict) else {}
                if isinstance(upper, dict):
                    task_instruction = str(upper.get("task_instruction") or "")
            searchable = " ".join([run_dir.name, str(summary.get("text") or ""), task_instruction, mode["label"], review["outcome"]]).casefold()
            if query and query not in searchable:
                continue
            matches.append(
                {
                    "name": run_dir.name,
                    "mode": mode["label"],
                    "outcome": RUN_OUTCOME_LABELS.get(review["outcome"], "未标记"),
                    "instruction": summary.get("text") or "No instruction recorded",
                    "url": url_for("run_view", run_name=run_dir.name, live=1),
                }
            )
            if len(matches) >= 12:
                break
        return jsonify({"ok": True, "results": matches})

    @app.route("/api/run/<run_name>/review", methods=["POST"])
    def api_update_run_review(run_name):
        """Update optional pin/outcome metadata for one existing experiment."""
        payload = request.get_json(force=True, silent=True) or {}
        if not isinstance(payload, dict) or not ({"pinned", "outcome"} & set(payload)):
            return jsonify({"ok": False, "error": "Provide pinned and/or outcome."}), 400
        if "pinned" in payload and not isinstance(payload["pinned"], bool):
            return jsonify({"ok": False, "error": "pinned must be a boolean."}), 400
        if "outcome" in payload and str(payload["outcome"] or "").strip().lower() not in RUN_OUTCOMES:
            return jsonify({"ok": False, "error": "outcome must be success, failed, or empty."}), 400
        try:
            run_dir = safe_run_dir(root, run_name)
            review = load_run_review(run_dir)
            if "pinned" in payload:
                review["pinned"] = payload["pinned"]
            if "outcome" in payload:
                review["outcome"] = str(payload["outcome"] or "").strip().lower()
            return jsonify({"ok": True, "review": save_run_review(run_dir, review)})
        except FileNotFoundError:
            return jsonify({"ok": False, "error": "Experiment does not exist."}), 404
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 500

    @app.route("/api/run/<run_name>/delete", methods=["POST"])
    def api_delete_run(run_name):
        # 删除必须经过前端确认，并且只能删除实验根目录下的 run 包。
        payload = request.get_json(force=True, silent=True) or {}
        if payload.get("confirm") is not True:
            return jsonify({"ok": False, "error": "Deletion was not confirmed."}), 400
        try:
            run_dir = safe_run_dir(root, run_name)
            if run_dir == root:
                return jsonify({"ok": False, "error": "Cannot delete experiment root."}), 400
            shutil.rmtree(run_dir)
            return jsonify({"ok": True, "deleted": run_name})
        except FileNotFoundError:
            return jsonify({"ok": False, "error": "Experiment does not exist."}), 404
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 500

    @app.route("/api/runs/delete", methods=["POST"])
    def api_delete_runs():
        payload = request.get_json(force=True, silent=True) or {}
        names = payload.get("names")
        if payload.get("confirm") is not True or not isinstance(names, list) or not names:
            return jsonify({"ok": False, "error": "Deletion was not confirmed or no runs were selected."}), 400
        if len(names) > 100:
            return jsonify({"ok": False, "error": "Too many runs selected."}), 400

        run_dirs = []
        try:
            for name in names:
                if not isinstance(name, str) or not name.strip():
                    return jsonify({"ok": False, "error": "Invalid run name."}), 400
                candidate = (Path(root).expanduser().resolve() / name).resolve()
                root_path = Path(root).expanduser().resolve()
                if candidate == root_path or root_path not in candidate.parents or not candidate.is_dir():
                    return jsonify({"ok": False, "error": f"Invalid experiment directory: {name}"}), 404
                run_dirs.append((name, candidate))
            for _, run_dir in run_dirs:
                shutil.rmtree(run_dir)
            return jsonify({"ok": True, "deleted": [name for name, _ in run_dirs]})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 500

    @app.route("/api/runtime-config", methods=["GET", "POST"])
    def api_runtime_config():
# Runtime Config 面板提交时走这里。
# 修改写到同一个 runtime_config.json，模型服务下一帧 refresh_runtime_config 后生效。
        if request.method == "POST":
            current = load_runtime_config(config_path)
            current.update(request.get_json(force=True, silent=True) or {})
            current.setdefault("runtime_config_path", str(config_path))
            config = save_runtime_config(config_path, current)
            return jsonify(sanitize_runtime_config(config))
        return jsonify(sanitize_runtime_config(load_viewer_config()))

    @app.route("/api/runtime-config/control", methods=["POST"])
    def api_runtime_config_control():
# Start Policy / Stop E-Stop 只切 service_enabled，不杀模型进程。
# 同时清掉 _upper_agent_pause，避免智能体旧暂停状态影响手动控制。
        current = load_runtime_config(config_path)
        payload = request.get_json(force=True, silent=True) or {}
        raw_enabled = payload.get("service_enabled")
        current["service_enabled"] = raw_enabled in {True, 1, "1", "true", "True", "on", "yes"}
        current.pop("_upper_agent_pause", None)
        current.setdefault("runtime_config_path", str(config_path))
        return jsonify(sanitize_runtime_config(save_runtime_config(config_path, current)))

    @app.route("/api/model-service", methods=["GET", "POST", "DELETE"])
    def api_model_service():
# Start Service / Stop Service 控制真实 HTTP/Zenoh 推理服务进程。
        if request.method == "POST":
            try:
                return jsonify(service_launcher.start(request.get_json(force=True, silent=True) or {}))
            except Exception as exc:
                return jsonify({"error": str(exc), "status": service_launcher.status()}), 400
        if request.method == "DELETE":
            try:
                return jsonify(service_launcher.stop())
            except Exception as exc:
                return jsonify({"error": str(exc), "status": service_launcher.status()}), 400
        return jsonify(service_launcher.status())

    @app.route("/api/local-qwen", methods=["GET", "POST", "DELETE"])
    def api_local_qwen():
        """Lifecycle for the local VLM only; it does not start or stop InternVLA."""
        try:
            if request.method == "POST":
                return jsonify(local_qwen_launcher.start(request.get_json(force=True, silent=True) or {}))
            if request.method == "DELETE":
                return jsonify(local_qwen_launcher.stop())
            return jsonify(local_qwen_launcher.status())
        except Exception as exc:
            return jsonify({"error": str(exc), "status": local_qwen_launcher.status()}), 400

    @app.route("/api/local-qwen/use", methods=["POST"])
    def api_local_qwen_use():
        """Select the local endpoint for all VLM-backed assistant features."""
        current = load_runtime_config(config_path)
        upper = get_upper_agent_config(current)
        upper["model"] = LOCAL_QWEN_MODEL
        current["upper_agent"] = upper
        current["voice_language_model"] = LOCAL_QWEN_MODEL
        current.setdefault("runtime_config_path", str(config_path))
        saved = save_runtime_config(config_path, current)
        return jsonify({"ok": True, "config": sanitize_runtime_config(saved), "local_qwen": local_qwen_launcher.status()})

    @app.route("/api/upper-agent/config", methods=["GET", "POST"])
    def api_upper_agent_config():
# Upper Agent 配置面板提交时走这里。
# api_key 空字符串表示保留旧 key，返回给前端时仍会脱敏。
        if request.method == "POST":
            try:
                upper = set_upper_agent_config(config_path, request.get_json(force=True, silent=True) or {})
                upper = dict(upper)
                upper["credential_configured"] = bool(get_upper_agent_config(load_runtime_config(config_path)).get("api_key"))
                upper.pop("api_key", None)
                return jsonify(upper)
            except Exception as exc:
                return jsonify({"error": str(exc)}), 400
        return jsonify(public_upper_agent_config(load_viewer_config()))

    @app.route("/api/upper-agent/memory/status")
    def api_upper_agent_memory_status():
        return jsonify({"ok": True, **long_term_memory_status()})

    @app.route("/api/demo-agent/libraries", methods=["GET", "POST"])
    def api_demo_agent_libraries():
        try:
            if request.method == "POST":
                library = upsert_demo_library(
                    demo_library_path,
                    request.get_json(force=True, silent=True) or {},
                )
                return jsonify({"ok": True, "library": library})
            store = load_demo_libraries(demo_library_path)
            return jsonify(
                {
                    "ok": True,
                    "libraries": store.get("libraries", []),
                    "updated_at": store.get("updated_at", ""),
                    "active": get_demo_state(load_viewer_config()),
                }
            )
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

    @app.route("/api/demo-agent/libraries/<library_id>", methods=["DELETE"])
    def api_demo_agent_library_delete(library_id):
        try:
            delete_demo_library(demo_library_path, library_id, runtime_config_path=config_path)
            return jsonify({"ok": True})
        except KeyError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 404
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

    @app.route("/api/demo-agent/parse", methods=["POST"])
    def api_demo_agent_parse():
        try:
            payload = request.get_json(force=True, silent=True) or {}
            return jsonify({"ok": True, "commands": parse_navigation_steps(payload.get("commands_text", ""))})
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

    @app.route("/api/demo-agent/activate/<library_id>", methods=["POST"])
    def api_demo_agent_activate(library_id):
        try:
            library = get_demo_library(demo_library_path, library_id)
            updated = activate_demo_library(config_path, library)
            return jsonify({"ok": True, "active": get_demo_state(updated)})
        except KeyError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 404
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

    @app.route("/api/demo-agent/control", methods=["POST"])
    def api_demo_agent_control():
        try:
            payload = request.get_json(force=True, silent=True) or {}
            updated = control_demo_agent(config_path, str(payload.get("action") or ""))
            return jsonify({"ok": True, "active": get_demo_state(updated)})
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

    @app.route("/api/upper-agent/evaluate/<run_name>", methods=["POST"])
    def api_upper_agent_evaluate(run_name):
# “Run Once” 按钮和 live 自动触发都会调用这里。
# 它读取 run 的最新帧，让 Upper Agent 产出下一条低层指令。
        run_dir = safe_run_dir(root, run_name)
        payload = request.get_json(force=True, silent=True) or {}
        try:
            return jsonify(
                evaluate_upper_agent_latest(
                    run_dir,
                    config_path,
                    force=bool(payload.get("force")),
                    settle_for_fresh_frame=bool(payload.get("settle")),
                )
            )
        except Exception as exc:
            print(f"upper-agent evaluation failed for {run_name}: {type(exc).__name__}: {exc}")
            return jsonify({"ok": False, "error": str(exc)}), 400

    def analysis_record_for_client(run_name, record):
        record = dict(record or {})
        evidence = []
        for item in record.get("evidence") or []:
            item = dict(item)
            filename = item.get("rgb_file") or item.get("vis_file")
            item["rgb_url"] = url_for("file_view", run_name=run_name, filename=filename) if filename else ""
            evidence.append(item)
        record["evidence"] = evidence
        return record

    @app.route("/api/experiment-analysis/<run_name>/status")
    def api_experiment_analysis_status(run_name):
        run_dir = safe_run_dir(root, run_name)
        return jsonify(
            {
                "ok": True,
                "status": experiment_index_status(run_dir),
                "object_instances": instance_index_status(run_dir),
                "progress": load_analysis_progress(run_dir),
            }
        )

    @app.route("/api/experiment-analysis/<run_name>/progress")
    def api_experiment_analysis_progress(run_name):
        run_dir = safe_run_dir(root, run_name)
        return jsonify({"ok": True, "progress": load_analysis_progress(run_dir)})

    @app.route("/api/experiment-analysis/<run_name>/instances/index", methods=["POST"])
    def api_experiment_instance_index(run_name):
        run_dir = safe_run_dir(root, run_name)
        payload = request.get_json(force=True, silent=True) or {}
        try:
            write_analysis_progress(run_dir, "starting", 1, "正在准备全帧目标检测")
            prepared = ensure_full_frame_instance_index(
                run_dir,
                force_detector=bool(payload.get("force_detector", False)),
            )
            index = prepared["index"]
            write_analysis_progress(run_dir, "complete", 100, "全帧目标检测与实例索引已完成")
            return jsonify(
                {
                    "ok": True,
                    "detector_ran": prepared["detector_ran"],
                    "instance_count": len(index.get("instances") or []),
                    "status": instance_index_status(run_dir),
                }
            )
        except Exception as exc:
            write_analysis_progress(run_dir, "failed", 100, f"实例索引失败：{exc}")
            return jsonify(
                {"ok": False, "error": str(exc), "status": instance_index_status(run_dir)}
            ), 400

    @app.route("/api/experiment-analysis/<run_name>/index", methods=["POST"])
    def api_experiment_analysis_index(run_name):
        run_dir = safe_run_dir(root, run_name)
        try:
            write_analysis_progress(run_dir, "indexing", 5, "正在构建场景与动作索引")
            index = build_experiment_index(run_dir)
            write_analysis_progress(run_dir, "complete", 100, "场景与动作索引已完成")
            return jsonify({"ok": True, "summary": index.get("summary"), "status": experiment_index_status(run_dir)})
        except Exception as exc:
            write_analysis_progress(run_dir, "failed", 100, f"索引构建失败：{exc}")
            return jsonify({"ok": False, "error": str(exc)}), 400

    @app.route("/api/experiment-analysis/<run_name>/ask", methods=["POST"])
    def api_experiment_analysis_ask(run_name):
        run_dir = safe_run_dir(root, run_name)
        payload = request.get_json(force=True, silent=True) or {}
        question = str(payload.get("question") or "").strip()
        try:
            write_analysis_progress(run_dir, "starting", 1, "正在检查实验索引")
            # Object detection/tracking is the common, cached first stage for
            # every QA request. Once an experiment has been scanned this is a
            # quick cache validation, not another full-frame inference pass.
            write_analysis_progress(run_dir, "indexing", 3, "正在检查全帧目标与对象索引")
            ensure_full_frame_instance_index(run_dir)
            write_analysis_progress(run_dir, "retrieving", 70, "正在检索动作、事件与对象证据")
            upper_config = dict(get_upper_agent_config(load_viewer_config()))
            requested_model = str(payload.get("model") or "").strip()
            if requested_model:
                if not re.fullmatch(r"[A-Za-z0-9._:/-]{1,128}", requested_model):
                    raise ValueError("Analysis model name contains unsupported characters.")
                upper_config["model"] = requested_model
            max_images = max(2, min(12, int(payload.get("max_images", 8))))
            write_analysis_progress(run_dir, "reasoning", 88, "正在调用分析模型核验证据")
            record = answer_experiment_question(run_dir, question, upper_config, max_images=max_images)
            write_analysis_progress(run_dir, "complete", 100, "实验分析完成")
            return jsonify(
                {
                    "ok": True,
                    "record": analysis_record_for_client(run_name, record),
                    "status": experiment_index_status(run_dir),
                }
            )
        except Exception as exc:
            write_analysis_progress(run_dir, "failed", 100, f"实验分析失败：{exc}")
            return jsonify({"ok": False, "error": str(exc)}), 400

    @app.route("/api/experiment-analysis/<run_name>/history")
    def api_experiment_analysis_history(run_name):
        run_dir = safe_run_dir(root, run_name)
        records = [analysis_record_for_client(run_name, item) for item in load_experiment_qa_history(run_dir)]
        return jsonify({"ok": True, "records": records})

    def frame_payload(run_name, frame_index=None):
# 前端播放/倒放/实时刷新统一使用这个 payload：
# 包含 metadata JSON 和三张图的 URL。
        run_dir = safe_run_dir(root, run_name)
        metadata_paths = list_frame_metadata(run_dir)
        if not metadata_paths:
            abort(404)
        if frame_index is None:
            frame_index = len(metadata_paths) - 1
        frame_index = max(0, min(int(frame_index), len(metadata_paths) - 1))
        metadata_path = metadata_paths[frame_index]
        metadata = sanitize_metadata_for_client(load_metadata(metadata_path))
        return {
            "index": frame_index,
            "count": len(metadata_paths),
            "metadata": metadata,
            "vis_url": url_for("file_view", run_name=run_name, filename=metadata.get("vis_file")),
            "rgb_url": url_for("file_view", run_name=run_name, filename=metadata.get("rgb_file")),
            "depth_url": url_for("file_view", run_name=run_name, filename=metadata.get("depth_file")),
        }

    @app.route("/api/run/<run_name>/frame")
    def api_frame_base(run_name):
        return jsonify(frame_payload(run_name, 0))

    @app.route("/api/run/<run_name>/frame/<int:frame_index>")
    def api_frame(run_name, frame_index):
        return jsonify(frame_payload(run_name, frame_index))

    @app.route("/api/run/<run_name>/latest")
    def api_latest(run_name):
        return jsonify(frame_payload(run_name))

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", default="output/realworld_experiments")
    parser.add_argument("--refresh_vis", default=None, help="Run name to redraw visualizations, or 'all'.")
    parser.add_argument("--make_gif", default=None, help="Run name to export as GIF.")
    parser.add_argument("--source", default="vis", choices=["vis", "rgb", "depth"])
    parser.add_argument("--duration_ms", type=int, default=250)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8899)
    parser.add_argument("--runtime_config_path", default=None)
    parser.add_argument("--tunnel_user", default=os.environ.get("USER", "chris"))
    parser.add_argument("--tunnel_host", default="")
    args = parser.parse_args()

    log_dir = Path(args.log_dir).expanduser().resolve()

    if args.refresh_vis:
        targets = list_runs(log_dir) if args.refresh_vis == "all" else [safe_path(log_dir, args.refresh_vis)]
        for run_dir in targets:
            updated = refresh_run_visualizations(run_dir)
            print(f"refreshed {updated} frames in {run_dir}")

    if args.make_gif:
        run_dir = safe_path(log_dir, args.make_gif)
        output_path = create_gif_from_run(run_dir, source=args.source, duration_ms=args.duration_ms)
        print(output_path)

    if args.serve:
        runtime_config_path = args.runtime_config_path or default_runtime_config_path(log_dir)
        app = create_viewer_app(log_dir, runtime_config_path=runtime_config_path)
        tunnel_host = args.tunnel_host or ("<server-ip>" if args.host in {"0.0.0.0", "127.0.0.1", "localhost"} else args.host)
        print("\nInternNav Experiment Viewer")
        print(f"  Server URL: http://{args.host}:{args.port}/")
        print(f"  Local browser URL after SSH tunnel: http://127.0.0.1:{args.port}/")
        print("  Run this on your local computer if direct access does not work:")
        print(f"  ssh -L {args.port}:127.0.0.1:{args.port} {args.tunnel_user}@{tunnel_host}\n")
        app.run(host=args.host, port=args.port)


def safe_path(log_dir, run_name):
    # CLI 版本的 run 目录校验。
    root = Path(log_dir).expanduser().resolve()
    run_dir = (root / run_name).resolve()
    if root not in run_dir.parents:
        raise ValueError(f"Invalid run name: {run_name}")
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    return run_dir


if __name__ == "__main__":
    main()
