import base64
import json
import math
import os
import re
import time
from datetime import datetime
from pathlib import Path

import requests
from PIL import Image

from agent_long_term_memory import (
    build_memory_candidate,
    build_retrieval_query,
    get_long_term_memory_manager,
    long_term_memory_status,
)
from local_qwen_service import is_local_qwen_model, resolve_api_key, resolve_api_url, resolve_model_name

from demo_agent import (
    constrain_demo_agent_output,
    demo_prompt_context,
    ensure_demo_step_reference,
)
from runtime_config import load_runtime_config, sanitize_runtime_config, save_runtime_config


# Upper Agent 是“上层智能体”的独立逻辑文件：
# 1. 从实验目录读取最新帧和低层大脑输出。
# 2. 结合总任务 task_instruction、路线记忆 route_memory、历史决策 recent_events。
# 3. 调用 Qwen-VL，让它产出短时 current_subgoal。
# 4. 可选地把 current_subgoal 写回 runtime_config["instruction"]，供 InternVLA-N1 低层大脑执行。
# 注意：这里的 task_instruction 是用户给上层智能体的总任务；
# runtime_config["instruction"] 是上层智能体拆出来、真正发给低层大脑的短指令。
DEFAULT_UPPER_AGENT_CONFIG = {
    "enabled": False,
# 是否启用上层智能体自动观察最新帧。
    "auto_apply_instruction": False,
# 是否把智能体输出的 current_subgoal 自动写回低层大脑 instruction。
    "pause_policy_while_thinking": True,
# 智能体调用 Qwen 思考期间，是否临时暂停低层策略，避免旧指令继续驱动机器人。
    "enable_route_memory": True,
# 是否为每个实验 run 保存路线记忆 upper_agent_memory.json。
    "enable_long_term_memory": True,
# 是否启用跨实验 Mem0 长期记忆；检索超时或依赖异常时自动跳过，不影响导航。
    "enable_graph_memory_capture": True,
# 是否记录场景—地点—地标—动作—结果图事件，为后续图数据库阶段积累数据。
    "long_term_memory_top_k": 3,
# 每次只取最相关的少量长期记忆，避免旧经验淹没当前视觉证据。
    "long_term_memory_char_budget": 600,
# 送入上层模型的长期记忆总字符预算。
    "long_term_memory_timeout_ms": 180,
# 本地检索时限；超过后本轮直接不用长期记忆，保证实时控制速度。
    "safety_mode": True,
# 是否启用转弯、盲区、墙边等场景的安全提示。
    "guidance_style": "directional",
# 上层智能体输出风格：directional 更偏短时方向，subgoal 更偏阶段目标，cautious 更保守。
    "task_instruction": "",
# 用户给上层智能体的完整任务，例如“跟随穿黑裤子的人并在沙发旁停止”。
    "api_key": "",
    "api_key_env": "QWEN_API_KEY",
    "api_url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    "model": "qwen-vl-plus",
    "read_every_n_frames": 5,
# 每隔多少个已保存帧才触发一次上层智能体。
    "min_seconds_between_calls": 2.0,
# 两次 Qwen 调用之间的最小时间间隔，防止“走两步就停下来想一次”。
    "replan_settle_seconds": 1.0,
# 低层触发重规划后，保持停止并等待最新相机帧写入的时间。
    "history_events": 6,
# prompt 中带入最近多少条上层智能体决策记录。
    "max_memory_items": 12,
# 路线记忆中每类列表最多保留多少条。
    "max_subgoal_age_frames": 120,
# 同一子目标最多允许覆盖多少个保存帧；超时后强制进入下一阶段，避免卡在同一地标。
    # Local control decisions use a compact JSON schema. Keep enough headroom
    # for valid JSON while avoiding long natural-language reports every frame.
    "max_tokens": 256,
    "temperature": 0.2,
    # Preserve the complete camera view, but reduce each of the four motion
    # frames before sending them to the local VLM.
    "max_image_width": 640,
    "motion_context_frames": 8,
# Demo Agent 判断动作进度时发送的时序帧数：步骤起点、间隔帧和最新帧。
    "demo_completion_confidence_threshold": 0.75,
# Demo 子任务完成置信度门槛；不足时保持当前指令并重新执行。
    "demo_transition_mode": "balanced",
# Demo 步骤切换策略：strict 强视觉确认，balanced 融合时序画面与低层转向证据。
    "feedback_language": "zh",
# 任务反馈语言：zh 中文、en 英文、auto 跟随任务语言。
    "auto_speak_task_feedback": True,
# Viewer 打开时，任务完成或失败后自动语音播报最终反馈。
    "system_prompt": "",
}

INT_LIMITS = {
    "read_every_n_frames": (1, 1000),
    "history_events": (0, 30),
    "max_tokens": (64, 4096),
    "max_image_width": (224, 1600),
    "max_memory_items": (1, 50),
    "max_subgoal_age_frames": (10, 10000),
    "motion_context_frames": (4, 12),
    "long_term_memory_top_k": (1, 5),
    "long_term_memory_char_budget": (160, 1200),
    "long_term_memory_timeout_ms": (20, 1000),
}

FLOAT_LIMITS = {
    "min_seconds_between_calls": (0.0, 3600.0),
    "replan_settle_seconds": (0.0, 10.0),
    "temperature": (0.0, 2.0),
    "demo_completion_confidence_threshold": (0.5, 1.0),
}

UPPER_AGENT_SYSTEM_PROMPT = """
You are a fast upper-level navigation controller for a quadruped robot.
Inspect the ordered camera frames and compact state, then return one JSON object.

Required JSON schema:
{
  "task_status": "running|completed|failed",
  "navigation_phase": "observe|advance|turn|scan|approach|stop",
  "current_subgoal": "short executable command, empty when completed/failed",
  "demo_step_index": 0,
  "demo_step_decision": "hold|advance|complete|failed",
  "execution_assessment": {
    "subgoal_completed": false,
    "forward_completed": false,
    "turn_completed": false,
    "observed_turn_direction": "none|left|right|around|uncertain",
    "completion_confidence": 0.0
  },
  "visual_evidence": "at most 20 words"
}

Optional only when there is a new landmark, step transition, finding, completion,
or failure:
{"memory":{"current_place":"short","landmarks_seen":["short"],"completed_subgoal":"short","next_direction_hint":"short","failure_reason":"short"},"task_feedback":{"summary":"short","count":null,"count_label":"","findings":[],"failure_reason":"","recommendation":""}}

Rules:
- When demo_agent is present in the context, act only as its visual step
  supervisor. Do not invent or rewrite navigation commands. Set demo_step_index
  to the current step or the immediately following step, and set
  demo_step_decision to hold, advance, complete, or failed. Advance only after
  the step-start reference image, current image, and low-level response support
  that the current command has been executed. A materially changed heading or
  route after a turn is completion evidence even though the turn itself is not
  visible in the current static frame. STOP/replan_required only marks the end
  of the current review clip. It is never evidence that the command completed.
  Complete only after the final command is visibly complete.
  The server will copy the exact selected library command into current_subgoal.
- Demo motion images are ordered from the step start to the current observation.
  Judge translation and turning by comparing the whole sequence. Never claim
  that the robot has not moved based on the current image alone. If the first
  and last views show a changed position or heading consistent with the command,
  advance even when no single frame depicts the turn in progress.
- Fill execution_assessment from the whole ordered clip. For a command containing
  a turn, combine the ordered images with demo_execution_evidence. In balanced
  mode, sustained matching turn outputs plus a material start-to-end scene change
  and a plausible new route or area are sufficient evidence of completion; do
  not demand that the final static frame display an explicit corridor axis. When
  matching turn outputs occur repeatedly in multiple segments and the endpoint
  scene has materially changed, advance unless the images clearly prove the
  opposite turn direction. Hold when the requested turn never appears, the
  direction conflicts, or the start and end views remain materially unchanged.
- Treat task_instruction as the authoritative full user task.
  current_low_level_instruction is only the previously issued command, not the
  full task and not a command that must be repeated.
- Base every decision primarily on the current image and current low-level
  response. Route memory is historical context, not proof that an old landmark
  is still visible or still ahead.
- When the requested target is clearly visible and reachable, prioritize making
  progress toward that target instead of navigating toward an unrelated route
  event. The requested final target is a destination, not a turn trigger.
- Keep current_subgoal short, directly executable, and limited to the immediate
  navigation stage. Keep later route stages in memory rather than placing the
  whole route in current_subgoal.
- Do not include a stop action in an intermediate current_subgoal. If uncertain,
  blocked, or near a blind corner, choose an appropriate observation, recovery,
  or safe navigation action without falsely declaring task completion.
- If the current image clearly confirms a dead end or no usable forward passage,
  choose an appropriate recovery action. Do not infer a dead end from a temporary
  pedestrian, partial occlusion, or route memory alone.
- Re-evaluate the previous subgoal against every new frame. Reuse it only while
  its visual evidence remains valid and it has not been completed, passed, or
  invalidated.
- If subgoal_watchdog.must_transition is true, or the low-level response reports
  STOP/replan_required while the task is still running, review the execution clip.
  Keep the same current_subgoal unless execution_assessment confirms completion;
  timeout or STOP alone must never advance the Demo instruction index.
- Use compact memory to avoid loops. Only return a memory patch when it changed.
- Set task_status to completed only when the current image confirms that the
  task's final stopping condition has been reached. When completed, return an
  empty current_subgoal. Otherwise keep task_status running and provide the next
  executable subgoal.
- Keep navigation and reporting separate. current_subgoal contains only robot
  actions. Put counts, inspection results, anomalies, damage, obstacles, task
  conclusions, and failure explanations only in task_feedback.
- For counting tasks, count only clearly visible task-relevant objects. Use
  route memory and recent findings to avoid double-counting the same object
  across adjacent frames. Keep count null while uncertain. On completion,
  return the best grounded integer count and name it in count_label.
- For inspection tasks, add one finding per visible issue. Never invent damage,
  blockage, or anomalies not supported by the current image. Include a useful
  location and evidence; use warning or critical only when justified.
- While running, omit task_feedback unless there is a newly observed task finding.
  When completed, summary must state the result clearly. When failed,
  current_subgoal must be empty, summary must explain the outcome, and
  failure_reason must state the most likely grounded cause. Do not mark failed
  for one uncertain frame or a temporary occlusion; first attempt reasonable
  observation or recovery.
- Write task_feedback in Chinese when feedback_language is zh, in English when
  it is en, and follow the user's task language when it is auto.
- Keep every free-text value short. Do not include markdown or text outside JSON.
""".strip()


def normalize_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def clamp_int(value, low, high, default):
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = default
    return max(low, min(high, value))


def clamp_float(value, low, high, default):
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = default
    return max(low, min(high, value))


def normalize_upper_agent_config(config):
# 统一清洗网页/API 写入的配置，避免空字符串、非法数字、checkbox 字符串导致运行异常。
    normalized = dict(DEFAULT_UPPER_AGENT_CONFIG)
    if isinstance(config, dict):
        normalized.update(config)

    normalized["enabled"] = normalize_bool(normalized.get("enabled"))
    normalized["auto_apply_instruction"] = normalize_bool(normalized.get("auto_apply_instruction"))
    normalized["pause_policy_while_thinking"] = normalize_bool(normalized.get("pause_policy_while_thinking"))
    normalized["enable_route_memory"] = normalize_bool(normalized.get("enable_route_memory"))
    normalized["enable_long_term_memory"] = normalize_bool(normalized.get("enable_long_term_memory"))
    normalized["enable_graph_memory_capture"] = normalize_bool(normalized.get("enable_graph_memory_capture"))
    normalized["safety_mode"] = normalize_bool(normalized.get("safety_mode"))
    normalized["auto_speak_task_feedback"] = normalize_bool(normalized.get("auto_speak_task_feedback"))
    normalized["guidance_style"] = str(normalized.get("guidance_style") or "directional").strip() or "directional"
    transition_mode = str(normalized.get("demo_transition_mode") or "balanced").strip().lower()
    normalized["demo_transition_mode"] = transition_mode if transition_mode in {"strict", "balanced"} else "balanced"
    feedback_language = str(normalized.get("feedback_language") or "zh").strip().lower()
    normalized["feedback_language"] = feedback_language if feedback_language in {"zh", "en", "auto"} else "zh"

    for key in ["api_key", "api_key_env", "api_url", "model", "system_prompt", "task_instruction"]:
        normalized[key] = str(normalized.get(key) or "").strip()

    for key, (low, high) in INT_LIMITS.items():
        normalized[key] = clamp_int(normalized.get(key), low, high, DEFAULT_UPPER_AGENT_CONFIG[key])

    for key, (low, high) in FLOAT_LIMITS.items():
        normalized[key] = clamp_float(normalized.get(key), low, high, DEFAULT_UPPER_AGENT_CONFIG[key])

    normalized["updated_at"] = str(normalized.get("updated_at") or "")
    return normalized


def get_upper_agent_config(runtime_config):
    return normalize_upper_agent_config((runtime_config or {}).get("upper_agent") or {})


def set_upper_agent_config(runtime_config_path, updates):
# Web 可视化页面提交上层智能体配置时走这里。
# 空 api_key 表示“沿用旧 key”，这样页面刷新不会把已有 key 清空。
    current = load_runtime_config(runtime_config_path)
    existing = get_upper_agent_config(current)
    updates = dict(updates or {})

    # Empty password field means "keep the old key".
    if "api_key" in updates and not str(updates.get("api_key") or "").strip():
        updates.pop("api_key")

    new_task = str(updates.get("task_instruction") or "").strip() if "task_instruction" in updates else None
    task_changed = new_task is not None and new_task != str(existing.get("task_instruction") or "").strip()
    existing.update(updates)
    if task_changed:
        # A new high-level task must never start with the previous task's
        # low-level command or completion state.
        current["instruction"] = ""
        existing["last_task_status"] = "running"
        existing["last_subgoal"] = ""
        existing["last_decision_at"] = ""
        existing["replan_requested"] = True
        existing["hard_reset_requested"] = False
        existing.pop("last_task_feedback", None)
        existing.pop("last_task_report_at", None)
        for key in ("hard_reset_reason", "hard_reset_requested_at", "hard_reset_subgoal"):
            existing.pop(key, None)
        demo_state = existing.get("demo_agent") if isinstance(existing.get("demo_agent"), dict) else None
        if demo_state and demo_state.get("enabled"):
            demo_state = dict(demo_state)
            demo_state["enabled"] = False
            demo_state["status"] = "stopped"
            demo_state["last_transition_reason"] = "replaced_by_regular_upper_agent_task"
            demo_state["updated_at"] = datetime.now().isoformat(timespec="seconds")
            existing["demo_agent"] = demo_state
        current.pop("_upper_agent_pause", None)
    current["upper_agent"] = normalize_upper_agent_config(existing)
    current["upper_agent"]["updated_at"] = datetime.now().isoformat(timespec="seconds")
    return save_runtime_config(runtime_config_path, current)["upper_agent"]


def acquire_policy_pause(runtime_config_path, config):
# 上层智能体思考前的“软暂停”：
# 只在 auto_apply_instruction 打开时启用，因为只有这种情况下智能体会真正改低层指令。
# 使用 token 记录暂停归属，避免多个异步请求互相覆盖。
    if not config.get("pause_policy_while_thinking") or not config.get("auto_apply_instruction"):
        return None
    token = f"upper-agent-{time.time_ns()}"
    current = load_runtime_config(runtime_config_path)
    previous_service_enabled = bool(current.get("service_enabled", True))
    if not previous_service_enabled:
        return None
    current["service_enabled"] = False
    current["_upper_agent_pause"] = {
        "active": True,
        "token": token,
        "previous_service_enabled": previous_service_enabled,
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    save_runtime_config(runtime_config_path, current)
    return {"token": token, "previous_service_enabled": previous_service_enabled}


def release_policy_pause(
    runtime_config_path,
    pause_state,
    low_level_instruction=None,
    auto_apply_instruction=False,
    task_completed=False,
):
# 上层智能体思考结束后的恢复：
# 只有 token 对得上，才允许恢复 service_enabled 并写入 current_subgoal。
# 这样如果用户手动 E-Stop，不会被一个迟到的智能体结果又自动启动。
    current = load_runtime_config(runtime_config_path)
    pause = current.get("_upper_agent_pause") or {}
    if pause_state and pause.get("token") == pause_state.get("token"):
        if task_completed:
            current["instruction"] = ""
        elif auto_apply_instruction and low_level_instruction:
            current["instruction"] = str(low_level_instruction).strip()
        current["service_enabled"] = bool(pause_state.get("previous_service_enabled", True))
        current.pop("_upper_agent_pause", None)
    save_runtime_config(runtime_config_path, current)
    return current


def persist_upper_agent_decision(
    runtime_config_path,
    parsed,
    auto_apply_instruction=False,
    request_low_level_hard_reset=False,
    hard_reset_reason="",
):
    """Publish the latest high-level decision for the model-service protocol."""
    current = load_runtime_config(runtime_config_path)
    upper = current.get("upper_agent") if isinstance(current.get("upper_agent"), dict) else {}
    upper = dict(upper)
    previous_subgoal = str(upper.get("last_subgoal") or "").strip()
    upper["last_task_status"] = str(parsed.get("task_status") or "running").strip().lower()
    upper["last_subgoal"] = str(parsed.get("current_subgoal") or "").strip()
    upper["last_task_feedback"] = make_json_safe(parsed.get("task_feedback") or {})
    upper["last_task_report_at"] = datetime.now().isoformat(timespec="seconds")
    task_terminal = upper["last_task_status"] in {"completed", "failed"}
    upper["last_decision_at"] = datetime.now().isoformat(timespec="seconds")
    upper["replan_requested"] = False
    upper["last_error"] = ""
    # 这个令牌只用于“低层已经卡在 STOP，但上层仍在 running”的恢复。
    # 它不是新 episode：server 消费后仅 reset InternVLA 内部缓存，实验 run 保持不变。
    subgoal_changed = bool(
        previous_subgoal
        and upper["last_subgoal"]
        and normalize_subgoal_text(previous_subgoal) != normalize_subgoal_text(upper["last_subgoal"])
    )
    should_hard_reset = not task_terminal and bool(request_low_level_hard_reset or subgoal_changed)
    if should_hard_reset and auto_apply_instruction and upper["last_subgoal"]:
        upper["hard_reset_requested"] = True
        upper["hard_reset_reason"] = str(
            hard_reset_reason if request_low_level_hard_reset else "upper_agent_subgoal_changed"
        )
        upper["hard_reset_requested_at"] = datetime.now().isoformat(timespec="seconds")
        upper["hard_reset_subgoal"] = upper["last_subgoal"]
    if task_terminal:
        for key in (
            "hard_reset_requested",
            "hard_reset_reason",
            "hard_reset_requested_at",
            "hard_reset_subgoal",
        ):
            upper.pop(key, None)
        current["instruction"] = ""
    elif auto_apply_instruction and upper["last_subgoal"]:
        current["instruction"] = upper["last_subgoal"]
    current["upper_agent"] = upper
    return save_runtime_config(runtime_config_path, current)


def persist_upper_agent_error(runtime_config_path, error):
    """Persist a concise, non-secret failure reason for the web viewer."""
    current = load_runtime_config(runtime_config_path)
    upper = current.get("upper_agent") if isinstance(current.get("upper_agent"), dict) else {}
    upper = dict(upper)
    upper["last_error"] = str(error).strip()[:1200]
    upper["last_error_at"] = datetime.now().isoformat(timespec="seconds")
    current["upper_agent"] = upper
    return save_runtime_config(runtime_config_path, current)


def latest_frame_metadata(run_dir):
    paths = sorted(Path(run_dir).glob("frame_*_waypoint.json"), key=frame_sort_key)
    if not paths:
        return None, None
    with open(paths[-1]) as f:
        return paths[-1], json.load(f)


def frame_sort_key(path):
    match = re.search(r"frame_(\d+)", Path(path).name)
    return int(match.group(1)) if match else -1


def event_log_path(run_dir):
# 每个实验 run 都会有一份上层智能体事件流，便于复盘智能体每次“想了什么”。
    return Path(run_dir) / "upper_agent_events.jsonl"


def latest_event_path(run_dir):
    return Path(run_dir) / "upper_agent_latest.json"


def memory_path(run_dir):
# 路线记忆和实验 run 绑定，不同实验之间不共享，避免旧任务污染新任务。
    return Path(run_dir) / "upper_agent_memory.json"


def task_report_path(run_dir):
    return Path(run_dir) / "upper_agent_report.json"


def default_route_memory():
# route_memory 是给上层智能体的轻量“任务笔记本”：
# 记录当前位置、已见 landmark、完成过的子目标、危险点、最近观察。
    return {
        "current_place": "",
        "active_subgoal": "",
        "next_direction_hint": "",
        "visited_landmarks": [],
        "completed_subgoals": [],
        "hazards": [],
        "observations": [],
        "failure_reasons": [],
        "task_findings": [],
        "latest_task_feedback": {},
        "last_frame_idx": None,
        "active_subgoal_started_frame": None,
        "same_subgoal_decisions": 0,
        "subgoal_watchdog_forced_count": 0,
        "updated_at": "",
    }


def load_route_memory(run_dir):
    path = memory_path(run_dir)
    if not path.exists():
        return default_route_memory()
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return default_route_memory()
    memory = default_route_memory()
    if isinstance(data, dict):
        memory.update(data)
    return memory


def append_unique(items, value, limit):
    value = str(value or "").strip()
    if not value:
        return items[-limit:]
    items = [item for item in items if item != value]
    items.append(value)
    return items[-limit:]


def merge_task_findings(existing, new_findings, limit):
    """Merge grounded findings while avoiding adjacent-frame duplicates."""
    merged = [item for item in (existing or []) if isinstance(item, dict)]
    for finding in new_findings or []:
        if not isinstance(finding, dict) or not finding.get("description"):
            continue
        signature = (
            str(finding.get("type") or "").strip().casefold(),
            str(finding.get("description") or "").strip().casefold(),
            str(finding.get("location") or "").strip().casefold(),
        )
        merged = [
            item
            for item in merged
            if (
                str(item.get("type") or "").strip().casefold(),
                str(item.get("description") or "").strip().casefold(),
                str(item.get("location") or "").strip().casefold(),
            )
            != signature
        ]
        merged.append(make_json_safe(finding))
    return merged[-limit:]


def normalize_subgoal_text(value):
    """Compare subgoals without casing, whitespace, or trailing punctuation noise."""
    return re.sub(r"\s+", " ", str(value or "").strip().rstrip(".!?").lower())


def subgoal_watchdog_status(memory, frame_idx, config):
    """Expose deterministic progress state to the VLM and the fallback policy."""
    memory = memory or {}
    active_subgoal = str(memory.get("active_subgoal") or "").strip()
    try:
        started_frame = int(memory.get("active_subgoal_started_frame"))
    except (TypeError, ValueError):
        started_frame = int(frame_idx or 0)
    current_frame = int(frame_idx or 0)
    age_frames = max(0, current_frame - started_frame)
    max_age_frames = int(config.get("max_subgoal_age_frames", 120))
    return {
        "active_subgoal": active_subgoal,
        "started_frame": started_frame,
        "age_frames": age_frames,
        "max_age_frames": max_age_frames,
        "must_transition": bool(active_subgoal and age_frames >= max_age_frames),
    }


def update_route_memory(memory, output, frame_idx, config):
# 将 Qwen 输出的 memory 合并到本地记忆。
# 这里不是做复杂地图，而是保留足够少、足够有用的语义线索，帮助智能体避免重复绕圈。
    memory = dict(memory or default_route_memory())
    limit = int(config.get("max_memory_items", 12))
    update = output.get("memory") if isinstance(output, dict) else {}
    if not isinstance(update, dict):
        update = output.get("route_memory_update") if isinstance(output, dict) else {}
    if not isinstance(update, dict):
        update = {}

    current_place = str(update.get("current_place") or "").strip()
    if current_place:
        memory["current_place"] = current_place

    previous_subgoal = str(memory.get("active_subgoal") or "").strip()
    current_subgoal = str(output.get("current_subgoal") or "").strip() if isinstance(output, dict) else ""
    if current_subgoal:
        memory["active_subgoal"] = current_subgoal
        if normalize_subgoal_text(previous_subgoal) == normalize_subgoal_text(current_subgoal):
            memory["same_subgoal_decisions"] = int(memory.get("same_subgoal_decisions") or 0) + 1
        else:
            memory["active_subgoal_started_frame"] = int(frame_idx or 0)
            memory["same_subgoal_decisions"] = 1
    if output.get("subgoal_watchdog_forced"):
        memory["subgoal_watchdog_forced_count"] = int(memory.get("subgoal_watchdog_forced_count") or 0) + 1

    next_direction_hint = str(update.get("next_direction_hint") or output.get("direction_guidance") or "").strip()
    if next_direction_hint:
        memory["next_direction_hint"] = next_direction_hint

    memory["visited_landmarks"] = append_unique(
        list(memory.get("visited_landmarks") or []),
        update.get("visited_landmark"),
        limit,
    )
    for landmark in (update.get("landmarks_seen") or output.get("landmarks_seen") or []):
        memory["visited_landmarks"] = append_unique(memory["visited_landmarks"], landmark, limit)

    memory["completed_subgoals"] = append_unique(
        list(memory.get("completed_subgoals") or []),
        update.get("completed_subgoal"),
        limit,
    )
    memory["failure_reasons"] = append_unique(
        list(memory.get("failure_reasons") or []),
        update.get("failure_reason"),
        limit,
    )
    feedback = output.get("task_feedback") if isinstance(output.get("task_feedback"), dict) else {}
    memory["task_findings"] = merge_task_findings(
        memory.get("task_findings") or [],
        feedback.get("findings") or [],
        limit,
    )
    memory["latest_task_feedback"] = make_json_safe(feedback)
    memory["failure_reasons"] = append_unique(
        list(memory.get("failure_reasons") or []),
        feedback.get("failure_reason"),
        limit,
    )
    hazards = list(memory.get("hazards") or [])
    for hazard in output.get("hazards") or []:
        hazards = append_unique(hazards, hazard, limit)
    memory["hazards"] = hazards[-limit:]

    observation = {
        "frame_idx": frame_idx,
        "place": memory.get("current_place", ""),
        "guidance": memory.get("next_direction_hint", ""),
        "instruction": output.get("current_subgoal", ""),
        "evidence": output.get("visual_evidence", ""),
        "feedback": feedback.get("summary", ""),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    observations = list(memory.get("observations") or [])
    observations.append(observation)
    memory["observations"] = observations[-limit:]
    memory["last_frame_idx"] = frame_idx
    memory["updated_at"] = datetime.now().isoformat(timespec="seconds")
    return memory


def save_route_memory(run_dir, memory):
    path = memory_path(run_dir)
    path.write_text(json.dumps(make_json_safe(memory), indent=2, ensure_ascii=False))
    return memory


def save_task_report(run_dir, task_instruction, output, route_memory, frame_idx):
    """Persist the latest accumulated user-facing report for one experiment."""
    feedback = output.get("task_feedback") if isinstance(output.get("task_feedback"), dict) else {}
    report = {
        "task_instruction": str(task_instruction or "").strip(),
        "task_status": str(output.get("task_status") or "running").strip().lower(),
        "frame_idx": frame_idx,
        "summary": str(feedback.get("summary") or "").strip(),
        "count": feedback.get("count"),
        "count_label": str(feedback.get("count_label") or "").strip(),
        "findings": make_json_safe(route_memory.get("task_findings") or feedback.get("findings") or []),
        "failure_reason": str(feedback.get("failure_reason") or "").strip(),
        "recommendation": str(feedback.get("recommendation") or "").strip(),
        "final": str(output.get("task_status") or "running").strip().lower() in {"completed", "failed"},
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    path = task_report_path(run_dir)
    temporary_path = path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    os.replace(temporary_path, path)
    return report


def load_events(run_dir, limit=6):
    path = event_log_path(run_dir)
    if not path.exists() or limit <= 0:
        return []
    lines = path.read_text().splitlines()[-limit:]
    events = []
    for line in lines:
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def load_latest_event(run_dir):
    path = latest_event_path(run_dir)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def find_demo_step_reference_event(run_dir, demo_context):
    """Find the earliest recorded observation for the active Demo step."""
    if not demo_context:
        return None
    expected_step = demo_context.get("current_step_index")
    expected_command = normalize_subgoal_text(demo_context.get("current_command"))
    for event in load_events(run_dir, limit=100000):
        output = event.get("output") if isinstance(event, dict) else {}
        demo_output = output.get("demo_agent") if isinstance(output, dict) else {}
        event_step = demo_output.get("step_index") if isinstance(demo_output, dict) else None
        if event_step is None and isinstance(output, dict):
            event_step = output.get("demo_step_index")
        event_command = normalize_subgoal_text(
            demo_output.get("command")
            if isinstance(demo_output, dict) and demo_output.get("command")
            else output.get("current_subgoal")
        )
        if event_step == expected_step and event_command == expected_command and event.get("image_file"):
            return {
                "frame_idx": event.get("frame_idx"),
                "image_file": event.get("image_file"),
            }
    return None


def find_first_saved_frame_at_or_after(run_dir, frame_idx_hint, current_frame_idx=None):
    if frame_idx_hint is None:
        return None
    try:
        minimum = int(frame_idx_hint)
        maximum = int(current_frame_idx) if current_frame_idx is not None else None
    except (TypeError, ValueError):
        return None
    pattern = re.compile(r"frame_(\d+)_rgb\.(?:jpg|jpeg|png)$", re.IGNORECASE)
    candidates = []
    for path in Path(run_dir).glob("frame_*_rgb.*"):
        match = pattern.match(path.name)
        if not match:
            continue
        frame_idx = int(match.group(1))
        if frame_idx >= minimum and (maximum is None or frame_idx <= maximum):
            candidates.append((frame_idx, path.name))
    if not candidates:
        return None
    frame_idx, image_file = min(candidates)
    return {"frame_idx": frame_idx, "image_file": image_file}


def select_motion_context_frames(
    run_dir,
    start_image_file,
    current_image_file,
    frame_count=8,
    current_command="",
):
    """Select an execution clip with motion transitions and pre-STOP context."""
    run_dir = Path(run_dir)
    pattern = re.compile(r"frame_(\d+)_rgb\.(?:jpg|jpeg|png)$", re.IGNORECASE)
    available = []
    for path in run_dir.glob("frame_*_rgb.*"):
        match = pattern.match(path.name)
        if match:
            available.append((int(match.group(1)), path.name))
    available.sort()
    if not available:
        return []

    by_name = {name: idx for idx, name in available}
    start_idx = by_name.get(str(start_image_file or ""), available[0][0])
    current_idx = by_name.get(str(current_image_file or ""), available[-1][0])
    start_idx = min(start_idx, current_idx)
    eligible = [(idx, name) for idx, name in available if start_idx <= idx <= current_idx]
    if not eligible:
        return []

    count = max(4, min(int(frame_count or 8), 12, len(eligible)))
    required_turn = required_turn_from_command(current_command)
    motion_labels = []
    for frame_idx, _ in eligible:
        metadata_path = run_dir / f"frame_{frame_idx:06d}_waypoint.json"
        motion = "unknown"
        try:
            metadata = json.loads(metadata_path.read_text())
            motion = summarize_low_level_motion(metadata.get("response") or {}).get(
                "planned_motion", "unknown"
            )
        except (OSError, json.JSONDecodeError):
            pass
        motion_labels.append(motion)

    recent_count = min(3, max(1, count // 4))
    mandatory = {0, len(eligible) - 1}
    mandatory.update(range(max(0, len(eligible) - recent_count), len(eligible)))
    for position in range(1, len(motion_labels)):
        if motion_labels[position] != motion_labels[position - 1]:
            mandatory.update({position - 1, position})
    if required_turn in {"left", "right"}:
        matching = [i for i, motion in enumerate(motion_labels) if motion == required_turn]
        if matching:
            mandatory.update({matching[0], matching[-1]})

    uniform = {
        round(position * (len(eligible) - 1) / max(1, count - 1))
        for position in range(count)
    }
    candidates = sorted(mandatory | uniform)
    if len(candidates) > count:
        must_keep = sorted(
            {0, len(eligible) - 1}
            | set(range(max(0, len(eligible) - recent_count), len(eligible)))
        )
        remaining = [position for position in candidates if position not in must_keep]
        slots = max(0, count - len(must_keep))
        if slots and remaining:
            chosen = {
                remaining[round(i * (len(remaining) - 1) / max(1, slots - 1))]
                for i in range(slots)
            }
        else:
            chosen = set()
        positions = sorted(set(must_keep) | chosen)
    else:
        positions = candidates
    return [
        {"frame_idx": eligible[position][0], "image_file": eligible[position][1]}
        for position in positions
    ]


def required_turn_from_command(command):
    command = str(command or "").casefold()
    if "turn around" in command or "u-turn" in command or "u turn" in command:
        return "turn_around"
    if "turn left" in command:
        return "left"
    if "turn right" in command:
        return "right"
    return "none"


def summarize_low_level_motion(response):
    """Convert one low-level response into compact trajectory/action evidence."""
    response = response if isinstance(response, dict) else {}
    actions = response.get("discrete_action")
    if isinstance(actions, list) and actions:
        normalized_actions = [int(value) for value in actions if isinstance(value, (int, float))]
        if 2 in normalized_actions and 3 not in normalized_actions:
            motion = "left"
        elif 3 in normalized_actions and 2 not in normalized_actions:
            motion = "right"
        elif normalized_actions and all(value == 0 for value in normalized_actions):
            motion = "stop"
        elif normalized_actions and all(value == 1 for value in normalized_actions):
            motion = "straight"
        else:
            motion = "mixed"
        return {
            "output_type": "discrete_action",
            "planned_motion": motion,
            "discrete_action": normalized_actions,
        }

    trajectory = response.get("trajectory")
    points = []
    if isinstance(trajectory, list):
        for point in trajectory:
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                try:
                    points.append((float(point[0]), float(point[1])))
                except (TypeError, ValueError):
                    continue
    if not points:
        return {"output_type": "none", "planned_motion": "unknown"}

    endpoint_x, endpoint_y = points[-1]
    path_length = sum(
        math.hypot(x2 - x1, y2 - y1)
        for (x1, y1), (x2, y2) in zip(points, points[1:])
    )
    turn_angle_deg = math.degrees(math.atan2(endpoint_y, max(0.05, endpoint_x)))
    if endpoint_y >= 0.15 and turn_angle_deg >= 10.0:
        motion = "left"
    elif endpoint_y <= -0.15 and turn_angle_deg <= -10.0:
        motion = "right"
    else:
        motion = "straight"
    sample_count = min(8, len(points))
    sample_positions = {
        round(position * (len(points) - 1) / max(1, sample_count - 1))
        for position in range(sample_count)
    }
    return {
        "output_type": "trajectory",
        "planned_motion": motion,
        "trajectory_point_count": len(points),
        "endpoint_xy_m": [round(endpoint_x, 3), round(endpoint_y, 3)],
        "path_length_m": round(path_length, 3),
        "endpoint_bearing_deg": round(turn_angle_deg, 1),
        "trajectory_sample_xy_m": [
            [round(points[position][0], 3), round(points[position][1], 3)]
            for position in sorted(sample_positions)
        ],
    }


def build_demo_execution_evidence(
    run_dir,
    motion_frames,
    current_command,
    completion_confidence_threshold=0.75,
    transition_mode="balanced",
):
    """Attach sampled trajectories and scan the whole step for required turns."""
    if not motion_frames:
        return None
    start_idx = int(motion_frames[0]["frame_idx"])
    end_idx = int(motion_frames[-1]["frame_idx"])
    sampled_indices = {int(item["frame_idx"]) for item in motion_frames}
    sampled = []
    timeline = []
    motion_counts = {"left": 0, "right": 0, "straight": 0, "stop": 0, "mixed": 0, "unknown": 0}

    for path in sorted(Path(run_dir).glob("frame_*_waypoint.json"), key=frame_sort_key):
        match = re.search(r"frame_(\d+)_waypoint\.json$", path.name)
        if not match:
            continue
        frame_idx = int(match.group(1))
        if frame_idx < start_idx or frame_idx > end_idx:
            continue
        try:
            metadata = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        summary = summarize_low_level_motion(metadata.get("response") or {})
        motion = summary.get("planned_motion", "unknown")
        motion_counts[motion] = motion_counts.get(motion, 0) + 1
        timeline.append({"frame_idx": frame_idx, "planned_motion": motion})
        if frame_idx in sampled_indices:
            sampled.append(
                {
                    "frame_idx": frame_idx,
                    "instruction": str(metadata.get("instruction") or ""),
                    **summary,
                }
            )

    required_turn = required_turn_from_command(current_command)
    matching_turn_frames = motion_counts.get(required_turn, 0) if required_turn in {"left", "right"} else 0
    segments = []
    for item in timeline:
        if not segments or segments[-1]["planned_motion"] != item["planned_motion"]:
            segments.append(
                {
                    "planned_motion": item["planned_motion"],
                    "start_frame_idx": item["frame_idx"],
                    "end_frame_idx": item["frame_idx"],
                    "frame_count": 1,
                }
            )
        else:
            segments[-1]["end_frame_idx"] = item["frame_idx"]
            segments[-1]["frame_count"] += 1
    matching_indices = [
        item["frame_idx"] for item in timeline if item["planned_motion"] == required_turn
    ]
    matching_turn_segment_count = sum(
        1 for item in segments if item["planned_motion"] == required_turn
    )
    endpoint_visual_change_score = 0.0
    if len(motion_frames) >= 2:
        try:
            with Image.open(Path(run_dir) / motion_frames[0]["image_file"]) as start_image:
                start_pixels = list(start_image.convert("RGB").resize((48, 36)).getdata())
            with Image.open(Path(run_dir) / motion_frames[-1]["image_file"]) as end_image:
                end_pixels = list(end_image.convert("RGB").resize((48, 36)).getdata())
            absolute_difference = sum(
                abs(start_channel - end_channel)
                for start_pixel, end_pixel in zip(start_pixels, end_pixels)
                for start_channel, end_channel in zip(start_pixel, end_pixel)
            )
            endpoint_visual_change_score = absolute_difference / max(
                1, len(start_pixels) * 3 * 255
            )
        except (OSError, KeyError):
            pass
    return {
        "clip_start_frame_idx": start_idx,
        "clip_end_frame_idx": end_idx,
        "clip_frame_count": max(0, end_idx - start_idx + 1),
        "clip_ended_with_stop": bool(timeline and timeline[-1]["planned_motion"] == "stop"),
        "coordinate_convention": "trajectory x=forward meters, y=left-positive/right-negative meters",
        "required_turn": required_turn,
        "matching_turn_output_count": matching_turn_frames,
        "matching_turn_segment_count": matching_turn_segment_count,
        "first_matching_turn_frame_idx": matching_indices[0] if matching_indices else None,
        "last_matching_turn_frame_idx": matching_indices[-1] if matching_indices else None,
        "has_required_turn_output": required_turn not in {"left", "right"} or matching_turn_frames > 0,
        "completion_confidence_threshold": float(completion_confidence_threshold),
        "transition_mode": str(transition_mode or "balanced"),
        "endpoint_visual_change_score": round(endpoint_visual_change_score, 4),
        "motion_output_counts_for_full_step": motion_counts,
        "motion_segments": segments,
        "sampled_frame_outputs": sampled,
        "important_limitation": (
            "Trajectory is the low-level planned path, not measured odometry. "
            "Use it with the ordered camera frames, never as sole proof of physical execution."
        ),
    }


def append_event(run_dir, event):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    safe_event = make_json_safe(event)
    with open(event_log_path(run_dir), "a") as f:
        f.write(json.dumps(safe_event, ensure_ascii=False) + "\n")
    latest_event_path(run_dir).write_text(json.dumps(safe_event, indent=2, ensure_ascii=False))
    return safe_event


def make_json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(v) for v in value]
    return value


def image_to_data_url(path, max_width=768):
# Qwen-VL API 需要图片 data URL。这里会先压缩宽度，降低网络耗时和 token/费用。
    image = Image.open(path).convert("RGB")
    if max_width and image.width > max_width:
        new_height = max(1, int(image.height * max_width / image.width))
        image = image.resize((max_width, new_height))
    from io import BytesIO

    buf = BytesIO()
    image.save(buf, format="JPEG", quality=85)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def extract_json(text):
# 模型有时会返回 ```json ... ``` 或在 JSON 前后加解释文本；
# 这里做容错抽取，尽量拿到中间的 JSON 对象。
    if not text:
        return None
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(stripped[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


def contains_cjk(text):
    """Return whether text still contains Chinese/Japanese/Korean ideographs."""
    return bool(re.search(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", str(text or "")))


def normalize_agent_output(output):
# 把模型输出规范成固定字段，缺字段时给默认值。
# 这样后续的安全改写、记忆更新、网页展示都能依赖稳定 schema。
    if not isinstance(output, dict):
        output = {}
    memory = output.get("memory") if isinstance(output.get("memory"), dict) else {}
    if not memory and isinstance(output.get("route_memory_update"), dict):
        memory = output.get("route_memory_update")
    task_status = str(output.get("task_status") or "running").strip().lower()
    if task_status not in {"running", "completed", "failed"}:
        task_status = "running"
    current_subgoal = str(output.get("current_subgoal") or output.get("next_instruction") or "").strip()
    if task_status in {"completed", "failed"}:
        current_subgoal = ""
    task_feedback = normalize_task_feedback(output.get("task_feedback"), task_status, memory)
    raw_assessment = output.get("execution_assessment")
    raw_assessment = raw_assessment if isinstance(raw_assessment, dict) else {}
    try:
        completion_confidence = max(0.0, min(1.0, float(raw_assessment.get("completion_confidence") or 0.0)))
    except (TypeError, ValueError):
        completion_confidence = 0.0
    evidence_frame_indices = []
    for value in raw_assessment.get("evidence_frame_indices") or []:
        try:
            evidence_frame_indices.append(int(value))
        except (TypeError, ValueError):
            continue
    normalized = {
        "task_status": task_status,
        "navigation_phase": str(output.get("navigation_phase") or "advance"),
        "current_subgoal": current_subgoal,
        "demo_step_index": output.get("demo_step_index"),
        "demo_step_decision": str(output.get("demo_step_decision") or "hold").strip().lower(),
        "execution_assessment": {
            "subgoal_completed": bool(raw_assessment.get("subgoal_completed", False)),
            "forward_completed": bool(raw_assessment.get("forward_completed", False)),
            "turn_started": bool(raw_assessment.get("turn_started", False)),
            "turn_completed": bool(raw_assessment.get("turn_completed", False)),
            "observed_turn_direction": str(raw_assessment.get("observed_turn_direction") or "uncertain"),
            "completion_confidence": completion_confidence,
            "evidence_frame_indices": evidence_frame_indices[:20],
            "reason": str(raw_assessment.get("reason") or "").strip()[:1000],
        },
        "visual_evidence": str(output.get("visual_evidence") or "").strip(),
        "task_feedback": task_feedback,
        "memory": {
            "current_place": str(memory.get("current_place") or "").strip(),
            "landmarks_seen": memory.get("landmarks_seen") if isinstance(memory.get("landmarks_seen"), list) else [],
            "completed_subgoal": str(memory.get("completed_subgoal") or "").strip(),
            "next_direction_hint": str(memory.get("next_direction_hint") or "").strip(),
            "failure_reason": str(memory.get("failure_reason") or "").strip(),
        },
    }
    return normalized


def normalize_task_feedback(feedback, task_status="running", memory=None):
    """Normalize user-facing task results without leaking them into navigation."""
    feedback = feedback if isinstance(feedback, dict) else {}
    memory = memory if isinstance(memory, dict) else {}
    raw_count = feedback.get("count")
    count = None
    if raw_count is not None and not isinstance(raw_count, bool):
        try:
            count = max(0, int(raw_count))
        except (TypeError, ValueError):
            count = None

    findings = []
    for item in feedback.get("findings") or []:
        if not isinstance(item, dict):
            continue
        description = str(item.get("description") or "").strip()
        if not description:
            continue
        severity = str(item.get("severity") or "info").strip().lower()
        if severity not in {"info", "warning", "critical"}:
            severity = "info"
        findings.append(
            {
                "type": str(item.get("type") or "observation").strip().lower()[:40],
                "description": description[:500],
                "location": str(item.get("location") or "").strip()[:300],
                "severity": severity,
                "evidence": str(item.get("evidence") or "").strip()[:500],
            }
        )
        if len(findings) >= 20:
            break

    failure_reason = str(
        feedback.get("failure_reason")
        or (memory.get("failure_reason") if task_status == "failed" else "")
        or ""
    ).strip()
    return {
        "summary": str(feedback.get("summary") or "").strip()[:1200],
        "count": count,
        "count_label": str(feedback.get("count_label") or "").strip()[:120],
        "findings": findings,
        "failure_reason": failure_reason[:1000],
        "recommendation": str(feedback.get("recommendation") or "").strip()[:800],
    }


def normalize_subgoal_format(output):
    """Normalize formatting without changing the model's navigation decision."""
    output = dict(output)
    instruction = str(output.get("current_subgoal") or "").strip()
    instruction = re.sub(r"\s+", " ", instruction).strip()
    instruction = re.sub(r"\s+([,.!?])", r"\1", instruction)
    output["current_subgoal"] = instruction
    return output


def safety_rewrite_instruction(output, config):
# 保留接口兼容性；不改变智能体生成的导航语义。
    output = dict(output)
    instruction = str(output.get("current_subgoal") or "").strip()
    if not instruction:
        return output
    if not config.get("safety_mode", True):
        return output

    output["current_subgoal"] = instruction
    return output


def compact_low_level_response(response):
    """Keep control-relevant low-level state without serializing full trajectories."""
    response = sanitize_runtime_config(response if isinstance(response, dict) else {})
    compact = {}
    for key in ("discrete_action", "replan_required", "instruction", "pixel_goal"):
        if key in response:
            compact[key] = response[key]
    trajectory = response.get("trajectory")
    if isinstance(trajectory, list) and trajectory:
        compact["trajectory_points"] = len(trajectory)
        compact["trajectory_start"] = trajectory[0]
        compact["trajectory_end"] = trajectory[-1]
    return compact


def compact_route_memory(memory):
    """The VLM needs state, not the full append-only experiment notebook."""
    memory = memory if isinstance(memory, dict) else {}
    return {
        "active_subgoal": str(memory.get("active_subgoal") or ""),
        "active_subgoal_started_frame": memory.get("active_subgoal_started_frame"),
        "current_place": str(memory.get("current_place") or ""),
        "next_direction_hint": str(memory.get("next_direction_hint") or ""),
        "visited_landmarks": list(memory.get("visited_landmarks") or [])[-6:],
        "completed_subgoals": list(memory.get("completed_subgoals") or [])[-3:],
        "failure_reasons": list(memory.get("failure_reasons") or [])[-2:],
        "same_subgoal_decisions": int(memory.get("same_subgoal_decisions") or 0),
    }


def compact_demo_execution_evidence(evidence):
    """Retain temporal turn evidence while dropping the verbose full-step trace."""
    if not isinstance(evidence, dict):
        return None
    return {
        key: evidence.get(key)
        for key in (
            "required_turn",
            "matching_turn_output_count",
            "matching_turn_segment_count",
            "first_matching_turn_frame_idx",
            "last_matching_turn_frame_idx",
            "endpoint_visual_change_score",
            "clip_start_frame_idx",
            "clip_end_frame_idx",
            "clip_ended_with_stop",
            "has_required_turn_output",
        )
    }


def compact_recent_events(events):
    """Use only the two newest control decisions as short recurrence context."""
    compact = []
    for event in list(events or [])[-2:]:
        output = event.get("output") if isinstance(event, dict) else {}
        output = output if isinstance(output, dict) else {}
        assessment = output.get("execution_assessment") if isinstance(output.get("execution_assessment"), dict) else {}
        compact.append(
            {
                "frame_idx": event.get("frame_idx"),
                "status": output.get("task_status"),
                "subgoal": output.get("current_subgoal"),
                "decision": output.get("demo_step_decision"),
                "completed": assessment.get("subgoal_completed"),
            }
        )
    return compact


def build_prompt(
    metadata,
    recent_events,
    runtime_config,
    route_memory,
    config,
    demo_execution_evidence=None,
    long_term_memories=None,
):
# 组装发给上层智能体的 prompt。
# 关键隔离：
# - task_instruction：用户给上层智能体的完整任务。
# - current_low_level_instruction：上一条已经发给 InternVLA-N1 的短指令。
# 智能体必须根据完整任务和当前图像，产出新的 current_subgoal。
    response = sanitize_runtime_config(metadata.get("response") or {})
    low_level_instruction = runtime_config.get("instruction", "") or metadata.get("instruction") or ""
    task_instruction = config.get("task_instruction") or metadata.get("agent_task_instruction") or low_level_instruction
    context = {
        "current_frame_idx": metadata.get("frame_idx"),
        "task_instruction": task_instruction,
        "current_low_level_instruction": low_level_instruction,
        "low_level_response": compact_low_level_response(response),
        "route_memory": compact_route_memory(route_memory),
        # Cross-run memory is advisory only. Current images and safety state
        # always take priority over a retrieved historical route.
        "cross_experiment_memory": list(long_term_memories or []),
        "recent_control_decisions": compact_recent_events(recent_events),
        "subgoal_watchdog": subgoal_watchdog_status(route_memory, metadata.get("frame_idx"), config),
        "demo_agent": demo_prompt_context(runtime_config),
        "demo_execution_evidence": compact_demo_execution_evidence(demo_execution_evidence),
    }
    demo_instruction = ""
    if context["demo_agent"]:
        demo_instruction = (
            "Demo Agent is active: choose hold, advance one step, complete, or failed. "
            "Do not rewrite its command. Use the ordered images plus compact turn evidence; "
            "STOP alone never proves completion.\n"
        )
    return (
        "Inspect the ordered images and compact state. Produce the required fast control JSON.\n"
        f"{demo_instruction}"
        "task_instruction is the user task. current_low_level_instruction is only the previous command. "
        "cross_experiment_memory is verified advisory history; current images and safety evidence always win. "
        "Reuse success memories only when the current scene matches. Treat failure memories as warnings to avoid, "
        "never as actions to copy. Treat finding memories as past observations, not proof of the current view. "
        "Return an empty current_subgoal only when completed or failed.\n"
        f"Context:{json.dumps(context, ensure_ascii=False, separators=(',', ':'))}"
    )


def call_qwen_vl(
    config,
    image_data_url,
    prompt,
    reference_image_data_url=None,
    motion_context_images=None,
    max_tokens_override=None,
    temperature_override=None,
):
# 真正调用 Qwen-VL 的地方。
# api_key 只用于 Authorization header，不会写入事件日志，也不会返回给前端。
    api_key = resolve_api_key(config)
    if not api_key:
        raise RuntimeError("Upper agent API key is missing. Fill API key or set the configured env var.")
    api_url = resolve_api_url(config) or DEFAULT_UPPER_AGENT_CONFIG["api_url"]
    system_prompt = config.get("system_prompt") or UPPER_AGENT_SYSTEM_PROMPT
    user_content = [{"type": "text", "text": prompt}]
    if motion_context_images:
        user_content.append(
            {
                "type": "text",
                "text": "Demo motion sequence, ordered from step start to current frame:",
            }
        )
        for position, item in enumerate(motion_context_images):
            user_content.extend(
                [
                    {
                        "type": "text",
                        "text": (
                            f"Motion frame {position + 1}/{len(motion_context_images)}, "
                            f"saved frame_idx={item.get('frame_idx')}:"
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": item["data_url"]}},
                ]
            )
        user_content.append(
            {
                "type": "text",
                "text": "Compare all motion frames before deciding hold or advance.",
            }
        )
    elif reference_image_data_url:
        user_content.extend(
            [
                {"type": "text", "text": "Demo step-start reference image:"},
                {"type": "image_url", "image_url": {"url": reference_image_data_url}},
                {"type": "text", "text": "Current image to evaluate:"},
            ]
        )
    if not motion_context_images:
        user_content.append({"type": "image_url", "image_url": {"url": image_data_url}})
    # The current schema contains navigation, execution assessment, feedback,
    # and memory. Cloud models can use a larger JSON budget; the local service
    # intentionally runs with a 4096-token context window, so it needs a
    # smaller output budget to leave room for multi-frame visual context.
    requested_max_tokens = int(
        max_tokens_override if max_tokens_override is not None else config.get("max_tokens", 512)
    )
    local_qwen = is_local_qwen_model(config.get("model"))
    output_budget = (
        max(256, min(512, requested_max_tokens))
        if local_qwen
        else max(1024, min(4096, requested_max_tokens))
    )
    payload = {
        "model": resolve_model_name(config) or DEFAULT_UPPER_AGENT_CONFIG["model"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": user_content,
            },
        ],
        "max_tokens": output_budget,
        "temperature": float(
            temperature_override
            if temperature_override is not None
            else config.get("temperature", 0.2)
        ),
        "response_format": {"type": "json_object"},
    }
    if local_qwen:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    started = time.time()
    response = requests.post(api_url, headers=headers, json=payload, timeout=60)
    # Some older or third-party OpenAI-compatible endpoints do not implement
    # response_format. Keep compatibility while preferring JSON mode whenever
    # the configured Qwen endpoint supports it.
    if response.status_code == 400 and "response_format" in response.text.lower():
        payload.pop("response_format", None)
        response = requests.post(api_url, headers=headers, json=payload, timeout=60)
    # Token counts are computed by the serving model, not by our lightweight
    # client. If an unusually long visual prompt still exceeds the local
    # 4096-token window, retry once with the smallest useful JSON budget.
    if (
        local_qwen
        and response.status_code == 400
        and "maximum context length" in response.text.lower()
        and payload["max_tokens"] > 256
    ):
        payload["max_tokens"] = 256
        response = requests.post(api_url, headers=headers, json=payload, timeout=60)
    elapsed = time.time() - started
    if not response.ok:
        detail = response.text.strip().replace("\n", " ")[:1000]
        raise RuntimeError(f"Qwen API HTTP {response.status_code}: {detail or response.reason}")
    data = response.json()
    text = data["choices"][0]["message"]["content"]
    if isinstance(text, list):
        text = " ".join(str(item.get("text") or "") for item in text if isinstance(item, dict))
    return text, data, elapsed


def rewrite_spoken_navigation_instruction(config, transcript, target="upper_task"):
    """Translate and polish a spoken instruction without exposing credentials."""
    config = normalize_upper_agent_config(config)
    api_key = resolve_api_key(config)
    if not api_key:
        raise RuntimeError("Upper agent API key is missing. Fill API key or set the configured env var.")
    transcript = str(transcript or "").strip()
    if not transcript:
        raise ValueError("语音转写结果为空。")
    if target not in {"low_level", "upper_task"}:
        raise ValueError("不支持的语音指令目标。")

    target_rule = (
        "Rewrite it as one concise, immediately executable low-level navigation command. "
        "Use imperative motion verbs, preserve every stated direction and landmark, and avoid explanations."
        if target == "low_level"
        else
        "Rewrite it as a clear high-level navigation task for an upper agent. Preserve the full goal, "
        "ordered route steps, landmarks, and stopping condition, while removing filler and speech disfluencies."
    )
    system_prompt = (
        "You are a navigation-instruction editor. Translate spoken Chinese or mixed-language input into natural English. "
        "Never invent destinations, directions, landmarks, or stopping conditions. Return only the final English instruction, "
        "with no JSON, markdown, quotation marks, commentary, or alternatives."
    )
    payload = {
        "model": resolve_model_name(config) or DEFAULT_UPPER_AGENT_CONFIG["model"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{target_rule}\n\nSpoken transcript:\n{transcript}"},
        ],
        "max_tokens": min(256, int(config.get("max_tokens", 512))),
        "temperature": min(0.2, float(config.get("temperature", 0.2))),
    }
    if is_local_qwen_model(config.get("model")):
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    response = requests.post(
        resolve_api_url(config) or DEFAULT_UPPER_AGENT_CONFIG["api_url"],
        headers=headers,
        json=payload,
        timeout=60,
    )
    if not response.ok:
        detail = response.text.strip().replace("\n", " ")[:1000]
        raise RuntimeError(f"Qwen API HTTP {response.status_code}: {detail or response.reason}")
    content = response.json()["choices"][0]["message"]["content"]
    if isinstance(content, list):
        content = " ".join(str(item.get("text") or "") for item in content if isinstance(item, dict))
    instruction = str(content or "").strip().strip('"').strip("'").strip()
    if not instruction:
        raise RuntimeError("Qwen 没有返回有效的英文指令。")
    if contains_cjk(instruction):
        raise RuntimeError("Qwen 返回结果仍包含中文，英文转换未完成。")
    return instruction


def classify_spoken_navigation_command(
    config,
    transcript,
    target="low_level",
    confidence_threshold=0.78,
):
    """Conservatively classify speech and normalize only explicit robot commands."""
    config = normalize_upper_agent_config(config)
    api_key = resolve_api_key(config)
    if not api_key:
        raise RuntimeError("Upper agent API key is missing. Fill API key or set the configured env var.")
    transcript = re.sub(r"\s+", " ", str(transcript or "")).strip()
    if not transcript:
        raise ValueError("语音转写结果为空。")
    if target != "low_level":
        raise ValueError("交互模式目前只支持低层导航指令。")

    system_prompt = """
You are a conservative command gate for a quadruped navigation robot.
Decide whether the transcript is a complete, explicit command directed at the
robot. Accept navigation or motion commands such as go straight, turn, follow,
find, approach, stop, wait, or return. Translate accepted Chinese or mixed
speech into one concise English imperative instruction for the low-level
navigation model while preserving all stated directions, landmarks, order, and
stopping conditions.

Use command_type "stop" only when the complete intent is to stop immediately,
with no navigation action before it. Commands such as "go to the sofa and stop"
are command_type "navigate" because STOP is only their final route condition.
Use command_type "none" for rejected speech.

Reject background conversation, speech addressed to another person, questions
about the robot, descriptions of what the robot is doing, discussion or quoting
of a possible command, acknowledgements, filler, incomplete fragments, ASR
noise, and ambiguous speech. When uncertain, reject. Never invent an action,
direction, landmark, destination, or stopping condition.

Return only valid JSON:
{
  "classification": "command|non_command|uncertain",
  "command_type": "navigate|stop|none",
  "confidence": 0.0,
  "instruction": "English command, or empty when rejected",
  "reason": "brief reason"
}
""".strip()
    payload = {
        "model": resolve_model_name(config) or DEFAULT_UPPER_AGENT_CONFIG["model"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Speech transcript:\n{transcript}"},
        ],
        "max_tokens": min(256, int(config.get("max_tokens", 512))),
        "temperature": 0.0,
    }
    if is_local_qwen_model(config.get("model")):
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    response = requests.post(
        resolve_api_url(config) or DEFAULT_UPPER_AGENT_CONFIG["api_url"],
        headers=headers,
        json=payload,
        timeout=60,
    )
    if not response.ok:
        detail = response.text.strip().replace("\n", " ")[:1000]
        raise RuntimeError(f"Qwen API HTTP {response.status_code}: {detail or response.reason}")
    content = response.json()["choices"][0]["message"]["content"]
    if isinstance(content, list):
        content = " ".join(str(item.get("text") or "") for item in content if isinstance(item, dict))
    parsed = extract_json(str(content or ""))
    if not isinstance(parsed, dict):
        raise RuntimeError("Qwen 没有返回有效的命令判别 JSON。")

    classification = str(parsed.get("classification") or "uncertain").strip().lower()
    if classification not in {"command", "non_command", "uncertain"}:
        classification = "uncertain"
    try:
        confidence = max(0.0, min(1.0, float(parsed.get("confidence") or 0.0)))
    except (TypeError, ValueError):
        confidence = 0.0
    instruction = re.sub(r"\s+", " ", str(parsed.get("instruction") or "")).strip().strip('"').strip("'")
    command_type = str(parsed.get("command_type") or "none").strip().lower()
    if command_type not in {"navigate", "stop", "none"}:
        command_type = "none"
    normalized_instruction = re.sub(r"[^a-z]+", " ", instruction.lower()).strip()
    explicit_stop_phrases = {
        "stop",
        "stop now",
        "stop here",
        "stop moving",
        "stop the robot",
        "halt",
        "halt now",
        "stay still",
    }
    if classification == "command" and normalized_instruction in explicit_stop_phrases:
        command_type = "stop"
    elif classification == "command" and command_type == "none":
        command_type = "navigate"
    try:
        confidence_threshold = float(confidence_threshold)
    except (TypeError, ValueError):
        confidence_threshold = 0.78
    confidence_threshold = max(0.5, min(1.0, confidence_threshold))
    accepted = classification == "command" and confidence >= confidence_threshold and bool(instruction)
    if accepted and contains_cjk(instruction):
        # Interaction mode uses the command classifier instead of the regular
        # refine route. Enforce the same English-only contract here as well.
        instruction = rewrite_spoken_navigation_instruction(config, instruction, target="low_level")
    if not accepted:
        instruction = ""
    return {
        "accepted": accepted,
        "classification": classification,
        "command_type": command_type if accepted else "none",
        "confidence": confidence,
        "confidence_threshold": confidence_threshold,
        "instruction": instruction,
        "reason": str(parsed.get("reason") or "").strip()[:300],
    }


def should_evaluate(run_dir, metadata, config, force=False):
# 判断这一帧要不要触发上层智能体：
# 1. enabled 没开就跳过。
# 2. read_every_n_frames 控制按帧触发频率。
# 3. min_seconds_between_calls 控制按时间限流。
    if force:
        return True, ""
    if not config.get("enabled"):
        return False, "upper agent disabled"
    if str(config.get("last_task_status") or "running").strip().lower() in {"completed", "failed"}:
        return False, "task already reached a terminal state"
    frame_idx = int(metadata.get("frame_idx") or 0)
    every_n = int(config.get("read_every_n_frames") or 1)
    if every_n > 1 and frame_idx % every_n != 0:
        return False, f"waiting for frame interval: frame_idx {frame_idx} is not divisible by {every_n}"

    latest = load_latest_event(run_dir)
    if latest:
        if int(latest.get("frame_idx") or -1) == frame_idx:
            return False, "latest frame already evaluated"
        min_gap = float(config.get("min_seconds_between_calls") or 0.0)
        if min_gap > 0:
            last_ts = float(latest.get("timestamp") or 0.0)
            if time.time() - last_ts < min_gap:
                return False, "minimum call interval not reached"
    return True, ""


def upper_agent_pause_is_active(runtime_config):
# 防并发保护：如果已经有一个智能体请求正在思考，新的请求直接跳过。
    pause = (runtime_config or {}).get("_upper_agent_pause") or {}
    return bool(pause.get("active") and pause.get("token"))


def evaluate_latest(
    run_dir,
    runtime_config_path,
    force=False,
    request_low_level_hard_reset=False,
    settle_for_fresh_frame=False,
):
# 上层智能体主入口：
# 读取最新帧 -> 判断是否该调用 -> 构造 prompt -> 暂停低层策略 -> 调 Qwen ->
# 解析/安全改写 -> 更新记忆 -> 写回低层 instruction -> 记录事件。
    runtime_config = load_runtime_config(runtime_config_path)
    config = get_upper_agent_config(runtime_config)
    task_instruction = config.get("task_instruction") or runtime_config.get("instruction", "")
    metadata_path, metadata = latest_frame_metadata(run_dir)
    if metadata is None:
        raise FileNotFoundError(f"No frame metadata found in {run_dir}")

    if upper_agent_pause_is_active(runtime_config):
        return {
            "ok": True,
            "skipped": True,
            "reason": "upper agent is already thinking",
            "latest": load_latest_event(run_dir),
        }

    ok, reason = should_evaluate(run_dir, metadata, config, force=force)
    if not ok:
        return {
            "ok": True,
            "skipped": True,
            "reason": reason,
            "latest": load_latest_event(run_dir),
        }

    pause_state = None
    settle_seconds = 0.0
    parsed = None
    event = None
    try:
        if settle_for_fresh_frame:
            # Pause before sleeping: client requests received in this window
            # keep logging fresh RGB-D while its controller holds position.
            pause_state = acquire_policy_pause(runtime_config_path, config)
            if pause_state:
                settle_seconds = float(config.get("replan_settle_seconds") or 0.0)
                if settle_seconds > 0:
                    time.sleep(settle_seconds)
                # The frame chosen before the wait is stale by construction.
                # Re-read both config and latest metadata after the settle.
                runtime_config = load_runtime_config(runtime_config_path)
                config = get_upper_agent_config(runtime_config)
                task_instruction = config.get("task_instruction") or runtime_config.get("instruction", "")
                metadata_path, metadata = latest_frame_metadata(run_dir)
                if metadata is None:
                    raise FileNotFoundError(f"No frame metadata found in {run_dir}")

        rgb_file = metadata.get("rgb_file")
        if not rgb_file:
            raise FileNotFoundError("Frame metadata does not contain rgb_file.")
        image_path = Path(run_dir) / rgb_file
        if not image_path.exists():
            raise FileNotFoundError(image_path)

        route_memory = load_route_memory(run_dir) if config.get("enable_route_memory", True) else default_route_memory()
        previous_route_memory = dict(route_memory)
        recent_events = load_events(run_dir, config.get("history_events", 6))
        long_term_memories = []
        memory_retrieval_ms = 0.0
        memory_manager = None
        if config.get("enable_long_term_memory") or config.get("enable_graph_memory_capture"):
            memory_manager = get_long_term_memory_manager()
        if config.get("enable_long_term_memory") and memory_manager is not None:
            memory_query = build_retrieval_query(
                task_instruction,
                route_memory,
                runtime_config.get("instruction", ""),
            )
            retrieval_started = time.perf_counter()
            long_term_memories = memory_manager.retrieve(
                memory_query,
                top_k=config.get("long_term_memory_top_k", 3),
                char_budget=config.get("long_term_memory_char_budget", 600),
                timeout_ms=config.get("long_term_memory_timeout_ms", 180),
            )
            memory_retrieval_ms = (time.perf_counter() - retrieval_started) * 1000.0
        demo_context = demo_prompt_context(runtime_config)
        attempt_reference = find_first_saved_frame_at_or_after(
            run_dir,
            (demo_context or {}).get("attempt_started_frame_idx_hint"),
            metadata.get("frame_idx"),
        )
        historical_reference = attempt_reference or find_demo_step_reference_event(run_dir, demo_context)
        demo_reference = ensure_demo_step_reference(
            runtime_config_path,
            Path(run_dir).name,
            historical_reference.get("frame_idx") if historical_reference else metadata.get("frame_idx"),
            historical_reference.get("image_file") if historical_reference else rgb_file,
        )
        if demo_reference:
            runtime_config = load_runtime_config(runtime_config_path)
        image_data_url = image_to_data_url(image_path, config.get("max_image_width", 768))
        reference_image_data_url = None
        if demo_reference and demo_reference.get("image_file"):
            reference_path = Path(run_dir) / demo_reference["image_file"]
            if reference_path.exists() and reference_path != image_path:
                reference_image_data_url = image_to_data_url(
                    reference_path,
                    config.get("max_image_width", 768),
                )
        motion_context_frames = []
        motion_context_images = []
        if demo_reference:
            motion_context_frames = select_motion_context_frames(
                run_dir,
                demo_reference.get("image_file"),
                rgb_file,
                config.get("motion_context_frames", 8),
                current_command=(demo_context or {}).get("current_command", ""),
            )
            for item in motion_context_frames:
                motion_path = Path(run_dir) / item["image_file"]
                if motion_path.exists():
                    motion_context_images.append(
                        {
                            **item,
                            "data_url": image_to_data_url(
                                motion_path,
                                config.get("max_image_width", 768),
                            ),
                        }
                    )
        demo_execution_evidence = build_demo_execution_evidence(
            run_dir,
            motion_context_frames,
            (demo_context or {}).get("current_command", ""),
            config.get("demo_completion_confidence_threshold", 0.75),
            config.get("demo_transition_mode", "balanced"),
        )
        prompt = build_prompt(
            metadata,
            recent_events,
            runtime_config,
            route_memory,
            config,
            demo_execution_evidence=demo_execution_evidence,
            long_term_memories=long_term_memories,
        )
        if pause_state is None:
            pause_state = acquire_policy_pause(runtime_config_path, config)
        raw_text, raw_response, call_time = call_qwen_vl(
            config,
            image_data_url,
            prompt,
            reference_image_data_url=reference_image_data_url,
            motion_context_images=motion_context_images,
        )
        parsed = extract_json(raw_text)
        json_retry_count = 0
        if parsed is None:
            # Most historical failures were valid JSON prefixes cut off by the
            # output-token limit. Retry once with a larger deterministic budget
            # before surfacing an analysis failure to the operator.
            retry_config = dict(config)
            retry_config["system_prompt"] = (
                str(config.get("system_prompt") or UPPER_AGENT_SYSTEM_PROMPT)
                + "\nReturn compact valid JSON. Keep free-text fields concise."
            )
            retry_text, retry_response, retry_time = call_qwen_vl(
                retry_config,
                image_data_url,
                prompt,
                reference_image_data_url=reference_image_data_url,
                motion_context_images=motion_context_images,
                max_tokens_override=(
                    min(512, int(config.get("max_tokens", 512)))
                    if is_local_qwen_model(config.get("model"))
                    else max(1536, int(config.get("max_tokens", 512)) * 2)
                ),
                temperature_override=0.0,
            )
            json_retry_count = 1
            call_time += retry_time
            retry_parsed = extract_json(retry_text)
            if retry_parsed is not None:
                raw_text = retry_text
                raw_response = retry_response
                parsed = retry_parsed
        if parsed is None:
            parsed = {
                "task_status": "running",
                "navigation_phase": "observe",
                "current_subgoal": route_memory.get("active_subgoal", ""),
                "visual_evidence": raw_text,
                "task_feedback": {
                    "summary": "",
                    "count": None,
                    "count_label": "",
                    "findings": [],
                    "failure_reason": "model output was not valid JSON",
                    "recommendation": "Retry upper-agent evaluation.",
                },
                "memory": {
                    "current_place": route_memory.get("current_place", ""),
                    "landmarks_seen": [],
                    "completed_subgoal": "",
                    "next_direction_hint": route_memory.get("next_direction_hint", ""),
                    "failure_reason": "model output was not valid JSON",
                },
            }
        parsed = normalize_agent_output(parsed)
        parsed, demo_command_changed = constrain_demo_agent_output(
            runtime_config_path,
            parsed,
            run_name=Path(run_dir).name,
            frame_idx=metadata.get("frame_idx"),
            image_file=rgb_file,
            execution_evidence=demo_execution_evidence,
        )
        parsed = safety_rewrite_instruction(parsed, config)
        parsed = normalize_subgoal_format(parsed)
        route_memory = update_route_memory(route_memory, parsed, metadata.get("frame_idx"), config)
        if config.get("enable_route_memory", True):
            save_route_memory(run_dir, route_memory)
        memory_candidate = build_memory_candidate(
            previous_route_memory,
            parsed,
            run_name=Path(run_dir).name,
            frame_idx=metadata.get("frame_idx"),
            task_instruction=task_instruction,
        )
        memory_write_queued = bool(
            memory_manager
            and memory_manager.remember_async(
                memory_candidate,
                capture_graph=config.get("enable_graph_memory_capture", True),
            )
        )
        task_report = save_task_report(
            run_dir,
            task_instruction,
            parsed,
            route_memory,
            metadata.get("frame_idx"),
        )

        event = {
            "ok": True,
            "skipped": False,
            "timestamp": time.time(),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "frame_idx": metadata.get("frame_idx"),
            "metadata_path": metadata_path,
            "image_file": rgb_file,
            "demo_reference_image_file": (
                demo_reference.get("image_file") if demo_reference else ""
            ),
            "motion_context_frames": motion_context_frames,
            "demo_execution_evidence": demo_execution_evidence,
            "task_instruction": task_instruction,
            "model": config.get("model"),
            "call_time": call_time,
            "fresh_frame_wait_seconds": settle_seconds,
            "output": parsed,
            "raw_text": raw_text,
            "usage": raw_response.get("usage", {}),
            "json_retry_count": json_retry_count,
            "paused_policy_while_thinking": bool(pause_state),
            "route_memory": route_memory,
            "long_term_memory": {
                "retrieved": long_term_memories,
                "retrieval_ms": round(memory_retrieval_ms, 2),
                "write_queued": memory_write_queued,
                "status": long_term_memory_status(),
            },
            "task_report": task_report,
        }
    except Exception as exc:
        persist_upper_agent_error(runtime_config_path, f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if pause_state:
            low_level_instruction = parsed.get("current_subgoal") if isinstance(parsed, dict) else None
            updated_runtime_config = release_policy_pause(
                runtime_config_path,
                pause_state,
                low_level_instruction=low_level_instruction,
                auto_apply_instruction=config.get("auto_apply_instruction"),
                task_completed=bool(
                    isinstance(parsed, dict) and parsed.get("task_status") in {"completed", "failed"}
                ),
            )
            if event is not None and config.get("auto_apply_instruction") and low_level_instruction:
                event["applied_instruction"] = updated_runtime_config.get("instruction", "")

    if parsed:
        updated_runtime_config = persist_upper_agent_decision(
            runtime_config_path,
            parsed,
            auto_apply_instruction=config.get("auto_apply_instruction"),
            request_low_level_hard_reset=request_low_level_hard_reset or demo_command_changed,
            hard_reset_reason=(
                "demo_agent_step_changed"
                if demo_command_changed
                else "low_level_stop_before_upper_completion"
            ),
        )
        if event is not None and config.get("auto_apply_instruction") and parsed.get("current_subgoal"):
            event["applied_instruction"] = updated_runtime_config.get("instruction", "")

    event = append_event(run_dir, event)
    return {
        "ok": True,
        "skipped": False,
        "event": event,
        "low_level_hard_reset_requested": bool(
            (request_low_level_hard_reset or demo_command_changed)
            and parsed.get("task_status") == "running"
            and config.get("auto_apply_instruction")
            and parsed.get("current_subgoal")
        ),
    }
