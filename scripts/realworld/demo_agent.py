"""Deterministic instruction-library support for the Upper Agent."""

import json
import os
import re
import shlex
import threading
import uuid
from datetime import datetime
from pathlib import Path

from runtime_config import load_runtime_config, save_runtime_config


STORE_VERSION = 1
MAX_LIBRARIES = 200
MAX_COMMANDS = 64
MAX_COMMAND_LENGTH = 500
_store_lock = threading.RLock()

ATOMIC_COMMANDS = {
    "前进": "Go straight.",
    "直行": "Go straight.",
    "向前": "Go straight.",
    "左转": "Turn left.",
    "向左": "Turn left.",
    "右转": "Turn right.",
    "向右": "Turn right.",
    "掉头": "Turn around.",
    "停止": "Stop.",
    "停下": "Stop.",
    "等待": "Wait.",
    "forward": "Go straight.",
    "straight": "Go straight.",
    "left": "Turn left.",
    "right": "Turn right.",
    "around": "Turn around.",
    "stop": "Stop.",
    "wait": "Wait.",
}

ATOMIC_PHRASES = {
    "go straight": "Go straight.",
    "continue straight": "Continue straight.",
    "turn left": "Turn left.",
    "turn right": "Turn right.",
    "turn around": "Turn around.",
    "stop": "Stop.",
    "wait": "Wait.",
}


def utc_now():
    return datetime.now().isoformat(timespec="seconds")


def default_demo_library_path(runtime_config_path):
    return Path(runtime_config_path).expanduser().resolve().parent / "demo_agent_libraries.json"


def parse_navigation_steps(text):
    """Parse a compact library while preserving multi-word commands."""
    text = str(text or "").strip()
    if not text:
        return []

    if re.search(r"[;；,，\n\r]", text):
        candidates = re.split(r"\s*[;；,，\n\r]+\s*", text)
    elif '"' in text or "'" in text:
        try:
            candidates = shlex.split(text)
        except ValueError as exc:
            raise ValueError(f"指令引号没有闭合：{exc}") from exc
    else:
        phrase_pattern = re.compile(
            r"\b(?:continue\s+straight|go\s+straight|turn\s+left|turn\s+right|turn\s+around|stop|wait)\b",
            flags=re.IGNORECASE,
        )
        phrase_matches = list(phrase_pattern.finditer(text))
        phrase_remainder = phrase_pattern.sub("", text)
        if len(phrase_matches) > 1 and not phrase_remainder.strip():
            return [ATOMIC_PHRASES[re.sub(r"\s+", " ", match.group(0).casefold())] for match in phrase_matches]
        if re.search(r"\s{2,}", text):
            candidates = re.split(r"\s{2,}", text)
        else:
            words = text.split()
            if len(words) > 1 and all(word.strip().casefold() in ATOMIC_COMMANDS for word in words):
                candidates = words
            else:
                candidates = [text]

    commands = []
    for candidate in candidates:
        command = re.sub(r"^\s*(?:step\s*)?\d+[.)、:]\s*", "", str(candidate), flags=re.IGNORECASE)
        command = re.sub(r"\s+", " ", command).strip()
        if not command:
            continue
        command = ATOMIC_COMMANDS.get(command.casefold(), command)
        if len(command) > MAX_COMMAND_LENGTH:
            raise ValueError(f"单条导航指令不能超过 {MAX_COMMAND_LENGTH} 个字符。")
        commands.append(command)
    if len(commands) > MAX_COMMANDS:
        raise ValueError(f"一个指令库最多包含 {MAX_COMMANDS} 条指令。")
    return commands


def _empty_store():
    return {"version": STORE_VERSION, "libraries": [], "updated_at": ""}


def load_demo_libraries(path):
    path = Path(path).expanduser().resolve()
    with _store_lock:
        if not path.exists():
            return _empty_store()
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return _empty_store()
        if not isinstance(data, dict) or not isinstance(data.get("libraries"), list):
            return _empty_store()
        libraries = [item for item in data["libraries"] if isinstance(item, dict)]
        return {"version": STORE_VERSION, "libraries": libraries[-MAX_LIBRARIES:], "updated_at": data.get("updated_at", "")}


def _save_store(path, store):
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    store = {
        "version": STORE_VERSION,
        "libraries": list(store.get("libraries") or [])[-MAX_LIBRARIES:],
        "updated_at": utc_now(),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(store, indent=2, ensure_ascii=False))
    os.replace(temporary, path)
    return store


def normalize_library(payload, existing=None):
    payload = payload if isinstance(payload, dict) else {}
    existing = existing if isinstance(existing, dict) else {}
    name = str(payload.get("name") or existing.get("name") or "").strip()
    scene = str(payload.get("scene") or existing.get("scene") or "").strip()
    notes = str(payload.get("notes") or existing.get("notes") or "").strip()
    raw_commands = payload.get("commands_text")
    if raw_commands is None:
        raw_commands = payload.get("commands", existing.get("commands", []))
    if isinstance(raw_commands, list):
        commands = parse_navigation_steps("\n".join(str(item) for item in raw_commands))
    else:
        commands = parse_navigation_steps(raw_commands)
    if not name:
        raise ValueError("指令库名称不能为空。")
    if not scene:
        raise ValueError("场景名称不能为空。")
    if not commands:
        raise ValueError("至少需要一条导航指令。")
    return {
        "id": str(existing.get("id") or payload.get("id") or uuid.uuid4().hex),
        "name": name[:120],
        "scene": scene[:120],
        "notes": notes[:2000],
        "commands": commands,
        "created_at": str(existing.get("created_at") or utc_now()),
        "updated_at": utc_now(),
    }


def upsert_demo_library(path, payload):
    with _store_lock:
        store = load_demo_libraries(path)
        library_id = str((payload or {}).get("id") or "").strip()
        existing = next((item for item in store["libraries"] if str(item.get("id")) == library_id), None)
        library = normalize_library(payload, existing=existing)
        libraries = [item for item in store["libraries"] if str(item.get("id")) != library["id"]]
        libraries.append(library)
        store["libraries"] = libraries
        _save_store(path, store)
        return library


def get_demo_library(path, library_id):
    store = load_demo_libraries(path)
    library = next((item for item in store["libraries"] if str(item.get("id")) == str(library_id)), None)
    if library is None:
        raise KeyError(f"Demo Agent 指令库不存在：{library_id}")
    return library


def delete_demo_library(path, library_id, runtime_config_path=None):
    with _store_lock:
        if runtime_config_path:
            runtime = load_runtime_config(runtime_config_path)
            state = ((runtime.get("upper_agent") or {}).get("demo_agent") or {})
            if state.get("enabled") and str(state.get("library_id")) == str(library_id):
                raise ValueError("当前指令库正在运行，请先停止 Demo Agent。")
        store = load_demo_libraries(path)
        before = len(store["libraries"])
        store["libraries"] = [item for item in store["libraries"] if str(item.get("id")) != str(library_id)]
        if len(store["libraries"]) == before:
            raise KeyError(f"Demo Agent 指令库不存在：{library_id}")
        _save_store(path, store)


def get_demo_state(runtime_config):
    upper = runtime_config.get("upper_agent") if isinstance(runtime_config.get("upper_agent"), dict) else {}
    state = upper.get("demo_agent") if isinstance(upper.get("demo_agent"), dict) else {}
    return dict(state)


def activate_demo_library(runtime_config_path, library):
    current = load_runtime_config(runtime_config_path)
    upper = dict(current.get("upper_agent") or {})
    first_command = str((library.get("commands") or [""])[0]).strip()
    activated_at = utc_now()
    state = {
        "enabled": True,
        "library_id": library["id"],
        "library_name": library["name"],
        "scene": library["scene"],
        "notes": library.get("notes", ""),
        "commands": list(library["commands"]),
        "current_step_index": 0,
        "current_command": first_command,
        "status": "running",
        "execution_attempt": 1,
        "started_at": activated_at,
        "updated_at": activated_at,
        "last_transition_reason": "library_activated",
    }
    upper.update(
        {
            "enabled": True,
            "auto_apply_instruction": True,
            "task_instruction": f"Execute Demo Agent library '{library['name']}' for scene '{library['scene']}' in order.",
            "last_task_status": "running",
            "last_subgoal": first_command,
            "last_decision_at": activated_at,
            "replan_requested": False,
            "hard_reset_requested": True,
            "hard_reset_reason": "demo_agent_library_activated",
            "hard_reset_requested_at": activated_at,
            "hard_reset_subgoal": first_command,
            "voice_stop_active": False,
            "demo_agent": state,
        }
    )
    for key in ("last_task_feedback", "last_task_report_at", "last_error"):
        upper.pop(key, None)
    current["upper_agent"] = upper
    current["instruction"] = first_command
    current["service_enabled"] = True
    current.pop("_upper_agent_pause", None)
    return save_runtime_config(runtime_config_path, current)


def control_demo_agent(runtime_config_path, action):
    current = load_runtime_config(runtime_config_path)
    upper = dict(current.get("upper_agent") or {})
    state = dict(upper.get("demo_agent") or {})
    commands = list(state.get("commands") or [])
    if action not in {"pause", "resume", "reset", "stop"}:
        raise ValueError("不支持的 Demo Agent 控制操作。")
    if not state:
        raise ValueError("尚未启动 Demo Agent 指令库。")

    if action == "pause":
        state["status"] = "paused"
        current["service_enabled"] = False
    elif action == "stop":
        state["enabled"] = False
        state["status"] = "stopped"
        current["instruction"] = ""
        current["service_enabled"] = False
    elif action == "reset":
        state.update({"enabled": True, "status": "starting", "current_step_index": 0, "current_command": ""})
        upper["last_task_status"] = "running"
        upper["last_subgoal"] = ""
        upper["replan_requested"] = True
        current["instruction"] = ""
        current["service_enabled"] = True
    elif action == "resume":
        index = max(0, min(int(state.get("current_step_index") or 0), max(0, len(commands) - 1)))
        command = commands[index] if commands else ""
        state.update({"enabled": True, "status": "running", "current_step_index": index, "current_command": command})
        upper["last_task_status"] = "running"
        upper["last_subgoal"] = command
        upper["replan_requested"] = False
        if command:
            upper["hard_reset_requested"] = True
            upper["hard_reset_reason"] = "demo_agent_resumed"
            upper["hard_reset_requested_at"] = utc_now()
            upper["hard_reset_subgoal"] = command
        current["instruction"] = command
        current["service_enabled"] = True

    state["updated_at"] = utc_now()
    state["last_transition_reason"] = action
    upper["demo_agent"] = state
    current["upper_agent"] = upper
    current.pop("_upper_agent_pause", None)
    return save_runtime_config(runtime_config_path, current)


def demo_prompt_context(runtime_config):
    state = get_demo_state(runtime_config)
    if not state.get("enabled") or state.get("status") in {"paused", "stopped", "completed", "failed"}:
        return None
    commands = list(state.get("commands") or [])
    if not commands:
        return None
    index = max(0, min(int(state.get("current_step_index") or 0), len(commands) - 1))
    return {
        "library_id": state.get("library_id"),
        "library_name": state.get("library_name"),
        "scene": state.get("scene"),
        "notes": state.get("notes"),
        "commands": commands,
        "current_step_index": index,
        "current_command": commands[index],
        "next_step_index": index + 1 if index + 1 < len(commands) else None,
        "next_command": commands[index + 1] if index + 1 < len(commands) else "",
        "is_final_step": index == len(commands) - 1,
        "status": state.get("status"),
        "execution_attempt": int(state.get("execution_attempt") or 0),
        "attempt_started_frame_idx_hint": state.get("attempt_started_frame_idx_hint"),
        "step_started_run_name": state.get("step_started_run_name", ""),
        "step_started_frame_idx": state.get("step_started_frame_idx"),
        "step_started_image_file": state.get("step_started_image_file", ""),
    }


def ensure_demo_step_reference(runtime_config_path, run_name, frame_idx, image_file):
    """Record the first observed frame for the active Demo Agent step."""
    current = load_runtime_config(runtime_config_path)
    upper = dict(current.get("upper_agent") or {})
    state = dict(upper.get("demo_agent") or {})
    if not state.get("enabled") or state.get("status") in {"paused", "stopped", "completed", "failed"}:
        return None

    commands = list(state.get("commands") or [])
    if not commands:
        return None
    step_index = max(0, min(int(state.get("current_step_index") or 0), len(commands) - 1))
    reference_matches_step = (
        state.get("step_started_step_index") == step_index
        and state.get("step_started_run_name") == str(run_name or "")
        and state.get("step_started_image_file")
    )
    if not reference_matches_step:
        state.update(
            {
                "step_started_step_index": step_index,
                "step_started_run_name": str(run_name or ""),
                "step_started_frame_idx": frame_idx,
                "step_started_image_file": str(image_file or ""),
            }
        )
        upper["demo_agent"] = state
        current["upper_agent"] = upper
        save_runtime_config(runtime_config_path, current)
    return {
        "step_index": step_index,
        "run_name": state.get("step_started_run_name", ""),
        "frame_idx": state.get("step_started_frame_idx"),
        "image_file": state.get("step_started_image_file", ""),
    }


def constrain_demo_agent_output(
    runtime_config_path,
    output,
    run_name="",
    frame_idx=None,
    image_file="",
    execution_evidence=None,
):
    """Force model decisions onto the active ordered instruction library."""
    current = load_runtime_config(runtime_config_path)
    upper = dict(current.get("upper_agent") or {})
    state = dict(upper.get("demo_agent") or {})
    if not state.get("enabled") or state.get("status") in {"paused", "stopped", "completed", "failed"}:
        return output, False
    commands = list(state.get("commands") or [])
    if not commands:
        return output, False

    output = dict(output)
    old_index = max(0, min(int(state.get("current_step_index") or 0), len(commands) - 1))
    old_status = str(state.get("status") or "starting")
    decision = str(output.get("demo_step_decision") or "hold").strip().lower()
    model_status = str(output.get("task_status") or "running").strip().lower()
    required_turn = str((execution_evidence or {}).get("required_turn") or "none")
    has_required_turn_output = bool((execution_evidence or {}).get("has_required_turn_output", True))
    assessment = output.get("execution_assessment")
    assessment = assessment if isinstance(assessment, dict) else {}
    try:
        completion_confidence = float(assessment.get("completion_confidence") or 0.0)
    except (TypeError, ValueError):
        completion_confidence = 0.0
    completion_threshold = float(
        (execution_evidence or {}).get("completion_confidence_threshold") or 0.75
    )
    transition_mode = str(
        (execution_evidence or {}).get("transition_mode") or "balanced"
    ).strip().lower()
    matching_turn_output_count = int(
        (execution_evidence or {}).get("matching_turn_output_count") or 0
    )
    matching_turn_segment_count = int(
        (execution_evidence or {}).get("matching_turn_segment_count") or 0
    )
    visual_change_score = float(
        (execution_evidence or {}).get("endpoint_visual_change_score") or 0.0
    )
    observed_turn_direction = str(
        assessment.get("observed_turn_direction") or "uncertain"
    )
    model_supported_transition = bool(
        transition_mode == "balanced"
        and old_index < len(commands) - 1
        and required_turn in {"left", "right"}
        and has_required_turn_output
        and matching_turn_output_count >= 3
        and bool((execution_evidence or {}).get("clip_ended_with_stop"))
        and visual_change_score >= 0.12
        and bool(assessment.get("turn_started"))
        and observed_turn_direction == required_turn
        and completion_confidence + 1e-9 >= max(0.4, completion_threshold - 0.30)
    )
    strong_temporal_transition = bool(
        transition_mode == "balanced"
        and old_index < len(commands) - 1
        and required_turn in {"left", "right"}
        and has_required_turn_output
        and matching_turn_output_count >= 8
        and matching_turn_segment_count >= 2
        and bool((execution_evidence or {}).get("clip_ended_with_stop"))
        and visual_change_score >= 0.16
        and observed_turn_direction in {required_turn, "none", "uncertain"}
    )
    balanced_transition_supported = model_supported_transition or strong_temporal_transition
    if decision == "hold" and model_status == "running" and balanced_transition_supported:
        decision = "advance"
        output["demo_step_decision"] = "advance"
        assessment["subgoal_completed"] = True
        assessment["turn_completed"] = True
        assessment["reason"] = (
            f"Balanced transition: {matching_turn_output_count} matching {required_turn} outputs "
            f"across {matching_turn_segment_count} segments, endpoint visual change "
            f"{visual_change_score:.2f}, and no opposite-direction evidence."
        )
        output["execution_assessment"] = assessment
    wants_transition = decision in {"advance", "complete"} or model_status == "completed"
    assessment_gate_blocked = bool(
        execution_evidence
        and wants_transition
        and not balanced_transition_supported
        and (
            not bool(assessment.get("subgoal_completed"))
            or completion_confidence < completion_threshold
        )
    )
    turn_gate_blocked = (
        required_turn in {"left", "right"}
        and wants_transition
        and not balanced_transition_supported
        and (
            not has_required_turn_output
            or not bool(assessment.get("turn_started"))
            or not bool(assessment.get("turn_completed"))
            or str(assessment.get("observed_turn_direction") or "uncertain") != required_turn
        )
    )
    if turn_gate_blocked or assessment_gate_blocked:
        decision = "hold"
        model_status = "running"
        output["demo_step_decision"] = "hold"
        output["task_status"] = "running"
        evidence = str(output.get("visual_evidence") or "").strip()
        if turn_gate_blocked:
            gate_note = (
                f"Transition blocked: the current command requires a completed {required_turn} turn, "
                "but the execution clip does not confirm both turn start and visual turn completion."
            )
        else:
            gate_note = (
                "Transition blocked: execution_assessment did not confirm complete execution "
                f"at confidence >= {completion_threshold:.2f}."
            )
        output["visual_evidence"] = f"{evidence} {gate_note}".strip()

    if model_status == "failed" or decision == "failed":
        new_index = old_index
        new_status = "failed"
        output["task_status"] = "failed"
        output["current_subgoal"] = ""
    elif (model_status == "completed" or decision == "complete") and old_index == len(commands) - 1:
        new_index = old_index
        new_status = "completed"
        output["task_status"] = "completed"
        output["current_subgoal"] = ""
    else:
        if old_status == "starting":
            new_index = old_index
        elif decision == "advance":
            new_index = min(old_index + 1, len(commands) - 1)
        else:
            new_index = old_index
        new_status = "running"
        output["task_status"] = "running"
        output["current_subgoal"] = commands[new_index]

    command_changed = old_status == "starting" or new_index != old_index or state.get("current_command") != output.get("current_subgoal")
    state.update(
        {
            "enabled": new_status not in {"completed", "failed"},
            "current_step_index": new_index,
            "current_command": output.get("current_subgoal", ""),
            "status": new_status,
            "updated_at": utc_now(),
            "last_transition_reason": decision,
            "last_visual_evidence": str(output.get("visual_evidence") or "")[:1000],
        }
    )
    if new_index != old_index:
        # The frame that closes one step is also the visual baseline for the
        # next step. Future decisions can compare viewpoint and route progress.
        state.update(
            {
                "step_started_step_index": new_index,
                "step_started_run_name": str(run_name or ""),
                "step_started_frame_idx": frame_idx,
                "step_started_image_file": str(image_file or ""),
            }
        )
    upper["demo_agent"] = state
    current["upper_agent"] = upper
    save_runtime_config(runtime_config_path, current)
    output["demo_agent"] = {
        "library_id": state.get("library_id"),
        "library_name": state.get("library_name"),
        "scene": state.get("scene"),
        "step_index": new_index,
        "step_number": new_index + 1,
        "total_steps": len(commands),
        "decision": decision,
        "status": new_status,
        "command": output.get("current_subgoal", ""),
        "turn_transition_gate_blocked": turn_gate_blocked,
        "completion_transition_gate_blocked": assessment_gate_blocked,
        "balanced_transition_supported": balanced_transition_supported,
        "strong_temporal_transition": strong_temporal_transition,
        "transition_mode": transition_mode,
    }
    return output, command_changed
