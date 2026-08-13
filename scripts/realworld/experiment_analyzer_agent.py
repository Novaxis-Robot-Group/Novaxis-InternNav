"""Offline experiment indexing and evidence-grounded question answering.

This module is deliberately isolated from the online navigation stack. It reads
one completed (or still growing) experiment directory and writes only analysis
artifacts inside that directory. It never changes runtime configuration, reset
tokens, policy state, or robot commands.
"""

import base64
import json
import math
import os
import re
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw

from local_qwen_service import is_local_qwen_model, resolve_api_key, resolve_api_url, resolve_model_name

from experiment_instance_analyzer import instance_evidence, load_instance_index


INDEX_FILENAME = "experiment_analysis_index.json"
HISTORY_FILENAME = "experiment_qa_history.jsonl"
ANALYZER_VERSION = 2
ACTION_LABELS = {0: "STOP", 1: "FWD", 2: "LEFT", 3: "RIGHT", 5: "LOOK_DOWN"}


ANALYZER_SYSTEM_PROMPT = """
You are an offline robot experiment analysis agent. You analyze one recorded
experiment; you do not control the robot. Answer only from the supplied ordered
keyframes and structured logs. Never invent an object, room, event, time, frame,
or unique-object count.

Return only valid JSON with this schema:
{
  "answer": "concise answer to the user question",
  "confidence": 0.0,
  "uncertainty": "what cannot be confirmed, or empty string",
  "time_ranges": [{"start_frame": 0, "end_frame": 0, "reason": "why relevant"}],
  "evidence": [{"frame_idx": 0, "reason": "what this frame proves"}],
  "data_sources": ["RGB keyframes", "action log"]
}

Rules:
- A frame proves only what is visible in that frame. Logs prove only recorded
  robot state or model output.
- Use multiple ordered frames to reason about motion and repeated sightings.
- Do not treat repeated views as different physical objects. If tracking or
  identity evidence is insufficient, give a range or state that an exact unique
  count cannot be confirmed.
- For anomaly or pause questions, use STOP/replan events and nearby frames.
- Cite only frame_idx values supplied in the evidence bundle.
- Keep the response compact: answer and uncertainty must each be at most 60
  Chinese characters (or 35 English words); include at most 3 time_ranges and
  3 evidence entries, with each reason at most 30 Chinese characters (or 18
  English words).
- Keep confidence between 0 and 1. Do not include markdown.
""".strip()


def _atomic_write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    os.replace(temporary, path)


def _frame_idx(path):
    match = re.search(r"frame_(\d+)", Path(path).name)
    return int(match.group(1)) if match else -1


def _load_json(path, default=None):
    try:
        value = json.loads(Path(path).read_text())
        return value
    except (OSError, json.JSONDecodeError):
        return {} if default is None else default


def _response_action(response):
    response = response if isinstance(response, dict) else {}
    actions = response.get("discrete_action")
    if isinstance(actions, list) and actions:
        labels = [ACTION_LABELS.get(int(item), str(item)) for item in actions]
        return {
            "kind": "discrete",
            "labels": labels,
            "primary": labels[0],
            "is_stop": all(label == "STOP" for label in labels),
        }

    trajectory = response.get("trajectory")
    if not isinstance(trajectory, list) or not trajectory:
        return {"kind": "none", "labels": [], "primary": "UNKNOWN", "is_stop": False}
    points = []
    for point in trajectory:
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            try:
                points.append((float(point[0]), float(point[1])))
            except (TypeError, ValueError):
                continue
    if not points:
        return {"kind": "trajectory", "labels": [], "primary": "UNKNOWN", "is_stop": False}
    endpoint_x, endpoint_y = points[-1]
    path_length = sum(
        math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1])
        for i in range(1, len(points))
    )
    if path_length < 0.05:
        primary = "STOP"
    elif endpoint_y > 0.15:
        primary = "LEFT"
    elif endpoint_y < -0.15:
        primary = "RIGHT"
    else:
        primary = "FWD"
    return {
        "kind": "trajectory",
        "labels": [primary],
        "primary": primary,
        "is_stop": primary == "STOP",
        "point_count": len(points),
        "endpoint": [endpoint_x, endpoint_y],
        "path_length": round(path_length, 4),
    }


def _visual_signature(image_path):
    try:
        image = Image.open(image_path).convert("L").resize((32, 24))
        return np.asarray(image, dtype=np.float32) / 255.0
    except (OSError, ValueError):
        return None


def _scene_difference(previous, current):
    if previous is None or current is None:
        return 1.0
    return float(np.mean(np.abs(previous - current)))


def _timeline_entry(run_dir, metadata_path):
    metadata = _load_json(metadata_path, {})
    frame_idx = int(metadata.get("frame_idx", _frame_idx(metadata_path)))
    response = metadata.get("response") if isinstance(metadata.get("response"), dict) else {}
    runtime = metadata.get("runtime_config") if isinstance(metadata.get("runtime_config"), dict) else {}
    upper = runtime.get("upper_agent") if isinstance(runtime.get("upper_agent"), dict) else {}
    action = _response_action(response)
    return {
        "frame_idx": frame_idx,
        "saved_at": str(metadata.get("saved_at") or ""),
        "instruction": str(metadata.get("instruction") or ""),
        "agent_task_instruction": str(
            metadata.get("agent_task_instruction") or upper.get("task_instruction") or ""
        ),
        "action": action,
        "service_state": str(response.get("service_state") or ""),
        "replan_required": bool(response.get("replan_required", False)),
        "pixel_goal": response.get("pixel_goal"),
        "rgb_file": str(metadata.get("rgb_file") or f"frame_{frame_idx:06d}_rgb.jpg"),
        "depth_file": str(metadata.get("depth_file") or f"frame_{frame_idx:06d}_depth.png"),
        "vis_file": str(metadata.get("vis_file") or f"frame_{frame_idx:06d}_vis.jpg"),
    }


def _load_upper_events(run_dir):
    events = []
    path = Path(run_dir) / "upper_agent_events.jsonl"
    if not path.exists():
        return events
    for line in path.read_text().splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        output = event.get("output") if isinstance(event.get("output"), dict) else {}
        memory = output.get("memory") if isinstance(output.get("memory"), dict) else {}
        events.append(
            {
                "frame_idx": event.get("frame_idx"),
                "task_status": output.get("task_status"),
                "navigation_phase": output.get("navigation_phase"),
                "current_subgoal": output.get("current_subgoal"),
                "visual_evidence": output.get("visual_evidence"),
                "current_place": memory.get("current_place"),
                "landmarks_seen": memory.get("landmarks_seen") or [],
            }
        )
    return events


def build_experiment_index(run_dir, max_keyframes=120, sample_interval=15, scene_threshold=0.12):
    """Build a compact, deterministic scene/action index for one experiment."""
    run_dir = Path(run_dir).resolve()
    metadata_paths = sorted(run_dir.glob("frame_*_waypoint.json"), key=_frame_idx)
    if not metadata_paths:
        raise ValueError("This experiment has no waypoint metadata to index.")

    timeline = [_timeline_entry(run_dir, path) for path in metadata_paths]
    selected = set()
    reasons = {}

    def select(position, reason):
        if 0 <= position < len(timeline):
            selected.add(position)
            reasons.setdefault(position, []).append(reason)

    select(0, "experiment start")
    select(len(timeline) - 1, "experiment end/latest frame")
    previous_signature = None
    previous_instruction = None
    previous_action = None
    stop_runs = []
    active_stop_start = None

    for position, entry in enumerate(timeline):
        if position % max(1, int(sample_interval)) == 0:
            select(position, "periodic coverage")
        instruction = entry["instruction"]
        action = entry["action"].get("primary")
        if previous_instruction is not None and instruction != previous_instruction:
            select(position, "instruction changed")
            select(position - 1, "frame before instruction change")
        if previous_action is not None and action != previous_action:
            select(position, "action changed")
        if entry["replan_required"] or entry["service_state"]:
            select(position, "replan or service-state event")

        rgb_path = run_dir / entry["rgb_file"]
        signature = _visual_signature(rgb_path)
        if position > 0 and _scene_difference(previous_signature, signature) >= float(scene_threshold):
            select(position, "visual scene changed")
        if signature is not None:
            previous_signature = signature

        if entry["action"].get("is_stop"):
            if active_stop_start is None:
                active_stop_start = position
        elif active_stop_start is not None:
            stop_runs.append((active_stop_start, position - 1))
            select(active_stop_start, "STOP sequence started")
            select(position - 1, "STOP sequence ended")
            active_stop_start = None
        previous_instruction = instruction
        previous_action = action

    if active_stop_start is not None:
        stop_runs.append((active_stop_start, len(timeline) - 1))
        select(active_stop_start, "STOP sequence started")
        select(len(timeline) - 1, "STOP sequence active at end")

    if len(selected) > max_keyframes:
        mandatory = {0, len(timeline) - 1}
        event_positions = sorted(
            position for position in selected
            if any("periodic" not in reason and "visual scene" not in reason for reason in reasons.get(position, []))
        )
        keep = sorted(mandatory)
        event_budget = min(len(event_positions), max(0, max_keyframes // 2 - len(keep)))
        if event_budget:
            event_indexes = np.linspace(0, len(event_positions) - 1, event_budget, dtype=int)
            keep.extend(event_positions[int(index)] for index in event_indexes)
        keep = list(dict.fromkeys(keep))
        remaining = max_keyframes - len(keep)
        candidates = [position for position in sorted(selected) if position not in keep]
        if remaining > 0 and candidates:
            indexes = np.linspace(0, len(candidates) - 1, min(remaining, len(candidates)), dtype=int)
            keep.extend(candidates[int(index)] for index in indexes)
        selected = set(keep)

    upper_events = _load_upper_events(run_dir)
    upper_by_frame = {event.get("frame_idx"): event for event in upper_events}
    keyframes = []
    for position in sorted(selected):
        entry = dict(timeline[position])
        entry["selection_reasons"] = reasons.get(position, [])
        if entry["frame_idx"] in upper_by_frame:
            entry["upper_agent"] = upper_by_frame[entry["frame_idx"]]
        keyframes.append(entry)

    stop_events = [
        {
            "start_frame": timeline[start]["frame_idx"],
            "end_frame": timeline[end]["frame_idx"],
            "saved_frame_count": end - start + 1,
        }
        for start, end in stop_runs
    ]
    latest_mtime_ns = max(path.stat().st_mtime_ns for path in metadata_paths)
    index = {
        "analyzer_version": ANALYZER_VERSION,
        "backend": "lightweight_scene_event_index",
        "run_name": run_dir.name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": {"frame_count": len(timeline), "latest_metadata_mtime_ns": latest_mtime_ns},
        "summary": {
            "keyframe_count": len(keyframes),
            "upper_agent_event_count": len(upper_events),
            "stop_sequence_count": len(stop_events),
            "instructions": list(dict.fromkeys(item["instruction"] for item in timeline if item["instruction"])),
        },
        "keyframes": keyframes,
        "timeline": timeline,
        "stop_events": stop_events,
        "upper_agent_events": upper_events,
        "capabilities": {
            "scene_keyframes": True,
            "action_timeline": True,
            "depth_references": True,
            "object_detector": False,
            "object_tracking": False,
            "semantic_room_map": False,
        },
    }
    _atomic_write_json(run_dir / INDEX_FILENAME, index)
    return index


def load_experiment_index(run_dir, rebuild_if_stale=True):
    run_dir = Path(run_dir).resolve()
    path = run_dir / INDEX_FILENAME
    index = _load_json(path, None) if path.exists() else None
    metadata_paths = sorted(run_dir.glob("frame_*_waypoint.json"), key=_frame_idx)
    if not metadata_paths:
        raise ValueError("This experiment has no waypoint metadata to analyze.")
    stale = not isinstance(index, dict)
    if isinstance(index, dict):
        source = index.get("source") if isinstance(index.get("source"), dict) else {}
        stale = index.get("analyzer_version") != ANALYZER_VERSION
        stale = stale or source.get("frame_count") != len(metadata_paths)
        stale = stale or source.get("latest_metadata_mtime_ns") != max(path.stat().st_mtime_ns for path in metadata_paths)
    if stale and rebuild_if_stale:
        return build_experiment_index(run_dir)
    return index


def _question_tokens(question):
    return set(re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]{1,4}", str(question).casefold()))


def is_visual_question(question):
    question_lower = str(question).casefold()
    return any(
        word in question_lower
        for word in (
            "person", "people", "sofa", "box", "plant", "potted plant", "room", "object", "visible", "see", "saw",
            "人", "沙发", "纸箱", "绿植", "植物", "盆栽", "会议室", "房间", "物体", "目标", "看到", "几个", "是否有",
        )
    )


def is_overview_question(question):
    """Recognize questions that require a whole-run report, not one event."""
    lower = str(question or "").casefold()
    return any(
        term in lower
        for term in (
            "overall", "summary", "report", "whole experiment", "entire experiment",
            "整体", "全程", "总结", "简报", "汇报", "报告", "复盘", "概况",
            "情况如何", "巡检情况", "过程怎么样",
        )
    )


def is_anomaly_question(question):
    lower = str(question or "").casefold()
    return any(
        term in lower
        for term in (
            "anomaly", "abnormal", "failure", "stuck", "stop", "pause",
            "异常", "故障", "卡住", "停顿", "停止", "问题",
        )
    )


def _report_detector_summary(object_instance_index, frame_count):
    """Condense the full-frame object index into bounded report facts."""
    if not isinstance(object_instance_index, dict):
        return {"available": False, "objects": []}
    summary = object_instance_index.get("detection_summary") or {}
    minimum_frames = max(5, int(frame_count) // 100)
    objects = []
    for category, item in summary.items():
        if not isinstance(item, dict):
            continue
        detected = int(item.get("detected_frame_count", 0))
        if detected < minimum_frames:
            continue
        windows = []
        for window in (item.get("appearance_windows") or [])[:2]:
            windows.append(
                {
                    key: window.get(key)
                    for key in ("start_frame", "end_frame", "best_frame_idx", "max_simultaneous")
                }
            )
        objects.append(
            {
                "category": str(category),
                "detected_frame_count": detected,
                "max_simultaneous": int(item.get("max_simultaneous", 0)),
                "appearance_windows": windows,
            }
        )
    objects.sort(
        key=lambda item: (item["detected_frame_count"], item["max_simultaneous"]), reverse=True
    )
    # Preserve semantically useful patrol findings even if a common furniture
    # class has more raw detections than them.
    priority_categories = {"person", "potted plant", "couch", "chair", "backpack", "box"}
    priority = [item for item in objects if item["category"].casefold() in priority_categories]
    remaining = [item for item in objects if item not in priority]
    capabilities = object_instance_index.get("capabilities") or {}
    return {
        "available": True,
        "cross_loop_unique_count": bool(capabilities.get("cross_loop_unique_count", False)),
        "objects": (priority + remaining)[:8],
    }


def _experiment_overview(index, object_instance_index=None):
    """Return bounded global facts for an offline whole-experiment report."""
    timeline = index.get("timeline") if isinstance(index, dict) else []
    timeline = timeline if isinstance(timeline, list) else []
    action_counts = {}
    replan_count = 0
    for entry in timeline:
        action = entry.get("action") if isinstance(entry, dict) else {}
        label = str(action.get("primary") or "UNKNOWN") if isinstance(action, dict) else "UNKNOWN"
        action_counts[label] = action_counts.get(label, 0) + 1
        replan_count += int(bool(entry.get("replan_required"))) if isinstance(entry, dict) else 0
    summary = index.get("summary") if isinstance(index, dict) else {}
    summary = summary if isinstance(summary, dict) else {}
    overview = {
        "recorded_frame_count": len(timeline),
        "frame_range": [
            int(timeline[0].get("frame_idx", -1)) if timeline else -1,
            int(timeline[-1].get("frame_idx", -1)) if timeline else -1,
        ],
        "unique_low_level_instructions": summary.get("instructions", [])[:12],
        "action_frame_counts": action_counts,
        "stop_sequence_count": int(summary.get("stop_sequence_count", 0)),
        "replan_frame_count": replan_count,
        "upper_agent_event_count": int(summary.get("upper_agent_event_count", 0)),
    }
    overview["detector_observations"] = _report_detector_summary(
        object_instance_index, len(timeline)
    )
    return overview


def _inspection_report_context(overview):
    """Expose environmental findings, not policy/runtime diagnostics, to reports."""
    overview = overview if isinstance(overview, dict) else {}
    return {
        "report_mode": True,
        "recorded_frame_count": overview.get("recorded_frame_count", 0),
        "frame_range": overview.get("frame_range", []),
        "detector_observations": overview.get("detector_observations", {}),
    }


def retrieve_evidence(index, question, limit=8):
    """Rank indexed keyframes using logs first, then preserve temporal diversity."""
    question_lower = str(question).casefold()
    tokens = _question_tokens(question)
    stop_query = any(word in question_lower for word in ("stop", "pause", "stuck", "停", "卡", "异常"))
    motion_query = any(word in question_lower for word in ("turn", "route", "trajectory", "转弯", "轨迹", "经过"))
    visual_query = is_visual_question(question)
    overview_query = is_overview_question(question)
    anomaly_query = is_anomaly_question(question)
    scored = []
    keyframes = index.get("keyframes") or []
    for position, entry in enumerate(keyframes):
        upper = entry.get("upper_agent") if isinstance(entry.get("upper_agent"), dict) else {}
        text = " ".join(
            [
                str(entry.get("instruction") or ""),
                str(entry.get("agent_task_instruction") or ""),
                str(entry.get("action") or ""),
                str(entry.get("selection_reasons") or ""),
                str(upper.get("visual_evidence") or ""),
                str(upper.get("current_place") or ""),
                str(upper.get("landmarks_seen") or ""),
            ]
        ).casefold()
        score = sum(2 for token in tokens if token and token in text)
        if stop_query and entry.get("action", {}).get("is_stop"):
            score += 8
        if stop_query and (entry.get("replan_required") or entry.get("service_state")):
            score += 6
        if motion_query and "action changed" in entry.get("selection_reasons", []):
            score += 5
        if visual_query and "visual scene changed" in entry.get("selection_reasons", []):
            score += 3
        score += 0.25 if position in {0, len(keyframes) - 1} else 0
        scored.append((score, position, entry))

    chosen = []
    ranked = sorted(scored, key=lambda item: (-item[0], item[1]))
    # A whole-run question must not inherit early-frame tie breaking. Select
    # start, middle and end evidence before adding event-specific frames.
    if overview_query and keyframes:
        coverage_count = min(len(keyframes), max(3, int(limit)))
        coverage_positions = np.linspace(0, len(keyframes) - 1, coverage_count, dtype=int)
        chosen.extend(keyframes[int(position)] for position in coverage_positions)
    if anomaly_query and keyframes:
        # An anomaly report needs temporally diverse event evidence. Selecting
        # only the earliest STOP/replan event makes a long run look like it
        # never progressed.
        event_entries = [
            entry
            for entry in keyframes
            if entry.get("action", {}).get("is_stop")
            or entry.get("replan_required")
            or entry.get("service_state")
            or "action changed" in (entry.get("selection_reasons") or [])
        ]
        source = event_entries or keyframes
        coverage_count = min(len(source), max(3, int(limit)))
        coverage_positions = np.linspace(0, len(source) - 1, coverage_count, dtype=int)
        chosen.extend(source[int(position)] for position in coverage_positions)
    # Object/room questions often have no matching text before VLM inspection.
    # Reserve half the budget for temporal coverage so late-run targets are not
    # hidden by equally scored keyframes from the beginning of the experiment.
    if visual_query and keyframes:
        coverage_count = min(len(keyframes), max(2, int(limit) // 2))
        coverage_positions = np.linspace(0, len(keyframes) - 1, coverage_count, dtype=int)
        chosen.extend(keyframes[int(position)] for position in coverage_positions)
    for _, _, entry in ranked:
        if len(chosen) >= max(1, int(limit)):
            break
        if entry in chosen:
            continue
        chosen.append(entry)
    return sorted(chosen, key=lambda entry: int(entry.get("frame_idx", -1)))


def _contact_sheet_data_url(run_dir, entries, columns=4, thumb_size=(224, 168)):
    """Pack labeled thumbnails into one overview image for coarse VLM search."""
    rows = max(1, math.ceil(len(entries) / columns))
    label_height = 24
    sheet = Image.new("RGB", (columns * thumb_size[0], rows * (thumb_size[1] + label_height)), (8, 15, 13))
    draw = ImageDraw.Draw(sheet)
    for position, entry in enumerate(entries):
        row, column = divmod(position, columns)
        x = column * thumb_size[0]
        y = row * (thumb_size[1] + label_height)
        image_path = Path(run_dir) / str(entry.get("rgb_file") or "")
        try:
            image = Image.open(image_path).convert("RGB")
            image.thumbnail(thumb_size)
            tile = Image.new("RGB", thumb_size, (0, 0, 0))
            tile.paste(image, ((thumb_size[0] - image.width) // 2, (thumb_size[1] - image.height) // 2))
            sheet.paste(tile, (x, y + label_height))
        except (OSError, ValueError):
            pass
        draw.rectangle((x, y, x + thumb_size[0] - 1, y + label_height - 1), fill=(10, 35, 28))
        draw.text((x + 7, y + 5), f"frame {entry.get('frame_idx')}", fill=(174, 255, 220))
    buffer = BytesIO()
    sheet.save(buffer, format="JPEG", quality=86)
    return f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"


def _coarse_visual_retrieve(config, question, index, run_dir, max_candidates=12):
    """Locate likely evidence on contact sheets before opening full frames."""
    keyframes = index.get("keyframes") or []
    if not keyframes:
        return [], 0.0
    api_key = resolve_api_key(config)
    api_url = resolve_api_url(config)
    if not api_key or not api_url:
        return [], 0.0

    content = [{
        "type": "text",
        "text": (
            f"Question: {question}\nScan every labeled thumbnail in every overview. "
            "Return all frames that may contain relevant visual evidence. Favor recall."
        ),
    }]
    batch_size = 20
    for offset in range(0, len(keyframes), batch_size):
        batch = keyframes[offset : offset + batch_size]
        content.append({
            "type": "text",
            "text": f"Overview {offset // batch_size + 1}; available frame labels: {[item.get('frame_idx') for item in batch]}",
        })
        content.append({
            "type": "image_url",
            "image_url": {"url": _contact_sheet_data_url(run_dir, batch)},
        })
    payload = {
        "model": resolve_model_name(config) or "qwen3-vl-flash",
        "messages": [
            {
                "role": "system",
                "content": (
                    "Retrieve visual evidence from robot experiment overviews. Return only JSON with schema "
                    '{"candidate_frames":[0],"reason":"short retrieval reason"}. '
                    "Use only visible frame labels and do not answer the original question."
                ),
            },
            {"role": "user", "content": content},
        ],
        "max_tokens": 256,
        "temperature": 0.0,
    }
    if is_local_qwen_model(config.get("model")):
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    started = time.time()
    response = requests.post(
        api_url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=90,
    )
    elapsed = time.time() - started
    if not response.ok:
        return [], elapsed
    try:
        output = _extract_json(response.json()["choices"][0]["message"]["content"])
    except (KeyError, TypeError, RuntimeError, ValueError):
        return [], elapsed
    by_frame = {int(entry.get("frame_idx")): entry for entry in keyframes}
    candidates = []
    for raw_frame in output.get("candidate_frames") or []:
        try:
            entry = by_frame.get(int(raw_frame))
        except (TypeError, ValueError):
            entry = None
        if entry is not None and entry not in candidates:
            candidates.append(entry)
        if len(candidates) >= max_candidates:
            break
    return candidates, elapsed


def _image_data_url(path, max_width=768):
    image = Image.open(path).convert("RGB")
    if max_width and image.width > max_width:
        height = max(1, int(image.height * max_width / image.width))
        image = image.resize((max_width, height))
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=82)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _extract_json(text):
    if isinstance(text, list):
        text = " ".join(
            str(item.get("text") or "") for item in text if isinstance(item, dict)
        )
    text = str(text or "").strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start() :])
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            continue
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    raise RuntimeError("The analysis model did not return valid JSON.")


def _repair_invalid_json(config, raw_text):
    """Repair a non-JSON model reply without resending images or experiment data."""
    raw_text = str(raw_text or "").strip()
    if not raw_text:
        raise RuntimeError("The analysis model returned an empty response.")
    api_key = resolve_api_key(config)
    api_url = resolve_api_url(config)
    payload = {
        "model": resolve_model_name(config) or "qwen3-vl-flash",
        "messages": [
            {
                "role": "system",
                "content": (
                    "Convert the supplied answer into one valid JSON object only. "
                    "Use this schema: {answer:string,confidence:number,uncertainty:string,"
                    "time_ranges:[{start_frame:number,end_frame:number,reason:string}],"
                    "evidence:[{frame_idx:number,reason:string}],data_sources:[string]}."
                ),
            },
            {"role": "user", "content": raw_text[:6000]},
        ],
        "max_tokens": 512,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    if is_local_qwen_model(config.get("model")):
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    response = requests.post(
        api_url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=45,
    )
    if not response.ok:
        detail = response.text.strip().replace("\n", " ")[:500]
        raise RuntimeError(f"Experiment QA JSON repair failed: {detail or response.reason}")
    return _extract_json(response.json()["choices"][0]["message"]["content"])


def _save_model_response(run_dir, raw_text, filename="experiment_qa_last_response.json"):
    """Keep a local debugging artifact; it is never sent to the browser."""
    _atomic_write_json(
        Path(run_dir) / filename,
        {"created_at": datetime.now().isoformat(timespec="seconds"), "raw_response": str(raw_text or "")},
    )


def _retry_with_context_budget(response, payload, api_url, headers, timeout):
    """Retry once with the output budget advertised by a vLLM context error."""
    if response.status_code != 400 or "maximum context length" not in response.text.lower():
        return response
    match = re.search(
        r"maximum context length is\s*(\d+).*?prompt contains at least\s*(\d+)\s*input tokens",
        response.text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return response
    context_limit, input_tokens = (int(value) for value in match.groups())
    # Multi-modal chat templates can add more tokens on a retry. Reserve 256
    # tokens so a boundary error cannot recur when image-token accounting shifts.
    safe_budget = max(128, min(int(payload.get("max_tokens") or 512), context_limit - input_tokens - 256))
    if safe_budget >= int(payload.get("max_tokens") or 512):
        return response
    payload["max_tokens"] = safe_budget
    return requests.post(api_url, headers=headers, json=payload, timeout=timeout)


def _default_evidence_budget(question, maximum):
    if is_overview_question(question) or is_anomaly_question(question):
        return min(maximum, 5)
    if is_visual_question(question):
        return min(maximum, 4)
    return min(maximum, 3)


def _plan_evidence_budget(config, question, overview, maximum):
    """Ask the analysis model how much temporal evidence this question needs."""
    maximum = max(2, min(6, int(maximum)))
    fallback = _default_evidence_budget(question, maximum)
    api_key = resolve_api_key(config)
    api_url = resolve_api_url(config)
    if not api_key or not api_url:
        return fallback, {"source": "fallback", "reason": "analysis API is unavailable"}, 0.0
    payload = {
        "model": resolve_model_name(config) or "qwen3-vl-flash",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a robot experiment evidence planner. Return only JSON: "
                    '{"evidence_frame_count":2,"reason":"short"}. Choose an integer from 2 to 6. '
                    "Choose more frames for whole-run reports, anomalies, repeated events, or counting; "
                    "choose fewer for a narrow object-presence question. Do not answer the question."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {"question": question, "experiment_summary": overview}, ensure_ascii=False
                ),
            },
        ],
        "max_tokens": 96,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    if is_local_qwen_model(config.get("model")):
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    started = time.time()
    try:
        response = requests.post(
            api_url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )
        if response.status_code == 400 and "response_format" in response.text.lower():
            payload.pop("response_format", None)
            response = requests.post(
                api_url,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=30,
            )
        if not response.ok:
            raise RuntimeError(response.text[:200])
        plan = _extract_json(response.json()["choices"][0]["message"]["content"])
        count = int(plan.get("evidence_frame_count", fallback))
        count = max(2, min(maximum, count))
        if is_overview_question(question):
            # A meaningful patrol report must cover more than start/middle/end.
            count = max(min(maximum, 5), count)
        return count, {"source": "model", "reason": str(plan.get("reason") or "")}, time.time() - started
    except (KeyError, TypeError, ValueError, RuntimeError, requests.RequestException):
        return fallback, {"source": "fallback", "reason": "planner response unavailable"}, time.time() - started


def _compact_detection_summary(summary, instances):
    """Keep only detector facts relevant to the instances retrieved for QA."""
    if not isinstance(summary, dict):
        return {}
    categories = {
        str(item.get("category") or "").casefold()
        for item in (instances or [])
        if isinstance(item, dict) and item.get("category")
    }
    compact = {}
    for category in sorted(categories):
        value = summary.get(category)
        if not isinstance(value, dict):
            continue
        compact[category] = {
            key: value.get(key)
            for key in ("detected_frame_count", "max_simultaneous", "appearance_window_count")
        }
    return compact


def _call_vlm(config, question, evidence, run_dir, instance_context=None, experiment_context=None):
    api_key = resolve_api_key(config)
    if not api_key:
        raise RuntimeError("Upper Agent API key is not configured; Experiment QA uses the same server-side credential.")
    api_url = resolve_api_url(config)
    if not api_url:
        raise RuntimeError("Experiment QA API URL is not configured.")

    compact_evidence = []
    content = [
        {
            "type": "text",
            "text": (
                f"User question: {question}\n"
                "The following evidence frames are ordered in experiment time. "
                "Structured metadata for them is included below."
            ),
        }
    ]
    if isinstance(experiment_context, dict):
        report_mode = bool(experiment_context.get("report_mode"))
        content.append(
            {
                "type": "text",
                "text": (
                    "Whole-experiment structured summary (not visual proof by itself):\n"
                    + json.dumps(experiment_context, ensure_ascii=False)
                    + (
                        "\nWrite an environmental inspection report only. The answer must contain these "
                        "concise labelled parts: 异常结论、巡检发现、综合评价、复核建议. Assess only "
                        "what was encountered in the environment: people, obstacles, blocked passages, "
                        "damage, spills, unsafe objects, or other visible anomalies. Routine furniture is "
                        "not an anomaly by itself. Do not mention model behavior, low-level instructions, "
                        "STOP/replan counts, navigation efficiency, waiting, or policy execution. "
                        "Include detector_observations only as candidates and require visual evidence before "
                        "calling anything an anomaly. Use all ordered evidence; do not describe only the "
                        "first frame."
                        if report_mode
                        else "\nFor an overall question, summarize the full run using this summary and the "
                        "ordered start/middle/end evidence. Do not describe only the first frame."
                    )
                ),
            }
        )
    if isinstance(instance_context, dict):
        capabilities = instance_context.get("capabilities") or {}
        matched_instances = instance_context.get("instances") or []
        detection_summary = _compact_detection_summary(
            instance_context.get("detection_summary"), matched_instances
        )
        compact_instances = []
        for item in matched_instances[:12]:
            compact_instances.append(
                {
                    key: item.get(key)
                    for key in (
                        "instance_id",
                        "category",
                        "first_seen_frame",
                        "last_seen_frame",
                        "visible_frame_count",
                        "confidence",
                    )
                }
            )
        content.append(
            {
                "type": "text",
                "text": (
                    "Full-frame local detector/tracker instance index:\n"
                    + json.dumps(
                        {
                            "capabilities": capabilities,
                            "detection_summary": detection_summary,
                            "matched_instance_count": len(matched_instances),
                            "instances": compact_instances,
                        },
                        ensure_ascii=False,
                    )
                    + "\nUse the supplied representative images to visually verify candidates. "
                    "max_simultaneous is a detector-supported lower-bound candidate count in one frame. "
                    "If cross_loop_unique_count is false, do not claim an exact physical-object count "
                    "when separate tracks may be repeated views of one object."
                ),
            }
        )
    qa_image_width = int(config.get("max_image_width", 768))
    if is_local_qwen_model(config.get("model")):
        # Keep total vision-token use bounded while allowing the evidence
        # planner to choose 2-6 frames for different questions.
        width_by_count = {2: 512, 3: 384, 4: 320, 5: 288, 6: 256}
        qa_image_width = min(qa_image_width, width_by_count.get(len(evidence), 256))
    for entry in evidence:
        compact = {
            key: entry.get(key)
            for key in ("frame_idx", "saved_at", "instruction", "agent_task_instruction", "action", "service_state", "replan_required")
        }
        compact["upper_agent"] = entry.get("upper_agent")
        compact_evidence.append(compact)
        rgb_path = Path(run_dir) / str(entry.get("rgb_file") or "")
        content.append({"type": "text", "text": f"Evidence frame_idx={entry.get('frame_idx')}: {json.dumps(compact, ensure_ascii=False)}"})
        if rgb_path.is_file():
            content.append({"type": "image_url", "image_url": {"url": _image_data_url(rgb_path, qa_image_width)}})

    payload = {
        "model": resolve_model_name(config) or "qwen3-vl-flash",
        "messages": [
            {"role": "system", "content": ANALYZER_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        # Experiment QA is offline and needs enough room for cited evidence;
        # do not inherit the smaller real-time control response budget.
        "max_tokens": 512 if is_local_qwen_model(config.get("model")) else min(
            1200, max(512, int(config.get("max_tokens", 512)))
        ),
        "temperature": min(0.3, max(0.0, float(config.get("temperature", 0.2)))),
        "response_format": {"type": "json_object"},
    }
    if is_local_qwen_model(config.get("model")):
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    started = time.time()
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    response = requests.post(api_url, headers=headers, json=payload, timeout=90)
    if response.status_code == 400 and "response_format" in response.text.lower():
        payload.pop("response_format", None)
        response = requests.post(api_url, headers=headers, json=payload, timeout=90)
    for _ in range(2):
        retried = _retry_with_context_budget(response, payload, api_url, headers, timeout=90)
        if retried is response:
            break
        response = retried
    elapsed = time.time() - started
    if not response.ok:
        detail = response.text.strip().replace("\n", " ")[:1000]
        raise RuntimeError(f"Experiment QA API HTTP {response.status_code}: {detail or response.reason}")
    data = response.json()
    raw_text = data["choices"][0]["message"]["content"]
    _save_model_response(run_dir, raw_text)
    try:
        output = _extract_json(raw_text)
    except RuntimeError:
        _save_model_response(run_dir, raw_text, "experiment_qa_last_invalid_response.json")
        repair_started = time.time()
        output = _repair_invalid_json(config, raw_text)
        elapsed += time.time() - repair_started
    return output, elapsed


def _normalize_answer(output, evidence):
    allowed = {int(entry["frame_idx"]): entry for entry in evidence}
    cited = []
    for item in output.get("evidence") or []:
        if not isinstance(item, dict):
            continue
        try:
            frame_idx = int(item.get("frame_idx"))
        except (TypeError, ValueError):
            continue
        if frame_idx not in allowed:
            continue
        source = allowed[frame_idx]
        cited.append(
            {
                "frame_idx": frame_idx,
                "reason": str(item.get("reason") or "Relevant visual evidence."),
                "rgb_file": source.get("rgb_file"),
                "vis_file": source.get("vis_file"),
                "depth_file": source.get("depth_file"),
                "saved_at": source.get("saved_at"),
            }
        )
    confidence = output.get("confidence", 0.0)
    try:
        confidence = max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "answer": str(output.get("answer") or "No supported answer was produced."),
        "confidence": confidence,
        "uncertainty": str(output.get("uncertainty") or ""),
        "time_ranges": output.get("time_ranges") if isinstance(output.get("time_ranges"), list) else [],
        "evidence": cited,
        "data_sources": [str(item) for item in (output.get("data_sources") or [])],
    }


def answer_question(run_dir, question, config, max_images=8, rebuild_index=False):
    run_dir = Path(run_dir).resolve()
    question = str(question or "").strip()
    if not question:
        raise ValueError("Question cannot be empty.")
    # The full-frame tracker has already scanned every RGB frame. The planner
    # can choose up to six representative frames within the local 8K budget.
    if is_local_qwen_model(config.get("model")):
        max_images = min(int(max_images), 6)
    visual_question = is_visual_question(question)
    overview_question = is_overview_question(question)
    anomaly_question = is_anomaly_question(question)
    # The visualizer ensures this cache exists before every question. Keep it
    # available for all QA types so detector capabilities are part of the
    # experiment context, while object-specific evidence remains opt-in.
    object_instance_index = load_instance_index(run_dir)
    # Probe the full-frame instance index before touching the legacy thumbnail
    # index. A detector hit already identifies the relevant original frames, so
    # rebuilding contact sheets would duplicate a potentially expensive pass.
    initial_tracked_evidence, _ = instance_evidence(
        object_instance_index if visual_question else None,
        question,
        limit=max_images,
    )
    use_instance_only = bool(visual_question and initial_tracked_evidence and not rebuild_index)
    if use_instance_only:
        index = None
    else:
        index = build_experiment_index(run_dir) if rebuild_index else load_experiment_index(run_dir)
    overview_context = _experiment_overview(index, object_instance_index)
    if overview_question:
        overview_context["report_mode"] = True
    evidence_count, evidence_plan, planning_elapsed = _plan_evidence_budget(
        config, question, overview_context, max_images
    )
    tracked_evidence, tracked_instances = instance_evidence(
        object_instance_index if visual_question else None, question, limit=evidence_count
    )
    # A full-frame detector hit is already the highest-recall retrieval stage.
    # Avoid rebuilding the legacy thumbnail/keyframe index for the same visual
    # question; on long runs that duplicate pass can read thousands of images.
    if visual_question and tracked_evidence and not rebuild_index:
        baseline_evidence = []
    else:
        baseline_evidence = retrieve_evidence(index, question, limit=evidence_count)
    coarse_evidence = []
    coarse_elapsed = 0.0
    if visual_question and not tracked_evidence:
        coarse_evidence, coarse_elapsed = _coarse_visual_retrieve(
            config, question, index, run_dir, max_candidates=evidence_count
        )
    evidence = []
    for entry in tracked_evidence + coarse_evidence + baseline_evidence:
        if entry not in evidence:
            evidence.append(entry)
        if len(evidence) >= evidence_count:
            break
    evidence.sort(key=lambda entry: int(entry.get("frame_idx", -1)))
    vlm_instance_context = None
    if isinstance(object_instance_index, dict):
        vlm_instance_context = {
            "capabilities": object_instance_index.get("capabilities") or {},
            "detection_summary": object_instance_index.get("detection_summary") or {},
            "instances": tracked_instances,
        }
    output, answer_elapsed = _call_vlm(
        config,
        question,
        evidence,
        run_dir,
        instance_context=vlm_instance_context,
        experiment_context=(
            _inspection_report_context(overview_context)
            if overview_question
            else overview_context
            if anomaly_question
            else None
        ),
    )
    elapsed = planning_elapsed + coarse_elapsed + answer_elapsed
    answer = _normalize_answer(output, evidence)
    if visual_question and not isinstance(object_instance_index, dict):
        limitation = (
            "Full-frame detector/tracker index is unavailable; keyframe retrieval cannot confirm "
            "that a briefly visible object was absent."
        )
        answer["uncertainty"] = " ".join(
            item for item in (answer.get("uncertainty", ""), limitation) if item
        )
        answer["confidence"] = min(float(answer.get("confidence", 0.0)), 0.65)
    record = {
        "question": question,
        **answer,
        "retrieved_frame_indexes": [item.get("frame_idx") for item in evidence],
        "search_strategy": (
            "full_frame_instance_index_then_vlm"
            if tracked_evidence
            else "full_frame_index_plus_event_timeline"
            if isinstance(object_instance_index, dict)
            else "contact_sheet_then_detail"
            if coarse_evidence
            else "structured_keyframe_retrieval"
        ),
        "instance_candidates": [item.get("instance_id") for item in tracked_instances],
        "instance_index_available": isinstance(object_instance_index, dict),
        "evidence_plan": {"frame_count": evidence_count, **evidence_plan},
        "retrieval_time": coarse_elapsed,
        "model": str(config.get("model") or "qwen3-vl-flash"),
        "call_time": elapsed,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "analyzer_version": ANALYZER_VERSION,
    }
    with open(run_dir / HISTORY_FILENAME, "a") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def load_qa_history(run_dir, limit=30):
    path = Path(run_dir) / HISTORY_FILENAME
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines()[-max(1, int(limit)):]:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return list(reversed(records))


def index_status(run_dir):
    run_dir = Path(run_dir).resolve()
    path = run_dir / INDEX_FILENAME
    if not path.exists():
        return {"exists": False, "stale": True, "path": str(path)}
    index = _load_json(path, {})
    metadata_paths = list(run_dir.glob("frame_*_waypoint.json"))
    source = index.get("source") if isinstance(index.get("source"), dict) else {}
    stale = index.get("analyzer_version") != ANALYZER_VERSION
    stale = stale or source.get("frame_count") != len(metadata_paths)
    if metadata_paths:
        stale = stale or source.get("latest_metadata_mtime_ns") != max(item.stat().st_mtime_ns for item in metadata_paths)
    return {
        "exists": True,
        "stale": stale,
        "path": str(path),
        "created_at": index.get("created_at"),
        "backend": index.get("backend"),
        "source": source,
        "summary": index.get("summary") or {},
        "capabilities": index.get("capabilities") or {},
    }
