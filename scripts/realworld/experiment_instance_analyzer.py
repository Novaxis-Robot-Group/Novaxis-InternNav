"""Offline full-frame object-instance indexing for Experiment QA.

The detector/tracker runs in a separate environment and writes one JSON object
per frame to ``experiment_instance_detections.jsonl``. This module performs the
lightweight, deterministic half of the pipeline in the InternNav environment:
track consolidation, RGB-D localization, optional world-frame fusion, and
representative-frame selection. It never changes policy or runtime state.
"""

import argparse
import importlib.util
import json
import math
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image


DETECTIONS_FILENAME = "experiment_instance_detections.jsonl"
INSTANCE_INDEX_FILENAME = "experiment_instance_index.json"
DETECTOR_METADATA_FILENAME = "experiment_instance_detector_meta.json"
INSTANCE_ANALYZER_VERSION = 1
DEFAULT_CONCEPTS = ["sofa", "couch", "armchair", "upholstered seating furniture"]


def _load_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return {} if default is None else default


def _atomic_write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    os.replace(temporary, path)


def instance_backend_status(run_dir=None):
    """Report detector readiness without importing CUDA-heavy packages."""
    runner = Path(os.environ.get("INTERNNAV_SAM3_RUNNER", "")).expanduser()
    python = Path(os.environ.get("INTERNNAV_SAM3_PYTHON", "")).expanduser()
    checkpoint = Path(os.environ.get("INTERNNAV_SAM3_CHECKPOINT", "")).expanduser()
    detection_path = Path(run_dir) / DETECTIONS_FILENAME if run_dir else None
    project_root = Path(__file__).resolve().parents[2]
    yolo_model = Path(
        os.environ.get("INTERNNAV_INSTANCE_MODEL", str(project_root / "yolo11s.pt"))
    ).expanduser()
    return {
        "yolo_in_current_environment": importlib.util.find_spec("ultralytics") is not None,
        "yolo_model_configured": yolo_model.is_file(),
        "yolo_model_path": str(yolo_model),
        "sam3_in_current_environment": importlib.util.find_spec("sam3") is not None,
        "external_python_configured": bool(str(python)) and python.is_file(),
        "runner_configured": bool(str(runner)) and runner.is_file(),
        "checkpoint_configured": bool(str(checkpoint)) and checkpoint.is_file(),
        "detections_exist": bool(detection_path and detection_path.is_file()),
        "detections_path": str(detection_path) if detection_path else "",
        "ready_to_launch": (
            importlib.util.find_spec("ultralytics") is not None and yolo_model.is_file()
        ) or (python.is_file() and runner.is_file() and checkpoint.is_file()),
        "note": (
            "The lightweight YOLO backend can run in the navigation environment. "
            "SAM 3.1 remains an optional external detector using the same JSONL contract."
        ),
    }


def _frame_metadata(run_dir):
    records = {}
    for path in sorted(Path(run_dir).glob("frame_*_waypoint.json")):
        value = _load_json(path, {})
        match = re.search(r"frame_(\d+)", path.name)
        frame_idx = int(value.get("frame_idx", match.group(1) if match else -1))
        records[frame_idx] = value
    return records


def load_frame_detections(run_dir, minimum_score=0.15):
    path = Path(run_dir) / DETECTIONS_FILENAME
    if not path.is_file():
        raise FileNotFoundError(
            f"Full-frame detector output is missing: {path}. Run the SAM 3.1 worker first."
        )
    frames = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid detector JSONL at line {line_number}: {exc}") from exc
        detections = []
        for raw in item.get("detections") or []:
            try:
                bbox = [float(value) for value in raw["bbox_xyxy"]]
                score = float(raw.get("score", 1.0))
            except (KeyError, TypeError, ValueError):
                continue
            if len(bbox) != 4 or score < float(minimum_score):
                continue
            detections.append(
                {
                    "source_track_id": str(raw.get("track_id") or ""),
                    "label": str(raw.get("label") or "object").strip().lower(),
                    "score": score,
                    "bbox_xyxy": bbox,
                    "mask_centroid": raw.get("mask_centroid"),
                    "mask_area": raw.get("mask_area"),
                    "embedding": raw.get("embedding"),
                }
            )
        frames.append({"frame_idx": int(item["frame_idx"]), "detections": detections})
    return sorted(frames, key=lambda item: item["frame_idx"])


def _bbox_iou(left, right):
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _same_concept(left, right):
    seating = {"sofa", "couch", "armchair", "seating", "upholstered seating furniture"}
    return left == right or (left in seating and right in seating)


def consolidate_tracks(frame_detections, iou_threshold=0.25, max_frame_gap=8):
    """Consolidate propagated masks; prefer stable source IDs when available."""
    tracks = []
    next_id = 1
    source_to_track = {}
    for frame in frame_detections:
        frame_idx = int(frame["frame_idx"])
        used_tracks = set()
        for detection in sorted(frame["detections"], key=lambda item: item["score"], reverse=True):
            track = None
            source_id = detection.get("source_track_id")
            source_key = f"{detection['label']}:{source_id}" if source_id else ""
            if source_key and source_key in source_to_track:
                candidate = source_to_track[source_key]
                if frame_idx - candidate["last_frame"] <= max_frame_gap:
                    track = candidate
            if track is None:
                candidates = [
                    item
                    for item in tracks
                    if item["instance_id"] not in used_tracks
                    and frame_idx - item["last_frame"] <= max_frame_gap
                    and _same_concept(item["label"], detection["label"])
                ]
                candidates.sort(
                    key=lambda item: _bbox_iou(item["observations"][-1]["bbox_xyxy"], detection["bbox_xyxy"]),
                    reverse=True,
                )
                if candidates and _bbox_iou(
                    candidates[0]["observations"][-1]["bbox_xyxy"], detection["bbox_xyxy"]
                ) >= iou_threshold:
                    track = candidates[0]
            if track is None:
                track = {
                    "instance_id": f"track_{next_id:04d}",
                    "label": detection["label"],
                    "first_frame": frame_idx,
                    "last_frame": frame_idx,
                    "observations": [],
                }
                next_id += 1
                tracks.append(track)
            observation = {"frame_idx": frame_idx, **detection}
            track["observations"].append(observation)
            track["last_frame"] = frame_idx
            used_tracks.add(track["instance_id"])
            if source_key:
                source_to_track[source_key] = track
    return tracks


def _depth_at_detection(run_dir, metadata, detection):
    depth_file = metadata.get("depth_file")
    if not depth_file or not (Path(run_dir) / depth_file).is_file():
        return None
    depth = np.asarray(Image.open(Path(run_dir) / depth_file))
    centroid = detection.get("mask_centroid")
    if not isinstance(centroid, (list, tuple)) or len(centroid) < 2:
        x1, y1, x2, y2 = detection["bbox_xyxy"]
        centroid = [(x1 + x2) / 2.0, (y1 + y2) / 2.0]
    u, v = int(round(float(centroid[0]))), int(round(float(centroid[1])))
    radius = 3
    patch = depth[max(0, v - radius) : v + radius + 1, max(0, u - radius) : u + radius + 1]
    valid = patch[np.isfinite(patch) & (patch > 0)]
    if not valid.size:
        return None
    value = float(np.median(valid))
    return value / 1000.0 if value > 20 else value


def _camera_intrinsic(metadata):
    value = metadata.get("camera_intrinsic") or (metadata.get("request_json") or {}).get("camera_intrinsic")
    try:
        matrix = np.asarray(value, dtype=np.float64).reshape(3, 3)
    except (TypeError, ValueError):
        return None
    return matrix


def _world_center(metadata, detection, depth_m):
    if depth_m is None:
        return None
    intrinsic = _camera_intrinsic(metadata)
    if intrinsic is None:
        return None
    centroid = detection.get("mask_centroid")
    if not isinstance(centroid, (list, tuple)) or len(centroid) < 2:
        x1, y1, x2, y2 = detection["bbox_xyxy"]
        centroid = [(x1 + x2) / 2.0, (y1 + y2) / 2.0]
    u, v = float(centroid[0]), float(centroid[1])
    camera_point = np.array(
        [(u - intrinsic[0, 2]) * depth_m / intrinsic[0, 0],
         (v - intrinsic[1, 2]) * depth_m / intrinsic[1, 1], depth_m, 1.0]
    )
    transform = metadata.get("world_T_camera") or (metadata.get("request_json") or {}).get("world_T_camera")
    try:
        world_transform = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    except (TypeError, ValueError):
        return None
    return (world_transform @ camera_point)[:3].tolist()


def _representative_observations(observations, limit=4):
    if not observations:
        return []
    candidates = [observations[0], max(observations, key=lambda item: item["score"]), observations[len(observations) // 2], observations[-1]]
    result = []
    for item in candidates:
        if item not in result:
            result.append(item)
        if len(result) >= limit:
            break
    return result


def _merge_world_instances(instances, distance_threshold=0.75):
    """Merge repeated 2D tracks only when 3D evidence supports identity."""
    merged = []
    for instance in sorted(instances, key=lambda item: item["first_seen_frame"]):
        center = instance.get("world_center")
        match = None
        if center is not None:
            for candidate in merged:
                candidate_center = candidate.get("world_center")
                if candidate_center is None or not _same_concept(candidate["category"], instance["category"]):
                    continue
                if float(np.linalg.norm(np.asarray(center) - np.asarray(candidate_center))) <= distance_threshold:
                    match = candidate
                    break
        if match is None:
            merged.append(dict(instance))
            continue
        left_weight = max(1, int(match["visible_frame_count"]))
        right_weight = max(1, int(instance["visible_frame_count"]))
        match["world_center"] = (
            (np.asarray(match["world_center"]) * left_weight + np.asarray(center) * right_weight)
            / (left_weight + right_weight)
        ).round(3).tolist()
        match["first_seen_frame"] = min(match["first_seen_frame"], instance["first_seen_frame"])
        match["last_seen_frame"] = max(match["last_seen_frame"], instance["last_seen_frame"])
        match["visible_frame_count"] += instance["visible_frame_count"]
        match["confidence"] = max(match["confidence"], instance["confidence"])
        combined = match["representative_frames"] + instance["representative_frames"]
        match["representative_frames"] = list(
            {item["frame_idx"]: item for item in combined}.values()
        )[:4]
        match.setdefault("merged_track_ids", []).append(instance["instance_id"])
    for position, instance in enumerate(merged, 1):
        instance["instance_id"] = f"{instance['category'].replace(' ', '_')}_{position:03d}"
    return merged


def _build_detection_summary(frame_detections, appearance_gap=30):
    """Summarize full-frame hits without pretending fragmented tracks are unique objects."""
    category_frames = {}
    for frame in frame_detections:
        grouped = {}
        for detection in frame.get("detections") or []:
            grouped.setdefault(detection["label"], []).append(detection)
        for category, detections in grouped.items():
            category_frames.setdefault(category, []).append(
                {
                    "frame_idx": int(frame["frame_idx"]),
                    "count": len(detections),
                    "best_score": max(float(item["score"]) for item in detections),
                    "score_sum": sum(float(item["score"]) for item in detections),
                }
            )

    summary = {}
    for category, observations in category_frames.items():
        windows = []
        for observation in observations:
            if not windows or observation["frame_idx"] - windows[-1]["end_frame"] > appearance_gap:
                windows.append(
                    {
                        "start_frame": observation["frame_idx"],
                        "end_frame": observation["frame_idx"],
                        "detected_frame_count": 0,
                        "max_simultaneous": 0,
                        "best_frame_idx": observation["frame_idx"],
                        "best_frame_count": observation["count"],
                        "best_frame_score": observation["score_sum"],
                    }
                )
            window = windows[-1]
            window["end_frame"] = observation["frame_idx"]
            window["detected_frame_count"] += 1
            window["max_simultaneous"] = max(window["max_simultaneous"], observation["count"])
            current_rank = (observation["count"], observation["score_sum"])
            best_rank = (window["best_frame_count"], window["best_frame_score"])
            if current_rank > best_rank:
                window["best_frame_idx"] = observation["frame_idx"]
                window["best_frame_count"] = observation["count"]
                window["best_frame_score"] = observation["score_sum"]
        for window in windows:
            window["best_frame_score"] = round(float(window["best_frame_score"]), 4)
        summary[category] = {
            "detected_frame_count": len(observations),
            "max_simultaneous": max(item["count"] for item in observations),
            "appearance_window_count": len(windows),
            "appearance_windows": windows,
        }
    return summary


def build_instance_index(run_dir, minimum_score=0.15, iou_threshold=0.25, max_frame_gap=8):
    run_dir = Path(run_dir).resolve()
    metadata_by_frame = _frame_metadata(run_dir)
    frames = load_frame_detections(run_dir, minimum_score=minimum_score)
    tracks = consolidate_tracks(frames, iou_threshold=iou_threshold, max_frame_gap=max_frame_gap)
    instances = []
    for track in tracks:
        world_points = []
        depth_values = []
        for observation in track["observations"]:
            metadata = metadata_by_frame.get(observation["frame_idx"], {})
            depth_m = _depth_at_detection(run_dir, metadata, observation)
            if depth_m is not None:
                depth_values.append(depth_m)
            center = _world_center(metadata, observation, depth_m)
            if center is not None:
                world_points.append(center)
        representatives = []
        for observation in _representative_observations(track["observations"]):
            metadata = metadata_by_frame.get(observation["frame_idx"], {})
            representatives.append(
                {
                    "frame_idx": observation["frame_idx"],
                    "score": round(float(observation["score"]), 4),
                    "bbox_xyxy": [round(value, 2) for value in observation["bbox_xyxy"]],
                    "rgb_file": metadata.get("rgb_file") or f"frame_{observation['frame_idx']:06d}_rgb.jpg",
                    "depth_file": metadata.get("depth_file") or "",
                }
            )
        instances.append(
            {
                "instance_id": track["instance_id"],
                "category": track["label"],
                "first_seen_frame": track["first_frame"],
                "last_seen_frame": track["last_frame"],
                "visible_frame_count": len(track["observations"]),
                "confidence": round(max(item["score"] for item in track["observations"]), 4),
                "median_depth_m": round(float(np.median(depth_values)), 3) if depth_values else None,
                "world_center": np.median(np.asarray(world_points), axis=0).round(3).tolist() if world_points else None,
                "representative_frames": representatives,
            }
        )
    instances = _merge_world_instances(instances)
    has_world_pose = any(item["world_center"] is not None for item in instances)
    detector_metadata = _load_json(run_dir / DETECTOR_METADATA_FILENAME, {})
    backend = str(detector_metadata.get("backend") or "full_frame_detector_jsonl_plus_rgbd_fusion")
    index = {
        "analyzer_version": INSTANCE_ANALYZER_VERSION,
        "backend": backend,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_name": run_dir.name,
        "source": {
            "frame_count": len(metadata_by_frame),
            "detection_frame_count": len(frames),
            "detector": detector_metadata,
        },
        "capabilities": {
            "full_frame_detection": True,
            "continuous_tracking": True,
            "depth_localization": any(item["median_depth_m"] is not None for item in instances),
            "world_pose_fusion": has_world_pose,
            "cross_loop_unique_count": has_world_pose,
            "vlm_verified": False,
        },
        "thresholds": {
            "minimum_score": float(minimum_score),
            "iou_threshold": float(iou_threshold),
            "max_frame_gap": int(max_frame_gap),
        },
        "detection_summary": _build_detection_summary(frames),
        "instances": instances,
    }
    _atomic_write_json(run_dir / INSTANCE_INDEX_FILENAME, index)
    return index


def load_instance_index(run_dir):
    path = Path(run_dir) / INSTANCE_INDEX_FILENAME
    return _load_json(path, None) if path.is_file() else None


def instance_index_status(run_dir):
    path = Path(run_dir) / INSTANCE_INDEX_FILENAME
    index = load_instance_index(run_dir)
    return {
        "exists": isinstance(index, dict),
        "path": str(path),
        "backend": index.get("backend") if isinstance(index, dict) else "",
        "created_at": index.get("created_at") if isinstance(index, dict) else "",
        "instance_count": len(index.get("instances") or []) if isinstance(index, dict) else 0,
        "capabilities": index.get("capabilities") if isinstance(index, dict) else {},
        "backend_status": instance_backend_status(run_dir),
    }


def instance_evidence(index, question, limit=12):
    """Return representative frame records for object-related QA."""
    if not isinstance(index, dict):
        return [], []
    lower = str(question or "").casefold()
    concept_groups = {
        "sofa": ("sofa", "couch", "armchair", "seating", "沙发", "座椅"),
        "person": ("person", "people", "human", "人", "人员"),
        "box": ("box", "carton", "纸箱", "箱子"),
        "plant": ("plant", "potted plant", "绿植", "植物", "盆栽"),
    }
    wanted = None
    for group, terms in concept_groups.items():
        if any(term in lower for term in terms):
            wanted = group
            break
    selected_instances = []
    frames = []
    matching_instances = []
    for instance in index.get("instances") or []:
        label = str(instance.get("category") or "").casefold()
        if wanted and not any(term in label for term in concept_groups[wanted] if term.isascii()):
            continue
        matching_instances.append(instance)

    # Full-frame appearance windows preserve temporal coverage. This avoids
    # spending the complete VLM image budget on fragmented tracks from the
    # beginning of a long experiment.
    summary = index.get("detection_summary") or {}
    relevant_windows = []
    for category, category_summary in summary.items():
        label = str(category).casefold()
        if wanted and not any(term in label for term in concept_groups[wanted] if term.isascii()):
            continue
        for window in category_summary.get("appearance_windows") or []:
            # Keep low detector thresholds for recall, but do not spend VLM
            # image budget on a very weak two-frame flicker unless no stronger
            # evidence exists later.
            if (
                int(window.get("detected_frame_count", 0)) < 3
                and float(window.get("best_frame_score", 0.0)) < 0.10
            ):
                continue
            relevant_windows.append({"category": category, **window})
    relevant_windows.sort(key=lambda item: int(item.get("start_frame", -1)))
    if len(relevant_windows) > limit:
        relevant_windows = sorted(
            relevant_windows,
            key=lambda item: (
                int(item.get("detected_frame_count", 0)),
                int(item.get("max_simultaneous", 0)),
                float(item.get("best_frame_score", 0.0)),
            ),
            reverse=True,
        )[:limit]
        relevant_windows.sort(key=lambda item: int(item.get("start_frame", -1)))
    for window in relevant_windows:
        frame_idx = int(window["best_frame_idx"])
        frames.append(
            {
                "frame_idx": frame_idx,
                "rgb_file": f"frame_{frame_idx:06d}_rgb.jpg",
                "depth_file": f"frame_{frame_idx:06d}_depth.png",
                "vis_file": "",
                "instruction": "",
                "agent_task_instruction": "",
                "action": {},
                "service_state": "",
                "replan_required": False,
                "instance_category": window["category"],
                "appearance_window": window,
            }
        )

    matching_instances.sort(
        key=lambda item: (int(item.get("visible_frame_count", 0)), float(item.get("confidence", 0.0))),
        reverse=True,
    )
    selected_instances = matching_instances[: max(limit * 2, limit)]
    if frames:
        return frames, selected_instances

    for instance in selected_instances:
        for representative in instance.get("representative_frames") or []:
            entry = {
                "frame_idx": representative.get("frame_idx"),
                "rgb_file": representative.get("rgb_file"),
                "depth_file": representative.get("depth_file"),
                "vis_file": "",
                "instruction": "",
                "agent_task_instruction": "",
                "action": {},
                "service_state": "",
                "replan_required": False,
                "instance_id": instance.get("instance_id"),
                "instance_category": instance.get("category"),
            }
            if entry not in frames:
                frames.append(entry)
            if len(frames) >= limit:
                break
        if len(frames) >= limit:
            break
    return frames, selected_instances


def main():
    parser = argparse.ArgumentParser(description="Build an Experiment QA object-instance index.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--minimum-score", type=float, default=0.10)
    parser.add_argument("--iou-threshold", type=float, default=0.25)
    parser.add_argument("--max-frame-gap", type=int, default=8)
    args = parser.parse_args()
    index = build_instance_index(
        args.run_dir,
        minimum_score=args.minimum_score,
        iou_threshold=args.iou_threshold,
        max_frame_gap=args.max_frame_gap,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "run_name": index.get("run_name"),
                "backend": index.get("backend"),
                "instance_count": len(index.get("instances") or []),
                "source": index.get("source"),
                "capabilities": index.get("capabilities"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
