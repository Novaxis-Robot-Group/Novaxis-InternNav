"""Run a lightweight detector over every saved experiment RGB frame.

The worker intentionally has no dependency on the InternVLA model. It writes
the JSONL contract consumed by ``experiment_instance_analyzer.py`` so a small
YOLO model, Grounded SAM, or SAM 3.x can be swapped without changing QA.
"""

import argparse
import json
import os
import re
import time
from pathlib import Path


DEFAULT_CLASS_NAMES = ("couch", "chair")


def _frame_idx(path):
    match = re.search(r"frame_(\d+)", Path(path).name)
    return int(match.group(1)) if match else -1


def _resolve_class_ids(names, requested_names):
    normalized = {str(name).strip().casefold() for name in requested_names if str(name).strip()}
    available = dict(names) if isinstance(names, dict) else dict(enumerate(names))
    if not normalized:
        return sorted(int(class_id) for class_id in available)
    selected = [int(class_id) for class_id, name in available.items() if str(name).casefold() in normalized]
    if not selected:
        raise ValueError(
            f"None of the requested classes {sorted(normalized)} exist in model classes: "
            f"{sorted(str(value) for value in available.values())}"
        )
    return selected


def _result_detections(result, names, selected_class_ids):
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.detach().cpu().tolist()
    scores = boxes.conf.detach().cpu().tolist()
    class_ids = [int(value) for value in boxes.cls.detach().cpu().tolist()]
    detections = []
    for position, (bbox, score, class_id) in enumerate(zip(xyxy, scores, class_ids)):
        if class_id not in selected_class_ids:
            continue
        x1, y1, x2, y2 = [float(value) for value in bbox]
        detections.append(
            {
                "track_id": "",
                "label": str(names[class_id]).strip().lower(),
                "score": float(score),
                "bbox_xyxy": [x1, y1, x2, y2],
                "mask_centroid": [(x1 + x2) / 2.0, (y1 + y2) / 2.0],
                "mask_area": max(0.0, x2 - x1) * max(0.0, y2 - y1),
                "source_detection_id": f"{position}",
            }
        )
    return detections


def run(args):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is not installed. Install it with: "
            "python -m pip install ultralytics"
        ) from exc

    run_dir = Path(args.run_dir).resolve()
    output_path = Path(args.output or run_dir / "experiment_instance_detections.jsonl").resolve()
    metadata_path = output_path.with_name("experiment_instance_detector_meta.json")
    frames = sorted(run_dir.glob("frame_*_rgb.jpg"), key=_frame_idx)
    frames = [path for path in frames if _frame_idx(path) >= args.frame_start]
    if args.frame_end >= 0:
        frames = [path for path in frames if _frame_idx(path) <= args.frame_end]
    frames = frames[:: max(1, int(args.stride))]
    if args.max_frames > 0:
        frames = frames[: args.max_frames]
    if not frames:
        raise ValueError(f"No matching frame_*_rgb.jpg files found in {run_dir}")

    model = YOLO(args.model)
    names = model.names
    requested_names = [item.strip() for item in args.classes.split(",")]
    selected_class_ids = _resolve_class_ids(names, requested_names)
    all_classes = not any(requested_names)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_suffix(output_path.suffix + ".tmp")
    started_at = time.time()
    detection_count = 0
    detected_frames = 0

    with open(temporary_output, "w") as handle:
        for offset in range(0, len(frames), args.batch_size):
            batch = frames[offset : offset + args.batch_size]
            results = model.predict(
                source=[str(path) for path in batch],
                conf=args.confidence,
                iou=args.nms_iou,
                imgsz=args.image_size,
                device=args.device,
                half=args.half,
                classes=None if all_classes else selected_class_ids,
                verbose=False,
                stream=False,
            )
            for path, result in zip(batch, results):
                detections = _result_detections(result, names, selected_class_ids)
                detection_count += len(detections)
                detected_frames += int(bool(detections))
                handle.write(
                    json.dumps(
                        {"frame_idx": _frame_idx(path), "rgb_file": path.name, "detections": detections},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            print(
                f"processed {min(offset + len(batch), len(frames))}/{len(frames)} frames, "
                f"detections={detection_count}",
                flush=True,
            )

    os.replace(temporary_output, output_path)
    metadata = {
        "backend": "ultralytics_yolo_full_frame",
        "model": str(args.model),
        "classes": [str(names[class_id]) for class_id in selected_class_ids],
        "all_classes": all_classes,
        "confidence": float(args.confidence),
        "nms_iou": float(args.nms_iou),
        "image_size": int(args.image_size),
        "stride": int(args.stride),
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "processed_frame_count": len(frames),
        "detected_frame_count": detected_frames,
        "detection_count": detection_count,
        "elapsed_seconds": round(time.time() - started_at, 3),
        "output": str(output_path),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(json.dumps({"ok": True, **metadata}, ensure_ascii=False))
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Detect object candidates in every saved RGB frame.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output")
    parser.add_argument("--model", default="yolo11s.pt")
    parser.add_argument(
        "--classes",
        default="",
        help="Comma-separated model classes. Empty means all classes for general Experiment QA.",
    )
    parser.add_argument("--confidence", type=float, default=0.05)
    parser.add_argument("--nms-iou", type=float, default=0.60)
    parser.add_argument("--image-size", type=int, default=1280)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-end", type=int, default=-1)
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()
    if not 0.0 < args.confidence < 1.0:
        parser.error("--confidence must be between 0 and 1")
    run(args)


if __name__ == "__main__":
    main()
