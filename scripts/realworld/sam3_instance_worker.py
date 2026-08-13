"""SAM 3/3.1 full-video detection worker.

Run this script from a dedicated SAM environment, not the InternNav inference
environment. It emits the transport-neutral JSONL consumed by
``experiment_instance_analyzer.py``.
"""

import argparse
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np


def frame_idx(path):
    match = re.search(r"frame_(\d+)", Path(path).name)
    return int(match.group(1)) if match else -1


def prepare_video_frames(run_dir, work_dir):
    source_frames = sorted(Path(run_dir).glob("frame_*_rgb.jpg"), key=frame_idx)
    if not source_frames:
        raise ValueError(f"No frame_*_rgb.jpg files found in {run_dir}")
    mapping = []
    frame_dir = Path(work_dir) / "frames"
    frame_dir.mkdir(parents=True)
    for position, source in enumerate(source_frames):
        target = frame_dir / f"{position:06d}.jpg"
        try:
            target.symlink_to(source.resolve())
        except OSError:
            shutil.copy2(source, target)
        mapping.append(frame_idx(source))
    return frame_dir, mapping


def to_numpy(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return np.asarray(value)


def detections_from_output(output, label):
    output = output if isinstance(output, dict) else {}
    masks = to_numpy(output.get("out_binary_masks") if "out_binary_masks" in output else output.get("masks"))
    if masks is None:
        return []
    while masks.ndim > 3 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim == 2:
        masks = masks[None, ...]
    object_ids = to_numpy(output.get("out_obj_ids") if "out_obj_ids" in output else output.get("object_ids"))
    scores = to_numpy(output.get("scores") if "scores" in output else output.get("out_probs"))
    detections = []
    for position, raw_mask in enumerate(masks):
        mask = np.asarray(raw_mask).squeeze() > 0
        ys, xs = np.nonzero(mask)
        if not len(xs):
            continue
        detections.append(
            {
                "track_id": str(object_ids[position] if object_ids is not None and position < len(object_ids) else position),
                "label": label,
                "score": float(scores[position] if scores is not None and position < len(scores) else 1.0),
                "bbox_xyxy": [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)],
                "mask_centroid": [float(xs.mean()), float(ys.mean())],
                "mask_area": int(mask.sum()),
            }
        )
    return detections


def build_predictor(checkpoint, bpe_path, multiplex):
    if multiplex:
        from sam3.model_builder import build_sam3_multiplex_video_predictor

        return build_sam3_multiplex_video_predictor(
            checkpoint,
            bpe_path,
            use_fa3=False,
            async_loading_frames=False,
            use_rope_real=False,
        )
    from sam3.model_builder import build_sam3_video_predictor

    kwargs = {"checkpoint_path": checkpoint} if checkpoint else {}
    return build_sam3_video_predictor(**kwargs)


def run(args):
    run_dir = Path(args.run_dir).resolve()
    output_path = Path(args.output or run_dir / "experiment_instance_detections.jsonl").resolve()
    concepts = [item.strip() for item in args.concepts.split(".") if item.strip()]
    if not concepts:
        raise ValueError("At least one text concept is required.")
    # Use one broad concept per pass. Multiple sessions keep object IDs isolated
    # and are merged by the lightweight indexer afterwards.
    by_frame = {}
    with tempfile.TemporaryDirectory(prefix="internnav_sam3_") as temporary:
        frame_dir, mapping = prepare_video_frames(run_dir, temporary)
        for concept in concepts:
            predictor = build_predictor(args.checkpoint, args.bpe_path, args.multiplex)
            response = predictor.handle_request(
                request={"type": "start_session", "resource_path": str(frame_dir), "offload_video_to_cpu": True}
            )
            session_id = response["session_id"]
            predictor.handle_request(
                request={"type": "add_prompt", "session_id": session_id, "frame_index": 0, "text": concept}
            )
            try:
                for propagated in predictor.propagate_in_video(session_id, "forward"):
                    position = int(propagated["frame_index"])
                    if position < 0 or position >= len(mapping):
                        continue
                    actual_frame = mapping[position]
                    detections = detections_from_output(propagated.get("outputs"), concept)
                    by_frame.setdefault(actual_frame, []).extend(detections)
            finally:
                try:
                    predictor.handle_request(request={"type": "close_session", "session_id": session_id})
                except Exception:
                    pass
    temporary_output = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(temporary_output, "w") as handle:
        for actual_frame in mapping:
            handle.write(json.dumps({"frame_idx": actual_frame, "detections": by_frame.get(actual_frame, [])}) + "\n")
    os.replace(temporary_output, output_path)
    print(json.dumps({"ok": True, "frames": len(mapping), "output": str(output_path)}))


def main():
    parser = argparse.ArgumentParser(description="Run SAM 3.1 over every saved RGB frame.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output")
    parser.add_argument("--checkpoint", default=os.environ.get("INTERNNAV_SAM3_CHECKPOINT", ""))
    parser.add_argument("--bpe-path", default=os.environ.get("INTERNNAV_SAM3_BPE_PATH", "sam3/assets/bpe_simple_vocab_16e6.txt.gz"))
    parser.add_argument("--concepts", default="sofa.couch.armchair.upholstered seating furniture")
    parser.add_argument("--multiplex", action="store_true", help="Use the SAM 3.1 Object Multiplex predictor.")
    args = parser.parse_args()
    if args.multiplex and not args.checkpoint:
        parser.error("--checkpoint is required for the multiplex predictor")
    run(args)


if __name__ == "__main__":
    main()
