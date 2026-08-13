import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image


def encode_rgb_jpeg(image_path):
    image = Image.open(image_path).convert("RGB")
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="JPEG")
    image_bytes.seek(0)
    return image, image_bytes


def encode_depth_png(width, height, depth_m):
    # Matches http_internvla_client.py: meters -> uint16 PNG, server divides by 10000.
    depth = np.full((height, width), int(depth_m * 10000.0), dtype=np.uint16)
    depth_image = Image.fromarray(depth)
    depth_bytes = io.BytesIO()
    depth_image.save(depth_bytes, format="PNG")
    depth_bytes.seek(0)
    return depth_bytes


def post_frame(session, url, rgb_path, idx, reset, instruction, depth_m):
    image, image_bytes = encode_rgb_jpeg(rgb_path)
    depth_bytes = encode_depth_png(image.width, image.height, depth_m)
    payload = {
        "reset": reset,
        "idx": idx,
        "instruction": instruction,
    }
    files = {
        "image": ("rgb_image", image_bytes, "image/jpeg"),
        "depth": ("depth_image", depth_bytes, "image/png"),
    }
    started_at = time.time()
    response = session.post(url, files=files, data={"json": json.dumps(payload)}, timeout=180)
    elapsed = time.time() - started_at
    print(f"\nframe={idx} reset={reset} image={rgb_path.name} status={response.status_code} time={elapsed:.2f}s")
    print(response.text)
    response.raise_for_status()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:5801/eval_dual")
    parser.add_argument("--scene_dir", default="/data/users/chris/InternNav/assets/realworld_sample_data1")
    parser.add_argument("--instruction", default="")
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--interval", type=float, default=0.3)
    parser.add_argument("--depth_m", type=float, default=1.0)
    args = parser.parse_args()

    scene_dir = Path(args.scene_dir)
    rgb_paths = sorted(scene_dir.glob("debug_raw_*.jpg"))
    rgb_paths = [p for p in rgb_paths if "look_down" not in p.name]
    selected = rgb_paths[args.start :: args.stride][: args.num_frames]
    if not selected:
        raise FileNotFoundError(f"No debug_raw_*.jpg images found in {scene_dir}")

    session = requests.Session()
    session.trust_env = False
    for idx, rgb_path in enumerate(selected):
        post_frame(
            session=session,
            url=args.url,
            rgb_path=rgb_path,
            idx=idx,
            reset=(idx == 0),
            instruction=args.instruction,
            depth_m=args.depth_m,
        )
        if idx != len(selected) - 1:
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
