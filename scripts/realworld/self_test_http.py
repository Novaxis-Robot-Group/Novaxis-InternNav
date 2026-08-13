import argparse
import io
import json
from pathlib import Path

import numpy as np
import requests
from PIL import Image


def make_depth_png(width, height, depth_m):
    depth_uint16 = np.full((height, width), int(depth_m * 10000.0), dtype=np.uint16)
    depth_image = Image.fromarray(depth_uint16)
    depth_bytes = io.BytesIO()
    depth_image.save(depth_bytes, format="PNG")
    depth_bytes.seek(0)
    return depth_bytes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:5801/eval_dual")
    parser.add_argument(
        "--image",
        default="/data/users/chris/InternNav/assets/realworld_sample_data1/debug_raw_0000.jpg",
    )
    parser.add_argument("--depth_m", type=float, default=1.0)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--instruction", default="")
    args = parser.parse_args()

    image_path = Path(args.image)
    image = Image.open(image_path).convert("RGB")
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="JPEG")
    image_bytes.seek(0)

    depth_bytes = make_depth_png(image.width, image.height, args.depth_m)
    payload = {
        "reset": args.reset,
        "instruction": args.instruction,
    }
    files = {
        "image": ("rgb.jpg", image_bytes, "image/jpeg"),
        "depth": ("depth.png", depth_bytes, "image/png"),
    }

    session = requests.Session()
    session.trust_env = False
    response = session.post(args.url, files=files, data={"json": json.dumps(payload)}, timeout=180)
    print("status:", response.status_code)
    print(response.text)
    response.raise_for_status()


if __name__ == "__main__":
    main()
