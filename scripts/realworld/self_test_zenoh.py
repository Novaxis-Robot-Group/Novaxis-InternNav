import argparse
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from zenoh_internvla_rpc import ZenohInternVLAClient, add_zenoh_config_args, env_default_key, import_zenoh_or_raise


def make_depth_png(width, height, depth_m):
    depth_uint16 = np.full((height, width), int(depth_m * 10000.0), dtype=np.uint16)
    depth_image = Image.fromarray(depth_uint16)
    depth_bytes = io.BytesIO()
    depth_image.save(depth_bytes, format="PNG")
    depth_bytes.seek(0)
    return depth_bytes.getvalue()


def main():
    zenoh = import_zenoh_or_raise()
    zenoh.init_log_from_env_or("error")

    parser = argparse.ArgumentParser(description="Send one InternVLA request through Zenoh.")
    add_zenoh_config_args(parser)
    parser.set_defaults(key=env_default_key())
    parser.add_argument("--image", default="/data/users/chris/InternNav/assets/realworld_sample_data1/debug_raw_0000.jpg")
    parser.add_argument("--depth_m", type=float, default=1.0)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--instruction", default="")
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    image = Image.open(Path(args.image)).convert("RGB")
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="JPEG")

    payload = {
        "reset": args.reset,
        "instruction": args.instruction,
    }
    client = ZenohInternVLAClient.from_args(args, zenoh)
    try:
        response = client.eval_dual(image_bytes.getvalue(), make_depth_png(image.width, image.height, args.depth_m), payload)
    finally:
        client.close()

    print(json.dumps(response, indent=2, ensure_ascii=False))
    if not response.get("ok", False):
        raise RuntimeError(response.get("error", "Zenoh server returned an error."))


if __name__ == "__main__":
    main()
