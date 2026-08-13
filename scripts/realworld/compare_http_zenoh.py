import argparse
import io
import json
import statistics
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image

from zenoh_internvla_rpc import ZenohInternVLAClient, add_zenoh_config_args, env_default_key, import_zenoh_or_raise


def make_depth_png(width, height, depth_m):
    depth_uint16 = np.full((height, width), int(depth_m * 10000.0), dtype=np.uint16)
    depth_image = Image.fromarray(depth_uint16)
    depth_bytes = io.BytesIO()
    depth_image.save(depth_bytes, format="PNG")
    depth_bytes.seek(0)
    return depth_bytes.getvalue()


def make_rgb_jpeg(image_path):
    image = Image.open(Path(image_path)).convert("RGB")
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="JPEG")
    return image, image_bytes.getvalue()


def request_http(url, image_bytes, depth_bytes, payload, timeout, session=None):
    files = {
        "image": ("rgb.jpg", io.BytesIO(image_bytes), "image/jpeg"),
        "depth": ("depth.png", io.BytesIO(depth_bytes), "image/png"),
    }
    owns_session = session is None
    if session is None:
        session = requests.Session()
        session.trust_env = False
    started_at = time.time()
    try:
        response = session.post(url, files=files, data={"json": json.dumps(payload)}, timeout=timeout)
        elapsed = time.time() - started_at
        return response.status_code, response.text, elapsed
    finally:
        if owns_session:
            session.close()


def percentile(values, pct):
    if not values:
        return None
    ordered = sorted(values)
    idx = int(round((pct / 100.0) * (len(ordered) - 1)))
    return ordered[max(0, min(idx, len(ordered) - 1))]


def seconds_to_ms(value):
    if value is None:
        return None
    return float(value) * 1000.0


def format_ms(value):
    if value is None:
        return "n/a"
    return f"{seconds_to_ms(value):.3f}"


def payload_bytes_http(image_bytes, depth_bytes, payload):
    return len(image_bytes) + len(depth_bytes) + len(json.dumps(payload).encode("utf-8"))


def print_timing_table(http_elapsed=None, http_result=None, zenoh_elapsed=None, zenoh_response=None):
    http_result = http_result or {}
    zenoh_response = zenoh_response or {}
    http_timing = http_result.get("_timing", {}) if isinstance(http_result, dict) else {}
    zenoh_server_timing = zenoh_response.get("_timing", {})

    rows = [
        (
            "end_to_end_total",
            http_elapsed,
            zenoh_elapsed,
            "client observed total request/reply time",
        ),
        (
            "server_total",
            http_timing.get("server_total_time"),
            zenoh_server_timing.get("server_total_time"),
            "server receive/decode + inference + response build",
        ),
        (
            "inference",
            http_timing.get("inference_time"),
            zenoh_server_timing.get("inference_time"),
            "agent.step/model generation time",
        ),
        (
            "server_core",
            http_timing.get("server_core_time"),
            zenoh_server_timing.get("server_core_time"),
            "run_dual_inference total including logging",
        ),
        (
            "image_depth_decode",
            http_timing.get("image_depth_decode_time"),
            zenoh_server_timing.get("image_depth_decode_time"),
            "PIL decode RGB JPEG + depth PNG",
        ),
        (
            "request_read_or_payload_decode",
            http_timing.get("request_read_time"),
            zenoh_server_timing.get("payload_decode_time"),
            "HTTP stream read vs Zenoh binary header parse",
        ),
        (
            "client_payload_encode",
            None,
            zenoh_response.get("_client_encode_time"),
            "Zenoh binary framing encode time",
        ),
    ]

    http_comm = None
    if http_timing.get("server_total_time") is not None:
        http_comm = max(0.0, http_elapsed - http_timing["server_total_time"])
    zenoh_comm = None
    if zenoh_server_timing.get("server_total_time") is not None:
        zenoh_comm = max(0.0, zenoh_elapsed - zenoh_server_timing["server_total_time"])
    rows.append(
        (
            "estimated_comm_overhead",
            http_comm,
            zenoh_comm,
            "end_to_end_total - server_total",
        )
    )

    print("\n[Timing Breakdown: ms]")
    print(f"{'metric':<32} {'HTTP':>12} {'Zenoh':>12}  note")
    print("-" * 86)
    for metric, http_value, zenoh_value, note in rows:
        print(f"{metric:<32} {format_ms(http_value):>12} {format_ms(zenoh_value):>12}  {note}")


def print_size_table(http_request_bytes=None, http_response_text=None, zenoh_response=None):
    zenoh_response = zenoh_response or {}
    print("\n[Payload Sizes: bytes]")
    print(f"{'metric':<28} {'HTTP':>12} {'Zenoh':>12}")
    print("-" * 56)
    http_response_bytes = "n/a"
    if http_response_text is not None:
        http_response_bytes = len(http_response_text.encode("utf-8"))
    print(
        f"{'request_payload':<28} {http_request_bytes if http_request_bytes is not None else 'n/a':>12} {zenoh_response.get('_request_payload_bytes', 'n/a'):>12}"
    )
    print(
        f"{'response_payload':<28} {http_response_bytes:>12} {zenoh_response.get('_reply_payload_bytes', 'n/a'):>12}"
    )


def summarize_result(name, ok, text, elapsed):
    print(f"\n[{name}] ok={ok} elapsed={elapsed:.3f}s")
    print(text)


def extract_timing_row(result, elapsed, transport):
    result = result or {}
    timing = result.get("_timing", {})
    server_total = timing.get("server_total_time")
    inference = timing.get("inference_time")
    if transport == "http":
        request_decode = timing.get("request_read_time")
        client_encode = None
        payload_bytes = None
        reply_bytes = None
    else:
        request_decode = timing.get("payload_decode_time")
        client_encode = result.get("_client_encode_time")
        payload_bytes = result.get("_request_payload_bytes")
        reply_bytes = result.get("_reply_payload_bytes")
    return {
        "elapsed": elapsed,
        "server_total": server_total,
        "inference": inference,
        "server_core": timing.get("server_core_time"),
        "image_depth_decode": timing.get("image_depth_decode_time"),
        "request_decode": request_decode,
        "client_encode": client_encode,
        "comm_overhead": max(0.0, elapsed - server_total) if server_total is not None else None,
        "payload_bytes": payload_bytes,
        "reply_bytes": reply_bytes,
    }


def print_repeat_summary(name, rows):
    if len(rows) <= 1:
        return
    metrics = [
        ("end_to_end_total", "elapsed"),
        ("server_total", "server_total"),
        ("inference", "inference"),
        ("estimated_comm_overhead", "comm_overhead"),
        ("image_depth_decode", "image_depth_decode"),
        ("request_decode", "request_decode"),
        ("client_encode", "client_encode"),
    ]
    print(f"\n[{name} Repeat Summary: ms]")
    print(f"{'metric':<30} {'mean':>10} {'median':>10} {'std':>10} {'p90':>10} {'min':>10} {'max':>10}")
    print("-" * 96)
    for label, key in metrics:
        values = [row[key] for row in rows if row.get(key) is not None]
        if not values:
            continue
        print(
            f"{label:<30} "
            f"{statistics.mean(values) * 1000:>10.3f} "
            f"{statistics.median(values) * 1000:>10.3f} "
            f"{(statistics.stdev(values) * 1000 if len(values) > 1 else 0.0):>10.3f} "
            f"{percentile(values, 90) * 1000:>10.3f} "
            f"{min(values) * 1000:>10.3f} "
            f"{max(values) * 1000:>10.3f}"
        )

    slowest = max(range(len(rows)), key=lambda i: rows[i]["elapsed"])
    fastest = min(range(len(rows)), key=lambda i: rows[i]["elapsed"])
    print(f"fastest_request_index: {fastest + 1}, elapsed_ms: {rows[fastest]['elapsed'] * 1000:.3f}")
    print(f"slowest_request_index: {slowest + 1}, elapsed_ms: {rows[slowest]['elapsed'] * 1000:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Compare InternVLA HTTP and Zenoh request/reply on the same frame.")
    add_zenoh_config_args(parser)
    parser.set_defaults(key=env_default_key())
    parser.add_argument(
        "--transport",
        choices=["http", "zenoh", "both"],
        default="both",
        help="Run only one transport to avoid two model servers competing for GPU resources.",
    )
    parser.add_argument("--http_url", default="http://127.0.0.1:8848/eval_dual")
    parser.add_argument("--image", default="/data/users/chris/InternNav/assets/realworld_sample_data1/debug_raw_0000.jpg")
    parser.add_argument("--depth_m", type=float, default=1.0)
    parser.add_argument("--instruction", default="")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--no_reset", action="store_true", help="Do not send reset=True to either service.")
    parser.add_argument("--repeat", type=int, default=1, help="Repeat requests in one process and reuse the same session.")
    parser.add_argument("--interval", type=float, default=0.0, help="Sleep seconds between repeated requests.")
    parser.add_argument("--reset_each", action="store_true", help="Send reset=True on every repeated request.")
    parser.add_argument(
        "--http_reuse_session",
        action="store_true",
        help="Reuse one requests.Session for HTTP. Default keeps production-like behavior: one session per request.",
    )
    args = parser.parse_args()

    image, image_bytes = make_rgb_jpeg(args.image)
    depth_bytes = make_depth_png(image.width, image.height, args.depth_m)
    base_payload = {"instruction": args.instruction}

    http_text = None
    http_result = None
    http_elapsed = None
    http_rows = []
    if args.transport in ("http", "both"):
        http_session = None
        if args.http_reuse_session:
            http_session = requests.Session()
            http_session.trust_env = False
        try:
            for request_idx in range(args.repeat):
                payload = dict(base_payload)
                payload["idx"] = request_idx
                payload["reset"] = args.reset_each or (request_idx == 0 and not args.no_reset)
                http_status, http_text, http_elapsed = request_http(
                    args.http_url, image_bytes, depth_bytes, payload, args.timeout, session=http_session
                )
                http_result = json.loads(http_text)
                http_rows.append(extract_timing_row(http_result, http_elapsed, "http"))
                summarize_result(f"HTTP #{request_idx + 1}", 200 <= http_status < 300, http_text, http_elapsed)
                if args.interval and request_idx != args.repeat - 1:
                    time.sleep(args.interval)
        finally:
            if http_session is not None:
                http_session.close()

    zenoh_response = None
    zenoh_elapsed = None
    zenoh_rows = []
    if args.transport in ("zenoh", "both"):
        zenoh = import_zenoh_or_raise()
        zenoh.init_log_from_env_or("error")
        client = ZenohInternVLAClient.from_args(args, zenoh)
        try:
            for request_idx in range(args.repeat):
                payload = dict(base_payload)
                payload["idx"] = request_idx
                payload["reset"] = args.reset_each or (request_idx == 0 and not args.no_reset)
                started_at = time.perf_counter()
                zenoh_response = client.eval_dual(image_bytes, depth_bytes, payload)
                zenoh_elapsed = time.perf_counter() - started_at
                zenoh_rows.append(extract_timing_row(zenoh_response, zenoh_elapsed, "zenoh"))
                summarize_result(
                    f"Zenoh #{request_idx + 1}",
                    zenoh_response.get("ok", False),
                    json.dumps(zenoh_response, ensure_ascii=False),
                    zenoh_elapsed,
                )
                if args.interval and request_idx != args.repeat - 1:
                    time.sleep(args.interval)
        finally:
            client.close()

    print_timing_table(http_elapsed, http_result, zenoh_elapsed, zenoh_response)
    print_size_table(payload_bytes_http(image_bytes, depth_bytes, base_payload), http_text, zenoh_response)
    print_repeat_summary("HTTP", http_rows)
    print_repeat_summary("Zenoh", zenoh_rows)

    if args.transport == "both":
        zenoh_result = zenoh_response.get("result", {})
        print("\n[Schema]")
        print(f"HTTP keys:  {sorted(k for k in http_result.keys() if k != '_timing')}")
        print(f"Zenoh keys: {sorted(zenoh_result.keys())}")


if __name__ == "__main__":
    main()
