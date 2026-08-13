import argparse
import io
import json
import logging
import statistics
import threading
import time
from pathlib import Path

import numpy as np
import requests
from flask import Flask, jsonify, request
from PIL import Image
from werkzeug.serving import make_server

from zenoh_internvla_rpc import (
    ZenohInternVLAClient,
    add_zenoh_config_args,
    decode_eval_request,
    encode_eval_request,
    import_zenoh_or_raise,
    make_zenoh_config,
    payload_to_bytes,
    DEFAULT_UDP_LOCAL_CONNECT,
)


def make_rgb_depth_payload(image_path, depth_m):
    image = Image.open(Path(image_path)).convert("RGB")
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="JPEG")

    depth = np.full((image.height, image.width), int(depth_m * 10000.0), dtype=np.uint16)
    depth_bytes = io.BytesIO()
    Image.fromarray(depth).save(depth_bytes, format="PNG")
    return image_bytes.getvalue(), depth_bytes.getvalue()


def percentile(values, pct):
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(round((pct / 100.0) * (len(values) - 1)))))
    return values[idx]


def summarize(name, latencies, request_bytes, response_bytes):
    print(f"\n[{name}]")
    print(f"requests: {len(latencies)}")
    print(f"request_bytes: {request_bytes}")
    print(f"response_bytes: {response_bytes}")
    print(f"mean_ms: {statistics.mean(latencies) * 1000:.3f}")
    print(f"median_ms: {statistics.median(latencies) * 1000:.3f}")
    print(f"p90_ms: {percentile(latencies, 90) * 1000:.3f}")
    print(f"p99_ms: {percentile(latencies, 99) * 1000:.3f}")
    print(f"min_ms: {min(latencies) * 1000:.3f}")
    print(f"max_ms: {max(latencies) * 1000:.3f}")


class HttpEchoServer:
    def __init__(self, host, port):
        logging.getLogger("werkzeug").setLevel(logging.ERROR)
        app = Flask("internvla_http_echo")

        @app.route("/echo", methods=["POST"])
        def echo():
            image_file = request.files.get("image")
            depth_file = request.files.get("depth")
            json_data = request.form.get("json", "{}")
            if image_file is None or depth_file is None:
                return jsonify({"ok": False, "error": "image and depth are required"}), 400
            image_bytes = image_file.stream.read()
            depth_bytes = depth_file.stream.read()
            data = json.loads(json_data)
            return jsonify(
                {
                    "ok": True,
                    "image_bytes": len(image_bytes),
                    "depth_bytes": len(depth_bytes),
                    "idx": data.get("idx"),
                }
            )

        self.server = make_server(host, port, app)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.server.shutdown()
        self.thread.join(timeout=5)


class ZenohEchoServer:
    def __init__(self, zenoh, args):
        self.key = args.key
        conf = make_zenoh_config(args, zenoh)
        self.session = zenoh.open(conf)
        self.queryable = self.session.declare_queryable(args.key)
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.stopped = threading.Event()

    def _serve(self):
        while not self.stopped.is_set():
            try:
                with self.queryable.recv() as query:
                    payload = payload_to_bytes(query.payload)
                    image_bytes, depth_bytes, data = decode_eval_request(payload)
                    reply = json.dumps(
                        {
                            "ok": True,
                            "image_bytes": len(image_bytes),
                            "depth_bytes": len(depth_bytes),
                            "idx": data.get("idx"),
                        },
                        separators=(",", ":"),
                    )
                    query.reply(self.key, reply)
            except Exception:
                if not self.stopped.is_set():
                    raise

    def start(self):
        self.thread.start()

    def stop(self):
        self.stopped.set()
        self.session.close()
        self.thread.join(timeout=5)


def bench_http(url, image_bytes, depth_bytes, n, warmup):
    session = requests.Session()
    session.trust_env = False
    latencies = []
    last_response = ""
    for i in range(n + warmup):
        files = {
            "image": ("rgb.jpg", io.BytesIO(image_bytes), "image/jpeg"),
            "depth": ("depth.png", io.BytesIO(depth_bytes), "image/png"),
        }
        payload = {"idx": i, "reset": i == 0}
        started_at = time.perf_counter()
        response = session.post(url, files=files, data={"json": json.dumps(payload)}, timeout=30)
        elapsed = time.perf_counter() - started_at
        response.raise_for_status()
        last_response = response.text
        if i >= warmup:
            latencies.append(elapsed)
    return latencies, len(last_response.encode("utf-8"))


def bench_zenoh(client, image_bytes, depth_bytes, n, warmup):
    latencies = []
    last_response = {}
    for i in range(n + warmup):
        payload = {"idx": i, "reset": i == 0}
        started_at = time.perf_counter()
        response = client.eval_dual(image_bytes, depth_bytes, payload)
        elapsed = time.perf_counter() - started_at
        last_response = response
        if i >= warmup:
            latencies.append(elapsed)
    return latencies, len(json.dumps(last_response, separators=(",", ":")).encode("utf-8"))


def main():
    parser = argparse.ArgumentParser(description="Benchmark pure HTTP vs Zenoh transport without loading InternVLA.")
    parser.add_argument("--image", default="/data/users/chris/InternNav/assets/realworld_sample_data1/debug_raw_0000.jpg")
    parser.add_argument("--depth_m", type=float, default=1.0)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--http_host", default="127.0.0.1")
    parser.add_argument("--http_port", type=int, default=18848)
    add_zenoh_config_args(parser)
    parser.set_defaults(
        key="internvla/bench/echo",
        mode="peer",
        connect=[DEFAULT_UDP_LOCAL_CONNECT.replace(":7447", ":17447")],
        no_multicast_scouting=True,
    )
    parser.add_argument("--zenoh_listen", default=DEFAULT_UDP_LOCAL_CONNECT.replace(":7447", ":17447"))
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    image_bytes, depth_bytes = make_rgb_depth_payload(args.image, args.depth_m)
    print(f"rgb_jpeg_bytes: {len(image_bytes)}")
    print(f"depth_png_bytes: {len(depth_bytes)}")

    http_server = HttpEchoServer(args.http_host, args.http_port)
    http_server.start()
    time.sleep(0.2)
    http_url = f"http://{args.http_host}:{args.http_port}/echo"
    http_latencies, http_response_bytes = bench_http(http_url, image_bytes, depth_bytes, args.requests, args.warmup)
    http_request_bytes = len(image_bytes) + len(depth_bytes)
    summarize("HTTP multipart echo", http_latencies, http_request_bytes, http_response_bytes)
    http_server.stop()

    zenoh = import_zenoh_or_raise()
    zenoh.init_log_from_env_or("error")
    server_args = argparse.Namespace(**vars(args))
    server_args.mode = "peer"
    server_args.connect = None
    server_args.listen = [args.zenoh_listen]
    zenoh_server = ZenohEchoServer(zenoh, server_args)
    zenoh_server.start()
    time.sleep(0.5)

    client = ZenohInternVLAClient.from_args(args, zenoh)
    try:
        zenoh_latencies, zenoh_response_bytes = bench_zenoh(client, image_bytes, depth_bytes, args.requests, args.warmup)
    finally:
        client.close()
        zenoh_server.stop()
    zenoh_request_bytes = len(encode_eval_request(image_bytes, depth_bytes, {"idx": 0, "reset": True}))
    summarize("Zenoh query echo", zenoh_latencies, zenoh_request_bytes, zenoh_response_bytes)


if __name__ == "__main__":
    main()
