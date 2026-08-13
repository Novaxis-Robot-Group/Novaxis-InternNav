import argparse
import time

from flask import Flask, jsonify, request

import internvla_service_core as service_core


app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    return jsonify(service_core.service_info())


@app.route("/eval_dual", methods=["POST"])
def eval_dual():
    request_started_at = time.perf_counter()

    image_file = request.files.get("image")
    depth_file = request.files.get("depth")
    json_data = request.form.get("json", "{}")
    if image_file is None or depth_file is None:
        return jsonify({"error": "Both multipart files 'image' and 'depth' are required."}), 400

    data = service_core.parse_json_data(json_data)
    image_bytes = image_file.stream.read()
    depth_bytes = depth_file.stream.read()
    read_time = time.perf_counter() - request_started_at
    image, depth, decode_time = service_core.decode_rgb_depth(image_bytes, depth_bytes)
    print(f"read http data cost {read_time + decode_time}")

    core_started_at = time.perf_counter()
    json_output, timing = service_core.run_dual_inference(image, depth, data)
    server_total_time = time.perf_counter() - request_started_at
    timing.update(
        {
            "transport": "http",
            "request_read_time": read_time,
            "image_depth_decode_time": decode_time,
            "server_core_time": time.perf_counter() - core_started_at,
            "server_total_time": server_total_time,
            "request_image_bytes": len(image_bytes),
            "request_depth_bytes": len(depth_bytes),
        }
    )
    json_output["_timing"] = timing
    service_core.update_saved_timing(timing)
    return jsonify(json_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    service_core.add_common_model_args(parser)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8848)
    args = parser.parse_args()

    service_core.init_service(args)
    app.run(host=args.host, port=args.port)
