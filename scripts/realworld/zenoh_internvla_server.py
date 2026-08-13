import argparse
import json
import time
import traceback

import internvla_service_core as service_core
from zenoh_internvla_rpc import (
    add_zenoh_config_args,
    decode_eval_request,
    import_zenoh_or_raise,
    make_zenoh_config,
    payload_to_bytes,
    validate_zenoh_args,
    zenoh_args_summary,
)


def main():
    zenoh = import_zenoh_or_raise()
    zenoh.init_log_from_env_or("error")

    parser = argparse.ArgumentParser(description="InternVLA-N1 real-world Zenoh queryable server.")
    service_core.add_common_model_args(parser)
    add_zenoh_config_args(parser)
    parser.add_argument("--complete", action="store_true", help="Declare the queryable complete for this key.")
    args = parser.parse_args()
    args = validate_zenoh_args(args)
    print(f"Zenoh server config: {json.dumps(zenoh_args_summary(args), ensure_ascii=False)}")

    service_core.init_service(args)
    conf = make_zenoh_config(args, zenoh)

    print("Opening Zenoh session...")
    with zenoh.open(conf) as session:
        print(f"Declaring InternVLA queryable on '{args.key}'...")
        queryable = session.declare_queryable(args.key, complete=args.complete)
        print("Zenoh InternVLA server is ready. Press CTRL-C to quit.")

        while True:
            with queryable.recv() as query:
                request_started_at = time.perf_counter()
                try:
                    payload_started_at = time.perf_counter()
                    request_payload = payload_to_bytes(query.payload)
                    payload_to_bytes_time = time.perf_counter() - payload_started_at
                    decode_payload_started_at = time.perf_counter()
                    image_bytes, depth_bytes, data = decode_eval_request(request_payload)
                    payload_decode_time = time.perf_counter() - decode_payload_started_at
                    image, depth, image_decode_time = service_core.decode_rgb_depth(image_bytes, depth_bytes)
                    print(f"read zenoh data cost {time.perf_counter() - request_started_at}")
                    core_started_at = time.perf_counter()
                    json_output, timing = service_core.run_dual_inference(image, depth, data)
                    server_total_time = time.perf_counter() - request_started_at
                    timing.update(
                        {
                            "transport": "zenoh",
                            "payload_to_bytes_time": payload_to_bytes_time,
                            "payload_decode_time": payload_decode_time,
                            "image_depth_decode_time": image_decode_time,
                            "server_core_time": time.perf_counter() - core_started_at,
                            "server_total_time": server_total_time,
                            "request_payload_bytes": len(request_payload),
                            "request_image_bytes": len(image_bytes),
                            "request_depth_bytes": len(depth_bytes),
                        }
                    )
                    json_output["_timing"] = timing
                    service_core.update_saved_timing(timing)
                    reply_payload = json.dumps(
                        {
                            "ok": True,
                            "result": json_output,
                            "generate_time": timing["inference_time"],
                            "_timing": timing,
                        },
                        separators=(",", ":"),
                    )
                except Exception as exc:
                    traceback.print_exc()
                    reply_payload = json.dumps(
                        {
                            "ok": False,
                            "error": str(exc),
                        },
                        separators=(",", ":"),
                    )
                query.reply(args.key, reply_payload)


if __name__ == "__main__":
    main()
