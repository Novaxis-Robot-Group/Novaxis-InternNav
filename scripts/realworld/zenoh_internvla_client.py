import argparse
import json
import os
import threading
import time

import rclpy

import http_internvla_client as base_client
from zenoh_internvla_rpc import (
    ZenohInternVLAClient,
    add_zenoh_config_args,
    env_default_key,
    import_zenoh_or_raise,
    validate_zenoh_args,
    zenoh_args_summary,
)


zenoh_client = None


def payload_bytes(value):
    """Accept both legacy BytesIO payloads and current Orbbec raw bytes."""
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    return value.getvalue()


def dual_sys_eval_zenoh(image_bytes, depth_bytes, front_image_bytes=None, url=None):
    global zenoh_client

    data = {"reset": base_client.policy_init, "idx": base_client.http_idx}
    instruction = os.environ.get("INTERNVLA_INSTRUCTION")
    if instruction:
        data["instruction"] = instruction

    base_client.policy_init = False
    started_at = time.time()
    response = zenoh_client.eval_dual(payload_bytes(image_bytes), payload_bytes(depth_bytes), data)
    if not response.get("ok", False):
        raise RuntimeError(response.get("error", "Zenoh InternVLA server returned an error."))

    base_client.http_idx += 1
    if base_client.http_idx == 0:
        base_client.first_running_time = time.time()
    print(f"idx: {base_client.http_idx} after zenoh {time.time() - started_at}")
    print(f"response {json.dumps(response, ensure_ascii=False)}")
    return response["result"]


def main():
    global zenoh_client

    base_client.acquire_client_lock()
    zenoh = import_zenoh_or_raise()
    zenoh.init_log_from_env_or("error")

    parser = argparse.ArgumentParser(description="Go2 real-world client using Zenoh RPC for InternVLA inference.")
    add_zenoh_config_args(parser)
    parser.set_defaults(key=env_default_key())
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()
    args = validate_zenoh_args(args)
    print(f"Zenoh client config: {json.dumps(zenoh_args_summary(args), ensure_ascii=False)}")

    zenoh_client = ZenohInternVLAClient.from_args(args, zenoh)
    base_client.dual_sys_eval = dual_sys_eval_zenoh

    control_thread_instance = threading.Thread(target=base_client.control_thread)
    planning_thread_instance = threading.Thread(target=base_client.planning_thread)
    control_thread_instance.daemon = True
    planning_thread_instance.daemon = True
    rclpy.init()

    try:
        base_client.manager = base_client.Go2Manager()
        control_thread_instance.start()
        planning_thread_instance.start()
        rclpy.spin(base_client.manager)
    except KeyboardInterrupt:
        pass
    finally:
        if base_client.manager is not None:
            base_client.manager.shutdown()
            base_client.manager.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if zenoh_client is not None:
            zenoh_client.close()


if __name__ == "__main__":
    main()
