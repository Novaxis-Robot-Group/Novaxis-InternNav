import json
import os
import re
import struct
import time


DEFAULT_KEY = "internvla/eval_dual"
DEFAULT_UDP_PORT = 7447
DEFAULT_UDP_SERVER_LISTEN = f"udp/0.0.0.0:{DEFAULT_UDP_PORT}"
DEFAULT_UDP_LOCAL_CONNECT = f"udp/127.0.0.1:{DEFAULT_UDP_PORT}"
PAYLOAD_MAGIC = b"IVLAZ1\0\0"
PAYLOAD_HEADER_STRUCT = struct.Struct("!Q")


def add_zenoh_config_args(parser):
    parser.add_argument("--key", default=DEFAULT_KEY, help="Zenoh key expression used for InternVLA request/reply.")
    parser.add_argument("--mode", "-m", choices=["peer", "client"], help="Zenoh session mode.")
    parser.add_argument(
        "--connect",
        "-e",
        action="append",
        help="Zenoh endpoints to connect to, e.g. udp/192.168.8.251:7447 or tcp/192.168.8.251:7447.",
    )
    parser.add_argument("--listen", "-l", action="append", help="Zenoh endpoints to listen on, e.g. udp/0.0.0.0:7447.")
    parser.add_argument("--config", "-c", help="Zenoh configuration file.")
    parser.add_argument("--no-multicast-scouting", action="store_true", help="Disable Zenoh multicast scouting.")
    parser.add_argument(
        "--cfg",
        action="append",
        default=[],
        help="Arbitrary Zenoh config entry as KEY:VALUE, matching the official zenoh-python examples.",
    )
    return parser


def normalize_endpoint_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        endpoints = value
    else:
        endpoints = [value]
    return [str(endpoint).strip() for endpoint in endpoints if str(endpoint).strip()]


def validate_zenoh_endpoint(endpoint):
    endpoint = str(endpoint).strip()
    if not endpoint:
        raise ValueError("Zenoh endpoint cannot be empty.")
    if "http://" in endpoint or "https://" in endpoint or "[" in endpoint or "]" in endpoint or "(" in endpoint or ")" in endpoint:
        raise ValueError(
            "Invalid Zenoh endpoint. Use a raw Zenoh endpoint like 'udp/192.168.8.251:7447', "
            f"not a Markdown/HTTP link: {endpoint!r}"
        )
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*/[^/].+", endpoint):
        raise ValueError(
            "Invalid Zenoh endpoint format. Expected something like 'udp/192.168.8.251:7447' "
            f"or 'udp/0.0.0.0:7447', got: {endpoint!r}"
        )
    return endpoint


def validate_zenoh_args(args):
    args.connect = [validate_zenoh_endpoint(endpoint) for endpoint in normalize_endpoint_list(getattr(args, "connect", None))] or None
    args.listen = [validate_zenoh_endpoint(endpoint) for endpoint in normalize_endpoint_list(getattr(args, "listen", None))] or None
    key = str(getattr(args, "key", "") or "").strip()
    if not key:
        raise ValueError("Zenoh key cannot be empty.")
    if key.startswith("/") or " " in key:
        raise ValueError(f"Suspicious Zenoh key {key!r}. Expected a key like 'internvla/eval_dual'.")
    args.key = key
    return args


def zenoh_args_summary(args):
    return {
        "key": getattr(args, "key", None),
        "mode": getattr(args, "mode", None) or "default",
        "connect": getattr(args, "connect", None) or [],
        "listen": getattr(args, "listen", None) or [],
        "no_multicast_scouting": bool(getattr(args, "no_multicast_scouting", False)),
        "config": getattr(args, "config", None),
    }


def make_zenoh_config(args, zenoh):
    args = validate_zenoh_args(args)
    conf = zenoh.Config.from_file(args.config) if getattr(args, "config", None) else zenoh.Config()
    if getattr(args, "mode", None) is not None:
        conf.insert_json5("mode", json.dumps(args.mode))
    if getattr(args, "connect", None) is not None:
        conf.insert_json5("connect/endpoints", json.dumps(args.connect))
    if getattr(args, "listen", None) is not None:
        conf.insert_json5("listen/endpoints", json.dumps(args.listen))
    if getattr(args, "no_multicast_scouting", False):
        conf.insert_json5("scouting/multicast/enabled", json.dumps(False))
    for item in getattr(args, "cfg", []):
        key, value = item.split(":", 1)
        conf.insert_json5(key, value)
    return conf


def encode_eval_request(image_bytes, depth_bytes, data):
    header = json.dumps(
        {
            "data": data,
            "image_format": "jpeg",
            "depth_format": "png_uint16_m_x10000",
            "image_bytes": len(image_bytes),
            "depth_bytes": len(depth_bytes),
        },
        separators=(",", ":"),
    ).encode("utf-8")
    return b"".join([PAYLOAD_MAGIC, PAYLOAD_HEADER_STRUCT.pack(len(header)), header, image_bytes, depth_bytes])


def decode_eval_request(payload_bytes):
    if payload_bytes.startswith(PAYLOAD_MAGIC):
        offset = len(PAYLOAD_MAGIC)
        header_len = PAYLOAD_HEADER_STRUCT.unpack(payload_bytes[offset : offset + PAYLOAD_HEADER_STRUCT.size])[0]
        offset += PAYLOAD_HEADER_STRUCT.size
        header = json.loads(payload_bytes[offset : offset + header_len].decode("utf-8"))
        offset += header_len
        image_len = int(header["image_bytes"])
        depth_len = int(header["depth_bytes"])
        image_bytes = payload_bytes[offset : offset + image_len]
        offset += image_len
        depth_bytes = payload_bytes[offset : offset + depth_len]
        return image_bytes, depth_bytes, header.get("data", {})

    import base64

    payload = json.loads(payload_bytes.decode("utf-8"))
    image_bytes = base64.b64decode(payload["image_b64"])
    depth_bytes = base64.b64decode(payload["depth_b64"])
    return image_bytes, depth_bytes, payload.get("data", {})


def payload_to_bytes(payload):
    if payload is None:
        return b""
    if hasattr(payload, "to_bytes"):
        return bytes(payload.to_bytes())
    if hasattr(payload, "to_string"):
        return payload.to_string().encode("utf-8")
    if isinstance(payload, bytes):
        return payload
    if isinstance(payload, bytearray):
        return bytes(payload)
    return bytes(payload)


def reply_to_payload_bytes(reply):
    if hasattr(reply, "ok") and reply.ok is not None:
        return payload_to_bytes(reply.ok.payload)
    if hasattr(reply, "err") and reply.err is not None:
        raise RuntimeError(payload_to_bytes(reply.err.payload).decode("utf-8", errors="replace"))
    raise RuntimeError(f"Unsupported Zenoh reply object: {reply!r}")


class ZenohInternVLAClient:
    def __init__(self, session, key=DEFAULT_KEY, timeout=180.0):
        self.session = session
        self.key = key
        self.timeout = timeout

    @classmethod
    def from_args(cls, args, zenoh):
        conf = make_zenoh_config(args, zenoh)
        session = zenoh.open(conf)
        return cls(session=session, key=args.key, timeout=getattr(args, "timeout", 180.0))

    def close(self):
        close = getattr(self.session, "close", None)
        if close is not None:
            close()

    def eval_dual(self, image_bytes, depth_bytes, data):
        total_started_at = time.perf_counter()
        encode_started_at = time.perf_counter()
        request_payload = encode_eval_request(image_bytes, depth_bytes, data)
        encode_time = time.perf_counter() - encode_started_at
        target = getattr(__import__("zenoh"), "QueryTarget", None)
        target = target.BEST_MATCHING if target is not None else None
        kwargs = {"payload": request_payload, "timeout": self.timeout}
        if target is not None:
            kwargs["target"] = target
        replies = self.session.get(self.key, **kwargs)
        for reply in replies:
            response_payload = reply_to_payload_bytes(reply)
            response = json.loads(response_payload.decode("utf-8"))
            response["_transport_time"] = time.perf_counter() - total_started_at
            response["_client_encode_time"] = encode_time
            response["_request_payload_bytes"] = len(request_payload)
            response["_reply_payload_bytes"] = len(response_payload)
            return response
        raise TimeoutError(f"No Zenoh reply received for key '{self.key}' within {self.timeout}s.")


def import_zenoh_or_raise():
    try:
        import zenoh
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Python package 'zenoh' is not installed in this environment. "
            "Install it in the active env first, for example: pip install eclipse-zenoh"
        ) from exc
    return zenoh


def env_default_key():
    return os.environ.get("INTERNVLA_ZENOH_KEY", DEFAULT_KEY)
