#!/usr/bin/env python3
"""Pulse-style TCP network benchmark for the InternVLA real-world pipeline.

This is a transport benchmark, not an InternVLA inference client.  Run its
receiver on the H200 server and its sender on Thor/Go2.  Each packet is sent as
one complete "inference request" and acknowledged by the receiver, so the
reported RTT contains TCP transport plus receiver scheduling only.
"""

import argparse
import math
import socket
import statistics
import struct
import time
from dataclasses import dataclass


HEADER = struct.Struct("!Q")
ACK = b"OK"


@dataclass(frozen=True)
class Profile:
    name: str
    frequency_hz: float
    payload_bytes: int
    description: str


PROFILES = {
    # Current client: one 640x480 RGB-D frame, encoded as JPEG + PNG, and a
    # planning loop with DESIRED_TIME = 0.3 seconds.  39,226 B is a measured
    # sample request; use --payload-bytes or --payload-file for your own data.
    "internvla-current": Profile(
        "InternVLA-N1 current RGB-D request", 1.0 / 0.3, 39_226,
        "1 RGB-D request every 0.3 s; JPEG RGB + PNG depth",
    ),
    # Upper Agent runs on the H200 and reads frames already saved by the
    # low-level service.  It does not create a Thor-to-server video stream.
    "upper-agent": Profile(
        "Upper Agent (no extra edge video transport)", 1.0 / 7.0, 0,
        "server-local frame analysis; this profile intentionally sends no image payload",
    ),
    # Only for planning a future architecture.  It is NOT used by current
    # InternVLA: 4 * RGB8 640x480 plus 1 * uint16 depth 640x480 at 15 FPS.
    "future-4rgb-1depth-raw": Profile(
        "Future 4 RGB + 1 depth raw", 15.0, 4 * 640 * 480 * 3 + 640 * 480 * 2,
        "uncompressed 640x480 RGB8 x4 + uint16 depth x1",
    ),
}


def recvall(conn: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        piece = conn.recv(size - len(data))
        if not piece:
            raise ConnectionError("peer closed the connection")
        data.extend(piece)
    return bytes(data)


def percentile(values, p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(p * len(ordered)) - 1)
    return ordered[index]


def resolve_payload(args, profile: Profile) -> bytes:
    if args.payload_file:
        with open(args.payload_file, "rb") as file:
            payload = file.read()
        if not payload:
            raise ValueError("--payload-file is empty")
        return payload

    payload_bytes = args.payload_bytes if args.payload_bytes is not None else profile.payload_bytes
    return bytes(payload_bytes)


def run_receiver(args) -> None:
    received_bytes = 0
    received_requests = 0
    started_at = time.monotonic()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((args.host, args.port))
        listener.listen(1)
        print(f"[receiver] listening on {args.host}:{args.port}")
        conn, address = listener.accept()
        with conn:
            conn.settimeout(args.timeout)
            print(f"[receiver] sender connected: {address[0]}:{address[1]}")
            try:
                while True:
                    payload_size = HEADER.unpack(recvall(conn, HEADER.size))[0]
                    recvall(conn, payload_size)
                    conn.sendall(ACK)
                    received_requests += 1
                    received_bytes += payload_size
                    if received_requests % args.report_every == 0:
                        elapsed = max(time.monotonic() - started_at, 1e-6)
                        print(
                            f"[receiver] requests={received_requests} "
                            f"avg_rx={received_bytes * 8 / elapsed / 1e6:.2f} Mbps"
                        )
            except (ConnectionError, socket.timeout):
                pass
    elapsed = max(time.monotonic() - started_at, 1e-6)
    print(
        f"[receiver] finished: {received_requests} requests, "
        f"{received_bytes / 1e6:.2f} MB, avg_rx={received_bytes * 8 / elapsed / 1e6:.2f} Mbps"
    )


def run_sender(args, profile: Profile) -> None:
    payload = resolve_payload(args, profile)
    if not payload:
        print("[sender] Upper Agent profile uses no edge video transport; nothing to send.")
        return

    interval = 1.0 / args.frequency_hz
    total_steps = max(1, math.ceil(args.duration * args.frequency_hz))
    rtts_ms = []
    schedule_slips_ms = []
    sent_bytes = 0

    print("=" * 76)
    print(f"[Thor] profile: {profile.name}")
    print(f"[strategy] {profile.description}")
    print(f"[schedule] {args.frequency_hz:.3f} Hz | interval={interval:.3f} s | requests={total_steps}")
    print(f"[payload] {len(payload):,} B ({len(payload) / 1024:.1f} KiB) per request")
    print("=" * 76)

    with socket.create_connection((args.host, args.port), timeout=args.timeout) as conn:
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        conn.settimeout(args.timeout)
        started_at = time.monotonic()
        for step in range(1, total_steps + 1):
            scheduled_at = started_at + (step - 1) * interval
            now = time.monotonic()
            if now < scheduled_at:
                time.sleep(scheduled_at - now)
            schedule_slip_ms = max(0.0, (time.monotonic() - scheduled_at) * 1000.0)

            sent_at = time.monotonic()
            conn.sendall(HEADER.pack(len(payload)))
            conn.sendall(payload)
            if recvall(conn, len(ACK)) != ACK:
                raise RuntimeError("invalid receiver ACK")
            rtt_ms = (time.monotonic() - sent_at) * 1000.0

            rtts_ms.append(rtt_ms)
            schedule_slips_ms.append(schedule_slip_ms)
            sent_bytes += len(payload)
            print(
                f"[{profile.name}] request={step:>4}/{total_steps} "
                f"payload={len(payload) / 1024:>7.1f} KiB "
                f"rtt={rtt_ms:>7.2f} ms schedule_slip={schedule_slip_ms:>6.2f} ms"
            )

    elapsed = max(time.monotonic() - started_at, 1e-6)
    print("\n[summary]")
    print(f"  effective rate: {total_steps / elapsed:.3f} Hz")
    print(f"  average throughput: {sent_bytes * 8 / elapsed / 1e6:.3f} Mbps")
    print(
        f"  RTT ms: mean={statistics.mean(rtts_ms):.2f}, "
        f"p50={percentile(rtts_ms, 0.50):.2f}, p95={percentile(rtts_ms, 0.95):.2f}, "
        f"max={max(rtts_ms):.2f}"
    )
    print(
        f"  scheduling slip ms: p95={percentile(schedule_slips_ms, 0.95):.2f}, "
        f"max={max(schedule_slips_ms):.2f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("receiver", "sender"))
    parser.add_argument("--host", default="0.0.0.0", help="receiver bind address or receiver IP")
    parser.add_argument("--port", type=int, default=19090)
    parser.add_argument("--profile", choices=tuple(PROFILES), default="internvla-current")
    parser.add_argument("--frequency-hz", type=float, help="override profile request frequency")
    parser.add_argument("--payload-bytes", type=int, help="override profile packet size")
    parser.add_argument("--payload-file", help="repeat exact bytes from this file as a request payload")
    parser.add_argument("--duration", type=float, default=120.0, help="sender duration in seconds")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--report-every", type=int, default=20, help="receiver progress interval")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = PROFILES[args.profile]
    if args.frequency_hz is None:
        args.frequency_hz = profile.frequency_hz
    if args.frequency_hz <= 0:
        raise ValueError("--frequency-hz must be positive")
    if args.payload_bytes is not None and args.payload_bytes < 0:
        raise ValueError("--payload-bytes must be non-negative")
    if args.mode == "receiver":
        run_receiver(args)
    else:
        run_sender(args, profile)


if __name__ == "__main__":
    main()
