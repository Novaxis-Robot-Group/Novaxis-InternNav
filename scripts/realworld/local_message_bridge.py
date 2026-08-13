import argparse
import json
import os
import re
import sys
from typing import Any, Dict, Optional

import requests
from flask import Flask, jsonify, request


app = Flask(__name__)


class LocalMessageBridge:
    def __init__(self, target_url: Optional[str] = None, reply_url: Optional[str] = None):
        self.target_url = target_url or os.environ.get("BRIDGE_TARGET_URL")
        self.reply_url = reply_url or os.environ.get("BRIDGE_REPLY_URL")

    def parse_command(self, text: str) -> Dict[str, Any]:
        text = (text or "").strip().lower()
        if not text:
            return {"action": "stop"}

        if re.search(r"(前进|forward|move|go)", text):
            dist_match = re.search(r"([0-9.]+)\s*(米|m)", text)
            distance = float(dist_match.group(1)) if dist_match else 0.5
            return {"action": "move", "direction": "forward", "distance": distance}
        if re.search(r"(左转|left|turn left)", text):
            angle_match = re.search(r"([0-9.]+)\s*(度|deg)", text)
            angle = float(angle_match.group(1)) if angle_match else 90.0
            return {"action": "turn", "direction": "left", "angle": angle}
        if re.search(r"(右转|right|turn right)", text):
            angle_match = re.search(r"([0-9.]+)\s*(度|deg)", text)
            angle = float(angle_match.group(1)) if angle_match else 90.0
            return {"action": "turn", "direction": "right", "angle": angle}
        if re.search(r"(停|stop|停止)", text):
            return {"action": "stop"}
        return {"action": "unknown", "text": text}

    def build_reply_text(self, command: Dict[str, Any], result: Dict[str, Any]) -> str:
        action = command.get("action", "unknown")
        if action == "move":
            return f"已收到：{action}，距离 {command.get('distance', 'unknown')} 米。"
        if action == "turn":
            return f"已收到：{action}，方向 {command.get('direction', 'unknown')}，角度 {command.get('angle', 'unknown')} 度。"
        if action == "stop":
            return "已收到停止指令。"
        return f"已收到：{action}。"

    def forward_to_target(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.target_url:
            return {"ok": True, "note": "No target URL configured; dry run."}
        response = requests.post(self.target_url, json=payload, timeout=10)
        response.raise_for_status()
        try:
            return response.json()
        except ValueError:
            return {"raw": response.text}

    def send_reply(self, payload: Dict[str, Any], text: str) -> Dict[str, Any]:
        if not self.reply_url:
            return {"ok": True, "note": "No reply URL configured; skipped reply."}
        reply_payload = {"text": text, **payload}
        response = requests.post(self.reply_url, json=reply_payload, timeout=10)
        response.raise_for_status()
        try:
            return response.json()
        except ValueError:
            return {"raw": response.text}

    def handle_message(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        event = payload.get("event", {})
        message = event.get("message", {})
        content = message.get("content", "")
        if isinstance(content, str):
            text = content
        else:
            text = json.dumps(content, ensure_ascii=False)

        if isinstance(text, str) and text.startswith("{"):
            try:
                parsed = json.loads(text)
            except ValueError:
                parsed = {"text": text}
            text = parsed.get("text", text)

        command = self.parse_command(text)
        forward_result = self.forward_to_target({"command": command, "text": text, "message": message})
        reply_text = self.build_reply_text(command, forward_result)
        reply_result = self.send_reply({"command": command, "message": message}, reply_text)
        return {"ok": True, "command": command, "forward": forward_result, "reply": reply_result}


bridge = LocalMessageBridge()


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "service": "local_message_bridge"})


@app.route("/inbound", methods=["POST"])
def inbound():
    payload = request.get_json(silent=True) or {}
    return jsonify(bridge.handle_message(payload))


@app.route("/test", methods=["POST"])
def test_handler():
    payload = request.get_json(silent=True) or {}
    return jsonify({"ok": True, "received": payload})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local bridge that receives messages and forwards them")
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "9002")))
    parser.add_argument("--target-url", default=os.environ.get("BRIDGE_TARGET_URL"))
    parser.add_argument("--reply-url", default=os.environ.get("BRIDGE_REPLY_URL"))
    args = parser.parse_args()

    bridge.target_url = args.target_url
    bridge.reply_url = args.reply_url
    app.run(host=args.host, port=args.port, debug=False)
