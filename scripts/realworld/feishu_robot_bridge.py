import json
import os
import re
from typing import Any, Dict, Optional

import requests


def build_reply_text(command: Dict[str, Any], result: Dict[str, Any]) -> str:
    action = command.get("action", "unknown")
    if action == "move":
        return f"已收到指令：{action}，距离 {command.get('distance', 'unknown')} 米。"
    if action == "turn":
        return f"已收到指令：{action}，方向 {command.get('direction', 'unknown')}，角度 {command.get('angle', 'unknown')} 度。"
    if action == "stop":
        return "已收到停止指令。"
    return f"已收到指令：{action}。"


class RobotCommandDispatcher:
    def __init__(self, endpoint: Optional[str] = None):
        self.endpoint = endpoint or os.environ.get("ROBOT_COMMAND_ENDPOINT")

    def dispatch(self, command: Dict[str, Any]) -> Dict[str, Any]:
        if not self.endpoint:
            return {"ok": True, "command": command, "note": "No endpoint configured; dry run."}

        payload = {"command": command.get("action"), **command}
        response = requests.post(self.endpoint, json=payload, timeout=10)
        response.raise_for_status()
        try:
            body = response.json()
        except ValueError:
            body = {"raw": response.text}
        return {"ok": True, "remote": body, "command": command}


def parse_robot_command(text: str) -> Dict[str, Any]:
    text = (text or "").strip().lower()
    if not text:
        return {"action": "stop"}

    if re.search(r"(前进|forward|move|go)", text):
        distance_match = re.search(r"([0-9.]+)\s*(米|m)", text)
        distance = float(distance_match.group(1)) if distance_match else 0.5
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


class FeishuRobotBridge:
    def __init__(self, dispatcher: Optional[RobotCommandDispatcher] = None):
        self.dispatcher = dispatcher or RobotCommandDispatcher()
        self.feishu_app_id = os.environ.get("FEISHU_APP_ID")
        self.feishu_app_secret = os.environ.get("FEISHU_APP_SECRET")
        self.feishu_bot_url = os.environ.get("FEISHU_BOT_URL")

    def send_reply(self, chat_id: str, message_id: str, text: str) -> Dict[str, Any]:
        bot_url = self.feishu_bot_url or os.environ.get("FEISHU_BOT_URL")
        if not bot_url:
            return {"ok": True, "note": "Feishu bot URL not configured; skipped reply."}
        payload = {
            "chat_id": chat_id,
            "msg_type": "text",
            "content": {"text": text},
            "reply_message_id": message_id,
        }
        response = requests.post(self.feishu_bot_url, json=payload, timeout=10)
        response.raise_for_status()
        try:
            body = response.json()
        except ValueError:
            body = {"raw": response.text}
        return {"ok": True, "reply": body}

    def handle_event(self, payload: Dict[str, Any]) -> Dict[str, Any]:
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

        command = parse_robot_command(text)
        result = self.dispatcher.dispatch(command)
        if result.get("note") == "No endpoint configured; dry run.":
            result["mock"] = True
        reply_text = build_reply_text(command, result)
        chat_id = message.get("chat_id") or payload.get("chat_id")
        message_id = message.get("message_id") or payload.get("message_id")
        if chat_id and message_id:
            reply_result = self.send_reply(chat_id, message_id, reply_text)
            result["reply"] = reply_result
        return {"ok": True, "command": command, **result}


if __name__ == "__main__":
    bridge = FeishuRobotBridge()
    print(json.dumps(bridge.handle_event({"event": {"message": {"content": "前进 0.5 米"}}}), ensure_ascii=False))
