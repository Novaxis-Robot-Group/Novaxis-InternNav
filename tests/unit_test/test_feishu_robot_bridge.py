import json
from unittest.mock import patch

import pytest

from scripts.realworld.feishu_robot_bridge import (
    FeishuRobotBridge,
    RobotCommandDispatcher,
    build_reply_text,
    parse_robot_command,
)
from scripts.realworld.feishu_robot_server import app


def test_parse_robot_command_move_forward():
    cmd = parse_robot_command("前进 1 米")
    assert cmd["action"] == "move"
    assert cmd["direction"] == "forward"
    assert cmd["distance"] == 1.0


def test_parse_robot_command_turn_left():
    cmd = parse_robot_command("左转 90 度")
    assert cmd["action"] == "turn"
    assert cmd["direction"] == "left"
    assert cmd["angle"] == 90.0


def test_dispatcher_posts_to_robot_endpoint():
    dispatcher = RobotCommandDispatcher(endpoint="http://example.test/robot/command")
    with patch("requests.post") as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"ok": True}
        result = dispatcher.dispatch({"action": "stop"})

    assert result["ok"] is True
    payload = mock_post.call_args.kwargs["json"]
    assert payload["command"] == "stop"


def test_bridge_handles_feishu_text_event():
    dispatcher = RobotCommandDispatcher(endpoint=None)
    bridge = FeishuRobotBridge(dispatcher=dispatcher)
    payload = {
        "schema": "2.0",
        "header": {"event_id": "evt-1", "event_type": "im.message.receive_v1"},
        "event": {
            "message": {
                "message_id": "msg-1",
                "chat_id": "chat-1",
                "content": json.dumps({"text": "前进 0.5 米"}),
                "message_type": "text",
            }
        },
    }

    result = bridge.handle_event(payload)
    assert result["ok"] is True
    assert result["command"]["action"] == "move"
    assert result["command"]["distance"] == 0.5


def test_build_reply_text_includes_summary():
    text = build_reply_text({"action": "move", "distance": 0.5}, {"ok": True})
    assert "move" in text.lower()
    assert "0.5" in text


def test_mock_webhook_flow_without_robot_endpoint():
    client = app.test_client()
    response = client.post(
        "/feishu/webhook",
        json={
            "event": {
                "message": {
                    "chat_id": "chat-1",
                    "message_id": "msg-1",
                    "content": "前进 0.5 米",
                }
            }
        },
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    assert payload["mock"] is True


def test_send_reply_uses_feishu_api():
    bridge = FeishuRobotBridge(dispatcher=RobotCommandDispatcher(endpoint=None))
    bridge.feishu_bot_url = "https://example.test/feishu"
    with patch("requests.post") as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"code": 0}
        result = bridge.send_reply("chat-1", "msg-1", "已收到")

    assert result["ok"] is True
    assert mock_post.call_count >= 1
