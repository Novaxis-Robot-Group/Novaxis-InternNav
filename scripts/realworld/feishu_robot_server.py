import argparse
import json
import os

from flask import Flask, jsonify, request

try:
    from scripts.realworld.feishu_robot_bridge import FeishuRobotBridge, RobotCommandDispatcher
except ImportError:  # pragma: no cover
    from feishu_robot_bridge import FeishuRobotBridge, RobotCommandDispatcher


app = Flask(__name__)
bridge = FeishuRobotBridge(dispatcher=RobotCommandDispatcher())


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "service": "feishu_robot_bridge"})


@app.route("/robot/command", methods=["POST"])
def robot_command():
    payload = request.get_json(silent=True) or {}
    command = payload.get("command")
    if isinstance(command, dict):
        result = bridge.dispatcher.dispatch(command)
    else:
        text = payload.get("text") or payload.get("message") or ""
        result = bridge.dispatcher.dispatch({"action": "unknown", "text": text})
    return jsonify(result)


@app.route("/feishu/webhook", methods=["POST"])
def feishu_webhook():
    payload = request.get_json(silent=True) or {}
    result = bridge.handle_event(payload)
    return jsonify(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feishu robot bridge server")
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "9000")))
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=False)
