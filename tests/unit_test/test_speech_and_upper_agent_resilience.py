import io
import json
import sys
from pathlib import Path


REALWORLD_DIR = Path(__file__).resolve().parents[2] / "scripts" / "realworld"
if str(REALWORLD_DIR) not in sys.path:
    sys.path.insert(0, str(REALWORLD_DIR))

import experiment_visualizer  # noqa: E402
import upper_agent  # noqa: E402
from runtime_config import save_runtime_config  # noqa: E402


class FakeResponse:
    def __init__(self, content, status_code=200, text=""):
        self.status_code = status_code
        self.ok = status_code < 400
        self.text = text
        self.reason = text or "test"
        self._content = content

    def json(self):
        return {"choices": [{"message": {"content": self._content}}], "usage": {}}


def test_rewrite_rejects_output_that_is_still_chinese(monkeypatch):
    monkeypatch.setattr(
        upper_agent.requests,
        "post",
        lambda *args, **kwargs: FakeResponse("直行，然后左转。"),
    )
    config = {"api_key": "server-only", "api_url": "https://example.invalid", "model": "qwen"}
    try:
        upper_agent.rewrite_spoken_navigation_instruction(config, "直行然后左转", target="low_level")
    except RuntimeError as exc:
        assert "仍包含中文" in str(exc)
    else:
        raise AssertionError("Chinese rewrite output must not be accepted as English.")


def test_speech_refine_failure_does_not_return_raw_text_for_auto_apply(tmp_path, monkeypatch):
    log_root = tmp_path / "runs"
    log_root.mkdir()
    config_path = tmp_path / "runtime.json"
    save_runtime_config(
        config_path,
        {"instruction": "Previous English command.", "upper_agent": {"api_key": "secret"}},
    )
    monkeypatch.setattr(
        experiment_visualizer.SpeechTranscriber,
        "transcribe",
        lambda self, *args, **kwargs: {"text": "直行然后左转"},
    )
    monkeypatch.setattr(
        experiment_visualizer,
        "rewrite_spoken_navigation_instruction",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("translation unavailable")),
    )
    app = experiment_visualizer.create_viewer_app(log_root, config_path)
    app.testing = True
    response = app.test_client().post(
        "/api/speech/transcribe",
        data={
            "audio": (io.BytesIO(b"fake audio"), "recording.webm"),
            "language": "zh",
            "refine": "true",
            "target": "low_level",
        },
        content_type="multipart/form-data",
    )
    payload = response.get_json()
    assert response.status_code == 502
    assert payload["ok"] is False
    assert payload["text"] == ""
    assert "直行然后左转" not in payload.get("error", "")
    assert json.loads(config_path.read_text())["instruction"] == "Previous English command."


def test_upper_agent_request_uses_json_mode_and_safe_token_floor(monkeypatch):
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured.update(json)
        return FakeResponse('{"task_status":"running"}')

    monkeypatch.setattr(upper_agent.requests, "post", fake_post)
    text, _, _ = upper_agent.call_qwen_vl(
        {
            "api_key": "server-only",
            "api_url": "https://example.invalid",
            "model": "qwen",
            "max_tokens": 512,
        },
        "data:image/jpeg;base64,AA==",
        "Return JSON.",
    )
    assert json.loads(text)["task_status"] == "running"
    assert captured["response_format"] == {"type": "json_object"}
    assert captured["max_tokens"] >= 1024
