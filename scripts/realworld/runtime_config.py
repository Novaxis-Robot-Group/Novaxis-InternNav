import json
import os
from datetime import datetime
from pathlib import Path


DEFAULT_RUNTIME_CONFIG = {
    "service_enabled": True,
    "instruction": "",
    "resize_w": 384,
    "resize_h": 384,
    "num_history": 8,
    "plan_step_gap": 4,
    "return_traj_points": 10,
    "save_frame_interval": 1,
    "low_level_stop_replan_threshold": 3,
    "voice_silence_seconds": 1.4,
    "voice_command_confidence_threshold": 0.78,
    "speech_to_text_backend": "faster-whisper",
    "speech_to_text_model": "small",
    "speech_to_text_device": "cpu",
    "funasr_model_path": "checkpoints/Fun-ASR-Nano-2512",
    "sensevoice_model_path": "checkpoints/SenseVoiceSmall",
    "voice_language_model": "",
}

INT_LIMITS = {
    "resize_w": (224, 768),
    "resize_h": (224, 768),
    "num_history": (0, 16),
    "plan_step_gap": (1, 32),
    "return_traj_points": (1, 33),
    "save_frame_interval": (0, 1000),
    "low_level_stop_replan_threshold": (1, 20),
}

FLOAT_LIMITS = {
    "voice_silence_seconds": (0.4, 5.0),
    "voice_command_confidence_threshold": (0.5, 1.0),
}


def normalize_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def default_runtime_config_path(log_dir="output/realworld_experiments"):
    root = Path(log_dir).expanduser().resolve()
    return root.parent / "realworld_runtime_config.json"


def normalize_runtime_config(config):
    normalized = dict(DEFAULT_RUNTIME_CONFIG)
    if isinstance(config, dict):
        normalized.update(config)

    normalized["instruction"] = str(normalized.get("instruction") or "").strip()
    normalized["speech_to_text_model"] = str(
        normalized.get("speech_to_text_model") or DEFAULT_RUNTIME_CONFIG["speech_to_text_model"]
    ).strip()
    backend = str(normalized.get("speech_to_text_backend") or "faster-whisper").strip().lower()
    normalized["speech_to_text_backend"] = (
        backend if backend in {"faster-whisper", "funasr-nano", "sensevoice"} else "faster-whisper"
    )
    device = str(normalized.get("speech_to_text_device") or "cpu").strip().lower()
    normalized["speech_to_text_device"] = device if device == "cpu" or device.startswith("cuda") else "cpu"
    normalized["funasr_model_path"] = str(
        normalized.get("funasr_model_path") or DEFAULT_RUNTIME_CONFIG["funasr_model_path"]
    ).strip()
    normalized["sensevoice_model_path"] = str(
        normalized.get("sensevoice_model_path") or DEFAULT_RUNTIME_CONFIG["sensevoice_model_path"]
    ).strip()
    normalized["voice_language_model"] = str(normalized.get("voice_language_model") or "").strip()
    normalized["service_enabled"] = normalize_bool(normalized.get("service_enabled"))
    for key, (low, high) in INT_LIMITS.items():
        value = normalized.get(key, DEFAULT_RUNTIME_CONFIG[key])
        try:
            value = int(value)
        except (TypeError, ValueError):
            value = DEFAULT_RUNTIME_CONFIG[key]
        normalized[key] = max(low, min(high, value))
    for key, (low, high) in FLOAT_LIMITS.items():
        value = normalized.get(key, DEFAULT_RUNTIME_CONFIG[key])
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = DEFAULT_RUNTIME_CONFIG[key]
        normalized[key] = max(low, min(high, value))
    normalized["updated_at"] = str(normalized.get("updated_at") or "")
    return normalized


def load_runtime_config(path):
    path = Path(path).expanduser()
    if not path.exists():
        return normalize_runtime_config({})
    with open(path) as f:
        return normalize_runtime_config(json.load(f))


def save_runtime_config(path, config):
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = normalize_runtime_config(config)
    normalized["updated_at"] = datetime.now().isoformat(timespec="seconds")
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with open(temp_path, "w") as f:
        json.dump(normalized, f, indent=2, ensure_ascii=False)
    os.replace(temp_path, path)
    return normalized


def sanitize_runtime_config(config):
    if isinstance(config, dict):
        sanitized = {}
        for key, value in config.items():
            lower_key = str(key).lower()
            if lower_key in {"api_key", "authorization", "token", "secret", "password"}:
                continue
            else:
                sanitized[key] = sanitize_runtime_config(value)
        return sanitized
    if isinstance(config, list):
        return [sanitize_runtime_config(item) for item in config]
    return config
