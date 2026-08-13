"""Local speech-to-text support for the real-world experiment viewer."""

import importlib.util
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import wave
from pathlib import Path


MAX_AUDIO_BYTES = 16 * 1024 * 1024
KNOWN_STT_MODELS = ("tiny", "base", "small", "medium", "large-v3")


def _huggingface_hub_cache():
    explicit = os.environ.get("HF_HUB_CACHE")
    if explicit:
        return Path(explicit).expanduser()
    hf_home = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    return hf_home / "hub"


def cached_speech_models():
    """Return known models whose snapshot contains a resolved model.bin."""
    hub_cache = _huggingface_hub_cache()
    cached = []
    for name in KNOWN_STT_MODELS:
        snapshots = hub_cache / f"models--Systran--faster-whisper-{name}" / "snapshots"
        if any(path.exists() and path.stat().st_size > 0 for path in snapshots.glob("*/model.bin")):
            cached.append(name)
    return cached


def _module_available(module_name):
    """Check an optional module without failing on test-injected modules."""
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return module_name in __import__("sys").modules


def _funasr_model_ready(model_path):
    model_path = Path(model_path).expanduser().resolve()
    return model_path.is_dir() and (model_path / "model.pt").is_file() and (model_path / "config.yaml").is_file()


def speech_backend_status(
    model_name=None,
    backend="faster-whisper",
    device=None,
    funasr_model_path="checkpoints/Fun-ASR-Nano-2512",
    sensevoice_model_path="checkpoints/SenseVoiceSmall",
):
    """Return public backend status without loading an ASR model."""
    whisper_available = _module_available("faster_whisper")
    funasr_available = _module_available("funasr")
    torchaudio_available = _module_available("torchaudio")
    funasr_path = Path(funasr_model_path).expanduser().resolve()
    sensevoice_path = Path(sensevoice_model_path).expanduser().resolve()
    funasr_model_ready = _funasr_model_ready(funasr_path)
    sensevoice_model_ready = _funasr_model_ready(sensevoice_path)
    backend = str(backend or "faster-whisper")
    backend_ready = {
        "faster-whisper": whisper_available,
        "funasr-nano": funasr_available and torchaudio_available and funasr_model_ready,
        "sensevoice": funasr_available and torchaudio_available and sensevoice_model_ready,
    }
    missing = []
    if backend in {"funasr-nano", "sensevoice"}:
        if not funasr_available:
            missing.append("funasr")
        if not torchaudio_available:
            missing.append("torchaudio")
        selected_ready = funasr_model_ready if backend == "funasr-nano" else sensevoice_model_ready
        if not selected_ready:
            missing.append("本地模型")
    return {
        "available": bool(backend_ready.get(backend, False)),
        "backend": backend,
        "model": str(model_name or os.environ.get("INTERNNAV_STT_MODEL", "small")),
        "device": str(device or os.environ.get("INTERNNAV_STT_DEVICE", "cpu")),
        "cached_models": cached_speech_models(),
        "backend_ready": backend_ready,
        "funasr_installed": funasr_available,
        "torchaudio_installed": torchaudio_available,
        "funasr_model_ready": funasr_model_ready,
        "sensevoice_model_ready": sensevoice_model_ready,
        "message": "ready" if backend_ready.get(backend, False) else f"缺少：{', '.join(missing) or '可用后端'}",
    }


class SpeechTranscriber:
    """Lazily load one faster-whisper model and serialize transcription calls."""

    def __init__(self):
        self._model = None
        self._model_signature = None
        self._loaded_backend = ""
        self._lock = threading.Lock()
        self._state = "idle"
        self._state_started_at = None
        self._last_error = ""

    def status(self):
        elapsed = None
        if self._state_started_at is not None:
            elapsed = max(0.0, time.monotonic() - self._state_started_at)
        return {
            "worker_state": self._state,
            "worker_state_seconds": elapsed,
            "loaded_model": self._model_signature[0] if self._model_signature else "",
            "loaded_backend": self._loaded_backend,
            "last_error": self._last_error,
        }

    def _load_model(self, model_name=None):
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError("语音转写组件未安装，请执行：pip install faster-whisper") from exc

        signature = (
            str(model_name or os.environ.get("INTERNNAV_STT_MODEL", "small")),
            os.environ.get("INTERNNAV_STT_DEVICE", "cpu"),
            os.environ.get("INTERNNAV_STT_COMPUTE_TYPE", "int8"),
        )
        if self._model is None or self._model_signature != signature:
            if signature[0] in KNOWN_STT_MODELS and signature[0] not in cached_speech_models():
                raise RuntimeError(
                    f"语音模型 {signature[0]!r} 未完整下载到本地缓存。"
                    "请先完成模型下载，再回到网页选择。"
                )
            try:
                self._model = WhisperModel(
                    signature[0],
                    device=signature[1],
                    compute_type=signature[2],
                    local_files_only=True,
                )
            except Exception as cache_exc:
                raise RuntimeError(
                    f"语音模型 {signature[0]!r} 未完整下载到本地缓存。"
                    "为避免交互请求长期阻塞，请先在终端单独下载模型，再回到网页选择。"
                ) from cache_exc
            self._model_signature = signature
            self._loaded_backend = "faster-whisper"
        return self._model

    def _load_funasr_model(self, backend, model_path, device):
        try:
            from funasr import AutoModel
        except (ImportError, ModuleNotFoundError) as exc:
            missing = getattr(exc, "name", None) or "FunASR dependency"
            raise RuntimeError(
                f"FunASR 后端缺少依赖 {missing!r}。"
                "请安装 funasr>=1.3.3，并安装与当前 torch 版本匹配的 torchaudio。"
            ) from exc

        model_path = Path(model_path).expanduser().resolve()
        if not model_path.is_dir():
            raise RuntimeError(f"{backend} 本地模型目录不存在：{model_path}")
        signature = (str(model_path), str(device or "cpu"), backend)
        if self._model is None or self._model_signature != signature:
            try:
                self._model = AutoModel(
                    model=str(model_path),
                    trust_remote_code=True,
                    device=str(device or "cpu"),
                )
            except Exception as exc:
                raise RuntimeError(f"无法加载 {backend} 模型 {model_path}：{exc}") from exc
            self._model_signature = signature
            self._loaded_backend = backend
        return self._model

    @staticmethod
    def _convert_to_wav(source_path, output_path):
        """Decode browser audio to 16 kHz mono PCM without requiring ffmpeg."""
        try:
            import av
        except ImportError:
            av = None

        if av is not None:
            try:
                resampler = av.audio.resampler.AudioResampler(
                    format="s16",
                    layout="mono",
                    rate=16000,
                )
                with av.open(str(source_path), mode="r") as container, wave.open(str(output_path), "wb") as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(16000)
                    for frame in container.decode(audio=0):
                        for converted in resampler.resample(frame):
                            wav_file.writeframes(converted.to_ndarray().tobytes())
                    for converted in resampler.resample(None):
                        wav_file.writeframes(converted.to_ndarray().tobytes())
                return
            except (av.error.FFmpegError, OSError, ValueError, wave.Error):
                pass

        # A standalone ffmpeg binary remains a fallback for unusual codecs.
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("无法解码录音：PyAV 转换失败，且服务器未安装 ffmpeg。")
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(source_path),
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    str(output_path),
                ],
                check=True,
                capture_output=True,
                timeout=20,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("FunASR 音频转换需要 ffmpeg，但当前环境未安装。") from exc
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            detail = getattr(exc, "stderr", b"") or b""
            raise RuntimeError(f"无法把浏览器录音转换成 WAV：{detail.decode(errors='ignore')[:300]}") from exc

    @staticmethod
    def _extract_funasr_text(result):
        if not isinstance(result, list) or not result or not isinstance(result[0], dict):
            raise RuntimeError("FunASR 没有返回预期的识别结果。")
        text = str(result[0].get("text") or "").strip()
        text = re.sub(r"<\|[^|]+\|>", "", text).strip()
        return text

    def transcribe(
        self,
        audio_bytes,
        filename="recording.webm",
        language=None,
        model_name=None,
        backend="faster-whisper",
        device=None,
        funasr_model_path="checkpoints/Fun-ASR-Nano-2512",
        sensevoice_model_path="checkpoints/SenseVoiceSmall",
    ):
        if not audio_bytes:
            raise ValueError("没有收到录音数据。")
        if len(audio_bytes) > MAX_AUDIO_BYTES:
            raise ValueError("录音文件过大，请将单次录音控制在 60 秒以内。")

        suffix = Path(filename or "recording.webm").suffix.lower()
        if suffix not in {".webm", ".ogg", ".wav", ".mp3", ".m4a", ".mp4"}:
            suffix = ".webm"
        language = (language or "").strip().lower() or None
        if language not in {None, "zh", "en"}:
            raise ValueError("不支持的识别语言。")
        backend = str(backend or "faster-whisper").strip().lower()
        if backend not in {"faster-whisper", "funasr-nano", "sensevoice"}:
            raise ValueError(f"不支持的语音识别后端：{backend}")

        with tempfile.NamedTemporaryFile(suffix=suffix) as audio_file:
            audio_file.write(audio_bytes)
            audio_file.flush()
            if not self._lock.acquire(timeout=10.0):
                raise RuntimeError(
                    "语音转写器仍被上一条请求占用。请重启 Viewer 释放旧请求，"
                    "并确认所选模型已完整下载。"
                )
            try:
                self._state = "loading"
                self._state_started_at = time.monotonic()
                self._last_error = ""
                if backend == "faster-whisper":
                    model = self._load_model(model_name=model_name)
                else:
                    selected_path = funasr_model_path if backend == "funasr-nano" else sensevoice_model_path
                    model = self._load_funasr_model(backend, selected_path, device or "cpu")
                self._state = "transcribing"
                if backend == "faster-whisper":
                    segments, info = model.transcribe(
                        audio_file.name,
                        language=language,
                        beam_size=5,
                        vad_filter=True,
                        condition_on_previous_text=False,
                    )
                    text = " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()
                    detected_language = getattr(info, "language", language or "unknown")
                    language_probability = float(getattr(info, "language_probability", 0.0) or 0.0)
                    duration = float(getattr(info, "duration", 0.0) or 0.0)
                else:
                    with tempfile.NamedTemporaryFile(suffix=".wav") as wav_file:
                        self._convert_to_wav(audio_file.name, wav_file.name)
                        generation_kwargs = {
                            "input": [wav_file.name],
                            "cache": {},
                            "batch_size": 1,
                        }
                        if backend == "funasr-nano":
                            generation_kwargs.update({"language": "中文", "itn": True})
                        else:
                            generation_kwargs.update({"language": "auto", "use_itn": True})
                        text = self._extract_funasr_text(model.generate(**generation_kwargs))
                    detected_language = "zh" if language in {None, "zh"} else language
                    language_probability = 0.0
                    duration = 0.0
            except Exception as exc:
                self._last_error = str(exc)
                raise
            finally:
                self._state = "idle"
                self._state_started_at = None
                self._lock.release()

        if not text:
            raise ValueError("没有识别到清晰语音，请靠近麦克风后重试。")
        return {
            "text": text,
            "language": detected_language,
            "language_probability": language_probability,
            "duration": duration,
            "model": self._model_signature[0] if self._model_signature else str(model_name or "small"),
            "backend": backend,
        }
