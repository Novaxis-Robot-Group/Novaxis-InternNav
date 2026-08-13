"""Local Qwen3.6-VL service lifecycle and OpenAI-compatible endpoint helpers."""

import json
import os
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path

import requests


LOCAL_QWEN_MODEL = "local-qwen3.6-vl"
LOCAL_QWEN_SERVED_MODEL = "qwen3.6-local"
LOCAL_QWEN_API_URL = "http://127.0.0.1:8000/v1/chat/completions"
QWEN36_BF16_SHARD_COUNT = 26


def is_local_qwen_model(model_name):
    return str(model_name or "").strip().lower() in {
        LOCAL_QWEN_MODEL,
        LOCAL_QWEN_SERVED_MODEL,
    }


def resolve_model_name(config):
    model_name = str((config or {}).get("model") or "").strip()
    return LOCAL_QWEN_SERVED_MODEL if is_local_qwen_model(model_name) else model_name


def resolve_api_url(config):
    if is_local_qwen_model((config or {}).get("model")):
        return LOCAL_QWEN_API_URL
    return str((config or {}).get("api_url") or "").strip()


def resolve_api_key(config):
    if is_local_qwen_model((config or {}).get("model")):
        # vLLM accepts a placeholder bearer token. No DashScope credential is used.
        return "EMPTY"
    config = config or {}
    return str(config.get("api_key") or os.environ.get(config.get("api_key_env") or "") or "").strip()


def local_qwen_default_config(repo_root):
    repo_root = Path(repo_root).expanduser().resolve()
    return {
        "model_path": str(repo_root.parent / "models" / "Qwen3.6-35B-A3B"),
        "python_path": str(repo_root.parent / "envs" / "qwen36-vl" / "bin" / "python"),
        "gpu": "1",
        "host": "127.0.0.1",
        "port": 8000,
        # Keep this 35B BF16 VLM cooperative with the 30GB InternVLA process
        # already resident on GPU 1. Throughput is intentionally conservative.
        # 8K is needed for upper-agent prompts that include several motion
        # frames. This still leaves ample H200 memory for InternVLA on GPU 1.
        "gpu_memory_utilization": 0.60,
        "max_model_len": 8192,
        "max_num_seqs": 1,
        "served_model_name": LOCAL_QWEN_SERVED_MODEL,
    }


class LocalQwenServiceLauncher:
    """Launch Qwen in an isolated environment without touching InternVLA."""

    def __init__(self, repo_root, state_path):
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.state_path = Path(state_path).expanduser().resolve()
        self.log_dir = self.state_path.parent / "local_qwen_service_logs"

    def default_config(self):
        return local_qwen_default_config(self.repo_root)

    def read_state(self):
        try:
            return json.loads(self.state_path.read_text())
        except (OSError, json.JSONDecodeError):
            return {}

    def write_state(self, state):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temporary.write_text(json.dumps(state, indent=2, ensure_ascii=False))
        os.replace(temporary, self.state_path)

    def normalize_config(self, config):
        result = self.default_config()
        result.update({key: value for key, value in dict(config or {}).items() if value not in (None, "")})
        result["model_path"] = str(Path(result["model_path"]).expanduser().resolve())
        result["python_path"] = str(Path(result["python_path"]).expanduser().resolve())
        result["gpu"] = str(result["gpu"]).strip()
        result["host"] = str(result["host"]).strip() or "127.0.0.1"
        result["port"] = max(1, min(65535, int(result["port"])))
        result["gpu_memory_utilization"] = min(0.85, max(0.20, float(result["gpu_memory_utilization"])))
        result["max_model_len"] = max(1024, min(32768, int(result["max_model_len"])))
        result["max_num_seqs"] = max(1, min(16, int(result["max_num_seqs"])))
        result["served_model_name"] = str(result["served_model_name"] or LOCAL_QWEN_SERVED_MODEL).strip()
        return result

    @staticmethod
    def process_alive(pid):
        try:
            os.kill(int(pid), 0)
            return True
        except (OSError, TypeError, ValueError):
            return False

    def pid_is_local_qwen(self, pid):
        cmdline = Path(f"/proc/{pid}/cmdline")
        try:
            command = cmdline.read_bytes().replace(b"\x00", b" ").decode("utf-8", errors="replace")
        except OSError:
            return False
        return (
            ("vllm serve" in command or "vllm.entrypoints.openai.api_server" in command)
            and "Qwen3.6-35B-A3B" in command
        )

    def model_status(self, config=None):
        config = self.normalize_config(config)
        model_path = Path(config["model_path"])
        index_path = model_path / "model.safetensors.index.json"
        expected = []
        if index_path.is_file():
            try:
                expected = sorted(set(json.loads(index_path.read_text()).get("weight_map", {}).values()))
            except (OSError, ValueError, TypeError):
                expected = []
        if expected:
            present = [name for name in expected if (model_path / name).is_file()]
            expected_count = len(expected)
        else:
            # The index JSON is usually downloaded last. Keep progress useful
            # while Hugging Face is still fetching the official 26 BF16 shards.
            present = list(model_path.glob("model-*-of-*.safetensors")) if model_path.is_dir() else []
            expected_count = QWEN36_BF16_SHARD_COUNT
        size_bytes = sum(path.stat().st_size for path in model_path.rglob("*") if path.is_file()) if model_path.is_dir() else 0
        return {
            "path": str(model_path),
            "exists": model_path.is_dir(),
            "expected_shards": expected_count,
            "present_shards": len(present),
            "complete": bool(expected) and len(present) == len(expected),
            "size_gib": round(size_bytes / 1024**3, 2),
        }

    def build_command(self, config):
        return [
            config["python_path"], "-m", "vllm.entrypoints.openai.api_server",
            "--model", config["model_path"],
            "--host", config["host"],
            "--port", str(config["port"]),
            "--served-model-name", config["served_model_name"],
            "--dtype", "bfloat16",
            "--gpu-memory-utilization", str(config["gpu_memory_utilization"]),
            "--max-model-len", str(config["max_model_len"]),
            "--max-num-seqs", str(config["max_num_seqs"]),
            # A shared robotics GPU benefits more from predictable startup and
            # latency than from CUDA graph / torch.compile throughput gains.
            "--enforce-eager",
            # Avoid a long first-run FlashInfer JIT build for Qwen3.6's GDN
            # prefill kernel. Triton is slower only for that narrow path but
            # starts reliably on the shared navigation GPU.
            "--gdn-prefill-backend", "triton",
        ]

    def status(self):
        state = self.read_state()
        config = self.normalize_config(state.get("config"))
        pid = state.get("pid")
        running = bool(pid) and self.process_alive(pid) and self.pid_is_local_qwen(pid)
        ready = False
        if running:
            try:
                response = requests.get(f"http://{config['host']}:{config['port']}/v1/models", timeout=0.7)
                ready = response.ok
            except requests.RequestException:
                pass
        return {
            "running": running,
            "ready": ready,
            "pid": int(pid) if running else None,
            "config": config,
            "model": self.model_status(config),
            "log_path": state.get("log_path"),
            "started_at": state.get("started_at"),
            "stopped_at": state.get("stopped_at"),
        }

    def start(self, config=None):
        current = self.status()
        if current["running"]:
            return current
        config = self.normalize_config(config or current["config"])
        model = self.model_status(config)
        if not model["complete"]:
            raise RuntimeError(
                f"Qwen weights are incomplete: {model['present_shards']}/{model['expected_shards']} shards, {model['size_gib']} GiB."
            )
        if not Path(config["python_path"]).is_file():
            raise RuntimeError(f"Qwen Python environment not found: {config['python_path']}")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / f"qwen36_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = config["gpu"]
        env["PYTHONUNBUFFERED"] = "1"
        # FlashInfer JIT invokes the `ninja` executable. Keep the selected
        # virtual environment's bin directory on PATH for that child process.
        python_bin_dir = str(Path(config["python_path"]).parent)
        env["PATH"] = python_bin_dir + os.pathsep + env.get("PATH", "")
        # Prevent one-time kernel work from saturating the host while the
        # navigation server remains live.
        env.setdefault("MAX_JOBS", "2")
        env.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
        env.setdefault("OMP_NUM_THREADS", "4")
        env.setdefault("MKL_NUM_THREADS", "4")
        with open(log_path, "a") as log_file:
            process = subprocess.Popen(
                self.build_command(config), cwd=self.repo_root, stdout=log_file,
                stderr=subprocess.STDOUT, env=env, start_new_session=True,
            )
        state = {
            "pid": process.pid,
            "config": config,
            "log_path": str(log_path),
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "stopped_at": "",
        }
        self.write_state(state)
        time.sleep(0.8)
        if process.poll() is not None:
            state["stopped_at"] = datetime.now().isoformat(timespec="seconds")
            self.write_state(state)
            raise RuntimeError(f"Local Qwen exited during startup; see {log_path}")
        return self.status()

    def stop(self):
        state = self.read_state()
        pid = state.get("pid")
        if pid and self.process_alive(pid) and self.pid_is_local_qwen(pid):
            try:
                os.killpg(int(pid), signal.SIGTERM)
            except OSError:
                os.kill(int(pid), signal.SIGTERM)
            deadline = time.time() + 10
            while time.time() < deadline and self.process_alive(pid):
                time.sleep(0.2)
            if self.process_alive(pid):
                try:
                    os.killpg(int(pid), signal.SIGKILL)
                except OSError:
                    os.kill(int(pid), signal.SIGKILL)
        state["stopped_at"] = datetime.now().isoformat(timespec="seconds")
        self.write_state(state)
        return self.status()
