import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from zenoh_internvla_rpc import DEFAULT_UDP_SERVER_LISTEN


ALLOWED_TRANSPORTS = {"http", "zenoh"}
SERVICE_SCRIPTS = {
    "http": "scripts/realworld/http_internvla_server.py",
    "zenoh": "scripts/realworld/zenoh_internvla_server.py",
}


def default_service_state_path(log_dir="output/realworld_experiments"):
    root = Path(log_dir).expanduser().resolve()
    return root.parent / "realworld_service_state.json"


def normalize_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def split_endpoints(value):
    if not value:
        return []
    if isinstance(value, list):
        raw_items = value
    else:
        raw_items = str(value).replace(",", "\n").splitlines()
    return [item.strip() for item in raw_items if item and item.strip()]


def normalize_launcher_config(config):
    config = dict(config or {})
    transport = str(config.get("transport") or "http").strip().lower()
    if transport not in ALLOWED_TRANSPORTS:
        transport = "http"

    normalized = {
        "transport": transport,
        "device": str(config.get("device") or "cuda:0").strip(),
        "model_path": str(config.get("model_path") or "checkpoints/InternVLA-N1-DualVLN").strip(),
        "host": str(config.get("host") or "0.0.0.0").strip(),
        "http_port": int(config.get("http_port") or 8848),
        "zenoh_key": str(config.get("zenoh_key") or "internvla/eval_dual").strip(),
        "zenoh_mode": str(config.get("zenoh_mode") or ("peer" if transport == "zenoh" else "")).strip(),
        "zenoh_connect": "\n".join(split_endpoints(config.get("zenoh_connect"))),
        "zenoh_listen": "\n".join(split_endpoints(config.get("zenoh_listen") or (DEFAULT_UDP_SERVER_LISTEN if transport == "zenoh" else ""))),
        "zenoh_no_multicast_scouting": normalize_bool(config.get("zenoh_no_multicast_scouting", transport == "zenoh")),
        "no_warmup": normalize_bool(config.get("no_warmup")),
    }
    normalized["http_port"] = max(1, min(65535, normalized["http_port"]))
    if normalized["zenoh_mode"] not in {"", "peer", "client"}:
        normalized["zenoh_mode"] = ""
    return normalized


class RealworldServiceLauncher:
    def __init__(self, repo_root, runtime_config_path, experiment_log_dir, state_path=None):
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.runtime_config_path = Path(runtime_config_path).expanduser().resolve()
        self.experiment_log_dir = Path(experiment_log_dir).expanduser().resolve()
        self.state_path = Path(state_path or default_service_state_path(self.experiment_log_dir)).expanduser().resolve()
        self.log_dir = self.state_path.parent / "realworld_service_logs"

    def default_config(self):
        return normalize_launcher_config(
            {
                "transport": "http",
                "device": "cuda:0",
                "model_path": str(self.repo_root / "checkpoints/InternVLA-N1-DualVLN"),
                "host": "0.0.0.0",
                "http_port": 8848,
                "zenoh_key": "internvla/eval_dual",
                "zenoh_mode": "peer",
                "zenoh_connect": "",
                "zenoh_listen": DEFAULT_UDP_SERVER_LISTEN,
                "zenoh_no_multicast_scouting": True,
                "no_warmup": False,
            }
        )

    def read_state(self):
        if not self.state_path.exists():
            return {}
        try:
            with open(self.state_path) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}

    def write_state(self, state):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        with open(temp_path, "w") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        os.replace(temp_path, self.state_path)

    def pid_is_allowed_service(self, pid):
        cmdline_path = Path(f"/proc/{pid}/cmdline")
        if not cmdline_path.exists():
            return True
        try:
            cmdline = cmdline_path.read_bytes().replace(b"\x00", b" ").decode("utf-8", errors="replace")
        except OSError:
            return False
        return any(script in cmdline for script in SERVICE_SCRIPTS.values())

    def pid_is_running(self, pid):
        if not pid:
            return False
        try:
            os.kill(int(pid), 0)
        except (OSError, ValueError):
            return False
        return self.pid_is_allowed_service(int(pid))

    def status(self):
        state = self.read_state()
        pid = state.get("pid")
        running = self.pid_is_running(pid)
        if not running and state.get("running"):
            state["running"] = False
            state["stopped_at"] = datetime.now().isoformat(timespec="seconds")
            self.write_state(state)
        return {
            "running": running,
            "pid": pid if running else None,
            "transport": state.get("transport"),
            "command": state.get("command", []),
            "log_path": state.get("log_path"),
            "started_at": state.get("started_at"),
            "stopped_at": state.get("stopped_at"),
            "config": normalize_launcher_config(state.get("config") or self.default_config()),
            "state_path": str(self.state_path),
        }

    def build_command(self, config):
        config = normalize_launcher_config(config)
        script = SERVICE_SCRIPTS[config["transport"]]
        cmd = [
            sys.executable,
            str(self.repo_root / script),
            "--device",
            config["device"],
            "--model_path",
            config["model_path"],
            "--experiment_log_dir",
            str(self.experiment_log_dir),
            "--runtime_config_path",
            str(self.runtime_config_path),
        ]
        if config["no_warmup"]:
            cmd.append("--no_warmup")

        if config["transport"] == "http":
            cmd.extend(["--host", config["host"], "--port", str(config["http_port"])])
        else:
            cmd.extend(["--key", config["zenoh_key"]])
            if config["zenoh_mode"]:
                cmd.extend(["--mode", config["zenoh_mode"]])
            for endpoint in split_endpoints(config["zenoh_connect"]):
                cmd.extend(["--connect", endpoint])
            for endpoint in split_endpoints(config["zenoh_listen"]):
                cmd.extend(["--listen", endpoint])
            if config["zenoh_no_multicast_scouting"]:
                cmd.append("--no-multicast-scouting")
        return cmd

    def start(self, config):
        current = self.status()
        if current["running"]:
            raise RuntimeError(f"Model service is already running with pid {current['pid']}. Stop it before starting another transport.")

        config = normalize_launcher_config(config)
        cmd = self.build_command(config)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / f"{config['transport']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        log_file = open(log_path, "a")
        try:
            process = subprocess.Popen(
                cmd,
                cwd=self.repo_root,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
        finally:
            log_file.close()

        time.sleep(0.8)
        if process.poll() is not None:
            state = {
                "running": False,
                "pid": process.pid,
                "transport": config["transport"],
                "command": cmd,
                "log_path": str(log_path),
                "started_at": datetime.now().isoformat(timespec="seconds"),
                "stopped_at": datetime.now().isoformat(timespec="seconds"),
                "config": config,
            }
            self.write_state(state)
            raise RuntimeError(
                f"{config['transport'].upper()} service exited during startup with code {process.returncode}. "
                f"Check log: {log_path}"
            )

        state = {
            "running": True,
            "pid": process.pid,
            "transport": config["transport"],
            "command": cmd,
            "log_path": str(log_path),
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "stopped_at": "",
            "config": config,
        }
        self.write_state(state)
        return self.status()

    def stop(self):
        state = self.read_state()
        pid = state.get("pid")
        if not self.pid_is_running(pid):
            state["running"] = False
            state["stopped_at"] = datetime.now().isoformat(timespec="seconds")
            self.write_state(state)
            return self.status()

        pid = int(pid)
        try:
            os.killpg(pid, signal.SIGTERM)
        except OSError:
            os.kill(pid, signal.SIGTERM)

        deadline = time.time() + 8.0
        while time.time() < deadline:
            if not self.pid_is_running(pid):
                break
            time.sleep(0.2)

        if self.pid_is_running(pid):
            try:
                os.killpg(pid, signal.SIGKILL)
            except OSError:
                os.kill(pid, signal.SIGKILL)

        state["running"] = False
        state["stopped_at"] = datetime.now().isoformat(timespec="seconds")
        self.write_state(state)
        return self.status()
