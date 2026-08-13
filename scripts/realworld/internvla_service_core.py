import argparse
import io
import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from experiment_visualizer import ExperimentLogger
from internnav.agent.internvla_n1_agent_realworld import InternVLAN1AsyncAgent
from runtime_config import load_runtime_config, normalize_runtime_config, sanitize_runtime_config, save_runtime_config
from upper_agent import evaluate_latest as evaluate_upper_agent_latest


idx = 0
start_time = time.time()
output_dir = ""
args = None
agent = None
experiment_logger = None
inference_lock = threading.Lock()
upper_replan_lock = threading.Lock()
runtime_config = normalize_runtime_config({})
runtime_config_mtime = None
base_instruction = ""
consecutive_low_level_stop_count = 0
upper_replan_retry_after = 0.0

EXPLICIT_STOP_INSTRUCTIONS = {
    "stop",
    "stop now",
    "stop here",
    "stop moving",
    "stop the robot",
    "halt",
    "halt now",
    "stay still",
    "please stop",
    "please stop now",
    "停止",
    "停下",
    "马上停下",
    "别动",
}

# internvla_service_core.py 是 HTTP/Zenoh server 共用的推理核心。
# Go2 client 每次上传 RGB-D 后，都会走 run_dual_inference -> agent.step。
# runtime_config.json 是网页、Upper Agent、模型服务之间共享的“控制面板”：
# - upper_agent.task_instruction：用户给上层智能体的总任务。
# - instruction：上层智能体给 InternVLA-N1 低层大脑的当前短指令。
# - service_enabled：软启动/急停 gate，不杀进程，只决定是否真正推理。


def is_explicit_stop_instruction(instruction):
    """Match immediate stop commands without catching routes that end in stop."""
    normalized = " ".join(str(instruction or "").strip().lower().split())
    normalized = normalized.strip(".,!?;:。！？，；：")
    return normalized in EXPLICIT_STOP_INSTRUCTIONS


def request_upper_agent_replan(run_dir, request_low_level_hard_reset=True):
    """Schedule a replan without cutting short the active subgoal window."""
    global upper_replan_retry_after
    now = time.monotonic()
    if now < upper_replan_retry_after:
        return False
    if not upper_replan_lock.acquire(blocking=False):
        return False
    # Avoid a failed API call being retried on every incoming camera frame.
    upper_replan_retry_after = now + 2.0

    def worker():
        try:
            # Do not bypass min_seconds_between_calls here. That setting is the
            # low-level subgoal execution window: until it expires, the old
            # subgoal may continue and no policy_pause is written. Once the
            # window opens, evaluate_latest pauses the policy only for the
            # actual Qwen request and consumes the freshest saved frame.
            result = None
            while True:
                result = evaluate_upper_agent_latest(
                    run_dir,
                    args.runtime_config_path,
                    force=False,
                    request_low_level_hard_reset=request_low_level_hard_reset,
                    settle_for_fresh_frame=True,
                )
                if not result.get("skipped") or result.get("reason") != "minimum call interval not reached":
                    break
                time.sleep(0.25)
            event = result.get("event") or {}
            print(
                "upper-agent replan result: "
                f"ok={result.get('ok')} skipped={result.get('skipped')} "
                f"subgoal={(event.get('output') or {}).get('current_subgoal', '')}"
            )
        except Exception as exc:
            print(f"upper-agent replan failed: {exc}")
        finally:
            upper_replan_lock.release()

    threading.Thread(target=worker, name="upper-agent-replan", daemon=True).start()
    return True

def add_common_model_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model_path", type=str, default="checkpoints/InternVLA-N1-DualVLN")
    parser.add_argument(
        "--instruction",
        type=str,
        default="",
    )
    parser.add_argument("--resize_w", type=int, default=384)
    parser.add_argument("--resize_h", type=int, default=384)
    parser.add_argument("--num_history", type=int, default=8)
    parser.add_argument("--plan_step_gap", type=int, default=4)
    parser.add_argument("--experiment_log_dir", type=str, default="output/realworld_experiments")
    parser.add_argument("--save_frame_interval", type=int, default=0)
    parser.add_argument("--return_traj_points", type=int, default=10)
    parser.add_argument("--runtime_config_path", type=str, default="output/realworld_runtime_config.json")
    parser.add_argument("--no_warmup", action="store_true", help="Skip the startup warmup inference.")
    return parser


def init_service(parsed_args):
    global args, agent, experiment_logger, runtime_config, runtime_config_mtime, base_instruction

    args = parsed_args
    args.model_path = str(Path(args.model_path).expanduser().resolve())
    args.runtime_config_path = str(Path(args.runtime_config_path).expanduser().resolve())
    base_instruction = args.instruction
    args.camera_intrinsic = np.array(
        [[386.5, 0.0, 328.9, 0.0], [0.0, 386.5, 244, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    runtime_config = annotate_runtime_config(ensure_runtime_config_file())
    save_runtime_config(args.runtime_config_path, runtime_config)
    apply_runtime_config(runtime_config)
    experiment_logger = ExperimentLogger(args.experiment_log_dir, args.save_frame_interval)
    agent = InternVLAN1AsyncAgent(args)
    runtime_config_mtime = Path(args.runtime_config_path).stat().st_mtime if Path(args.runtime_config_path).exists() else None

    if not args.no_warmup:
        warmup_agent()


def service_info():
    if args is None:
        return {"status": "not_initialized"}
    return {
        "status": "ok",
        "model_path": args.model_path,
        "device": args.device,
        "runtime_config": sanitize_runtime_config(runtime_config),
    }


def ensure_runtime_config_file():
    path = Path(args.runtime_config_path)
    if path.exists():
        return load_runtime_config(path)
    return save_runtime_config(
        path,
        {
            "instruction": "",
            "resize_w": args.resize_w,
            "resize_h": args.resize_h,
            "num_history": args.num_history,
            "plan_step_gap": args.plan_step_gap,
            "return_traj_points": args.return_traj_points,
            "save_frame_interval": args.save_frame_interval,
            "low_level_stop_replan_threshold": 3,
        },
    )


def annotate_runtime_config(config):
    config = dict(config)
    config["device"] = args.device
    config["runtime_config_path"] = args.runtime_config_path
    return config


def apply_runtime_config(config):
# 将磁盘 runtime_config 同步到模型服务进程内的 args/agent。
# 注意：这里的 instruction 是低层大脑指令，不是 Upper Agent 总任务。
    args.resize_w = config["resize_w"]
    args.resize_h = config["resize_h"]
    args.num_history = config["num_history"]
    args.plan_step_gap = config["plan_step_gap"]
    args.return_traj_points = config["return_traj_points"]
    args.save_frame_interval = config["save_frame_interval"]
    args.instruction = config["instruction"] or base_instruction
    if agent is not None:
        agent.resize_w = args.resize_w
        agent.resize_h = args.resize_h
        agent.num_history = args.num_history
        agent.PLAN_STEP_GAP = args.plan_step_gap
    if experiment_logger is not None:
        experiment_logger.save_frame_interval = args.save_frame_interval


def refresh_runtime_config():
# 每帧推理前检查 runtime_config.json 是否被网页/Upper Agent 修改。
# 如果 mtime 变了，就重新加载配置，让新 instruction、plan_gap 等热生效。
    global runtime_config, runtime_config_mtime
    path = Path(args.runtime_config_path)
    if not path.exists():
        runtime_config = annotate_runtime_config(ensure_runtime_config_file())
        runtime_config_mtime = path.stat().st_mtime
        apply_runtime_config(runtime_config)
        return runtime_config

    mtime = path.stat().st_mtime
    if runtime_config_mtime is None or mtime > runtime_config_mtime:
        runtime_config = annotate_runtime_config(load_runtime_config(path))
        runtime_config_mtime = mtime
        apply_runtime_config(runtime_config)
        print(f"runtime config updated: {sanitize_runtime_config(runtime_config)}")
    return runtime_config


def warmup_agent():
    print("Running warmup inference...")
    dummy_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_depth = np.ones((480, 640), dtype=np.float32)
    dummy_pose = np.eye(4, dtype=np.float32)

    agent.reset()
    agent.step(
        dummy_rgb,
        dummy_depth,
        dummy_pose,
        "hello",
        intrinsic=args.camera_intrinsic,
    )
    agent.reset()
    print("Warmup inference finished.")


def decode_rgb_depth(image_bytes, depth_bytes):
    started_at = time.perf_counter()
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert("RGB")
    image = np.asarray(image)

    depth = Image.open(io.BytesIO(depth_bytes))
    depth = depth.convert("I")
    depth = np.asarray(depth)
    depth = depth.astype(np.float32) / 10000.0
    return image, depth, time.perf_counter() - started_at


def run_dual_inference(image, depth, data):
    with inference_lock:
        return _run_dual_inference_unlocked(image, depth, data)


def _run_dual_inference_unlocked(image, depth, data):
    global idx, output_dir, start_time, runtime_config, runtime_config_mtime, consecutive_low_level_stop_count

    if agent is None or args is None:
        raise RuntimeError("InternVLA service is not initialized. Call init_service(args) first.")

    config = refresh_runtime_config()
    camera_pose = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    request_instruction = str(data.get("instruction") or "").strip()
    policy_init = bool(data.get("reset", False))
    upper_agent_config = config.get("upper_agent") if isinstance(config.get("upper_agent"), dict) else {}
    upper_agent_enabled = bool(upper_agent_config.get("enabled"))
    upper_task_status = str(upper_agent_config.get("last_task_status") or "running").strip().lower()
    if policy_init and upper_agent_enabled:
# 新 episode 的首帧不能沿用上一次低层子目标。
# request_instruction 是新总任务；没有传入时，沿用网页刚设置的上层总任务。
        latest_config = load_runtime_config(args.runtime_config_path)
        latest_upper = latest_config.get("upper_agent") if isinstance(latest_config.get("upper_agent"), dict) else {}
        latest_upper = dict(latest_upper)
        if request_instruction:
            latest_upper["task_instruction"] = request_instruction
        latest_upper["last_task_status"] = "running"
        latest_upper["last_subgoal"] = ""
        latest_upper["last_decision_at"] = ""
        latest_upper["replan_requested"] = True
        latest_upper.pop("last_task_feedback", None)
        latest_upper.pop("last_task_report_at", None)
        # 新任务不能继承上一任务遗留的低层硬重置令牌。
        for key in (
            "hard_reset_requested",
            "hard_reset_reason",
            "hard_reset_requested_at",
            "hard_reset_subgoal",
        ):
            latest_upper.pop(key, None)
        latest_config["upper_agent"] = latest_upper
        latest_config["instruction"] = ""
        latest_config.pop("_upper_agent_pause", None)
        config = annotate_runtime_config(save_runtime_config(args.runtime_config_path, latest_config))
        runtime_config = config
        runtime_config_mtime = Path(args.runtime_config_path).stat().st_mtime
        apply_runtime_config(config)
        upper_agent_config = latest_upper
        upper_task_status = "running"
    elif upper_agent_enabled and request_instruction and not upper_agent_config.get("task_instruction"):
# 隔离逻辑：
# 如果 Upper Agent 开启，Go2 client 请求里的 instruction 视为“总任务”。
# 它会写入 upper_agent.task_instruction，不能直接喂给低层 InternVLA-N1。
        upper_agent_config = dict(upper_agent_config)
        upper_agent_config["task_instruction"] = request_instruction
        config["upper_agent"] = upper_agent_config
        config = annotate_runtime_config(save_runtime_config(args.runtime_config_path, config))
        runtime_config = config
        runtime_config_mtime = Path(args.runtime_config_path).stat().st_mtime
        apply_runtime_config(config)

    # Upper Agent 在低层 STOP 后给出新子目标时，请求一次“硬重置低层”。
    # 这不同于请求 reset=True：不重置 idx，不新建 ExperimentLogger run，
    # 也不会清空 Upper Agent 的总任务和路线记忆。
    hard_reset_applied = False
    upper_agent_config = config.get("upper_agent") if isinstance(config.get("upper_agent"), dict) else {}
    if bool(upper_agent_config.get("hard_reset_requested")) and not policy_init:
        reset_subgoal = str(upper_agent_config.get("hard_reset_subgoal") or config.get("instruction") or "").strip()
        print(f"hard reset low-level agent in current run; next subgoal: {reset_subgoal}")
        agent.reset()
        hard_reset_applied = True

        latest_config = load_runtime_config(args.runtime_config_path)
        latest_upper = latest_config.get("upper_agent") if isinstance(latest_config.get("upper_agent"), dict) else {}
        latest_upper = dict(latest_upper)
        latest_upper["hard_reset_requested"] = False
        latest_upper["hard_reset_applied_at"] = datetime.now().isoformat(timespec="seconds")
        demo_state = latest_upper.get("demo_agent")
        if isinstance(demo_state, dict) and demo_state.get("enabled"):
            demo_state = dict(demo_state)
            demo_state["execution_attempt"] = int(demo_state.get("execution_attempt") or 0) + 1
            demo_state["attempt_started_frame_idx_hint"] = idx + 1
            demo_state["attempt_started_at"] = datetime.now().isoformat(timespec="seconds")
            for key in (
                "step_started_step_index",
                "step_started_run_name",
                "step_started_frame_idx",
                "step_started_image_file",
            ):
                demo_state.pop(key, None)
            latest_upper["demo_agent"] = demo_state
        latest_upper.pop("hard_reset_reason", None)
        latest_upper.pop("hard_reset_requested_at", None)
        latest_upper.pop("hard_reset_subgoal", None)
        latest_upper.pop("hard_reset_command_id", None)
        latest_config["upper_agent"] = latest_upper
        config = annotate_runtime_config(save_runtime_config(args.runtime_config_path, latest_config))
        runtime_config = config
        runtime_config_mtime = Path(args.runtime_config_path).stat().st_mtime
        apply_runtime_config(config)
        upper_agent_config = latest_upper

    waiting_for_upper_agent_instruction = (
        upper_agent_enabled
        and upper_task_status not in {"completed", "failed"}
        and not config["instruction"]
    )
    if waiting_for_upper_agent_instruction:
# Upper Agent 开启但还没产出低层短指令时，返回 STOP 等待。
# 这样低层大脑不会拿用户的长任务直接乱跑。
        instruction_source = "upper_agent_waiting"
        instruction = "Stop and wait for upper-level navigation instruction."
    else:
# 正常优先级：
# 1. runtime_config["instruction"]：Upper Agent 或网页写入的低层短指令。
# 2. request_instruction：兼容未启用 Upper Agent 的旧 HTTP client 调用。
# 3. args.instruction：启动命令里的默认值。
        instruction_source = "runtime_config" if config["instruction"] else ("request" if request_instruction else "default")
        instruction = config["instruction"] or request_instruction or args.instruction
    explicit_stop_instruction = is_explicit_stop_instruction(instruction)
    if policy_init:
        start_time = time.time()
        idx = 0
        consecutive_low_level_stop_count = 0
        output_dir = "output/runs" + datetime.now().strftime("%m-%d-%H%M")
        os.makedirs(output_dir, exist_ok=True)
        experiment_logger.new_run()
        print("init reset model!!!")
        agent.reset()

    idx += 1

    look_down = False
    infer_started_at = time.perf_counter()
    ran_low_level_agent = False
    if not config.get("service_enabled", True):
        json_output = {"discrete_action": [0], "service_state": "stopped"}
        upper_pause = config.get("_upper_agent_pause") if isinstance(config.get("_upper_agent_pause"), dict) else {}
        if upper_pause.get("active") and upper_pause.get("token"):
            json_output.update(
                {
                    "service_state": "upper_agent_thinking",
                    "replan_required": True,
                    "replan_reason": "upper_agent_thinking",
                }
            )
        agent_debug = agent.get_debug_snapshot()
    elif explicit_stop_instruction:
        consecutive_low_level_stop_count = 0
        json_output = {
            "discrete_action": [0],
            "service_state": "explicit_stop_instruction",
            "explicit_stop": True,
        }
        agent_debug = agent.get_debug_snapshot()
    elif waiting_for_upper_agent_instruction:
        json_output = {
            "discrete_action": [0],
            "service_state": "waiting_upper_agent_instruction",
            "replan_required": True,
            "replan_reason": "waiting_upper_agent_instruction",
        }
        agent_debug = agent.get_debug_snapshot()
    elif upper_agent_enabled and upper_task_status in {"completed", "failed"}:
        # Terminal protocol: do not call the low-level model again,
        # otherwise its previous context can emit a fresh trajectory after the
        # Upper Agent has already declared the task completed or failed.
        consecutive_low_level_stop_count = 0
        json_output = {
            "discrete_action": [0],
            "task_completed": upper_task_status == "completed",
            "task_failed": upper_task_status == "failed",
            "service_state": f"task_{upper_task_status}",
            "completion_reason": f"upper_agent_{upper_task_status}",
            "task_feedback": upper_agent_config.get("last_task_feedback") or {},
        }
        agent_debug = agent.get_debug_snapshot()
    else:
# 只有这里才真正调用 InternVLA-N1 大脑。
# instruction 会进入 internvla_n1_agent_realworld.step，并在 System2 触发时写进 VLM prompt。
        ran_low_level_agent = True
        dual_sys_output = agent.step(
            image, depth, camera_pose, instruction, intrinsic=args.camera_intrinsic, look_down=look_down
        )
        if dual_sys_output.output_action is not None and dual_sys_output.output_action == [5]:
            look_down = True
            dual_sys_output = agent.step(
                image, depth, camera_pose, instruction, intrinsic=args.camera_intrinsic, look_down=look_down
            )

        json_output = {}
        if dual_sys_output.output_action is not None:
            json_output["discrete_action"] = dual_sys_output.output_action
        else:
            json_output["trajectory"] = dual_sys_output.output_trajectory.tolist()
            json_output["trajectory"] = json_output["trajectory"][: args.return_traj_points]
            if dual_sys_output.output_pixel is not None:
                json_output["pixel_goal"] = dual_sys_output.output_pixel
        agent_debug = agent.get_debug_snapshot()

    # A one-off low-level STOP can be a transient model hesitation. Count only
    # real agent.step() outputs; synthetic STOPs used while waiting/thinking do
    # not participate in the hard-replan protocol.
    replan_required = bool(json_output.get("replan_required"))
    replan_triggered = False
    initial_upper_agent_plan = False
    stop_replan_threshold = int(config.get("low_level_stop_replan_threshold", 3))
    low_level_stop = ran_low_level_agent and json_output.get("discrete_action") == [0]
    if low_level_stop:
        consecutive_low_level_stop_count += 1
    else:
        consecutive_low_level_stop_count = 0

    if (
        upper_agent_enabled
        and upper_task_status not in {"completed", "failed"}
        and low_level_stop
        and consecutive_low_level_stop_count >= stop_replan_threshold
        and config.get("service_enabled", True)
    ):
        replan_required = True
        json_output["replan_required"] = True
        json_output["replan_reason"] = "consecutive_low_level_stops_before_upper_completion"
        json_output["low_level_stop_count"] = consecutive_low_level_stop_count
        json_output["low_level_stop_replan_threshold"] = stop_replan_threshold
        # Clear immediately so a delayed Upper Agent request cannot be spawned
        # repeatedly by every subsequent frame.
        consecutive_low_level_stop_count = 0

        latest_config = load_runtime_config(args.runtime_config_path)
        latest_upper = latest_config.get("upper_agent") if isinstance(latest_config.get("upper_agent"), dict) else {}
        latest_upper = dict(latest_upper)
        latest_upper["replan_requested"] = True
        latest_upper["last_low_level_stop_at"] = datetime.now().isoformat(timespec="seconds")
        latest_config["upper_agent"] = latest_upper
        runtime_config = annotate_runtime_config(save_runtime_config(args.runtime_config_path, latest_config))
        runtime_config_mtime = Path(args.runtime_config_path).stat().st_mtime
        config = runtime_config

    if waiting_for_upper_agent_instruction and upper_agent_enabled:
        # Initial planning must be owned by the model server, not by an open
        # browser tab. Once the first RGB-D frame is saved below, schedule the
        # Upper Agent against that run and keep returning STOP until its first
        # current_subgoal is atomically written to runtime_config.
        initial_upper_agent_plan = True
        replan_required = True
        json_output["replan_required"] = True
        json_output["replan_reason"] = "initial_upper_agent_instruction_required"

    generate_time = time.perf_counter() - infer_started_at
    log_request_data = dict(data)
    # Keep calibration with every saved RGB-D frame for offline instance
    # localization. Existing clients do not need to change. If a future client
    # also sends odom/world_T_camera, ExperimentLogger preserves it alongside
    # this intrinsic matrix and enables cross-loop 3D identity merging.
    log_request_data.setdefault("camera_intrinsic", np.asarray(args.camera_intrinsic).tolist())
    metadata_path = experiment_logger.save_frame(
        idx,
        image,
        depth,
        log_request_data,
        instruction,
        json_output,
        generate_time,
        agent_debug,
        sanitize_runtime_config(runtime_config),
    )
    if replan_required and metadata_path is not None:
        replan_triggered = request_upper_agent_replan(
            Path(metadata_path).parent,
            request_low_level_hard_reset=not initial_upper_agent_plan,
        )
        json_output["replan_triggered"] = replan_triggered
    print(f"dual sys step {generate_time}")
    print(f"json_output {json_output}")
    return json_output, {
        "inference_time": generate_time,
        "episode_idx": idx,
        "output_type": "discrete_action" if "discrete_action" in json_output else "trajectory",
        "effective_instruction": instruction,
        "instruction_source": instruction_source,
        "upper_task_status": upper_task_status if upper_agent_enabled else None,
        "replan_required": replan_required,
        "replan_triggered": replan_triggered,
        "low_level_hard_reset_applied": hard_reset_applied,
        "consecutive_low_level_stop_count": consecutive_low_level_stop_count,
        "low_level_stop_replan_threshold": stop_replan_threshold,
        "runtime_config": sanitize_runtime_config(runtime_config),
        "experiment_metadata_path": str(metadata_path) if metadata_path is not None else None,
    }


def update_saved_timing(timing):
    metadata_path = timing.get("experiment_metadata_path")
    if not metadata_path:
        return

    path = Path(metadata_path)
    if not path.exists():
        return

    with open(path) as f:
        metadata = json.load(f)
    response = metadata.setdefault("response", {})
    response["_timing"] = timing
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def parse_json_data(json_data):
    if isinstance(json_data, bytes):
        json_data = json_data.decode("utf-8")
    if not json_data:
        return {}
    return json.loads(json_data)
