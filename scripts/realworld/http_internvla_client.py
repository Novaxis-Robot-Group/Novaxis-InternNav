import atexit
import copy
import fcntl
import json
import math
import os
import sys
import termios
import threading
import time
from collections import deque
from enum import Enum

import cv2
import numpy as np
import rclpy
import requests
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from pyorbbecsdk import Config, OBFormat, OBSensorType, Pipeline

frame_data = {}
# 缓存最近若干帧的 RGB、Depth 和 odom，便于调试和轨迹对齐。
frame_idx = 0
# 本地帧编号，当前文件里主要作为保留变量。
# user-specific
from controllers import Mpc_controller, PID_controller
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from thread_utils import ReadWriteLock


_client_lock_handle = None


def _env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def acquire_client_lock():
# Orbbec 网络相机的 RTP 会话只能由一个 client 稳定接收。
# 用锁替代启动时 pkill/kill -9，避免误杀另一个实验或 Zenoh client。
    global _client_lock_handle
    lock_path = os.environ.get("INTERNVLA_CAMERA_LOCK", "/tmp/internvla_orbbec_client.lock")
    _client_lock_handle = open(lock_path, "w")
    try:
        fcntl.flock(_client_lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(
            "Another updated InternVLA camera client is already running. "
            "Stop that client before starting a second RTP receiver."
        ) from exc
    _client_lock_handle.write(str(os.getpid()))
    _client_lock_handle.flush()


def release_client_lock():
    global _client_lock_handle
    if _client_lock_handle is not None:
        try:
            fcntl.flock(_client_lock_handle.fileno(), fcntl.LOCK_UN)
            _client_lock_handle.close()
        finally:
            _client_lock_handle = None


atexit.register(release_client_lock)


def _getkey():
# 读取单键而不需要回车；非交互终端（例如 systemd）不启用该功能。
    if not sys.stdin.isatty():
        raise OSError("stdin is not a TTY")
    fd = sys.stdin.fileno()
    original = termios.tcgetattr(fd)
    changed = termios.tcgetattr(fd)
    changed[3] &= ~(termios.ICANON | termios.ECHO)
    changed[6][termios.VMIN] = 1
    changed[6][termios.VTIME] = 0
    try:
        termios.tcsetattr(fd, termios.TCSADRAIN, changed)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, original)


class ControlMode(Enum):
# 控制模式枚举：根据 server 返回的是离散动作还是连续轨迹，选择 PID 或 MPC。
    PID_Mode = 1
# PID 模式：用于执行 discrete_action 生成的小目标位姿。
    MPC_Mode = 2
# MPC 模式：用于追踪 server 返回的连续 trajectory。


# global variable
policy_init = True
# 是否是新任务第一次请求；True 时会通知 server reset。
mpc = None
# MPC 控制器对象，只有收到连续 trajectory 后才会初始化。
pid = PID_controller(Kp_trans=2.0, Kd_trans=0.0, Kp_yaw=1.5, Kd_yaw=0.0, max_v=0.6, max_w=0.5)
# PID/PD 控制器：离散动作模式下把目标位姿误差转换成线速度 v 和角速度 w。
http_idx = -1
# HTTP 请求编号，每发一次请求给 server 就加 1。
first_running_time = 0.0
# 第一次 HTTP 请求完成后的时间戳，主要用于统计运行时间。
last_pixel_goal = None
# 上一次像素目标点，当前 realworld client 中主要是保留状态。
last_s2_step = -1
# 上一次调用 System 2 的 step 编号，当前主要由 server/agent 内部控制。
manager = None
# Go2Manager 的全局引用，负责 odom、速度发布、Orbbec 相机采集和状态缓存。
current_control_mode = ControlMode.MPC_Mode
# 当前控制模式，收到 trajectory 时切 MPC，收到 discrete_action 时切 PID。
trajs_in_world = None
# 当前世界/odom 坐标系下的参考轨迹，供 MPC 追踪。

desired_v, desired_w = 0.0, 0.0
# 当前控制器期望输出的线速度和角速度，主要用于调试/状态观察。
debug_mode = _env_flag("INTERNVLA_DEBUG_MODE", False)
# debug 模式只打印底盘速度，不下发给狗。
# 直接运行 HTTP client 时默认先暂停，导入本模块的 Zenoh client 默认不暂停。
paused = _env_flag("INTERNVLA_START_PAUSED", __name__ == "__main__")
# 收到 replan_required 后保持底盘静止，直到 server 返回已应用新子目标的正常结果。
upper_replan_hold = False
rgb_depth_rw_lock = ReadWriteLock()
# 保护 RGB/depth 数据的读写锁，避免回调线程和规划线程同时读写。
odom_rw_lock = ReadWriteLock()
# 保护 odom 位姿和速度数据的读写锁。
mpc_rw_lock = ReadWriteLock()
# 保护 MPC 控制器对象的读写锁。


def debug_input_thread():
# 交互快捷键：R 开始/暂停规划，D 切换底盘动作拦截，Q 退出。
    global debug_mode, paused, mpc, current_control_mode, trajs_in_world, desired_v, desired_w
    global http_idx, policy_init
    while rclpy.ok():
        try:
            key = _getkey().lower()
        except (EOFError, OSError):
            return
        if key == "d":
            debug_mode = not debug_mode
            print(f"[CTRL] debug command interception: {'ON' if debug_mode else 'OFF'}")
        elif key == "r":
            paused = not paused
            if paused:
                mpc = None
                current_control_mode = ControlMode.MPC_Mode
                trajs_in_world = None
                desired_v, desired_w = 0.0, 0.0
                if manager is not None:
                    manager.hold_current_position()
                print("[PLAN] paused; controller state cleared")
            else:
                # 手动恢复被视为用户发起的新任务，保留原有 policy_init 语义。
                http_idx = -1
                policy_init = True
                print("[PLAN] resumed; next request starts a new episode")
        elif key == "q":
            print("[INFO] shutdown requested")
            rclpy.shutdown()
            return


def dual_sys_eval(image_bytes, depth_bytes, front_image_bytes, url=None):
# 把当前 RGB-D 帧通过 HTTP 发给模型 server，并解析 server 返回的动作/轨迹。
    global policy_init, http_idx, first_running_time
    if url is None:
        url = os.environ.get("INTERNVLA_SERVER_URL", "http://127.0.0.1:8848/eval_dual")
# 模型 server 地址可通过环境变量 INTERNVLA_SERVER_URL 覆盖。

    data = {"reset": policy_init, "idx": http_idx}
# reset 用来告诉 server 是否开始新 episode，idx 是当前请求序号。
    instruction = os.environ.get("INTERNVLA_INSTRUCTION")
# 可通过环境变量 INTERNVLA_INSTRUCTION 给真实部署传导航指令。
    if instruction:
        data["instruction"] = instruction
    json_data = json.dumps(data)

    policy_init = False
# 第一次请求发出后，后续请求不再 reset server。
    files = {
        'image': ('rgb_image', image_bytes, 'image/jpeg'),
        'depth': ('depth_image', depth_bytes, 'image/png'),
    }
# 按 server 约定，以 multipart/form-data 发送 RGB JPEG 和 Depth PNG。
    start = time.time()
    session = requests.Session()
    session.trust_env = False
# 禁用系统代理环境变量，避免本地 HTTP 请求被代理到错误端口。
    response = session.post(url, files=files, data={'json': json_data}, timeout=100)
# 调用 server 的 /eval_dual 接口，timeout 给得较长以覆盖首次推理或慢帧。
    print(f"response {response.text}")
    http_idx += 1
    if http_idx == 0:
        first_running_time = time.time()
    print(f"idx: {http_idx} after http {time.time() - start}")

    return json.loads(response.text)


def control_thread():
# 控制线程：持续根据当前 PID/MPC 模式输出 /cmd_vel_bridge 速度命令。
    global desired_v, desired_w
    while True:
        global current_control_mode
        if paused:
            if manager is not None:
                manager.move(0.0, 0.0, 0.0)
            time.sleep(0.1)
            continue
        if current_control_mode == ControlMode.MPC_Mode:
# MPC 模式：追踪连续轨迹 trajectory。
            odom_rw_lock.acquire_read()
            odom = manager.odom.copy() if manager.odom else None
            odom_rw_lock.release_read()
            if mpc is not None and manager is not None and odom is not None:
                local_mpc = mpc
                opt_u_controls, opt_x_states = local_mpc.solve(np.array(odom))
# MPC 根据当前 odom 和参考轨迹求解未来控制序列。
                v, w = opt_u_controls[0, 0], opt_u_controls[0, 1]
# 只取 MPC 求出的第一个控制量作为当前时刻速度命令。

                desired_v, desired_w = v, w
                manager.move(v, 0.0, w)
        elif current_control_mode == ControlMode.PID_Mode:
# PID 模式：追踪由离散动作累加出来的小目标位姿。
            odom_rw_lock.acquire_read()
            odom = manager.odom.copy() if manager.odom else None
            odom_rw_lock.release_read()
            homo_odom = manager.homo_odom.copy() if manager.homo_odom is not None else None
            vel = manager.vel.copy() if manager.vel is not None else None
            homo_goal = manager.homo_goal.copy() if manager.homo_goal is not None else None

            if homo_odom is not None and vel is not None and homo_goal is not None:
                v, w, e_p, e_r = pid.solve(homo_odom, homo_goal, vel)
# PID 根据当前位置、目标位姿和当前速度计算线速度 v 与角速度 w。
                if v < 0.0:
                    v = 0.0
                desired_v, desired_w = v, w
                manager.move(v, 0.0, w)

        time.sleep(0.1)


def planning_thread():
# 规划线程：等待新 RGB-D 帧，匹配 odom，发给 server，并更新 PID/MPC 目标。
    global trajs_in_world, upper_replan_hold, current_control_mode

    while True:
        start_time = time.time()
        DESIRED_TIME = 0.3
        time.sleep(0.05)

        if not manager.new_image_arrived:
# 没有新图像就跳过，避免重复用旧帧请求模型。
            time.sleep(0.01)
            continue
        manager.new_image_arrived = False
        rgb_depth_rw_lock.acquire_read()
        rgb_bytes = copy.deepcopy(manager.rgb_bytes)
        depth_bytes = copy.deepcopy(manager.depth_bytes)
        infer_rgb = copy.deepcopy(manager.rgb_image)
        infer_depth = copy.deepcopy(manager.depth_image)
        rgb_time = manager.rgb_time
        rgb_depth_rw_lock.release_read()
        odom_rw_lock.acquire_read()
        min_diff = 1e10
        # time_diff = 1e10
        odom_infer = None
        for odom in manager.odom_queue:
            diff = abs(odom[0] - rgb_time)
            if diff < min_diff:
                min_diff = diff
                odom_infer = copy.deepcopy(odom[1])
                # time_diff = odom[0] - rgb_time
        # odom_time = manager.odom_timestamp
        odom_rw_lock.release_read()

        if odom_infer is not None and rgb_bytes is not None and depth_bytes is not None:
# 只有 RGB、Depth、Odom 都齐了，才发给 server 做导航推理。
            global frame_data
            frame_data[http_idx] = {
                'infer_rgb': copy.deepcopy(infer_rgb),
                'infer_depth': copy.deepcopy(infer_depth),
                'infer_odom': copy.deepcopy(odom_infer),
            }
            if len(frame_data) > 100:
                del frame_data[min(frame_data.keys())]
            if paused:
                continue
            response = dual_sys_eval(rgb_bytes, depth_bytes, None)
# 调用模型 server，返回 discrete_action 或 trajectory。

            # replan_required is a protocol-level hold, not an ordinary STOP.
            # Keep ignoring stale low-level replies until the server has written
            # and applied the Upper Agent's next instruction.
            if response.get('replan_required'):
                upper_replan_hold = True
                manager.hold_current_position()
                current_control_mode = ControlMode.PID_Mode
                print(f"low-level policy held for upper-agent replan: {response.get('replan_reason', '')}")
                continue
            if upper_replan_hold:
                upper_replan_hold = False
                print("upper-agent replan released; applying the fresh low-level response")

            traj_len = 0.0
            if 'trajectory' in response:
# 连续轨迹输出：使用 MPC 控制。
                trajectory = response['trajectory']
                trajs_in_world = []
                odom = odom_infer
                traj_len = np.linalg.norm(trajectory[-1][:2])
# 轨迹末端到当前机器人的局部距离，用于观察轨迹长度。
                print(f"traj len {traj_len}")
                for i, traj in enumerate(trajectory):
                    if i < 3:
                        continue
                    x_, y_, yaw_ = odom[0], odom[1], odom[2]

                    w_T_b = np.array(
                        [
                            [np.cos(yaw_), -np.sin(yaw_), 0, x_],
                            [np.sin(yaw_), np.cos(yaw_), 0, y_],
                            [0.0, 0.0, 1.0, 0],
                            [0.0, 0.0, 0.0, 1.0],
                        ]
                    )
                    w_P = (w_T_b @ (np.array([traj[0], traj[1], 0.0, 1.0])).T)[:2]
# 把模型输出的机器人局部坐标点转换到 odom/world 坐标系。
                    trajs_in_world.append(w_P)
                trajs_in_world = np.array(trajs_in_world)
                print(f"{time.time()} update traj")

                manager.last_trajs_in_world = trajs_in_world
                mpc_rw_lock.acquire_write()
                global mpc
                if mpc is None:
                    mpc = Mpc_controller(np.array(trajs_in_world))
# 第一次收到 trajectory 时创建 MPC 控制器。
                else:
                    mpc.update_ref_traj(np.array(trajs_in_world))
# 后续收到 trajectory 时更新 MPC 参考轨迹。
                manager.request_cnt += 1
                mpc_rw_lock.release_write()
                current_control_mode = ControlMode.MPC_Mode
            elif 'discrete_action' in response:
# 离散动作输出：使用 PID 控制。
                actions = response['discrete_action']
                if actions != [5] and actions != [9]:
                    manager.incremental_change_goal(actions)
# 把离散动作序列转换成一个新的目标位姿 homo_goal。
                    current_control_mode = ControlMode.PID_Mode
# 切换到 PID 模式追踪这个目标位姿。
        else:
            print(
                f"skip planning. odom_infer: {odom_infer is not None} rgb_bytes: {rgb_bytes is not None} depth_bytes: {depth_bytes is not None}"
            )
            time.sleep(0.1)

        time.sleep(max(0, DESIRED_TIME - (time.time() - start_time)))


class Go2Manager(Node):
# Go2 ROS 管理节点：负责 odom、速度控制和 Orbbec RGB-D 采集。
    def __init__(self):
        super().__init__('go2_manager')

        qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)
        self.odom_sub = self.create_subscription(Odometry, "/odom_bridge", self.odom_callback, qos_profile)
# odom topic，提供机器人在 odom/world 坐标下的位置、朝向和速度。

        # publisher
        self.control_pub = self.create_publisher(Twist, '/cmd_vel_bridge', 5)
# 速度控制 topic，最终发布给狗子底盘执行。

        # Shared RGB-D cache consumed by planning_thread.
        self.rgb_image = None
        self.rgb_bytes = None
        self.depth_image = None
        self.depth_bytes = None
        self.new_image_arrived = False
        self.new_vis_image_arrived = False
        self.rgb_time = 0.0

        self.odom = None
        self.linear_vel = 0.0
        self.angular_vel = 0.0
        self.request_cnt = 0
        self.odom_cnt = 0
        self.odom_queue = deque(maxlen=50)
        self.odom_timestamp = 0.0

        self.last_s2_step = -1
        self.last_trajs_in_world = None
        self.last_all_trajs_in_world = None
        self.homo_odom = None
        self.homo_goal = None
        self.vel = None

        self._running = True
        self.frame_timeout_ms = max(500, int(os.environ.get("INTERNVLA_CAMERA_FRAME_TIMEOUT_MS", "1000")))
        self.reconnect_after_timeouts = max(1, int(os.environ.get("INTERNVLA_CAMERA_RECONNECT_AFTER_TIMEOUTS", "3")))
        self.pipeline = None
        for attempt in range(3):
            try:
                # 无参 Pipeline 让 Orbbec SDK 自己完成网络设备发现与 RTP 会话管理。
                self.pipeline = Pipeline()
                if self._start_orbbec_pipeline():
                    print(f"[CAM] Orbbec connected: {self.pipeline.get_device().get_device_info().get_name()}")
                    break
            except Exception as exc:
                print(f"[CAM] connection attempt {attempt + 1}/3 failed: {exc}")
            self._release_camera()
            if attempt < 2:
                time.sleep(2)
        else:
            raise RuntimeError("[CAM] unable to connect to Orbbec camera after 3 attempts")

        self.orbbec_thread = threading.Thread(target=self.orbbec_capture_thread, name="orbbec-capture", daemon=True)
        self.orbbec_thread.start()

    def _select_profile(self, sensor_type, width, height, fps, fmt=OBFormat.UNKNOWN_FORMAT):
        profile_list = self.pipeline.get_stream_profile_list(sensor_type)
        try:
            profile = profile_list.get_video_stream_profile(width, height, fmt, fps)
            if profile is not None:
                return profile
        except Exception as exc:
            print(f"[CAM] requested profile unavailable; using default: {exc}")
        return profile_list.get_default_video_stream_profile()

    def _start_orbbec_pipeline(self):
        config = Config()
        try:
            config.enable_stream(self._select_profile(OBSensorType.COLOR_SENSOR, 640, 480, 15, OBFormat.MJPG))
            config.enable_stream(self._select_profile(OBSensorType.DEPTH_SENSOR, 640, 480, 15))
            # SW depth-to-color alignment is intentionally disabled because it
            # can stall some network cameras during wait_for_frames().
        except Exception as exc:
            print(f"[CAM] stream profile configuration failed: {exc}")
            return False
        try:
            self.pipeline.start(config)
        except Exception as exc:
            print(f"[CAM] pipeline start failed: {exc}")
            return False

        print("[CAM] waiting for Orbbec stream warmup...")
        time.sleep(5)
        for attempt in range(10):
            if self.pipeline.wait_for_frames(2000) is not None:
                print(f"[CAM] stream ready after {attempt + 1} attempt(s)")
                return True
            print(f"[CAM] stream wait timeout {attempt + 1}/10")
        return False

    def _color_frame_to_bgr(self, color_frame):
        fmt = color_frame.get_format()
        width, height = color_frame.get_width(), color_frame.get_height()
        data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
        if fmt in (OBFormat.MJPG, OBFormat.COMPRESSED):
            return cv2.imdecode(data, cv2.IMREAD_COLOR)
        if fmt == OBFormat.RGB:
            return cv2.cvtColor(data.reshape((height, width, 3)), cv2.COLOR_RGB2BGR)
        if fmt == OBFormat.BGR:
            return data.reshape((height, width, 3))
        if fmt == OBFormat.RGBA:
            return cv2.cvtColor(data.reshape((height, width, 4)), cv2.COLOR_RGBA2BGR)
        if fmt == OBFormat.BGRA:
            return cv2.cvtColor(data.reshape((height, width, 4)), cv2.COLOR_BGRA2BGR)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)

    def _release_camera(self):
        try:
            if self.pipeline is not None:
                self.pipeline.stop()
        except Exception as exc:
            print(f"[CAM] pipeline stop failed: {exc}")
        finally:
            self.pipeline = None

    def _reconnect_camera(self):
        print("[CAM] reconnecting Orbbec camera...")
        self._release_camera()
        time.sleep(1)
        for attempt in range(3):
            try:
                self.pipeline = Pipeline()
                if self._start_orbbec_pipeline():
                    print(f"[CAM] reconnect succeeded on attempt {attempt + 1}")
                    return True
            except Exception as exc:
                print(f"[CAM] reconnect attempt {attempt + 1}/3 failed: {exc}")
            self._release_camera()
            if attempt < 2:
                time.sleep(2)
        self._release_camera()
        return False

    def orbbec_capture_thread(self):
        consecutive_failures = 0
        while self._running:
            if self.pipeline is None:
                time.sleep(0.5)
                continue
            try:
                frames = self.pipeline.wait_for_frames(self.frame_timeout_ms)
            except Exception as exc:
                print(f"[CAM] frame receive error: {exc}")
                frames = None
            if frames is None:
                consecutive_failures += 1
                if consecutive_failures == 1:
                    print("[CAM] no RGB-D frame received; checking RTP stream...")
                if consecutive_failures >= self.reconnect_after_timeouts:
                    print(
                        f"[CAM] {consecutive_failures} consecutive frame timeouts; "
                        "recreating the RTP session."
                    )
                    consecutive_failures = 0
                    self._reconnect_camera()
                continue
            consecutive_failures = 0
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if color_frame is None or depth_frame is None:
                continue

            bgr_image = self._color_frame_to_bgr(color_frame)
            if bgr_image is None:
                continue
            jpeg_ok, jpeg_buffer = cv2.imencode(".jpg", bgr_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            if not jpeg_ok:
                continue

            height, width = depth_frame.get_height(), depth_frame.get_width()
            depth_raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape((height, width))
            depth_m = depth_raw.astype(np.float32) * depth_frame.get_depth_scale() / 1000.0
            depth_m[~np.isfinite(depth_m)] = 0
            depth_m[depth_m < 0] = 0
            depth_uint16 = np.clip(depth_m * 10000.0, 0, 65535).astype(np.uint16)
            png_ok, png_buffer = cv2.imencode(".png", depth_uint16)
            if not png_ok:
                continue

            frame_time = time.time()
            rgb_depth_rw_lock.acquire_write()
            self.rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            self.rgb_bytes = jpeg_buffer.tobytes()
            self.depth_image = depth_m
            self.depth_bytes = png_buffer.tobytes()
            self.rgb_time = frame_time
            self.depth_time = frame_time
            self.last_rgb_time = frame_time
            self.last_depth_time = frame_time
            rgb_depth_rw_lock.release_write()
            self.new_vis_image_arrived = True
            self.new_image_arrived = True

    def shutdown(self):
        self._running = False
        self._release_camera()

    def odom_callback(self, msg):
# odom 回调：更新机器人当前 x、y、yaw 和速度。
        self.odom_cnt += 1
        odom_rw_lock.acquire_write()
        zz = msg.pose.pose.orientation.z
        ww = msg.pose.pose.orientation.w
        yaw = math.atan2(2 * zz * ww, 1 - 2 * zz * zz)
# 从四元数中提取 yaw 角，代码假设主要在平面运动。
        self.odom = [msg.pose.pose.position.x, msg.pose.pose.position.y, yaw]
# odom 的核心状态：[世界/odom 坐标 x, 世界/odom 坐标 y, 朝向 yaw]。
        self.odom_queue.append((time.time(), copy.deepcopy(self.odom)))
# 保存最近一段 odom 队列，用于和 RGB 时间戳做近似匹配。
        self.odom_timestamp = time.time()
        self.linear_vel = msg.twist.twist.linear.x
        self.angular_vel = msg.twist.twist.angular.z
# 当前机器人线速度和角速度，用于 PID 的速度反馈项。
        odom_rw_lock.release_write()

        R0 = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        self.homo_odom = np.eye(4)
# 当前 odom 位姿的 4x4 齐次变换矩阵。
        self.homo_odom[:2, :2] = R0
        self.homo_odom[:2, 3] = [msg.pose.pose.position.x, msg.pose.pose.position.y]
        self.vel = [msg.twist.twist.linear.x, msg.twist.twist.angular.z]
# PID 控制器使用的当前速度 [线速度, 角速度]。

        if self.odom_cnt == 1:
            self.homo_goal = self.homo_odom.copy()
# 第一帧 odom 到来时，把当前位姿作为初始目标，避免目标为空。

    def incremental_change_goal(self, actions):
# 把 server 返回的离散动作序列累加成一个新的目标位姿 homo_goal。
        if self.homo_goal is None:
            raise ValueError("Please initialize homo_goal before change it!")
        homo_goal = self.homo_odom.copy()
# 每次根据当前 odom 重新构造目标，避免旧目标累积过远。
        for each_action in actions:
            if each_action == 0:
                pass
# 0 表示 STOP/pass，不改变目标。
            elif each_action == 1:
                yaw = math.atan2(homo_goal[1, 0], homo_goal[0, 0])
                homo_goal[0, 3] += 0.25 * np.cos(yaw)
                homo_goal[1, 3] += 0.25 * np.sin(yaw)
# 1 表示向当前朝向前进 0.25 米。
            elif each_action == 2:
                angle = math.radians(15)
                rotation_matrix = np.array(
                    [[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]]
                )
                homo_goal[:3, :3] = np.dot(rotation_matrix, homo_goal[:3, :3])
# 2 表示左转 15 度。
            elif each_action == 3:
                angle = -math.radians(15.0)
                rotation_matrix = np.array(
                    [[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]]
                )
                homo_goal[:3, :3] = np.dot(rotation_matrix, homo_goal[:3, :3])
# 3 表示右转 15 度。
        self.homo_goal = homo_goal
# 更新 PID 要追踪的新目标位姿。

    def hold_current_position(self):
# 重规划期间的安全保持：立刻发布零速度，并把 PID 目标锁定在当前 odom 位姿。
# 这不改变 policy_init，也不会新开实验；收到新计划后会被正常目标覆盖。
        if self.homo_odom is not None:
            self.homo_goal = self.homo_odom.copy()
        self.move(0.0, 0.0, 0.0)

    def move(self, vx, vy, vyaw):
# 发布速度命令到 /cmd_vel_bridge。
        # Debug 只拦截非零运动命令；显式停车必须始终发给底盘。
        if debug_mode and not paused and any(abs(value) > 1e-6 for value in (vx, vy, vyaw)):
            print(f"[CTRL] intercepted vx={vx:.3f}, vy={vy:.3f}, yaw={vyaw:.3f} ({current_control_mode.name})")
            return
        request = Twist()
        request.linear.x = vx
        request.linear.y = 0.0
        request.angular.z = vyaw
# Go2 底盘主要使用前向线速度 vx 和偏航角速度 vyaw。

        self.control_pub.publish(request)


if __name__ == '__main__':
    acquire_client_lock()
    control_thread_instance = threading.Thread(target=control_thread)
    planning_thread_instance = threading.Thread(target=planning_thread)
    control_thread_instance.daemon = True
    planning_thread_instance.daemon = True
    rclpy.init()

    if sys.stdin.isatty():
        debug_thread_instance = threading.Thread(target=debug_input_thread, name="client-keyboard", daemon=True)
        debug_thread_instance.start()
        print("[INFO] press R to start/pause, D to toggle command interception, Q to exit")
    else:
        print("[INFO] non-interactive client; set INTERNVLA_START_PAUSED=0 to start planning immediately")

    try:
        manager = Go2Manager()

        control_thread_instance.start()
        planning_thread_instance.start()

        rclpy.spin(manager)
    except KeyboardInterrupt:
        pass
    finally:
        if manager is not None:
            manager.shutdown()
            manager.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
