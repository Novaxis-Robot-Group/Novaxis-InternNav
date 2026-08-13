"""Standalone Go2 HTTP client based on the user-provided camera/control client.

This file deliberately leaves ``http_internvla_client.py`` untouched.  It keeps
the original start-paused, keyboard-control, old-process cleanup and Orbbec
Pipeline() behaviour, while retaining the Upper-Agent replan safety handoff.
"""

import atexit
import copy
import json
import math
import os
import subprocess
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
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from controllers import Mpc_controller, PID_controller
from thread_utils import ReadWriteLock


def _kill_old_clients():
    """Keep the supplied client's cleanup behaviour for stale RTP receivers."""
    script_name, current_pid = os.path.basename(sys.argv[0]), os.getpid()
    try:
        output = subprocess.check_output(["pgrep", "-f", script_name], timeout=5, stderr=subprocess.STDOUT).decode()
        for value in output.strip().splitlines():
            pid = int(value)
            if pid not in (0, current_pid):
                os.kill(pid, 9)
    except (subprocess.CalledProcessError, ValueError, OSError):
        pass


class ControlMode(Enum):
    PID_Mode = 1
    MPC_Mode = 2


frame_data = {}
policy_init = True
mpc = None
pid = PID_controller(Kp_trans=2.0, Kd_trans=0.0, Kp_yaw=1.5, Kd_yaw=0.0, max_v=0.6, max_w=0.5)
http_idx = -1
first_running_time = 0.0
manager = None
current_control_mode = ControlMode.MPC_Mode
trajs_in_world = None
desired_v = desired_w = 0.0
debug_mode = False
paused = True
upper_replan_hold = False
rgb_depth_rw_lock = ReadWriteLock()
odom_rw_lock = ReadWriteLock()
mpc_rw_lock = ReadWriteLock()


def _getkey():
    fd = sys.stdin.fileno()
    if not hasattr(_getkey, "old_settings"):
        _getkey.old_settings = termios.tcgetattr(fd)
        atexit.register(lambda: termios.tcsetattr(fd, termios.TCSADRAIN, _getkey.old_settings))
    settings = termios.tcgetattr(fd)
    settings[3] &= ~(termios.ICANON | termios.ECHO)
    settings[6][termios.VMIN], settings[6][termios.VTIME] = 1, 0
    try:
        termios.tcsetattr(fd, termios.TCSADRAIN, settings)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, _getkey.old_settings)


def debug_input_thread():
    global debug_mode, paused, mpc, current_control_mode, trajs_in_world, desired_v, desired_w, http_idx, policy_init
    while True:
        try:
            key = _getkey().lower()
        except (EOFError, OSError):
            return
        if key == "d":
            debug_mode = not debug_mode
            print(f"[DEBUG] {'ON: commands intercepted' if debug_mode else 'OFF: commands published'}")
        elif key == "r":
            paused = not paused
            if paused:
                mpc, trajs_in_world, desired_v, desired_w = None, None, 0.0, 0.0
                current_control_mode = ControlMode.MPC_Mode
                if manager is not None:
                    manager.hold_current_position()
                print("[PLAN] paused")
            else:
                http_idx, policy_init = -1, True
                print("[PLAN] resumed; next request resets the model episode")
        elif key == "q":
            print("[INFO] shutdown requested")
            rclpy.shutdown()
            return


def dual_sys_eval(image_bytes, depth_bytes, front_image_bytes=None, url=None):
    global policy_init, http_idx, first_running_time
    url = url or os.environ.get("INTERNVLA_SERVER_URL", "http://127.0.0.1:8848/eval_dual")
    data = {"reset": policy_init, "idx": http_idx}
    instruction = os.environ.get("INTERNVLA_INSTRUCTION")
    if instruction:
        data["instruction"] = instruction
    policy_init = False
    session = requests.Session()
    session.trust_env = False
    started = time.time()
    response = session.post(url, files={"image": ("rgb_image", image_bytes, "image/jpeg"), "depth": ("depth_image", depth_bytes, "image/png")}, data={"json": json.dumps(data)}, timeout=100)
    response.raise_for_status()
    http_idx += 1
    if http_idx == 0:
        first_running_time = time.time()
    print(f"[HTTP] idx={http_idx} elapsed={time.time() - started:.3f}s")
    return response.json()


def control_thread():
    global desired_v, desired_w
    while rclpy.ok():
        if paused:
            manager.move(0.0, 0.0, 0.0)
        elif current_control_mode == ControlMode.MPC_Mode:
            with_read(odom_rw_lock)
            odom = manager.odom.copy() if manager.odom else None
            release_read(odom_rw_lock)
            if mpc is not None and odom is not None:
                controls, _ = mpc.solve(np.array(odom))
                desired_v, desired_w = controls[0, 0], controls[0, 1]
                manager.move(desired_v, 0.0, desired_w)
        else:
            homo_odom = manager.homo_odom.copy() if manager.homo_odom is not None else None
            goal = manager.homo_goal.copy() if manager.homo_goal is not None else None
            vel = manager.vel.copy() if manager.vel is not None else None
            if homo_odom is not None and goal is not None and vel is not None:
                desired_v, desired_w, _, _ = pid.solve(homo_odom, goal, vel)
                manager.move(max(0.0, desired_v), 0.0, desired_w)
        time.sleep(0.1)


def with_read(lock):
    lock.acquire_read()


def release_read(lock):
    lock.release_read()


def planning_thread():
    global mpc, trajs_in_world, current_control_mode, upper_replan_hold
    while rclpy.ok():
        started = time.time()
        time.sleep(0.05)
        if not manager.new_image_arrived:
            continue
        manager.new_image_arrived = False
        rgb_depth_rw_lock.acquire_read()
        rgb, depth, rgb_time = manager.rgb_bytes, manager.depth_bytes, manager.rgb_time
        rgb_depth_rw_lock.release_read()
        odom_rw_lock.acquire_read()
        candidates = list(manager.odom_queue)
        odom_rw_lock.release_read()
        odom = min(candidates, key=lambda item: abs(item[0] - rgb_time))[1] if candidates else None
        if paused or rgb is None or depth is None or odom is None:
            continue
        response = dual_sys_eval(rgb, depth)
        if response.get("replan_required"):
            upper_replan_hold = True
            manager.hold_current_position()
            current_control_mode = ControlMode.PID_Mode
            print(f"[PLAN] holding for Upper-Agent replan: {response.get('replan_reason', '')}")
            continue
        if upper_replan_hold:
            upper_replan_hold = False
            print("[PLAN] Upper-Agent replan released; applying fresh response")
        if "trajectory" in response:
            trajectory = response["trajectory"]
            world = []
            x, y, yaw = odom
            transform = np.array([[np.cos(yaw), -np.sin(yaw), 0, x], [np.sin(yaw), np.cos(yaw), 0, y], [0, 0, 1, 0], [0, 0, 0, 1]])
            for point in trajectory[3:]:
                world.append((transform @ np.array([point[0], point[1], 0, 1]))[:2])
            if world:
                trajs_in_world = np.array(world)
                mpc_rw_lock.acquire_write()
                if mpc is None:
                    mpc = Mpc_controller(trajs_in_world)
                else:
                    mpc.update_ref_traj(trajs_in_world)
                mpc_rw_lock.release_write()
                current_control_mode = ControlMode.MPC_Mode
        elif "discrete_action" in response:
            if response["discrete_action"] not in ([5], [9]):
                manager.incremental_change_goal(response["discrete_action"])
                current_control_mode = ControlMode.PID_Mode
        time.sleep(max(0.0, 0.3 - (time.time() - started)))


class Go2Manager(Node):
    def __init__(self):
        super().__init__("go2_manager")
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)
        self.create_subscription(Odometry, "/odom_bridge", self.odom_callback, qos)
        self.control_pub = self.create_publisher(Twist, "/cmd_vel_bridge", 5)
        self.rgb_bytes = self.depth_bytes = self.rgb_image = self.depth_image = None
        self.rgb_time = 0.0
        self.new_image_arrived = False
        self.odom = self.homo_odom = self.homo_goal = self.vel = None
        self.odom_queue = deque(maxlen=50)
        self._running = True
        self.frame_timeout_ms = max(100, int(os.environ.get("INTERNVLA_CAMERA_FRAME_TIMEOUT_MS", "1000")))
        self.reconnect_after_timeouts = max(1, int(os.environ.get("INTERNVLA_CAMERA_RECONNECT_AFTER_TIMEOUTS", "15")))
        self.pipeline = None
        for attempt in range(3):
            self.pipeline = Pipeline()
            if self._start_pipeline():
                break
            self._release_camera()
            time.sleep(2)
        else:
            raise RuntimeError("[CAM] unable to connect to Orbbec after 3 attempts")
        threading.Thread(target=self.orbbec_capture_thread, name="orbbec-capture", daemon=True).start()

    def _profile(self, sensor, width, height, fps, fmt=OBFormat.UNKNOWN_FORMAT):
        profiles = self.pipeline.get_stream_profile_list(sensor)
        try:
            return profiles.get_video_stream_profile(width, height, fmt, fps) or profiles.get_default_video_stream_profile()
        except Exception:
            return profiles.get_default_video_stream_profile()

    def _start_pipeline(self):
        try:
            config = Config()
            config.enable_stream(self._profile(OBSensorType.COLOR_SENSOR, 640, 480, 15, OBFormat.MJPG))
            config.enable_stream(self._profile(OBSensorType.DEPTH_SENSOR, 640, 480, 15))
            self.pipeline.start(config)
            time.sleep(5)
            return any(self.pipeline.wait_for_frames(2000) is not None for _ in range(10))
        except Exception as exc:
            print(f"[CAM] pipeline start failed: {exc}")
            return False

    def _release_camera(self):
        try:
            if self.pipeline is not None:
                self.pipeline.stop()
        except Exception:
            pass
        self.pipeline = None

    def _reconnect_camera(self):
        print("[CAM] reconnecting Orbbec RTP session")
        self._release_camera()
        time.sleep(1)
        self.pipeline = Pipeline()
        if not self._start_pipeline():
            self._release_camera()

    @staticmethod
    def _bgr(frame):
        data = np.frombuffer(frame.get_data(), dtype=np.uint8)
        if frame.get_format() in (OBFormat.MJPG, OBFormat.COMPRESSED):
            return cv2.imdecode(data, cv2.IMREAD_COLOR)
        return data.reshape((frame.get_height(), frame.get_width(), 3))

    def orbbec_capture_thread(self):
        failures = 0
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
                failures += 1
                if failures >= self.reconnect_after_timeouts:
                    print(f"[CAM] {failures} consecutive RTP timeouts; reconnecting")
                    failures = 0
                    self._reconnect_camera()
                continue
            failures = 0
            color, depth = frames.get_color_frame(), frames.get_depth_frame()
            if color is None or depth is None:
                continue
            bgr = self._bgr(color)
            if bgr is None:
                continue
            ok_jpg, jpg = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            raw = np.frombuffer(depth.get_data(), dtype=np.uint16).reshape((depth.get_height(), depth.get_width()))
            meters = raw.astype(np.float32) * depth.get_depth_scale() / 1000.0
            ok_png, png = cv2.imencode(".png", np.clip(meters * 10000.0, 0, 65535).astype(np.uint16))
            if not ok_jpg or not ok_png:
                continue
            rgb_depth_rw_lock.acquire_write()
            self.rgb_image, self.depth_image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), meters
            self.rgb_bytes, self.depth_bytes, self.rgb_time = jpg.tobytes(), png.tobytes(), time.time()
            rgb_depth_rw_lock.release_write()
            self.new_image_arrived = True

    def odom_callback(self, msg):
        z, w = msg.pose.pose.orientation.z, msg.pose.pose.orientation.w
        yaw = math.atan2(2 * z * w, 1 - 2 * z * z)
        odom_rw_lock.acquire_write()
        self.odom = [msg.pose.pose.position.x, msg.pose.pose.position.y, yaw]
        self.odom_queue.append((time.time(), copy.deepcopy(self.odom)))
        odom_rw_lock.release_write()
        self.homo_odom = np.eye(4)
        self.homo_odom[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
        self.homo_odom[:2, 3] = self.odom[:2]
        self.vel = [msg.twist.twist.linear.x, msg.twist.twist.angular.z]
        if self.homo_goal is None:
            self.homo_goal = self.homo_odom.copy()

    def incremental_change_goal(self, actions):
        if self.homo_odom is None:
            return
        goal = self.homo_odom.copy()
        for action in actions:
            yaw = math.atan2(goal[1, 0], goal[0, 0])
            if action == 1:
                goal[0, 3] += 0.25 * math.cos(yaw)
                goal[1, 3] += 0.25 * math.sin(yaw)
            elif action in (2, 3):
                angle = math.radians(15 if action == 2 else -15)
                rotation = np.array([[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]])
                goal[:3, :3] = rotation @ goal[:3, :3]
        self.homo_goal = goal

    def hold_current_position(self):
        if self.homo_odom is not None:
            self.homo_goal = self.homo_odom.copy()
        self.move(0.0, 0.0, 0.0)

    def move(self, vx, vy, vyaw):
        if debug_mode and not paused:
            print(f"[CTRL] intercepted vx={vx:.3f}, vyaw={vyaw:.3f}")
            return
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.angular.z = vx, 0.0, vyaw
        self.control_pub.publish(msg)

    def shutdown(self):
        self._running = False
        self._release_camera()


if __name__ == "__main__":
    _kill_old_clients()
    rclpy.init()
    print("[INFO] starts paused: R=run/pause, D=debug, Q=quit")
    threading.Thread(target=debug_input_thread, daemon=True).start()
    try:
        manager = Go2Manager()
        threading.Thread(target=control_thread, daemon=True).start()
        threading.Thread(target=planning_thread, daemon=True).start()
        rclpy.spin(manager)
    except KeyboardInterrupt:
        pass
    finally:
        if manager is not None:
            manager.shutdown()
            manager.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
