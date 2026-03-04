"""
wheeltec_vln_client.py
======================
ROS2 client for StreamVLN real-world deployment on a Wheeltec differential-drive robot.

Ported from go2_vln_client.py (Unitree Go2) with the following changes:
  - Motion command: geometry_msgs/Twist on /cmd_vel  (vs Unitree SDK)
  - Odometry:       nav_msgs/Odometry (quaternion)    (vs SportModeState)
  - Image capture:  ApproximateTimeSynchronizer for synced RGB+Depth
  - Instruction:    supplied via CLI arg, forwarded to server each request
  - Speed limits:   0.25 m/s linear / 0.175 rad/s angular (in-place, <10 deg/s)

Architecture (2 threads + ROS spin):
  ROS spin   – rgb_depth_callback / odom_callback populate shared state
  Planning   – waits for should_plan flag, calls HTTP server, updates homo_goal
  Control    – 10 Hz PID loop, publishes Twist, triggers replan on convergence
"""

import argparse
import io
import json
import math
import sys
import threading
import time

import numpy as np
import PIL.Image as PIL_Image
import requests
import rclpy
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Image

# Reuse the PID controller and ReadWriteLock from Go2 deployment
from pid_controller import PID_controller, ReadWriteLock

# ── Camera configs ────────────────────────────────────────────────────────────
# Gemini 336L native: 1280×800 @ 30fps, RGB FOV H94°×V68°, Depth FOV H90°×V65°
# Two common output configurations depending on SDK setting:
#   "gemini_336l"     – 1280×800 native, intrinsics from FOV (deploy-time calibration recommended)
#   "gemini_336l_720" – 1280×720 cropped, calibrated values from InternNav project
CAMERA_CONFIGS = {
    "gemini_336l": {
        "width": 1280,
        "height": 800,
        "fx": 597.0,   # ≈ 640 / tan(47°), from RGB H94° FOV
        "fy": 593.0,   # ≈ 400 / tan(34°), from RGB V68° FOV
        "cx": 640.0,
        "cy": 400.0,
        "rgb_height": 0.35,  # camera mounting height above ground (metres)
    },
    "gemini_336l_720": {
        # SDK configured to output 1280×720 (crops top/bottom of native 1280×800)
        # Intrinsics from InternNav real-world calibration
        "width": 1280,
        "height": 720,
        "fx": 607.45,
        "fy": 607.40,
        "cx": 639.19,
        "cy": 361.75,
        "rgb_height": 0.35,
    },
    "astra_s": {
        "width": 640,
        "height": 480,
        "fx": 570.3,
        "fy": 570.3,
        "cx": 319.5,
        "cy": 239.5,
        "rgb_height": 0.35,
    },
}

# ── Wheeltec hardware limits ──────────────────────────────────────────────────
MAX_V = 0.25    # m/s  (recommended navigation: 0.15-0.25)
MAX_W = 0.175   # rad/s = 10 deg/s  (in-place rotation hardware recommendation)

# ── Safety / depth (Gemini 336L optimal range: 0.25-6.0 m) ───────────────────
COLLISION_DIST    = 0.60   # m  – stop if closer than this
DEPTH_MIN_VALID   = 0.17   # m  (Gemini 336L minimum range)
DEPTH_MAX_VALID   = 20.0   # m  (Gemini 336L maximum range)

# ── Replan convergence thresholds ────────────────────────────────────────────
REPLAN_POS_THRESH = 0.10   # m
REPLAN_YAW_THRESH = 0.10   # rad

# ── Odom downsampling (keep control loop light) ───────────────────────────────
ODOM_DOWNSAMPLE = 5

# ── Global singletons ─────────────────────────────────────────────────────────
policy_init = True
manager: "WheeltecVlnManager | None" = None
pid = PID_controller(
    Kp_trans=3.0, Kd_trans=0.5,
    Kp_yaw=3.0,   Kd_yaw=0.5,
    max_v=MAX_V,   max_w=MAX_W,
)

rgb_rw_lock   = ReadWriteLock()
depth_rw_lock = ReadWriteLock()
odom_rw_lock  = ReadWriteLock()


# ── Helpers ───────────────────────────────────────────────────────────────────
def yaw_from_quaternion(q) -> float:
    """Extract yaw from a geometry_msgs/Quaternion."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def collision_check(depth: np.ndarray) -> bool:
    """Return True if the centre region of the depth image is too close."""
    h, w = depth.shape
    roi = depth[h // 3 : 2 * h // 3, w // 3 : 2 * w // 3]
    valid = roi[(roi > DEPTH_MIN_VALID) & (roi < DEPTH_MAX_VALID)]
    return len(valid) > 0 and float(valid.min()) < COLLISION_DIST


def build_intrinsic_matrix(cfg: dict) -> np.ndarray:
    fx, fy, cx, cy = cfg["fx"], cfg["fy"], cfg["cx"], cfg["cy"]
    return np.array([
        [fx,  0., cx, 0.],
        [ 0., fy, cy, 0.],
        [ 0.,  0.,  1., 0.],
        [ 0.,  0.,  0., 1.],
    ], dtype=np.float64)


# ── HTTP server interface ─────────────────────────────────────────────────────
def eval_vln(image_bgr: np.ndarray, instruction: str, url: str) -> list:
    """Send an RGB frame to the StreamVLN inference server; return action list."""
    global policy_init

    # BGR (OpenCV) → RGB (PIL)
    image_rgb = image_bgr[:, :, ::-1]
    pil_img = PIL_Image.fromarray(image_rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="jpeg")
    buf.seek(0)

    payload = json.dumps({"reset": policy_init, "instruction": instruction})
    policy_init = False

    t0 = time.time()
    response = requests.post(
        url,
        files={"image": ("rgb_image", buf, "image/jpeg")},
        data={"json": payload},
        timeout=150,
    )
    print(f"[eval_vln] server round-trip: {time.time() - t0:.3f}s | "
          f"response: {response.text[:80]}")

    return json.loads(response.text)["action"]


# ── Thread: control (10 Hz) ───────────────────────────────────────────────────
def control_thread():
    while True:
        homo_odom = manager.homo_odom.copy() if manager.homo_odom is not None else None
        vel       = manager.vel.copy()       if manager.vel       is not None else None
        homo_goal = manager.homo_goal.copy() if manager.homo_goal is not None else None

        e_p = e_r = 0.0
        if homo_odom is not None and vel is not None and homo_goal is not None:
            v, w, e_p, e_r = pid.solve(homo_odom, homo_goal, vel)
            manager.move(v, w)

        if abs(e_p) < REPLAN_POS_THRESH and abs(e_r) < REPLAN_YAW_THRESH:
            manager.trigger_replan()

        time.sleep(0.1)


# ── Thread: planning ──────────────────────────────────────────────────────────
def planning_thread():
    while True:
        if not manager.should_plan:
            time.sleep(0.05)
            continue

        print("[planning] acquiring new plan …")

        rgb_rw_lock.acquire_read()
        rgb_image   = manager.rgb_image
        instruction = manager.instruction
        server_url  = manager.server_url
        rgb_rw_lock.release_read()

        if rgb_image is None:
            time.sleep(0.1)
            continue

        # Optional safety: skip planning while obstacle is too close
        depth_rw_lock.acquire_read()
        depth = manager.depth_image
        depth_rw_lock.release_read()
        if depth is not None and collision_check(depth):
            print("[planning] obstacle detected – waiting …")
            manager.move(0.0, 0.0)
            time.sleep(0.5)
            continue

        actions = eval_vln(rgb_image, instruction, server_url)
        print(f"[planning] actions received: {actions}")

        odom_rw_lock.acquire_write()
        manager.should_plan = False
        manager.request_cnt += 1
        manager.incremental_change_goal(actions)
        odom_rw_lock.release_write()

        time.sleep(0.1)


# ── ROS2 Node ─────────────────────────────────────────────────────────────────
class WheeltecVlnManager(Node):
    def __init__(self, instruction: str, server_url: str, camera: str):
        super().__init__("wheeltec_vln_manager")

        self.instruction = instruction
        self.server_url  = server_url
        self.cam_cfg     = CAMERA_CONFIGS[camera]

        _sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # ── RGB subscriber (always required) ─────────────────────────────────
        self._rgb_sub = self.create_subscription(
            Image, "/camera/color/image_raw", self._rgb_callback, _sensor_qos
        )

        # ── Depth subscriber (optional – enables collision check) ─────────────
        self._depth_sub = self.create_subscription(
            Image, "/camera/depth/image_raw", self._depth_callback, _sensor_qos
        )

        # ── Odometry subscriber ───────────────────────────────────────────────
        self._odom_sub = self.create_subscription(
            Odometry, "/odom", self._odom_callback, 10
        )

        # ── Velocity command publisher ────────────────────────────────────────
        self._cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        # ── Shared state ──────────────────────────────────────────────────────
        self._bridge      = CvBridge()
        self.rgb_image    = None   # BGR numpy (H, W, 3)
        self.depth_image  = None   # float32 metres (H, W)
        self.homo_odom    = None   # 4×4 np.ndarray
        self.homo_goal    = None   # 4×4 np.ndarray
        self.vel          = None   # [vx, wz]
        self.request_cnt  = 0
        self._odom_cnt    = 0
        self.should_plan  = False

    # ── Callbacks ─────────────────────────────────────────────────────────────
    def _rgb_callback(self, rgb_msg: Image):
        rgb_rw_lock.acquire_write()
        self.rgb_image = self._bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        rgb_rw_lock.release_write()

    def _depth_callback(self, depth_msg: Image):
        depth_rw_lock.acquire_write()
        raw = self._bridge.imgmsg_to_cv2(depth_msg, "16UC1").astype(np.float32)
        raw /= 1000.0
        raw[~np.isfinite(raw)] = 0.0
        raw[raw < 0.0] = 0.0
        self.depth_image = raw
        depth_rw_lock.release_write()

    def _odom_callback(self, msg: Odometry):
        self._odom_cnt += 1
        if self._odom_cnt % ODOM_DOWNSAMPLE != 0:
            return

        yaw    = yaw_from_quaternion(msg.pose.pose.orientation)
        cos_y  = math.cos(yaw)
        sin_y  = math.sin(yaw)

        odom_rw_lock.acquire_write()
        self.homo_odom = np.eye(4)
        self.homo_odom[0, 0] =  cos_y
        self.homo_odom[0, 1] = -sin_y
        self.homo_odom[1, 0] =  sin_y
        self.homo_odom[1, 1] =  cos_y
        self.homo_odom[0, 3] = msg.pose.pose.position.x
        self.homo_odom[1, 3] = msg.pose.pose.position.y
        self.vel = [
            msg.twist.twist.linear.x,
            msg.twist.twist.angular.z,
        ]
        # Initialise goal pose on first valid odom message
        if self._odom_cnt == ODOM_DOWNSAMPLE:
            self.homo_goal  = self.homo_odom.copy()
            self.should_plan = True
        odom_rw_lock.release_write()

    # ── Goal management ───────────────────────────────────────────────────────
    def trigger_replan(self):
        self.should_plan = True

    def incremental_change_goal(self, actions: list):
        """Translate a sequence of discrete actions into an updated homo_goal."""
        if self.homo_goal is None:
            raise RuntimeError("homo_goal has not been initialised yet")

        g = self.homo_goal.copy()
        for a in actions:
            if a == 0:
                # STOP – do not update goal so control loop converges in place
                pass
            elif a == 1:
                # Forward 0.25 m along current heading
                yaw = math.atan2(g[1, 0], g[0, 0])
                g[0, 3] += 0.25 * math.cos(yaw)
                g[1, 3] += 0.25 * math.sin(yaw)
            elif a == 2:
                # Rotate left 15°
                angle = math.radians(15.0)
                c, s = math.cos(angle), math.sin(angle)
                R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
                g[:3, :3] = R @ g[:3, :3]
            elif a == 3:
                # Rotate right 15°
                angle = math.radians(-15.0)
                c, s = math.cos(angle), math.sin(angle)
                R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
                g[:3, :3] = R @ g[:3, :3]
        self.homo_goal = g

    # ── Motion command ────────────────────────────────────────────────────────
    def move(self, vx: float, wz: float):
        msg = Twist()
        msg.linear.x  = float(np.clip(vx, -MAX_V, MAX_V))
        msg.angular.z = float(np.clip(wz, -MAX_W, MAX_W))
        self._cmd_vel_pub.publish(msg)

    def stop(self):
        self.move(0.0, 0.0)


# ── Entry point ───────────────────────────────────────────────────────────────
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="StreamVLN ROS2 client for Wheeltec robot"
    )
    parser.add_argument(
        "--server",
        default="http://localhost:8909/eval_vln",
        help="URL of the StreamVLN HTTP inference server",
    )
    parser.add_argument(
        "--camera",
        default="gemini_336l",
        choices=list(CAMERA_CONFIGS.keys()),
        help="gemini_336l (1280×800 native) | gemini_336l_720 (1280×720, calibrated) | astra_s",
    )
    parser.add_argument(
        "--instruction",
        default="Navigate to the destination.",
        help="Navigation instruction sent to the VLN model",
    )
    return parser.parse_args(argv)


def main():
    global manager

    args = parse_args()

    control_t  = threading.Thread(target=control_thread,  daemon=True)
    planning_t = threading.Thread(target=planning_thread, daemon=True)

    rclpy.init()
    try:
        manager = WheeltecVlnManager(
            instruction=args.instruction,
            server_url=args.server,
            camera=args.camera,
        )

        control_t.start()
        planning_t.start()

        rclpy.spin(manager)
    except KeyboardInterrupt:
        pass
    finally:
        if manager is not None:
            manager.stop()
            manager.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
