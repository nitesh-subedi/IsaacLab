#!/usr/bin/env python3
import os
import time
import numpy as np
import cv2

import rospy
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist

# Import your agent
# Make sure this path matches your actual project layout.
# If your class file is at: /workspace/robotvlm/decouple_rl/models/cross_attention_infer.py
# then importing should work like below after adding /workspace/robotvlm to sys.path.
import sys
sys.path.insert(0, "/workspace/robotvlm")

from decouple_rl.models.cross_attention_infer import VLMCrossAttentionAgent


class PolicyNode:
    def __init__(self):
        # Params
        self.topic = rospy.get_param("~topic", "/camera/image/compressed")
        self.prompt = rospy.get_param("~prompt", "Navigate to the green plant right of the red chair.")  # you can update per goal
        self.model_type = rospy.get_param("~model_type", "siglip")

        self.adapter_path = rospy.get_param(
            "~adapter_path",
            "/workspace/IsaacLab/custom_models/adapters/adapter_ep2_siglip_working.pt",
        )
        self.action_head_path = rospy.get_param(
            "~action_head_path",
            "/workspace/IsaacLab/custom_models/action_head_from_new_model.pt",
        )

        device = rospy.get_param("~device", "cuda")  # "cuda" or "cpu"

        rospy.loginfo("Loading policy...")
        self.policy = VLMCrossAttentionAgent(
            self.adapter_path,
            self.action_head_path,
            device=None if device == "cuda" else __import__("torch").device("cpu"),
            model_type=self.model_type,
        )

        # Jetbot differential drive geometry (override via ROS params)
        self.wheel_radius = rospy.get_param("~wheel_radius", 0.0325)   # metres — 65 mm wheel (Waveshare Jetbot)
        self.wheel_base   = rospy.get_param("~wheel_base",   0.1120)   # metres — wheel centre-to-centre (Waveshare Jetbot)

        # Smoothing: EMA alpha (0 < alpha <= 1). Lower = smoother but more lag.
        self.alpha = rospy.get_param("~smooth_alpha", 0.2)
        # Max velocity clamps
        self.max_v = rospy.get_param("~max_v", 0.3)   # m/s
        self.max_w = rospy.get_param("~max_w", 1.5)   # rad/s

        # EMA state
        self._v_smooth = 0.0
        self._w_smooth = 0.0

        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

        # Image saving
        self.save_dir = rospy.get_param("~save_dir", "/workspace/IsaacLab")
        self._latest_path = os.path.join(self.save_dir, "latest_frame.jpg")
        # Each run gets its own timestamped folder inside vlm_test_lab/
        run_ts = time.strftime("%Y%m%d_%H%M%S")
        self._frames_dir = os.path.join(self.save_dir, "vlm_test_lab", run_ts)
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self._frames_dir, exist_ok=True)
        self._frame_idx = 0
        rospy.loginfo("Saving frames to %s", self._frames_dir)

        self.last_t = time.time()
        self.count = 0

        self.sub = rospy.Subscriber(
            self.topic,
            CompressedImage,
            self.cb,
            queue_size=1,
            buff_size=2**24,  # important for big compressed frames
        )
        rospy.loginfo("Subscribed to %s", self.topic)

    def cb(self, msg: CompressedImage):
        # Decode JPEG bytes -> BGR image
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            rospy.logwarn_throttle(2.0, "Failed to decode compressed image.")
            return

        # Convert to RGB numpy for your agent (it accepts np.ndarray)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Save latest frame (overwrites)
        ok = cv2.imwrite(self._latest_path, frame_bgr)
        if not ok:
            rospy.logwarn_throttle(5.0, "cv2.imwrite failed — cannot write to %s", self._latest_path)

        # Save timestamped frame
        ts_path = os.path.join(self._frames_dir, f"frame_{self._frame_idx:06d}.jpg")
        cv2.imwrite(ts_path, frame_bgr)
        self._frame_idx += 1

        # Run policy
        t0 = time.time()
        action = self.policy.predict(frame_rgb, self.prompt)
        dt = (time.time() - t0) * 1000.0

        # action = [omega_left, omega_right]  (rad/s per wheel)
        omega_l = float(action[0])
        omega_r = float(action[1])

        # Differential drive kinematics -> v (m/s), w (rad/s)
        r = self.wheel_radius
        L = self.wheel_base
        v_raw = r * (omega_r + omega_l) / 2.0
        w_raw = r * (omega_r - omega_l) / L

        # Clamp raw values before smoothing
        v_raw = float(np.clip(v_raw, -self.max_v, self.max_v))
        w_raw = float(np.clip(w_raw, -self.max_w, self.max_w))

        # Exponential moving average smoothing
        a = self.alpha
        self._v_smooth = a * v_raw + (1.0 - a) * self._v_smooth
        self._w_smooth = a * w_raw + (1.0 - a) * self._w_smooth

        v = self._v_smooth
        w = self._w_smooth

        twist = Twist()
        twist.linear.x  = v
        twist.angular.z = -w * 0.2
        self.cmd_vel_pub.publish(twist)

        self.count += 1
        now = time.time()
        if now - self.last_t >= 2.0:
            hz = self.count / (now - self.last_t)
            rospy.loginfo(
                "Policy rate: %.2f Hz | last infer: %.1f ms | "
                "omega_l=%.3f omega_r=%.3f -> v=%.3f w=%.3f",
                hz, dt, omega_l, omega_r, v, w,
            )
            self.last_t = now
            self.count = 0


def main():
    rospy.init_node("vlm_policy_sub", anonymous=False)
    _ = PolicyNode()
    rospy.spin()


if __name__ == "__main__":
    main()