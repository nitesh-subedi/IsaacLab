#!/usr/bin/env python3
import os
import time
import numpy as np
import cv2

import rospy
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist

import threading
import requests
try:
    import speech_recognition as sr
except ImportError:
    sr = None
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
        
        self.enable_speech = rospy.get_param("~enable_speech", True)
        if self.enable_speech:
            self.prompt = None  # Wait for user speech before moving
        else:
            self.prompt = rospy.get_param("~prompt", "Navigate to the green plant right of the red chair.")
            
        self.manual_override = None
        self.speed_multiplier = 1.0

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

        # Speech control
        if self.enable_speech:
            if sr is None:
                rospy.logwarn("Speech control enabled but speech_recognition not found. Please install: pip install SpeechRecognition pyaudio openai-whisper")
                self.prompt = rospy.get_param("~prompt", "Navigate to the green plant right of the red chair.") # fallback
            else:
                self.ollama_model = rospy.get_param("~ollama_model", "llama3")
                self.speech_thread = threading.Thread(target=self._speech_loop)
                self.speech_thread.daemon = True
                self.speech_thread.start()

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

    def _speech_loop(self):
        recognizer = sr.Recognizer()
        recognizer.dynamic_energy_threshold = True
        
        while not rospy.is_shutdown():
            try:
                with sr.Microphone() as source:
                    rospy.loginfo_throttle(10.0, "Listening for speech command...")
                    audio = recognizer.listen(source, timeout=2.0, phrase_time_limit=5.0)
                
                rospy.loginfo("Audio captured. Transcribing with local Whisper...")
                text = recognizer.recognize_whisper(audio, model="base.en")
                rospy.loginfo(f"Speech transcription: '{text}'")
                
                if text.strip():
                    new_prompt = self._get_ollama_prompt(text)
                    if new_prompt:
                        cmd = new_prompt.strip().upper()
                        if cmd.startswith("NAVIGATE:"):
                            self.prompt = new_prompt[9:].strip()
                            self.manual_override = None
                            rospy.loginfo(f"Updating VLM prompt to: '{self.prompt}'")
                        elif cmd.startswith("STOP"):
                            self.manual_override = "STOP"
                            rospy.loginfo("Commanded to STOP.")
                        elif cmd.startswith("FORWARD"):
                            self.manual_override = "FORWARD"
                            rospy.loginfo("Commanded to MOVE FORWARD.")
                        elif cmd.startswith("REVERSE") or cmd.startswith("BACKWARD"):
                            self.manual_override = "REVERSE"
                            rospy.loginfo("Commanded to REVERSE.")
                        elif cmd.startswith("TURN: LEFT"):
                            self.manual_override = "TURN_LEFT"
                            rospy.loginfo("Commanded to TURN LEFT.")
                        elif cmd.startswith("TURN: RIGHT"):
                            self.manual_override = "TURN_RIGHT"
                            rospy.loginfo("Commanded to TURN RIGHT.")
                        elif cmd.startswith("FASTER"):
                            self.speed_multiplier += 0.5
                            rospy.loginfo(f"Speed increased to x{self.speed_multiplier}")
                        elif cmd.startswith("SLOWER"):
                            self.speed_multiplier -= 0.5
                            self.speed_multiplier = max(0.0, self.speed_multiplier)
                            rospy.loginfo(f"Speed decreased to x{self.speed_multiplier}")
                        else:
                            # Fallback just in case LLM outputs something unexpected
                            self.prompt = new_prompt
                            self.manual_override = None
                            rospy.loginfo(f"Updating VLM prompt to: '{self.prompt}' (Fallback)")
                        
            except sr.WaitTimeoutError:
                pass
            except Exception as e:
                rospy.logwarn_throttle(5.0, f"Speech recognition exception: {e}")
                time.sleep(1.0)

    def _get_ollama_prompt(self, text):
        url = "http://localhost:11434/api/generate"
        system_prompt = (
            "You are a helpful assistant for a mobile robot. "
            "Convert the user's spoken command into one of the following exact formats. "
            "Do not output anything else or any conversational filler.\n"
            "- If it's a target to navigate to: NAVIGATE: [target description]\n"
            "- If the user says to stop or halt: STOP\n"
            "- If the user says to move forward or go straight: FORWARD\n"
            "- If the user says to reverse, go back, or back up: REVERSE\n"
            "- If the user says to turn left: TURN: LEFT\n"
            "- If the user says to turn right: TURN: RIGHT\n"
            "- If the user says to go faster: FASTER\n"
            "- If the user says to go slower: SLOWER\n"
            "Only output the command matching the format above."
        )
        payload = {
            "model": self.ollama_model,
            "prompt": f"User spoken command: {text}",
            "system": system_prompt,
            "stream": False
        }
        try:
            res = requests.post(url, json=payload, timeout=10.0)
            if res.status_code == 200:
                return res.json().get("response", "").strip()
            else:
                rospy.logwarn(f"Ollama API error: {res.status_code} - {res.text}")
        except Exception as e:
            rospy.logwarn(f"Failed to query Ollama at {url}: {e}")
        return None

    def cb(self, msg: CompressedImage):
        # Decode JPEG bytes -> BGR image
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            rospy.logwarn_throttle(2.0, "Failed to decode compressed image.")
            return

        # Center crop to maximum square resolution
        h, w = frame_bgr.shape[:2]
        size = min(h, w)
        y = (h - size) // 2
        x = (w - size) // 2
        frame_bgr = frame_bgr[y:y+size, x:x+size]

        # Convert to RGB numpy for your agent (it accepts np.ndarray)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Save latest frame (overwrites)
        ok = cv2.imwrite(self._latest_path, frame_bgr)
        if not ok:
            rospy.logwarn_throttle(5.0, "cv2.imwrite failed — cannot write to %s", self._latest_path)

        # # Save timestamped frame
        # ts_path = os.path.join(self._frames_dir, f"frame_{self._frame_idx:06d}.jpg")
        # cv2.imwrite(ts_path, frame_bgr)
        # self._frame_idx += 1

        if self.prompt is None and self.manual_override is None:
            rospy.loginfo_throttle(5.0, "Waiting for initial speech command before moving...")
            twist = Twist()
            self.cmd_vel_pub.publish(twist)
            return

        dt = 0.0
        omega_l = 0.0
        omega_r = 0.0

        if self.manual_override == "STOP":
            v_raw, w_raw = 0.0, 0.0
        elif self.manual_override == "FORWARD":
            v_raw, w_raw = 0.2, 0.0
        elif self.manual_override == "REVERSE":
            v_raw, w_raw = -0.2, 0.0
        elif self.manual_override == "TURN_LEFT":
            v_raw, w_raw = self._v_smooth, 0.5
        elif self.manual_override == "TURN_RIGHT":
            v_raw, w_raw = self._v_smooth, -0.5
        else:
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

        v = self._v_smooth * self.speed_multiplier
        w = self._w_smooth * self.speed_multiplier

        twist = Twist()
        twist.linear.x  = v
        twist.angular.z = w * 0.2
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