#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import JointState, Image  # noqa: F401
from geometry_msgs.msg import WrenchStamped  # noqa: F401
from cv_bridge import CvBridge
import numpy as np
from pathlib import Path
import cv2
import sys
import select
import termios
import tty
import signal
import threading
import pickle
import shutil
from imitation_utils.modality_config import ModalityConfig
import yaml

rospy.init_node("data_collector")
bridge = CvBridge()

config_path = rospy.get_param("~config", None)
cfg = ModalityConfig(config_path)

with open(config_path if config_path else cfg.config_path) as f:
    config = yaml.safe_load(f)


class DataCollector:
    def __init__(self, root_dir):
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.interrupted = False
        self.lock = threading.Lock()

        self.latest_data = {m.name: None for m in cfg.state_modalities}
        self.latest_data.update({m.name: None for m in cfg.env_state_modalities})
        self.latest_data.update({m.name: None for m in cfg.image_modalities})

        for mod in cfg.state_modalities:
            msg_type = self._get_msg_class(mod.msg_type)
            rospy.Subscriber(
                mod.topic,
                msg_type,
                lambda msg, name=mod.name, field=mod.field: self._state_cb(
                    msg, name, field
                ),
            )

        for mod in cfg.env_state_modalities:
            msg_type = self._get_msg_class(mod.msg_type)
            rospy.Subscriber(
                mod.topic,
                msg_type,
                lambda msg, name=mod.name, field=mod.field: self._env_state_cb(
                    msg, name, field
                ),
            )

        for mod in cfg.image_modalities:
            rospy.Subscriber(
                mod.topic, Image, lambda msg, name=mod.name: self._image_cb(msg, name)
            )

        print("Waiting for all topics...")
        self.wait_for_topics()

        self.load_existing_episodes()

    def _get_msg_class(self, msg_type_str):
        pkg, msg = msg_type_str.split("/")
        if pkg == "sensor_msgs":
            from sensor_msgs.msg import JointState

            return JointState
        elif pkg == "geometry_msgs":
            from geometry_msgs.msg import WrenchStamped

            return WrenchStamped
        raise ValueError(f"Unknown msg type: {msg_type_str}")

    def _state_cb(self, msg, name, field):
        with self.lock:
            if "." in field:
                obj = msg
                for attr in field.split("."):
                    obj = getattr(obj, attr)
                self.latest_data[name] = obj
            else:
                self.latest_data[name] = getattr(msg, field)

    def _env_state_cb(self, msg, name, field):
        with self.lock:
            if "." in field:
                obj = msg
                for attr in field.split("."):
                    obj = getattr(obj, attr)
                val = obj
            else:
                val = getattr(msg, field)
            if hasattr(val, 'x') and hasattr(val, 'y') and hasattr(val, 'z'):
                self.latest_data[name] = [val.x, val.y, val.z]
            elif hasattr(val, 'x') and hasattr(val, 'y'):
                self.latest_data[name] = [val.x, val.y]
            else:
                self.latest_data[name] = val

    def _image_cb(self, msg, name):
        with self.lock:
            self.latest_data[name] = msg

    def wait_for_topics(self):
        while not rospy.is_shutdown():
            with self.lock:
                missing = [k for k, v in self.latest_data.items() if v is None]
            if not missing:
                print("All topics ready")
                return True
            print(f"Waiting for: {missing}")
            rospy.sleep(0.5)
        return False

    def load_existing_episodes(self):
        episode_files = sorted(self.root.glob("episode_*.pkl"))
        if episode_files:
            print(f"Found {len(episode_files)} existing episodes")
            choice = input("Continue from existing data? (y/n): ").strip().lower()
            if choice != "y":
                backup_dir = (
                    self.root.parent
                    / f"{self.root.name}_backup_{int(rospy.Time.now().to_sec())}"
                )
                shutil.move(self.root, backup_dir)
                self.root.mkdir(parents=True, exist_ok=True)
                print(f"Moved existing data to {backup_dir}")

    def get_next_episode_number(self):
        episode_files = list(self.root.glob("episode_*.pkl"))
        if not episode_files:
            return 0
        numbers = [int(f.stem.split("_")[1]) for f in episode_files]
        return max(numbers) + 1

    def kbhit(self):
        return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

    def record_episode(self):
        frames = []
        ep_num = self.get_next_episode_number()
        print(f"\n=== Episode {ep_num} ===")
        print("Recording... Press SPACE to finish episode")

        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            rate = rospy.Rate(30)

            while not rospy.is_shutdown() and not self.interrupted:
                if self.kbhit():
                    ch = sys.stdin.read(1)
                    if ch == " ":
                        break

                try:
                    with self.lock:
                        state_list = []
                        for mod in cfg.state_modalities:
                            val = self.latest_data[mod.name]
                            if isinstance(val, (list, tuple)):
                                state_list.extend(val)
                            else:
                                state_list.append(val)
                        state = np.array(state_list, dtype=np.float32)

                        env_state_list = []
                        for mod in cfg.env_state_modalities:
                            val = self.latest_data[mod.name]
                            if isinstance(val, (list, tuple)):
                                env_state_list.extend(val)
                            else:
                                env_state_list.append(val)
                        env_state = (
                            np.array(env_state_list, dtype=np.float32)
                            if env_state_list
                            else None
                        )

                        frame_data = {"state": state}
                        if env_state is not None:
                            frame_data["env_state"] = env_state

                        for mod in cfg.image_modalities:
                            img_msg = self.latest_data[mod.name]
                            img = bridge.imgmsg_to_cv2(img_msg, "rgb8")
                            img = cfg.crop_image(img, mod.name)
                            img = cv2.resize(img, mod.resolution)
                            frame_data[mod.data_key] = img

                    frames.append(frame_data)
                    print(f"\rFrames: {len(frames)}", end="", flush=True)
                except Exception:
                    pass

                rate.sleep()
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

        if self.interrupted:
            if len(frames) > 10:
                print(f"\nInterrupted. Episode has {len(frames)} frames")
                choice = input("Save this episode? (y/n): ").strip().lower()
                if choice == "y":
                    self.save_episode(ep_num, frames)
                    return True
            return False

        if len(frames) < 10:
            print(f"\nEpisode too short ({len(frames)} frames), discarded")
            return False

        choice = (
            input(f"\nEpisode recorded ({len(frames)} frames). Accept? (y/n): ")
            .strip()
            .lower()
        )
        if choice != "y":
            print("Episode discarded")
            return False

        self.save_episode(ep_num, frames)
        return True

    def save_episode(self, ep_num, frames):
        filepath = self.root / f"episode_{ep_num:06d}.pkl"
        with open(filepath, "wb") as f:
            pickle.dump(frames, f)
        print(f"\n✓ Saved {filepath}")


collector = None


def signal_handler(sig, frame):
    global collector
    print("\n\nCtrl+C detected. All episodes saved.")
    if collector:
        collector.interrupted = True
        episode_files = sorted(collector.root.glob("episode_*.pkl"))
        print(f"Total episodes: {len(episode_files)}")
    sys.exit(0)


if __name__ == "__main__":
    collector = DataCollector(config["paths"]["data_dir"])
    signal.signal(signal.SIGINT, signal_handler)

    num_episodes = rospy.get_param("~num_episodes", 60)
    collected = 0
    target = num_episodes + collector.get_next_episode_number()

    while collector.get_next_episode_number() < target and not collector.interrupted:
        if not collector.wait_for_topics():
            print("Topic check failed, exiting")
            break
        if collector.record_episode():
            collected += 1
            if collector.get_next_episode_number() < target:
                try:
                    input("Press Enter to start next episode...")
                except KeyboardInterrupt:
                    break

    episode_files = sorted(collector.root.glob("episode_*.pkl"))
    print(f"\nCollection complete. Total episodes: {len(episode_files)}")
