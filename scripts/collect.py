#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import JointState, Image  # noqa: F401
from geometry_msgs.msg import WrenchStamped  # noqa: F401
from cv_bridge import CvBridge
import numpy as np
from pathlib import Path
import cv2
import sys
import json
import select
import termios
import tty
import signal
import threading
import pickle
import shutil
from std_msgs.msg import String
from imitation_utils.modality_config import ModalityConfig
from imitation_utils.dataset_conversion import resolve_data_path
from imitation_utils.lerobot_format import ManualLeRobotDatasetWriter, load_lerobot_episode_count
import yaml

rospy.init_node("data_collector")
bridge = CvBridge()

config_path = rospy.get_param("~config", None)
cfg = ModalityConfig(config_path)

with open(config_path if config_path else cfg.config_path) as f:
    config = yaml.safe_load(f)


class DataCollector:
    def __init__(self, root_dir, *, dataset_format, task_name, pickle_dir=None, save_pickle=False):
        self.root = Path(root_dir)
        self.dataset_format = dataset_format
        self.task_name = task_name
        self.pickle_root = Path(pickle_dir) if pickle_dir is not None else self.root
        self.save_pickle_enabled = save_pickle or dataset_format == "pickle"
        self.root.mkdir(parents=True, exist_ok=True)
        if self.save_pickle_enabled:
            self.pickle_root.mkdir(parents=True, exist_ok=True)
        self.interrupted = False
        self.lock = threading.Lock()
        self.command_lock = threading.Lock()
        self.pending_commands = []
        self.status = {
            "phase": "initializing",
            "episode_index": 0,
            "saved_episodes": 0,
            "target_episodes": None,
            "frames": 0,
            "dataset_format": self.dataset_format,
            "task_name": self.task_name,
            "root": str(self.root),
            "last_message": "",
            "connections": {},
        }
        self.manual_start = rospy.get_param("~manual_start", False)
        self.manual_accept = rospy.get_param("~manual_accept", False)
        self.status_pub = rospy.Publisher("~status", String, queue_size=1, latch=True)
        self.command_sub = rospy.Subscriber("~command", String, self._command_cb)
        self.lerobot_writer = None

        if self.dataset_format == "lerobot":
            self.lerobot_writer = ManualLeRobotDatasetWriter(
                root=self.root,
                repo_id=config["paths"]["repo_id"],
                fps=config["robot"]["fps"],
                robot_type=config["robot"]["type"],
                features=cfg.get_lerobot_features(),
            )

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
        self.publish_status(phase="ready", saved_episodes=self.get_next_episode_number())

    def _command_cb(self, msg):
        command = msg.data.strip().lower()
        if command not in {"start", "finish", "accept", "reject", "stop"}:
            self.publish_status(last_message=f"Ignored unknown command: {msg.data}")
            return
        with self.command_lock:
            self.pending_commands.append(command)
        self.publish_status(last_message=f"Received command: {command}")

    def consume_command(self, *allowed):
        with self.command_lock:
            for idx, command in enumerate(self.pending_commands):
                if not allowed or command in allowed:
                    return self.pending_commands.pop(idx)
        return None

    def publish_status(self, **updates):
        with self.lock:
            connections = {}
            for mod in cfg.state_modalities:
                role = "hand" if "hand" in mod.name or "hand" in mod.topic or "rhand" in mod.name or "lhand" in mod.name else "robot"
                connections[mod.name] = {
                    "role": role,
                    "topic": mod.topic,
                    "connected": self.latest_data.get(mod.name) is not None,
                }
            for mod in cfg.env_state_modalities:
                connections[mod.name] = {
                    "role": "env",
                    "topic": mod.topic,
                    "connected": self.latest_data.get(mod.name) is not None,
                }
            for mod in cfg.image_modalities:
                connections[mod.name] = {
                    "role": "image",
                    "topic": mod.topic,
                    "connected": self.latest_data.get(mod.name) is not None,
                }
        updates["connections"] = connections
        updates["robot_connected"] = all(
            item["connected"] for item in connections.values() if item["role"] == "robot"
        )
        updates["hand_connected"] = all(
            item["connected"] for item in connections.values() if item["role"] == "hand"
        )
        updates["images_connected"] = all(
            item["connected"] for item in connections.values() if item["role"] == "image"
        )
        self.status.update(updates)
        self.status.setdefault("saved_episodes", self.get_next_episode_number())
        self.status["stamp"] = rospy.Time.now().to_sec()
        self.status_pub.publish(String(data=json.dumps(self.status, ensure_ascii=False)))

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
            self.publish_status(phase="waiting_topics", missing_topics=missing)
            if not missing:
                print("All topics ready")
                return True
            print(f"Waiting for: {missing}")
            rospy.sleep(0.5)
        return False

    def load_existing_episodes(self):
        if self.dataset_format == "lerobot":
            existing_count = load_lerobot_episode_count(self.root)
        else:
            existing_count = len(sorted(self.pickle_root.glob("episode_*.pkl")))

        if existing_count:
            print(f"Found {existing_count} existing episodes")
            if not sys.stdin.isatty():
                continue_existing = rospy.get_param("~continue_existing", True)
                if continue_existing:
                    print("stdin is not a TTY. Continuing from existing data.")
                    return
                else:
                    print("stdin is not a TTY. Backing up existing data.")
                    backup_dir = (
                        self.root.parent
                        / f"{self.root.name}_backup_{int(rospy.Time.now().to_sec())}"
                    )
                    shutil.move(self.root, backup_dir)
                    self.root.mkdir(parents=True, exist_ok=True)
                    if self.dataset_format == "lerobot":
                        self.lerobot_writer = ManualLeRobotDatasetWriter(
                            root=self.root,
                            repo_id=config["paths"]["repo_id"],
                            fps=config["robot"]["fps"],
                            robot_type=config["robot"]["type"],
                            features=cfg.get_lerobot_features(),
                        )
                    print(f"Moved existing data to {backup_dir}")
                    return

            choice = input("Continue from existing data? (y/n): ").strip().lower()
            if choice != "y":
                backup_dir = (
                    self.root.parent
                    / f"{self.root.name}_backup_{int(rospy.Time.now().to_sec())}"
                )
                shutil.move(self.root, backup_dir)
                self.root.mkdir(parents=True, exist_ok=True)
                if self.dataset_format == "lerobot":
                    self.lerobot_writer = ManualLeRobotDatasetWriter(
                        root=self.root,
                        repo_id=config["paths"]["repo_id"],
                        fps=config["robot"]["fps"],
                        robot_type=config["robot"]["type"],
                        features=cfg.get_lerobot_features(),
                    )
                print(f"Moved existing data to {backup_dir}")

    def get_next_episode_number(self):
        if self.dataset_format == "lerobot":
            return load_lerobot_episode_count(self.root)

        episode_files = list(self.pickle_root.glob("episode_*.pkl"))
        if not episode_files:
            return 0
        numbers = [int(f.stem.split("_")[1]) for f in episode_files]
        return max(numbers) + 1

    def kbhit(self):
        return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

    def wait_for_start(self, target_episodes):
        ep_num = self.get_next_episode_number()
        self.publish_status(
            phase="idle",
            episode_index=ep_num,
            saved_episodes=ep_num,
            target_episodes=target_episodes,
            frames=0,
            last_message="Waiting for start",
        )
        print(f"\nReady for episode {ep_num}. Press Enter or send 'start'.")
        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and not self.interrupted:
            command = self.consume_command("start", "stop")
            if command == "start":
                return True
            if command == "stop":
                self.interrupted = True
                return False
            if sys.stdin.isatty() and self.kbhit():
                ch = sys.stdin.read(1)
                if ch in ("\n", "\r", " "):
                    return True
                if ch.lower() == "q":
                    self.interrupted = True
                    return False
            rate.sleep()
        return False

    def wait_for_decision(self, ep_num, frames):
        self.publish_status(
            phase="review",
            episode_index=ep_num,
            frames=len(frames),
            last_message="Waiting for accept/reject",
        )
        print(f"\nEpisode recorded ({len(frames)} frames). Send accept/reject or press y/n.")
        old_settings = None
        try:
            if sys.stdin.isatty():
                old_settings = termios.tcgetattr(sys.stdin)
                tty.setcbreak(sys.stdin.fileno())
            rate = rospy.Rate(10)
            while not rospy.is_shutdown() and not self.interrupted:
                command = self.consume_command("accept", "reject", "stop")
                if command == "accept":
                    return True
                if command == "reject":
                    return False
                if command == "stop":
                    self.interrupted = True
                    return False
                if sys.stdin.isatty() and self.kbhit():
                    ch = sys.stdin.read(1).lower()
                    if ch == "y":
                        return True
                    if ch == "n":
                        return False
                rate.sleep()
        finally:
            if old_settings is not None:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        return False

    def record_episode(self):
        frames = []
        ep_num = self.get_next_episode_number()
        print(f"\n=== Episode {ep_num} ===")
        interactive = sys.stdin.isatty()
        fixed_duration = rospy.get_param("~episode_duration", 10.0)

        if interactive:
            print("Recording... Press SPACE to finish episode")
        else:
            print(
                "Recording... stdin is not a TTY, "
                f"recording for {fixed_duration} seconds"
            )

        old_settings = None
        start_time = rospy.Time.now()
        last_status_time = rospy.Time(0)
        self.publish_status(
            phase="recording",
            episode_index=ep_num,
            frames=0,
            last_message="Recording",
        )

        try:
            if interactive:
                old_settings = termios.tcgetattr(sys.stdin)
                tty.setcbreak(sys.stdin.fileno())

            rate = rospy.Rate(30)

            while not rospy.is_shutdown() and not self.interrupted:
                command = self.consume_command("finish", "reject", "stop")
                if command == "finish":
                    break
                if command == "reject":
                    frames = []
                    break
                if command == "stop":
                    self.interrupted = True
                    break

                if interactive:
                    if self.kbhit():
                        ch = sys.stdin.read(1)
                        if ch == " ":
                            break
                else:
                    if (rospy.Time.now() - start_time).to_sec() >= fixed_duration:
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
                    now = rospy.Time.now()
                    if (now - last_status_time).to_sec() >= 0.25:
                        self.publish_status(
                            phase="recording",
                            episode_index=ep_num,
                            frames=len(frames),
                            saved_episodes=self.get_next_episode_number(),
                        )
                        last_status_time = now
                except Exception:
                    pass

                rate.sleep()
        finally:
            if interactive and old_settings is not None:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

        if self.interrupted:
            if len(frames) > 10:
                print(f"\nInterrupted. Episode has {len(frames)} frames")
                if not sys.stdin.isatty():
                    print("stdin is not a TTY. Auto-saving interrupted episode.")
                    self.save_episode(ep_num, frames)
                    return True

                choice = input("Save this episode? (y/n): ").strip().lower()
                if choice == "y":
                    self.save_episode(ep_num, frames)
                    return True
            return False

        if len(frames) < 10:
            print(f"\nEpisode too short ({len(frames)} frames), discarded")
            self.publish_status(
                phase="discarded",
                episode_index=ep_num,
                frames=len(frames),
                last_message="Episode too short, discarded",
            )
            return False

        if self.manual_accept:
            if not self.wait_for_decision(ep_num, frames):
                print("Episode discarded")
                self.publish_status(
                    phase="discarded",
                    episode_index=ep_num,
                    frames=len(frames),
                    last_message="Episode rejected",
                )
                return False
            self.save_episode(ep_num, frames)
            return True

        if not sys.stdin.isatty():
            print(f"\nEpisode recorded ({len(frames)} frames). Auto-accepted.")
            self.save_episode(ep_num, frames)
            return True

        choice = (
            input(f"\nEpisode recorded ({len(frames)} frames). Accept? (y/n): ")
            .strip()
            .lower()
        )
        if choice != "y":
            print("Episode discarded")
            self.publish_status(
                phase="discarded",
                episode_index=ep_num,
                frames=len(frames),
                last_message="Episode rejected",
            )
            return False

        self.save_episode(ep_num, frames)
        return True

    def save_episode(self, ep_num, frames):
        if self.dataset_format == "lerobot":
            saved_ep = self.lerobot_writer.add_episode(frames, task=self.task_name)
            print(f"\nSaved LeRobot episode {saved_ep:06d} under {self.root}")

        if self.save_pickle_enabled:
            filepath = self.pickle_root / f"episode_{ep_num:06d}.pkl"
            with open(filepath, "wb") as f:
                pickle.dump(frames, f)
            print(f"Saved pickle backup {filepath}")
        self.publish_status(
            phase="saved",
            episode_index=ep_num,
            saved_episodes=self.get_next_episode_number(),
            frames=len(frames),
            last_message="Episode saved",
        )

collector = None

def signal_handler(sig, frame):
    global collector
    print("\n\nCtrl+C detected. All episodes saved.")
    if collector:
        collector.interrupted = True
        print(f"Total episodes: {collector.get_next_episode_number()}")
    sys.exit(0)


if __name__ == "__main__":
    dataset_format = rospy.get_param("~dataset_format", "lerobot").lower()
    if dataset_format not in {"lerobot", "pickle"}:
        raise ValueError("~dataset_format must be 'lerobot' or 'pickle'")

    task_name = rospy.get_param(
        "~task_name",
        config.get("policy", {}).get("openpi", {}).get("task_name", "manipulation"),
    )
    save_pickle = rospy.get_param("~save_pickle", False)
    resolved_config_path = Path(config_path if config_path else cfg.config_path).resolve()
    lerobot_dir = resolve_data_path(resolved_config_path, config["paths"]["local_dir"])
    pickle_dir = resolve_data_path(resolved_config_path, config["paths"]["data_dir"])
    root_dir = lerobot_dir if dataset_format == "lerobot" else pickle_dir

    collector = DataCollector(
        root_dir,
        dataset_format=dataset_format,
        task_name=task_name,
        pickle_dir=pickle_dir,
        save_pickle=save_pickle,
    )
    signal.signal(signal.SIGINT, signal_handler)

    num_episodes = rospy.get_param("~num_episodes", 60)
    collected = 0
    target = num_episodes + collector.get_next_episode_number()
    collector.publish_status(target_episodes=target, saved_episodes=collector.get_next_episode_number())

    while collector.get_next_episode_number() < target and not collector.interrupted:
        if not collector.wait_for_topics():
            print("Topic check failed, exiting")
            break
        if collector.manual_start and not collector.wait_for_start(target):
            break
        if collector.record_episode():
            collected += 1
            if collector.get_next_episode_number() < target:
                if sys.stdin.isatty():
                    try:
                        input("Press Enter to start next episode...")
                    except KeyboardInterrupt:
                        break
                else:
                    rospy.sleep(1.0)

    print(f"\nCollection complete. Total episodes: {collector.get_next_episode_number()}")
