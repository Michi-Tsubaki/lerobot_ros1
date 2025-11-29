#!/usr/bin/env python3

import pickle
from pathlib import Path
import shutil
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import yaml
from imitation_utils.modality_config import ModalityConfig
import rospy

rospy.init_node("lerobot_uploader", anonymous=True)

config_path = rospy.get_param("~config", None)
cfg = ModalityConfig(config_path)

with open(config_path if config_path else cfg.config_path) as f:
    config = yaml.safe_load(f)

data_dir = Path(config["paths"]["data_dir"])
repo_id = config["paths"]["repo_id"]
local_dir = Path(config["paths"]["local_dir"])
fps = config["robot"]["fps"]

if local_dir.exists():
    shutil.rmtree(local_dir)

features = cfg.get_lerobot_features()

dataset = LeRobotDataset.create(
    repo_id=repo_id,
    fps=fps,
    root=local_dir,
    robot_type=config["robot"]["type"],
    features=features,
)

episode_files = sorted(data_dir.glob("episode_*.pkl"))
print(f"Processing {len(episode_files)} episodes")

for ep_idx, ep_file in enumerate(episode_files):
    with open(ep_file, "rb") as f:
        frames = pickle.load(f)

    for i, frame in enumerate(frames):
        action = frames[i + 1]["state"] if i < len(frames) - 1 else frame["state"]

        frame_data = {"observation.state": frame["state"], "action": action}

        if "env_state" in frame:
            frame_data["observation.environment_state"] = frame["env_state"]

        for mod in cfg.image_modalities:
            frame_data[f"observation.images.{mod.name}"] = frame[mod.data_key]

        dataset.add_frame(
            frame_data,
            task=rospy.get_param("~task_name", "manipulation"),
            timestamp=i / fps,
        )

    dataset.save_episode()
    print(f"Episode {ep_idx + 1}/{len(episode_files)} saved")

if rospy.get_param("~push_to_hub", True):
    dataset.push_to_hub()
    print("Upload Completed!")
else:
    print("Local save completed. Skipping upload.")
