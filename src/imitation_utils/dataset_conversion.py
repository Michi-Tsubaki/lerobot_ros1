from __future__ import annotations

import pickle
import shutil
from pathlib import Path

import yaml
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from imitation_utils.modality_config import ModalityConfig


def resolve_config_path(config_path: str | Path | None, fallback: str | Path) -> Path:
    if config_path is None:
        return Path(fallback).resolve()
    return Path(config_path).resolve()


def resolve_data_path(config_path: str | Path, path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (Path(config_path).resolve().parent / path).resolve()


def load_config_dict(config_path: str | Path | None) -> tuple[Path, ModalityConfig, dict]:
    cfg = ModalityConfig(config_path)
    resolved_config_path = resolve_config_path(config_path, cfg.config_path)
    with open(resolved_config_path) as f:
        config = yaml.safe_load(f)
    return resolved_config_path, cfg, config


def convert_pickle_dataset_to_lerobot(
    *,
    config_path: str | Path | None,
    task_name: str,
    push_to_hub: bool = False,
    local_dir_override: str | Path | None = None,
) -> Path:
    resolved_config_path, cfg, config = load_config_dict(config_path)

    data_dir = resolve_data_path(resolved_config_path, config["paths"]["data_dir"])
    repo_id = config["paths"]["repo_id"]
    local_dir = (
        resolve_data_path(resolved_config_path, local_dir_override)
        if local_dir_override is not None
        else resolve_data_path(resolved_config_path, config["paths"]["local_dir"])
    )
    fps = config["robot"]["fps"]

    if local_dir.exists():
        shutil.rmtree(local_dir)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        root=local_dir,
        robot_type=config["robot"]["type"],
        features=cfg.get_lerobot_features(),
    )

    episode_files = sorted(data_dir.glob("episode_*.pkl"))
    for ep_file in episode_files:
        with open(ep_file, "rb") as f:
            frames = pickle.load(f)

        for i, frame in enumerate(frames):
            action = frames[i + 1]["state"] if i < len(frames) - 1 else frame["state"]
            frame_data = {
                "observation.state": frame["state"],
                "action": action,
            }

            if "env_state" in frame:
                frame_data["observation.environment_state"] = frame["env_state"]

            for mod in cfg.image_modalities:
                frame_data[f"observation.images.{mod.name}"] = frame[mod.data_key]

            dataset.add_frame(
                frame_data,
                task=task_name,
                timestamp=i / fps,
            )

        dataset.save_episode()

    if push_to_hub:
        dataset.push_to_hub()

    return local_dir
