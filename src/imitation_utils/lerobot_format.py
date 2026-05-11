from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


CODEBASE_VERSION = "v2.1"
CHUNKS_SIZE = 1000
DEFAULT_PARQUET_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
DEFAULT_VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"

DEFAULT_FEATURES = {
    "timestamp": {"dtype": "float32", "shape": (1,), "names": None},
    "frame_index": {"dtype": "int64", "shape": (1,), "names": None},
    "episode_index": {"dtype": "int64", "shape": (1,), "names": None},
    "index": {"dtype": "int64", "shape": (1,), "names": None},
    "task_index": {"dtype": "int64", "shape": (1,), "names": None},
}


def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False, default=_json_default)


def _append_jsonl(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(data, ensure_ascii=False, default=_json_default) + "\n")


def _read_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _feature_shape(feature: dict) -> tuple[int, ...]:
    return tuple(feature["shape"])


def _empty_info(*, fps: int, features: dict, robot_type: str, use_videos: bool = True) -> dict:
    return {
        "codebase_version": CODEBASE_VERSION,
        "robot_type": robot_type,
        "total_episodes": 0,
        "total_frames": 0,
        "total_tasks": 0,
        "total_videos": 0,
        "total_chunks": 0,
        "chunks_size": CHUNKS_SIZE,
        "fps": fps,
        "splits": {},
        "data_path": DEFAULT_PARQUET_PATH,
        "video_path": DEFAULT_VIDEO_PATH if use_videos else None,
        "features": features,
    }


def lerobot_features_with_defaults(features: dict) -> dict:
    return {**features, **DEFAULT_FEATURES}


def load_lerobot_episode_count(root: str | Path) -> int:
    info_path = Path(root) / "meta" / "info.json"
    if not info_path.exists():
        return 0
    return int(_read_json(info_path).get("total_episodes", 0))


def is_lerobot_dataset(root: str | Path) -> bool:
    root = Path(root)
    return (root / "meta" / "info.json").exists() and (root / "data").exists()


def _stats_for_array(array: np.ndarray, *, image: bool = False) -> dict[str, np.ndarray]:
    array = np.asarray(array)
    if image:
        # LeRobot image/video stats are channel-first, normalized, and shaped (3, 1, 1).
        array = array.astype(np.float32) / 255.0
        array = np.transpose(array, (0, 3, 1, 2))
        axis = (0, 2, 3)
        keepdims = True
    else:
        axis = 0
        keepdims = array.ndim == 1

    stats = {
        "min": np.min(array, axis=axis, keepdims=keepdims),
        "max": np.max(array, axis=axis, keepdims=keepdims),
        "mean": np.mean(array, axis=axis, keepdims=keepdims),
        "std": np.std(array, axis=axis, keepdims=keepdims),
        "count": np.array([len(array)], dtype=np.int64),
    }
    if image:
        stats = {key: value if key == "count" else np.squeeze(value, axis=0) for key, value in stats.items()}
    return stats


def compute_episode_stats(episode_data: dict[str, list | np.ndarray], features: dict) -> dict:
    stats = {}
    for key, data in episode_data.items():
        if key not in features or features[key]["dtype"] == "string":
            continue
        if features[key]["dtype"] == "video":
            stats[key] = _stats_for_array(np.asarray(data), image=True)
        else:
            stats[key] = _stats_for_array(np.asarray(data))
    return stats


def aggregate_stats(stats_list: list[dict[str, dict]]) -> dict[str, dict]:
    if not stats_list:
        return {}

    stats_list = [_stats_to_numpy(stats) for stats in stats_list]
    result = {}
    for key in sorted({key for stats in stats_list for key in stats}):
        key_stats = [stats[key] for stats in stats_list if key in stats]
        means = np.stack([stats["mean"] for stats in key_stats])
        variances = np.stack([stats["std"] ** 2 for stats in key_stats])
        counts = np.stack([stats["count"] for stats in key_stats])
        total_count = counts.sum(axis=0)

        while counts.ndim < means.ndim:
            counts = np.expand_dims(counts, axis=-1)

        weighted_means = means * counts
        total_mean = weighted_means.sum(axis=0) / total_count
        delta_means = means - total_mean
        weighted_variances = (variances + delta_means**2) * counts
        total_variance = weighted_variances.sum(axis=0) / total_count

        result[key] = {
            "min": np.min(np.stack([stats["min"] for stats in key_stats]), axis=0),
            "max": np.max(np.stack([stats["max"] for stats in key_stats]), axis=0),
            "mean": total_mean,
            "std": np.sqrt(total_variance),
            "count": total_count,
        }
    return result


def _stats_to_numpy(stats: dict[str, dict]) -> dict[str, dict]:
    return {
        key: {stat_key: np.asarray(value) for stat_key, value in values.items()}
        for key, values in stats.items()
    }


def _task_index(root: Path, task: str) -> int:
    tasks_path = root / "meta" / "tasks.jsonl"
    tasks = _read_jsonl(tasks_path)
    for item in tasks:
        if item["task"] == task:
            return int(item["task_index"])

    task_idx = len(tasks)
    _append_jsonl(tasks_path, {"task_index": task_idx, "task": task})
    return task_idx


def _write_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {path}")
    try:
        for frame in frames:
            rgb = np.asarray(frame, dtype=np.uint8)
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


@dataclass
class ManualLeRobotDatasetWriter:
    root: Path
    repo_id: str
    fps: int
    robot_type: str
    features: dict
    overwrite: bool = False

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.features = lerobot_features_with_defaults(self.features)
        if self.overwrite and self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.info_path = self.root / "meta" / "info.json"
        if not self.info_path.exists():
            info = _empty_info(
                fps=self.fps,
                features=self.features,
                robot_type=self.robot_type,
                use_videos=bool(self.video_keys),
            )
            _write_json(self.info_path, info)
        self.info = _read_json(self.info_path)

    @property
    def video_keys(self) -> list[str]:
        return [key for key, ft in self.features.items() if ft["dtype"] == "video"]

    def next_episode_index(self) -> int:
        return int(self.info.get("total_episodes", 0))

    def add_episode(self, frames: list[dict], *, task: str) -> int:
        if len(frames) < 1:
            raise ValueError("Cannot save an empty episode")

        episode_index = self.next_episode_index()
        episode_chunk = episode_index // int(self.info["chunks_size"])
        task_index = _task_index(self.root, task)
        start_index = int(self.info.get("total_frames", 0))

        rows: list[dict] = []
        episode_arrays: dict[str, list] = {key: [] for key in self.features}
        image_frames: dict[str, list[np.ndarray]] = {key: [] for key in self.video_keys}

        for frame_index, frame in enumerate(frames):
            action = frames[frame_index + 1]["state"] if frame_index < len(frames) - 1 else frame["state"]
            row = {
                "observation.state": np.asarray(frame["state"], dtype=np.float32).tolist(),
                "action": np.asarray(action, dtype=np.float32).tolist(),
                "timestamp": np.float32(frame_index / self.fps),
                "frame_index": np.int64(frame_index),
                "episode_index": np.int64(episode_index),
                "index": np.int64(start_index + frame_index),
                "task_index": np.int64(task_index),
            }
            if "env_state" in frame and "observation.environment_state" in self.features:
                row["observation.environment_state"] = np.asarray(frame["env_state"], dtype=np.float32).tolist()

            for key in row:
                episode_arrays[key].append(row[key])

            for video_key in self.video_keys:
                image_name = video_key.removeprefix("observation.images.")
                data_key = image_name.replace("_camera", "_image")
                image = frame.get(data_key)
                if image is None:
                    image = frame.get(image_name)
                if image is None:
                    raise KeyError(f"Frame is missing image data for {video_key}")
                image_frames[video_key].append(np.asarray(image, dtype=np.uint8))

            rows.append(row)

        parquet_path = self.root / self.info["data_path"].format(
            episode_chunk=episode_chunk,
            episode_index=episode_index,
        )
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(parquet_path, index=False)

        for video_key, frames_for_video in image_frames.items():
            video_path = self.root / self.info["video_path"].format(
                episode_chunk=episode_chunk,
                video_key=video_key,
                episode_index=episode_index,
            )
            _write_video(video_path, frames_for_video, self.fps)
            episode_arrays[video_key] = frames_for_video

        episode_stats = compute_episode_stats(episode_arrays, self.features)
        _append_jsonl(
            self.root / "meta" / "episodes_stats.jsonl",
            {"episode_index": episode_index, "stats": episode_stats},
        )

        episode_record = {"episode_index": episode_index, "tasks": [task], "length": len(frames)}
        _append_jsonl(self.root / "meta" / "episodes.jsonl", episode_record)

        self.info["total_episodes"] += 1
        self.info["total_frames"] += len(frames)
        self.info["total_tasks"] = len(_read_jsonl(self.root / "meta" / "tasks.jsonl"))
        self.info["total_chunks"] = max(self.info["total_chunks"], episode_chunk + 1)
        self.info["total_videos"] += len(self.video_keys)
        self.info["splits"] = {"train": f"0:{self.info['total_episodes']}"}

        all_episode_stats = [
            item["stats"] for item in _read_jsonl(self.root / "meta" / "episodes_stats.jsonl")
        ]
        _write_json(self.root / "meta" / "stats.json", aggregate_stats(all_episode_stats))
        _write_json(self.info_path, self.info)
        return episode_index


def push_lerobot_dataset_to_hub(
    root: str | Path,
    repo_id: str,
    *,
    private: bool = False,
    branch: str | None = None,
) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    if branch is not None:
        api.create_branch(repo_id=repo_id, repo_type="dataset", branch=branch, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        revision=branch,
        folder_path=str(root),
        ignore_patterns=["images/"],
    )
    try:
        api.create_tag(repo_id, tag=CODEBASE_VERSION, repo_type="dataset", revision=branch)
    except Exception:
        pass
