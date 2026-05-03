from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from imitation_utils.dataset_conversion import (
    convert_pickle_dataset_to_lerobot,
    load_config_dict,
    resolve_data_path,
)


def is_openpi_policy(config: dict[str, Any]) -> bool:
    return config.get("policy", {}).get("type") == "openpi"


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_openpi_root() -> Path:
    return get_repo_root() / "models" / "openpi"


def ensure_openpi_imports() -> Path:
    openpi_root = get_openpi_root()
    if not openpi_root.exists():
        raise FileNotFoundError(
            f"openpi repository not found at {openpi_root}. "
            "Initialize the git submodule before using policy.type=openpi."
        )

    extra_paths = [
        openpi_root / "src",
        openpi_root / "packages" / "openpi-client" / "src",
    ]
    for path in extra_paths:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    return openpi_root


def sanitize_name(value: str) -> str:
    return value.lower().replace(" ", "_").replace("/", "_").replace("-", "_")


def get_openpi_section(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("policy", {}).get("openpi", {})


def get_training_section(config: dict[str, Any]) -> dict[str, Any]:
    training = dict(config.get("training", {}))
    training.update(get_openpi_section(config).get("training", {}))
    return training


def get_openpi_paths(config_path: str | Path, config: dict[str, Any]) -> dict[str, Path]:
    base_output_dir = resolve_data_path(config_path, config["paths"]["output_dir"])
    local_dir = resolve_data_path(config_path, config["paths"]["local_dir"])
    paths_cfg = config.get("paths", {})
    return {
        "data_dir": resolve_data_path(config_path, paths_cfg["data_dir"]),
        "output_dir": base_output_dir,
        "local_dir": local_dir,
        "assets_base_dir": resolve_data_path(
            config_path,
            paths_cfg.get("openpi_assets_dir", str(base_output_dir / "openpi_assets")),
        ),
        "checkpoint_base_dir": resolve_data_path(
            config_path,
            paths_cfg.get("openpi_checkpoint_dir", str(base_output_dir / "openpi_checkpoints")),
        ),
    }


def configure_openpi_environment(config_path: str | Path, config: dict[str, Any]) -> dict[str, Path]:
    ensure_openpi_imports()
    paths = get_openpi_paths(config_path, config)
    os.environ["LEROBOT_HOME"] = str(paths["local_dir"])
    return paths


def prepare_openpi_dataset(
    *,
    config_path: str | Path | None,
    task_name: str | None = None,
) -> Path:
    resolved_config_path, _, config = load_config_dict(config_path)
    task = task_name or get_openpi_section(config).get("task_name", "manipulation")
    return convert_pickle_dataset_to_lerobot(
        config_path=resolved_config_path,
        task_name=task,
        push_to_hub=False,
        local_dir_override=config["paths"]["local_dir"],
    )


def _parse_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    if image.ndim != 3:
        raise ValueError(f"Expected image with 3 dimensions, got shape {image.shape}")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.transpose(image, (1, 2, 0))
    return image


def get_openpi_runtime_spec(config: dict[str, Any], modality_cfg) -> dict[str, Any]:
    openpi_cfg = get_openpi_section(config)
    camera_mapping = openpi_cfg.get("camera_mapping", {})
    image_names = [mod.name for mod in modality_cfg.image_modalities]

    base_camera = camera_mapping.get("base", image_names[0] if image_names else None)
    left_wrist_camera = camera_mapping.get(
        "left_wrist",
        image_names[1] if len(image_names) > 1 else None,
    )
    right_wrist_camera = camera_mapping.get("right_wrist")

    model_name = openpi_cfg.get("model_name", "pi05").lower()
    if model_name not in {"pi0", "pi05", "pi0_fast"}:
        raise ValueError(f"Unsupported openpi model_name: {model_name}")

    action_dim = modality_cfg.total_action_dim
    model_action_dim = int(openpi_cfg.get("model_action_dim", 32 if model_name == "pi05" else action_dim))
    action_horizon = int(openpi_cfg.get("action_horizon", 15 if model_name == "pi05" else 10))
    max_token_len = int(openpi_cfg.get("max_token_len", 250 if action_dim > 16 else 180))
    n_action_steps = int(openpi_cfg.get("n_action_steps", 1))

    checkpoint_dir = openpi_cfg.get("checkpoint_dir")
    if checkpoint_dir is None:
        checkpoint_dir = {
            "pi0": "gs://openpi-assets/checkpoints/pi0_base",
            "pi05": "gs://openpi-assets/checkpoints/pi05_base",
            "pi0_fast": "gs://openpi-assets/checkpoints/pi0_fast_base",
        }[model_name]

    return {
        "model_name": model_name,
        "base_camera": base_camera,
        "left_wrist_camera": left_wrist_camera,
        "right_wrist_camera": right_wrist_camera,
        "action_dim": action_dim,
        "model_action_dim": model_action_dim,
        "action_horizon": action_horizon,
        "max_token_len": max_token_len,
        "n_action_steps": n_action_steps,
        "default_prompt": openpi_cfg.get("default_prompt"),
        "prompt_from_task": bool(openpi_cfg.get("prompt_from_task", True)),
        "use_delta_joint_actions": bool(openpi_cfg.get("use_delta_joint_actions", True)),
        "checkpoint_dir": checkpoint_dir,
        "policy_metadata": openpi_cfg.get("policy_metadata"),
        "runtime": openpi_cfg.get("runtime", "local"),
        "server_host": openpi_cfg.get("server_host", "127.0.0.1"),
        "server_port": int(openpi_cfg.get("server_port", 8000)),
        "api_key": openpi_cfg.get("api_key"),
        "discrete_state_input": bool(openpi_cfg.get("discrete_state_input", False)),
        "exp_name": openpi_cfg.get("exp_name", "default"),
    }


def build_openpi_train_config(config_path: str | Path, config: dict[str, Any], modality_cfg):
    configure_openpi_environment(config_path, config)

    from typing_extensions import override

    import openpi.models.model as _model
    import openpi.models.pi0_config as pi0_config
    import openpi.models.pi0_fast as pi0_fast
    import openpi.training.config as _config
    import openpi.training.optimizer as _optimizer
    import openpi.training.weight_loaders as weight_loaders
    import openpi.transforms as _transforms

    runtime_spec = get_openpi_runtime_spec(config, modality_cfg)
    training_cfg = get_training_section(config)
    repo_id = config["paths"]["repo_id"]

    def _camera_slots() -> dict[str, str | None]:
        return {
            "base_0_rgb": runtime_spec["base_camera"],
            "left_wrist_0_rgb": runtime_spec["left_wrist_camera"],
            "right_wrist_0_rgb": runtime_spec["right_wrist_camera"],
        }

    @dataclasses.dataclass(frozen=True)
    class RobotInputs(_transforms.DataTransformFn):
        model_type: _model.ModelType

        def __call__(self, data: dict) -> dict:
            state = np.asarray(data["state"], dtype=np.float32)
            actions = np.asarray(data["actions"], dtype=np.float32) if "actions" in data else None
            source_images = data["images"]

            base_camera = runtime_spec["base_camera"]
            if base_camera is None or base_camera not in source_images:
                raise ValueError(
                    "openpi requires at least one image modality mapped to camera_mapping.base"
                )

            base_image = _parse_image(source_images[base_camera])
            images = {}
            image_masks = {}

            for slot_name, source_name in _camera_slots().items():
                if source_name is not None and source_name in source_images:
                    images[slot_name] = _parse_image(source_images[source_name])
                    image_masks[slot_name] = np.True_
                else:
                    images[slot_name] = np.zeros_like(base_image)
                    image_masks[slot_name] = (
                        np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_
                    )

            result = {
                "state": state,
                "image": images,
                "image_mask": image_masks,
            }
            if actions is not None:
                result["actions"] = actions
            if "prompt" in data:
                prompt = data["prompt"]
                if isinstance(prompt, bytes):
                    prompt = prompt.decode("utf-8")
                result["prompt"] = prompt
            return result

    @dataclasses.dataclass(frozen=True)
    class RobotOutputs(_transforms.DataTransformFn):
        action_dim: int

        def __call__(self, data: dict) -> dict:
            return {"actions": np.asarray(data["actions"][:, : self.action_dim])}

    @dataclasses.dataclass(frozen=True)
    class LeRobotRobotDataConfig(_config.DataConfigFactory):
        @override
        def create(self, assets_dirs: Path, model_config: _model.BaseModelConfig) -> _config.DataConfig:
            repack_structure = {
                "images": {
                    camera_name: f"observation.images.{camera_name}"
                    for camera_name in {
                        runtime_spec["base_camera"],
                        runtime_spec["left_wrist_camera"],
                        runtime_spec["right_wrist_camera"],
                    }
                    if camera_name is not None
                },
                "state": "observation.state",
                "actions": "action",
            }
            if runtime_spec["prompt_from_task"]:
                repack_structure["prompt"] = "prompt"

            data_transforms = _transforms.Group(
                inputs=[RobotInputs(model_type=model_config.model_type)],
                outputs=[RobotOutputs(action_dim=runtime_spec["action_dim"])],
            )

            if runtime_spec["use_delta_joint_actions"]:
                joint_dims = len(modality_cfg.joint_names)
                hand_dims = max(runtime_spec["action_dim"] - joint_dims, 0)
                delta_mask = _transforms.make_bool_mask(joint_dims, -hand_dims)
                data_transforms = data_transforms.push(
                    inputs=[_transforms.DeltaActions(delta_mask)],
                    outputs=[_transforms.AbsoluteActions(delta_mask)],
                )

            return dataclasses.replace(
                self.create_base_config(assets_dirs, model_config),
                repack_transforms=_transforms.Group(
                    inputs=[_transforms.RepackTransform(repack_structure)]
                ),
                data_transforms=data_transforms,
                model_transforms=_config.ModelTransformFactory(
                    default_prompt=runtime_spec["default_prompt"]
                )(model_config),
                action_sequence_keys=("action",),
            )

    if runtime_spec["model_name"] == "pi05":
        model = pi0_config.Pi0Config(
            pi05=True,
            action_dim=runtime_spec["model_action_dim"],
            action_horizon=runtime_spec["action_horizon"],
            max_token_len=runtime_spec["max_token_len"],
            discrete_state_input=runtime_spec["discrete_state_input"],
        )
        initial_checkpoint = "gs://openpi-assets/checkpoints/pi05_base/params"
    elif runtime_spec["model_name"] == "pi0_fast":
        model = pi0_fast.Pi0FASTConfig(
            action_dim=runtime_spec["model_action_dim"],
            action_horizon=runtime_spec["action_horizon"],
            max_token_len=runtime_spec["max_token_len"],
        )
        initial_checkpoint = "gs://openpi-assets/checkpoints/pi0_fast_base/params"
    else:
        model = pi0_config.Pi0Config(
            action_dim=runtime_spec["model_action_dim"],
            action_horizon=runtime_spec["action_horizon"],
            max_token_len=runtime_spec["max_token_len"],
        )
        initial_checkpoint = "gs://openpi-assets/checkpoints/pi0_base/params"

    paths = get_openpi_paths(config_path, config)
    lr = float(training_cfg.get("learning_rate", 5e-5))
    num_train_steps = int(training_cfg.get("total_steps", 20_000))
    batch_size = int(training_cfg.get("batch_size", 32))
    num_workers = int(training_cfg.get("num_workers", 2))
    overwrite = bool(training_cfg.get("overwrite", False))
    resume = bool(training_cfg.get("resume", False))
    wandb_enabled = bool(training_cfg.get("wandb_enabled", True))
    ema_decay = training_cfg.get("ema_decay", 0.999)

    config_name = sanitize_name(
        get_openpi_section(config).get(
            "config_name",
            f"{config['robot']['type']}_{runtime_spec['model_name']}",
        )
    )
    exp_name = get_openpi_section(config).get("exp_name", "default")

    return _config.TrainConfig(
        name=config_name,
        exp_name=exp_name,
        model=model,
        data=LeRobotRobotDataConfig(
            repo_id=repo_id,
            assets=_config.AssetsConfig(asset_id=repo_id),
            base_config=_config.DataConfig(
                prompt_from_task=runtime_spec["prompt_from_task"],
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            get_openpi_section(config).get("initial_checkpoint", initial_checkpoint)
        ),
        assets_base_dir=str(paths["assets_base_dir"]),
        checkpoint_base_dir=str(paths["checkpoint_base_dir"]),
        batch_size=batch_size,
        num_workers=num_workers,
        num_train_steps=num_train_steps,
        overwrite=overwrite,
        resume=resume,
        wandb_enabled=wandb_enabled,
        ema_decay=None if ema_decay is None else float(ema_decay),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=int(training_cfg.get("warmup_steps", 1_000)),
            peak_lr=lr,
            decay_steps=int(training_cfg.get("decay_steps", max(num_train_steps, 1_000))),
            decay_lr=float(training_cfg.get("decay_learning_rate", lr)),
        ),
        optimizer=_optimizer.AdamW(
            clip_gradient_norm=float(training_cfg.get("grad_clip_norm", 1.0)),
            weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
        ),
        policy_metadata=runtime_spec["policy_metadata"],
    )


def compute_openpi_norm_stats(train_config, *, max_frames: int | None = None) -> Path:
    import openpi.scripts.compute_norm_stats as compute_norm_stats
    import openpi.shared.normalize as normalize

    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = compute_norm_stats.create_rlds_dataloader(
            data_config,
            train_config.model.action_horizon,
            train_config.batch_size,
            max_frames=max_frames,
        )
    else:
        data_loader, num_batches = compute_norm_stats.create_torch_dataloader(
            data_config,
            train_config.model.action_horizon,
            train_config.batch_size,
            train_config.model,
            train_config.num_workers,
            max_frames=max_frames,
        )

    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    for batch in data_loader:
        for key in ("state", "actions"):
            stats[key].update(np.asarray(batch[key]))
        num_batches -= 1
        if num_batches == 0:
            break

    norm_stats = {key: value.get_statistics() for key, value in stats.items()}
    output_path = train_config.assets_dirs / data_config.repo_id
    normalize.save(output_path, norm_stats)
    return output_path


def run_openpi_training(config_path: str | Path | None):
    resolved_config_path, modality_cfg, config = load_config_dict(config_path)
    prepare_openpi_dataset(config_path=resolved_config_path)
    train_config = build_openpi_train_config(resolved_config_path, config, modality_cfg)
    compute_openpi_norm_stats(train_config)

    framework = get_openpi_section(config).get("training_framework", "jax").lower()
    if framework == "pytorch":
        import openpi.scripts.train_pytorch as train_pytorch

        train_pytorch.init_logging()
        train_pytorch.train_loop(train_config)
    else:
        import openpi.scripts.train as train_jax

        train_jax.main(train_config)

    return train_config


def make_identity_norm_stats(action_dim: int):
    import openpi.shared.normalize as normalize

    zeros = np.zeros(action_dim, dtype=np.float32)
    ones = np.ones(action_dim, dtype=np.float32)
    return {
        "state": normalize.NormStats(mean=zeros, std=ones, q01=-ones, q99=ones),
        "actions": normalize.NormStats(mean=zeros, std=ones, q01=-ones, q99=ones),
    }


def maybe_load_local_openpi_norm_stats(train_config):
    import openpi.shared.normalize as normalize

    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    stats_dir = train_config.assets_dirs / data_config.repo_id
    if stats_dir.exists():
        return normalize.load(stats_dir)
    return None


def create_openpi_policy(
    *,
    config_path: str | Path | None,
    checkpoint_dir: str | Path | None = None,
):
    resolved_config_path, modality_cfg, config = load_config_dict(config_path)
    train_config = build_openpi_train_config(resolved_config_path, config, modality_cfg)
    runtime_spec = get_openpi_runtime_spec(config, modality_cfg)

    if checkpoint_dir is None:
        checkpoint_dir = runtime_spec["checkpoint_dir"]

    norm_stats = maybe_load_local_openpi_norm_stats(train_config)
    if norm_stats is None and str(checkpoint_dir).startswith("gs://"):
        norm_stats = make_identity_norm_stats(runtime_spec["model_action_dim"])

    from openpi.policies import policy_config

    return policy_config.create_trained_policy(
        train_config,
        checkpoint_dir,
        default_prompt=runtime_spec["default_prompt"],
        norm_stats=norm_stats,
    )


def create_openpi_client(
    *,
    config_path: str | Path | None,
):
    resolved_config_path, modality_cfg, config = load_config_dict(config_path)
    runtime_spec = get_openpi_runtime_spec(config, modality_cfg)
    configure_openpi_environment(resolved_config_path, config)

    from openpi_client import websocket_client_policy

    return websocket_client_policy.WebsocketClientPolicy(
        host=runtime_spec["server_host"],
        port=runtime_spec["server_port"],
        api_key=runtime_spec["api_key"],
    )


def build_openpi_observation(
    *,
    state: np.ndarray,
    images: dict[str, np.ndarray],
    prompt: str | None,
    config: dict[str, Any],
    modality_cfg,
) -> dict[str, Any]:
    runtime_spec = get_openpi_runtime_spec(config, modality_cfg)
    return {
        "state": np.asarray(state, dtype=np.float32),
        "images": {
            name: _parse_image(image)
            for name, image in images.items()
        },
        "prompt": prompt or runtime_spec["default_prompt"] or "",
    }


class OpenPIActionRunner:
    def __init__(self, policy, n_action_steps: int):
        self.policy = policy
        self.n_action_steps = max(1, n_action_steps)
        self._chunk: np.ndarray | None = None
        self._index = 0

    def reset(self) -> None:
        self._chunk = None
        self._index = 0

    def next_action(self, observation: dict[str, Any]) -> np.ndarray:
        if self._chunk is None or self._index >= min(len(self._chunk), self.n_action_steps):
            result = self.policy.infer(observation)
            self._chunk = np.asarray(result["actions"], dtype=np.float32)
            self._index = 0

        action = self._chunk[self._index]
        self._index += 1
        return action
