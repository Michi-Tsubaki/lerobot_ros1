#!/usr/bin/env python3

from pathlib import Path
import torch
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.configs.types import PolicyFeature, FeatureType
from itertools import cycle
import wandb
from torch.amp import GradScaler
from torchvision.transforms import v2
import yaml
from imitation_utils.modality_config import ModalityConfig
import rospy

rospy.init_node("policy_trainer", anonymous=True)

config_path = rospy.get_param("~config", None)
cfg = ModalityConfig(config_path)

with open(config_path if config_path else cfg.config_path) as f:
    config = yaml.safe_load(f)


class RobotDataset(Dataset):
    def __init__(self, data_dir, cfg, chunk_size=48, transforms=None):
        self.data_dir = Path(data_dir)
        self.cfg = cfg
        episode_files = sorted(self.data_dir.glob("episode_*.pkl"))
        self.episodes = []
        for ep_file in episode_files:
            with open(ep_file, "rb") as f:
                self.episodes.append(pickle.load(f))
        self.chunk_size = chunk_size
        self.transforms = transforms
        self.samples = []
        for ep in self.episodes:
            for i in range(len(ep) - chunk_size):
                self.samples.append((ep, i))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ep, start = self.samples[idx]
        action = np.array(
            [ep[i]["state"] for i in range(start + 1, start + self.chunk_size + 1)]
        )

        result = {
            "observation.state": torch.FloatTensor(ep[start]["state"]),
            "action": torch.FloatTensor(action),
            "action_is_pad": torch.zeros(self.chunk_size, dtype=torch.bool),
        }

        if "env_state" in ep[start]:
            result["observation.environment_state"] = torch.FloatTensor(
                ep[start]["env_state"]
            )

        for img_mod in self.cfg.image_modalities:
            img = (
                torch.FloatTensor(ep[start][img_mod.data_key]).permute(2, 0, 1) / 255.0
            )
            if self.transforms:
                img = self.transforms(img)
            result[f"observation.images.{img_mod.name}"] = img

        return result


def compute_stats(data_dir, cfg):
    data_dir = Path(data_dir)
    episode_files = sorted(data_dir.glob("episode_*.pkl"))
    all_states = []
    all_env_states = []

    for ep_file in episode_files:
        with open(ep_file, "rb") as f:
            episode = pickle.load(f)
        for frame in episode:
            all_states.append(frame["state"])
            if "env_state" in frame:
                all_env_states.append(frame["env_state"])

    all_states = np.array(all_states)

    stats = {
        "observation.state": {
            "mean": torch.FloatTensor(all_states.mean(axis=0)),
            "std": torch.FloatTensor(all_states.std(axis=0) + 1e-6),
            "min": torch.FloatTensor(all_states.min(axis=0)),
            "max": torch.FloatTensor(all_states.max(axis=0)),
        },
        "action": {
            "mean": torch.FloatTensor(all_states.mean(axis=0)),
            "std": torch.FloatTensor(all_states.std(axis=0) + 1e-6),
            "min": torch.FloatTensor(all_states.min(axis=0)),
            "max": torch.FloatTensor(all_states.max(axis=0)),
        },
    }

    if all_env_states:
        all_env_states = np.array(all_env_states)
        stats["observation.environment_state"] = {
            "mean": torch.FloatTensor(all_env_states.mean(axis=0)),
            "std": torch.FloatTensor(all_env_states.std(axis=0) + 1e-6),
            "min": torch.FloatTensor(all_env_states.min(axis=0)),
            "max": torch.FloatTensor(all_env_states.max(axis=0)),
        }

    for img_mod in cfg.image_modalities:
        stats[f"observation.images.{img_mod.name}"] = {
            "mean": torch.FloatTensor([0.5, 0.5, 0.5]).reshape(3, 1, 1),
            "std": torch.FloatTensor([0.5, 0.5, 0.5]).reshape(3, 1, 1),
            "min": torch.FloatTensor([0.0, 0.0, 0.0]).reshape(3, 1, 1),
            "max": torch.FloatTensor([1.0, 1.0, 1.0]).reshape(3, 1, 1),
        }

    return stats


def build_transforms(aug_cfg):
    if not any(
        [
            aug_cfg.get("use_blur"),
            aug_cfg.get("use_occlusion"),
            aug_cfg.get("use_brightness"),
            aug_cfg.get("use_saturation"),
            aug_cfg.get("use_sharpness"),
        ]
    ):
        return None

    transforms_list = []

    if aug_cfg.get("use_blur"):
        blur_tfms = [
            v2.GaussianBlur(kernel_size=(1, 1), sigma=(0.01, 0.01)),
            v2.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
            v2.GaussianBlur(kernel_size=(9, 9), sigma=(0.1, 2.0)),
            v2.GaussianBlur(kernel_size=(13, 13), sigma=(0.1, 2.0)),
            v2.GaussianBlur(kernel_size=(15, 15), sigma=(0.1, 2.0)),
        ]
        transforms_list.append(
            v2.RandomChoice(blur_tfms, p=[0.4, 0.15, 0.15, 0.15, 0.15])
        )

    if aug_cfg.get("use_occlusion"):
        transforms_list.append(
            v2.RandomErasing(
                p=aug_cfg.get("occlusion_prob", 0.1),
                scale=(0.005, 0.20),
                ratio=(0.5, 2.0),
                value=0,
            )
        )

    if aug_cfg.get("use_brightness"):
        br = aug_cfg.get("brightness_range", [0.8, 1.2])
        transforms_list.append(v2.ColorJitter(brightness=(br[0], br[1])))

    if aug_cfg.get("use_saturation"):
        sr = aug_cfg.get("saturation_range", [0.8, 1.2])
        transforms_list.append(v2.ColorJitter(saturation=(sr[0], sr[1])))

    if aug_cfg.get("use_sharpness"):
        transforms_list.append(
            v2.RandomAdjustSharpness(
                sharpness_factor=aug_cfg.get("sharpness_factor", 2.0),
                p=aug_cfg.get("sharpness_prob", 0.5),
            )
        )

    return v2.Compose(transforms_list) if transforms_list else None


policy_type = config["policy"]["type"]
train_cfg = config["training"]
aug_cfg = config["augmentation"]
paths = config["paths"]

output_directory = Path(paths["output_dir"])
output_directory.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

stats = compute_stats(paths["data_dir"], cfg)
transforms = build_transforms(aug_cfg)

input_features = cfg.get_act_input_features()
output_features = {
    "action": PolicyFeature(shape=[cfg.total_action_dim], type=FeatureType.ACTION)
}

if policy_type == "act":
    policy_cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        n_obs_steps=1,
        chunk_size=int(config["policy"]["act"]["chunk_size"]),
        n_action_steps=int(config["policy"]["act"]["n_action_steps"]),
        optimizer_lr=float(train_cfg["learning_rate"]),
        use_amp=True,
    )
    policy = ACTPolicy(policy_cfg, dataset_stats=stats)
    chunk_size = config["policy"]["act"]["chunk_size"]

elif policy_type == "diffusion":
    policy_cfg = DiffusionConfig(
        input_features=input_features,
        output_features=output_features,
        horizon=int(config["policy"]["diffusion"]["horizon"]),
        n_action_steps=int(config["policy"]["diffusion"]["n_action_steps"]),
        num_inference_steps=int(config["policy"]["diffusion"]["num_inference_steps"]),
        optimizer_lr=float(train_cfg["learning_rate"]),
        use_amp=True,
    )
    policy = DiffusionPolicy(policy_cfg, dataset_stats=stats)
    chunk_size = config["policy"]["diffusion"]["horizon"]

else:
    raise ValueError(f"Unknown policy type: {policy_type}")

wandb.init(
    project=f"{config['robot']['type']}-{policy_type}",
    config={**config["policy"][policy_type], **train_cfg, **aug_cfg},
)

policy.train()
policy.to(device)

dataset = RobotDataset(
    paths["data_dir"], cfg, chunk_size=chunk_size, transforms=transforms
)
optimizer = torch.optim.AdamW(
    policy.parameters(), lr=float(train_cfg["learning_rate"]), weight_decay=1e-4
)
grad_scaler = GradScaler(device.type, enabled=True)

dataloader = DataLoader(
    dataset,
    batch_size=train_cfg["batch_size"],
    shuffle=True,
    num_workers=train_cfg["num_workers"],
    pin_memory=device.type == "cuda",
    drop_last=False,
)

dataloader_iter = cycle(dataloader)

for step in range(train_cfg["total_steps"]):
    batch = next(dataloader_iter)
    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    with torch.autocast(device_type=device.type):
        output = policy.forward(batch)
        loss = output["loss"] if isinstance(output, dict) else output[0]
    grad_scaler.scale(loss).backward()
    grad_scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(), train_cfg["grad_clip_norm"], error_if_nonfinite=False
    )
    grad_scaler.step(optimizer)
    grad_scaler.update()
    optimizer.zero_grad()
    wandb.log({"loss": loss.item(), "grad_norm": grad_norm.item()})
    if step % 100 == 0:
        print(
            f"step: {step}/{train_cfg['total_steps']} loss: {loss.item():.3f} grad_norm: {grad_norm.item():.3f}"
        )
    if step % 5000 == 0 and step > 0:
        checkpoint_dir = output_directory / f"step_{step}"
        checkpoint_dir.mkdir(exist_ok=True)
        policy.save_pretrained(checkpoint_dir)
        torch.save(stats, checkpoint_dir / "dataset_stats.pt")
        print(f"Saved checkpoint at step {step}")

policy.save_pretrained(output_directory)
torch.save(stats, output_directory / "dataset_stats.pt")
wandb.save(str(output_directory / "**/*"), base_path=str(output_directory.parent))
wandb.finish()
print(f"Training complete. Model saved to {output_directory}")
