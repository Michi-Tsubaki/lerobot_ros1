#!/usr/bin/env python3

import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
from pathlib import Path
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
import yaml
from imitation_utils.modality_config import ModalityConfig
import rospy

rospy.init_node("policy_evaluator", anonymous=True)

config_path = rospy.get_param("~config", None)
cfg = ModalityConfig(config_path)

with open(config_path if config_path else cfg.config_path) as f:
    config = yaml.safe_load(f)

policy_type = config["policy"]["type"]

output_dir = Path(config["paths"]["output_dir"])
if not output_dir.is_absolute():
    output_dir = Path(config_path).parent / output_dir
model_dir = output_dir.resolve()

if not model_dir.exists() or not (model_dir / "model.safetensors").exists():
    print(f"Model not found in {model_dir}")
    exit(1)

print(f"Using local model: {model_dir}")

data_dir = Path(config["paths"]["data_dir"])
if not data_dir.is_absolute():
    data_dir = Path(config_path).parent / data_dir
data_dir = data_dir.resolve()
episode_files = sorted(data_dir.glob("episode_*.pkl"))

print(f"Available episodes: {len(episode_files)}")
ep_idx = rospy.get_param("~episode_index", 0)
test_ep = pickle.load(open(data_dir / f"episode_{ep_idx:06d}.pkl", "rb"))
print(f"Testing on episode {ep_idx} with {len(test_ep)} frames")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

stats = torch.load(model_dir / "dataset_stats.pt")

if policy_type == "act":
    policy = ACTPolicy.from_pretrained(str(model_dir), dataset_stats=stats)
elif policy_type == "diffusion":
    policy = DiffusionPolicy.from_pretrained(str(model_dir), dataset_stats=stats)
else:
    raise ValueError(f"Unknown policy type: {policy_type}")

policy.eval()
policy.to(device)

states = np.array([f["state"] for f in test_ep])

images = {}
for mod in cfg.image_modalities:
    img_list = []
    for f in test_ep:
        img = f[mod.data_key]
        img = cfg.crop_image(img, mod.name)
        img = cv2.resize(img, mod.resolution)
        img_list.append(img)
    img_array = np.array(img_list)
    img_array = img_array.transpose(0, 3, 1, 2) / 255.0
    images[mod.name] = img_array

predictions = []
with torch.no_grad():
    for i in range(len(states) - 1):
        batch = {"observation.state": torch.FloatTensor(states[i : i + 1]).to(device)}

        if "env_state" in test_ep[i]:
            batch["observation.environment_state"] = (
                torch.FloatTensor(test_ep[i]["env_state"]).unsqueeze(0).to(device)
            )

        for mod in cfg.image_modalities:
            batch[f"observation.images.{mod.name}"] = torch.FloatTensor(
                images[mod.name][i : i + 1]
            ).to(device)

        action = policy.select_action(batch).cpu().numpy()

        if action.ndim == 3:
            action = action[0, 0]
        elif action.ndim == 2:
            action = action[0]

        predictions.append(action)

predictions = np.array(predictions)
ground_truth = states[1 : len(predictions) + 1]
errors = np.abs(ground_truth - predictions)

print(f"\nPredictions shape: {predictions.shape}")
print(f"Ground truth shape: {ground_truth.shape}")
print(f"NaN count in predictions: {np.isnan(predictions).sum()}")

results_dir = Path(rospy.get_param("~results_dir", "../results"))
results_dir.mkdir(exist_ok=True)

state_dim = ground_truth.shape[1]
print(f"Actual state dimension from data: {state_dim}")

joint_names = cfg.joint_names + cfg.hand_joint_names
if len(joint_names) != state_dim:
    joint_names = [f"Dim_{i}" for i in range(state_dim)]

rows = (state_dim + 2) // 3
cols = 3
fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows))

if rows == 1:
    axes = axes.reshape(1, -1)

axes = axes.flatten()

for i in range(state_dim):
    ax = axes[i]
    ax.plot(ground_truth[:, i], label="GT", linewidth=2)
    ax.plot(predictions[:, i], label="Pred", linestyle="--", linewidth=2)
    ax.set_title(joint_names[i])
    ax.set_xlabel("Frame")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True)

for i in range(state_dim, len(axes)):
    axes[i].axis("off")

plt.tight_layout()
plt.savefig(results_dir / f"{policy_type}_ep{ep_idx}_state.png", dpi=150)
print(f"Saved: {results_dir / f'{policy_type}_ep{ep_idx}_state.png'}")

if not np.isnan(errors).any():
    print(f"\nMAE: {np.mean(errors):.4f} rad ({np.degrees(np.mean(errors)):.2f} deg)")
    print(f"Max Error: {np.max(errors):.4f} rad ({np.degrees(np.max(errors)):.2f} deg)")

    print("\nPer-dimension MAE:")
    for i in range(state_dim):
        mae = np.mean(errors[:, i])
        print(f"  {joint_names[i]}: {mae:.4f} rad ({np.degrees(mae):.2f} deg)")
else:
    print("\nERROR: Predictions contain NaN values!")

if rospy.get_param("~show_plot", False):
    plt.show()
