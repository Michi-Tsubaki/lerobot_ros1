import numpy as np
import yaml
import rospkg
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple

@dataclass
class StateModalityConfig:
    name: str
    topic: str
    msg_type: str
    field: str
    dim: int

@dataclass
class ImageModalityConfig:
    name: str
    topic: str
    msg_type: str
    resolution: Tuple[int, int]
    crop: Tuple[float, float, float, float] = None
    
    @property
    def data_key(self):
        return self.name.replace("_camera", "_image")

class ModalityConfig:
    def __init__(self, config_path=None):
        if config_path is None:
            rospack = rospkg.RosPack()
            pkg_path = rospack.get_path('imitation_utils')
            config_path = Path(pkg_path) / 'config' / 'config.yaml'

        self.config_path = config_path
        
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)
        
        self.robot_type = self.cfg["robot"]["type"]
        self.joint_names = self.cfg["robot"].get("joint_names", [])
        self.hand_joint_names = self.cfg["robot"].get("hand_joint_names", [])
        self.init_pose = np.array(self.cfg["robot"].get("init_pose", []))
        self.action_clients = self.cfg["robot"].get("action_clients", {})
        
        self.state_modalities = [
            StateModalityConfig(**m) for m in self.cfg["modalities"]["state"]
        ]
        
        self.env_state_modalities = []
        if "env_state" in self.cfg["modalities"]:
            self.env_state_modalities = [
                StateModalityConfig(**m) for m in self.cfg["modalities"]["env_state"]
            ]
        
        self.image_modalities = [
            ImageModalityConfig(**m) for m in self.cfg["modalities"]["images"]
        ]
        
    @property
    def total_state_dim(self):
        return sum(m.dim for m in self.state_modalities)
    
    @property
    def total_env_state_dim(self):
        return sum(m.dim for m in self.env_state_modalities)
    
    @property
    def total_action_dim(self):
        return self.total_state_dim
    
    @property
    def state_names(self):
        return [m.name for m in self.state_modalities]
    
    @property
    def env_state_names(self):
        return [m.name for m in self.env_state_modalities]
    
    @property
    def image_names(self):
        return [m.name for m in self.image_modalities]
    
    def get_state_topic_info(self):
        return [(m.topic, m.msg_type, m.field) for m in self.state_modalities]
    
    def get_env_state_topic_info(self):
        return [(m.topic, m.msg_type, m.field) for m in self.env_state_modalities]
    
    def get_image_topic_info(self):
        return [(m.topic, m.msg_type) for m in self.image_modalities]
    
    def get_lerobot_features(self):
        features = {}
        features["observation.state"] = {
            "dtype": "float32",
            "shape": (self.total_state_dim,),
            "names": None
        }
        if self.env_state_modalities:
            features["observation.environment_state"] = {
                "dtype": "float32",
                "shape": (self.total_env_state_dim,),
                "names": None
            }
        features["action"] = {
            "dtype": "float32",
            "shape": (self.total_action_dim,),
            "names": None
        }
        for img_mod in self.image_modalities:
            h, w = img_mod.resolution[1], img_mod.resolution[0]
            features[f"observation.images.{img_mod.name}"] = {
                "dtype": "video",
                "shape": (h, w, 3),
                "names": None
            }
        return features
    
    def get_act_input_features(self):
        from lerobot.configs.types import PolicyFeature, FeatureType
        features = {
            "observation.state": PolicyFeature(
                shape=[self.total_state_dim],
                type=FeatureType.STATE
            )
        }
        if self.env_state_modalities:
            features["observation.environment_state"] = PolicyFeature(
                shape=[self.total_env_state_dim],
                type=FeatureType.STATE
            )
        for img_mod in self.image_modalities:
            h, w = img_mod.resolution[1], img_mod.resolution[0]
            features[f"observation.images.{img_mod.name}"] = PolicyFeature(
                shape=[3, h, w],
                type=FeatureType.VISUAL
            )
        return features
    
    def crop_image(self, img, modality_name):
        mod = next((m for m in self.image_modalities if m.name == modality_name), None)
        if mod is None or mod.crop is None:
            return img
        h, w = img.shape[:2]
        y1, y2, x1, x2 = mod.crop
        return img[int(h*y1):int(h*y2), int(w*x1):int(w*x2)]