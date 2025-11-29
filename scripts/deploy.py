#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import JointState, Image  # noqa: F401
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from cv_bridge import CvBridge
import torch
import numpy as np
import cv2
import actionlib
from lerobot.policies.act.modeling_act import ACTPolicy, ACTTemporalEnsembler
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
import threading
import yaml
from imitation_utils.modality_config import ModalityConfig

rospy.init_node("policy_deployer")
bridge = CvBridge()


class PolicyDeployer:
    def __init__(self, max_episode_steps=500):
        config_path = rospy.get_param("~config", None)
        model_path = rospy.get_param("~model_path", None)

        if config_path:
            with open(config_path) as f:
                config = yaml.safe_load(f)
        else:
            self.cfg = ModalityConfig()
            with open(self.cfg.config_path) as f:
                config = yaml.safe_load(f)

        self.cfg = ModalityConfig(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_episode_steps = max_episode_steps
        policy_type = config["policy"]["type"]

        if model_path is None:
            model_path = config["paths"]["output_dir"]

        if policy_type == "act":
            self.policy = ACTPolicy.from_pretrained(model_path)
            if self.policy.config.temporal_ensemble_coeff is None:
                self.policy.config.temporal_ensemble_coeff = config["policy"][
                    "act"
                ].get("temporal_ensemble_coeff", None)
                self.policy.temporal_ensembler = ACTTemporalEnsembler(
                    self.policy.config.temporal_ensemble_coeff,
                    self.policy.config.chunk_size,
                )
                rospy.loginfo(
                    f"Enabled temporal ensemble with coeff={self.policy.config.temporal_ensemble_coeff}"
                )

        elif policy_type == "diffusion":
            self.policy = DiffusionPolicy.from_pretrained(model_path)
            rospy.loginfo("Loaded Diffusion Policy")

        else:
            raise ValueError(f"Unknown policy type: {policy_type}")

        self.policy.eval()
        self.policy.to(self.device)

        self.body_client = actionlib.SimpleActionClient(
            self.cfg.action_clients["body"], FollowJointTrajectoryAction
        )
        self.rhand_client = actionlib.SimpleActionClient(
            self.cfg.action_clients["rhand"], FollowJointTrajectoryAction
        )
        self.lhand_client = actionlib.SimpleActionClient(
            self.cfg.action_clients["lhand"], FollowJointTrajectoryAction
        )

        self.body_client.wait_for_server()
        self.rhand_client.wait_for_server()
        self.lhand_client.wait_for_server()

        self.body_joint_names = self.cfg.joint_names
        self.init_pose = self.cfg.init_pose

        self.lock = threading.Lock()
        self.latest_data = {m.name: None for m in self.cfg.state_modalities}
        self.latest_data.update({m.name: None for m in self.cfg.env_state_modalities})
        self.latest_data.update({m.name: None for m in self.cfg.image_modalities})

        for mod in self.cfg.state_modalities:
            msg_type = self._get_msg_class(mod.msg_type)
            rospy.Subscriber(
                mod.topic,
                msg_type,
                lambda msg, name=mod.name, field=mod.field: self._state_cb(
                    msg, name, field
                ),
            )

        for mod in self.cfg.env_state_modalities:
            msg_type = self._get_msg_class(mod.msg_type)
            rospy.Subscriber(
                mod.topic,
                msg_type,
                lambda msg, name=mod.name, field=mod.field: self._env_state_cb(
                    msg, name, field
                ),
            )

        for mod in self.cfg.image_modalities:
            rospy.Subscriber(
                mod.topic, Image, lambda msg, name=mod.name: self._image_cb(msg, name)
            )

        self.debug_pubs = {}
        for mod in self.cfg.image_modalities:
            self.debug_pubs[mod.name] = rospy.Publisher(
                f"/debug/{mod.name}_input", Image, queue_size=1
            )

        while not rospy.is_shutdown():
            with self.lock:
                if all(v is not None for v in self.latest_data.values()):
                    break
            rospy.sleep(0.1)

        self.rate = rospy.Rate(30)

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
                self.latest_data[name] = obj
            else:
                self.latest_data[name] = getattr(msg, field)

    def _image_cb(self, msg, name):
        with self.lock:
            self.latest_data[name] = msg

    def move_to_init_pose(self):
        body_goal = FollowJointTrajectoryGoal()
        body_goal.trajectory.joint_names = self.body_joint_names
        body_point = JointTrajectoryPoint(
            positions=self.init_pose[: len(self.body_joint_names)].tolist(),
            time_from_start=rospy.Duration(2.0),
        )
        body_goal.trajectory.points.append(body_point)

        rhand_goal = FollowJointTrajectoryGoal()
        rhand_goal.trajectory.joint_names = [self.cfg.hand_joint_names[0]]
        rhand_point = JointTrajectoryPoint(
            positions=[self.init_pose[len(self.body_joint_names)]],
            time_from_start=rospy.Duration(2.0),
        )
        rhand_goal.trajectory.points.append(rhand_point)

        lhand_goal = FollowJointTrajectoryGoal()
        lhand_goal.trajectory.joint_names = [self.cfg.hand_joint_names[1]]
        lhand_point = JointTrajectoryPoint(
            positions=[self.init_pose[len(self.body_joint_names) + 1]],
            time_from_start=rospy.Duration(2.0),
        )
        lhand_goal.trajectory.points.append(lhand_point)

        self.body_client.send_goal(body_goal)
        self.rhand_client.send_goal(rhand_goal)
        self.lhand_client.send_goal(lhand_goal)

        self.body_client.wait_for_result()
        rospy.sleep(1.0)

    def get_observation(self):
        with self.lock:
            state_list = []
            for mod in self.cfg.state_modalities:
                val = self.latest_data[mod.name]
                if isinstance(val, (list, tuple)):
                    state_list.extend(val)
                else:
                    state_list.append(val)
            state = np.array(state_list, dtype=np.float32)

            env_state_list = []
            for mod in self.cfg.env_state_modalities:
                val = self.latest_data[mod.name]
                if isinstance(val, (list, tuple)):
                    env_state_list.extend(val)
                else:
                    env_state_list.append(val)
            env_state = (
                np.array(env_state_list, dtype=np.float32) if env_state_list else None
            )

            images = {}
            for mod in self.cfg.image_modalities:
                img_msg = self.latest_data[mod.name]
                img = bridge.imgmsg_to_cv2(img_msg, "rgb8")
                img = self.cfg.crop_image(img, mod.name)
                img = cv2.resize(img, mod.resolution)
                images[mod.name] = img

        for name, img in images.items():
            self.debug_pubs[name].publish(bridge.cv2_to_imgmsg(img, "rgb8"))

        state = (
            torch.from_numpy(state)
            .to(torch.float32)
            .to(self.device, non_blocking=True)
            .unsqueeze(0)
        )

        result = {"observation.state": state}

        if env_state is not None:
            env_state = (
                torch.from_numpy(env_state)
                .to(torch.float32)
                .to(self.device, non_blocking=True)
                .unsqueeze(0)
            )
            result["observation.environment_state"] = env_state

        for mod in self.cfg.image_modalities:
            img_tensor = torch.from_numpy(images[mod.name]).to(torch.float32) / 255.0
            img_tensor = (
                img_tensor.permute(2, 0, 1)
                .to(self.device, non_blocking=True)
                .unsqueeze(0)
            )
            result[f"observation.images.{mod.name}"] = img_tensor

        return result

    def execute_action(self, action):
        body_goal = FollowJointTrajectoryGoal()
        body_goal.trajectory.joint_names = self.body_joint_names
        body_point = JointTrajectoryPoint(
            positions=action[: len(self.body_joint_names)].tolist(),
            time_from_start=rospy.Duration(0.1),
        )
        body_goal.trajectory.points.append(body_point)

        rhand_goal = FollowJointTrajectoryGoal()
        rhand_goal.trajectory.joint_names = [self.cfg.hand_joint_names[0]]
        rhand_point = JointTrajectoryPoint(
            positions=[action[len(self.body_joint_names)]],
            time_from_start=rospy.Duration(0.1),
        )
        rhand_goal.trajectory.points.append(rhand_point)

        lhand_goal = FollowJointTrajectoryGoal()
        lhand_goal.trajectory.joint_names = [self.cfg.hand_joint_names[1]]
        lhand_point = JointTrajectoryPoint(
            positions=[action[len(self.body_joint_names) + 1]],
            time_from_start=rospy.Duration(0.1),
        )
        lhand_goal.trajectory.points.append(lhand_point)

        self.body_client.send_goal(body_goal)
        self.rhand_client.send_goal(rhand_goal)
        self.lhand_client.send_goal(lhand_goal)

    def run(self):
        self.move_to_init_pose()
        while not rospy.is_shutdown():
            self.policy.reset()

            for step in range(self.max_episode_steps):
                if rospy.is_shutdown():
                    break

                observation = self.get_observation()

                with torch.inference_mode():
                    action = self.policy.select_action(observation)

                numpy_action = action.squeeze(0).to("cpu").numpy()

                self.execute_action(numpy_action)
                self.rate.sleep()


if __name__ == "__main__":
    deployer = PolicyDeployer()
    deployer.run()
