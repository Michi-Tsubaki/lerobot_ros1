#!/usr/bin/env python3

import rospy

from imitation_utils.dataset_conversion import convert_pickle_dataset_to_lerobot

rospy.init_node("lerobot_uploader", anonymous=True)

local_dir = convert_pickle_dataset_to_lerobot(
    config_path=rospy.get_param("~config", None),
    task_name=rospy.get_param("~task_name", "manipulation"),
    push_to_hub=rospy.get_param("~push_to_hub", True),
)

if rospy.get_param("~push_to_hub", True):
    print(f"Upload completed from {local_dir}")
else:
    print(f"Local save completed at {local_dir}. Skipping upload.")
