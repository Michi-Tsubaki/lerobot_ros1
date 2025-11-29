cmake_minimum_required(VERSION 3.0.2)
project(imitation_utils)

if("$ENV{ROS_DISTRO}" STREQUAL "indigo" OR
   "$ENV{ROS_DISTRO}" STREQUAL "kinetic" OR
   "$ENV{ROS_DISTRO}" STREQUAL "melodic")
  message(WARNING "Skipping package for unsupported ROS distro: $ENV{ROS_DISTRO}")
  find_package(catkin REQUIRED)
  catkin_package()
  return()
endif()

find_package(catkin REQUIRED COMPONENTS
  catkin_virtualenv
  geometry_msgs
  rospy
  sensor_msgs
  std_msgs
  trajectory_msgs
)

catkin_python_setup()

catkin_generate_virtualenv(
  INPUT_REQUIREMENTS requirements.in
  PYTHON_INTERPRETER python3.10
  USE_SYSTEM_PACKAGES FALSE
  ISOLATE_REQUIREMENTS TRUE
)

catkin_package(
)

catkin_install_python(PROGRAMS
  scripts/collect.py
  scripts/deploy.py
  scripts/eval.py
  scripts/train.py
  scripts/upload_lerobot.py
  scripts/visualize_lerobot.py
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/scripts
)

install(FILES
  requirements.in
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  )

install(DIRECTORY config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  )
