#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This example demonstrates how to solve inverse kinematics for the Franka Emika Panda robot."""

# Standard Library
import os
import statistics
from math import pi

# Third Party
import numpy as np

# cuMotion
import cumotion
try:
    from cumotion_vis.visualizer import FrankaVisualization, RenderableType, Visualizer

    ENABLE_VIS = True
except ImportError:
    print("Visualizer not installed. Disabling visualization.")
    ENABLE_VIS = False

# Set cuMotion root directory
CUMOTION_ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))


def cumotion_print_status(success):
    """Print the final status of the example."""
    if success:
        print("CUMOTION EXAMPLE COMPLETED SUCCESSFULLY")
    else:
        print("CUMOTION EXAMPLE FAILED")


def create_target_poses():
    """Create a set of target poses for the Franka's "right_gripper" frame.

    Positions are arranged along a rectangle on the YZ plane in front of Franka.
    """
    poses = []

    # Use constant orientation target with gripper pointing away from Franka along the x-axis
    orientation = cumotion.Rotation3.from_axis_angle(np.array([0.0, 1.0, 0.0]), 0.5 * pi)

    # Define rectangle on YZ plane (with x offset).
    x = 0.5
    min_y = -0.4
    max_y = 0.4
    min_z = 0.5
    max_z = 0.8

    # Discretize positions along rectangle.
    step = 0.01  # 1 cm between poses.
    for y in np.arange(min_y, max_y, step):
        poses.append(cumotion.Pose3(orientation, np.array([x, y, max_z])))

    for z in np.arange(max_z, min_z, -step):
        poses.append(cumotion.Pose3(orientation, np.array([x, max_y, z])))

    for y in np.arange(max_y, min_y, -step):
        poses.append(cumotion.Pose3(orientation, np.array([x, y, min_z])))

    for z in np.arange(min_z, max_z, step):
        poses.append(cumotion.Pose3(orientation, np.array([x, min_y, z])))

    return poses


if __name__ == '__main__':
    # Set directory for Fabric configuration and robot description YAML files.
    config_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'nvidia', 'shared')

    # Set absolute path to the XRDF for Franka.
    xrdf_path = os.path.join(config_path, 'franka.xrdf')

    # Set absolute path to URDF file for Franka.
    urdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'third_party', 'franka', 'franka.urdf')

    # Load robot description.
    robot_description = cumotion.load_robot_from_file(xrdf_path, urdf_path)

    # Load kinematics.
    kinematics = robot_description.kinematics()

    # Set end effector frame for Franka.
    end_effector_frame = 'right_gripper'

    # Define target poses along a rectangle.
    target_poses = create_target_poses()

    # Initialize visualization.
    if ENABLE_VIS:
        visualizer = Visualizer()

        # Add robot arm visualization to scene.
        mesh_folder = os.path.join(CUMOTION_ROOT_DIR, 'content', 'third_party', 'franka', 'meshes')
        robot_visualization = FrankaVisualization(robot_description, mesh_folder, visualizer,
                                                  np.zeros(kinematics.num_cspace_coords()))

        # Add visualization marker for end effector target positions to scene.
        target_config = {
            'radius': 0.01,
            'color': [1.0, 0.5, 0.0]  # orange
        }
        for count, target_pose in enumerate(target_poses):
            target_config['position'] = target_pose.translation
            visualizer.add(RenderableType.MARKER, 'target_' + str(count), target_config)

    # Create configuration for inverse kinematics, after first solve the most recent solution will
    # be added to config for use as a c-space seed.
    config = cumotion.IkConfig()

    # For each target pose, compute and visualize solution to inverse kinematics.
    # The target poses will be looped through repeatedly until visualization window is manually
    # closed.
    success = True
    translation_errors = []
    orientation_errors = []
    for target_pose in target_poses:
        # Solve inverse kinematics.
        results = cumotion.solve_ik(kinematics, target_pose, end_effector_frame, config)

        # Expect IK solution for each pose in this example.
        if not results.success:
            success = False

        # Check final pose error.
        tool_pose = kinematics.pose(results.cspace_position, end_effector_frame)
        translation_errors.append(np.linalg.norm(tool_pose.translation - target_pose.translation))
        orientation_errors.append(
            cumotion.Rotation3.distance(tool_pose.rotation, target_pose.rotation))

        # Add seed to "warm-start" IK to nearby solution.
        config.cspace_seeds = [results.cspace_position]

        # If the visualization window has not been manually closed, update visualization.
        if ENABLE_VIS and visualizer.is_active():
            robot_visualization.set_joint_positions(results.cspace_position)
            visualizer.update()

    print("Translation Error:")
    print("  Mean:    ", statistics.mean(translation_errors))
    print("  Median:  ", statistics.median(translation_errors))
    print("  Std Dev: ", statistics.stdev(translation_errors))
    print("Orientation Error:")
    print("  Mean:    ", statistics.mean(orientation_errors))
    print("  Median:  ", statistics.median(orientation_errors))
    print("  Std Dev: ", statistics.stdev(orientation_errors))
    success = success and statistics.median(translation_errors) < 1e-6
    success = success and statistics.median(orientation_errors) < 1e-3
    cumotion_print_status(success)

    # Close visualization
    if ENABLE_VIS:
        visualizer.close()
