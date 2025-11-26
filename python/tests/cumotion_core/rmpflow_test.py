# SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES.
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

"""Unit tests for rmpflow_python.h."""

# Standard Library
import os

# Third Party
import numpy as np
import pytest

# cuMotion
import cumotion

# Local directory
from ._test_helper import CUMOTION_ROOT_DIR, warnings_disabled


@pytest.fixture
def configure_franka_rmpflow_test():
    """Test fixture to configure robot description and RMPFlow objects."""
    def _configure_franka_rmpflow_test():
        # Set directory for RMPflow configuration and robot description YAML files.
        config_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'nvidia', 'shared')

        # Set absolute path to RMPflow configuration for Franka.
        rmpflow_config_path = os.path.join(
            config_path, 'franka_rmpflow_config_without_point_cloud.yaml')

        # Set absolute path to the XRDF for Franka.
        xrdf_path = os.path.join(config_path, "franka.xrdf")

        # Set absolute path to URDF file for Franka.
        urdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'third_party', 'franka',
                                 'franka.urdf')

        # Set end effector frame for Franka.
        end_effector_frame_name = 'right_gripper'

        # Load robot description.
        robot_description = cumotion.load_robot_from_file(xrdf_path, urdf_path)

        # Create world for obstacles.
        world = cumotion.create_world()
        world_view = world.add_world_view()

        # Create RMPflow configuration.
        rmpflow_config = cumotion.create_rmpflow_config_from_file(rmpflow_config_path,
                                                                  robot_description,
                                                                  end_effector_frame_name,
                                                                  world_view)

        # Create RMPflow policy.
        rmpflow = cumotion.create_rmpflow(rmpflow_config)

        # Return robot_description and rmpflow
        return (robot_description, world, world_view, rmpflow)

    return _configure_franka_rmpflow_test


def test_franka_rmpflow(configure_franka_rmpflow_test):
    """Test the Franka RMPFlow interface."""
    # Init robot_description and rmpflow from test fixture
    robot_description, world, world_view, rmpflow = configure_franka_rmpflow_test()

    # Expect that c-space values of all zeros exceed c-space limits (due to "panda_joint4").
    zero = np.zeros(7)
    with warnings_disabled:
        assert robot_description.kinematics().within_cspace_limits(zero, True) is False

    # Test that valid joint positions are within bounds.
    valid_position = np.zeros(7)
    valid_position[3] = -1.5
    assert robot_description.kinematics().within_cspace_limits(valid_position, True)

    # Add spherical obstacle in front of robot arm.
    sphere_obstacle = cumotion.create_obstacle(cumotion.Obstacle.Type.SPHERE)
    sphere_obstacle_radius = 0.05
    sphere_obstacle.set_attribute(cumotion.Obstacle.Attribute.RADIUS, sphere_obstacle_radius)
    sphere_obstacle_pose = cumotion.Pose3.from_translation(np.array([0.3, 0.0, 0.6]))
    world.add_obstacle(sphere_obstacle, sphere_obstacle_pose)

    # Add box obstacle in front of robot arm.
    box_obstacle = cumotion.create_obstacle(cumotion.Obstacle.Type.CUBOID)
    box_obstacle_length = 0.1
    box_obstacle.set_attribute(cumotion.Obstacle.Attribute.SIDE_LENGTHS,
                               box_obstacle_length * np.ones(3))
    box_obstacle_pose = cumotion.Pose3.from_translation(np.array([0.4, 0.0, 0.2]))
    world.add_obstacle(box_obstacle, box_obstacle_pose)

    # Update world view after adding obstacles.
    world_view.update()

    # Set default configuration
    default_cspace_config = np.array([0.00, -1.3, 0.00, -2.87, 0.00, 2.00, 0.75])

    # Set end effector position target.
    end_effector_position_target = np.array([0.8, 0.0, 0.35])
    rmpflow.set_end_effector_position_attractor(end_effector_position_target)
    rmpflow.set_cspace_attractor(default_cspace_config)

    # Set joint positions and joint velocities
    joint_position = default_cspace_config
    joint_velocity = np.zeros(robot_description.num_cspace_coords())

    # Initialize joint acceleration vector
    joint_accel = np.zeros(robot_description.num_cspace_coords())

    # Evaluate acceleration from joint state
    rmpflow.eval_accel(joint_position, joint_velocity, joint_accel)

    # Test acceleration values
    joint_accel_expected = np.array([-0.149459,
                                     -24.1595,
                                     -0.167684,
                                     -40.4155,
                                     0.0096364,
                                     -16.815,
                                     0.0494])

    for idx, (expected, actual) in enumerate(zip(joint_accel_expected, joint_accel)):
        assert expected == pytest.approx(actual, abs=1e-3), 'joint_accel[{}]'.format(idx)

    # Evaluate force and metric from joint state
    joint_force, joint_metric = rmpflow.eval_force_and_metric(joint_position, joint_velocity)

    # Test force values
    joint_force_expected = np.array([-61991.6877,
                                     -7534653.33,
                                     -69744.7161,
                                     -13006465.844,
                                     4135.46402,
                                     -5397935.177,
                                     20835.0061])

    for expected, actual in zip(joint_force_expected, joint_force):
        assert expected == pytest.approx(actual, abs=1e-3)

    # Test metric values
    joint_metric_expected = np.array([[775.650655,
                                       -14.3174561,
                                       686.355989,
                                       336.982731,
                                       105.714143,
                                       125.344953,
                                       -155.738298],
                                      [-14.3174561,
                                       3073.90049,
                                       -19.2232271,
                                       -907.944895,
                                       3.64276457,
                                       63.5595243,
                                       -11.1856694],
                                      [686.355989,
                                       -19.2232271,
                                       754.118668,
                                       384.611192,
                                       41.9560397,
                                       142.726214,
                                       -175.208925],
                                      [336.982731,
                                       -907.944895,
                                       384.611192,
                                       10517.9771,
                                       -27.0197790,
                                       3770.55119,
                                       -110.482394],
                                      [105.714143,
                                       3.64276457,
                                       41.9560397,
                                       -27.0197790,
                                       136.706724,
                                       -9.66406428,
                                       11.4533163],
                                      [125.344953,
                                       63.5595243,
                                       142.726214,
                                       3770.55119,
                                       -9.66406428,
                                       1591.13314,
                                       -42.3265189],
                                      [-155.738298,
                                       -11.1856694,
                                       -175.208925,
                                       -110.482394,
                                       11.4533163,
                                       -42.3265189,
                                       105.208768]])

    for expected_vector, actual_vector in zip(joint_metric_expected, joint_metric):
        for expected, actual in zip(expected_vector, actual_vector):
            assert expected == pytest.approx(actual, abs=1e-3)
