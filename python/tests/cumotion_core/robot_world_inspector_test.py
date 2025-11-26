# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES.
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

"""Unit tests for robot_world_inspector_python.h."""

# Standard Library
import math
import os

# Third Party
import numpy as np
import pytest

# cuMotion
import cumotion

# Local directory
from ._test_helper import CUMOTION_ROOT_DIR


@pytest.fixture
def configure_three_link_arm_robot_world_inspector_test():
    """Test fixture to configure robot description and world objects."""
    def _configure_three_link_arm_robot_world_inspector_test():
        # Set absolute path to the XRDF for the three-link arm.
        xrdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'nvidia', 'shared',
                                 "three_link_arm.xrdf")

        # Set absolute path to URDF file for three-link arm.
        urdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'nvidia', 'robots',
                                 'three_link_arm.urdf')

        # Load robot description.
        robot_description = cumotion.load_robot_from_file(xrdf_path, urdf_path)

        # Create world for obstacles.
        world = cumotion.create_world()

        # Return robot_description and world.
        return robot_description, world

    return _configure_three_link_arm_robot_world_inspector_test


def check_distance(inspector, cspace_position,
                   obstacle_sphere_handle, obstacle_sphere_position, obstacle_sphere_radius,
                   world_collision_sphere_index, collision_sphere_position,
                   collision_sphere_radius):
    """Check that distance from inspector.distance_to_obstacle() matches manual calculation."""
    expected_distance = (np.linalg.norm(obstacle_sphere_position - collision_sphere_position)
                         - obstacle_sphere_radius - collision_sphere_radius)
    actual_distance = inspector.distance_to_obstacle(obstacle_sphere_handle,
                                                     world_collision_sphere_index,
                                                     cspace_position)
    assert expected_distance == pytest.approx(actual_distance)
    return actual_distance


def test_three_link_arm_robot_world_inspector(configure_three_link_arm_robot_world_inspector_test):
    """Test the three-link arm `RobotWorldInspector` interface for collision detection."""
    # Init robot_description and world from test fixture.
    robot_description, world = configure_three_link_arm_robot_world_inspector_test()

    # Add spherical obstacle in front of robot arm.
    sphere_obstacle = cumotion.create_obstacle(cumotion.Obstacle.Type.SPHERE)
    sphere_obstacle_radius = 0.5
    sphere_obstacle.set_attribute(cumotion.Obstacle.Attribute.RADIUS, sphere_obstacle_radius)
    sphere_obstacle_pose = cumotion.Pose3.from_translation(np.array([1.0, 0.0, 0.0]))
    obstacle_handle = world.add_obstacle(sphere_obstacle, sphere_obstacle_pose)

    # Create `RobotWorldInspector`.
    inspector = cumotion.create_robot_world_inspector(robot_description, world.add_world_view())

    # Check that policy has the expected number of collision spheres.
    assert 33 == inspector.num_world_collision_spheres()
    assert 22 == inspector.num_self_collision_spheres()

    # Check that world-collision spheres are associated with the expected frames.
    for i in range(0, 11):
        assert "link2" == inspector.world_collision_sphere_frame_name(i)
    for i in range(11, 22):
        assert "link1" == inspector.world_collision_sphere_frame_name(i)
    for i in range(22, 33):
        assert "link0" == inspector.world_collision_sphere_frame_name(i)

    # Expect failure to query world-collision frame name for invalid indices.
    with pytest.raises(Exception):
        inspector.world_collision_sphere_frame_name(-1)
    with pytest.raises(Exception):
        inspector.world_collision_sphere_frame_name(inspector.num_world_collision_spheres())

    # Check that self-collision spheres are associated with the expected frames.
    for i in range(0, 11):
        assert "link2" == inspector.self_collision_sphere_frame_name(i)
    for i in range(11, 22):
        assert "link0" == inspector.self_collision_sphere_frame_name(i)

    # Expect failure to query self-collision frame name for invalid indices.
    with pytest.raises(Exception):
        inspector.self_collision_sphere_frame_name(-1)
    with pytest.raises(Exception):
        inspector.self_collision_sphere_frame_name(inspector.num_self_collision_spheres())

    # Expect that all world-collision spheres will have the same radius.
    world_collision_sphere_radii = inspector.world_collision_sphere_radii()
    assert inspector.num_world_collision_spheres() == len(world_collision_sphere_radii)
    for radius in world_collision_sphere_radii:
        assert 0.1 == radius

    # Expect that all self-collision spheres will have the same radius.
    self_collision_sphere_radii = inspector.self_collision_sphere_radii()
    assert inspector.num_self_collision_spheres() == len(self_collision_sphere_radii)
    for radius in self_collision_sphere_radii:
        assert 0.1 == radius

    # Set a c-space position expected to be in collision with obstacle.
    collision_position = np.array([0.5 * math.pi, -0.5 * math.pi, -0.5 * math.pi])
    assert inspector.in_collision_with_obstacle(collision_position)

    # Get world-collision sphere positions from inspector.
    world_collision_sphere_positions = (
        inspector.world_collision_sphere_positions(collision_position))

    # Get self-collision sphere positions from inspector.
    self_collision_sphere_positions = (
        inspector.self_collision_sphere_positions(collision_position))

    # Check distances for spheres in first link. First link has 11 equidistant spheres spanning
    # (0,0,0) to (0,1,0).
    sphere_radii = 0.1
    for i in range(22, 33):
        # Compute and test expected sphere position.
        expected_sphere_position = np.array([0.0, 0.1 * (i - 22), 0.0])
        assert expected_sphere_position == pytest.approx(world_collision_sphere_positions[i])
        # NOTE: Self-collision spheres for the first link correspond to indices [11, 22).
        assert expected_sphere_position == pytest.approx(self_collision_sphere_positions[i - 11])

        check_distance(inspector, collision_position,
                       obstacle_handle, sphere_obstacle_pose.translation, sphere_obstacle_radius,
                       i, expected_sphere_position, sphere_radii)

    # Check distances for spheres in second link. Second link has 11 equidistant spheres spanning
    # (0,1,0) to (1,1,0).
    for i in range(11, 22):
        # Compute and test expected sphere position.
        expected_sphere_position = np.array([0.1 * (i - 11), 1.0, 0.0])
        assert expected_sphere_position == pytest.approx(world_collision_sphere_positions[i])
        # NOTE: There are no self-collision spheres for the second link.

        check_distance(inspector, collision_position,
                       obstacle_handle, sphere_obstacle_pose.translation, sphere_obstacle_radius,
                       i, expected_sphere_position, sphere_radii)

    # Check distances for spheres in third link. Third link has 11 equidistant spheres spanning
    # (1,1,0) to (1,0,0).
    for i in range(11):
        # Compute and test expected sphere position.
        expected_sphere_position = np.array([1.0, 1.0 - 0.1 * i, 0.0])
        assert expected_sphere_position == pytest.approx(world_collision_sphere_positions[i])
        assert expected_sphere_position == pytest.approx(self_collision_sphere_positions[i])

        distance = check_distance(inspector, collision_position,
                                  obstacle_handle, sphere_obstacle_pose.translation,
                                  sphere_obstacle_radius,
                                  i, expected_sphere_position, sphere_radii)

        # Expect final 6 spheres to be in collision.
        if i >= 5:
            assert 0.0 >= distance

    # Set a c-space position expected to NOT be in collision with obstacle.
    collision_position = np.array([0.5 * math.pi, 0.0, 0.0])
    assert inspector.in_collision_with_obstacle(collision_position) is False

    # Get world-collision sphere positions from inspector.
    world_collision_sphere_positions = (
        inspector.world_collision_sphere_positions(collision_position))

    # Get self-collision sphere positions from inspector.
    self_collision_sphere_positions = (
        inspector.self_collision_sphere_positions(collision_position))

    # Check distances for spheres, where all spheres are arranged along (0,0,0) to (0,3,0).
    for i in range(33):
        y_pos = 0.1 * (i - 22) if (i > 21) else 1.0 + 0.1 * (i - 11) if (i > 10) else 2.0 + 0.1 * i

        # Compute and test expected sphere position.
        expected_sphere_position = np.array([0.0, y_pos, 0.0])
        assert expected_sphere_position == pytest.approx(world_collision_sphere_positions[i])
        # NOTE: There are only 22 self-collision spheres, with positions corresponding to
        #       world-collision sphere indices [0, 11) and [22, 33).
        if i >= 22:
            assert expected_sphere_position == pytest.approx(
                self_collision_sphere_positions[i - 11])
        elif i < 11:
            assert expected_sphere_position == pytest.approx(self_collision_sphere_positions[i])

        check_distance(inspector, collision_position,
                       obstacle_handle, sphere_obstacle_pose.translation, sphere_obstacle_radius,
                       i, expected_sphere_position, sphere_radii)


def test_three_link_arm_self_collision(configure_three_link_arm_robot_world_inspector_test):
    """Test `RobotWorldInspector` self-collision detection for the three-link arm."""
    # Init robot_description and world from test fixture.
    robot_description, world = configure_three_link_arm_robot_world_inspector_test()

    # Create `RobotWorldInspector`.
    inspector_with_world_view = cumotion.create_robot_world_inspector(robot_description,
                                                                      world.add_world_view())
    inspector_without_world_view = cumotion.create_robot_world_inspector(robot_description, None)

    # Expect the outstretched three-link arm to not collide with itself.
    q0 = np.zeros(3)
    assert not inspector_with_world_view.in_self_collision(q0)
    assert len(inspector_with_world_view.frames_in_self_collision(q0)) == 0
    assert not inspector_without_world_view.in_self_collision(q0)
    assert len(inspector_without_world_view.frames_in_self_collision(q0)) == 0

    # Expect the three-link arm to collide with itself when configured such that the first and
    # third links intersect.
    q1 = np.array([0.0, 0.75 * np.pi, 0.75 * np.pi])
    assert inspector_with_world_view.in_self_collision(q1)
    collisions = inspector_with_world_view.frames_in_self_collision(q1)
    assert inspector_without_world_view.in_self_collision(q1)
    assert inspector_without_world_view.frames_in_self_collision(q1) == collisions

    assert len(collisions) == 1
    # NOTE: The order of "link2" and "link0" is arbitrary and may change if implementation changes.
    #       This test is checking that a single pair of frames collide for the given configuration
    #       and self-collision mask.
    assert collisions[0][0] == "link2"
    assert collisions[0][1] == "link0"


def test_three_link_arm_world_management(configure_three_link_arm_robot_world_inspector_test):
    """Test `RobotWorldInspector` world management."""
    # Init robot_description and world from test fixture.
    robot_description, world = configure_three_link_arm_robot_world_inspector_test()

    # Add spherical obstacle in front of robot arm.
    sphere_obstacle = cumotion.create_obstacle(cumotion.Obstacle.Type.SPHERE)
    sphere_obstacle_radius = 0.5
    sphere_obstacle.set_attribute(cumotion.Obstacle.Attribute.RADIUS, sphere_obstacle_radius)
    sphere_obstacle_pose = cumotion.Pose3.from_translation(np.array([1.0, 0.0, 0.0]))
    sphere_handle = world.add_obstacle(sphere_obstacle, sphere_obstacle_pose)
    world_view = world.add_world_view()

    # Create `RobotWorldInspector`.
    inspector_1 = cumotion.create_robot_world_inspector(robot_description, None)
    inspector_2 = cumotion.create_robot_world_inspector(robot_description, world_view)

    # Set a c-space position expected to be in collision with obstacle.
    collision_position = np.array([0.5 * math.pi, -0.5 * math.pi, -0.5 * math.pi])
    assert not inspector_1.in_collision_with_obstacle(collision_position)
    assert inspector_2.in_collision_with_obstacle(collision_position)

    inspector_2.clear_world_view()
    assert not inspector_2.in_collision_with_obstacle(collision_position)

    inspector_1.set_world_view(world_view)
    assert inspector_1.in_collision_with_obstacle(collision_position)

    world.remove_obstacle(sphere_handle)
    world_view.update()
    assert not inspector_1.in_collision_with_obstacle(collision_position)


def test_min_distance_to_obstacle(configure_three_link_arm_robot_world_inspector_test):
    """Test `RobotWorldInspector.min_distance_to_obstacle()`."""
    # Init robot_description and world from test fixture.
    robot_description, world = configure_three_link_arm_robot_world_inspector_test()

    # Add a two spherical obstacles at (0, 1, 0) and (-0.8, 0, 0).
    obstacle_handles = []
    sphere_radii = 0.1
    sphere_obstacle_1 = cumotion.create_obstacle(cumotion.Obstacle.Type.SPHERE)
    sphere_obstacle_1.set_attribute(cumotion.Obstacle.Attribute.RADIUS, sphere_radii)
    sphere_obstacle_1_pose = cumotion.Pose3.from_translation(np.array([0.0, 1.0, 0.0]))
    obstacle_handles.append(world.add_obstacle(sphere_obstacle_1, sphere_obstacle_1_pose))

    sphere_obstacle_2 = cumotion.create_obstacle(cumotion.Obstacle.Type.SPHERE)
    sphere_obstacle_2.set_attribute(cumotion.Obstacle.Attribute.RADIUS, sphere_radii)
    sphere_obstacle_2_pose = cumotion.Pose3.from_translation(np.array([-0.8, 0.0, 0.0]))
    obstacle_handles.append(world.add_obstacle(sphere_obstacle_2, sphere_obstacle_2_pose))

    robot_position = np.array([0.0, 0.0, 0.0])

    # Create `RobotWorldInspector`.
    inspector = cumotion.create_robot_world_inspector(robot_description, world.add_world_view())

    # Compute the distance to the obstacle from the nearest sphere.
    expected_min_distance_to_obstacle = float('inf')
    for sphere_index in range(inspector.num_world_collision_spheres()):
        for obstacle_handle in obstacle_handles:
            distance = inspector.distance_to_obstacle(obstacle_handle, sphere_index, robot_position)
            if distance < expected_min_distance_to_obstacle:
                expected_min_distance_to_obstacle = distance

    min_distance_to_obstacle = inspector.min_distance_to_obstacle(robot_position)
    assert min_distance_to_obstacle == expected_min_distance_to_obstacle
