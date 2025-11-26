# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
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

"""Unit tests for path_conversion_python.h."""

# Standard Library
import os

# Third Party
import math
import numpy as np
import pytest

# cuMotion
import cumotion

# Local Folder
from ._test_helper import CUMOTION_ROOT_DIR


def test_task_space_path_conversion_config():
    """Test `TaskSpacePathConversionConfig`."""
    # Create task space path conversion config.
    config = cumotion.TaskSpacePathConversionConfig()

    assert config.min_position_deviation == 0.001
    config.min_position_deviation = 0.002
    assert config.min_position_deviation == 0.002

    assert config.max_position_deviation == 0.003
    config.max_position_deviation = 0.006
    assert config.max_position_deviation == 0.006

    assert config.initial_s_step_size == 0.05
    config.initial_s_step_size = 0.1
    assert config.initial_s_step_size == 0.1

    assert config.initial_s_step_size_delta == 0.005
    config.initial_s_step_size_delta = 0.01
    assert config.initial_s_step_size_delta == 0.01

    assert config.min_s_step_size == 1e-5
    config.min_s_step_size = 0.001
    assert config.min_s_step_size == 0.001

    assert config.min_s_step_size_delta == 1e-5
    config.min_s_step_size_delta = 0.0001
    assert config.min_s_step_size_delta == 0.0001

    assert config.max_iterations == 50
    config.max_iterations = 20
    assert config.max_iterations == 20

    assert config.alpha == 1.4
    config.alpha = 12.3
    assert config.alpha == 12.3


def approximate_task_space_path(cspace_trajectory, kinematics, control_frame, num_segments=500):
    """Convert `cspace_trajectory` to an approximate task space path.

    The `control_frame` is mapped from c-space to task space using `kinematics` with `num_segments`
    linear segments.
    """
    # Compute initial pose and generate tasks space path builder.
    t0 = cspace_trajectory.domain().lower
    q0 = cspace_trajectory.eval(t0)
    pose0 = kinematics.pose(q0, control_frame)
    path_spec = cumotion.create_task_space_path_spec(pose0)

    # Generate linear path specifications.
    for t in np.linspace(cspace_trajectory.domain().lower, cspace_trajectory.domain().upper,
                         num_segments)[1:]:
        # Compute pose.
        q = cspace_trajectory.eval(t)
        pose = kinematics.pose(q, control_frame)

        # Add linear path segment.
        path_spec.add_linear_path(pose)

    # Generate and return task space path.
    return path_spec.generate_path()


def compute_max_position_deviation_from_circular_path(task_space_path, center, radius):
    """Compute the maximum position deviation between `task_space_path` and a circle.

    The circle is assumed to be on the YZ-plane with `center` and `radius`.
    """
    # Number of points sampled uniformly from `task_space_path` domain to approximate maximum
    # position deviation.
    num_test_points = 1000

    max_sq_deviation = 0.0

    for s in np.linspace(task_space_path.domain().lower, task_space_path.domain().upper,
                         num_test_points):
        pose = task_space_path.eval(s)
        # Compute squared distance from `pose` to circle.
        in_plane_dist = np.linalg.norm(pose.translation[1:] - center[1:]) - radius
        out_of_plane_dist = pose.translation[0] - center[0]
        sq_distance = (in_plane_dist * in_plane_dist) + (out_of_plane_dist * out_of_plane_dist)

        # Update maximum squared deviation.
        max_sq_deviation = max(max_sq_deviation, sq_distance)

    return math.sqrt(max_sq_deviation)


def test_trajectory_from_task_space_path_spec():
    """Test `convert_task_space_path_spec_to_cspace()` with Franka."""
    # Load Franka kinematics and set control frame.
    config_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'nvidia', 'shared')
    xrdf_path = os.path.join(config_path, 'franka.xrdf')
    urdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'third_party', 'franka', 'franka.urdf')
    robot_description = cumotion.load_robot_from_file(xrdf_path, urdf_path)
    kinematics = robot_description.kinematics()
    control_frame = "right_gripper"

    # Use `kinematics` to create c-space trajectory generator.
    cspace_trajectory_generator = cumotion.create_cspace_trajectory_generator(kinematics)

    # Use default parameters for task space trajectory generation.
    path_conversion_config = cumotion.TaskSpacePathConversionConfig()

    # Create simple circular path.
    q0 = robot_description.default_cspace_configuration()
    pose0 = kinematics.pose(q0, control_frame)
    path_spec = cumotion.create_task_space_path_spec(pose0)
    radius = 0.2
    center = pose0.translation + np.array([0.0, 0.0, radius])
    x1 = center + np.array([0.0, radius, 0.0])
    x2 = center + np.array([0.0, 0.0, radius])
    path_spec.add_three_point_arc(x2, x1)
    path_spec.add_tangent_arc(pose0.translation)

    # Set a series of position deviation limits for testing.
    position_deviation_limits = [[0.001, 0.003], [0.005, 0.01]]

    # For each set of position deviation limits, set the expected number of c-space waypoints that
    # will be generated.
    expected_nums_cspace_waypoints = [21, 12]

    # For each set of position deviation limits, set the expected position deviation of the final
    # c-space trajectory.
    # NOTE: the `position_deviation_limits` do *NOT* serve as a strict upper bound due to the spline
    #       interpolation in the `cspace_trajectory_generator` not adhering to straight lines in
    #       c-space.
    expected_max_position_deviations = [0.0036, 0.0023]

    # For each set of position deviation limits, set the expected time span of the final c-space
    # trajectory.
    # In general, the time span is expected to be higher for tighter tolerances.
    expected_trajectory_time_spans = [2.4, 2.22]

    # Test trajectory generation for each set of position deviation limits.
    for limits, expected_num_cspace_waypoints, expected_max_position_deviation, \
        expected_trajectory_time_span \
            in zip(position_deviation_limits,
                   expected_nums_cspace_waypoints,
                   expected_max_position_deviations,
                   expected_trajectory_time_spans):
        # Set position deviation limits.
        path_conversion_config.min_position_deviation = limits[0]
        path_conversion_config.max_position_deviation = limits[1]

        # Generate c-space path form `TaskSpacePathSpec`.
        cspace_path = cumotion.convert_task_space_path_spec_to_cspace(path_spec,
                                                                      kinematics,
                                                                      control_frame,
                                                                      path_conversion_config)

        # Test that the number of waypoints is as expected.
        assert expected_num_cspace_waypoints == len(cspace_path.waypoints())

        # Generate task space trajectory, expecting success.
        trajectory = cspace_trajectory_generator.generate_trajectory(cspace_path.waypoints())
        assert trajectory is not None

        # Test that maximum position deviation is as expected.
        approx_path = approximate_task_space_path(trajectory, kinematics, control_frame)
        assert expected_max_position_deviation == \
            pytest.approx(compute_max_position_deviation_from_circular_path(
                approx_path, center, radius), abs=1e-4)

        # Test that time span is as expected.
        assert expected_trajectory_time_span == pytest.approx(trajectory.domain().span(), abs=5e-2)


def create_rectangular_task_space_path_spec(x0):
    """Create a rectangular path in the YZ plane from the initial pose `x0`."""
    path_spec = cumotion.create_task_space_path_spec(x0)
    width = 0.3
    height = 0.1
    blend_radius = 0.02
    path_spec.add_translation(np.array([x0.translation[0],
                                        x0.translation[1],
                                        x0.translation[2] - height]), blend_radius)
    path_spec.add_translation(np.array([x0.translation[0],
                                        x0.translation[1] + width,
                                        x0.translation[2] - height]), blend_radius)
    path_spec.add_translation(np.array([x0.translation[0],
                                        x0.translation[1] + width,
                                        x0.translation[2]]), blend_radius)
    path_spec.add_translation(x0.translation)
    return path_spec


def test_trajectory_from_composite_path_spec():
    """Test `convert_composite_path_spec_to_cspace()` with Franka."""
    # Load Franka kinematics and set control frame.
    config_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'nvidia', 'shared')
    xrdf_path = os.path.join(config_path, 'franka.xrdf')
    urdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'third_party', 'franka', 'franka.urdf')
    robot_description = cumotion.load_robot_from_file(xrdf_path, urdf_path)
    kinematics = robot_description.kinematics()
    control_frame = "panda_rightfingertip"

    # Use `kinematics` to create c-space trajectory generator.
    cspace_trajectory_generator = cumotion.create_cspace_trajectory_generator(kinematics)

    # Create rectangular task space path spec with arbitrary initial pose.
    x0 = cumotion.Pose3(cumotion.Rotation3.from_axis_angle(np.array([0.0, 1.0, 0.0]), -math.pi),
                        np.array([0.4, 0.0, 0.3]))
    task_space_path_spec = create_rectangular_task_space_path_spec(x0)

    # Set transition modes for testing.
    transition_modes = [cumotion.CompositePathSpec.TransitionMode.FREE,
                        cumotion.CompositePathSpec.TransitionMode.LINEAR_TASK_SPACE,
                        cumotion.CompositePathSpec.TransitionMode.SKIP]

    # For each transition mode, set the expected number of c-space waypoints in the intermediate
    # `LinearCSpacePath`.
    expected_num_waypoints = [19, 21, 23]

    # Test adding `TaskSpacePathSpec` to `CompositePathSpec` with each transition mode.
    for i, transition_mode in enumerate(transition_modes):
        # Create `CompositePathSpec` from the default c-space position.
        composite_spec = cumotion.create_composite_path_spec(
            robot_description.default_cspace_configuration())

        # Add rectangular task space path specification.
        composite_spec.add_task_space_path_spec(task_space_path_spec, transition_mode)

        # Generate c-space path form `CompositePathSpec`.
        cspace_path = cumotion.convert_composite_path_spec_to_cspace(composite_spec,
                                                                     kinematics,
                                                                     control_frame)

        # Check that the number of c-space waypoints is as expected.
        assert expected_num_waypoints[i] == len(cspace_path.waypoints())

        # Expect that the first waypoint is at the default c-space configuration.
        assert robot_description.default_cspace_configuration() == \
               pytest.approx(cspace_path.waypoints()[0])

        # Expect that the final waypoint corresponds to the initial task space position.
        assert x0.translation == \
               pytest.approx(kinematics.position(cspace_path.waypoints()[-1], control_frame),
                             abs=2e-6)

        if transition_mode != cumotion.CompositePathSpec.TransitionMode.SKIP:
            # For the moves that do *NOT* skip the initial position of the task space path, expect
            # the orientation to match `x0` as well.
            assert x0.rotation.matrix() == \
                   pytest.approx(kinematics.orientation(cspace_path.waypoints()[-1],
                                                        control_frame).matrix(), abs=0.003)
        else:
            # For the case where the initial pose is skipped, the rectangle will translate from the
            # robot's default c-space configuration.
            assert kinematics.orientation(robot_description.default_cspace_configuration(),
                                          control_frame).matrix() == \
                   pytest.approx(kinematics.orientation(cspace_path.waypoints()[-1],
                                                        control_frame).matrix(), abs=0.003)

        # Generate task space trajectory, expecting success.
        trajectory = cspace_trajectory_generator.generate_trajectory(cspace_path.waypoints())
        assert trajectory is not None

        # NOTE: The resulting trajectory is not meaningfully tested without visual inspection. The
        #       corresponding c++ test in `path_conversion_test.cpp` can be used to generate such a
        #       visualization. Very loose time span checks are used to check that a
        #       "reasonable" trajectory has been generated. Significant divergence in time span
        #       (+/- 0.5 sec) is expected across different platforms due to divergence in IK
        #       solutions when converting from task-space specification to c-space waypoints. These
        #       differing c-space waypoints (with nearly equivalent task-space poses) produce
        #       different trajectories when the intervals between c-space waypoints are optimized
        #       to minimize time span.
        assert trajectory.domain().span() > 2.5
        assert trajectory.domain().span() < 3.5
