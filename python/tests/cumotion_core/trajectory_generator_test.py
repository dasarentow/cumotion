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

"""Unit tests for trajectory_generator_python.h."""

# Standard Library
import os

# Third Party
import numpy as np
import pytest

# cuMotion
import cumotion

# Local directory
from ._test_helper import CUMOTION_ROOT_DIR, warnings_disabled


def test_cspace_trajectory_generator():
    """Test trajectory generation for 2-DoF system."""
    # Create trajectory generator.
    trajectory_generator = cumotion.create_cspace_trajectory_generator(2)
    assert 2 == trajectory_generator.num_cspace_coords()

    # Set limits.
    trajectory_generator.set_position_limits(np.array([-4.0, -4.0]), np.array([4.0, 4.0]))
    velocity_limits = np.array([10.0, 10.0])
    trajectory_generator.set_velocity_limits(velocity_limits)
    acceleration_limits = np.array([20.0, 40.0])
    trajectory_generator.set_acceleration_limits(acceleration_limits)
    jerk_limits = np.array([1000.0, 200.0])
    trajectory_generator.set_jerk_limits(jerk_limits)

    # Set position waypoints.
    waypoints = [np.zeros(2), np.array([-2.0, 2.0]), np.array([2.0, -2.0]), np.zeros(2)]

    # Attempt to generate a trajectory and expect a valid trajectory to be generated.
    trajectory = trajectory_generator.generate_trajectory(waypoints)
    assert trajectory is not None

    # Expect that one element of maximum acceleration magnitude (nearly) saturates the acceleration
    # limit.
    assert acceleration_limits[0] == pytest.approx(trajectory.max_acceleration_magnitude()[0], 0.02)

    # Test that derivative limits are as expected.
    eps = 1e-4
    assert np.array([5.1882, 5.1882]) == pytest.approx(trajectory.max_velocity_magnitude(), eps)
    assert np.array([19.9863, 19.9863]) == pytest.approx(trajectory.max_acceleration_magnitude(),
                                                         eps)
    assert np.array([76.5785, 76.5785]) == pytest.approx(trajectory.max_jerk_magnitude(), eps)

    # // Test that position limits are as expected.
    assert np.array([-2.08647, -2.08647]) == pytest.approx(trajectory.min_position(), eps)
    assert np.array([2.08647, 2.08647]) == pytest.approx(trajectory.max_position(), eps)

    # Test that domain is as expected.
    assert 0.0 == trajectory.domain().lower
    assert 2.9087247007150596 == pytest.approx(trajectory.domain().upper, 1e-9)
    assert trajectory.domain().upper == trajectory.domain().span()

    # Test equivalence of `eval()` and `eval_all()`.
    t = 1.5  # Arbitrary time value
    pos_a, vel_a, accel_a, jerk_a = trajectory.eval_all(t)
    pos_b = trajectory.eval(t)
    pos_c = trajectory.eval(t, 0)
    vel_b = trajectory.eval(t, 1)
    accel_b = trajectory.eval(t, 2)
    jerk_b = trajectory.eval(t, 3)
    assert pos_a == pytest.approx(pos_b)
    assert pos_b == pytest.approx(pos_c)
    assert vel_a == pytest.approx(vel_b)
    assert accel_a == pytest.approx(accel_b)
    assert jerk_a == pytest.approx(jerk_b)

    # Test that if position limits are reduced s.t. bounds and waypoints are within limits, but the
    # computed path exceeds limits, then trajectory generation will fail and `None` will be
    # returned.
    trajectory_generator.set_position_limits(np.array([-2.05, -2.05]), np.array([2.05, 2.05]))
    with warnings_disabled:
        null_trajectory = trajectory_generator.generate_trajectory(waypoints)
    assert null_trajectory is None

    # NOTE: This behaviour is different from setting position limits s.t. waypoints are out of
    # bounds. In this case a fatal error will occur due to invalid problem formulation.
    trajectory_generator.set_position_limits(np.array([-1.95, -1.95]), np.array([1.95, 1.95]))
    with pytest.raises(Exception):
        trajectory_generator.generate_trajectory(waypoints)


@pytest.fixture
def configure_franka_robot_description():
    """Test fixture to configure Franka robot description object."""
    def _configure_franka_robot_description():
        # Set directory for robot description YAML files.
        config_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'nvidia', 'shared')

        # Set absolute path to the XRDF for Franka robot.
        xrdf_path = os.path.join(config_path, 'franka.xrdf')

        # Set absolute path to URDF for Franka robot.
        urdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'third_party', 'franka',
                                 'franka.urdf')

        # Load and return robot description.
        return cumotion.load_robot_from_file(xrdf_path, urdf_path)

    return _configure_franka_robot_description


def test_franka_trajectory_generation(configure_franka_robot_description):
    """Test trajectory generation for Franka."""
    # Init kinematics from test fixture
    kinematics = configure_franka_robot_description().kinematics()
    assert 7 == kinematics.num_cspace_coords()

    # Create trajectory generator.
    trajectory_generator = cumotion.create_cspace_trajectory_generator(kinematics)
    assert 7 == trajectory_generator.num_cspace_coords()

    # Set initial and final state.
    waypoints = [np.array([0.0, 0.0, 0.0, -0.1, 0.0, 0.0, 0.0]),
                 np.array([1.0, 1.2, -2.1, -2.5, 1.3, 2.4, -0.5])]

    # Attempt to generate a trajectory and expect a valid trajectory to be generated.
    trajectory = trajectory_generator.generate_trajectory(waypoints)
    assert trajectory is not None

    # Check domain is as expected.
    assert 0.0 == trajectory.domain().lower
    assert 1.7946536358179246 == pytest.approx(trajectory.domain().upper, 1e-9)

    # Set times for interpolation
    times = [0.0, 3.0]

    # Expect interpolation with cubic splines to succeed.
    cubic = cumotion.CSpaceTrajectoryGenerator.InterpolationMode.CUBIC_SPLINE
    cubic_trajectory = trajectory_generator.generate_time_stamped_trajectory(waypoints, times,
                                                                             cubic)
    assert cubic_trajectory is not None

    # Check domain is as expected.
    assert 0.0 == cubic_trajectory.domain().lower
    assert 3.0 == cubic_trajectory.domain().upper

    # Expect linear interpolation to succeed.
    linear = cumotion.CSpaceTrajectoryGenerator.InterpolationMode.LINEAR
    linear_trajectory = trajectory_generator.generate_time_stamped_trajectory(waypoints, times,
                                                                              linear)
    assert linear_trajectory is not None

    # Check domain is as expected.
    assert 0.0 == linear_trajectory.domain().lower
    assert 3.0 == linear_trajectory.domain().upper

    # Expect default interpolation to succeed.
    default_trajectory = trajectory_generator.generate_time_stamped_trajectory(waypoints, times)
    assert default_trajectory is not None

    # Check domain is as expected.
    assert 0.0 == default_trajectory.domain().lower
    assert 3.0 == default_trajectory.domain().upper


def test_cspace_trajectory_generator_set_solver_param():
    """Test setting solver params for trajectory generation."""
    # Create trajectory generator.
    trajectory_generator = cumotion.create_cspace_trajectory_generator(2)
    assert 2 == trajectory_generator.num_cspace_coords()

    # Test setting valid value for `max_segment_iterations`.
    assert trajectory_generator.set_solver_param("max_segment_iterations", 34)
    assert trajectory_generator.set_solver_param("max_segment_iterations", 0)
    # Test that setting a negative value (i.e., an invalid value) will not update the
    # `max_segment_iterations`.
    with warnings_disabled:
        assert not trajectory_generator.set_solver_param("max_segment_iterations", -3)
    # Test that the wrong value type (e.g., floating point or string) will throw an error.
    with pytest.raises(Exception):
        trajectory_generator.set_solver_param("max_segment_iterations", 23.0)
    with pytest.raises(Exception):
        trajectory_generator.set_solver_param("max_segment_iterations", "twelve")

    # Test setting valid value for `max_aggregate_iterations`.
    assert trajectory_generator.set_solver_param("max_aggregate_iterations", 34)
    assert trajectory_generator.set_solver_param("max_aggregate_iterations", 0)
    # Test that setting a negative value (i.e., an invalid value) will not update the
    # `max_aggregate_iterations`.
    with warnings_disabled:
        assert not trajectory_generator.set_solver_param("max_aggregate_iterations", -3)
    # Test that the wrong value type (e.g., floating point or string) will throw an error.
    with pytest.raises(Exception):
        trajectory_generator.set_solver_param("max_aggregate_iterations", 23.0)
    with pytest.raises(Exception):
        trajectory_generator.set_solver_param("max_aggregate_iterations", "twelve")

    # Test setting valid value for `convergence_dt`.
    assert trajectory_generator.set_solver_param("convergence_dt", 0.5)
    # Test that setting a negative or zero value (i.e., an invalid value) will not update the
    # `convergence_dt`.
    with warnings_disabled:
        assert not trajectory_generator.set_solver_param("convergence_dt", -0.01)
        assert not trajectory_generator.set_solver_param("convergence_dt", 0.0)
    # Test that the wrong value type (e.g., integer or string) will throw an error.
    with pytest.raises(Exception):
        trajectory_generator.set_solver_param("convergence_dt", 12)
    with pytest.raises(Exception):
        trajectory_generator.set_solver_param("convergence_dt", "twelve")

    # Test setting valid value for `max_dilation_iterations`.
    assert trajectory_generator.set_solver_param("max_dilation_iterations", 34)
    assert trajectory_generator.set_solver_param("max_dilation_iterations", 0)
    # Test that setting a negative value (i.e., an invalid value) will not update the
    # `max_dilation_iterations`.
    with warnings_disabled:
        assert not trajectory_generator.set_solver_param("max_dilation_iterations", -3)
    # Test that the wrong value type (e.g., floating point or string) will throw an error.
    with pytest.raises(Exception):
        trajectory_generator.set_solver_param("max_dilation_iterations", 23.0)
    with pytest.raises(Exception):
        trajectory_generator.set_solver_param("max_dilation_iterations", "twelve")

    # Test setting valid value for `dilation_dt`.
    assert trajectory_generator.set_solver_param("dilation_dt", 0.5)
    # Test that setting a negative or zero value (i.e., an invalid value) will not update the
    # `dilation_dt`.
    with warnings_disabled:
        assert not trajectory_generator.set_solver_param("dilation_dt", -0.01)
        assert not trajectory_generator.set_solver_param("dilation_dt", 0.0)
    # Test that the wrong value type (e.g., integer or string) will throw an error.
    with pytest.raises(Exception):
        trajectory_generator.set_solver_param("dilation_dt", 12)
    with pytest.raises(Exception):
        trajectory_generator.set_solver_param("dilation_dt", "twelve")

    # Test setting valid value for `min_time_span`.
    assert trajectory_generator.set_solver_param("min_time_span", 0.5)
    # Test that setting a negative or zero value (i.e., an invalid value) will not update
    # `min_time_span`.
    with warnings_disabled:
        assert not trajectory_generator.set_solver_param("min_time_span", -0.01)
        assert not trajectory_generator.set_solver_param("min_time_span", 0.0)
    # Test that the wrong value type (e.g., integer or string) will throw an error.
    with pytest.raises(Exception):
        trajectory_generator.set_solver_param("min_time_span", 12)
    with pytest.raises(Exception):
        trajectory_generator.set_solver_param("min_time_span", "twelve")

    # Test setting valid values for `time_split_method`.
    assert trajectory_generator.set_solver_param("time_split_method", "uniform")
    assert trajectory_generator.set_solver_param("time_split_method", "chord_length")
    assert trajectory_generator.set_solver_param("time_split_method", "centripetal")
    # Test that setting an invalid method will not update the `time_split_method`.
    with warnings_disabled:
        assert not trajectory_generator.set_solver_param("time_split_method", "every_span_is_five")
    # Test that the wrong value type (e.g., integer or floating point) will throw an error.
    with pytest.raises(Exception):
        trajectory_generator.set_solver_param("time_split_method", 12)
    with pytest.raises(Exception):
        trajectory_generator.set_solver_param("time_split_method", 12.0)
