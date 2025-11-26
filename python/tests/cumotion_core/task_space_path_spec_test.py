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

"""Unit tests for task_space_path_spec_python.h."""

# Third Party
import math
import numpy as np
import pytest

# cuMotion
import cumotion

# Local directory
from ._test_helper import errors_disabled


def test_task_space_path_spec_all_path_types():
    """Test `TaskSpacePathSpec` with all supported path types."""
    # Create task space path specification.
    spec = cumotion.create_task_space_path_spec(cumotion.Pose3.identity())

    # Add linear path.
    R0 = cumotion.Rotation3.from_axis_angle(np.array([0.0, 0.0, 1.0]), math.pi)
    assert spec.add_linear_path(cumotion.Pose3(R0, np.array([0.5, 0.0, 0.0])))

    # Add translation.
    assert spec.add_translation(np.array([0.5, 1.0, 0.0]))

    # Add rotation.
    assert spec.add_rotation(cumotion.Rotation3.from_axis_angle(np.array([0.0, 1.0, 0.0]), math.pi))

    # Add three-point arc with constant orientation.
    assert spec.add_three_point_arc(np.array([0.5, 2.0, 0.0]), np.array([1.0, 1.5, 0.0]), True)

    # Add three-point arc with tangent orientation.
    assert spec.add_three_point_arc(np.array([0.5, 3.0, 0.0]), np.array([0.0, 2.5, 0.0]), False)

    # Add three-point arc with orientation target.
    assert spec.add_three_point_arc_with_orientation_target(
        cumotion.Pose3(R0, np.array([0.5, 4.0, 0.0])), np.array([0.0, 3.5, 0.0]))

    # Add tangent arc with constant orientation.
    assert spec.add_tangent_arc(np.array([2.0, 2.0, 0.0]), True)

    # Add tangent arc with tangent orientation.
    assert spec.add_tangent_arc(np.array([5.0, 1.0, 0.0]), False)

    # Add tangent arc with orientation target.
    assert spec.add_tangent_arc_with_orientation_target(
        cumotion.Pose3(R0, np.array([6.0, 2.0, 0.0])))

    # Generate path.
    path = spec.generate_path()

    # Expect path domain is as expected.
    assert 0.0 == path.domain().lower
    assert 19.026766797874675 == path.domain().upper

    # Expect path length is as expected.
    assert 15.41879210706054 == path.path_length()

    # Expect accumulated rotation is as expected.
    assert 18.6977301407855 == path.accumulated_rotation()

    assert cumotion.Pose3.identity().matrix() == pytest.approx(path.eval(0.0).matrix())

    expected_pose_10 = np.array([[-1.0, 0.0, 0.0, 0.782041],
                                 [0.0, -1.0, 0.0, 3.97433],
                                 [0.0, 0.0, 1.0, 0.0],
                                 [0.0, 0.0, 0.0, 1.0]])
    assert expected_pose_10 == pytest.approx(path.eval(10.0).matrix(), 5e-6)

    expected_pose_15 = np.array([[-0.0704836, 0.997513, 0.0, 2.96999],
                                 [-0.997513, -0.0704836, 0.0, 0.0729794],
                                 [0.0, 0.0, 1.0, 0.0],
                                 [0.0, 0.0, 0.0, 1.0]])
    assert expected_pose_15 == pytest.approx(path.eval(15.0).matrix(), 3e-6)


def test_task_space_path_spec_invalid_tangent_arc():
    """Test `TaskSpacePathSpec` with invalid tangent arc specifications."""
    # Create task space path specification.
    spec = cumotion.create_task_space_path_spec(cumotion.Pose3.identity())

    # Expect that no type of tangent arc can be added to an empty `TaskSpacePathSpec`.
    with errors_disabled:
        assert not spec.add_tangent_arc(np.array([2.0, 2.0, 0.0]), True)
        assert not spec.add_tangent_arc(np.array([5.0, 1.0, 0.0]), False)
        assert not spec.add_tangent_arc_with_orientation_target(
            cumotion.Pose3.from_translation(np.array([6.0, 2.0, 0.0])))

    # Add a pure rotation.
    assert spec.add_rotation(cumotion.Rotation3.from_axis_angle(np.array([0.0, 1.0, 0.0]), math.pi))

    # Expect that no type of tangent arc can be added to a `TaskSpaceSpec` with only rotations.
    with errors_disabled:
        assert not spec.add_tangent_arc(np.array([2.0, 2.0, 0.0]), True)
        assert not spec.add_tangent_arc(np.array([5.0, 1.0, 0.0]), False)
        assert not spec.add_tangent_arc_with_orientation_target(
            cumotion.Pose3.from_translation(np.array([6.0, 2.0, 0.0])))

    # Add a pure translation.
    assert spec.add_translation(np.array([0.0, 2.0, 0.0]))

    # Expect that all types of tangent arc can be added to a path that contains translation.
    # Expect that no type of tangent arc can be added to a `TaskSpaceSpec` with only rotations.
    assert spec.add_tangent_arc(np.array([2.0, 2.0, 0.0]), True)
    assert spec.add_tangent_arc(np.array([5.0, 1.0, 0.0]), False)
    assert spec.add_tangent_arc_with_orientation_target(
        cumotion.Pose3.from_translation(np.array([6.0, 2.0, 0.0])))

    # Generate path.
    path = spec.generate_path()

    # Expect path domain is as expected.
    assert 0.0 == path.domain().lower
    assert 13.970097117057179 == path.domain().upper

    assert cumotion.Pose3.identity().matrix() == pytest.approx(path.eval(0.0).matrix())

    expected_pose_6 = np.array([[-1.0, 0.0, 0.0, 0.346356],
                                [0.0, 1.0, 0.0, 2.7568],
                                [0.0, 0.0, -1.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0]])
    assert expected_pose_6 == pytest.approx(path.eval(6.0).matrix(), 3e-6)

    expected_pose_12 = np.array([[0.604544, -0.796572, 0.0, 4.67424],
                                 [-0.796572, -0.604544, 0.0, 0.672381],
                                 [0.0, 0.0, -1.0, 0.0],
                                 [0.0, 0.0, 0.0, 1.0]])
    assert expected_pose_12 == pytest.approx(path.eval(12.0).matrix(), 1e-6)
