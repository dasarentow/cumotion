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

"""Unit tests for pose3_python.h."""

# Third Party
import numpy as np
import pytest

# cuMotion
import cumotion


def test_rotation3():
    """Test the cumotion::Pose3 Python wrapper."""
    # Homogeneous transform matrix constructor
    mat = np.array([
        [-1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 2.0],
        [0.0, 0.0, -1.0, 3.0],
        [0.0, 0.0, 0.0, 1.0]])
    pose = cumotion.Pose3(mat)
    assert 0 == pytest.approx(pose.rotation.w())
    assert 0 == pytest.approx(pose.rotation.x())
    assert 1 == pytest.approx(pose.rotation.y())
    assert 0 == pytest.approx(pose.rotation.z())
    assert np.array([1, 2, 3]) == pytest.approx(pose.translation)

    # Rotation and translation constructor
    pose = cumotion.Pose3(cumotion.Rotation3(1, 2, 3, 4), np.array([5, 6, 7]))
    assert 1 / np.sqrt(30) == pytest.approx(pose.rotation.w())
    assert 2 / np.sqrt(30) == pytest.approx(pose.rotation.x())
    assert 3 / np.sqrt(30) == pytest.approx(pose.rotation.y())
    assert 4 / np.sqrt(30) == pytest.approx(pose.rotation.z())
    assert np.array([5, 6, 7]) == pytest.approx(pose.translation)

    # From rotation
    pose = cumotion.Pose3.from_rotation(cumotion.Rotation3(1, 2, 3, 4))
    assert 1 / np.sqrt(30) == pytest.approx(pose.rotation.w())
    assert 2 / np.sqrt(30) == pytest.approx(pose.rotation.x())
    assert 3 / np.sqrt(30) == pytest.approx(pose.rotation.y())
    assert 4 / np.sqrt(30) == pytest.approx(pose.rotation.z())
    assert np.array([0, 0, 0]) == pytest.approx(pose.translation)

    # From translation
    pose = cumotion.Pose3.from_translation(np.array([5, 6, 7]))
    assert 1 == pytest.approx(pose.rotation.w())
    assert 0 == pytest.approx(pose.rotation.x())
    assert 0 == pytest.approx(pose.rotation.y())
    assert 0 == pytest.approx(pose.rotation.z())
    assert np.array([5, 6, 7]) == pytest.approx(pose.translation)

    # Identity
    pose = cumotion.Pose3.identity()
    assert 1 == pytest.approx(pose.rotation.w())
    assert 0 == pytest.approx(pose.rotation.x())
    assert 0 == pytest.approx(pose.rotation.y())
    assert 0 == pytest.approx(pose.rotation.z())
    assert np.array([0, 0, 0]) == pytest.approx(pose.translation)

    # Matrix
    pose = cumotion.Pose3(cumotion.Rotation3.from_axis_angle(np.array([0, 0, 1]), np.pi / 2),
                          np.array([1, 2, 3]))
    assert np.array([
        [0, -1, 0, 1],
        [1, 0, 0, 2],
        [0, 0, 1, 3],
        [0, 0, 0, 1]]) == pytest.approx(pose.matrix())

    # Inverse and pose multiplication overload
    w = 0.73
    x = 0.183
    y = 0.365
    z = 0.548
    rot = cumotion.Rotation3(w, x, y, z)
    vec = np.random.uniform(0, 1, 3)
    pose = cumotion.Pose3(rot, vec)
    assert np.identity(4) == pytest.approx((pose * pose.inverse()).matrix())
    assert np.identity(4) == pytest.approx((pose.inverse() * pose).matrix())

    # Vector multiplication overload
    rot = cumotion.Rotation3.from_axis_angle(np.array([0, 0, 1]), np.pi / 2)
    vec = np.array([1, 0, 0])
    pose = cumotion.Pose3(rot, vec)
    vec1 = np.array([1, 0, 0])
    assert np.array([1, 1, 0]) == pytest.approx(pose * vec1)

    # String representation
    pose = cumotion.Pose3.identity()
    expected_repr = ("cumotion.Pose3(rotation=cumotion.Rotation3(w=1, x=0, y=0, z=0), "
                     "translation=np.array([0, 0, 0]))")
    assert expected_repr == repr(pose)
