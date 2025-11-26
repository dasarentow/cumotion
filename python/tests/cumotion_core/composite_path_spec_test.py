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

"""Unit tests for composite_path_spec_python.h."""

# Third Party
import numpy as np
import pytest

# cuMotion
import cumotion

# Local directory
from ._test_helper import errors_disabled


def create_arbitrary_2d_cspace_path_spec():
    """Create an arbitrary `CSpacePathSpec` with 2 c-space coordinates."""
    spec = cumotion.create_cspace_path_spec(np.array([5.0, 7.0]))
    spec.add_cspace_waypoint(np.array([12.0, 7.0]))
    spec.add_cspace_waypoint(np.array([12.0, 25.0]))
    return spec


def create_arbitrary_3d_cspace_path_spec():
    """Create an arbitrary `CSpacePathSpec` with 3 c-space coordinates."""
    spec = cumotion.create_cspace_path_spec(np.array([3.0, -4.0, 0.0]))
    spec.add_cspace_waypoint(np.array([3.0, -4.0, 10.0]))
    spec.add_cspace_waypoint(np.array([3.0, 54.0, 10.0]))
    return spec


def create_arbitrary_task_space_path_spec():
    """Create an arbitrary `TaskSpacePathSpec`."""
    spec = cumotion.create_task_space_path_spec(cumotion.Pose3.identity())
    spec.add_translation(np.array([5.0, 0.0, 0.0]))
    spec.add_translation(np.array([5.0, 10.0, 0.0]))
    spec.add_translation(np.array([5.0, 10.0, 20.0]))
    return spec


def test_composite_path_spec():
    """Test `CompositePathSpec`."""
    # Create composite path specification.
    q0 = np.array([1.0, 2.0])
    composite_spec = cumotion.create_composite_path_spec(q0)

    # Expect the `composite_spec` to have 2 c-space coordinates.
    assert 2 == composite_spec.num_cspace_coords()

    # Expect `composite_spec` to have zero path specifications.
    assert 0 == composite_spec.num_path_specs()

    # Create a 3d `cspace_path_spec` and expect adding it to `composite_spec` to fail.
    cspace_3d_spec = create_arbitrary_3d_cspace_path_spec()
    with errors_disabled:
        assert not composite_spec.add_cspace_path_spec(
            cspace_3d_spec,
            cumotion.CompositePathSpec.TransitionMode.SKIP)

    # Create a 2d `cspace_path_spec` and expect adding it to `composite_spec` to succeed.
    cspace_2d_spec = create_arbitrary_2d_cspace_path_spec()
    assert composite_spec.add_cspace_path_spec(cspace_2d_spec,
                                               cumotion.CompositePathSpec.TransitionMode.SKIP)

    # Expect `composite_spec` to now have one path specification.
    assert 1 == composite_spec.num_path_specs()

    # Expect adding the same `cspace_2d_spec` with transition mode set to `LINEAR_TASK_SPACE` to
    # fail
    linear_mode = cumotion.CompositePathSpec.TransitionMode.LINEAR_TASK_SPACE
    with errors_disabled:
        assert not composite_spec.add_cspace_path_spec(cspace_2d_spec, linear_mode)

    # Expect `composite_spec` to still have only one path specification.
    assert 1 == composite_spec.num_path_specs()

    # Create a `task_space_path_spec` and expect adding it to `composite_spec` to succeed.
    task_space_spec = create_arbitrary_task_space_path_spec()
    assert composite_spec.add_task_space_path_spec(task_space_spec,
                                                   cumotion.CompositePathSpec.TransitionMode.SKIP)

    # Expect `composite_spec` to now have two path specifications.
    assert 2 == composite_spec.num_path_specs()

    # Expect the first path spec to be in c-space and the second in task space.
    assert cumotion.CompositePathSpec.PathSpecType.CSPACE == composite_spec.path_spec_type(0)
    assert cumotion.CompositePathSpec.PathSpecType.TASK_SPACE == composite_spec.path_spec_type(1)

    # Expect out of range calls to `path_spec_type()` to throw fatal errors.
    with pytest.raises(Exception):
        composite_spec.path_spec_type(-3)

    with pytest.raises(Exception):
        with errors_disabled:
            composite_spec.path_spec_type(2)

    # Test that a copy of the first input path specifications can be returned.
    cspace_2d_spec_copy = composite_spec.cspace_path_spec(0)
    assert 25.0 == cumotion.create_linear_cspace_path(cspace_2d_spec).path_length()
    assert 25.0 == cumotion.create_linear_cspace_path(cspace_2d_spec_copy).path_length()
    assert 3 == len(cumotion.create_linear_cspace_path(cspace_2d_spec).waypoints())
    assert 3 == len(cumotion.create_linear_cspace_path(cspace_2d_spec_copy).waypoints())

    # The path specification returned from `cspace_path_spec()` is a full copy and will *NOT* be
    # changed by changing the original input specification.
    cspace_2d_spec.add_cspace_waypoint(np.array([25.0, 25.0]))
    cspace_2d_spec_new_copy = composite_spec.cspace_path_spec(0)
    assert 38.0 == cumotion.create_linear_cspace_path(cspace_2d_spec).path_length()
    assert 25.0 == cumotion.create_linear_cspace_path(cspace_2d_spec_copy).path_length()
    assert 25.0 == cumotion.create_linear_cspace_path(cspace_2d_spec_new_copy).path_length()
    assert 4 == len(cumotion.create_linear_cspace_path(cspace_2d_spec).waypoints())
    assert 3 == len(cumotion.create_linear_cspace_path(cspace_2d_spec_copy).waypoints())
    assert 3 == len(cumotion.create_linear_cspace_path(cspace_2d_spec_new_copy).waypoints())

    # Test that a copy of the second input path specifications can be returned.
    task_space_spec_copy = composite_spec.task_space_path_spec(1)
    assert 35.0 == task_space_spec.generate_path().path_length()
    assert 35.0 == task_space_spec_copy.generate_path().path_length()

    # The path specification returned from `task_space_path_spec()` is a full copy and will *NOT* be
    # changed by changing the original input specification.
    task_space_spec.add_translation(np.array([25.0, 10.0, 20.0]))
    task_space_spec_new_copy = composite_spec.task_space_path_spec(1)
    assert 55.0 == task_space_spec.generate_path().path_length()
    assert 35.0 == task_space_spec_copy.generate_path().path_length()
    assert 35.0 == task_space_spec_new_copy.generate_path().path_length()

    # Expect that calling `cspace_path_spec()` or `task_space_path_spec()` with an out of range
    # index will return `None`.
    with errors_disabled:
        assert composite_spec.cspace_path_spec(2) is None
        assert composite_spec.task_space_path_spec(2) is None

    # Expect that calling `cspace_path_spec()` or `task_space_path_spec()` for a path specification
    # of the wrong type will return `None`.
    with errors_disabled:
        assert composite_spec.cspace_path_spec(1) is None
        assert composite_spec.task_space_path_spec(0) is None
