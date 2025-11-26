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

"""Unit tests for linear_cspace_path_python.h."""

# Third Party
import numpy as np
import pytest

# cuMotion
import cumotion


def test_linear_cspace_path_with_rectangle_path():
    """Test `LinearCSpacePath` with a simple rectangular path."""
    # Create c-space path specification.
    q0 = np.array([0.0, 0.0])
    spec = cumotion.create_cspace_path_spec(q0)

    # Expect the `spec` to have 2 c-space coordinates.
    assert 2 == spec.num_cspace_coords()

    # Add waypoints forming a 2d rectangle.
    width = 6.0
    height = 3.0
    waypoints = [np.array([width, 0.0]), np.array([width, height]), np.array([0.0, height]), q0]
    for waypoint in waypoints:
        assert spec.add_cspace_waypoint(waypoint)

    # Create a linearly interpolated c-space path from `spec`.
    path = cumotion.create_linear_cspace_path(spec)

    # Expect `path` to have 2 c-space coordinates.
    assert 2 == path.num_cspace_coords()

    # Expect domain span to be equivalent to rectangle perimeter.
    expected_domain_span = (2.0 * width) + (2.0 * height)
    assert expected_domain_span == path.domain().span()
    assert 0.0 == path.domain().lower
    assert path.domain().span() == path.domain().upper

    # Expect path to be evaluated correctly at waypoints.
    assert q0 == pytest.approx(path.eval(0.0))
    assert waypoints[0] == pytest.approx(path.eval(width))
    assert waypoints[1] == pytest.approx(path.eval(width + height))
    assert waypoints[2] == pytest.approx(path.eval(2.0 * width + height))
    assert waypoints[3] == pytest.approx(path.eval(expected_domain_span))

    # Expect path to be evaluated correctly between waypoints.
    assert 0.5 * (q0 + waypoints[0]) == pytest.approx(path.eval(0.5 * width))
    assert 0.5 * (waypoints[0] + waypoints[1]) == pytest.approx(path.eval(width + 0.5 * height))
    assert 0.5 * (waypoints[1] + waypoints[2]) == pytest.approx(path.eval(1.5 * width + height))
    assert 0.5 * (waypoints[2] + waypoints[3]) == \
           pytest.approx(path.eval(2.0 * width + 1.5 * height))

    # Expect path length to be equivalent to domain span.
    assert expected_domain_span == path.path_length()

    # Expect minimum position at origin ([0, 0]).
    assert q0 == pytest.approx(path.min_position())

    # Expect maximum position at origin ([width, height]).
    assert np.array([width, height]) == pytest.approx(path.max_position())

    # Expect `path.waypoints()` to include `q0` and `waypoints`.
    all_expected_waypoints = [q0]
    for waypoint in waypoints:
        all_expected_waypoints.append(waypoint)

    for expected, actual in zip(all_expected_waypoints, path.waypoints()):
        assert expected == pytest.approx(actual)
