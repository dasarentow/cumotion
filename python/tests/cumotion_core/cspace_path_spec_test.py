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

"""Unit tests for cspace_path_spec_python.h."""

# Third Party
import numpy as np

# cuMotion
import cumotion

# Local directory
from ._test_helper import errors_disabled


def test_cspace_path_spec():
    """Test `CSpacePathSpec`."""
    # Create c-space path specification.
    q0 = np.array([1.0, 2.0])
    spec = cumotion.create_cspace_path_spec(q0)

    # Expect the `spec` to have 2 c-space coordinates.
    assert 2 == spec.num_cspace_coords()

    # Add waypoints with the correct c-space dimension.
    waypoints = [np.array([3.0, 4.0]), np.array([5.0, 6.0])]
    for waypoint in waypoints:
        assert spec.add_cspace_waypoint(waypoint)

    # Expect that adding waypoints with the wrong dimension will fail.
    with errors_disabled:
        assert not spec.add_cspace_waypoint(np.array([1.0]))
        assert not spec.add_cspace_waypoint(np.array([3.0, 2.0, 1.0]))
