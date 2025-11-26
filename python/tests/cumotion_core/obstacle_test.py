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

"""Unit tests for obstacle_python.h."""

# Third Party
import numpy as np

# cuMotion
import cumotion


def test_obstacle():
    """Test the cumotion::Obstacle Python wrapper."""
    # Create cuboid.
    cuboid = cumotion.create_obstacle(cumotion.Obstacle.Type.CUBOID)
    cuboid.set_attribute(cumotion.Obstacle.Attribute.SIDE_LENGTHS, np.array([2.3, 1.2, 9.8]))
    assert cumotion.Obstacle.Type.CUBOID == cuboid.type()

    # Create capsule.
    capsule = cumotion.create_obstacle(cumotion.Obstacle.Type.CAPSULE)
    capsule.set_attribute(cumotion.Obstacle.Attribute.RADIUS, 4.7)
    capsule.set_attribute(cumotion.Obstacle.Attribute.HEIGHT, 2.4)
    assert cumotion.Obstacle.Type.CAPSULE == capsule.type()

    # Create sphere.
    sphere = cumotion.create_obstacle(cumotion.Obstacle.Type.SPHERE)
    sphere.set_attribute(cumotion.Obstacle.Attribute.RADIUS, 9.3)
    assert cumotion.Obstacle.Type.SPHERE == sphere.type()

    # Create SDF.
    sdf = cumotion.create_obstacle(cumotion.Obstacle.Type.SDF)
    sdf.set_attribute(cumotion.Obstacle.Attribute.GRID,
                      cumotion.Obstacle.Grid(4, 6, 8, 0.1,
                                             cumotion.Obstacle.GridPrecision.FLOAT,
                                             cumotion.Obstacle.GridPrecision.HALF))
    assert cumotion.Obstacle.Type.SDF == sdf.type()
