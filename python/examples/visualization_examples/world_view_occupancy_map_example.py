#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


"""Example: Visualize SDF of overlapping spheres using WorldViewOccupancyVisualizer."""


# Third Party
import numpy as np


# cuMotion
import cumotion
try:
    from cumotion_vis.visualizer import WorldViewOccupancyVisualizer, Visualizer
except ImportError:
    print("Visualizer not installed. Cannot run visualization example.")
    print("CUMOTION EXAMPLE SKIPPED")
    exit(0)


def add_single_sphere(world, radius, position):
    """Create a single sphere obstacle."""
    sphere = cumotion.create_obstacle(cumotion.Obstacle.Type.SPHERE)
    sphere.set_attribute(cumotion.Obstacle.Attribute.RADIUS, radius)
    world.add_obstacle(sphere, cumotion.Pose3.from_translation(np.array(position)))


def create_world_with_obstacles():
    """Create a world with two overlapping spheres."""
    world = cumotion.create_world()
    add_single_sphere(world, 0.8, [0.0, 0.0, 0.3])
    add_single_sphere(world, 0.4, [0.3, 0.5, 0.8])
    world_view = world.add_world_view()
    return world_view


def main():
    """Main function."""
    print("=== World View Occupancy Visualization Example ===")
    lower_bound = [-1.0, -1.0, -1.0]
    upper_bound = [1.5, 1.5, 1.5]
    resolution = 0.02

    print("Creating world...")
    world_view = create_world_with_obstacles()
    world_inspector = cumotion.create_world_inspector(world_view)
    print(f"Created world with {world_inspector.num_enabled_obstacles()} obstacles")

    print(f"Initializing 'WorldViewOccupancyVisualizer' (lower_bound={lower_bound}, "
          f"upper_bound={upper_bound}, resolution={resolution}):")
    viz = WorldViewOccupancyVisualizer(
        world_view, lower_bound=lower_bound, upper_bound=upper_bound, resolution=resolution)

    print("Visualizing world view with default visualizer.\nClose the window to continue.")
    viz.visualize_world_view()

    print("Visualizing world view with external visualizer.")
    external_visualizer = Visualizer()
    viz.visualize_world_view(external_visualizer)

    while True:
        external_visualizer.update()
        if not external_visualizer.is_active():
            break

    external_visualizer.close()
    print("CUMOTION EXAMPLE COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
