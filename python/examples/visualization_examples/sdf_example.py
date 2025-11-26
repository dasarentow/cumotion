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

"""Example demonstrating signed distance field (SDF) visualization using the SdfVisualizer class."""

# Third Party
import numpy as np

# cuMotion
import cumotion
try:
    from cumotion_vis.visualizer import SdfVisualizer
except ImportError:
    print("Visualizer not installed. Cannot run visualization example.")
    print("CUMOTION EXAMPLE SKIPPED")
    exit(0)


def create_cuboid_obstacle(world, size, position=None):
    """Create a cuboid obstacle."""
    if position is None:
        position = [0.0, 0.0, 0.25]
    if size is None:
        size = [0.5, 0.5, 0.5]

    cuboid = cumotion.create_obstacle(cumotion.Obstacle.Type.CUBOID)
    cuboid.set_attribute(cumotion.Obstacle.Attribute.SIDE_LENGTHS,
                         np.array(size))
    cuboid_pose = cumotion.Pose3.from_translation(np.array(position))
    world.add_obstacle(cuboid, cuboid_pose)

    config = {
        'type': 'cuboid',
        'size': size,
        'position': position
    }
    return config


def create_table_obstacle(world):
    """Create a table obstacle with legs."""
    configs = []

    # Table dimensions
    table_top_width = 1.2
    table_top_depth = 0.8
    table_top_thickness = 0.05
    table_height = 0.75
    leg_size = 0.05
    leg_height = table_height - table_top_thickness

    # Create table top
    table_top = cumotion.create_obstacle(cumotion.Obstacle.Type.CUBOID)
    table_top.set_attribute(cumotion.Obstacle.Attribute.SIDE_LENGTHS,
                            np.array([table_top_width, table_top_depth, table_top_thickness]))
    table_top_pose = cumotion.Pose3.from_translation(
        np.array([0.0, 0.0, table_height - table_top_thickness / 2]))
    world.add_obstacle(table_top, table_top_pose)

    # Table top config
    table_top_config = {
        'type': 'table_top',
        'width': table_top_width,
        'depth': table_top_depth,
        'thickness': table_top_thickness,
        'height': table_height
    }
    configs.append(table_top_config)

    # Calculate leg positions
    leg_positions = [
        [table_top_width / 2 - leg_size / 2, table_top_depth / 2 - leg_size / 2, leg_height / 2],
        [-table_top_width / 2 + leg_size / 2, table_top_depth / 2 - leg_size / 2, leg_height / 2],
        [table_top_width / 2 - leg_size / 2, -table_top_depth / 2 + leg_size / 2, leg_height / 2],
        [-table_top_width / 2 + leg_size / 2, -table_top_depth / 2 + leg_size / 2, leg_height / 2]
    ]

    # Create table legs
    for leg_pos in leg_positions:
        leg = cumotion.create_obstacle(cumotion.Obstacle.Type.CUBOID)
        leg.set_attribute(cumotion.Obstacle.Attribute.SIDE_LENGTHS,
                          np.array([leg_size, leg_size, leg_height]))
        leg_pose = cumotion.Pose3.from_translation(np.array(leg_pos))
        world.add_obstacle(leg, leg_pose)

        # Leg config
        leg_config = {
            'type': 'table_leg',
            'size': leg_size,
            'height': leg_height,
            'position': leg_pos
        }
        configs.append(leg_config)

    return configs


def create_example_world_obstacles(world):
    """Create the example environment: table + 2 spheres of same size."""
    configs = []

    # Table dimensions
    table_top_width = 1.2
    table_top_depth = 0.8
    table_top_thickness = 0.05
    table_height = 0.75

    # Create table top only (no legs for simplicity)
    table_top = cumotion.create_obstacle(cumotion.Obstacle.Type.CUBOID)
    table_top.set_attribute(cumotion.Obstacle.Attribute.SIDE_LENGTHS,
                            np.array([table_top_width, table_top_depth, table_top_thickness]))
    table_top_pose = cumotion.Pose3.from_translation(
        np.array([0.0, 0.0, table_height - table_top_thickness / 2]))
    world.add_obstacle(table_top, table_top_pose)

    # Table top config
    table_top_config = {
        'type': 'example_table_top',
        'width': table_top_width,
        'depth': table_top_depth,
        'thickness': table_top_thickness,
        'height': table_height
    }
    configs.append(table_top_config)

    # Create two spheres of the same size (using radius 0.15 for both)
    sphere_radius = 0.15

    # Sphere 1 position
    sphere1_position = [0.3, 0.2, table_height + sphere_radius]
    sphere1 = cumotion.create_obstacle(cumotion.Obstacle.Type.SPHERE)
    sphere1.set_attribute(cumotion.Obstacle.Attribute.RADIUS, sphere_radius)
    sphere1_pose = cumotion.Pose3.from_translation(np.array(sphere1_position))
    world.add_obstacle(sphere1, sphere1_pose)

    sphere1_config = {
        'type': 'sphere',
        'radius': sphere_radius,
        'position': sphere1_position
    }
    configs.append(sphere1_config)

    # Sphere 2 position (same radius)
    sphere2_position = [-0.2, -0.3, table_height + sphere_radius]
    sphere2 = cumotion.create_obstacle(cumotion.Obstacle.Type.SPHERE)
    sphere2.set_attribute(cumotion.Obstacle.Attribute.RADIUS, sphere_radius)
    sphere2_pose = cumotion.Pose3.from_translation(np.array(sphere2_position))
    world.add_obstacle(sphere2, sphere2_pose)

    sphere2_config = {
        'type': 'sphere',
        'radius': sphere_radius,
        'position': sphere2_position
    }
    configs.append(sphere2_config)

    return configs


def get_scene_configs():
    """Get available scene configurations.

    Note: The values for bounds and resolution are represented in meters.
    """
    return {
        "1": {
            "name": "Simple cuboid",
            "creators": [
                lambda world: create_cuboid_obstacle(
                    world, size=[0.5, 0.5, 0.5], position=[0.0, 0.0, 0.25])
            ],
            "bounds": (-1.5, 1.5),
            "resolution": 0.05
        },
        "2": {
            "name": "Table with legs",
            "creators": [create_table_obstacle],
            "bounds": (-2.0, 2.0),
            "resolution": 0.08
        },
        "3": {
            "name": "Cuboid + Table",
            "creators": [
                lambda world: create_cuboid_obstacle(
                    world, size=[0.3, 0.3, 0.3], position=[1.0, 1.0, 0.15]),
                create_table_obstacle
            ],
            "bounds": (-2.5, 2.5),
            "resolution": 0.1
        },
        "4": {
            "name": "Table + 2 Spheres (recommended)",
            "creators": [create_example_world_obstacles],
            "bounds": (-1.5, 1.5),
            "resolution": 0.05
        }
    }


def create_world_with_obstacles(obstacle_creators):
    """Create a world with specified obstacles."""
    world = cumotion.create_world()

    all_configs = []

    for creator_func in obstacle_creators:
        configs = creator_func(world)
        if isinstance(configs, list):
            all_configs.extend(configs)
        else:
            all_configs.append(configs)

    # Create world view for distance queries
    world_view = world.add_world_view()

    return world_view, all_configs


def main():
    """Demonstrate SDF visualization."""
    print("=== cuMotion SDF Visualization Example ===")
    print("This example demonstrates the SdfVisualizer class")
    print()

    scene_configs = get_scene_configs()
    print("Available scene configurations:")
    for key, config in scene_configs.items():
        print(f"{key}. {config['name']}")

    scene_choice = input("Enter scene choice (1-4) [default: 4]: ").strip() or "4"
    if scene_choice not in scene_configs:
        raise ValueError(f"Invalid scene choice: {scene_choice}")

    selected_config = scene_configs[scene_choice]
    print(f"\nCreating world for '{selected_config['name']}'...")
    world_view, obstacle_configs = create_world_with_obstacles(selected_config["creators"])
    print(f"Created world with {len(obstacle_configs)} obstacles")

    # Create SDF visualizer
    print(
        f"Initializing SdfVisualizer (bounds={selected_config['bounds']}, "
        f"resolution={selected_config['resolution']})...")
    sdf_viz = SdfVisualizer(
        world_view,
        obstacle_configs,
        bounds=selected_config['bounds'],
        resolution=selected_config['resolution'])

    # Choose visualization mode
    print("\nVisualization modes:")
    print("1. Static SDF visualization (voxel grid)")
    print("2. Interactive SDF plane sweep")

    vis_choice = input("Enter choice (1-2) [default: 2]: ").strip() or "2"
    if vis_choice not in ["1", "2"]:
        raise ValueError(f"Invalid visualization choice: {vis_choice}")

    if vis_choice == "1":
        print("\n=== Static SDF Visualization ===")
        print("This will show the complete SDF as a colored voxel grid")
        print("Legend:")
        print(" - Blue voxels: Inside/close to obstacles")
        print(" - Red voxels:  Far from obstacles")
        input("Press 'Enter' to start static visualization")
        sdf_viz.visualize_sdf_static()
    elif vis_choice == "2":
        print("\n=== Interactive SDF Visualization ===")
        print("This will allow you to sweep through SDF planes interactively")
        input("Press 'Enter' to start interactive visualization")
        sdf_viz.visualize_sdf_interactive()
    else:
        raise ValueError(f"Invalid visualization choice: {vis_choice}")
    print("CUMOTION EXAMPLE COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
