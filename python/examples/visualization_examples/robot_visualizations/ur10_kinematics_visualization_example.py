#!/usr/bin/env python3

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

"""This example demonstrates how to visualize a UR10 arm."""

# Standard Library
import argparse
import os

# Third Party
import math
import time

# cuMotion
import cumotion
try:
    from cumotion_vis.visualizer import UR10Visualization, Visualizer, CollisionSphereVisualization
except ImportError:
    print("Visualizer not installed. Cannot run visualization example.")
    print("CUMOTION EXAMPLE SKIPPED")
    exit(0)


def parse_arguments():
    """Parse command line arguments for showing collision spheres."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--show_world_collision_spheres', action='store_true',
                        default=False, help='Show world collision spheres')
    parser.add_argument('--show_self_collision_spheres', action='store_true',
                        default=False, help='Show self collision spheres')
    return parser.parse_args()


# Set cuMotion root directory
CUMOTION_ROOT_DIR = os.path.realpath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')
)

if __name__ == '__main__':
    # Set log level (optional, since `WARNING` is already the default).
    cumotion.set_log_level(cumotion.LogLevel.WARNING)

    # Set absolute path to the XRDF for the UR10.
    xrdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'nvidia', 'shared', 'ur10.xrdf')

    # Set absolute path to URDF file for the UR10.
    urdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'third_party', 'universal_robots',
                             'ur10', 'ur10_robot.urdf')

    # Load robot description.
    robot_description = cumotion.load_robot_from_file(xrdf_path, urdf_path)

    # Load kinematics.
    kinematics = robot_description.kinematics()

    # Initialize visualization.
    visualizer = Visualizer()

    # Add robot arm visualization to scene.
    mesh_folder = os.path.join(CUMOTION_ROOT_DIR, 'content', 'third_party', 'universal_robots',
                               'ur10', 'meshes', 'visual')
    ur10_visualization = UR10Visualization(robot_description, mesh_folder, visualizer)
    visualizer.fit_camera_to_scene()

    # Parse command line arguments.
    args = parse_arguments()

    # Show collision spheres if requested.
    show_collision_spheres = args.show_world_collision_spheres or args.show_self_collision_spheres
    if show_collision_spheres:
        collision_sphere_visualization = CollisionSphereVisualization(
            robot_description, visualizer,
            world_collision_spheres_color=[0.0, 0.2, 0.8],
            self_collision_spheres_color=[0.6, 0.0, 0.4],
            show_world_collision_spheres=args.show_world_collision_spheres,
            show_self_collision_spheres=args.show_self_collision_spheres)

    # Visualize oscillating robot until window is manually closed.
    while visualizer.is_active():
        # Update `t` to current time.
        t = time.perf_counter()

        # Set c-space position.
        q = robot_description.default_cspace_configuration()
        for i in range(kinematics.num_cspace_coords()):
            q[i] += 0.5 * math.sin(t)

        # Update UR10 visualization.
        ur10_visualization.set_joint_positions(q)
        if show_collision_spheres:
            collision_sphere_visualization.set_joint_positions(q)
        visualizer.update()

    # Close visualization
    visualizer.close()

    print("CUMOTION EXAMPLE COMPLETED SUCCESSFULLY")
