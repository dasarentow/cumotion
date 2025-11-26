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

"""This example demonstrates how to generate and execute a trajectory for Franka."""

# Standard Library
import os

# Third Party
import numpy as np
import time

# cuMotion
import cumotion
try:
    from cumotion_vis.visualizer import FrankaVisualization, RenderableType, Visualizer

    ENABLE_VIS = True
except ImportError:
    print("Visualizer not installed. Disabling visualization.")
    ENABLE_VIS = False

# Set cuMotion root directory
CUMOTION_ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))


def cumotion_print_status(success):
    """Print the final status of the example."""
    if success:
        print("CUMOTION EXAMPLE COMPLETED SUCCESSFULLY")
    else:
        print("CUMOTION EXAMPLE FAILED")


if __name__ == '__main__':
    # Set log level (optional, since `WARNING` is already the default).
    cumotion.set_log_level(cumotion.LogLevel.WARNING)

    # Set absolute path to the XRDF for Franka.
    xrdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'nvidia', 'shared', 'franka.xrdf')

    # Set absolute path to URDF file for Franka.
    urdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'third_party', 'franka', 'franka.urdf')

    # Load robot description.
    robot_description = cumotion.load_robot_from_file(xrdf_path, urdf_path)

    # Load kinematics.
    kinematics = robot_description.kinematics()

    # Use `kinematics` to create trajectory generator.
    trajectory_generator = cumotion.create_cspace_trajectory_generator(kinematics)

    # Expect number of c-space coordinates to be 7 for both `kinematics` and `trajectory_generator`.
    assert 7 == kinematics.num_cspace_coords()
    assert 7 == trajectory_generator.num_cspace_coords()
    num_coords = 7

    # Set position waypoints.
    waypoints = [np.array([0.0, 0.0, 0.0, -0.1, 0.0, 0.0, 0.0]),
                 np.array([0.5, 0.2, -0.5, -1.2, 0.5, 1.3, 0.5]),
                 np.array([0.0, 0.1, 1.5, -2.0, 0.1, 2.2, 0.5]),
                 np.array([0.7, 0.8, 0.5, -0.5, 0.0, 1.2, 0.5]),
                 np.array([1.0, 1.2, -2.1, -2.5, 1.3, 2.4, -0.5])]

    # Generate a trajectory with specified bounds and intermediate waypoint positions.
    start_time = time.perf_counter()
    trajectory = trajectory_generator.generate_trajectory(waypoints)

    # Print to console the computation time and time-span of generated trajectory.
    end_time = time.perf_counter()
    print("Trajectory generation time [ms]: ", 1000.0 * (end_time - start_time))
    print("Trajectory time span [s]: ", trajectory.domain().span())

    # Check that the trajectory span is in a reasonable range.  The expected value is
    # 5.205 seconds, but this has the potential to vary across architectures and compilers
    # due to differences in floating-point rounding.
    success = 1.0 <= trajectory.domain().span() <= 45.0
    cumotion_print_status(success)

    if not ENABLE_VIS:
        exit(0)

    # Initialize visualization.
    visualizer = Visualizer()

    # Add robot arm visualization to scene.
    mesh_folder = os.path.join(CUMOTION_ROOT_DIR, 'content', 'third_party', 'franka', 'meshes')
    franka_visualization = FrankaVisualization(robot_description, mesh_folder, visualizer)
    visualizer.fit_camera_to_scene()

    # Additionally, spherical markers will be added to visualize the end effector ("ee") position at
    # bounds and waypoint positions.
    ee_frame = "right_gripper"
    ee_positions = [kinematics.position(q, ee_frame) for q in waypoints]

    if ENABLE_VIS:
        sphere_config = {'radius': 0.05}
        # NOTE: Colors are set in "rainbow" order. See `franka_trajectory_example.cpp` for details.
        colors = [[1.0, 0.0, 0.0], [0.8, 1.0, 0.0], [0.0, 1.0, 0.4], [0.0, 0.4, 1.0],
                  [0.8, 0.0, 1.0]]
        for i, x in enumerate(ee_positions):
            sphere_config["color"] = colors[i]
            sphere_config["position"] = x
            visualizer.add(RenderableType.MARKER, "sphere_" + str(i), sphere_config)

    # Initialize Franka joint positions to beginning of trajectory.
    t = trajectory.domain().lower
    q = trajectory.eval(t)

    # Set "zero" time to be used to query trajectory.
    t0 = time.perf_counter()

    # Loop through visualization of `trajectory` until window is manually closed.
    while visualizer.is_active():
        # Update `t` to current time.
        t = time.perf_counter() - t0

        # If trajectory is complete, update `t` to beginning of `trajectory` and reset `timer`.
        if t > trajectory.domain().upper:
            t = trajectory.domain().lower
            t0 = time.perf_counter()

        # Update joint positions to the specified time.
        q = trajectory.eval(t)

        # Update Franka visualization.
        franka_visualization.set_joint_positions(q)
        visualizer.update()

    # Close visualization
    visualizer.close()
