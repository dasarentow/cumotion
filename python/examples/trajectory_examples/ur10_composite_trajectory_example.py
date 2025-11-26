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

"""This example demonstrates how to generate a trajectory for the UR10 from a composite path spec.

This path spec is composited from task space specifications and c-space waypoints.
"""

# Standard Library
import os

# Third Party
import math
import numpy as np
import time

# cuMotion
import cumotion
try:
    from cumotion_vis.visualizer import UR10Visualization, RenderableType, Visualizer

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


def approximate_task_space_path(trajectory, kinematics, control_frame, num_samples=1000):
    """Create an approximate `TaskSpacePath` for the `control_frame` along the `trajectory`."""
    x0 = kinematics.pose(trajectory.eval(trajectory.domain().lower), control_frame)
    path_spec = cumotion.create_task_space_path_spec(x0)

    # Sample `trajectory`, skipping lower bound.
    for t in np.linspace(trajectory.domain().lower, trajectory.domain().upper, num_samples)[1:]:
        path_spec.add_linear_path(kinematics.pose(trajectory.eval(t), control_frame))

    return path_spec.generate_path()


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

    # Set control frame to follow task space path.
    control_frame = "tool0"

    # Use `kinematics` to create trajectory generator with default configuration parameters.
    cspace_trajectory_generator = cumotion.create_cspace_trajectory_generator(kinematics)

    # Set four corners of a rectangle.
    R = cumotion.Rotation3.from_axis_angle(np.array([0.0, 1.0, 0.0]), 0.5 * math.pi)
    minX = 0.4
    maxX = 0.9
    y = 0.1
    minZ = 0.4
    maxZ = 0.9
    xA = np.array([minX, y, minZ])
    xB = np.array([maxX, y, minZ])
    xC = np.array([minX, y, maxZ])
    xD = np.array([maxX, y, maxZ])

    # Create a task space path spec between the four corners (in an hourglass pattern) with no blend
    # radius.
    task_space_path_spec = cumotion.create_task_space_path_spec(cumotion.Pose3(R, xA))
    task_space_path_spec.add_translation(xB)
    task_space_path_spec.add_translation(xC)
    task_space_path_spec.add_translation(xD)
    task_space_path_spec.add_translation(xA)

    # Create a task space path spec between the four corners (in an hourglass pattern) with a
    # non-zero blend radius.
    blend_radius = 0.03
    task_space_path_spec_blended = cumotion.create_task_space_path_spec(cumotion.Pose3(R, xA))
    task_space_path_spec_blended.add_translation(xB, blend_radius)
    task_space_path_spec_blended.add_translation(xC, blend_radius)
    task_space_path_spec_blended.add_translation(xD, blend_radius)
    task_space_path_spec_blended.add_translation(xA, blend_radius)

    # Convert four rectangle corners to c-space.
    ik_config = cumotion.IkConfig()
    qA = cumotion.solve_ik(kinematics, cumotion.Pose3(R, xA), control_frame,
                           ik_config).cspace_position
    qB = cumotion.solve_ik(kinematics, cumotion.Pose3(R, xB), control_frame,
                           ik_config).cspace_position
    qC = cumotion.solve_ik(kinematics, cumotion.Pose3(R, xC), control_frame,
                           ik_config).cspace_position
    qD = cumotion.solve_ik(kinematics, cumotion.Pose3(R, xD), control_frame,
                           ik_config).cspace_position

    # Create a c-space path spec between the four corners (in an hourglass pattern).
    cspace_path_spec = cumotion.create_cspace_path_spec(qA)
    cspace_path_spec.add_cspace_waypoint(qB)
    cspace_path_spec.add_cspace_waypoint(qC)
    cspace_path_spec.add_cspace_waypoint(qD)
    cspace_path_spec.add_cspace_waypoint(qA)

    # Create a composite path spec combining the three "hourglass" paths.
    composite_path_spec = cumotion.create_composite_path_spec(qA)
    composite_path_spec.add_task_space_path_spec(
        task_space_path_spec, cumotion.CompositePathSpec.TransitionMode.LINEAR_TASK_SPACE)
    composite_path_spec.add_task_space_path_spec(
        task_space_path_spec_blended, cumotion.CompositePathSpec.TransitionMode.SKIP)
    composite_path_spec.add_cspace_path_spec(
        cspace_path_spec, cumotion.CompositePathSpec.TransitionMode.SKIP)

    # Create timer to measure time to generate the trajectory from `composite_path_spec`.
    tic = time.perf_counter()

    # Generate a c-space path from a `CompositePathSpec`.
    path_conversion_config = cumotion.TaskSpacePathConversionConfig()
    path_conversion_config.min_position_deviation = 0.0005
    path_conversion_config.max_position_deviation = 0.001
    cspace_path = cumotion.convert_composite_path_spec_to_cspace(
        composite_path_spec, kinematics, control_frame, path_conversion_config)

    # Generate trajectory, expecting success.
    trajectory = cspace_trajectory_generator.generate_trajectory(cspace_path.waypoints())

    # Print to console the computation time and time-span of generated trajectory.
    toc = time.perf_counter()
    print("Trajectory generation time [ms]: ", 1000.0 * (toc - tic))
    print("Trajectory time span [s]: ", trajectory.domain().span())

    # Check that the trajectory span is in a reasonable range.  The expected value is
    # 26.28 seconds, but this can vary across architectures and compilers due to differences
    # in floating-point rounding resulting in different IK solutions during path conversion.
    success = 1.0 <= trajectory.domain().span() <= 45.0
    cumotion_print_status(success)

    # ==============================================================================================
    # NOTE: Everything below this point is purely for visualizing the trajectory generated above.
    # ==============================================================================================

    if not ENABLE_VIS:
        exit(0)

    # Initialize visualization.
    visualizer = Visualizer()

    # Add robot arm visualization to scene.
    mesh_folder = os.path.join(CUMOTION_ROOT_DIR, 'content', 'third_party', 'universal_robots',
                               'ur10', 'meshes', 'visual')
    robot_vis = UR10Visualization(robot_description, mesh_folder, visualizer)

    # Add end effector frame to visualization.
    visualizer.add(RenderableType.COORDINATE_FRAME, "control_frame_vis", {'size': 0.2})

    # Add markers for the four corners to visualization.
    marker_config = {'radius': 0.02, 'color': [1.0, 0.5, 0.0], 'position': xA}
    visualizer.add(RenderableType.MARKER, "A", marker_config)
    marker_config['position'] = xB
    visualizer.add(RenderableType.MARKER, "B", marker_config)
    marker_config['position'] = xC
    visualizer.add(RenderableType.MARKER, "C", marker_config)
    marker_config['position'] = xD
    visualizer.add(RenderableType.MARKER, "D", marker_config)

    # Visualize the actual task space path encoded into `trajectory`.
    actual_task_space_path = approximate_task_space_path(trajectory, kinematics, control_frame)
    visualizer.add_task_space_path(actual_task_space_path,
                                   path_color=[0.463, 0.725, 0.0],  # NVIDIA green
                                   coordinate_axis_size=0.02,
                                   num_coord_samples=100)

    # Adjust camera to fit all geometry in scene.
    visualizer.fit_camera_to_scene()

    # Initialize UR10 joint positions to beginning of trajectory.
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

        # Update UR10 visualization.
        robot_vis.set_joint_positions(q)

        # Update control frame visualization.
        visualizer.set_pose("control_frame_vis", kinematics.pose(q, control_frame))

        visualizer.update()

    # Close visualization
    visualizer.close()
