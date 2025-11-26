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

"""This example demonstrates how to generate a task space trajectory for the Fanuc M20-iA."""

# Standard Library
import argparse
import os

# Third Party
import math
import numpy as np
import time

# cuMotion
import cumotion
try:
    from cumotion_vis.visualizer import FanucM20iAVisualization, RenderableType, Visualizer

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


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a task space trajectory for the Fanuc M20-iA.")

    # Use "--path_selection" to select a task space path.
    #
    # The following paths are available for demonstration:
    #   [0] Simple rectangular path generated procedurally by `generate_rectangle_path()`.
    #   [1] Spiral path generated procedurally by `generate_spiral_path()`.
    #   [2] Stacked circle path loaded from file by `load_stacked_circle_path()`, which loads
    #       "content/nvidia/examples/trajectory_examples/task_space_paths/
    #       stacked_circle_task_space_path.yaml".
    #   [3] NVIDIA logo path loaded from file by `load_nvidia_logo_path()`, which loads
    #       "content/nvidia/examples/trajectory_examples/task_space_paths/
    #       nvidia_logo_task_space_path.yaml".
    #       NOTE: For this demonstration, it is recommended to set the path deviations to:
    #         --min_path_deviation=0.0003 --max_path_deviation=0.001
    parser.add_argument('--path_selection', type=int, default=3,
                        help="Select task space path (0-3).")

    # Set the minimum and maximum path deviations for converting from task space to c-space.
    #
    # See `TaskSpaceTrajectoryConfig` in `task_space_trajectory_generator.h` for details.
    parser.add_argument('--min_path_deviation', type=float, default=0.001,
                        help="Set minimum position deviation for path conversion.")
    parser.add_argument('--max_path_deviation', type=float, default=0.003,
                        help="Set maximum position deviation for path conversion.")

    # If "path_selection" is set to `0` (i.e., rectangular path), then the blend radius can be set
    # to smooth the corners of the rectangle.
    #
    # See `TaskSpacePathSpec.add_translation()` in `task_space_path_spec.h` for details.
    parser.add_argument('--blend_radius', type=float, default=0.02,
                        help="Set blend radius between linear path segments.")

    return parser.parse_args()


def generate_rectangle_path_spec(blend_radius):
    """Generate a task space path specification with a simple rectangle."""
    # Set initial orientation.
    R0 = cumotion.Rotation3.from_axis_angle(np.array([0.0, 1.0, 0.0]), 0.5 * math.pi)

    # Set bounds for rectangle on YZ-plane.
    x = 1.25
    min_y = -0.1
    max_y = 0.1
    min_z = 0.2
    max_z = 1.0

    # Set rectangle corners.
    corners = [np.array([x, max_y, min_z]),
               np.array([x, max_y, max_z]),
               np.array([x, min_y, max_z]),
               np.array([x, min_y, min_z])]

    # Generate and return task space path.
    spec = cumotion.create_task_space_path_spec(cumotion.Pose3(R0, corners[-1]))
    for corner in corners:
        spec.add_translation(corner, blend_radius)
    return spec


def generate_spiral_path_spec():
    """Generate a task space path specification with a spiral."""
    # Set initial orientation.
    R0 = cumotion.Rotation3.from_axis_angle(np.array([0.0, 1.0, 0.0]), 0.5 * math.pi)

    # Set constant positions for x, and z-endpoints
    x = 1.25
    z = 0.7

    # Initialize spec and add initial three-point arc (since path cannot begin with a tangent arc)
    spec = cumotion.create_task_space_path_spec(cumotion.Pose3(R0, [x, 0.0, z]))
    radius = 0.015
    y = 2.0 * radius
    spec.add_three_point_arc(np.array([x, y, z]), np.array([x, radius, z + radius]))

    # Continue spiral with a series of tangent arcs.
    for i in range(20):
        radius += 0.03
        y = y - radius if (i % 2 == 0) else y + radius
        spec.add_tangent_arc(np.array([x, y, z]))

    # Generate and return task space path.
    return spec


def load_path_spec(filename):
    """Load a YAML file containing a task space path specification.

    `filename` is expected to be located in the hard-coded directory `dir`.
    """
    dir = os.path.join(CUMOTION_ROOT_DIR, 'content', 'nvidia', 'examples', 'trajectory_examples',
                       'task_space_paths')
    filepath = os.path.join(dir, filename)
    return cumotion.load_task_space_path_spec_from_file(filepath)


def load_stacked_circle_path_spec():
    """Load a YAML file containing a task space path spec for a series of stacked circles."""
    return load_path_spec("stacked_circle_task_space_path.yaml")


def load_nvidia_logo_path_spec():
    """Load a YAML file containing a task space path specification for the NVIDIA logo."""
    return load_path_spec("nvidia_logo_task_space_path.yaml.big")


def approximate_task_space_path(trajectory, kinematics, control_frame, num_samples=1000):
    """Create an approximate `TaskSpacePath` for the `control_frame` along the `trajectory`."""
    x0 = kinematics.pose(trajectory.eval(trajectory.domain().lower), control_frame)
    path_spec = cumotion.create_task_space_path_spec(x0)

    # Sample `trajectory`, skipping lower bound.
    for t in np.linspace(trajectory.domain().lower, trajectory.domain().upper, num_samples)[1:]:
        path_spec.add_linear_path(kinematics.pose(trajectory.eval(t), control_frame))

    return path_spec.generate_path()


if __name__ == '__main__':
    flags = parse_arguments()

    # Set log level (optional, since `WARNING` is already the default).
    cumotion.set_log_level(cumotion.LogLevel.WARNING)

    # Set absolute path to robot description for the Fanuc M20-iA.
    robot_description_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'nvidia', 'shared',
                                          'fanuc_m20ia.xrdf')

    # Set absolute path to URDF file for Fanuc M20-iA.
    urdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'third_party', 'fanuc', 'm20ia',
                             'm20ia.urdf')

    # Load robot description.
    robot_description = cumotion.load_robot_from_file(robot_description_path, urdf_path)

    # Load kinematics.
    kinematics = robot_description.kinematics()

    # Set control frame to follow task space path.
    control_frame = "tool0"

    # Create task space path based on command line arguments.
    task_space_path_spec = \
        generate_rectangle_path_spec(flags.blend_radius) if flags.path_selection == 0 \
        else generate_spiral_path_spec() if flags.path_selection == 1 \
        else load_stacked_circle_path_spec() if flags.path_selection == 2 \
        else load_nvidia_logo_path_spec()
    task_space_path = task_space_path_spec.generate_path()

    # Use `kinematics` to create trajectory generator with default configuration parameters.
    cspace_trajectory_generator = cumotion.create_cspace_trajectory_generator(kinematics)

    # Create `TaskSpacePathConversionConfig` and update position deviation limits from command line
    # arguments.
    path_conversion_config = cumotion.TaskSpacePathConversionConfig()
    path_conversion_config.min_position_deviation = flags.min_path_deviation
    path_conversion_config.max_position_deviation = flags.max_path_deviation

    # Create timer to measure time to generate the task space trajectory.
    tic = time.perf_counter()

    # Generate a c-space path from a `TaskSpacePathSpec`.
    cspace_path = cumotion.convert_task_space_path_spec_to_cspace(
        task_space_path_spec, kinematics, control_frame, path_conversion_config)

    # Generate trajectory, expecting success.
    trajectory = cspace_trajectory_generator.generate_trajectory(cspace_path.waypoints())

    # Print to console the computation time and time-span of generated trajectory.
    toc = time.perf_counter()
    print("Trajectory generation time [ms]: ", 1000.0 * (toc - tic))
    print("Trajectory time span [s]: ", trajectory.domain().span())

    # Check that the trajectory span is in a reasonable range.  The expected value is
    # 8.937 seconds, but this can vary across architectures and compilers due to differences
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
    mesh_folder = os.path.join(CUMOTION_ROOT_DIR, 'content', 'third_party', 'fanuc', 'm20ia',
                               'meshes', 'visual')
    fanuc_visualization = FanucM20iAVisualization(robot_description, mesh_folder, visualizer)

    # Add end effector frame to visualization.
    visualizer.add(RenderableType.COORDINATE_FRAME, "control_frame_vis", {'size': 0.2})

    # Visualize the desired task space path.
    visualizer.add_task_space_path(task_space_path, coordinate_axis_size=0.02,
                                   num_coord_samples=100)

    # Visualize the actual task space path encoded into `trajectory`.
    actual_task_space_path = approximate_task_space_path(trajectory, kinematics, control_frame)
    visualizer.add_task_space_path(actual_task_space_path,
                                   path_color=[0.463, 0.725, 0.0],  # NVIDIA green
                                   coordinate_axis_size=0.02,
                                   num_coord_samples=100)

    # Adjust camera to fit all geometry in scene.
    visualizer.fit_camera_to_scene()

    # Initialize Fanuc M20-iA joint positions to beginning of trajectory.
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

        # Update Fanuc M20-iA visualization.
        fanuc_visualization.set_joint_positions(q)

        # Update control frame visualization.
        visualizer.set_pose("control_frame_vis", kinematics.pose(q, control_frame))

        visualizer.update()

    # Close visualization
    visualizer.close()
