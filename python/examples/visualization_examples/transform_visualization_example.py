#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES.
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

"""This example demonstrates how to set position, rotation, and pose for a pair of cylinders."""

# Standard Library
import time
from enum import Enum

# Third Party
import numpy as np

# cuMotion
import cumotion
try:
    from cumotion_vis.visualizer import RenderableType, Visualizer

except ImportError:
    print("Visualizer not installed. Cannot run visualization example.")
    print("CUMOTION EXAMPLE SKIPPED")
    exit(0)

if __name__ == '__main__':
    # Initialize visualization.
    visualizer = Visualizer()

    # Set origin
    origin = np.zeros(3)

    # Generate two cylinders.
    absolute_cylinder_config = {
        'position': origin,
        'radius': 3.0,
        'height': 15.0,
        'color': [1.0, 0.5, 0.0]  # orange
    }
    visualizer.add(RenderableType.CYLINDER, "absolute", absolute_cylinder_config)
    relative_cylinder_config = {
        'position': origin,
        'radius': 5.0,
        'height': 5.0,
        'color': [0.5, 0.0, 1.0]  # purple
    }
    visualizer.add(RenderableType.CYLINDER, "relative", relative_cylinder_config)

    class Mode(Enum):
        """Example includes demonstrations of setting position, rotation, and pose."""

        POSITION = 1
        ROTATION = 2
        POSE = 3

    # Begin with POSITION mode.
    mode = Mode.POSITION

    # Set up positions with which to update cylinders.
    absolute_position = np.zeros(3)
    relative_position = np.array([0.05, 0.03, 0.01])

    # Set up rotations with which to update cylinders.
    absolute_rotation = cumotion.Rotation3.identity()
    relative_rotation = cumotion.Rotation3.from_axis_angle(np.array([1.0, 1.0, 0.0]), 0.05)

    # Set up poses with which to update cylinders.
    absolute_pose = cumotion.Pose3.identity()
    rotation = cumotion.Rotation3.from_axis_angle(np.array([0.0, 1.0, 0.0]), 0.02)
    relative_pose = cumotion.Pose3(rotation, np.array([0.05, 0.05, 0.05]))

    # Configure timing for visualization
    total_time = 15.0
    dt = 0.01
    current_time = 0.0

    # Step forward through visualization
    while (current_time < total_time):
        # Switch mode from POSITION -> ROTATION -> POSE based on time.
        # Reset pose for both cylinders with each state change.
        if current_time > 5.0 and mode == Mode.POSITION:
            visualizer.set_pose("absolute", cumotion.Pose3.identity())
            visualizer.set_pose("relative", cumotion.Pose3.identity())
            mode = Mode.ROTATION
        elif current_time > 10.0 and mode == Mode.ROTATION:
            visualizer.set_pose("absolute", cumotion.Pose3.identity())
            visualizer.set_pose("relative", cumotion.Pose3.identity())
            mode = Mode.POSE

        if mode == Mode.POSITION:
            # Increment absolute_position by relative_position and update cylinders in
            # visualization.
            absolute_position += relative_position
            visualizer.set_position("absolute", absolute_position)
            visualizer.update_position("relative", relative_position)
        elif mode == Mode.ROTATION:
            # Increment absolute_rotation by relative_rotation and update cylinders in
            # visualization.
            absolute_rotation = relative_rotation * absolute_rotation
            visualizer.set_rotation("absolute", absolute_rotation)
            visualizer.update_rotation("relative", relative_rotation)
        elif mode == Mode.POSE:
            # Increment absolute_pose by relative_pose and update cylinders in visualization.
            absolute_pose = absolute_pose * relative_pose
            visualizer.set_pose("absolute", absolute_pose)
            visualizer.update_pose("relative", relative_pose)

        # Update visualization.
        visualizer.update()

        # Step forward in time.
        current_time += dt

        # Sleep for one timestep to approximate realtime visualization.
        time.sleep(dt)

    # Close visualization
    visualizer.close()
    print("CUMOTION EXAMPLE COMPLETED SUCCESSFULLY")
