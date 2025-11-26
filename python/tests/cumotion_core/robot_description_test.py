# SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES.
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

"""Unit tests for robot_description_python.h."""

# Standard Library
import os
import pathlib

# Third Party
import numpy as np
import pytest

# cuMotion
import cumotion

# Local Folder
from ._test_helper import CUMOTION_ROOT_DIR


def test_load_from_pathlib():
    """Test loading robot from pathlib types."""
    config_path = pathlib.Path(CUMOTION_ROOT_DIR) / 'content' / 'nvidia' / 'shared'
    xrdf_path = config_path / 'franka.xrdf'
    urdf_path = pathlib.Path(
        CUMOTION_ROOT_DIR) / 'content' / 'third_party' / 'franka' / 'franka.urdf'
    robot_description = cumotion.load_robot_from_file(xrdf_path, urdf_path)

    num_cspace_coords_expected = 7
    assert num_cspace_coords_expected == robot_description.num_cspace_coords()


@pytest.fixture
def configure_robot_description():
    """Test fixture to configure RMPFlow object."""

    def _configure_robot_description():
        # Set directory for RMPflow configuration and robot description YAML files.
        config_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'nvidia', 'shared')

        # Set absolute path to the XRDF for Franka.
        xrdf_path = os.path.join(config_path, "franka.xrdf")

        # Set absolute path to URDF file for Franka.
        urdf_path = os.path.join(CUMOTION_ROOT_DIR, 'content', 'third_party', 'franka',
                                 'franka.urdf')

        # Load and return robot description.
        return cumotion.load_robot_from_file(xrdf_path, urdf_path)

    return _configure_robot_description


def test_franka_robot_description(configure_robot_description):
    """Test the Franka robot description."""
    # Init robot_description from test fixture
    robot_description = configure_robot_description()

    # Run tests
    repr_expected = 'RobotDescription'
    assert repr_expected == robot_description.__repr__()

    num_cspace_coords_expected = 7
    assert num_cspace_coords_expected == robot_description.num_cspace_coords()

    cspace_coord_names_expected = [
        'panda_joint1',
        'panda_joint2',
        'panda_joint3',
        'panda_joint4',
        'panda_joint5',
        'panda_joint6',
        'panda_joint7']
    cspace_coord_names_actual = [robot_description.cspace_coord_name(
        idx) for idx in range(0, robot_description.num_cspace_coords())]
    assert cspace_coord_names_expected == cspace_coord_names_actual

    default_q_expected = np.array([0.00, -1.3, 0.00, -2.87, 0.00, 2.00, 0.75])
    default_q_actual = robot_description.default_cspace_configuration()
    assert default_q_expected == pytest.approx(default_q_actual)

    assert 1 == len(robot_description.tool_frame_names())
    assert "panda_leftfingertip" == robot_description.tool_frame_names()[0]
