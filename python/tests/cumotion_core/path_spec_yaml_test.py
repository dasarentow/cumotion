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

"""Unit tests for path_spec_from_yaml_python.h."""

# Standard Library
import os

# Third Party
import math
import numpy as np
import pytest

# cuMotion
import cumotion

# Local Folder
from ._test_helper import CUMOTION_ROOT_DIR


def get_abs_filepath_for_path_spec(filename):
    """Return absolute path to `filepath` within directory of path specs for testing."""
    dir = os.path.join(CUMOTION_ROOT_DIR, 'content', 'nvidia', 'tests', 'trajectory')
    return os.path.join(dir, filename)


def load_as_string(filepath):
    """Load `filepath` and return contents as string."""
    with open(filepath, 'r') as file:
        file_string = file.read()

    return file_string


def remove_blank_lines_and_comments(string: str) -> str:
    """Remove blank lines and lines starting with '#' from `string`."""
    return "\n".join(line for line in string.splitlines()
                     if line.strip() and not line.lstrip().startswith("#"))


def build_example_task_space_path_spec_procedurally():
    """Create an example task space path spec."""
    # Create `task_space_pathSpec`.
    spec = cumotion.create_task_space_path_spec(cumotion.Pose3.identity())

    # Add path segments.
    R0 = cumotion.Rotation3(0.0, 0.0, 0.0, 1.0)
    spec.add_linear_path(cumotion.Pose3(R0, np.array([0.5, 0.0, 0.0])))
    spec.add_translation(np.array([0.5, 1.0, 0.0]))
    spec.add_rotation(cumotion.Rotation3(0.0, 0.0, 1.0, 0.0))
    spec.add_three_point_arc(np.array([0.5, 2.0, 0.0]), np.array([1.0, 1.5, 0.0]), True)
    spec.add_three_point_arc(np.array([0.5, 3.0, 0.0]), np.array([0.0, 2.5, 0.0]), False)
    spec.add_three_point_arc_with_orientation_target(cumotion.Pose3(R0, np.array([0.5, 4.0, 0.0])),
                                                     np.array([0.0, 3.5, 0.0]))
    spec.add_tangent_arc(np.array([2.0, 2.0, 0.0]), True)
    spec.add_tangent_arc(np.array([5.0, 1.0, 0.0]), False)
    spec.add_tangent_arc_with_orientation_target(cumotion.Pose3(R0, np.array([6.0, 2.0, 0.0])))

    return spec


def test_load_task_space_path_spec_from_file():
    """Test `load_task_space_path_spec_from_file()` with all supported path types."""
    # Load path from file.
    filepath = get_abs_filepath_for_path_spec('task_space_path_specs/valid_path_spec.yaml')
    path_spec_from_file = cumotion.load_task_space_path_spec_from_file(filepath)
    path_from_file = path_spec_from_file.generate_path()

    # Generate equivalent path procedurally.
    expected_path = build_example_task_space_path_spec_procedurally().generate_path()

    # Expect paths to have equivalent domains.
    assert expected_path.domain().lower == path_from_file.domain().lower
    assert expected_path.domain().upper == path_from_file.domain().upper

    # Test that paths have equivalent poses throughout.
    for s in np.linspace(expected_path.domain().lower, expected_path.domain().upper, num=1000):
        assert expected_path.eval(s).matrix() == pytest.approx(path_from_file.eval(s).matrix())


def test_load_task_space_path_spec_from_memory():
    """Test `load_task_space_path_spec_from_memory()`."""
    # Load path from memory.
    filepath = get_abs_filepath_for_path_spec('task_space_path_specs/valid_path_spec.yaml')
    yaml_string = load_as_string(filepath)
    path_spec_from_memory = cumotion.load_task_space_path_spec_from_memory(yaml_string)
    path_from_memory = path_spec_from_memory.generate_path()

    # Generate equivalent path from file.
    path_spec_from_file = cumotion.load_task_space_path_spec_from_file(filepath)
    path_from_file = path_spec_from_file.generate_path()

    # Expect paths to have equivalent poses.
    assert path_from_file.domain().lower == path_from_memory.domain().lower
    assert path_from_file.domain().upper == path_from_memory.domain().upper

    # Test that paths have equivalent poses throughout.
    for s in np.linspace(path_from_file.domain().lower, path_from_file.domain().upper, num=1000):
        assert path_from_file.eval(s).matrix() == pytest.approx(path_from_memory.eval(s).matrix())


def build_example_cspace_path_spec_procedurally():
    """Create an example c-space path."""
    # Create `cspace_path_spec`.
    spec = cumotion.create_cspace_path_spec(np.array([1.0, 2.0, 3.0]))

    # Add waypoints.
    spec.add_cspace_waypoint(np.array([4.0, 5.0, 6.0]))
    spec.add_cspace_waypoint(np.array([7.0, 8.0, 9.0]))
    spec.add_cspace_waypoint(np.array([8.0, 7.0, 6.0]))
    spec.add_cspace_waypoint(np.array([5.0, 4.0, 3.0]))
    spec.add_cspace_waypoint(np.array([2.0, 1.0, 0.0]))

    return spec


def test_load_cspace_path_spec_from_file():
    """Test `load_cspace_path_spec_from_file()` with valid YAML file."""
    # Load path from file.
    filepath = get_abs_filepath_for_path_spec('cspace_path_specs/valid_path_spec.yaml')
    path_spec_from_file = cumotion.load_cspace_path_spec_from_file(filepath)
    path_from_file = cumotion.create_linear_cspace_path(path_spec_from_file)

    # Generate equivalent path procedurally.
    expected_path = cumotion.create_linear_cspace_path(
        build_example_cspace_path_spec_procedurally())

    # Expect paths to have equivalent domains.
    assert expected_path.domain().lower == path_from_file.domain().lower
    assert expected_path.domain().upper == path_from_file.domain().upper

    # Expect paths to have equivalent waypoints.
    assert len(expected_path.waypoints()) == len(path_from_file.waypoints())
    for expected_waypoint, waypoint_from_file in zip(expected_path.waypoints(),
                                                     path_from_file.waypoints()):
        assert expected_waypoint == pytest.approx(waypoint_from_file)

    # Test that paths have equivalent positions throughout.
    for s in np.linspace(expected_path.domain().lower, expected_path.domain().upper, num=1000):
        assert expected_path.eval(s) == pytest.approx(path_from_file.eval(s))


def test_load_cspace_path_spec_from_memory():
    """Test `load_cspace_path_spec_from_memory()` with valid YAML file."""
    # Load path from memory.
    filepath = get_abs_filepath_for_path_spec('cspace_path_specs/valid_path_spec.yaml')
    yaml_string = load_as_string(filepath)
    path_spec_from_memory = cumotion.load_cspace_path_spec_from_memory(yaml_string)
    path_from_memory = cumotion.create_linear_cspace_path(path_spec_from_memory)

    # Generate equivalent path from file.
    path_spec_from_file = cumotion.load_cspace_path_spec_from_file(filepath)
    path_from_file = cumotion.create_linear_cspace_path(path_spec_from_file)

    # Expect paths to have equivalent domains.
    assert path_from_file.domain().lower == path_from_memory.domain().lower
    assert path_from_file.domain().upper == path_from_memory.domain().upper

    # Expect paths to have equivalent waypoints.
    assert len(path_from_file.waypoints()) == len(path_from_memory.waypoints())
    for waypoint_from_file, waypoint_from_memory in zip(path_from_file.waypoints(),
                                                        path_from_memory.waypoints()):
        assert waypoint_from_file == pytest.approx(waypoint_from_memory)

    # Test that paths have equivalent positions throughout.
    for s in np.linspace(path_from_file.domain().lower, path_from_file.domain().upper, num=1000):
        assert path_from_file.eval(s) == pytest.approx(path_from_memory.eval(s))


def build_example_composite_path_procedurally():
    """Create an example composite path specification."""
    # Create composite path specification.
    composite_spec = cumotion.create_composite_path_spec(np.array([1.0, 2.0, 3.0]))

    # Create `cspace_pathSpec` and add to `composite_spec`.
    spec_0 = cumotion.create_cspace_path_spec(np.array([1.0, 2.0, 3.0]))
    spec_0.add_cspace_waypoint(np.array([4.0, 5.0, 6.0]))
    spec_0.add_cspace_waypoint(np.array([7.0, 8.0, 9.0]))
    spec_0.add_cspace_waypoint(np.array([8.0, 7.0, 6.0]))
    spec_0.add_cspace_waypoint(np.array([5.0, 4.0, 3.0]))
    spec_0.add_cspace_waypoint(np.array([2.0, 1.0, 0.0]))
    composite_spec.add_cspace_path_spec(spec_0, cumotion.CompositePathSpec.TransitionMode.SKIP)

    # Create `task_space_pathSpec` and add to `composite_spec`.
    spec_1 = cumotion.create_task_space_path_spec(cumotion.Pose3.identity())
    R0 = cumotion.Rotation3.from_axis_angle(np.array([0.0, 0.0, 1.0]), math.pi)
    spec_1.add_linear_path(cumotion.Pose3(R0, np.array([0.5, 0.0, 0.0])))
    spec_1.add_translation(np.array([0.5, 1.0, 0.0]))
    spec_1.add_rotation(cumotion.Rotation3.from_axis_angle(np.array([0.0, 1.0, 0.0]), math.pi))
    composite_spec.add_task_space_path_spec(
        spec_1,
        cumotion.CompositePathSpec.TransitionMode.LINEAR_TASK_SPACE)

    # Create `cspace_pathSpec` and add to `composite_spec`.
    spec_2 = cumotion.create_cspace_path_spec(np.array([9.0, 8.0, 7.0]))
    spec_2.add_cspace_waypoint(np.array([6.0, 5.0, 4.0]))
    spec_2.add_cspace_waypoint(np.array([3.0, 2.0, 1.0]))
    composite_spec.add_cspace_path_spec(spec_2, cumotion.CompositePathSpec.TransitionMode.FREE)

    return composite_spec


def test_load_composite_path_spec_from_file():
    """Test `load_composite_path_spec_from_file()` with valid YAML file."""
    # Load path from file.
    filepath = get_abs_filepath_for_path_spec('composite_path_specs/valid_path_spec.yaml')
    path_spec_from_file = cumotion.load_composite_path_spec_from_file(filepath)

    # Generate equivalent path spec procedurally.
    expected_path_spec = build_example_composite_path_procedurally()

    # Expect specs to have equivalent numbers of c-space coordinates.
    assert expected_path_spec.num_cspace_coords() == path_spec_from_file.num_cspace_coords()

    # Expect specs to have equivalent numbers of path specifications.
    assert expected_path_spec.num_path_specs() == path_spec_from_file.num_path_specs()

    for i in range(expected_path_spec.num_path_specs()):
        # Expect paths to have identical path types.
        assert expected_path_spec.path_spec_type(i) == path_spec_from_file.path_spec_type(i)

        # Expect each path spec to be identical.
        if expected_path_spec.path_spec_type(i) == cumotion.CompositePathSpec.PathSpecType.CSPACE:
            expected_cspace_path = cumotion.create_linear_cspace_path(
                expected_path_spec.cspace_path_spec(i))
            cspace_path_from_file = cumotion.create_linear_cspace_path(
                path_spec_from_file.cspace_path_spec(i))

            assert expected_cspace_path.domain().lower == cspace_path_from_file.domain().lower
            assert expected_cspace_path.domain().upper == cspace_path_from_file.domain().upper
            assert expected_cspace_path.path_length() == cspace_path_from_file.path_length()
            assert expected_cspace_path.min_position() == \
                   pytest.approx(cspace_path_from_file.min_position())
            assert expected_cspace_path.max_position() == \
                   pytest.approx(cspace_path_from_file.max_position())
            assert len(expected_cspace_path.waypoints()) == len(cspace_path_from_file.waypoints())
            for expected_waypoint, waypoint_from_file in zip(expected_cspace_path.waypoints(),
                                                             cspace_path_from_file.waypoints()):
                assert expected_waypoint == pytest.approx(waypoint_from_file)
        else:
            assert expected_path_spec.path_spec_type(i) == \
                   cumotion.CompositePathSpec.PathSpecType.TASK_SPACE
            expected_task_space_path = expected_path_spec.task_space_path_spec(i).generate_path()
            task_space_path_from_file = path_spec_from_file.task_space_path_spec(i).generate_path()

            assert expected_task_space_path.domain().lower == \
                   task_space_path_from_file.domain().lower
            assert expected_task_space_path.domain().upper == \
                   task_space_path_from_file.domain().upper
            assert expected_task_space_path.path_length() == task_space_path_from_file.path_length()
            assert expected_task_space_path.accumulated_rotation() == \
                   task_space_path_from_file.accumulated_rotation()
            assert expected_task_space_path.min_position() == \
                   pytest.approx(task_space_path_from_file.min_position())
            assert expected_task_space_path.max_position() == \
                   pytest.approx(task_space_path_from_file.max_position())


def test_load_composite_path_spec_from_memory():
    """Test `load_composite_path_spec_from_memory()` with valid YAML file."""
    # Load path from memory.
    filepath = get_abs_filepath_for_path_spec('composite_path_specs/valid_path_spec.yaml')
    yaml_string = load_as_string(filepath)
    path_spec_from_memory = cumotion.load_composite_path_spec_from_memory(yaml_string)

    # Generate equivalent path from file.
    path_spec_from_file = cumotion.load_composite_path_spec_from_file(filepath)

    # Expect specs to have equivalent numbers of c-space coordinates.
    assert path_spec_from_file.num_cspace_coords() == path_spec_from_memory.num_cspace_coords()

    # Expect specs to have equivalent numbers of path specifications.
    assert path_spec_from_file.num_path_specs() == path_spec_from_memory.num_path_specs()

    for i in range(path_spec_from_file.num_path_specs()):
        # Expect paths to have identical path types.
        assert path_spec_from_file.path_spec_type(i) == path_spec_from_memory.path_spec_type(i)

        # Expect each path spec to be identical.
        if path_spec_from_file.path_spec_type(i) == cumotion.CompositePathSpec.PathSpecType.CSPACE:
            cspace_path_from_file = cumotion.create_linear_cspace_path(
                path_spec_from_file.cspace_path_spec(i))
            cspace_path_from_memory = cumotion.create_linear_cspace_path(
                path_spec_from_memory.cspace_path_spec(i))

            assert cspace_path_from_file.domain().lower == cspace_path_from_memory.domain().lower
            assert cspace_path_from_file.domain().upper == cspace_path_from_memory.domain().upper
            assert cspace_path_from_file.path_length() == cspace_path_from_memory.path_length()
            assert cspace_path_from_file.min_position() == \
                   pytest.approx(cspace_path_from_memory.min_position())
            assert cspace_path_from_file.max_position() == \
                   pytest.approx(cspace_path_from_memory.max_position())
            assert len(cspace_path_from_file.waypoints()) == \
                   len(cspace_path_from_memory.waypoints())
            for waypoint_from_file, waypoint_from_memory in \
                    zip(cspace_path_from_file.waypoints(), cspace_path_from_memory.waypoints()):
                assert waypoint_from_file == pytest.approx(waypoint_from_memory)
        else:
            assert path_spec_from_file.path_spec_type(i) == \
                   cumotion.CompositePathSpec.PathSpecType.TASK_SPACE
            task_space_path_from_file = path_spec_from_file.task_space_path_spec(i).generate_path()
            task_space_path_from_memory = \
                path_spec_from_file.task_space_path_spec(i).generate_path()

            assert task_space_path_from_file.domain().lower == \
                   task_space_path_from_memory.domain().lower
            assert task_space_path_from_file.domain().upper == \
                   task_space_path_from_memory.domain().upper
            assert task_space_path_from_file.path_length() == \
                   task_space_path_from_memory.path_length()
            assert task_space_path_from_file.accumulated_rotation() == \
                   task_space_path_from_memory.accumulated_rotation()
            assert task_space_path_from_file.min_position() == \
                   pytest.approx(task_space_path_from_memory.min_position())
            assert task_space_path_from_file.max_position() == \
                   pytest.approx(task_space_path_from_memory.max_position())


def test_export_task_space_path_spec_to_memory():
    """Test `export_task_space_path_spec_to_memory()`."""
    spec = build_example_task_space_path_spec_procedurally()
    exported_spec_string = cumotion.export_task_space_path_spec_to_memory(spec)

    filepath = get_abs_filepath_for_path_spec('task_space_path_specs/reference_exported_spec.yaml')
    expected_spec_string = load_as_string(filepath)

    # Strip off manually-added license header from reference file.
    expected_spec_string = remove_blank_lines_and_comments(expected_spec_string)

    assert expected_spec_string == exported_spec_string


def test_export_cspace_path_spec_to_memory():
    """Test `export_cspace_path_spec_to_memory()`."""
    spec = build_example_cspace_path_spec_procedurally()
    exported_spec_string = cumotion.export_cspace_path_spec_to_memory(spec)

    filepath = get_abs_filepath_for_path_spec('cspace_path_specs/reference_exported_spec.yaml')
    expected_spec_string = load_as_string(filepath)

    # Strip off manually-added license header from reference file.
    expected_spec_string = remove_blank_lines_and_comments(expected_spec_string)

    assert expected_spec_string == exported_spec_string


def test_export_composite_path_spec_to_memory():
    """Test `export_composite_path_spec_to_memory()`."""
    spec = build_example_composite_path_procedurally()
    exported_spec_string = cumotion.export_composite_path_spec_to_memory(spec)

    filepath = get_abs_filepath_for_path_spec('composite_path_specs/reference_exported_spec.yaml')
    expected_spec_string = load_as_string(filepath)

    # Strip off manually-added license header from reference file.
    expected_spec_string = remove_blank_lines_and_comments(expected_spec_string)

    assert expected_spec_string == exported_spec_string
