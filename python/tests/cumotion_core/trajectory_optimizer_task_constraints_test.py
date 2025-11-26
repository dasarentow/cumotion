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

"""Unit tests for task constraints from trajectory_optimizer_python.h."""

# Third Party
import numpy as np
import pytest

# cuMotion
import cumotion


def test_translation_constraint_target():
    """Test construction of a `TranslationConstraint` using `target()`."""
    # Set arbitrary target.
    target = np.array([3.0, 7.0, -9.0])

    # Set deviation limits for testing.
    zero_limit = 0.0
    negative_limit = -10.0
    positive_limit = 10.0

    # Expect that a translation constraint can be created with no deviation limit.
    cumotion.TrajectoryOptimizer.TranslationConstraint.target(target)

    # Expect that a translation constraint can be created with a zero deviation limit.
    cumotion.TrajectoryOptimizer.TranslationConstraint.target(target, zero_limit)

    # Expect that a translation constraint can be created with a positive deviation limit.
    cumotion.TrajectoryOptimizer.TranslationConstraint.target(target, positive_limit)

    # Expect *failure* to create a translation constraint with a negative deviation limit.
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.TranslationConstraint.target(target, negative_limit)


def test_translation_constraint_linear_path_constraint():
    """Test construction of a `TranslationConstraint` using `linear_path_constraint()`."""
    # Set arbitrary target.
    target = np.array([3.0, 7.0, -9.0])

    # Set deviation limits for testing.
    zero_limit = 0.0
    negative_limit = -10.0
    positive_limit = 10.0
    more_positive_limit = 2.0 * positive_limit

    # Expect that a linear path translation constraint can be created with no deviation limits.
    cumotion.TrajectoryOptimizer.TranslationConstraint.linear_path_constraint(target)

    # Expect that a linear path translation constraint can be created with zero deviation limits.
    cumotion.TrajectoryOptimizer.TranslationConstraint.linear_path_constraint(
        target, zero_limit, zero_limit)

    # Expect that a linear path translation constraint can be created with positive deviation
    # limits.
    cumotion.TrajectoryOptimizer.TranslationConstraint.linear_path_constraint(
        target, positive_limit, positive_limit)

    # Expect *failure* to create a linear path translation constraint with a negative path
    # deviation limit.
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.TranslationConstraint.linear_path_constraint(
            target, negative_limit, positive_limit)
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.TranslationConstraint.linear_path_constraint(
            target, negative_limit, None)

    # Expect *failure* to create a linear path translation constraint with a negative terminal
    # deviation limit.
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.TranslationConstraint.linear_path_constraint(
            target, positive_limit, negative_limit)

    # Expect *failure* to create a linear path translation constraint with a path deviation limit
    # that is more restrictive than the terminal deviation limit.
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.TranslationConstraint.linear_path_constraint(
            target, positive_limit, more_positive_limit)

    # Expect *failure* to create a linear path translation constraint with a terminal deviation
    # limit specified, but no path deviation limit.
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.TranslationConstraint.linear_path_constraint(
            target, None, positive_limit)


def test_translation_constraint_goalset_target():
    """Test construction of a `TranslationConstraintGoalset` using `target()`."""
    # Set arbitrary targets.
    targets = [np.array([3.0, 7.0, -9.0]), np.array([2.0, 1.0, 8.0])]

    # Set deviation limits for testing.
    zero_limit = 0.0
    negative_limit = -10.0
    positive_limit = 10.0

    # Expect that translation constraints can be created with no deviation limit.
    cumotion.TrajectoryOptimizer.TranslationConstraintGoalset.target(targets)

    # Expect that translation constraints can be created with a zero deviation limit.
    cumotion.TrajectoryOptimizer.TranslationConstraintGoalset.target(targets, zero_limit)

    # Expect that translation constraints can be created with a positive deviation limit.
    cumotion.TrajectoryOptimizer.TranslationConstraintGoalset.target(targets, positive_limit)

    # Expect *failure* to create translation constraints with a negative deviation limit.
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.TranslationConstraintGoalset.target(targets, negative_limit)

    # Expect *failure* to create translation constraints with empty translation targets.
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.TranslationConstraintGoalset.target([])


def test_translation_constraint_goalset_linear_path_constraint():
    """Test construction of a `TranslationConstraintGoalset` using `linear_path_constraint()`."""
    targets = [np.array([3.0, 7.0, -9.0]), np.array([2.0, 1.0, 8.0])]

    # Set deviation limits for testing.
    zero_limit = 0.0
    negative_limit = -10.0
    positive_limit = 10.0
    more_positive_limit = 2.0 * positive_limit

    # Expect that linear path translation constraints can be created with no deviation limits.
    cumotion.TrajectoryOptimizer.TranslationConstraintGoalset.linear_path_constraint(targets)

    # Expect that linear path translation constraints can be created with zero deviation limits.
    cumotion.TrajectoryOptimizer.TranslationConstraintGoalset.linear_path_constraint(
        targets, zero_limit, zero_limit)

    # Expect that linear path translation constraints can be created with positive deviation limits.
    cumotion.TrajectoryOptimizer.TranslationConstraintGoalset.linear_path_constraint(
        targets, positive_limit, positive_limit)

    # Expect *failure* to create linear path translation constraints with a negative path deviation
    # limit.
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.TranslationConstraintGoalset.linear_path_constraint(
            targets, negative_limit, positive_limit)
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.TranslationConstraintGoalset.linear_path_constraint(
            targets, negative_limit, None)

    # Expect *failure* to create linear path translation constraints with a negative terminal
    # deviation limit.
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.TranslationConstraintGoalset.linear_path_constraint(
            targets, positive_limit, negative_limit)

    # Expect *failure* to create linear path translation constraints with a path deviation limit
    # that is more restrictive than the terminal deviation limit.
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.TranslationConstraintGoalset.linear_path_constraint(
            targets, positive_limit, more_positive_limit)

    # Expect *failure* to create linear path translation constraints with a terminal deviation
    # limit specified, but no path deviation limit.
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.TranslationConstraintGoalset.linear_path_constraint(
            targets, None, positive_limit)

    # Expect *failure* to create linear path translation constraints with empty translation targets.
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.TranslationConstraintGoalset.linear_path_constraint([])


def test_orientation_constraint_none():
    """Test construction of an `OrientationConstraint` using `none()`."""
    cumotion.TrajectoryOptimizer.OrientationConstraint.none()


def test_orientation_constraint_constant():
    """Test construction of an `OrientationConstraint` using `constant()`."""
    # Set deviation limits for testing.
    zero_limit = 0.0
    negative_limit = -1.0
    positive_limit = 1.0
    larger_positive_limit = 2.0 * positive_limit
    largest_positive_limit = 10.0 * positive_limit

    # Expect that a constant orientation constraint can be created with no deviation limits.
    cumotion.TrajectoryOptimizer.OrientationConstraint.constant()

    # Expect that a constant orientation constraint can be created with zero deviation limits.
    cumotion.TrajectoryOptimizer.OrientationConstraint.constant(zero_limit, zero_limit)

    # Expect that a constant orientation constraint can be created with positive deviation limits.
    cumotion.TrajectoryOptimizer.OrientationConstraint.constant(positive_limit, positive_limit)

    # Expect *failure* to create a constant orientation constraint with a negative path deviation
    # limit.
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.OrientationConstraint.constant(negative_limit, positive_limit)
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.OrientationConstraint.constant(negative_limit, None)

    # Expect *warning* when creating a constant orientation constraint with a path deviation limit
    # greater than pi.
    cumotion.TrajectoryOptimizer.OrientationConstraint.constant(
        largest_positive_limit, positive_limit)
    cumotion.TrajectoryOptimizer.OrientationConstraint.constant(largest_positive_limit, None)

    # Expect *failure* to create a constant orientation constraint with a negative terminal
    # deviation limit.
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.OrientationConstraint.constant(positive_limit, negative_limit)

    # Expect *warning* when creating a constant orientation constraint with a terminal deviation
    # limit greater than pi.
    cumotion.TrajectoryOptimizer.OrientationConstraint.constant(
        largest_positive_limit, largest_positive_limit)

    # Expect *failure* to create a constant orientation constraint with a path deviation limit
    # that is more restrictive than the terminal deviation limit.
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.OrientationConstraint.constant(
            positive_limit, larger_positive_limit)

    # Expect *failure* to create a constant orientation constraint with a terminal deviation
    # limit specified, but no path deviation limit.
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.OrientationConstraint.constant(None, positive_limit)


def test_orientation_constraint_terminal_target():
    """Test construction of an `OrientationConstraint` using `terminal_target()`."""
    # Set arbitrary orientation target.
    target = cumotion.Rotation3.identity()

    # Set deviation limits for testing.
    zero_limit = 0.0
    negative_limit = -1.0
    positive_limit = 1.0
    large_positive_limit = 6.0

    # Expect that a terminal target orientation constraint can be created with no deviation limits.
    cumotion.TrajectoryOptimizer.OrientationConstraint.terminal_target(target)

    # Expect that a terminal target orientation constraint can be created with 0 deviation limit.
    cumotion.TrajectoryOptimizer.OrientationConstraint.terminal_target(target, zero_limit)

    # Expect that a terminal target orientation constraint can be created with a positive
    # deviation limit.
    cumotion.TrajectoryOptimizer.OrientationConstraint.terminal_target(target, positive_limit)

    # Expect *failure* to create a terminal target orientation constraint with a negative deviation
    # limit.
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.OrientationConstraint.terminal_target(target, negative_limit)

    # Expect *warning* when creating a terminal target orientation constraint with a deviation limit
    # greater than pi.
    cumotion.TrajectoryOptimizer.OrientationConstraint.terminal_target(
        target, large_positive_limit)


def test_orientation_constraint_terminal_and_path_target():
    """Test construction of an `OrientationConstraint` using `terminal_and_path_target()`."""
    # Set arbitrary orientation target.
    target = cumotion.Rotation3.identity()

    # Set deviation limits for testing.
    zero_limit = 0.0
    negative_limit = -1.0
    positive_limit = 1.0
    larger_positive_limit = 2.0 * positive_limit
    largest_positive_limit = 10.0 * positive_limit

    # Expect that a target orientation constraint can be created with no deviation limits.
    cumotion.TrajectoryOptimizer.OrientationConstraint.terminal_and_path_target(target)

    # Expect that a target orientation constraint can be created with zero deviation limits.
    cumotion.TrajectoryOptimizer.OrientationConstraint.terminal_and_path_target(
        target, zero_limit, zero_limit)

    # Expect that a target orientation constraint can be created with positive deviation limits.
    cumotion.TrajectoryOptimizer.OrientationConstraint.terminal_and_path_target(
        target, positive_limit, positive_limit)

    # Expect that a target orientation constraint can be created with a terminal deviation limit,
    # but no path deviation limit.
    cumotion.TrajectoryOptimizer.OrientationConstraint.terminal_and_path_target(
        target, None, positive_limit)

    # Expect *failure* to create a target orientation constraint with a negative path deviation
    # limit.
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.OrientationConstraint.terminal_and_path_target(
            target, negative_limit, positive_limit)
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.OrientationConstraint.terminal_and_path_target(
            target, negative_limit, None)

    # Expect *warning* when creating a target orientation constraint with a path deviation limit
    # greater than pi.
    cumotion.TrajectoryOptimizer.OrientationConstraint.terminal_and_path_target(
        target, largest_positive_limit, positive_limit)
    cumotion.TrajectoryOptimizer.OrientationConstraint.terminal_and_path_target(
        target, largest_positive_limit, None)

    # Expect *failure* to create a target orientation constraint with a negative terminal
    # deviation limit.
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.OrientationConstraint.terminal_and_path_target(
            target, positive_limit, negative_limit)

    # Expect *warning* when creating a target orientation constraint with a terminal deviation limit
    # greater than pi.
    cumotion.TrajectoryOptimizer.OrientationConstraint.terminal_and_path_target(
        target, largest_positive_limit, largest_positive_limit)

    # Expect *failure* to create a target orientation constraint with a path deviation limit
    # that is more restrictive than the terminal deviation limit.
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.OrientationConstraint.terminal_and_path_target(
            target, positive_limit, larger_positive_limit)


def test_orientation_constraint_terminal_axis():
    """Test construction of an `OrientationConstraint` using `terminal_axis()`."""
    # Set arbitrary, non-normalized target axes.
    tool_frame_axis = np.array([5.0, 0.0, 0.0])
    world_target_axis = np.array([0.0, 3.0, 0.0])

    # Set deviation limits for testing.
    zero_limit = 0.0
    negative_limit = -1.0
    positive_limit = 1.0
    large_positive_limit = 6.0

    # Expect that a terminal axis orientation constraint can be created with no deviation limit.
    cumotion.TrajectoryOptimizer.OrientationConstraint.terminal_axis(
        tool_frame_axis, world_target_axis)

    # Expect that a terminal axis orientation constraint can be created with a zero deviation limit.
    cumotion.TrajectoryOptimizer.OrientationConstraint.terminal_axis(
        tool_frame_axis, world_target_axis, zero_limit)

    # Expect that a terminal axis orientation constraint can be created with a positive deviation
    # limit.
    cumotion.TrajectoryOptimizer.OrientationConstraint.terminal_axis(
        tool_frame_axis, world_target_axis, positive_limit)

    # Expect *failure* to create a terminal axis orientation constraint with a negative deviation
    # limit.
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.OrientationConstraint.terminal_axis(
            tool_frame_axis, world_target_axis, negative_limit)

    # Expect *warning* when creating a terminal axis orientation constraint with a deviation limit
    # greater than pi.
    cumotion.TrajectoryOptimizer.OrientationConstraint.terminal_axis(
        tool_frame_axis, world_target_axis, large_positive_limit)

    # Expect *failure* to create a terminal axis orientation constraint with a zero or nearly zero
    # tool frame axis.
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.OrientationConstraint.terminal_axis(
            np.array([0.0, 0.0, 0.0]), world_target_axis)
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.OrientationConstraint.terminal_axis(
            1e-10 * np.array([1.0, 0.0, 0.0]), world_target_axis)

    # Expect *failure* to create a terminal axis orientation constraint with a zero or nearly zero
    # world target axis.
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.OrientationConstraint.terminal_axis(
            tool_frame_axis, np.array([0.0, 0.0, 0.0]))
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.OrientationConstraint.terminal_axis(
            tool_frame_axis, 1e-10 * np.array([1.0, 0.0, 0.0]))


def test_orientation_constraint_terminal_and_path_axis():
    """Test construction of an `OrientationConstraint` using `terminal_and_path_axis()`."""
    # Set arbitrary, non-normalized target axes.
    tool_frame_axis = np.array([5.0, 0.0, 0.0])
    world_target_axis = np.array([0.0, 3.0, 0.0])

    # Set deviation limits for testing.
    zero_limit = 0.0
    negative_limit = -1.0
    positive_limit = 1.0
    larger_positive_limit = 2.0 * positive_limit
    largest_positive_limit = 10.0 * positive_limit

    # Expect that a terminal and path axis orientation constraint can be created with no deviation
    # limits.
    cumotion.TrajectoryOptimizer.OrientationConstraint.terminal_and_path_axis(
        tool_frame_axis, world_target_axis)

    # Expect that a terminal and path axis orientation constraint can be created with zero
    # deviation limits.
    cumotion.TrajectoryOptimizer.OrientationConstraint.terminal_and_path_axis(
        tool_frame_axis, world_target_axis, zero_limit, zero_limit)

    # Expect that a terminal and path axis orientation constraint can be created with positive
    # deviation limits.
    cumotion.TrajectoryOptimizer.OrientationConstraint.terminal_and_path_axis(
        tool_frame_axis, world_target_axis, positive_limit, positive_limit)

    # Expect that a terminal and path axis orientation constraint can be created with a terminal
    # deviation limit, but no path deviation limit.
    cumotion.TrajectoryOptimizer.OrientationConstraint.terminal_and_path_axis(
        tool_frame_axis, world_target_axis, None, positive_limit)

    # Expect *failure* to create a terminal and path axis orientation constraint with a negative
    # path deviation limit.
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.OrientationConstraint.terminal_and_path_axis(
            tool_frame_axis, world_target_axis, negative_limit, positive_limit)
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.OrientationConstraint.terminal_and_path_axis(
            tool_frame_axis, world_target_axis, negative_limit, None)

    # Expect *warning* when creating a terminal and path axis orientation constraint with a path
    # deviation limit greater than pi.
    cumotion.TrajectoryOptimizer.OrientationConstraint.terminal_and_path_axis(
        tool_frame_axis, world_target_axis, largest_positive_limit, positive_limit)
    cumotion.TrajectoryOptimizer.OrientationConstraint.terminal_and_path_axis(
        tool_frame_axis, world_target_axis, largest_positive_limit, None)

    # Expect *failure* to create a terminal and path axis orientation constraint with a negative
    # terminal deviation limit.
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.OrientationConstraint.terminal_and_path_axis(
            tool_frame_axis, world_target_axis, positive_limit, negative_limit)

    # Expect *warning* when creating a terminal and path axis orientation constraint with a terminal
    # deviation limit greater than pi.
    cumotion.TrajectoryOptimizer.OrientationConstraint.terminal_and_path_axis(
        tool_frame_axis, world_target_axis, largest_positive_limit, largest_positive_limit)

    # Expect *failure* to create a terminal and path axis orientation constraint with a path
    # deviation limit that is more restrictive than the terminal deviation limit.
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.OrientationConstraint.terminal_and_path_axis(
            tool_frame_axis, world_target_axis, positive_limit, larger_positive_limit)

    # Expect *failure* to create a terminal and path axis orientation constraint with a zero or
    # nearly zero tool frame axis.
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.OrientationConstraint.terminal_and_path_axis(
            np.array([0.0, 0.0, 0.0]), world_target_axis)
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.OrientationConstraint.terminal_and_path_axis(
            1e-10 * np.array([1.0, 0.0, 0.0]), world_target_axis)

    # Expect *failure* to create a terminal and path axis orientation constraint with a zero or
    # nearly zero world target axis.
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.OrientationConstraint.terminal_and_path_axis(
            tool_frame_axis, np.array([0.0, 0.0, 0.0]))
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.OrientationConstraint.terminal_and_path_axis(
            tool_frame_axis, 1e-10 * np.array([1.0, 0.0, 0.0]))


def test_orientation_constraint_terminal_target_and_path_axis():
    """Test construction of an `OrientationConstraint` using `terminal_target_and_path_axis()`."""
    # Set arbitrary orientation target.
    target = cumotion.Rotation3.identity()

    # Set arbitrary, non-normalized target axes.
    tool_frame_axis = np.array([5.0, 0.0, 0.0])
    world_target_axis = np.array([0.0, 3.0, 0.0])

    # Set deviation limits for testing.
    zero_limit = 0.0
    negative_limit = -1.0
    positive_limit = 1.0
    large_positive_limit = 10.0 * positive_limit

    # Expect that a terminal target and path axis orientation constraint can be created with no
    # deviation limits.
    cumotion.TrajectoryOptimizer.OrientationConstraint.terminal_target_and_path_axis(
        target, tool_frame_axis, world_target_axis)

    # Expect that a terminal target and path axis orientation constraint can be created with zero
    # deviation limits.
    cumotion.TrajectoryOptimizer.OrientationConstraint.terminal_target_and_path_axis(
        target, tool_frame_axis, world_target_axis, zero_limit, zero_limit)

    # Expect that a terminal target and path axis orientation constraint can be created with
    # a terminal deviation limit greater than the path deviation limit.
    smaller_limit = 0.5 * positive_limit
    cumotion.TrajectoryOptimizer.OrientationConstraint.terminal_target_and_path_axis(
        target, tool_frame_axis, world_target_axis, positive_limit, smaller_limit)

    # Expect that a terminal target and path axis orientation constraint can be created with
    # a terminal deviation limit, but no path deviation limit.
    cumotion.TrajectoryOptimizer.OrientationConstraint.terminal_target_and_path_axis(
        target, tool_frame_axis, world_target_axis, positive_limit, None)

    # Expect *failure* to create a terminal target and path axis orientation constraint with a
    # negative path deviation limit.
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.OrientationConstraint.terminal_target_and_path_axis(
            target, tool_frame_axis, world_target_axis, positive_limit, negative_limit)
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.OrientationConstraint.terminal_target_and_path_axis(
            target, tool_frame_axis, world_target_axis, None, negative_limit)

    # Expect *warning* when creating a terminal target and path axis orientation constraint with a
    # path deviation limit greater than pi.
    cumotion.TrajectoryOptimizer.OrientationConstraint.terminal_target_and_path_axis(
        target, tool_frame_axis, world_target_axis, positive_limit, large_positive_limit)
    cumotion.TrajectoryOptimizer.OrientationConstraint.terminal_target_and_path_axis(
        target, tool_frame_axis, world_target_axis, None, large_positive_limit)

    # Expect *failure* to create a terminal target and path axis orientation constraint with a
    # negative terminal deviation limit.
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.OrientationConstraint.terminal_target_and_path_axis(
            target, tool_frame_axis, world_target_axis, negative_limit, positive_limit)

    # Expect *warning* when creating a terminal target and path axis orientation constraint with a
    # terminal deviation limit greater than pi.
    cumotion.TrajectoryOptimizer.OrientationConstraint.terminal_target_and_path_axis(
        target, tool_frame_axis, world_target_axis, large_positive_limit, positive_limit)

    # Expect *failure* to create a terminal target and path axis orientation constraint with a
    # zero or nearly zero tool frame axis.
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.OrientationConstraint.terminal_target_and_path_axis(
            target, np.array([0.0, 0.0, 0.0]), world_target_axis)
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.OrientationConstraint.terminal_target_and_path_axis(
            target, 1e-10 * np.array([1.0, 0.0, 0.0]), world_target_axis)

    # Expect *failure* to create a terminal target and path axis orientation constraint with a
    # zero or nearly zero world target axis.
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.OrientationConstraint.terminal_target_and_path_axis(
            target, tool_frame_axis, np.array([0.0, 0.0, 0.0]))
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.OrientationConstraint.terminal_target_and_path_axis(
            target, tool_frame_axis, 1e-10 * np.array([1.0, 0.0, 0.0]))


def test_orientation_constraint_goalset_none():
    """Test construction of an `OrientationConstraintGoalset` using `none()`."""
    cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.none()


def test_orientation_constraint_goalset_constant():
    """Test construction of an `OrientationConstraintGoalset` using `constant()`."""
    # Set deviation limits for testing.
    zero_limit = 0.0
    negative_limit = -1.0
    positive_limit = 1.0
    larger_positive_limit = 2.0 * positive_limit
    largest_positive_limit = 10.0 * positive_limit

    # Expect that constant orientation constraints can be created with no deviation limits.
    cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.constant()

    # Expect that constant orientation constraints can be created with zero deviation limits.
    cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.constant(zero_limit, zero_limit)

    # Expect that constant orientation constraints can be created with positive deviation limits.
    cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.constant(
        positive_limit, positive_limit)

    # Expect *failure* to create constant orientation constraints with a negative path deviation
    # limit.
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.constant(
            negative_limit, positive_limit)
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.constant(negative_limit, None)

    # Expect *warning* when creating constant orientation constraints with a path deviation limit
    # greater than pi.
    cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.constant(
        largest_positive_limit, positive_limit)
    cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.constant(
        largest_positive_limit, None)

    # Expect *failure* to create constant orientation constraints with a negative terminal
    # deviation limit.
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.constant(
            positive_limit, negative_limit)

    # Expect *warning* when creating constant orientation constraints with a terminal deviation
    # limit greater than pi.
    cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.constant(
        largest_positive_limit, largest_positive_limit)

    # Expect *failure* to create constant orientation constraints with a path deviation limit
    # that is more restrictive than the terminal deviation limit.
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.constant(
            positive_limit, larger_positive_limit)

    # Expect *failure* to create constant orientation constraints with a terminal deviation
    # limit specified, but no path deviation limit.
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.constant(None, positive_limit)


def test_orientation_constraint_goalset_terminal_target():
    """Test construction of an `OrientationConstraintGoalset` using `terminal_target()`."""
    # Set arbitrary orientation targets.
    targets = [cumotion.Rotation3.identity(), cumotion.Rotation3.identity()]

    # Set deviation limits for testing.
    zero_limit = 0.0
    negative_limit = -1.0
    positive_limit = 1.0
    large_positive_limit = 6.0

    # Expect that terminal target orientation constraints can be created with no deviation limits.
    cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.terminal_target(targets)

    # Expect that terminal target orientation constraints can be created with a zero deviation
    # limit.
    cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.terminal_target(targets, zero_limit)

    # Expect that terminal target orientation constraints can be created with a positive deviation
    # limit.
    cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.terminal_target(
        targets, positive_limit)

    # Expect *failure* to create terminal target orientation constraints with a negative deviation
    # limit.
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.terminal_target(
            targets, negative_limit)

    # Expect *warning* when creating terminal target orientation constraints with a deviation limit
    # greater than pi.
    cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.terminal_target(
        targets, large_positive_limit)

    # Expect *failure* to create terminal target orientation constraints with empty orientation
    # targets.
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.terminal_target([])


def test_orientation_constraint_goalset_terminal_and_path_target():
    """Test construction of an `OrientationConstraintGoalset` using `terminal_and_path_target()`."""
    # Set arbitrary orientation targets.
    targets = [cumotion.Rotation3.identity(), cumotion.Rotation3.identity()]

    # Set deviation limits for testing.
    zero_limit = 0.0
    negative_limit = -1.0
    positive_limit = 1.0
    larger_positive_limit = 2.0 * positive_limit
    largest_positive_limit = 10.0 * positive_limit

    # Expect that target orientation constraints can be created with no deviation limits.
    cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.terminal_and_path_target(targets)

    # Expect that target orientation constraints can be created with zero deviation limits.
    cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.terminal_and_path_target(
        targets, zero_limit, zero_limit)

    # Expect that target orientation constraints can be created with positive deviation limits.
    cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.terminal_and_path_target(
        targets, positive_limit, positive_limit)

    # Expect that target orientation constraints can be created with terminal deviation limit, but
    # no path deviation limit.
    cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.terminal_and_path_target(
        targets, None, positive_limit)

    # Expect *failure* to create target orientation constraints with a negative path deviation
    # limit.
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.terminal_and_path_target(
            targets, negative_limit, positive_limit)
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.terminal_and_path_target(
            targets, negative_limit, None)

    # Expect *warning* when creating target orientation constraints with a path deviation limit
    # greater than pi.
    cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.terminal_and_path_target(
        targets, largest_positive_limit, positive_limit)
    cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.terminal_and_path_target(
        targets, largest_positive_limit, None)

    # Expect *failure* to create target orientation constraints with a negative terminal
    # deviation limit.
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.terminal_and_path_target(
            targets, positive_limit, negative_limit)

    # Expect *warning* when creating target orientation constraints with a terminal deviation limit
    # greater than pi.
    cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.terminal_and_path_target(
        targets, largest_positive_limit, largest_positive_limit)

    # Expect *failure* to create target orientation constraints with a path deviation limit
    # that is more restrictive than the terminal deviation limit.
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.terminal_and_path_target(
            targets, positive_limit, larger_positive_limit)

    # Expect *failure* to create target orientation constraints with empty orientation targets.
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.terminal_and_path_target([])


def test_orientation_constraint_goalset_terminal_axis():
    """Test construction of an `OrientationConstraintGoalset` using `terminal_axis()`."""
    # Set arbitrary, non-normalized target axes.
    tool_frame_axes = [np.array([5.0, 0.0, 0.0]), np.array([0.0, 0.0, 7.0])]
    world_target_axes = [np.array([0.0, 3.0, 0.0]), np.array([0.0, -2.0, 0.0])]

    # Set zero and nearly zero axes for testing.
    zero_axes = [np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])]
    nearly_zero_axes = [1e-10 * np.array([1.0, 0.0, 0.0]), 1e-10 * np.array([0.0, 1.0, 0.0])]

    # Create a set of too many axes for testing.
    too_many_axes = [np.array([5.0, 0.0, 0.0]), np.array([0.0, 0.0, 7.0]),
                     np.array([0.0, 6.0, 0.0])]

    # Set deviation limits for testing.
    zero_limit = 0.0
    negative_limit = -1.0
    positive_limit = 1.0
    large_positive_limit = 6.0

    # Expect that terminal axis orientation constraints can be created with no deviation limit.
    cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.terminal_axis(
        tool_frame_axes, world_target_axes)

    # Expect that terminal axis orientation constraints can be created with a zero deviation limit.
    cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.terminal_axis(
        tool_frame_axes, world_target_axes, zero_limit)

    # Expect that terminal axis orientation constraints can be created with a positive deviation
    # limit.
    cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.terminal_axis(
        tool_frame_axes, world_target_axes, positive_limit)

    # Expect *failure* to create terminal axis orientation constraints with a negative
    # deviation limit.
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.terminal_axis(
            tool_frame_axes, world_target_axes, negative_limit)

    # Expect *warning* when creating terminal axis orientation constraints with a deviation limit
    # greater than pi.
    cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.terminal_axis(
        tool_frame_axes, world_target_axes, large_positive_limit)

    # Expect *failure* to create terminal axis orientation constraints with a zero or nearly zero
    # tool frame axis.
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.terminal_axis(
            zero_axes, world_target_axes)
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.terminal_axis(
            nearly_zero_axes, world_target_axes)

    # Expect *failure* to create terminal axis orientation constraints with a zero or nearly zero
    # world target axis.
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.terminal_axis(
            tool_frame_axes, zero_axes)
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.terminal_axis(
            tool_frame_axes, nearly_zero_axes)

    # Expect *failure* to create terminal axis orientation constraints with differing numbers of
    # tool frame axes and world target axes.
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.terminal_axis(
            tool_frame_axes, too_many_axes)
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.terminal_axis(
            too_many_axes, world_target_axes)

    # Expect *failure* to create terminal axis orientation constraints with empty tool frame and
    # world target axes.
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.terminal_axis([], [])


def test_orientation_constraint_goalset_terminal_and_path_axis():
    """Test construction of an `OrientationConstraintGoalset` using `terminal_and_path_axis()`."""
    # Set arbitrary, non-normalized target axes.
    tool_frame_axes = [np.array([5.0, 0.0, 0.0]), np.array([0.0, 0.0, 7.0])]
    world_target_axes = [np.array([0.0, 3.0, 0.0]), np.array([0.0, -2.0, 0.0])]

    # Set zero and nearly zero axes for testing.
    zero_axes = [np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])]
    nearly_zero_axes = [1e-10 * np.array([1.0, 0.0, 0.0]), 1e-10 * np.array([0.0, 1.0, 0.0])]

    # Create a set of too many axes for testing.
    too_many_axes = [np.array([5.0, 0.0, 0.0]), np.array([0.0, 0.0, 7.0]),
                     np.array([0.0, 6.0, 0.0])]

    # Set deviation limits for testing.
    zero_limit = 0.0
    negative_limit = -1.0
    positive_limit = 1.0
    larger_positive_limit = 2.0 * positive_limit
    largest_positive_limit = 10.0 * positive_limit

    # Expect that terminal and path axis orientation constraints can be created with no
    # deviation limits.
    cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.terminal_and_path_axis(
        tool_frame_axes, world_target_axes)

    # Expect that terminal and path axis orientation constraints can be created with zero deviation
    # limits.
    cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.terminal_and_path_axis(
        tool_frame_axes, world_target_axes, zero_limit, zero_limit)

    # Expect that terminal and path axis orientation constraints can be created with positive
    # deviation limits.
    cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.terminal_and_path_axis(
        tool_frame_axes, world_target_axes, positive_limit, positive_limit)

    # Expect that terminal and path axis orientation constraints can be created with terminal
    # deviation limit, but no path deviation limit.
    cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.terminal_and_path_axis(
        tool_frame_axes, world_target_axes, None, positive_limit)

    # Expect *failure* to create terminal and path axis orientation constraints with a negative
    # path deviation limit.
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.terminal_and_path_axis(
            tool_frame_axes, world_target_axes, negative_limit, positive_limit)
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.terminal_and_path_axis(
            tool_frame_axes, world_target_axes, negative_limit, None)

    # Expect *warning* when creating terminal and path axis orientation constraints with a path
    # deviation limit greater than pi.
    cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.terminal_and_path_axis(
        tool_frame_axes, world_target_axes, largest_positive_limit, positive_limit)
    cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.terminal_and_path_axis(
        tool_frame_axes, world_target_axes, largest_positive_limit, None)

    # Expect *failure* to create terminal and path axis orientation constraints with a negative
    # terminal deviation limit.
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.terminal_and_path_axis(
            tool_frame_axes, world_target_axes, positive_limit, negative_limit)

    # Expect *warning* when creating terminal and path axis orientation constraints with a terminal
    # deviation limit greater than pi.
    cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.terminal_and_path_axis(
        tool_frame_axes, world_target_axes, largest_positive_limit, largest_positive_limit)

    # Expect *failure* to create terminal and path axis orientation constraints with a path
    # deviation limit that is more restrictive than the terminal deviation limit.
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.terminal_and_path_axis(
            tool_frame_axes, world_target_axes, positive_limit, larger_positive_limit)

    # Expect *failure* to create terminal and path axis orientation constraints with a zero or
    # nearly zero tool frame axis.
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.terminal_and_path_axis(
            zero_axes, world_target_axes)
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.terminal_and_path_axis(
            nearly_zero_axes, world_target_axes)

    # Expect *failure* to create terminal and path axis orientation constraints with a zero or
    # nearly zero world target axis.
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.terminal_and_path_axis(
            tool_frame_axes, zero_axes)
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.terminal_and_path_axis(
            tool_frame_axes, nearly_zero_axes)

    # Expect *failure* to create terminal and path axis orientation constraints with differing
    # numbers of tool frame axes and world target axes.
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.terminal_and_path_axis(
            tool_frame_axes, too_many_axes)
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.terminal_and_path_axis(
            too_many_axes, world_target_axes)

    # Expect *failure* to create terminal and path axis orientation constraints with empty tool
    # frame and world target axes.
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.terminal_and_path_axis([], [])


def test_orientation_constraint_goalset_terminal_target_and_path_axis():
    """Test construction using `OrientationConstraintGoalset.terminal_target_and_path_axis()`."""
    # Set arbitrary orientation targets.
    targets = [cumotion.Rotation3.identity(), cumotion.Rotation3.identity()]

    # Set arbitrary, non-normalized target axes.
    tool_frame_axes = [np.array([5.0, 0.0, 0.0]), np.array([0.0, 0.0, 7.0])]
    world_target_axes = [np.array([0.0, 3.0, 0.0]), np.array([0.0, -2.0, 0.0])]

    # Set zero and nearly zero axes for testing.
    zero_axes = [np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])]
    nearly_zero_axes = [1e-10 * np.array([1.0, 0.0, 0.0]), 1e-10 * np.array([0.0, 1.0, 0.0])]

    # Create a set of too many axes for testing.
    too_many_axes = [np.array([5.0, 0.0, 0.0]), np.array([0.0, 0.0, 7.0]),
                     np.array([0.0, 6.0, 0.0])]

    # Set deviation limits for testing.
    zero_limit = 0.0
    negative_limit = -1.0
    positive_limit = 1.0
    large_positive_limit = 10.0 * positive_limit

    # Expect that terminal target and path axis orientation constraints can be created with no
    # deviation limits.
    cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.terminal_target_and_path_axis(
        targets, tool_frame_axes, world_target_axes)

    # Expect that terminal target and path axis orientation constraints can be created with zero
    # deviation limits.
    cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.terminal_target_and_path_axis(
        targets, tool_frame_axes, world_target_axes, zero_limit, zero_limit)

    # Expect that terminal target and path axis orientation constraints can be created with
    # positive deviation limits.
    cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.terminal_target_and_path_axis(
        targets, tool_frame_axes, world_target_axes, positive_limit, positive_limit)

    # Expect that terminal target and path axis orientation constraints can be created with
    # a terminal deviation limit greater than the path deviation limit.
    smaller_limit = 0.5 * positive_limit
    cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.terminal_target_and_path_axis(
        targets, tool_frame_axes, world_target_axes, positive_limit, smaller_limit)

    # Expect that terminal target and path axis orientation constraints can be created with
    # a terminal deviation limit, but no path deviation limit.
    cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.terminal_target_and_path_axis(
        targets, tool_frame_axes, world_target_axes, positive_limit, None)

    # Expect *failure* to create terminal target and path axis orientation constraints with a
    # negative path deviation limit.
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.terminal_target_and_path_axis(
            targets, tool_frame_axes, world_target_axes, positive_limit, negative_limit)
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.terminal_target_and_path_axis(
            targets, tool_frame_axes, world_target_axes, None, negative_limit)

    # Expect *warning* when creating terminal target and path axis orientation constraints with a
    # path deviation limit greater than pi.
    cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.terminal_target_and_path_axis(
        targets, tool_frame_axes, world_target_axes, positive_limit, large_positive_limit)
    cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.terminal_target_and_path_axis(
        targets, tool_frame_axes, world_target_axes, None, large_positive_limit)

    # Expect *failure* to create terminal target and path axis orientation constraints with a
    # negative terminal deviation limit.
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.terminal_target_and_path_axis(
            targets, tool_frame_axes, world_target_axes, negative_limit, positive_limit)

    # Expect *warning* when creating terminal target and path axis orientation constraints with a
    # terminal deviation limit greater than pi.
    cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.terminal_target_and_path_axis(
        targets, tool_frame_axes, world_target_axes, large_positive_limit, positive_limit)

    # Expect *failure* to create terminal target and path axis orientation constraints with a
    # zero or nearly zero tool frame axis.
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.terminal_target_and_path_axis(
            targets, zero_axes, world_target_axes)
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.terminal_target_and_path_axis(
            targets, nearly_zero_axes, world_target_axes)

    # Expect *failure* to create terminal target and path axis orientation constraints with a
    # zero or nearly zero world target axis.
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.terminal_target_and_path_axis(
            targets, tool_frame_axes, zero_axes)
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.terminal_target_and_path_axis(
            targets, tool_frame_axes, nearly_zero_axes)

    # Expect *failure* to create terminal target and path axis orientation constraints with
    # differing numbers orientation targets and target axes.
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.terminal_target_and_path_axis(
            targets, too_many_axes, too_many_axes)

    # Expect *failure* to create terminal target and path axis orientation constraints with
    # differing numbers of tool frame axes and world target axes.
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.terminal_target_and_path_axis(
            targets, tool_frame_axes, too_many_axes)
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.terminal_target_and_path_axis(
            targets, too_many_axes, world_target_axes)

    # Expect *failure* to create terminal target and path axis orientation constraints with empty
    # orientation targets, tool frame axes, and world target axes.
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.terminal_target_and_path_axis(
            [], [], [])


def test_task_space_target():
    """Test construction of a `TaskSpaceTarget`."""
    # NOTE: Since any valid `TranslationConstraint` and any valid `OrientationConstraint` may be
    # combined to form a `TaskSpaceTarget`, extensive tests are not included here.

    # Create an arbitrary `TranslationConstraint`.
    terminal_limit = 1.0
    translation_target = np.array([3.0, 7.0, -9.0])
    translation_constraint = cumotion.TrajectoryOptimizer.TranslationConstraint.target(
        translation_target, terminal_limit)

    # Create an arbitrary `OrientationConstraint`.
    orientation_target = cumotion.Rotation3.identity()
    orientation_constraint = cumotion.TrajectoryOptimizer.OrientationConstraint.terminal_target(
        orientation_target)

    # Create `TaskSpaceTarget`, expecting no failures.
    cumotion.TrajectoryOptimizer.TaskSpaceTarget(translation_constraint, orientation_constraint)

    # Test that we can create a `TaskSpaceTarget` with default orientation constraint.
    cumotion.TrajectoryOptimizer.TaskSpaceTarget(translation_constraint)


def test_task_space_target_goalset():
    """Test construction of a `TaskSpaceTargetGoalset`."""
    # NOTE: The only invalid combination of a valid `TranslationConstraintGoalset` and a valid
    # `OrientationConstraintGoalset` arises if the number of constraints differs between the two
    # goalsets. Testing is limited to one valid combination and one invalid combination.

    # Create an arbitrary `TranslationConstraintGoalset`.
    translation_targets = [np.array([3.0, 7.0, -9.0]), np.array([1.0, 2.0, -3.0])]
    translation_constraints = cumotion.TrajectoryOptimizer.TranslationConstraintGoalset.target(
        translation_targets)

    # Create an arbitrary `OrientationConstraintGoalset`.
    orientation_constraints = cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.none()

    # Create `TaskSpaceTargetGoalset`, expecting no failures.
    cumotion.TrajectoryOptimizer.TaskSpaceTargetGoalset(
        translation_constraints, orientation_constraints)

    # Test that we can create a `TaskSpaceTargetGoalset` with default orientation constraints.
    cumotion.TrajectoryOptimizer.TaskSpaceTargetGoalset(translation_constraints)

    # Expect *failure* to create a `TaskSpaceTargetGoalset` with an invalid numbers of constraints.
    # Create `orientation_constraints` that are incompatible with `translation_constraints`.
    orientation_targets = [cumotion.Rotation3.identity()]
    orientation_constraints_incompatible = (
        cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.terminal_target(
            orientation_targets))

    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.TaskSpaceTargetGoalset(
            translation_constraints, orientation_constraints_incompatible)


def test_cspace_target_translation_path_constraint_none():
    """Test construction of a `CSpaceTarget.TranslationPathConstraint` using `none()`."""
    cumotion.TrajectoryOptimizer.CSpaceTarget.TranslationPathConstraint.none()


def test_cspace_target_translation_path_constraint_linear():
    """Test construction of a `CSpaceTarget.TranslationPathConstraint` using `linear()`."""
    # Set deviation limits for testing.
    zero_limit = 0.0
    negative_limit = -10.0
    positive_limit = 10.0

    # Expect that a linear path translation constraint can be created with no deviation limit.
    cumotion.TrajectoryOptimizer.CSpaceTarget.TranslationPathConstraint.linear()

    # Expect that a linear path translation constraint can be created with 0 deviation limit.
    cumotion.TrajectoryOptimizer.CSpaceTarget.TranslationPathConstraint.linear(zero_limit)

    # Expect that a linear path translation constraint can be created with a positive deviation
    # limit.
    cumotion.TrajectoryOptimizer.CSpaceTarget.TranslationPathConstraint.linear(positive_limit)

    # Expect *failure* to create a linear path translation constraint with a negative deviation
    # limit.
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.CSpaceTarget.TranslationPathConstraint.linear(negative_limit)


def test_cspace_target_orientation_path_constraint_none():
    """Test construction of a `CSpaceTarget.OrientationPathConstraint` using `none()`."""
    cumotion.TrajectoryOptimizer.CSpaceTarget.OrientationPathConstraint.none()


def test_cspace_target_orientation_path_constraint_constant():
    """Test construction of a `CSpaceTarget.OrientationPathConstraint` using `constant()`."""
    # Set deviation limits for testing.
    zero_limit = 0.0
    negative_limit = -10.0
    positive_limit = 1.0
    large_positive_limit = 10.0

    # Expect that a constant orientation constraint can be created with no deviation limit.
    cumotion.TrajectoryOptimizer.CSpaceTarget.OrientationPathConstraint.constant()

    # Expect that a constant orientation constraint can be created with no deviation limit and
    # setting orientation from the initial configuration.
    cumotion.TrajectoryOptimizer.CSpaceTarget.OrientationPathConstraint.constant(None, False)

    # Expect that a constant orientation constraint can be created with 0 deviation limit.
    cumotion.TrajectoryOptimizer.CSpaceTarget.OrientationPathConstraint.constant(zero_limit)

    # Expect that a constant orientation constraint can be created with a positive deviation limit.
    cumotion.TrajectoryOptimizer.CSpaceTarget.OrientationPathConstraint.constant(positive_limit)

    # Expect *failure* to create a constant orientation constraint with a negative deviation limit.
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.CSpaceTarget.OrientationPathConstraint.constant(negative_limit)

    # Expect *warning* when creating a constant orientation constraint with a deviation limit
    # greater than pi.
    cumotion.TrajectoryOptimizer.CSpaceTarget.OrientationPathConstraint.constant(
        large_positive_limit)


def test_cspace_target_orientation_path_constraint_axis():
    """Test construction of a `CSpaceTarget.OrientationPathConstraint` using `axis()`."""
    # Set arbitrary, non-normalized target axes.
    tool_frame_axis = np.array([5.0, 0.0, 0.0])
    world_target_axis = np.array([0.0, 3.0, 0.0])

    # Set deviation limits for testing.
    zero_limit = 0.0
    negative_limit = -10.0
    positive_limit = 1.0
    large_positive_limit = 10.0

    # Expect that a target axis orientation constraint can be created with no deviation limit.
    cumotion.TrajectoryOptimizer.CSpaceTarget.OrientationPathConstraint.axis(
        tool_frame_axis, world_target_axis)

    # Expect that a target axis orientation constraint can be created with 0 deviation limit.
    cumotion.TrajectoryOptimizer.CSpaceTarget.OrientationPathConstraint.axis(
        tool_frame_axis, world_target_axis, zero_limit)

    # Expect that a target axis orientation constraint can be created with a positive deviation
    # limit.
    cumotion.TrajectoryOptimizer.CSpaceTarget.OrientationPathConstraint.axis(
        tool_frame_axis, world_target_axis, positive_limit)

    # Expect *failure* to create a target axis orientation constraint with a negative deviation
    # limit.
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.CSpaceTarget.OrientationPathConstraint.axis(
            tool_frame_axis, world_target_axis, negative_limit)

    # Expect *warning* when creating a target axis orientation constraint with a deviation limit
    # greater than pi.
    cumotion.TrajectoryOptimizer.CSpaceTarget.OrientationPathConstraint.axis(
        tool_frame_axis, world_target_axis, large_positive_limit)

    # Expect *failure* to create a target axis orientation constraint with a zero or nearly zero
    # tool frame axis.
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.CSpaceTarget.OrientationPathConstraint.axis(
            np.array([0.0, 0.0, 0.0]), world_target_axis)
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.CSpaceTarget.OrientationPathConstraint.axis(
            1e-10 * np.array([1.0, 0.0, 0.0]), world_target_axis)

    # Expect *failure* to create a target axis orientation constraint with a zero or nearly zero
    # world target axis.
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.CSpaceTarget.OrientationPathConstraint.axis(
            tool_frame_axis, np.array([0.0, 0.0, 0.0]))
    with pytest.raises(Exception):
        cumotion.TrajectoryOptimizer.CSpaceTarget.OrientationPathConstraint.axis(
            tool_frame_axis, 1e-10 * np.array([1.0, 0.0, 0.0]))


def test_cspace_target():
    """Test construction of a `CSpaceTarget`."""
    # NOTE: Since any valid `TranslationPathConstraint` and any valid `OrientationPathConstraint`
    # may be combined to form a `CSpaceTarget`, extensive tests are not included here.

    # Test that we can create a `CSpaceTarget` with default path constraints.
    cspace_position = np.array([1.0, 2.0, 3.0, 4.0])
    cumotion.TrajectoryOptimizer.CSpaceTarget(cspace_position)

    # Test that we can create a `CSpaceTarget` with custom path constraints
    translation_path_constraint = (
        cumotion.TrajectoryOptimizer.CSpaceTarget.TranslationPathConstraint.linear())
    orientation_path_constraint = (
        cumotion.TrajectoryOptimizer.CSpaceTarget.OrientationPathConstraint.constant())
    cumotion.TrajectoryOptimizer.CSpaceTarget(
        cspace_position, translation_path_constraint, orientation_path_constraint)
