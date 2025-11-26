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

"""Unit tests for task constraints from collision_free_ik_solver_python.h."""

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
    cumotion.CollisionFreeIkSolver.TranslationConstraint.target(target)

    # Expect that a translation constraint can be created with a zero deviation limit.
    cumotion.CollisionFreeIkSolver.TranslationConstraint.target(target, zero_limit)

    # Expect that a translation constraint can be created with a positive deviation limit.
    cumotion.CollisionFreeIkSolver.TranslationConstraint.target(target, positive_limit)

    # Expect *failure* to create a translation constraint with a negative deviation limit.
    with pytest.raises(Exception):
        cumotion.CollisionFreeIkSolver.TranslationConstraint.target(target, negative_limit)


def test_translation_constraint_goalset_target():
    """Test construction of a `TranslationConstraintGoalset` using `target()`."""
    # Set arbitrary targets.
    targets = [np.array([3.0, 7.0, -9.0]), np.array([2.0, 1.0, 8.0])]

    # Set deviation limits for testing.
    zero_limit = 0.0
    negative_limit = -10.0
    positive_limit = 10.0

    # Expect that translation constraints can be created with no deviation limit.
    cumotion.CollisionFreeIkSolver.TranslationConstraintGoalset.target(targets)

    # Expect that translation constraints can be created with a zero deviation limit.
    cumotion.CollisionFreeIkSolver.TranslationConstraintGoalset.target(targets, zero_limit)

    # Expect that translation constraints can be created with a positive deviation limit.
    cumotion.CollisionFreeIkSolver.TranslationConstraintGoalset.target(targets, positive_limit)

    # Expect *failure* to create translation constraints with a negative deviation limit.
    with pytest.raises(Exception):
        cumotion.CollisionFreeIkSolver.TranslationConstraintGoalset.target(targets, negative_limit)

    # Expect *failure* to create translation constraints with empty translation targets.
    with pytest.raises(Exception):
        cumotion.CollisionFreeIkSolver.TranslationConstraintGoalset.target([])


def test_orientation_constraint_none():
    """Test construction of an `OrientationConstraint` using `none()`."""
    cumotion.CollisionFreeIkSolver.OrientationConstraint.none()


def test_orientation_constraint_target():
    """Test construction of an `OrientationConstraint` using `target()`."""
    # Set arbitrary orientation target.
    target = cumotion.Rotation3.identity()

    # Set deviation limits for testing.
    zero_limit = 0.0
    negative_limit = -1.0
    positive_limit = 1.0
    large_positive_limit = 6.0

    # Expect that a target orientation constraint can be created with no deviation limits.
    cumotion.CollisionFreeIkSolver.OrientationConstraint.target(target)

    # Expect that a target orientation constraint can be created with 0 deviation limit.
    cumotion.CollisionFreeIkSolver.OrientationConstraint.target(target, zero_limit)

    # Expect that a target orientation constraint can be created with a positive deviation limit.
    cumotion.CollisionFreeIkSolver.OrientationConstraint.target(target, positive_limit)

    # Expect *failure* to create a target orientation constraint with a negative deviation limit.
    with pytest.raises(Exception):
        cumotion.CollisionFreeIkSolver.OrientationConstraint.target(target, negative_limit)

    # Expect *warning* when creating a target orientation constraint with a deviation limit
    # greater than pi.
    cumotion.CollisionFreeIkSolver.OrientationConstraint.target(target, large_positive_limit)


def test_orientation_constraint_axis():
    """Test construction of an `OrientationConstraint` using `axis()`."""
    # Set arbitrary, non-normalized target axes.
    tool_frame_axis = np.array([5.0, 0.0, 0.0])
    world_target_axis = np.array([0.0, 3.0, 0.0])

    # Set deviation limits for testing.
    zero_limit = 0.0
    negative_limit = -1.0
    positive_limit = 1.0
    large_positive_limit = 6.0

    # Expect that an axis orientation constraint can be created with no deviation limit.
    cumotion.CollisionFreeIkSolver.OrientationConstraint.axis(tool_frame_axis, world_target_axis)

    # Expect that an axis orientation constraint can be created with a zero deviation limit.
    cumotion.CollisionFreeIkSolver.OrientationConstraint.axis(
        tool_frame_axis, world_target_axis, zero_limit)

    # Expect that an axis orientation constraint can be created with a positive deviation limit.
    cumotion.CollisionFreeIkSolver.OrientationConstraint.axis(
        tool_frame_axis, world_target_axis, positive_limit)

    # Expect *failure* to create an axis orientation constraint with a negative deviation limit.
    with pytest.raises(Exception):
        cumotion.CollisionFreeIkSolver.OrientationConstraint.axis(
            tool_frame_axis, world_target_axis, negative_limit)

    # Expect *warning* when creating an axis orientation constraint with a deviation limit
    # greater than pi.
    cumotion.CollisionFreeIkSolver.OrientationConstraint.axis(
        tool_frame_axis, world_target_axis, large_positive_limit)

    # Expect *failure* to create an axis orientation constraint with a zero or nearly zero
    # tool frame axis.
    with pytest.raises(Exception):
        cumotion.CollisionFreeIkSolver.OrientationConstraint.axis(
            np.array([0.0, 0.0, 0.0]), world_target_axis)
    with pytest.raises(Exception):
        cumotion.CollisionFreeIkSolver.OrientationConstraint.axis(
            1e-10 * np.array([1.0, 0.0, 0.0]), world_target_axis)

    # Expect *failure* to create an axis orientation constraint with a zero or nearly zero
    # world target axis.
    with pytest.raises(Exception):
        cumotion.CollisionFreeIkSolver.OrientationConstraint.axis(
            tool_frame_axis, np.array([0.0, 0.0, 0.0]))
    with pytest.raises(Exception):
        cumotion.CollisionFreeIkSolver.OrientationConstraint.axis(
            tool_frame_axis, 1e-10 * np.array([1.0, 0.0, 0.0]))


def test_orientation_constraint_goalset_none():
    """Test construction of an `OrientationConstraintGoalset` using `none()`."""
    cumotion.CollisionFreeIkSolver.OrientationConstraintGoalset.none()


def test_orientation_constraint_goalset_target():
    """Test construction of an `OrientationConstraintGoalset` using `target()`."""
    # Set arbitrary orientation targets.
    targets = [cumotion.Rotation3.identity(), cumotion.Rotation3.identity()]

    # Set deviation limits for testing.
    zero_limit = 0.0
    negative_limit = -1.0
    positive_limit = 1.0
    large_positive_limit = 6.0

    # Expect that target orientation constraints can be created with no deviation limits.
    cumotion.CollisionFreeIkSolver.OrientationConstraintGoalset.target(targets)

    # Expect that target orientation constraints can be created with a zero deviation limit.
    cumotion.CollisionFreeIkSolver.OrientationConstraintGoalset.target(targets, zero_limit)

    # Expect that target orientation constraints can be created with a positive deviation limit.
    cumotion.CollisionFreeIkSolver.OrientationConstraintGoalset.target(targets, positive_limit)

    # Expect *failure* to create target orientation constraints with a negative deviation limit.
    with pytest.raises(Exception):
        cumotion.CollisionFreeIkSolver.OrientationConstraintGoalset.target(targets, negative_limit)

    # Expect *warning* when creating target orientation constraints with a deviation limit
    # greater than pi.
    cumotion.CollisionFreeIkSolver.OrientationConstraintGoalset.target(
        targets, large_positive_limit)

    # Expect *failure* to create target orientation constraints with empty orientation targets.
    with pytest.raises(Exception):
        cumotion.CollisionFreeIkSolver.OrientationConstraintGoalset.target([])


def test_orientation_constraint_goalset_axis():
    """Test construction of an `OrientationConstraintGoalset` using `axis()`."""
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

    # Expect that axis orientation constraints can be created with no deviation limit.
    cumotion.CollisionFreeIkSolver.OrientationConstraintGoalset.axis(
        tool_frame_axes, world_target_axes)

    # Expect that axis orientation constraints can be created with a zero deviation limit.
    cumotion.CollisionFreeIkSolver.OrientationConstraintGoalset.axis(
        tool_frame_axes, world_target_axes, zero_limit)

    # Expect that axis orientation constraints can be created with a positive deviation limit.
    cumotion.CollisionFreeIkSolver.OrientationConstraintGoalset.axis(
        tool_frame_axes, world_target_axes, positive_limit)

    # Expect *failure* to create axis orientation constraints with a negative deviation limit.
    with pytest.raises(Exception):
        cumotion.CollisionFreeIkSolver.OrientationConstraintGoalset.axis(
            tool_frame_axes, world_target_axes, negative_limit)

    # Expect *warning* when creating axis orientation constraints with a deviation limit
    # greater than pi.
    cumotion.CollisionFreeIkSolver.OrientationConstraintGoalset.axis(
        tool_frame_axes, world_target_axes, large_positive_limit)

    # Expect *failure* to create axis orientation constraints with a zero or nearly zero
    # tool frame axis.
    with pytest.raises(Exception):
        cumotion.CollisionFreeIkSolver.OrientationConstraintGoalset.axis(
            zero_axes, world_target_axes)
    with pytest.raises(Exception):
        cumotion.CollisionFreeIkSolver.OrientationConstraintGoalset.axis(
            nearly_zero_axes, world_target_axes)

    # Expect *failure* to create axis orientation constraints with a zero or nearly zero
    # world target axis.
    with pytest.raises(Exception):
        cumotion.CollisionFreeIkSolver.OrientationConstraintGoalset.axis(
            tool_frame_axes, zero_axes)
    with pytest.raises(Exception):
        cumotion.CollisionFreeIkSolver.OrientationConstraintGoalset.axis(
            tool_frame_axes, nearly_zero_axes)

    # Expect *failure* to create axis orientation constraints with differing numbers of
    # tool frame axes and world target axes.
    with pytest.raises(Exception):
        cumotion.CollisionFreeIkSolver.OrientationConstraintGoalset.axis(
            tool_frame_axes, too_many_axes)
    with pytest.raises(Exception):
        cumotion.CollisionFreeIkSolver.OrientationConstraintGoalset.axis(
            too_many_axes, world_target_axes)

    # Expect *failure* to create axis orientation constraints with empty tool frame and world
    # target axes.
    with pytest.raises(Exception):
        cumotion.CollisionFreeIkSolver.OrientationConstraintGoalset.axis([], [])


def test_task_space_target():
    """Test construction of a `TaskSpaceTarget`."""
    # NOTE: Since any valid `TranslationConstraint` and any valid `OrientationConstraint` may be
    # combined to form a `TaskSpaceTarget`, extensive tests are not included here.

    # Create an arbitrary `TranslationConstraint`.
    limit = 1.0
    translation_target = np.array([3.0, 7.0, -9.0])
    translation_constraint = cumotion.CollisionFreeIkSolver.TranslationConstraint.target(
        translation_target, limit)

    # Create an arbitrary `OrientationConstraint`.
    orientation_target = cumotion.Rotation3.identity()
    orientation_constraint = cumotion.CollisionFreeIkSolver.OrientationConstraint.target(
        orientation_target)

    # Create `TaskSpaceTarget`, expecting no failures.
    cumotion.CollisionFreeIkSolver.TaskSpaceTarget(translation_constraint, orientation_constraint)

    # Test that we can create a `TaskSpaceTarget` with default orientation constraint.
    cumotion.CollisionFreeIkSolver.TaskSpaceTarget(translation_constraint)


def test_task_space_target_goalset():
    """Test construction of a `TaskSpaceTargetGoalset`."""
    # NOTE: The only invalid combination of a valid `TranslationConstraintGoalset` and a valid
    # `OrientationConstraintGoalset` arises if the number of constraints differs between the two
    # goalsets. Testing is limited to one valid combination and one invalid combination.

    # Create an arbitrary `TranslationConstraintGoalset`.
    translation_targets = [np.array([3.0, 7.0, -9.0]), np.array([1.0, 2.0, -3.0])]
    translation_constraints = cumotion.CollisionFreeIkSolver.TranslationConstraintGoalset.target(
        translation_targets)

    # Create an arbitrary `OrientationConstraintGoalset`.
    orientation_constraints = cumotion.CollisionFreeIkSolver.OrientationConstraintGoalset.none()

    # Create `TaskSpaceTargetGoalset`, expecting no failures.
    cumotion.CollisionFreeIkSolver.TaskSpaceTargetGoalset(
        translation_constraints, orientation_constraints)

    # Test that we can create a `TaskSpaceTargetGoalset` with default orientation constraints.
    cumotion.CollisionFreeIkSolver.TaskSpaceTargetGoalset(translation_constraints)

    # Expect *failure* to create a `TaskSpaceTargetGoalset` with an invalid numbers of constraints.
    # Create `orientation_constraints` that are incompatible with `translation_constraints`.
    orientation_targets = [cumotion.Rotation3.identity()]
    orientation_constraints_incompatible = (
        cumotion.CollisionFreeIkSolver.OrientationConstraintGoalset.target(orientation_targets))

    with pytest.raises(Exception):
        cumotion.CollisionFreeIkSolver.TaskSpaceTargetGoalset(
            translation_constraints, orientation_constraints_incompatible)
