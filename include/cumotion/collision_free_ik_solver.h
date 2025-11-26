// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "Eigen/Core"

#include "cumotion/robot_description.h"
#include "cumotion/rotation3.h"
#include "cumotion/world.h"

namespace cumotion {

//! Configuration parameters for a `CollisionFreeIkSolver`.
class CollisionFreeIkSolverConfig {
 public:
  virtual ~CollisionFreeIkSolverConfig() = default;

  //! Specify the value for a given parameter.
  //!
  //! The required `ParamValue` constructor for each param is detailed in the
  //! documentation for `setParam()`.
  struct ParamValue {
    //! Create `ParamValue` from `int`.
    ParamValue(int value);  // NOLINT Allow implicit conversion
    //! Create `ParamValue` from `double`.
    ParamValue(double value);  // NOLINT Allow implicit conversion

    struct Impl;
    std::shared_ptr<Impl> impl;
  };

  //! Set the value of the parameter.
  //!
  //! `setParam()` returns `true` if the parameter has been successfully updated and `false` if an
  //! error has occurred (e.g., invalid parameter).
  //!
  //! The following parameters can be set for `CollisionFreeIkSolver`:
  //!
  //! `task_space_position_tolerance` [`double`]
  //!   - Maximum allowable violation of the user-specified constraint on the position of the tool
  //!     frame.
  //!   - It determines when a solution is considered to have satisfied position targets.
  //!     Smaller values require more precise solutions but may be harder to achieve.
  //!   - Units are in meters.
  //!   - A default value of 1e-3 (1mm) is recommended for most use-cases.
  //!   - Must be positive.
  //!
  //! `task_space_orientation_tolerance` [`double`]
  //!   - Maximum allowable violation of the user-specified constraint on the orientation of the
  //!     tool frame.
  //!   - It determines when a solution is considered to have satisfied orientation targets.
  //!     Smaller values require more precise solutions but may be harder to achieve.
  //!   - Units are in radians.
  //!   - A default value of 5e-3 (approximately 0.29 degrees) is recommended for most use-cases.
  //!   - Must be positive.
  //!
  //! `num_seeds` [`int`]
  //!   - Number of seeds used to solve the inverse kinematics (IK) problem.
  //!   - The IK solver generates multiple pseudo-random c-space configurations and
  //!     optimizes them to find diverse collision-free configurations for the desired tool pose.
  //!     Higher values increase the likelihood of finding valid solutions but increase
  //!     computational cost.
  //!   - A default value of 25 is recommended for most use-cases.
  //!   - Must be positive.
  //!
  //! `max_reattempts` [`int`]
  //!   - Maximum number of times to restart the IK problem with different random seeds, in case of
  //!     failure, before giving up.
  //!   - Higher values increase the likelihood of finding valid IK solutions but increase
  //!     memory usage and the maximum possible time to find a solution. A value of 0 means no
  //!     retries (i.e. only perform the initial attempt).
  //!   - A default value of 10 is recommended for most use-cases.
  //!   - Must be non-negative.
  [[nodiscard]] virtual bool setParam(const std::string &param_name, ParamValue value) = 0;
};

//! Use default parameters to create a configuration for collision-free inverse kinematics.
//!
//! These default parameters are combined with `robot_description`, `tool_frame_name`, and
//! `world_view`.
//!
//! A configuration will *NOT* be created if:
//!   1. `robot_description` is invalid, *OR*
//!   2. `tool_frame_name` is not a valid frame in `robot_description`.
//!
//! In the case of failure, a non-fatal error will be logged and a `nullptr` will be returned.
std::unique_ptr<CollisionFreeIkSolverConfig>
CreateDefaultCollisionFreeIkSolverConfig(const RobotDescription &robot_description,
                                         const std::string &tool_frame_name,
                                         const WorldViewHandle &world_view);

//! Interface for using numerical optimization to generate collision-free c-space positions for
//! task-space targets.
class CollisionFreeIkSolver {
 public:
  virtual ~CollisionFreeIkSolver() = default;

  //! Translation constraints restrict the position of the origin of a tool frame.
  class TranslationConstraint {
   public:
    //! Create a `TranslationConstraint` such that the desired position is fully specified as
    //! `translation_target`.
    //!
    //! The optional parameter `deviation_limit` can be used to allow deviation from the
    //! `translation_target`. This limit specifies the maximum allowable deviation from the desired
    //! position.
    //!
    //! If `deviation_limit` is not input, then the deviation limit is assumed to be zero
    //! (i.e., it is desired that the tool frame position be exactly `translation_target`).
    //!
    //! A fatal error will be logged if:
    //!   1. `deviation_limit` is negative.
    //!
    //! NOTE: The `translation_target` is specified in world frame coordinates (i.e., relative to
    //! the base of the robot).
    static TranslationConstraint Target(const Eigen::Vector3d &translation_target,
                                        std::optional<double> deviation_limit = std::nullopt);

    struct Impl;
    std::shared_ptr<Impl> impl;
  };

  //! Variant of `TranslationConstraint` for "goalset" planning.
  //!
  //! For goalset planning, a set of `TranslationConstraint`s are considered concurrently. Each
  //! `TranslationConstraint` in the goalset must have the same mode (e.g., "target") but may have
  //! different data for each `TranslationConstraint`.
  class TranslationConstraintGoalset {
   public:
    //! Create a `TranslationConstraintGoalset` such that `translation_targets` are fully specified.
    //!
    //! See `TranslationConstraint::Target()` for details.
    //!
    //! A fatal error will be logged if:
    //!   1. Any condition of `TranslationConstraint::Target()` is not met, *OR*
    //!   2. `translation_targets` is empty.
    static TranslationConstraintGoalset Target(
      const std::vector<Eigen::Vector3d> &translation_targets,
      std::optional<double> deviation_limit = std::nullopt);

    struct Impl;
    std::shared_ptr<Impl> impl;
  };

  //! Orientation constraints restrict the orientation of a tool frame.
  //!
  //! Each constraint may fully or partially constrain the orientation.
  class OrientationConstraint {
   public:
    //! Create an `OrientationConstraint` such that no tool frame orientation constraints are
    //! active.
    static OrientationConstraint None();

    //! Create an `OrientationConstraint` such that a tool frame `orientation_target` is fully
    //! specified.
    //!
    //! The optional parameter `deviation_limit` can be used to allow deviation from the
    //! `orientation_target`. If input, `deviation_limit` is expressed in radians. This limit
    //! specifies the maximum allowable deviation from the desired orientation.
    //!
    //! If `deviation_limit` is not input, then the deviation limit is assumed to be zero (i.e., it
    //! is desired that orientation be exactly `orientation_target`).
    //!
    //! In general, it is suggested that the `deviation_limit` be set to a value less than pi.
    //! A limit that is near or greater than pi essentially disables the constraint (without culling
    //! the computation). Non-fatal warnings will be logged if:
    //!   1. `deviation_limit` is near or greater than pi.
    //!
    //! A fatal error will be logged if:
    //!   1. `deviation_limit` is negative.
    //!
    //! NOTE: The `orientation_target` is specified in world frame coordinates (i.e., relative to
    //! the base of the robot).
    static OrientationConstraint Target(const Rotation3 &orientation_target,
                                        std::optional<double> deviation_limit = std::nullopt);

    //! Create an `OrientationConstraint` such that the tool frame orientation is constrained to
    //! rotate about a "free axis".
    //!
    //! The "free axis" for rotation is defined by a `tool_frame_axis` (specified in the tool frame
    //! coordinates) and a corresponding `world_target_axis` (specified in world frame coordinates).
    //!
    //! The optional `axis_deviation_limit` can be used to allow deviation from the
    //! desired axis alignment. If input, `axis_deviation_limit` is expressed in radians and the
    //! limit specifies the maximum allowable deviation of the rotation axis. If
    //! `axis_deviation_limit` is not input, then the deviation limit is assumed to be zero
    //! (i.e., it is desired that the tool frame axis be exactly aligned with `world_target_axis`).
    //!
    //! In general, it is suggested that the `axis_deviation_limit` be set to a value less than pi.
    //! A limit that is near or greater than pi essentially disables the constraint (without culling
    //! the computation). Non-fatal warnings will be logged if:
    //!   1. `axis_deviation_limit` is near or greater than pi.
    //!
    //! A fatal error will be logged if:
    //!   1. `axis_deviation_limit` is negative,
    //!   2. `tool_frame_axis` is (nearly) zero, *OR*
    //!   3. `world_target_axis` is (nearly) zero.
    //!
    //! NOTE: `tool_frame_axis` and `world_target_axis` inputs will be normalized.
    static OrientationConstraint Axis(const Eigen::Vector3d &tool_frame_axis,
                                      const Eigen::Vector3d &world_target_axis,
                                      std::optional<double> axis_deviation_limit = std::nullopt);

    struct Impl;
    std::shared_ptr<Impl> impl;
  };

  //! Variant of `OrientationConstraint` for "goalset" planning.
  //!
  //! For goalset planning, a set of `OrientationConstraint`s are considered concurrently. Each
  //! `OrientationConstraint` in the goalset must have the same mode (e.g., "full target") but
  //! may have different data for each `OrientationConstraint`.
  class OrientationConstraintGoalset {
   public:
    //! Create an `OrientationConstraintGoalset` such that no tool frame orientation constraints
    //! are active.
    static OrientationConstraintGoalset None();

    //! Create an `OrientationConstraintGoalset` such that tool frame `orientation_targets` are
    //! fully specified.
    //!
    //! See `OrientationConstraint::Target()` for details.
    //!
    //! A fatal error will be logged if:
    //!   1. Any condition of `OrientationConstraint::Target()` is not met, *OR*
    //!   2. `orientation_targets` is empty.
    static OrientationConstraintGoalset Target(
        const std::vector<Rotation3> &orientation_targets,
        std::optional<double> deviation_limit = std::nullopt);

    //! Create an `OrientationConstraintGoalset` such that the tool frame orientation is constrained
    //! to rotate about a "free axis".
    //!
    //! See `OrientationConstraint::Axis()` for details.
    //!
    //! A fatal error will be logged if:
    //!   1. Any condition of `OrientationConstraint::Axis()` is not met,
    //!   2. `tool_frame_axes` and `world_target_axes` do not have the same number of elements, *OR*
    //!   3. `tool_frame_axes` and `world_target_axes` are empty.
    static OrientationConstraintGoalset Axis(
        const std::vector<Eigen::Vector3d> &tool_frame_axes,
        const std::vector<Eigen::Vector3d> &world_target_axes,
        std::optional<double> axis_deviation_limit = std::nullopt);

    struct Impl;
    std::shared_ptr<Impl> impl;
  };

  //! Task-space targets restrict the position and (optionally) orientation of the tool frame.
  struct TaskSpaceTarget {
    //! Create a task-space target.
    explicit TaskSpaceTarget(const TranslationConstraint &translation_constraint,
                             const OrientationConstraint &orientation_constraint =
                                 OrientationConstraint::None());

    struct Impl;
    std::shared_ptr<Impl> impl;
  };

  //! Variant of `TaskSpaceTarget` for "goalset" planning.
  //!
  //! For goalset planning, a set of pose constraints are considered concurrently. Each pose
  //! constraint in the goalset must have the same mode (e.g., "position target and no orientation
  //! constraints") but may have different data for each constraint.
  struct TaskSpaceTargetGoalset {
    //! Create a task-space target goalset.
    //!
    //! A fatal error will be logged if:
    //!   1. The number of `translation_constraints` does not match the number of
    //!      `orientation_constraints`.
    explicit TaskSpaceTargetGoalset(const TranslationConstraintGoalset &translation_constraints,
                                    const OrientationConstraintGoalset &orientation_constraints =
                                        OrientationConstraintGoalset::None());

    struct Impl;
    std::shared_ptr<Impl> impl;
  };

  //! Results from an inverse kinematics solve.
  class Results {
   public:
    //! Indicate the success or failure of the inverse kinematics solve.
    enum class Status {
      //! One or more valid c-space positions found.
      SUCCESS,

      //! The inverse kinematics solver failed to find a valid solution.
      INVERSE_KINEMATICS_FAILURE,
    };

    virtual ~Results() = default;

    //! Return the `Status` from the inverse kinematics solve.
    [[nodiscard]] virtual Status status() const = 0;

    //! If `status()` returns `SUCCESS`, then `cSpacePositions()` returns unique IK solutions.
    //!
    //! A `SUCCESS` status indicates that at least one IK solution was found. Returned
    //! solutions are ordered from the lowest cost to the highest cost.
    //!
    //! If `status()` returns `INVERSE_KINEMATICS_FAILURE`, then an empty vector will be returned.
    [[nodiscard]] virtual std::vector<Eigen::VectorXd> cSpacePositions() const = 0;

    //! Return the indices of the targets selected for each valid c-space position.
    //!
    //! For goalset planning (e.g., `solveGoalset()`), returned values represent indices into
    //! the constraint vectors used to create the goalset (e.g., `TaskSpaceTargetGoalset`).
    //!
    //! For single target planning (e.g., `solve()`), zero will be returned for each valid
    //! c-space position in `cSpacePositions()`.
    //!
    //! In all cases, the length of returned `targetIndices()` will be equal to the length of
    //! returned `cSpacePositions()`.
    [[nodiscard]] virtual std::vector<int> targetIndices() const = 0;
  };

  //! Attempt to find c-space solutions that satisfy the constraints specified in
  //! `task_space_target`.
  //!
  //! The `cspace_seeds` are optional inputs that can be used to "warm start" the IK optimization.
  //! NOTE: It is *not* required that the `cspace_seeds` represent valid c-space positions (i.e., be
  //!       within position limits, be collision-free, etc.).
  //!
  //! A fatal error will be logged if:
  //!   1. Any c-space position in `cspace_seeds` does not have the same number of c-space
  //!      coordinates as the `RobotDescription` used to configure the
  //!      `CollisionFreeIkSolverConfig`.
  [[nodiscard]] virtual std::unique_ptr<Results> solve(
      const TaskSpaceTarget &task_space_target,
      const std::vector<Eigen::VectorXd> &cspace_seeds = {}) const = 0;

  //! Attempt to find c-space solutions that satisfy the constraints specified in
  //! `task_space_target_goalset`.
  //!
  //! To be considered a valid solution, a c-space position must satisfy the constraints of any
  //! *one* of the targets specified in `task_space_target_goalset`.
  //!
  //! The `cspace_seeds` are optional inputs that can be used to "warm start" the IK optimization.
  //! NOTE: It is *not* required that the `cspace_seeds` represent valid c-space positions (i.e., be
  //!       within position limits, be collision-free, etc.).
  //!
  //! A fatal error will be logged if:
  //!   1. Any c-space position in `cspace_seeds` does not have the same number of c-space
  //!      coordinates as the `RobotDescription` used to configure the
  //!      `CollisionFreeIkSolverConfig`.
  [[nodiscard]] virtual std::unique_ptr<Results> solveGoalset(
      const TaskSpaceTargetGoalset &task_space_target_goalset,
      const std::vector<Eigen::VectorXd> &cspace_seeds = {}) const = 0;
};

//! Create a `CollisionFreeIkSolver` with the given `config`.
std::unique_ptr<CollisionFreeIkSolver>
CreateCollisionFreeIkSolver(const CollisionFreeIkSolverConfig &config);

}  // namespace cumotion
