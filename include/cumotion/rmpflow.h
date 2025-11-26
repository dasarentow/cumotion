// SPDX-FileCopyrightText: Copyright (c) 2019-2025 NVIDIA CORPORATION & AFFILIATES.
//                         All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

//! @file
//! @brief Public interface to cuMotion's RMPflow implementation

#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "Eigen/Core"

#include "cumotion/robot_description.h"
#include "cumotion/rotation3.h"
#include "cumotion/world.h"

namespace cumotion {

//! Interface class for loading and manipulating RMPflow parameters.
//! WARNING: This interface may change in a future release.
class RmpFlowConfig {
 public:
  virtual ~RmpFlowConfig() = default;

  //! Get the value of a parameter, given a "param_name" string of the form
  //! "<rmp_name>/<parameter_name>"
  [[nodiscard]] virtual double getParam(const std::string &param_name) const = 0;

  //! Set the value of the parameter.
  virtual void setParam(const std::string &param_name, double value) = 0;

  //! Get the names and values of all parameters.  The two vectors will be overwritten if not empty.
  virtual void getAllParams(std::vector<std::string> &names, std::vector<double> &values) const = 0;

  //! Set all parameters at once.  The vectors "names" and "values" must have the same size.
  //! The parameter corresponding to names[i] will be set to the value given by values[i].
  virtual void setAllParams(const std::vector<std::string> &names,
                            const std::vector<double> &values) = 0;

  //! Set the world view that will be used for obstacle avoidance. All enabled obstacles in
  //! `world_view` will be avoided by the RMPflow policy.
  virtual void setWorldView(const WorldViewHandle &world_view) = 0;
};

//! Load a set of RMPflow parameters from file, and combine with a robot description to create a
//! configuration object for consumption by CreateRmpFlow().  The "end_effector_frame" should
//! correspond to a link name as specified in the original URDF used to create the robot
//! description. All enabled obstacles in `world_view` will be avoided by the RMPflow policy.
std::unique_ptr<RmpFlowConfig> CreateRmpFlowConfigFromFile(
    const std::filesystem::path &rmpflow_config_file,
    const RobotDescription &robot_description,
    const std::string &end_effector_frame,
    const WorldViewHandle &world_view);

//! Load a set of RMPflow parameters from string, and combine with a robot description to create a
//! configuration object for consumption by CreateRmpFlow().  The "end_effector_frame" should
//! correspond to a link name as specified in the original URDF used to create the robot
//! description. All enabled obstacles in `world_view` will be avoided by the RMPflow policy.
std::unique_ptr<RmpFlowConfig> CreateRmpFlowConfigFromMemory(
    const std::string &rmpflow_config,
    const RobotDescription &robot_description,
    const std::string &end_effector_frame,
    const WorldViewHandle &world_view);

//! Interface class for building and evaluating a motion policy in the RMPflow framework
class RmpFlow {
 public:
  virtual ~RmpFlow() = default;

  //! Set an end-effector position attractor.
  //!
  //! The origin of the end effector frame will be driven towards the specified position.
  //!
  //! WARNING: This attractor-specification interface is planned to change in a future release to
  //!          support multiple task space attractors rather than a single, fixed end-effector
  //!          frame, and to allow setting relative weights for each attractor. It is recommended
  //!          that this function be used for setting end-effector position attractors until then.
  virtual void setEndEffectorPositionAttractor(const Eigen::Vector3d &position) = 0;

  //! Clear end-effector position attractor.
  //!
  //! The RMP driving the origin of the end effector frame towards a particular position will be
  //! deactivated.
  //!
  //! WARNING: This attractor-specification interface is planned to change in a future release to
  //!          support multiple task space attractors rather than a single, fixed end-effector
  //!          frame, and to allow setting relative weights for each attractor. It is recommended
  //!          that this function be used for clearing end-effector position attractors until then.
  virtual void clearEndEffectorPositionAttractor() = 0;

  //! Set an end-effector orientation attractor.
  //!
  //! The orientation of the end effector frame will be driven towards the specified orientation.
  //!
  //! WARNING: This attractor-specification interface is planned to change in a future release to
  //!          support multiple task space attractors rather than a single, fixed end-effector
  //!          frame, and to allow setting relative weights for each attractor. It is recommended
  //!          that this function be used for setting orientation attractors until then.
  virtual void setEndEffectorOrientationAttractor(const Rotation3 &orientation) = 0;

  //! Clear end-effector orientation attractor.
  //!
  //! The RMPs driving the orientation of the end effector frame towards a particular orientation
  //! will be deactivated.
  //!
  //! WARNING: This attractor-specification interface is planned to change in a future release to
  //!          support multiple task space attractors rather than a single, fixed end-effector
  //!          frame, and to allow setting relative weights for each attractor. It is recommended
  //!          that this function be used for clearing orientation attractors until then.
  virtual void clearEndEffectorOrientationAttractor() = 0;

  //! Set an attractor in generalized coordinates (configuration space).
  //!
  //! The c-space coordinates will be biased towards the specified configuration.
  //!
  //! NOTE:  Unlike the end effector attractors, there is always an active c-space attractor (either
  //!        set using `setCSpaceAttractor()` or using the default value loaded from the robot
  //!        description).
  //!
  //! WARNING: This attractor-specification interface is planned to change in a future release to
  //!          support multiple task space attractors rather than a single, fixed end-effector
  //!          frame, and to allow setting relative weights for each attractor. It is recommended
  //!          that this function be used for setting c-space attractors until then.
  virtual void setCSpaceAttractor(const Eigen::VectorXd &cspace_position) = 0;

  //! Compute configuration-space acceleration from motion policy, given input state. This takes
  //! into account the current C-space and/or end-effector targets, as well as any
  //! currently-enabled obstacles.
  virtual void evalAccel(const Eigen::VectorXd &cspace_position,
                         const Eigen::VectorXd &cspace_velocity,
                         Eigen::Ref<Eigen::VectorXd> cspace_accel) const = 0;

  //! Compute configuration-space force and metric from motion policy, given input state. This takes
  //! into account the current C-space and/or end-effector targets, as well as any
  //! currently-enabled obstacles.
  virtual void evalForceAndMetric(const Eigen::VectorXd &cspace_position,
                                  const Eigen::VectorXd &cspace_velocity,
                                  Eigen::Ref<Eigen::VectorXd> cspace_force,
                                  Eigen::Ref<Eigen::MatrixXd> cspace_metric) const = 0;
};

//! Create an instance of the RmpFlow interface from an RMPflow configuration.
std::unique_ptr<RmpFlow> CreateRmpFlow(const RmpFlowConfig &config);

}  // namespace cumotion
