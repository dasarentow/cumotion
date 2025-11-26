// SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES.
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
//! @brief Public interface for querying spatial relationships in a world.

#pragma once

#include <memory>
#include <vector>

#include "Eigen/Core"

#include "cumotion/world.h"

namespace cumotion {

//! Interface for querying properties and spatial relationships within a world.
class WorldInspector {
 public:
  virtual ~WorldInspector() = default;

  //! Test whether a sphere defined by `center` and `radius` is in collision with ANY enabled
  //! obstacle in the world.
  [[nodiscard]] virtual bool inCollision(const Eigen::Vector3d &center, double radius) const = 0;

  //! Test whether a sphere defined by `center` and `radius` is in collision with the obstacle
  //! specified by `obstacle`.
  [[nodiscard]] virtual bool inCollision(const World::ObstacleHandle &obstacle,
                                         const Eigen::Vector3d &center,
                                         double radius) const = 0;

  //! Compute the distance from `point` to the obstacle specified by `obstacle`.
  //!
  //! Returns distance between the `point` and `obstacle`. If the `point` intersects `obstacle`, a
  //! negative distance is returned.  The distance gradient is written to `gradient` if provided.
  virtual double distanceTo(const World::ObstacleHandle &obstacle,
                            const Eigen::Vector3d &point,
                            Eigen::Vector3d *gradient = nullptr) const = 0;

  //! Compute distances from `point` to all enabled obstacles.
  //!
  //! Distances between the `point` and each enabled obstacle are written to `distances`.
  //! If the `point` intersects an obstacle, the resulting distance will be negative. The distance
  //! gradients are written to `distance_gradients` if provided.
  //!
  //! The number of `distances` and/or `distance_gradients` that are written (i.e., the number of
  //! enabled obstacles) is returned.
  //!
  //! If the length of `distances` or `distance_gradients` is less than the number of enabled
  //! obstacles, the vectors will be resized to the number of enabled obstacles. If the vector
  //! lengths are larger than the number of enabled obstacles, the vectors will NOT be resized.
  //! Only the first N elements will be written to, where N is the number of enabled obstacles. The
  //! values of extra elements at the end of the input vectors will NOT be changed.
  virtual int distancesTo(const Eigen::Vector3d &point,
                          std::vector<double> *distances,
                          std::vector<Eigen::Vector3d> *distance_gradients = nullptr) const = 0;

  //! Compute the minimum distance from `point` to ANY enabled obstacle in the current
  //! view of the world.
  //!
  //! Returns distance between the `point` and the nearest obstacle. If the `point` is inside an
  //! obstacle, a negative distance is returned. The distance gradient is written to `gradient` if
  //! provided.
  virtual double minDistance(const Eigen::Vector3d &point,
                             Eigen::Vector3d *gradient = nullptr) const = 0;

  //! Return the number of enabled obstacles in the current view of the world.
  [[nodiscard]] virtual int numEnabledObstacles() const = 0;
};

//! Create a `WorldInspector` for a given `world_view`.
std::unique_ptr<WorldInspector> CreateWorldInspector(const WorldViewHandle &world_view);

}  // namespace cumotion
