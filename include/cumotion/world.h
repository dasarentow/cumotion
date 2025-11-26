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

#pragma once

#include <memory>
#include <optional>
#include <vector>

#include "cumotion/obstacle.h"
#include "cumotion/pose3.h"

namespace cumotion {

//! Forward declaration of `WorldViewHandle` for use by `World::addWorldView()`.
struct WorldViewHandle;

//! World represents a collection of obstacles.
class World {
 public:
  //! Opaque handle to an obstacle.
  struct ObstacleHandle {
    struct Impl;
    std::shared_ptr<Impl> impl;
  };

  virtual ~World() = default;

  //! Add `obstacle` to the world.
  //!
  //! All attributes of obstacle are copied to world and subsequent changes to `obstacle` will
  //! not be reflected in the world.
  //!
  //! If a `pose` is not provided for the `obstacle`, `Pose3::Identity()` will be used.
  //!
  //! Obstacles are automatically enabled when added.
  virtual ObstacleHandle addObstacle(const Obstacle &obstacle,
                                     std::optional<Pose3> pose = std::nullopt) = 0;

  //! Permanently remove obstacle, invalidating its handle.
  virtual void removeObstacle(const ObstacleHandle &obstacle) = 0;

  //! Enable an obstacle for the purpose of collision checks and distance evaluations.
  virtual void enableObstacle(const ObstacleHandle &obstacle) = 0;

  //! Disable an obstacle for the purpose of collision checks and distance evaluations.
  virtual void disableObstacle(const ObstacleHandle &obstacle) = 0;

  //! Set the pose of the given obstacle.
  virtual void setPose(const ObstacleHandle &obstacle, const Pose3 &pose) = 0;

  //! Set the grid values for an obstacle of type SDF using a host-resident `values` buffer.
  //!
  //! It is assumed that `values` is stored with the `z` index varying fastest and has dimensions
  //! given by the `Obstacle::Grid` associated with `obstacle`.  For example, for an obstacle with
  //! `Grid` parameters `num_voxels_x`, `num_voxels_y`, and `num_voxels_z`, the length of `values`
  //! should be `num_voxels_x * num_voxels_y * num_voxels_z`, and in the provided coordinates,
  //! adjacent elements in memory should correspond to voxels with adjacent Z coordinates, and
  //! voxels with adjacent X coordinates should be separated by `num_voxels_y * num_voxels_z`
  //! elements.
  //!
  //! `precision` specifies the floating-point type of `values`.  `Obstacle::Grid::Precision::HALF`
  //! corresponds to the `__half` data type defined in the `cuda_fp16.h` header.
  //!
  //! If the type of `obstacle` is not `Obstacle::Type::SDF`, a fatal error will be logged.
  virtual void setSdfGridValuesFromHost(const ObstacleHandle &obstacle,
                                        const void *values,
                                        Obstacle::Grid::Precision grid_precision) = 0;

  //! Set the grid values for an obstacle of type SDF using a device-resident buffer `values`.
  //!
  //! It is assumed that `values` is stored with the `z` index varying fastest and has dimensions
  //! given by the `Obstacle::Grid` associated with `obstacle`.  For example, for an obstacle with
  //! `Grid` parameters `num_voxels_x`, `num_voxels_y`, and `num_voxels_z`, the length of `values`
  //! should be `num_voxels_x * num_voxels_y * num_voxels_z`, and in the provided coordinates,
  //! adjacent elements in memory should correspond to voxels with adjacent Z coordinates, and
  //! voxels with adjacent X coordinates should be separated by `num_voxels_y * num_voxels_z`
  //! elements.
  //!
  //! `precision` specifies the floating-point type of `values`.  `Obstacle::Grid::Precision::HALF`
  //! corresponds to the `__half` data type defined in the `cuda_fp16.h` header.
  //!
  //! If the type of `obstacle` is not `Obstacle::Type::SDF`, a fatal error will be logged.
  virtual void setSdfGridValuesFromDevice(const ObstacleHandle &obstacle,
                                          const void *values,
                                          Obstacle::Grid::Precision grid_precision) = 0;

  //! Create a view into the world that can be used for collision checks and distance evaluations.
  //!
  //! Each world view will maintain a static view of the world until it is updated. When a
  //! world view is updated, it will reflect any changes to the world since its last update.
  virtual WorldViewHandle addWorldView() = 0;
};

//! Create an empty world with no obstacles.
std::shared_ptr<World> CreateWorld();

//! A handle to a view of a `cumotion::World`.
//!
//! This view can be independently updated to track updates made to a `cumotion::World` object.
//! A `WorldViewHandle` may be copied, with all copies sharing the same underlying view.
//!
//! To query spatial relationships in a world view, use `cumotion::WorldInspector`.
struct WorldViewHandle {
  //! Update world view such that any changes to the underlying world are reflected in this view.
  void update();

  struct Impl;
  std::shared_ptr<Impl> impl;
};

}  // namespace cumotion
