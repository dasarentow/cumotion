# SPDX-FileCopyrightText: Copyright (c) 2019-2025 NVIDIA CORPORATION & AFFILIATES.
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

cmake_minimum_required(VERSION 3.21)

project(cumotion_examples LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA.
find_package(CUDAToolkit 12.2 REQUIRED)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc" CACHE FILEPATH
  "Filepath to nvcc compiler, default installation path provided")
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  # Enable Blackwell if supported by CUDA toolkit.
  set(ENABLE_BLACKWELL OFF)
  if (${CUDAToolkit_VERSION} VERSION_GREATER_EQUAL 12.8)
    set(ENABLE_BLACKWELL ON)
  endif()
  message(STATUS "CUDA Toolkit VERSION: ${CUDAToolkit_VERSION}. Supports Blackwell? ${ENABLE_BLACKWELL}")
  if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
    set(CMAKE_CUDA_ARCHITECTURES 87)  # Jetson Orin
    if(${ENABLE_BLACKWELL})
      list(APPEND CMAKE_CUDA_ARCHITECTURES 110)  # Jetson Thor
    endif()
  else()
    set(CMAKE_CUDA_ARCHITECTURES "75;86;89")  # Turing (TU10x), Ampere (GA10x), and Ada (AD10x)
    if(${ENABLE_BLACKWELL})
      list(APPEND CMAKE_CUDA_ARCHITECTURES 120)  # Blackwell (GB20x)
    endif()
  endif()
endif ()
enable_language(CUDA)

# Find dependencies.
find_package(Eigen3 REQUIRED)

# Find cumotion if this project is the top-level project.
if (PROJECT_IS_TOP_LEVEL)
  find_package(cumotion REQUIRED)

  if (cumotion_FOUND)
    message(STATUS "cuMotion found at ${cumotion_DIR}")
  else ()
    message(STATUS "cuMotion not found!")
  endif ()
endif ()

# Get the location of the content/ directory.
if (NOT DEFINED CONTENT_DIR)
  # Find
  #   content/
  # Relative to
  #   lib/cmake/cumotion/cumotionConfig.cmake
  set(CONTENT_DIR "${cumotion_DIR}/../../../content")
endif ()
cmake_path(ABSOLUTE_PATH CONTENT_DIR NORMALIZE)
message(STATUS "cuMotion content directory: ${CONTENT_DIR}")

# Build the examples.
macro (add_example TARGET SRC)
  add_executable(${TARGET} ${SRC})
  target_link_libraries(${TARGET} cumotion::cumotion)
  target_compile_definitions(${TARGET} PRIVATE CONTENT_DIR="${CONTENT_DIR}")
  target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
endmacro()

# Kinematics example
add_example(kinematics_example src/kinematics_example.cpp)

# Franka collision-free IK example
add_example(franka_collision_free_ik_solver_example src/trajectory_optimizer_examples/franka_collision_free_ik_solver_example.cpp)

# Franka trajectory optimization example (with no obstacles)
add_example(franka_trajectory_optimizer_example src/trajectory_optimizer_examples/franka_trajectory_optimizer_example.cpp)

# Franka collision-free trajectory optimization example (with cuboid obstacles)
add_example(franka_trajectory_optimizer_obstacle_example src/trajectory_optimizer_examples/franka_trajectory_optimizer_obstacle_example.cpp)
