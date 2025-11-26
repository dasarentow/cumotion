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

"""Helper functions and constants for the cuMotion unit tests."""

# Standard Library
import os

# cuMotion
import cumotion

# Set cuMotion root directory
CUMOTION_ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))


class WarningSuppressionManager:
    """Helper class to temporarily suppress warnings inside `with` block."""

    def __init__(self, temporary_log_level):
        """Capture log level to be set for duration of the `with` block."""
        self.temporary_log_level = temporary_log_level

    def __enter__(self):
        """Set desired log level at start of `with` block."""
        cumotion.set_log_level(self.temporary_log_level)

    def __exit__(self, e_type, e_value, tb):
        """Restore log level to default (`WARNING`) at end of `with` block."""
        cumotion.set_log_level(cumotion.LogLevel.WARNING)


warnings_disabled = WarningSuppressionManager(cumotion.LogLevel.ERROR)
errors_disabled = WarningSuppressionManager(cumotion.LogLevel.FATAL)
