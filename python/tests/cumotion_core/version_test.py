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

"""Unit tests for the version attribute in the `cumotion` module."""

# Standard Library
import os

# cuMotion
import cumotion

# Local Folder
from ._test_helper import CUMOTION_ROOT_DIR


def get_version_from_file():
    """Get the version that is stored in the <cumotion root>/VERSION file."""
    version_file_path = os.path.join(CUMOTION_ROOT_DIR, "VERSION")
    with open(version_file_path, "r") as version_file:
        version_from_file = version_file.readlines()[0].strip()
    return version_from_file


def test_version_attr():
    """Test `cumotion.__version___` attribute."""
    assert hasattr(cumotion, "__version__")
    assert isinstance(cumotion.__version__, str)

    expected_version = get_version_from_file()
    assert expected_version == cumotion.__version__
