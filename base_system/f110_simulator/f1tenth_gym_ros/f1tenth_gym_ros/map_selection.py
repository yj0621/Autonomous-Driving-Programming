# Copyright 2026 ForzaETH
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Select the occupancy map used by the simulator."""

import os


def prefer_orig_map_yaml(map_yaml_path: str) -> str:
    """Use the raw SLAM map when it is available."""
    stem, extension = os.path.splitext(map_yaml_path)
    orig_path = f'{stem}_orig{extension}'
    return orig_path if os.path.exists(orig_path) else map_yaml_path


def map_yaml_stem(map_yaml_path: str) -> str:
    """Remove only the final extension from a map YAML path."""
    return os.path.splitext(map_yaml_path)[0]
