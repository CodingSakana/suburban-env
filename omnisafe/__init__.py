# Copyright 2023 OmniSafe Team. All Rights Reserved.
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
# ==============================================================================
"""OmniSafe: A comprehensive and reliable benchmark for safe reinforcement learning."""

from contextlib import suppress


with suppress(ImportError):
    from isaacgym import gymutil

from omnisafe import algorithms
from omnisafe.algorithms import ALGORITHMS
from omnisafe.algorithms.algo_wrapper import AlgoWrapper as Agent
from omnisafe.evaluator import Evaluator
from omnisafe.version import __version__

from omnisafe.policy_network.my_policy import PolicyProvider
