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
"""Off-policy algorithms.

Some algorithms (e.g., CRABS) depend on optional third-party packages
like `pytorch_lightning` and `safety_gymnasium`. We import those lazily
so users training unrelated algorithms don't have to install them.
"""

from omnisafe.algorithms.off_policy.ddpg import DDPG
from omnisafe.algorithms.off_policy.ddpg_lag import DDPGLag
from omnisafe.algorithms.off_policy.ddpg_pid import DDPGPID
from omnisafe.algorithms.off_policy.sac import SAC
from omnisafe.algorithms.off_policy.sac_lag import SACLag
from omnisafe.algorithms.off_policy.sac_pid import SACPID
from omnisafe.algorithms.off_policy.td3 import TD3
from omnisafe.algorithms.off_policy.td3_lag import TD3Lag
from omnisafe.algorithms.off_policy.td3_pid import TD3PID

# Try to import CRABS (optional). If unavailable, skip silently.
_HAS_CRABS = False
try:
    from omnisafe.algorithms.off_policy.crabs import CRABS  # type: ignore
    _HAS_CRABS = True
except Exception:
    _HAS_CRABS = False

__all__ = [
    'DDPG',
    'TD3',
    'SAC',
    'DDPGLag',
    'TD3Lag',
    'SACLag',
    'DDPGPID',
    'TD3PID',
    'SACPID',
]

if _HAS_CRABS:
    __all__.append('CRABS')
