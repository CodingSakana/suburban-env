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
"""Offline algorithms.

Some offline algorithms require optional packages (e.g., gdown for dataset
downloads). Import them lazily so users not using offline RL won't be forced
to install extra dependencies.
"""

from contextlib import suppress

__all__ = []

with suppress(Exception):
    from omnisafe.algorithms.offline.bcq import BCQ  # type: ignore
    __all__.append('BCQ')
with suppress(Exception):
    from omnisafe.algorithms.offline.bcq_lag import BCQLag  # type: ignore
    __all__.append('BCQLag')
with suppress(Exception):
    from omnisafe.algorithms.offline.c_crr import CCRR  # type: ignore
    __all__.append('CCRR')
with suppress(Exception):
    from omnisafe.algorithms.offline.coptidice import COptiDICE  # type: ignore
    __all__.append('COptiDICE')
with suppress(Exception):
    from omnisafe.algorithms.offline.crr import CRR  # type: ignore
    __all__.append('CRR')
with suppress(Exception):
    from omnisafe.algorithms.offline.vae_bc import VAEBC  # type: ignore
    __all__.append('VAEBC')
