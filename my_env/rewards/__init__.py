import torch

from my_env.rewards.reward_general_planning import reward_general_planning
from my_env.rewards.reward_relationship_space_to_space import reward_relationship_space_to_space

import my_env.layout_env
from my_env.device_provider import DeviceProvider

def weighting(layoutEnv: "my_env.layout_env.LayoutEnv", action) -> torch.Tensor:

    reward = torch.tensor(0, device=DeviceProvider.device)

    # TODO 权重未设置
    reward = reward + 1 * reward_relationship_space_to_space(layoutEnv, action)

    # 最后一步的奖励
    if layoutEnv.step_index == layoutEnv.max_step:
        reward = reward + 1 * reward_general_planning(layoutEnv, action)

    return reward