import torch

from my_env.rewards.reward_cluster                      import reward_cluster
from my_env.rewards.reward_general_planning             import reward_general_planning
from my_env.rewards.reward_relationship_space_to_space  import reward_relationship_space_to_space
from my_env.rewards.reward_road_distance_relationship   import reward_road_distance_relationship
from my_env.rewards.reward_wrap_squares                 import reward_wrap_squares

import my_env.layout_env
from config_provider import ConfigProvider, dprint

import utils

@utils.count_runtime(track=ConfigProvider.track_time)
def reward_weighting(layoutEnv: "my_env.layout_env.LayoutEnv", action) -> torch.Tensor:
    # 本次被放置的空间类型
    space_type = layoutEnv.spaces[layoutEnv.step_index][0]

    # accumulate as float to keep dtype consistent
    reward = torch.tensor(0.0, device=ConfigProvider.device, dtype=torch.float32)

    # TODO 权重未设置

    reward = reward + 0.2 * reward_area(action)

    if layoutEnv.step_index >= 1:
        reward = reward + 1 * reward_relationship_space_to_space(layoutEnv, action)

    if layoutEnv.step_index == 25: # todo 这个不会根据空间配置更改而更改
        reward = reward + 4 * reward_road_distance_relationship(layoutEnv, action)

    if space_type != 0:
        reward = reward + 0.5 * reward_wrap_squares(layoutEnv, action)

    if layoutEnv.step_index in [8, 23, 25]: # todo 这个不会根据空间配置更改而更改
        reward = reward + 1 * reward_cluster(layoutEnv, action)

    # 最后一步的奖励
    if layoutEnv.step_index == layoutEnv.max_step:
        reward = reward + 0.5 * reward_general_planning(layoutEnv, action)

    return reward.to(torch.float32)



def reward_area(action:torch.Tensor) -> torch.Tensor:
    return torch.max(
        torch.tensor(0.0, device=ConfigProvider.device, dtype=torch.float32),
        (action[2] - 0.5).abs().to(torch.float32),
    )
