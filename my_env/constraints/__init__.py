import torch
import my_env

from my_env.constraints.constraint_space_to_road    import constraint_space_to_road
from my_env.constraints.constraint_square_to_road   import constraint_square_to_road
from my_env.constraints.constraint_boundary         import constraint_boundary
from my_env.constraints.constraint_overlap          import constraint_overlap

from my_env import space

from config_provider import ConfigProvider

import utils


@utils.count_runtime(track=ConfigProvider.track_time)
def cost_weighting(layoutEnv: "my_env.layout_env.LayoutEnv", action) -> torch.Tensor:

    # accumulate as float to keep dtype consistent
    cost = torch.tensor(0.0, device=ConfigProvider.device, dtype=torch.float32)

    # 本次被放置的空间类型
    space_type = layoutEnv.lay_type(layoutEnv.step_index)

    #TODO 权重未设置
    if space_type == space.Square: # 如果是广场的话
        cost = cost + 1 * constraint_square_to_road(layoutEnv, action)

    else: # 如果空间类型为其它
        cost = cost + 1 * constraint_space_to_road(layoutEnv, action)

    cost = cost + 1 * constraint_boundary(layoutEnv, action)
    cost = cost + 1 * constraint_overlap(layoutEnv, action)

    return cost.to(torch.float32)
