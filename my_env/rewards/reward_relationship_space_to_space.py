import utils
from my_env.space import *
import my_env.curves as crv
import my_env

from my_env.my_functions.circle_to_circle_edge_distance import circle_to_circle_edge_distance
from config_provider import ConfigProvider, dprint

"""

    0: Square
    1: Restaurant
    2: Store
    3: Restroom
    4: Hotel

"""

# 吸引为负值，排斥为正值
relationship_matrix: torch.Tensor = torch.tensor([
    # 广场    餐厅    商店   卫生间   酒店
    [ 0.30, -0.50, -0.30, -0.10,  0.50], # 广场
    [-0.50,  0.00,  0.00, -0.50,  0.30], # 餐厅
    [-0.30,  0.00,  0.00,  0.00,  0.70], # 商店
    [-0.10, -0.50,  0.00,  0.50,  0.00], # 卫生间
    [ 0.50,  0.30,  0.70,  0.00,  0.30], # 酒店
], device=ConfigProvider.device)

space_margin = torch.tensor([
    #广场    餐厅    商店   卫生间   酒店
    -0.03,  0.00,  0.00,  0.00,  0.02
], device=ConfigProvider.device)


@utils.count_runtime(track=ConfigProvider.track_time)
def crv_tanh(x: torch.Tensor, p:torch.Tensor) -> torch.Tensor:
    return p * torch.tanh(5*(x-0.5))

@utils.count_runtime(track=ConfigProvider.track_time)
def crv_cos(x: torch.Tensor, p:torch.Tensor) -> torch.Tensor:
    return p * -torch.cos(torch.pi*x)



@utils.count_runtime(track=ConfigProvider.track_time or False)
def reward_relationship_space_to_space(env: "my_env.layout_env.LayoutEnv", action: torch.tensor) -> torch.Tensor:

    current_step:int = env.step_index
    current_space_type_index:int = int(env.spaces[current_step, 0].item())

    previous_spaces: torch.Tensor = env.spaces[:current_step, :]
    previous_types = previous_spaces[:, 0].int()

    distances = circle_to_circle_edge_distance(action, previous_spaces[:, 1:])

    unique_types = torch.unique(previous_types)
    mask = (previous_types.unsqueeze(1) == unique_types.unsqueeze(0))

    min_values = torch.min(torch.where(mask, distances.unsqueeze(1), 1), dim=0).values
    param = relationship_matrix[current_space_type_index][:unique_types.shape[0]]

    min_values -= space_margin[:unique_types.shape[0]] # 考虑一些margin

    reward_vector = crv.crv_relationship(min_values, param, 0.08, 0.16)
    reward_sum = reward_vector.sum() # / unique_types.shape[0]

    dprint(f"最近空间距离: {min_values}")
    dprint(f"current_space_type_index {current_space_type_index}")
    dprint(f"unique_types.shape[0]： {unique_types.shape[0]}")
    dprint(f"最近空间参数: {param}")
    dprint(f"最近空间奖励: {reward_vector} -> {reward_sum}")

    return reward_sum



if __name__ == "__main__":
    pass