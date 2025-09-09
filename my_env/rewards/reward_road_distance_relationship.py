import my_env.layout_env
import torch
import my_env.curves as crv

import utils
from config_provider import ConfigProvider, dprint

# from my_env.road_generator import calcu_p2p_road_distance


# 吸引为负值，排斥为正值
alpha_matrix: torch.Tensor = torch.tensor([
    # 广场    餐厅    商店   卫生间   酒店
    [ 0.00,  0.00,  0.00,  0.00,  0.00], # 广场
    [ 0.00,  0.00,  0.00,  0.00,  0.00], # 餐厅
    [ 0.00,  0.00,  0.00,  0.00,  0.00], # 商店
    [ 0.00,  0.00,  0.00,  1.00,  0.00], # 卫生间
    [ 0.00,  0.00,  0.00,  0.00,  0.00], # 酒店
], device=ConfigProvider.device)

d1_matrix: torch.Tensor = torch.tensor([
    # 广场    餐厅    商店   卫生间   酒店
    [ 0.00,  0.00,  0.00,  0.00,  0.00], # 广场
    [ 0.00,  0.00,  0.00,  0.00,  0.00], # 餐厅
    [ 0.00,  0.00,  0.00,  0.00,  0.00], # 商店
    [ 0.00,  0.00,  0.00,  0.00,  0.00], # 卫生间
    [ 0.00,  0.00,  0.00,  0.00,  0.00], # 酒店
], device=ConfigProvider.device)

d2_matrix: torch.Tensor = torch.tensor([
    # 广场    餐厅    商店   卫生间   酒店
    [ 0.20,  0.20,  0.20,  0.20,  0.20], # 广场
    [ 0.20,  0.20,  0.20,  0.20,  0.20], # 餐厅
    [ 0.20,  0.20,  0.20,  0.20,  0.20], # 商店
    [ 0.20,  0.20,  0.20,  0.20,  0.20], # 卫生间
    [ 0.20,  0.20,  0.20,  0.20,  0.20], # 酒店
], device=ConfigProvider.device)


@utils.count_runtime(track=ConfigProvider.track_time)
def reward_road_distance_relationship(env: "my_env.layout_env.LayoutEnv", action):
    # 占位：未启用此奖励，避免访问未定义属性
    return torch.tensor(0, device=ConfigProvider.device)

    current = env.space_param[env.step_index]
    previous = env.space_param[:env.step_index]

    distances = calcu_p2p_road_distance(current[0].to(torch.int32), previous[:, 0].to(torch.int32), current[1], previous[:, 1], env.road_param)

    dprint(f"当前点参数: {current}")
    dprint(f"distances: {distances}")

    current_space_type_index: int = int(env.spaces[env.step_index, 0].item())
    previous_spaces: torch.Tensor = env.spaces[:env.step_index, :]
    previous_types = previous_spaces[:, 0].int()

    alpha = alpha_matrix[current_space_type_index, previous_types]
    d1 = d1_matrix[current_space_type_index, previous_types]
    d2 = d2_matrix[current_space_type_index, previous_types]

    #考虑程度
    beta = crv.crv_relationship(
        current[3] + previous[:, 3], torch.tensor(-1), 0.02, 0.08
    )
    dprint(f"slice d sum {current[3] + previous[:, 3]}")
    dprint(f"beta {beta}")

    reward_vector = beta * crv.crv_relationship(distances, alpha, d1, d2)
    dprint(distances.shape, alpha.shape, d1.shape, d2.shape)
    dprint(f"reward_vector: {reward_vector}")
    dprint(f"road distance reward sum: {reward_vector.sum()}")

    return torch.tensor(0) # reward_vector.sum()
