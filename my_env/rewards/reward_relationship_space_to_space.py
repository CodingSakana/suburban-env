import torch
import math
from my_env.space import *
import my_env.curves as crv
import my_env


from typing import Type, Union
from my_env.my_functions.circle_to_circle_edge_distance import circle_to_circle_edge_distance

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
    [ 0.30, -0.30, -0.30, -0.10,  0.50], # 广场
    [-0.30,  0.00,  0.00, -0.50,  0.30], # 餐厅
    [-0.30,  0.00,  0.00,  0.00,  0.70], # 酒店
    [-0.10, -0.50,  0.00,  0.50,  0.00], # 卫生间
    [ 0.50,  0.30,  0.70,  0.00,  0.30], # 酒店
])

def space_type_to_index(space_type: Type[Union[Square, Restaurant, Store, Restroom, Hotel]]) -> int:
    return {
        Square:     0,
        Restaurant: 1,
        Store:      2,
        Restroom:   3,
        Hotel:      4,
    }[space_type]

def reward_relationship_space_to_space(self: "my_env.layout_env.LayoutEnv", action: torch.tensor) -> torch.Tensor:

    """
    >>> Square.get_space_type_index()
    0
    """

    current_step:int = self.step_index
    current_space_type_index:int = self.spaces[current_step, 0].item()
    data_store = self.spaces
    n = len(data_store)
    circles = data_store[:,-3:]
    circle_distance = circle_to_circle_edge_distance(action, circles)

    # 初始化奖励矩阵
    reward_matrix = torch.zeros(n, dtype=torch.float32)

    # 遍历所有建筑对 // todo 是否可以改成 broadcast 的方式？
    for i in range(n):
        type_i = data_store[i, 0].item()
        relation = relationship_matrix[type_i, current_space_type_index].item()
        distance = circle_distance[i]
        if relation == 0:
            continue  # 无约束关系跳过
        if relation < 0:
            reward = torch.tanh((distance - relation / 2) / 10)
            reward_matrix[i] = reward * 20
        elif relation > 0:
            penalty = torch.tanh((relation / 2 - distance) / 10)
            reward_matrix[i] = -penalty * 20

    # 计算总奖励（上三角矩阵的和）
    total_reward = torch.sum(reward_matrix)
    return total_reward
