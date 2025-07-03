

import torch
import utils
import random
import numpy as np
from my_env.space import *
import my_env.curves as crv
from my_env.my_functions.circle_to_circle_edge_distance import circle_to_circle_edge_distance
import my_env

from my_env.device_provider import DeviceProvider

def constraint_overlap(self: "my_env.layout_env.LayoutEnv", action) -> torch.Tensor:
    # # 判断空间是否为空
    # if not self.spaces:
    #     return torch.tensor(0.0)
    # 初始化惩罚值总量
    total_overlap = torch.tensor(0.0, device=DeviceProvider.device)
    # 将action转换为torch结构
    current_x, current_y, current_r = action[0], action[1], action[2]

    current_circle = torch.stack([current_x, current_y, current_r])

    # 对space进行广场与非广场的分类

    squares = [space for space in self.spaces if isinstance(space, Square)]
    non_squares = [space for space in self.spaces if not isinstance(space, Square)]

    if squares:
        # 自定义relu_reverse参数
        penalty_fn = crv.relu_reverse(base=3, switch=-0.4)
        # 将广场转换为张量 (n, 3)
        squares_tensor = torch.tensor([[space.x, space.y, space.radius] for space in squares])
        distances = circle_to_circle_edge_distance(current_circle, squares_tensor)
        # 将distances标量转换为张量并应用relu_reverse惩罚
        penalties = torch.tensor([penalty_fn(d.item()) for d in distances])
        total_overlap += torch.sum(penalties)

    if non_squares:
        # 自定义relu_reverse参数
        penalty_fn = crv.relu_reverse(base=3, switch=-0)
        # 将非广场转换为张量 (n, 3)
        non_squares_tensor = torch.tensor([[space[1], space[2], space[3]] for space in non_squares], device=DeviceProvider.device)
        distances = circle_to_circle_edge_distance(current_circle, non_squares_tensor)
        # 将distances标量转换为张量并应用relu_reverse惩罚
        penalties = torch.tensor([penalty_fn(d.item()) for d in distances], device=DeviceProvider.device)
        total_overlap += torch.sum(penalties)

    return total_overlap


if __name__ == "__main__":
    # 初始化一个假的环境对象（模拟 LayoutEnv）
    class MockEnv:
        def __init__(self):
            self.spaces = []


    env = MockEnv()

    # 测试场景1：添加一个广场（Square）
    env.spaces.append(Square(x=0.5, y=0.5, radius=0.1))
    action = torch.tensor([0.5, 0.5, 0.1])
    penalty = constraint_overlap(env, action)
    print("[广场完全重叠] cost:", penalty.item())

    # 测试场景2：添加一个餐厅（Restaurant），此时餐厅与广场大小一致
    env.spaces = [Restaurant(x=0.5, y=0.5, radius=0.1)]
    action = torch.tensor([0.5, 0.5, 0.1])
    penalty = constraint_overlap(env, action)
    print("[餐厅完全重叠] cost:", penalty.item())


