import utils
import torch
from my_env.space import *
import my_env.curves as crv
from my_env.my_functions.circle_to_circle_edge_distance import circle_to_circle_edge_distance
import my_env

from config_provider import ConfigProvider, dprint

@utils.count_runtime(track=ConfigProvider.track_time)
def constraint_overlap(env: "my_env.layout_env.LayoutEnv", current_circle) -> torch.Tensor:

    # 初始化惩罚值总量
    total_overlap = torch.tensor(0.0, device=ConfigProvider.device)

    spaces_considered = env.spaces[:env.step_index]

    # 对space进行广场与非广场的分类
    squares = spaces_considered[spaces_considered[:,0]==1]
    non_squares = spaces_considered[spaces_considered[:,0]!=1]


    # 使用 reversed_relu 曲线, 广场可以接受0.05距离的重叠
    square_distances = circle_to_circle_edge_distance(current_circle, squares[:, 1:])
    square_penalties = crv.crv_overlap(square_distances, 0.01, 0.05)
    total_overlap += torch.sum(square_penalties)

    # 其它空间接受0.02距离的重叠
    non_square_distances = circle_to_circle_edge_distance(current_circle, non_squares[:, 1:])
    non_square_penalties = crv.crv_overlap(non_square_distances, 0.01, 0.02)
    total_overlap += torch.sum(non_square_penalties)

    try:
        val = total_overlap.item() if torch.is_tensor(total_overlap) else total_overlap
        dprint(f"空间重叠约束 {val:.2f}")
    except Exception:
        dprint(f"空间重叠约束 {total_overlap}")
    return total_overlap


def __test1():
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

if __name__ == "__main__":

    test1()


