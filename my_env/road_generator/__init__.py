import torch

from my_env.device_provider import DeviceProvider


def road_to_roadSlice(roads):
    """
    roads: 形式更方便维护更改 (n, m, 2) n条路，m个节点，2个数值代表x,y坐标
    roadSlice: 方便广播计算 (a, 4)  a个道路线段，4个数值分别代表：第一个点坐标值 ax ay 第二个点坐标值 bx by
    """
    road_slices = []
    # 道路的shape需要转换成方便计算的形式
    # (n, m, 2) → (n1, 4)
    for road in roads:
        for pi in range(len(road) - 1):
            road_slices.append(
                [road[pi][0], road[pi][1], road[pi + 1][0], road[pi + 1][1]]
            )

    road_slices = torch.tensor(road_slices, device=DeviceProvider.device).transpose(0, 1)
    return road_slices


def generate(seed=None):
    # todo 现在只是固定了一种街道
    return [
        [[0, 0.6], [0.2, 0.5], [0.6, 0.6], [1, 0.76]],
        [[0.56, 0], [0.5, 0.3], [0.6, 0.6]]
    ]

if __name__ == '__main__':

    print(
        road_to_roadSlice(generate())
    )