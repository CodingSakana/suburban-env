import torch
import utils
from config_provider import ConfigProvider


@utils.count_runtime(track=ConfigProvider.track_time, threshold=1e4)
def space_protection(point: torch.Tensor, margin: torch.Tensor, triangles: torch.Tensor) -> torch.Tensor:
    assert point.shape == (2,)
    assert margin.shape == (1,)
    assert triangles.dim() == 3
    assert triangles.shape[1] == 3
    assert triangles.shape[2] == 2

    point_expanded = point.view(1,1,2)
    v_point = triangles - point_expanded

    cross_12 = v_point[:, 0, 0] * v_point[:, 1, 1] - v_point[:, 0, 1] * v_point[:, 1, 0]
    cross_23 = v_point[:, 1, 0] * v_point[:, 2, 1] - v_point[:, 1, 1] * v_point[:, 2, 0]
    cross_31 = v_point[:, 2, 0] * v_point[:, 0, 1] - v_point[:, 2, 1] * v_point[:, 0, 0]
    v_cross = torch.stack([cross_12, cross_23, cross_31], dim=1)

    signs = torch.sign(v_cross)
    signs_sum = torch.sum(signs, dim=1)
    v_flag = torch.max(torch.zeros_like(signs_sum), signs_sum - 2)
    print(v_flag.shape)
    return v_flag

if __name__ == '__main__':

    point = torch.tensor([0.4, 0.2])
    margin = torch.tensor([0.1])
    triangles = torch.tensor([
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
        [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    ])

    # point_on_edge = torch.tensor([0.5, 0.0])
    # triangles_on_edge = torch.tensor([
    #     [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
    #     [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
    #     [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    # ])
    #
    # point_outside = torch.tensor([2.0, 2.0])
    # triangles_outside = torch.tensor([
    #     [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
    #     [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
    #     [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    # ])

    print("测试点在三角形内部：")
    print(space_protection(point, margin, triangles))

    # RunningTimeTester(
    #     test_functions=[space_protection],
    #     test_wrapper=lambda func: func(point, margin, triangles),
    #     times=500
    # ).test()

    # print("测试点在三角形边界上：")
    # print(space_protection(point_on_edge, margin, triangles_on_edge))
    #
    # print("测试点在三角形外部：")
    # print(space_protection(point_outside, margin, triangles_outside))