import torch
from my_env.my_functions.space_protection import space_protection
import utils
from config_provider import ConfigProvider
from utils.running_time_tester import RunningTimeTester
from utils.stopwatch import Stopwatch


@utils.count_runtime(track=ConfigProvider.track_time, threshold=1e4)
def constraint_space_conservation(p_c, p_tri, r_c, r_margin) -> torch.Tensor:
    assert p_c.shape == (2,)
    assert p_tri.dim() == 3
    assert p_tri.shape[1] == 3
    assert p_tri.shape[2] == 2
    assert r_c.shape == (1,)
    assert r_margin.shape == (1,)

    stopwatch = Stopwatch("test111")

    # p_c_expanded = p_c.view(1,1,2)
    v_triangle = torch.stack([
        p_tri[:, 1] - p_tri[:, 0],
        p_tri[:, 2] - p_tri[:, 1],
        p_tri[:, 0] - p_tri[:, 2]
    ], dim=1)

    stopwatch.press("1")

    # distances = []
    # for i in range(3):
    #     p_tri_i = p_tri[:, i] #(N,2)
    #     v_tri_i = v_triangle[:, i] #(N,2)
    #
    #     v1 = p_c - p_tri_i
    #     v2 = v_tri_i
    #     #计算投影长度
    #     dot_product = torch.sum(v1 * v2, dim=1)
    #     v2_norm_squared = torch.sum(v2 * v2, dim=1)
    #     #0到1内的限制
    #     t_value = torch.clamp(dot_product / v2_norm_squared, 0.0, 1.0) #(N,)
    #
    #     projected_point = p_tri_i + t_value.unsqueeze(1) * v2 #(N,2)
    #
    #     distance = torch.norm(projected_point - p_c.unsqueeze(0), dim=1)
    #     distances.append(distance)

    p_tri_points = p_tri  #(N,3,2)
    v_tri = v_triangle  #(N,3,2)
    v1 = p_c - p_tri_points  #(N,3,2)
    v2 = v_tri  #(N,3,2)

    dot_product = torch.sum(v1 * v2, dim=2)  #(N,3)
    v2_norm_squared = torch.sum(v2 * v2, dim=2)  #(N,3)
    t_value = torch.clamp(dot_product / v2_norm_squared, 0.0, 1.0)  #(N,3)

    stopwatch.press("2")

    # 计算投影点并得到距离
    projected_points = p_tri_points + t_value.unsqueeze(2) * v2  #(N,3,2)
    p_c_batch = p_c.unsqueeze(0).unsqueeze(1)  #(1,1,2)
    distances = torch.norm(projected_points - p_c_batch, dim=2)  #(N,3)

    d_mindis = torch.min(distances, dim=1).values #(N,)

    r_max = r_c + r_margin

    stopwatch.press("3")

    value1 = d_mindis - r_max #(N,)
    v_flag = space_protection(p_c,r_margin, v_triangle)
    value2 = -v_flag * (r_max + torch.min(d_mindis, r_max)) #(N,)

    q = torch.min(value1, value2)

    stopwatch.press("4")

    cost = torch.where(
        q <= -r_margin,
        torch.tensor(1.0, device=q.device),
        torch.where(
            q >= 0,
            torch.tensor(0.0, device=q.device),
            (1.0 / -r_margin) * q
        )
    )

    stopwatch.press("5")

    return cost

if __name__ == '__main__':
    p_c = torch.tensor([1.0, 2.0])
    p_tri = torch.tensor([
        [[0.0, 0.0], [2.0, 0.0], [1.0, 2.0]],
        [[-1.0, -1.0], [1.0, -1.0], [0.0, 1.0]],
        [[0.0, 0.0], [0.0, 1.0], [2.0, 1.0]],
    ])
    r_c = torch.tensor([1.0])
    r_margin = torch.tensor([0.5])

    # 调用函数
    cost = constraint_space_conservation(p_c, p_tri, r_c, r_margin)
    print("Cost:", cost)

    RunningTimeTester(
        test_functions=[constraint_space_conservation],
        test_wrapper=lambda func: func(p_c, p_tri, r_c, r_margin),
        times=500
    ).test()



