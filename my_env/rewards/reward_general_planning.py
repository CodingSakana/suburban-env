import torch
import my_env
from config_provider import ConfigProvider, dprint
from my_env.map_manager import calcu_p2p_road_distance

# todo 以下的全局变量和道路有关！现在在代码中写死了
center_point_slice_index = torch.tensor(4, device=ConfigProvider.device)
center_point_t = torch.tensor([0.2855], device=ConfigProvider.device)
max_distance_to_center_point = 1 #1.07


# def function_n(x):
#     """
#     Args:
#         x: distance to the CENTER_POINT
#     Returns: n: parallel road count of distance_x
#     """
#     # parallel_road_count
#     prc = torch.tensor([
#         [0.11, 2],  # [0.11, 2],
#         [0.17, 3],  # [0.17, 3],
#         [0.09, 4],  # [0.09, 4],
#         [0.04, 3],  # [0.04, 3],
#         [0.19, 4],  # [0.19, 4],
#         [0.02, 3],  # [0.02, 3],
#         [0.45, 1],  # [0.45, 1], #
#     ], device=ConfigProvider.device)


parallel_road_limit = torch.tensor([
    0.11, 0.28, 0.37, 0.41, 0.60, 0.62
], device=ConfigProvider.device)

parallel_road_count = torch.tensor([
    2, 3, 4, 3, 4, 3, 1
], device=ConfigProvider.device)

divide_length = 3
avg_boundaries = torch.linspace(0, 1, divide_length+1, device=ConfigProvider.device)[1:-1]


def reward_general_planning(env: "my_env.layout_env.LayoutEnv", action: torch.tensor) -> torch.Tensor:
    """在最后一次布置完成后，返回最终布局奖励"""

    # todo 尚未完成
    # current_param = env.space_param[env.step_index]
    # distance_to_center_point = calcu_p2p_road_distance(
    #     current_param[0].int(), center_point_slice_index,
    #     current_param[1], center_point_t,
    #     env.road_param
    # )
    #
    # dprint(f"distance_to_center_point: {distance_to_center_point}")

    limit_floor = 0
    limit_ceil = 0.05
    all_space_flatten = env.space_param_all[:env.step_index+1].flatten(start_dim=0, end_dim=1) # [:env.step_index+1]
    valid_param = all_space_flatten[(limit_floor < all_space_flatten[:, 3]) & (all_space_flatten[:, 3] < limit_ceil)]
    # valid_param = env.space_param_min[(limit_floor < env.space_param_min[:, 3]) & (env.space_param_min[:, 3] < limit_ceil)]

    distances_to_center_point = calcu_p2p_road_distance(
        center_point_slice_index, valid_param[:, 0].int(),
        center_point_t, valid_param[:, 1],
        env.road_param
    )

    score_bucket = torch.zeros(divide_length, device=ConfigProvider.device)


    dprint(f"raw_distances: {distances_to_center_point.sort()}")
    hist = torch.histc(distances_to_center_point, bins=10, min=0, max=max_distance_to_center_point)
    dprint(f"center hist: {hist}")

    indices = torch.bucketize(distances_to_center_point, parallel_road_limit)

    hyper = torch.tensor([0., 1, 0.8, 0.6, 0.4], device=ConfigProvider.device)
    # hyper = torch.tensor([0., 1, 1, 1, 1], device=ConfigProvider.device)
    washed = hyper[parallel_road_count[indices]]
    indices_avg = torch.bucketize(distances_to_center_point, avg_boundaries)

    # score_bucket[indices_avg] = washed
    score_bucket.scatter_add_(0, indices_avg, washed)

    dprint(f"score_bucket: {score_bucket}")
    dprint(f"valid_length: {len(valid_param)}")
    # __show_score_bucket(score_bucket)

    alpha = torch.tensor([1, 0.75, 0.5], device=ConfigProvider.device)
    reward_vector = alpha * score_bucket

    return reward_vector.sum()


def __show_score_bucket(score_bucket: torch.Tensor):

    import matplotlib.pyplot as plt

    # 将张量转换为 NumPy 数组
    tensor_np = score_bucket.numpy()

    # 使用 matplotlib 绘制折线图
    plt.plot(tensor_np, marker='o')  # 添加标记点，便于观察
    plt.title("Tensor Visualization")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.grid(True)  # 添加网格线
    plt.show()