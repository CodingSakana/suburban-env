import my_env.layout_env
import torch
from config_provider import ConfigProvider, dprint

from utils.crv_tester import show_curve

alpha:torch.Tensor = torch.tensor([
    #餐厅  商店  卫生间  酒店
    1.00, 0.80, 0.80, 0.20
], device=ConfigProvider.device)


def reward_wrap_squares(env: "my_env.layout_env.LayoutEnv", action):
    space_type = env.spaces[env.step_index][0].int()

    min_edge_distance_to_square = torch.min(
        env.current_to_others_distances[(env.spaces[:, 0] == 0)[:env.step_index]]
    )

    reward = alpha[space_type-1] * __crv(min_edge_distance_to_square)

    # dprint(f"min_edge_distance_to_square: {min_edge_distance_to_square}")
    # dprint(f"alpha: {alpha[space_type-1]}")
    # dprint(f"crv_result: {__crv(min_edge_distance_to_square)}")
    dprint(f"包裹广场奖励: {reward}")

    return reward



def __crv(x):
    limit = -0.04
    l1 = 0.012
    d2 = 0.1
    l2 = l1 + d2
    y = torch.where(x < limit, 0, x)
    y = torch.where((limit <= x)&(x < l1), 1, y)
    y = torch.where((l1 <= x) & (x < l2), -(1 / d2) * (x - l2), y)
    y = torch.where(l2 <= x, 0, y)

    return y



if __name__ == '__main__':

    def wrapper(x):
        return __crv(x)

    show_curve(
        wrapper,
        -0.2, 0.5, 100
    )
