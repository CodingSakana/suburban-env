
import torch
import my_env
import my_env.curves as crv
import utils

from config_provider import ConfigProvider, dprint


@utils.count_runtime(track=ConfigProvider.track_time)
def constraint_boundary(env: "my_env.layout_env.LayoutEnv", action) -> torch.Tensor:
    """
    圆不能超出边界
    :param env:
    :param action:
    :return:
    """

    margin = 0.02

    xy = torch.tensor([action[0], action[1]], device=ConfigProvider.device).view(1, -1)
    r = action[2]
    arg = torch.tensor([[0,0],[1,1]], device=ConfigProvider.device)
    distance = torch.abs(xy - arg) - (r + margin)
    distance_min = distance.min()

    result = crv.crv_boundary(distance_min, r, margin)

    try:
        dmin_val = distance_min.item() if torch.is_tensor(distance_min) else distance_min
        res_val = result.item() if torch.is_tensor(result) else result
        dprint(f"空间到边界 {dmin_val:.2f} 映射到 {res_val:.2f}")
    except Exception:
        dprint(f"空间到边界 {distance_min} 映射到 {result}")
    return result


def __test():
    action = torch.tensor([0.2, 0.2, 0.18])
    print(
        constraint_boundary(
            None, action
        )
    )

    from my_env.layout_env import LayoutEnv
    layout = LayoutEnv()
    layout.draw_reference_action(action)
    layout.show_plot()


if __name__ == '__main__':

    pass
