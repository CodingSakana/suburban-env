
import torch
import my_env
import my_env.curves as crv
from my_env.curves import show_curve

from my_env.device_provider import DeviceProvider


def constraint_boundary(self: "my_env.layout_env.LayoutEnv", action) -> torch.Tensor:
    """
    圆不能超出边界
    todo 并且如果有必要的话 在临近边界的地方就给cost
    :param self:
    :param action:
    :return:
    """

    xy = torch.tensor([action[0], action[1]], device=DeviceProvider.device).view(1, -1)
    r = action[2]
    arg = torch.tensor([[0,0],[1,1]], device=DeviceProvider.device)
    distance = torch.abs(xy - arg) - r
    distance_sum = distance[distance<0].sum()

    return crv.crvDebug(
        "空间到边界的距离",
        crv.relu_reverse(2, 0.1),
        distance_sum
    )


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

    show_curve(
        crv.relu_reverse(-0.02, 0.02, 100, 100),
        -1, 1
    )