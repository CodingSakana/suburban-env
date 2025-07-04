import my_env.curves as crv
from config_provider import ConfigProvider

import torch

# @utils.count_runtime(track=ConfigProvider.debug, threshold=1e4)
def reversed_relu(switch:float=0):
    """
    反过来的类似relu的曲线
    :param switch:
    :return:
    """
    def wrap(x:float):
        return max(0., (-x+switch))
    return wrap


def reversed_relu_tensor(switch:float=0):
    switch_tensor = torch.tensor(switch, device=ConfigProvider.device)
    def wrap(x:torch.Tensor):
        return torch.max(
            torch.tensor(0, device=ConfigProvider.device),
            -x + switch_tensor
        )
    return wrap


def reversed_index_relu(base:float=2, switch:float=0):
    """
    反过来的类似relu的曲线
    :param base:
    :param switch:
    :return:
    """
    def wrap(x:float):
        x *= 5
        if x < switch:
            return base**(-x+switch) - 1
        return 0
    return wrap

if __name__ == '__main__':
    crv.show_curve(
        reversed_relu(0.5),
        -1, 1, num=200
    )

    # torch.vmap(crvs.reversed_relu(switch=-0.05))(torch.tensor([-1, 0, 1, 2]))
