import my_env.layout_env
import torch
from config_provider import ConfigProvider, dprint

from utils.crv_tester import show_curve


def reward_wrap_road(env: "my_env.layout_env.LayoutEnv", action):

    if env.step_index > 3: # todo 3这个索引是写死的！
        distances = env.current_to_others_distances[3: env.step_index]

        if torch.min(distances) < 0:
            dprint("发生碰撞 奖励清空")
            return torch.tensor(0, device=ConfigProvider.device)

    current_param = env.space_param_min[env.step_index]
    reward = __crv(current_param[-1])
    dprint(f"包裹道路奖励：{reward}")

    return reward



def __crv(x):
    l1 = 0.04
    d2 = 0.1
    l2 = l1+d2
    y = torch.where(x<l1, 1, x)
    y = torch.where((l1<=x)&(x<l2), -(1/d2)*(x-l2), y)
    y = torch.where(l2<=x, 0, y)

    return y


if __name__ == '__main__':
    def wrapper(x):
        return __crv(x)

    show_curve(
        wrapper,
        0, 1, 100
    )