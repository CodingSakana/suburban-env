import torch
import my_env
import utils
from config_provider import ConfigProvider, dprint
from utils.running_time_tester import RunningTimeTester


# get first_index from last_index
first_index = {
    8: 3,
    23: 9,
    25: 24,
}


def reward_cluster(env: "my_env.layout_env.LayoutEnv", action: torch.Tensor) -> torch.Tensor:

    dprint(f"clustering on {env.spaces[first_index[env.step_index]:env.step_index]}")

    return torch.tensor(0)




@utils.count_runtime()
def __test():
    i = 2
    if i in [1,2,3,4]:
        return i

tensor_a = torch.tensor([1, 2, 3, 4])
@utils.count_runtime()
def __test2():
    i = 2
    if torch.isin(i, tensor_a):
        return i



if __name__ == '__main__':

    def wrapper(func):
        func()

    RunningTimeTester(
        [__test, __test2],
        wrapper,
    ).test()