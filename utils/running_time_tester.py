import torch
import utils
import tqdm
import random

from typing import Callable, List

class RunningTimeTester:

    def __init__(self, test_functions:List[Callable], test_wrapper:Callable, times=1000):
        self.test_functions = test_functions
        self.test_wrapper = test_wrapper
        self.times = times

    def test(self):
        for _ in tqdm.tqdm(range(self.times)):
            dice = random.choice(self.test_functions)
            self.test_wrapper(dice)



if __name__ == '__main__':
    pass
    # start = time.process_time_ns()
    # a = torch.ones([3000, 400])
    # print(f"process ends in {time.process_time_ns() - start} ns")
    #
    # start = time.process_time_ns()
    # a = torch.tensor([[0 for _ in range(400)] for _ in range(3000)])
    # print(f"process ends in {time.process_time_ns() - start:,} ns")

    # import tqdm, random

    @utils.count_runtime()
    def method1(x: torch.Tensor):
        result1 = torch.where(x < 0, x - 1, x)
        return result1

    @utils.count_runtime()
    def method1_plus(x: torch.Tensor):
        condition = x < 0
        result1 = torch.where(condition, x - 1, x)
        return result1

    @utils.count_runtime()
    def method2(x: torch.Tensor):
        x_minus_1 = x - 1
        result2 = torch.where(x < 0, x_minus_1, x)
        return result2

    @utils.count_runtime()
    def method3(x: torch.Tensor):
        condition = x < 0
        x[condition] = x[condition] - 1
        return x

    def wrapper(func):
        temp = torch.randn(100)
        func(temp)

    RunningTimeTester(
        test_functions=[method1, method2, method3],
        test_wrapper=wrapper,
        times=300
    ).test()