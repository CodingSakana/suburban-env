
import torch

from typing import Callable
from matplotlib import pyplot as plt


def show_curve(crv:Callable[[torch.Tensor], torch.Tensor], floor:float, ceil:float, nums:int):
    x = torch.linspace(floor, ceil, nums)
    y = crv(x)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(x, y, color="black", linewidth=0.8)
    plt.show()


if __name__ == '__main__':

    def crv_segmented(x: torch.Tensor) -> torch.Tensor:
        # l1 = 0.05
        # l2 = 0.1
        # l3 = l2 + 0.5
        #
        # x = torch.where(x < l1, 1, x)
        # x = torch.where((l1 <= x) & (x < l2), 0, x)
        # x = torch.where((l2 <= x) & (x < l3), 2 * (x - l2), x)
        # x = torch.where(l3 <= x, 1, x)
        return x

    show_curve(crv_segmented, 0, 1, 100)