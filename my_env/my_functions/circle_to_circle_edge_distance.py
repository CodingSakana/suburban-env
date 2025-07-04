import math

import torch
import utils

from typing import List

@utils.count_runtime
def circle_to_circle_edge_distance(circle:torch.Tensor, circles:torch.Tensor) -> torch.tensor:
    """
    一个圆和其它n个圆之间的距离
    :param circle: tensor(3,) -> tensor(1, 3)
    :param circles: tensor(n, 3)
    :return: values: tensor(3,)
    """
    circle = circle.view(1, -1)
    circle_center = circle[:,:2]
    circles_center = circles[:,:2]
    circle_radius = circle[:,2]
    circles_radius = circles[:,2]

    circle_distance = torch.pow(circle_center - circles_center, 2)
    circle_distance = torch.sqrt(
        torch.sum(circle_distance, dim=1)
    ) - (circle_radius + circles_radius)

    return circle_distance


if __name__ == '__main__':

    circle = torch.tensor([0,0,1])
    circles = torch.tensor([[1.6,0,1], [2,2,1], [2,3,1]])

    print(
        circle_to_circle_edge_distance(circle, circles)
    )

    # for i in range(1000):
    #     circle_to_circle_edge_distance_python_loop(
    #         circle, circles
    #     )
    #
    #     circle_to_circle_edge_distance(
    #         circle, circles
    #     )