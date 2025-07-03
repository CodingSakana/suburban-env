import math

import torch
import utils

from typing import List

@utils.count_runtime
def circle_to_circle_edge_distance(circle:torch.tensor, circles:torch.tensor) -> torch.tensor:
    """
    一个圆和其它n个圆之间
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
    ) - (circle_radius + circle_radius)

    return circle_distance

@utils.count_runtime
def circle_to_circle_edge_distance_python_loop(circle:torch.tensor, circles:torch.tensor) -> torch.tensor:

    edge_distances = torch.zeros(circles.shape[0])
    for i, other_circle in enumerate(circles):
        other_x, other_y, other_r = other_circle
        dx = other_x - circle[0]
        dy = other_y - circle[1]
        center_distance = math.sqrt(dx ** 2 + dy ** 2)

        edge_distances[i] = center_distance - circle[2] - other_r

    return edge_distances


    pass

if __name__ == '__main__':

    circle = torch.tensor([0,0,1])
    circles = torch.tensor([[2,0,1], [2,2,1], [2,3,1]])

    for i in range(1000):
        circle_to_circle_edge_distance_python_loop(
            circle, circles
        )

        circle_to_circle_edge_distance(
            circle, circles
        )