
import torch
import utils
import random
import numpy as np

import my_env
import my_env.curves as crvs
from my_env.curves import show_curve

from my_env.my_functions.circle_edge_to_road_distance import circle_edge_to_road_distance


@utils.count_runtime
def constraint_square_to_road(self: "my_env.layout_env.LayoutEnv", action) -> torch.Tensor:

    px = action[0]
    py = action[1]
    r = action[2]

    ax = self.road_slices[0]
    ay = self.road_slices[1]
    bx = self.road_slices[2]
    by = self.road_slices[3]

    edge_distance_to_road = circle_edge_to_road_distance(px, py, r, ax, ay, bx, by)


    return crvs.crvDebug(
        "广场边界到道路的约束",
        crvs.index_sweetZone(-0.02, 0.02, 100, 100),
        edge_distance_to_road
    )



if __name__ == '__main__':
    show_curve(
        crvs.index_sweetZone(-0.02, 0.02, 100, 100),
        -0.1, 0.1
    )