
import torch
import utils
import random
import numpy as np

import my_env
import my_env.curves as crvs

from my_env.my_functions.circle_edge_to_road_distance import circle_edge_to_road_distance


@utils.count_runtime
def constraint_space_to_road(self: "my_env.layout_env.LayoutEnv", action) -> torch.Tensor:

    px = action[0]
    py = action[1]
    r = action[2]

    ax = self.road_slices[0]
    ay = self.road_slices[1]
    bx = self.road_slices[2]
    by = self.road_slices[3]

    edge_distance_to_road = circle_edge_to_road_distance(px, py, r, ax, ay, bx, by)

    # todo 加上有广场的情况
    return crvs.crvDebug(
        "边界到道路的约束",
        crvs.index_sweetZone(0.03, 0.04, 1e8, 10),
        # crvs.quadratic_sweetZone(0.03, 0.04),
        edge_distance_to_road
    )


def __test(sample_times=10240):
    """仅供测试"""

    from my_env import LayoutEnv

    @utils.count_runtime
    def random_test(layoutEnv: LayoutEnv, radius=0.01, num=sample_times):
        actions = [
            torch.tensor([random.random(), random.random(), radius])
            for _ in range(num)
        ]
        results = np.array([
            constraint_space_to_road(layoutEnv, action, ) for action in actions
        ])

        colors = utils.debug.map_value_to_color(results, ['#dddddd', '#ff0000'])
        colors_rgb = (colors[:, :-1] * 255).astype(int)
        colors_bgr = colors_rgb[:, ::-1]

        for a, c in zip(actions, colors_bgr):
            layoutEnv.draw_reference_action(a, c.tolist(), fix_radius=3)

        layoutEnv.show_plot()

    roads = [
        [(0, 0.6), (0.2, 0.5), (0.6, 0.6), (1, 0.76)],
        [(0.56, 0), (0.5, 0.3), (0.6, 0.6)]
    ]
    layoutEnv = LayoutEnv(size=256, roads=roads)

    random_test(layoutEnv)


if __name__ == '__main__':
    __test()