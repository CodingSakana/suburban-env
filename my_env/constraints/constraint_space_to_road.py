
import torch
import utils
import random
import numpy as np

import my_env
import my_env.curves as crv
from config_provider import ConfigProvider, dprint

from my_env.my_functions.circle_edge_to_road_distance import circle_edge_to_road_distance
from my_env.my_functions.circle_to_circle_edge_distance import circle_to_circle_edge_distance


@utils.count_runtime(track=ConfigProvider.track_time)
def constraint_space_to_road(env: "my_env.layout_env.LayoutEnv", action: torch.Tensor) -> torch.Tensor:

    edge_distance_to_road = env.current_to_road_min_distance

    min_edge_distance_to_square = torch.min(
        env.current_to_others_distances[(env.spaces[:, 0] == 0)[:env.step_index]]
    )

    dprint(f"edge_distance_to_square: {env.current_to_others_distances[(env.spaces[:, 0] == 0)[:env.step_index]]}")

    # todo 要加上一种情况：如果无法和道路相接触呢？那就只有和其它建筑相接

    if min_edge_distance_to_square <= edge_distance_to_road:
        # 如果更临近广场 则免cost
        result = crv.crv_space_edge_to_square_edge(min_edge_distance_to_square)
        try:
            v1 = min_edge_distance_to_square.item() if torch.is_tensor(min_edge_distance_to_square) else min_edge_distance_to_square
            v2 = result.item() if torch.is_tensor(result) else result
            dprint(f"实体空间到广场 {v1:.3f} 映射到 {v2:.3f}")
        except Exception:
            dprint(f"实体空间到广场 {min_edge_distance_to_square} 映射到 {result}")
    else:
        result = crv.crv_edge_to_road_plus(edge_distance_to_road)
        try:
            v1 = edge_distance_to_road.item() if torch.is_tensor(edge_distance_to_road) else edge_distance_to_road
            v2 = result.item() if torch.is_tensor(result) else result
            dprint(f"实体空间到道路 {v1:.3f} 映射到 {v2:.3f}")
        except Exception:
            dprint(f"实体空间到道路 {edge_distance_to_road} 映射到 {result}")


    return result


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
