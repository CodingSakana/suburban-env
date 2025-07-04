import torch
import utils

import my_env
import my_env.curves as crv
from config_provider import ConfigProvider, dprint

from my_env.my_functions.circle_edge_to_road_distance import circle_edge_to_road_distance



@utils.count_runtime(track=ConfigProvider.track_time)
def constraint_square_to_road(env: "my_env.layout_env.LayoutEnv", action) -> torch.Tensor:

    edge_distance_to_road = env.current_to_road_min_distance


    # result = crv.crv_edge_to_road_plus(edge_distance_to_road)

    result = torch.where(-0.03 < edge_distance_to_road < 0.02, 0, 1)

    dprint(f"广场到路的约束 {edge_distance_to_road:.2f} 映射到 {result:.2f}")
    return result



if __name__ == '__main__':

    pass