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

    cond = (edge_distance_to_road > torch.tensor(-0.03, device=ConfigProvider.device)) & (
        edge_distance_to_road < torch.tensor(0.02, device=ConfigProvider.device)
    )
    result = torch.where(
        cond,
        torch.tensor(0.0, device=ConfigProvider.device, dtype=torch.float32),
        torch.tensor(1.0, device=ConfigProvider.device, dtype=torch.float32),
    )

    try:
        ed_val = edge_distance_to_road.item() if torch.is_tensor(edge_distance_to_road) else edge_distance_to_road
        res_val = result.item() if torch.is_tensor(result) else result
        dprint(f"广场到路的约束 {ed_val:.2f} 映射到 {res_val:.2f}")
    except Exception:
        dprint(f"广场到路的约束 {edge_distance_to_road} 映射到 {result}")
    return result



if __name__ == '__main__':

    pass
