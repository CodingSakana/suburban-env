
import torch
from my_env.my_functions.point_to_line_distance import distance_point_to_line_segment

def circle_edge_to_road_distance(circle_x, circle_y, circle_r, ax, ay, bx, by):
    """
    计算圆的边界到道路的最小距离。如果圆包含了部分道路，则这个值为负
    :param circle_x: 圆心x
    :param circle_y: 圆心y
    :param circle_r: 圆半径
    :param ax: 道路线段起点x （tensor）
    :param ay: 道路线段起点y （tensor）
    :param bx: 道路线段终点x （tensor）
    :param by: 道路线段终点y （tensor）
    :return:
    """
    min_distance_to_road = torch.min(
        distance_point_to_line_segment(circle_x, circle_y, ax, ay, bx, by)
    )
    edge_distance_to_road = min_distance_to_road - circle_r

    return edge_distance_to_road

def __test():
    pass

if __name__ == '__main__':

    from my_env import LayoutEnv

    roads = [
        [(0, 0.6), (0.2, 0.5), (0.6, 0.6), (1, 0.76)],
        [(0.56, 0), (0.5, 0.3), (0.6, 0.6)]
    ]
    layoutEnv = LayoutEnv(size=256, roads=roads)

    action = torch.tensor([0.57, 0.3, 0.1])

    x, y, r = action

    layoutEnv.draw_reference_action(action)
    layoutEnv.show_plot()

    print(
        circle_edge_to_road_distance(
            x, y, r,
            layoutEnv.road_slices[0],
            layoutEnv.road_slices[1],
            layoutEnv.road_slices[2],
            layoutEnv.road_slices[3],
        )
    )

