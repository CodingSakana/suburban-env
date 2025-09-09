
from my_env.map_manager.road_distance import *


def road_to_roadSlice(roads):
    """
    roads: 形式更方便维护更改 (n1, m, 2) n1条路，m个节点，2个数值代表x,y坐标
    roadSlice: 方便广播计算 (4, n2)  n2个道路线段，4个数值分别代表：第一个点坐标值 ax ay 第二个点坐标值 bx by
    """
    # 道路的shape需要转换成方便计算的形式
    # (n1, m, 2) → (4, n2)
    # road_slices = __util_road_to_roadSlice(roads)
    # road_slices = torch.tensor(road_slices, device=ConfigProvider.device).transpose(0, 1)
    return __util_road_to_roadSlice(roads)


def generate(key: str='road_real'):
    # todo 现在只是固定了一种街道

    roads = {}
    conservation_area = {}

    roads['road_mock'] =  [
        [[0.00, 0.60], [0.20, 0.50], [0.60, 0.60], [1.00, 0.76]],
        [[0.56, 0.00], [0.50, 0.30], [0.60, 0.60]]
    ]

    roads['road_real'] = [
        [[0.00, 0.66], [0.16, 0.56], [0.40, 0.16], [0.50, 0.00]],
        [[0.40, 0.16], [0.45, 0.45], [0.73, 0.72]],
        [[0.55, 1.00], [0.65, 0.80], [0.73, 0.72], [0.84, 0.60], [1.00, 0.54]],
        [[0.45, 0.45], [0.40, 0.70]],
    ]
    conservation_area['road_real'] = [

    ]

    roads['road_real_simple'] = roads['road_real'][:3]

    return roads[key]

# 把路转化为线段的形式
def __util_road_to_roadSlice(roads:List) -> List:
    road_slices = []
    for road in roads:
        for i in range(len(road) - 1):
            road_slices.append(
                [road[i][0], road[i][1], road[i + 1][0], road[i + 1][1]]
            )

    return road_slices



if __name__ == '__main__':

    # print(
    #     road_to_roadSlice(generate(), transpose=False)
    # )

    road_slices = __util_road_to_roadSlice(generate())

    for index, item in enumerate(road_slices):
        print(f"{index}: {item}")

    print()

    from road_distance import *

    # 从邻接矩阵求得最短路径
    # adjacency_matrix = build_adjacency_matrix(road_slices)
    # print(bfs_shortest_path(adjacency_matrix, 1, 6))

    road_param = build_road_parameters(road_slices)

    # 从投影值、索引、道路参数求道路距离
    # distance = calcu_p2p_road_distance(9, 5, 0.16, 0.68, road_param)
    # print(f"distance: {distance}")

    road_slices_tensor = torch.tensor(road_slices, device=ConfigProvider.device).transpose(0, 1)

    point1 = torch.tensor([0.469, 0.050], device=ConfigProvider.device)
    point2 = torch.tensor([0.570, 0.961], device=ConfigProvider.device)
    index1, t1, d1 = build_point_param(point1, road_slices_tensor)
    index2, t2, d2 = build_point_param(point2, road_slices_tensor)

    print(index1, t1, d1)
    print(index2, t2, d2)

    distance = calcu_p2p_road_distance(index1, index2, t1, t2, road_param)
    distance_check = calcu_p2p_road_distance(index2, index1, t2, t1, road_param)

    print(distance)
    print(distance_check)

    point3 = torch.tensor([0.469, 0.050], device=ConfigProvider.device)
    point4 = torch.tensor([0.570, 0.961], device=ConfigProvider.device)

    index3, t3, d3 = build_point_param(point3, road_slices_tensor)
    index4, t4, d4 = build_point_param(point4, road_slices_tensor)
    indexes = torch.tensor([index1, index2, index3, index4], device=ConfigProvider.device)
    ts = torch.tensor([t1, t2, t3, t4], device=ConfigProvider.device)
    ds = torch.tensor([d1, d2, d3, d4], device=ConfigProvider.device)
    print(
        calcu_p2p_road_distance(indexes[0], indexes[1:], ts[0], ts[1:], road_param)
    )
    print(calcu_p2p_road_distance(index2, index2, t2, t2, road_param))

    point5 = torch.tensor([0.195, 0.502], device=ConfigProvider.device)
    point6 = torch.tensor([0.354, 0.237], device=ConfigProvider.device)
    index5, t5, d5 = build_point_param(point5, road_slices_tensor)
    index6, t6, d6 = build_point_param(point6, road_slices_tensor)
    print(
        calcu_p2p_road_distance(index5, index6, t5, t6, road_param)
    )

    center_point = torch.tensor([0.530, 0.527], device=ConfigProvider.device)
    print(
        build_point_param(center_point, road_slices_tensor)
    )
