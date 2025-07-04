import math
from typing import List

import torch

import utils
from config_provider import ConfigProvider, dprint
from collections import deque


def build_adjacency_matrix(road_slices):

    def adjacency_core(slice_i, slice_j):
        if slice_i[:2] == slice_j[:2]:
            return True
        elif slice_i[2:] == slice_j[:2]:
            return True
        elif slice_i[2:] == slice_j[2:]:
            return True
        elif slice_i[2:] == slice_j[:2]:
            return True
        return False

    adjacency_matrix = torch.zeros(len(road_slices), len(road_slices), device=ConfigProvider.device)
    for i in range(len(road_slices)):
        for j in range(i+1, len(road_slices)):
            if adjacency_core(road_slices[i], road_slices[j]):
                # print(i, j, road_slices[i], road_slices[j])
                adjacency_matrix[i][j] = 1

    for i in range(len(road_slices)):
        for j in range(0, i):
            adjacency_matrix[i][j] = adjacency_matrix[j][i]

    return adjacency_matrix


def build_road_parameters(road_slices):

    adjacency_matrix = build_adjacency_matrix(road_slices)

    slice_lengths = torch.zeros(len(road_slices), device=ConfigProvider.device)
    for index, item in enumerate(road_slices):
        slice_lengths[index] = math.sqrt((item[0] - item[2]) ** 2 + (item[1] - item[3]) ** 2)

    def param_core(tip_slice, middle_slice):
        """端头的起始坐标 是否在 中间段的坐标中"""
        if tip_slice[:2] == middle_slice[:2]:
            return True
        elif tip_slice[:2] == middle_slice[2:]:
            return True
        return False

    distance_matrix = torch.zeros(len(road_slices), len(road_slices), device=ConfigProvider.device)
    param_matrix = torch.ones(len(road_slices), len(road_slices), device=ConfigProvider.device)
    for i in range(len(road_slices)):
        for j in range(i, len(road_slices)):
            paths = bfs_shortest_path(adjacency_matrix, i, j)
            distance = sum([slice_lengths[i] for i in paths])
            distance_matrix[i][j] = distance

            # 组装 param_matrix
            if i != j:
                assert len(paths) >= 2
                if param_core(road_slices[i], road_slices[paths[1]]):
                    param_matrix[i][j] = 0
                if param_core(road_slices[j], road_slices[paths[-2]]):
                    param_matrix[j][i] = 0

    for i in range(len(road_slices)):
        for j in range(0, i):
            distance_matrix[i][j] = distance_matrix[j][i]

    params = {
        'lengths': slice_lengths,
        'distance_matrix': distance_matrix,
        'param_matrix': param_matrix,
    }

    return params

@utils.count_runtime()
def calcu_p2p_road_distance(begin_index:torch.Tensor, end_index:torch.Tensor, begin_t, end_t, road_param):

    slice_lengths = road_param['lengths']
    distance_matrix = road_param['distance_matrix']
    param_matrix = road_param['param_matrix']

    distance: torch.Tensor = distance_matrix[begin_index, end_index]#.unsqueeze(0)

    def calcu_alpha(a, flag):
        return a*flag + (1-a)*(1-flag)

    inplace_param = torch.where(end_index==begin_index, 0, 1)

    # 千万别写 distance -= something, 被坑惨了
    distance = distance - slice_lengths[begin_index] * calcu_alpha(begin_t, param_matrix[begin_index, end_index])
    distance = distance - slice_lengths[end_index] * calcu_alpha(end_t, param_matrix[end_index, begin_index] * inplace_param)

    return distance.abs()


@utils.count_runtime()
def build_point_param(point, road_slices_tensor):
    """
    Args:
        point: (2, )
        road_slices_tensor: (4， n)
    Returns:
    """

    px = point[0]
    py = point[1]
    ax = road_slices_tensor[0]
    ay = road_slices_tensor[1]
    bx = road_slices_tensor[2]
    by = road_slices_tensor[3]

    dx1, dy1 = (bx - ax).view(-1, 1), (by - ay).view(-1, 1)
    dx2, dy2 = px.view(1, -1) - ax.view(-1, 1), py.view(1, -1) - ay.view(-1, 1)
    length_squared = dx1 ** 2 + dy1 ** 2

    # 投影标量t
    t = torch.max(torch.tensor(0), torch.min(torch.tensor(1), (dx2 * dx1 + dy2 * dy1) / length_squared))

    # 投影坐标
    projx = ax.reshape(-1, 1) + t * dx1
    projy = ay.reshape(-1, 1) + t * dy1

    # P到投影点距离
    d = torch.sqrt((px.reshape(1, -1) - projx) ** 2 + (py.reshape(1, -1) - projy) ** 2)  # (k,n)

    min_index = torch.argmin(d)
    d_min = d[min_index]
    t_min = t[min_index]

    all_index = torch.arange(road_slices_tensor.size(1), device=ConfigProvider.device)

    return min_index, t_min, d_min, all_index, t[:,0], d[:,0]



def bfs_shortest_path(adj_matrix, start, end):
    n = len(adj_matrix)
    visited = [False] * n
    queue = deque([(start, [start])])  # 存储节点和到达该节点的路径
    visited[start] = True

    while queue:
        current, path = queue.popleft()
        if current == end:
            return path  # 找到最短路径

        for i in range(n):
            if adj_matrix[current][i] == 1 and not visited[i]:
                queue.append((i, path + [i]))
                visited[i] = True

    return None  # 如果没有路径，返回None

