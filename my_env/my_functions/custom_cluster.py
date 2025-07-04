import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import utils
from utils.running_time_tester import RunningTimeTester


def __draw_circles(plt_obj, circles, color):
    """
    在给定的 plt 对象上绘制圆形。

    参数:
        plt_obj: Matplotlib 的 plt 对象
        circles: torch.Tensor，形状为 (n, 3)，每行代表一个圆的 x, y, r
        color: 圆的颜色，可以是字符串（如 'r' 表示红色）或 RGBA 元组
    """
    for circle in circles:
        x, y, r = circle
        # 创建一个圆形的 patch
        circle_patch = patches.Circle((x, y), r, edgecolor=color, facecolor='none')
        # 添加到当前的 axes 中
        plt_obj.gca().add_patch(circle_patch)


def __draw_clusters(plt_obj, circles, clusters):
    unique_clusters = torch.unique(clusters)
    print("聚类结果：", len(unique_clusters))
    colors = ['red', 'blue', 'green', 'yellow', 'cyan', 'magenta', 'orange']
    for i in range(len(unique_clusters)):
        __draw_circles(plt_obj, circles[clusters == unique_clusters[i]], colors[i])


def show_clusters(circles, clusters):
    Ezplot.initialize()
    __draw_clusters(plt, circles, clusters)
    Ezplot.show_plot()

@utils.count_runtime()
def pairwise_distances(points):
    """计算所有点对的距离 减去两个圆之间的半径之和"""
    diff = points[:, :2].unsqueeze(1) - points[:, :2].unsqueeze(0)
    radius = points[:, 2].unsqueeze(1) + points[:, 2].unsqueeze(0)
    dist_matrix = torch.norm(diff, dim=-1) - radius
    return dist_matrix


@utils.count_runtime()
def adjacency_matrix(points, epsilon):
    """计算邻接矩阵"""
    dist_matrix = pairwise_distances(points)
    # 生成邻接矩阵
    adj_matrix = dist_matrix <= epsilon
    return adj_matrix


def initialize_clusters(points):
    return torch.arange(len(points))


@utils.count_runtime()
def merge_clusters(adj_matrix, clusters):
    """合并簇"""
    new_clusters = clusters.clone()
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            if adj_matrix[i, j] and clusters[i] != clusters[j]:
                new_clusters[clusters == clusters[j]] = clusters[i]
    return new_clusters


@utils.count_runtime()
def custom_clustering(points, epsilon):
    adj_matrix = adjacency_matrix(points, epsilon)
    clusters = initialize_clusters(points)
    prev_clusters = torch.tensor([])

    # 迭代合并簇，直到簇不再变化
    while not torch.equal(clusters, prev_clusters):
        prev_clusters = clusters.clone()
        clusters = merge_clusters(adj_matrix, clusters)

    return clusters


class Ezplot:
    @staticmethod
    def initialize():
        plt.figure()
        plt.gca().set_aspect('equal')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.gca().invert_yaxis()
        plt.gca().xaxis.set_ticks_position('top')

    @staticmethod
    def show_plot():
        plt.show()


if __name__ == '__main__':

    spaces = torch.tensor([
        [0.12, 0.13, 0.05],
        [0.23, 0.13, 0.05],
        [0.12, 0.43, 0.05],
        [0.60, 0.65, 0.05],
        [0.70, 0.76, 0.08],
        [0.16, 0.28, 0.09], # r = 0.09 | 0.06
    ])

    # 执行聚类
    clusters = custom_clustering(spaces, 0.02)

    # RunningTimeTester(
    #     test_functions=[custom_clustering],
    #     test_wrapper=lambda func: func(points, 0.02)
    # ).test()

    show_clusters(spaces, clusters)