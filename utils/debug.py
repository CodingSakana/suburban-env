from typing import List, Callable

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


matplotlib.rcParams['font.sans-serif'] = ['SimSun']  # 'SimSun' 是宋体
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号


def statistics(crv:Callable, *args):
    pass


def draw_hist_boxplot(data):
    # 创建一个图形和两个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6))  # 设置为正方形

    # 在第一个子图中绘制直方图
    counts, bins, patches = ax1.hist(data, bins=20, color='blue', alpha=0.7)
    ax1.set_title('Histogram')

    # 设置直方图的y轴刻度为实际值
    max_count = max(counts)
    # ax1.set_yticks(np.arange(0, max_count + 1, max_count // 10))  # 以最大计数的十分之一为步长

    # 在第二个子图中绘制箱型图
    boxplot = ax2.boxplot(data, vert=True, patch_artist=True, showfliers=False)  # vert=True使箱型图竖直显示
    ax2.set_title('Boxplot')

    # 设置箱型图的y轴刻度为实际值
    q1, median, q3 = np.percentile(data, [25, 50, 75])
    ax2.set_ylim(min(data), max(data))  # 设置y轴的范围为数据的最小值和最大值
    ax2.set_yticks([q1, median, q3, min(data), max(data)])  # 设置y轴刻度为实际的分位数值

    # 调整子图间距和箱型图所占的宽度
    # plt.subplots_adjust(wspace=0.1)  # 调整子图之间的宽度间距
    ax1.set_position([0.1, 0.1, 0.6, 0.8])  # 调整箱型图的位置和大小
    ax2.set_position([0.8, 0.1, 0.15, 0.8])  # 调整箱型图的位置和大小

    # 显示图表
    plt.show()


def draw_crv_boxplot(data_x, data_y, title=""):
    # # 创建一个图形和两个子图
    # fig, axs = plt.subplots(2, 2, figsize=(8, 6))  # 设置为正方形
    #
    # # 在第一个子图中绘制直方图
    # axs[0, 0].scatter(data_x, data_y)
    # axs[0, 0].set_title('Scatter Plot')
    #
    # # 在右上图（axs[0, 1]）绘制关于 y 的垂直箱型图
    # axs[0, 1].boxplot(data_y, vert=True)
    #
    # # 在左下图（axs[1, 0]）绘制关于 x 的水平箱型图
    # axs[1, 0].boxplot(data_x, vert=False)
    #
    # # 显示图表
    # plt.show()

    # 创建一个图形对象
    fig = plt.figure(figsize=(8, 7))

    if title:
        fig.suptitle(title, fontsize=20, fontweight='bold')

    # 创建一个 gridspec 布局对象，定义 2 行 2 列
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 2], width_ratios=[2, 1])

    # 在 gridspec 布局中定义子图的位置
    ax1 = fig.add_subplot(gs[0, 0])  # 第一行，第一列
    ax2 = fig.add_subplot(gs[0, 1])  # 第一行，第二列
    ax3 = fig.add_subplot(gs[1, 0])  # 第二行，第一列

    # 在第一行的两个子图上绘制关于 y 的垂直箱型图
    ax1.scatter(data_x, data_y)

    ax2.boxplot(data_y, vert=True)

    # 在第一列的两个子图上绘制关于 x 的水平箱型图
    ax3.boxplot(data_x, vert=False)


    # # 调整子图间距
    # plt.tight_layout()

    left_padding = 0.1
    bottom_padding = 0.08
    image_width = 0.75
    boxplot_width = 0.1
    ax1.set_position([left_padding, bottom_padding+boxplot_width, image_width, image_width]) # left, bottom, width, height
    ax2.set_position([left_padding+image_width, bottom_padding+boxplot_width, boxplot_width, image_width])  # left, bottom, width, height
    ax3.set_position([left_padding, bottom_padding, image_width,boxplot_width])  # left, bottom, width, height


    # 显示图表
    plt.show()


def map_value_to_color(data, colors:List[str]=None, n_bins = 100):

    # 创建一个颜色映射对象
    norm = plt.Normalize(vmin=data.min(), vmax=data.max())  # 归一化数据

    if not colors:
        cmap = plt.get_cmap('viridis')  # 选择一个颜色映射表
    else:
        from matplotlib.colors import LinearSegmentedColormap
        cmap_name = 'custom'
        cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    # 将数据映射到颜色
    colors = cmap(norm(data))

    return colors


if __name__ == '__main__':
    # 假设这是你的100个数据点

    import my_env.curves as crv
    data_x = np.random.normal(0, 1, 100)
    data_y = np.vectorize(crv.index_sweetZone)(data_x, 0.03, 0.06, 20, 10)

    draw_crv_boxplot(data_x, data_y)
