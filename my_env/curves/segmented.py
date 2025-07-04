
import torch
import utils

from config_provider import ConfigProvider
from utils.crv_tester import show_curve


@utils.count_runtime(track=ConfigProvider.track_time)
def crv_edge_to_road(x: torch.Tensor) -> torch.Tensor:
    """ 用来测试的 """
    l1 = 0.03
    l2 = 0.08
    l3 = 0.58

    y = torch.where(x < l1, 1, x)
    y = torch.where((l1 <= x) & (x < l2), 0, y)
    y = torch.where((l2 <= x) & (x < l3), 2 * x - 0.16, y)
    y = torch.where(l3 <= x, 1, y)
    return y

@utils.count_runtime(track=ConfigProvider.track_time)
def crv_edge_to_road_plus(x: torch.Tensor) -> torch.Tensor:
    """
    空间边界到道路距离的曲线
    https://www.desmos.com/calculator/cqvebl2r1f
    Args:
        x: 边界到道路距离
    Returns: ...
    """
    # l1 = 0.01
    # l2 = 0.02
    # d = 0.06
    # l3 = l2 + d
    #
    # y = torch.where(x < l1, 1, x)
    # y = torch.where((l1 <= x) & (x < l2), 0, y)
    # y = torch.where((l2 <= x) & (x < l3), (1/d) * (x - l2), y)
    # y = torch.where(l3 <= x, 1, y)
    # return y

    l1 = 0
    d1 = 0.01
    d2 = 0.024
    d3 = 0.1

    l2 = l1 + d1
    l3 = l2 + d2
    l4 = l3 + d3

    y = torch.where(x < l1, 1, x)
    y = torch.where((l1 <= x) & (x < l2), -(1/d1)*(x-l2), y)
    y = torch.where((l2 <= x) & (x < l3), 0, y)
    y = torch.where((l3 <= x) & (x < l4), (1/d3)*(x-l3), y)
    y = torch.where(l4 <= x, 1, y)

    return y


def crv_space_edge_to_square_edge(x: torch.Tensor) -> torch.Tensor:
    """
    实体空间边界到广场
    Args:
        x: 边界到道路距离
    Returns: ...
    """
    l2 = 0.02
    d = 0.06
    l3 = l2 + d

    y = torch.where(x < l2, 0, x)
    y = torch.where((l2 <= x) & (x < l3), (1/d) * (x - l2), y)
    y = torch.where(l3 <= x, 1, y)
    return y

@utils.count_runtime(track=ConfigProvider.track_time)
def crv_overlap(x:torch.Tensor, d1:float, d2:float) -> torch.Tensor:
    """
    重叠部分曲线
    https://www.desmos.com/calculator/1lciprp2rj
    Args:
        x: 空间边界间的距离
        d1: 倾斜部分在x轴上的长度
        d2: 容许空间之间有多少距离的重叠
    Returns:重叠的cost
    """
    if d1 == 0:
        d1 = 1e4
    l1 = 0 - d1 - d2
    l2 = l1 + d1
    k = 1 / (l1 - l2)
    b = 1 - k * l1

    y = torch.where(x < l1, 1, x)
    y = torch.where((l1 <= x) & (x < l2), k * x + b, y)
    y = torch.where(l2 <= x, 0, y)

    return y

@utils.count_runtime(track=ConfigProvider.track_time)
def crv_relationship(x: torch.Tensor, p:torch.Tensor, d1:float, d2:float) -> torch.Tensor:
    """
    空间关系曲线
    https://www.desmos.com/calculator/fnrw9t82eu
    Args:
        x: 当前布置的空间与其它既有空间的距离（边界距离）
        p: 当前的吸引/排斥系数
        d1: 如下图
        d2: 如下图
         |.............@@@@@@@
         |..........@@
         |.......@@
         |@@@@@@
         |--d1--|--d2--|
    Returns: ...
    """
    l1 = d1
    l2 = d1 + d2
    k = 1 / d2
    b = -d1 * k - 0.5

    y = torch.where(x < l1, -0.5, x)
    y = torch.where((l1 <= x) & (x < l2), k * x + b, y)
    y = torch.where(l2 <= x, 0.5, y)

    g = torch.where(0<x, p * y + torch.abs(p / 2), 0)
    return g

@utils.count_runtime(track=ConfigProvider.track_time)
def crv_relationship_mini(x: torch.Tensor, p:torch.Tensor) -> torch.Tensor:
    """ 用来测试的 性能不行 """
    y = torch.where(x<0.2, -1, x)
    y = torch.where((0.2<=x)&(x<0.6), 5*x-2, y)
    y = torch.where(0.6 <= x, 1, y)
    return 0.5*p*y + torch.abs(p/2)


@utils.count_runtime(track=ConfigProvider.track_time)
def crv_boundary(x: torch.Tensor, r:float, m: float) -> torch.Tensor:
    """
    带有margin的 圆到边界的距离
    https://www.desmos.com/calculator/3nrrqqhziu
    Args:
        x:
        r: radius
        m: margin
    Returns:
    """
    l1 = -m - r + 0.02
    l2 = 0
    k = 1 / l1 - l2

    y = torch.where(x < l1, 1, x)
    y = torch.where((l1 <= x) & (x < l2), k*x, y)
    y = torch.where(l2 <= x, 0, y)
    return y


def __test_edge_to_road():
    from utils.running_time_tester import RunningTimeTester

    def wrapper(func):
        x = torch.linspace(0, 1, 10)
        func(x)

    RunningTimeTester(
        test_functions=[crv_edge_to_road, crv_edge_to_road_plus],
        test_wrapper=wrapper,
        times=10000
    ).test()


def __test_relationship():
    from utils.running_time_tester import RunningTimeTester

    def wrapper(func):
        x = torch.tensor([0.15, 0.35, 0.55, 0.75])
        p = torch.tensor([-0.3, 0.2, 0.5, -0.7])
        func(x, p)

    RunningTimeTester(
        test_functions=[crv_relationship, crv_relationship_mini],
        test_wrapper=wrapper,
        times=1000
    ).test()


def __show_crv_overlap():
    def wrapper(x:torch.Tensor) -> torch.Tensor:
        return crv_overlap(x, 0.5, 0.25)

    show_curve(
        wrapper,
        -1, 1, 100
    )


def __show_crv_relationship():
    def wrapper(x: torch.Tensor) -> torch.Tensor:
        return crv_relationship(x, torch.ones_like(x) * 1, 0.2, 0.25)

    show_curve(
        wrapper,
        0, 1, 100
    )


def __show_crv_boundary():
    def wrapper(x:torch.Tensor) -> torch.Tensor:
        return crv_boundary(x, 0.15, 0.05)

    show_curve(
        wrapper,
        -0.5, 0.5, 100
    )


def __show_crv_edge_to_road_plus():
    def wrapper(x:torch.Tensor) -> torch.Tensor:
        return crv_edge_to_road_plus(x)

    show_curve(
        wrapper,
        0, 1, 100
    )


if __name__ == '__main__':

    def wrapper(x):
        return crv_edge_to_road_plus(x)

    show_curve(
        wrapper,
        0, 1, 100
    )

    pass
