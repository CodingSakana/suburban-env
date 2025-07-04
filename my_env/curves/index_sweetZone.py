import utils
import my_env.curves as crv
from config_provider import ConfigProvider


@utils.count_runtime(track=ConfigProvider.track_time, threshold=1e4)
def index_sweetZone(floor:float, ceil:float, l_base=2, r_base=10):
    """
    指数函数 有一定的甜区
    :param floor: 甜区地板
    :param ceil: 甜区天花板
    :param l_base: 左侧指数函数的底数
    :param r_base: 右侧指数函数的底数
    :return: 一个闭包
    """
    # todo 指数改成其它的方式，改成 torch.pow torch.exp
    def crv(x:float):
        if x < floor:
            return l_base**(-x+floor) -1
        elif x > ceil:
            return r_base**(x-ceil) - 1
        return 0
    return crv


if __name__ == '__main__':
    crv.show_curve(
        index_sweetZone(-0.02, 0.02, 100, 100),
        -0.1, 0.1
    )
