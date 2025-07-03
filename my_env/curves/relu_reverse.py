import utils
import my_env.curves as crv


@utils.count_runtime(threshold=1e4)
def relu_reverse(base:float=2, switch:float=0):
    """
    反过来的类似relu的曲线
    :param base:
    :param switch:
    :return:
    """
    def crv(x:float):
        x *= 5
        if x < switch:
            return base**(-x+switch) - 1
        return 0
    return crv


if __name__ == '__main__':
    crv.show_curve(
        relu_reverse(3, 0),
        -1, 1
    )
