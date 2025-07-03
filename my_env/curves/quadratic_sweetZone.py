import utils


# @utils.show_curve(-1, 1.2, (0.03, 0.03, 10))
@utils.count_runtime
def quadratic_sweetZone(floor:float, ceil:float, a:float=1):
    def crv(x:float):
        if x < floor:
            return a*(x-floor)**2
        elif x > ceil:
            return a*(x-ceil)**2
        return 0
    return crv


if __name__ == '__main__':
    ...
