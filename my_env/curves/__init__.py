import atexit

import numpy as np

from typing import Callable, Dict, List, Tuple
from matplotlib import pyplot as plt

from my_env.curves.quadratic_sweetZone import quadratic_sweetZone
from my_env.curves.index_sweetZone import index_sweetZone
from my_env.curves.relu import reversed_relu, reversed_relu_tensor
from my_env.curves.segmented import *


def toggle(): # turn False in production env
    from config_provider import ConfigProvider
    return ConfigProvider.use_curve_debug


curve_debugs:Dict[str, List[Tuple[float, float]]] = {}

# def on_exit():
#     if curve_debugs:
#         print(f"Curve Debug Toggle=True\n\t  at {toggle.__code__.co_filename}:{toggle.__code__.co_firstlineno}")
#         for k, v in curve_debugs.items():
#             data = np.array(v).T
#             utils.debug.draw_crv_boxplot(data[0], data[1], title=k)
#     else:
#         print(f"Curve Debug Disabled, to enable please toggle\n\t  at {toggle.__code__.co_filename}:{toggle.__code__.co_firstlineno}")
# atexit.register(on_exit)

@utils.count_runtime(track=ConfigProvider.track_time)
def crvDebug(title:str, crv: Callable[[float], float], data, debug=True):
    if not toggle():
        debug = False

    crv_result = crv(data)

    if debug:
        title += ":" + crv.__qualname__[:crv.__qualname__.find(".")]
        if title not in curve_debugs:
            curve_debugs[title] = []
        curve_debugs[title].append((data, crv_result))

    return crv_result


def show_curve(crv:Callable[[float], float], floor:float, ceil:float, num=50):
    x = np.linspace(floor, ceil, num)
    y = np.vectorize(crv)(x)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(x, y, color="black", linewidth=0.8)
    plt.show()
