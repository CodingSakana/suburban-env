#
# import utils
#
# @utils.count_runtime
# def test():
#     pass
#
# if __name__ == '__main__':
#     test()
#

color = "black"

import numpy as np
import matplotlib.pyplot as plt

a = np.linspace(0, 5, 100)

y1 = np.sin(2 * np.pi * a)
y2 = np.cos(2 * np.pi * a)

fig, ax1 = plt.subplots()

ax1.set_xlabel("time (s)")
ax1.set_ylabel("sin", color="red")
ax1.plot(a, y1, color=color)
ax1.tick_params(axis="y", labelcolor=color)

ax2 = ax1.twinx()
# ax2.set_ylabel("cos", color="green", fontsize=10)
ax2.set_ylabel('Right Y-Axis Label', labelpad=40)
# ax2.plot(a, y2, color=color)
ax2.tick_params(axis="y", labelcolor=color)

fig.tight_layout()
plt.show()