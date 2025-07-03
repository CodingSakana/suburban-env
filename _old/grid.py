
import numpy as np

class Grid:
    def __init__(self, nm):
        self.nm = nm
        self.grid = np.zeros((nm,nm))

    def add(self, x1, y1, x2, y2):
        gridy = lambda x: x // (1 / self.nm)
        getN = lambda p1, p2: int(abs(gridy(p1) - gridy(p2)) + 1)

        linear = lambda x, x1, x2, y1, y2: y1 + (x - x1) / (x2 - x1) * (y2 - y1)

        Nx = getN(x1, x2)
        Ny = getN(y1, y2)
        if Nx >= Ny:
            if x1 > x2:
                x1, x2 = x2, x1
                y1, y2 = y2, y1
            for n in range(Nx):
                i = int(gridy(x1) + n - 1)
                j = int(gridy(linear(x1 + 1/self.nm * (n-1), x1, x2, y1, y2)))
                self.grid[j, i] = 1
        else:
            if y1 > y2:
                x1, x2 = x2, x1
                y1, y2 = y2, y1
            for n in range(Ny):
                i = int(gridy(y1) + n - 1)
                j = int(gridy(linear(y1 + 1 / self.nm * (n - 1), y1, y2, x1, x2)))
                self.grid[i, j] = 1

    def show(self):
        for line in self.grid:
            print(line)

    def show_grid(self):
        for line in self.grid:
            print("[" + "  ".join(["." if i==0 else "#" for i in line]) + "]")


if __name__ == '__main__':

    m = Grid(20)
    m.show()
    m.add(0.17, 0.17, 0.83, 0.51)
    # m.add(0.17, 0.51, 0.83, 0.17)
    m.add(0.17, 0.17, 0.51, 0.83)
    m.add(0.57, 0.17, 0.23, 0.6 )
    m.add(0.8, 0.8, 0.1, 0.6)
    m.show_grid()
