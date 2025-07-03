from geometry import *
from grid import Grid


class SimpleEnv:

    def __init__(self):
        self.map = Map((512, 512))

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self):
        pass

    def get_state(self):
        pass


class Map:
    size: tuple[int] = (1, 1)
    geometries: list[Geometry] = []
    grid: Grid

    def __init__(self, size):
        self.size = size
        self.grid = Grid(self.size[0])

    def check_margin(self, *args: tuple[int]):
        for point in args:
            if point[0] > self.size[0] or point[0] < 0 or point[1] > self.size[1] or point[1] < 0:
                return False
        return True

    def add_line(self, start, end):
        if self.check_margin(start, end):
            pass

    def render_repr(self):
        self.grid.show_grid()

    def add(self, geometry: Geometry):
        for line in geometry.draw():
            self.grid.add(*line)

if __name__ == '__main__':
    map_test = Map((20, 20))
    map_test.render_repr()
    map_test.add(
        Quadrilateral(
            Point(0.1, 0.1),
            Point(0.8, 0.2),
            Point(0.8, 0.8),
            Point(0.1, 0.6)
        )
    )
    map_test.render_repr()
