from abc import ABC, abstractmethod


class Geometry(ABC):
    @abstractmethod
    def draw(self) -> list[tuple]:
        pass

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Line(Geometry):
    def __init__(self, start, end):
        self.start = start
        self.end = end


class MultiLine(Geometry):
    def __init__(self, *args: Point):
        if len(args) < 3:
            raise ValueError("Not enough points")


class Quadrilateral(Geometry):
    def __init__(self, *args: Point):
        self.points = args
        if len(args) < 4:
            raise ValueError("Not enough points")
        elif len(args) > 4:
            raise ValueError("Points length > 5")

    def draw(self):
        rt = []
        for i in range(4):
            p1 = self.points[i]
            p2 = self.points[(i+1) % 4]
            rt.append(
                (p1.x, p1.y, p2.x, p2.y)
            )
        print(rt)
        return rt