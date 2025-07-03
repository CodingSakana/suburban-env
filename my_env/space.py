
import numpy as np
from abc import ABC, abstractmethod

from typing import Tuple


class Space(ABC):
    x: float
    y: float
    radius: float
    color: Tuple[int, int, int]

    max_r = 0.5
    min_r = 0
    def __init__(self, x, y, radius, color):
        self.x = x
        self.y = y
        self.radius = self.linear_radius(radius)
        self.color = color
        self.area = np.pi * (self.radius ** 2)

    def linear_radius(self, radius):
        return self.min_r + (radius / 1) * (self.max_r - self.min_r)

    @abstractmethod
    def get_space_type_index(self) -> int:
        pass


class Square(Space):
    max_r = 0.1
    min_r = 0.06

    def __init__(self, x, y, radius):
        super().__init__(
            x, y, radius, (16, 242, 255)
        )

    @classmethod
    def get_space_type_index(self):
        return 0


class Restaurant(Space):
    max_r = 0.05
    min_r = 0.035
    def __init__(self, x, y, radius):
        super().__init__(
            x, y, radius, (36, 27, 238)
        )

    @classmethod
    def get_space_type_index(self):
        return 1


class Store(Space):
    max_r = 0.035
    min_r = 0.025
    def __init__(self, x, y, radius):
        super().__init__(
            x, y, radius, (74, 185, 80)
        )

    @classmethod
    def get_space_type_index(self):
        return 2


class Hotel(Space):
    max_r = 0.055
    min_r = 0.04
    def __init__(self, x, y, radius):
        super().__init__(
            x, y, radius, (175, 99, 62)
        )

    @classmethod
    def get_space_type_index(self):
        return 3


class Restroom(Space):
    max_r = 0.035
    min_r = 0.025
    def __init__(self, x, y, radius):
        super().__init__(
            x, y, radius, (255, 185, 80)
        )

    @classmethod
    def get_space_type_index(self):
        return 4