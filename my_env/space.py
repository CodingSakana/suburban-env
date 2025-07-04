
import numpy as np
from abc import ABC, abstractmethod

from typing import Tuple

import torch
from config_provider import ConfigProvider


class Space(ABC):
    x: float
    y: float
    radius: float
    color: Tuple[int, int, int]

    max_r = 0.5
    min_r = 0
    color = (255, 255, 255)
    def __init__(self, x, y, radius):
        self.x = x
        self.y = y
        self.radius = self.linear_radius(radius)
        self.area = np.pi * (self.radius ** 2)
        self.action:torch.Tensor = torch.tensor([x, y, self.radius], device=ConfigProvider.device)
        self.space:torch.Tensor = torch.cat((
            self.get_space_type_index(), self.action
        ))

    @classmethod
    def linear_radius(cls, radius):
        return cls.min_r + (radius / 1) * (cls.max_r - cls.min_r)

    @classmethod
    @abstractmethod
    def get_space_type_index(cls) -> torch.Tensor:
        pass


class Square(Space):
    max_r = 0.1
    min_r = 0.06
    color = (16, 242, 255)

    @classmethod
    def get_space_type_index(cls):
        return torch.tensor([0], device=ConfigProvider.device)


class Restaurant(Space):
    max_r = 0.05
    min_r = 0.035
    color = (36, 27, 238)

    @classmethod
    def get_space_type_index(cls):
        return torch.tensor([1], device=ConfigProvider.device)


class Store(Space):
    max_r = 0.043
    min_r = 0.031
    color = (74, 185, 80)

    @classmethod
    def get_space_type_index(cls):
        return torch.tensor([2], device=ConfigProvider.device)


class Restroom(Space):
    max_r = 0.042
    min_r = 0.03
    color = (255, 185, 80)

    @classmethod
    def get_space_type_index(cls):
        return torch.tensor([3], device=ConfigProvider.device)


class Hotel(Space):
    max_r = 0.055
    min_r = 0.04
    color = (175, 99, 62)

    @classmethod
    def get_space_type_index(cls):
        return torch.tensor([4], device=ConfigProvider.device)


