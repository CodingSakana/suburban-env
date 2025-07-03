from __future__ import annotations


import cv2
import matplotlib.pyplot as plt
import random
import torch
from typing import Type, Any, List, Dict, Tuple

# 自己维护的依赖
import utils
from my_env import road_generator
from my_env.space import *
from my_env import constraints, rewards
from my_env.device_provider import DeviceProvider


class LayoutEnv:
    """
    这是核心的布局模型类
    """

    # 道路颜色
    road_color: Tuple[int, int, int] = (0, 0, 0)

    @utils.count_runtime
    def __init__(
        self,
        size: int = 128,
        actual_size=100,
    ):

        self.image_size = size # 地图的宽高 默认为正方形
        self.actual_size = actual_size # image_size所代表的现实中的长度 单位米

        # 设置路的粗细
        self.road_thickness: int = size // 64
        if self.road_thickness == 0:
            self.road_thickness = 1

        # 生成道路
        self.road = road_generator.generate()
        self.road_slices = road_generator.road_to_roadSlice(self.road)

        # 绘制道路到一个初始化图像
        self.initial_img: np.ndarray = np.ones((self.image_size, self.image_size, 3), dtype=np.uint8) * 255
        self.draw_roads_to_initial_img(self.road)

        # 设置空间顺序
        types = [Square, Restaurant, Store, Restroom, Hotel]
        self.space_numbers = [3, 6, 15, 2, 4]
        self.step_sum = sum(self.space_numbers)
        self.max_step = self.step_sum - 1

        self.space_types: List[type] = []

        for index, num in enumerate(self.space_numbers):
            self.space_types += [
                types[index] for _ in range(num)
            ]

        # 初始化: 图像、模型中的空间列表、目前布置的步数
        self.spaces: torch.Tensor = torch.tensor([])  # 布置了的空间将会存储在这里
        self.image: np.ndarray = np.array([])  # 栅格化成(size,size)大小的地图图像
        self.step_index: int = 0  # 当前智能体决策时间步

        self.reset()  # 重置 已布置空间、观测图像、时间步


    def draw_roads_to_initial_img(self, roads):
        """
        将归一化坐标转换为像素坐标 并按照cv2的要求reshape
        这段代码只会在实例化的时候调用 性能影响不大
        """
        def rebuild(points):
            for point in points:
                for i in range(2):
                    point[i] = int(point[i] * self.image_size)

            return points.reshape((-1, 1, 2)).astype(np.int32)

        polylines = []
        for road in roads:
            polylines.append(
                rebuild(
                    np.array(road, np.float32)
                )
            )

        cv2.polylines(self.initial_img, polylines, isClosed=False, color=self.road_color, thickness=self.road_thickness)

    def get_obs(self):
        """返回当前的 observation"""
        image_flatten = torch.tensor(self.image, device=DeviceProvider.device).flatten()
        image_normalized = image_flatten / 255.0 # 0-255 映射到 0-1
        obs =  torch.cat((
            image_normalized, torch.tensor([self.step_index], device=DeviceProvider.device),
        )).to(torch.float32)

        return obs

    def reset(self):
        """重置图像为白色背景"""
        self.image = self.initial_img.copy()
        self.spaces = torch.tensor([[0, 0, 0, 0] for _ in range(30)], device=DeviceProvider.device)
        self.step_index = 0
        return self.get_obs()

    def show_plot(self):
        """使用plt渲染当前的图像"""
        image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        # plt.imshow(image_rgb)
        plt.imshow(image_rgb, extent=[0, 1, 1, 0])
        plt.gca().xaxis.set_ticks_position('top')
        plt.show()

    @utils.count_runtime()
    def lay_space(self, space_type, x, y, radius):
        """存入 self.spaces 并绘制在image上"""
        # 构造空间 存入
        new_space = space_type(x, y, radius)
        self.spaces[self.step_index] = torch.tensor(
            [space_type.get_space_type_index(), x, y, radius]
        )

        # 将归一化的尺寸转换为图像像素尺寸
        new_x = int(x * self.image_size)
        new_y = int(y * self.image_size)
        new_radius = int(new_space.radius * self.image_size)

        cv2.circle(self.image, (new_x, new_y), new_radius, new_space.color, -1)


    def nl2il(self, action:torch.tensor) -> torch.tensor:
        """
        normalized length to image length
        归一化的尺寸 映射回 图像像素尺寸
        """
        return (action * self.image_size).to(torch.int32).tolist()

    def step(
        self,
        action: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Dict[str, Any],
    ]:
        """
        Public API for CMDP
        :param action: [x, y, radius]
        :return:
            observation: [tensor(size, size, 3)]
            reward: The amount of reward returned after previous action.
            cost: The amount of cost returned after previous action.
            terminated: Whether the episode has ended.
            truncated: Whether the episode has been truncated due to a time limit.
            info: Some information logged by the environment.
        """

        assert action.shape == (3,), "Action shape is not (3)."

        truncated = torch.tensor([0], device=DeviceProvider.device)
        info = {'time_step': self.step_index}

        type_to_lay = self.lay_type(self.step_index)
        assert type_to_lay is not None, "The env is already terminated."

        px = action[0]
        py = action[1]
        r = action[2]

        self.lay_space(type_to_lay, px, py, r)

        cost = constraints.weighting(self, action)
        reward = rewards.weighting(self, action)

        # get observation
        obs = self.get_obs()

        # determine if terminated
        if self.step_index == self.step_sum - 1:
            terminated = torch.tensor(True, device=DeviceProvider.device)
            info['rest'] = True
        else:
            self.step_index += 1
            terminated = torch.tensor(False, device=DeviceProvider.device)


        return obs, reward, cost, terminated, truncated, info


    # @staticmethod
    def lay_type(self, step_index:int) -> (Type, torch.tensor):
        """
        获取当前步骤要摆放啥类型的空间
        :param step_index: 步骤的索引 第几步
        :return: 这一步所要布置的空间； 结束标志 terminated
        """
        assert step_index < self.step_sum, f"step_index={step_index} 超出了最大步数"
        return self.space_types[step_index]



    @staticmethod
    def get_random_action(limitation=0.2):
        """
        获取一个随机的action 用于测试
        limitation: 随机决策中 radius大小的限制
        """
        return torch.tensor([random.random() for _ in range(2)] + [random.random()*limitation])


    def draw_reference_action(self, action, reference_color = (134, 255, 102), fix_radius=None):
        """画一个参考点 纯测试使用 训练的时候千万别用"""
        action = self.nl2il(action)
        x, y, r = action
        cv2.circle(self.image, (x, y), 3, reference_color, -1)
        cv2.circle(self.image, (x, y), r, reference_color, 1) # rgb(102, 255, 134)



if __name__ == '__main__':

    layoutEnv = LayoutEnv(size=256)
    debug = True

    layoutEnv.step = utils.count_runtime(
        layoutEnv.step, track=True
    )

    for _ in range(3):

        for index, i in enumerate(range(30)):
            action = torch.tensor([random.random() for _ in range(3)])

            result = layoutEnv.step(action)

            if debug:
                print(f"step {layoutEnv.step_index} action: {action}")
                layoutEnv.show_plot()
                print(f"reward:{result[1]}, cost:{result[2]}")
                input()

        layoutEnv.show_plot()
        layoutEnv.reset()
