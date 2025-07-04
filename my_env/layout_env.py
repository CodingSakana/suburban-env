from __future__ import annotations


import cv2
import time
import matplotlib.pyplot as plt
import random
from typing import Type, Any, List, Dict

# 自己维护的依赖
import utils
from my_env import map_manager
from my_env.my_functions.circle_to_circle_edge_distance import circle_to_circle_edge_distance
from my_env.space import *
from my_env import constraints, rewards
from config_provider import ConfigProvider, dprint


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

        # 生成道路和保护区
        self.road = map_manager.generate()
        temp_slices = map_manager.road_to_roadSlice(self.road)
        self.road_slices = torch.tensor(temp_slices, device=ConfigProvider.device).transpose(0, 1)
        self.road_param = map_manager.build_road_parameters(temp_slices)
        self.road_polylines = []

        # 绘制道路到一个初始化图像
        self.initial_img: np.ndarray = np.ones((self.image_size, self.image_size, 3), dtype=np.uint8) * 255
        self.draw_roads_to_img(self.road, self.initial_img)

        # 设置空间顺序 todo 可以加上这个特征
        types = [Square, Restaurant, Store, Restroom, Hotel]
        self.space_numbers = [3, 6, 15, 2, 4]
        self.step_sum = sum(self.space_numbers)
        self.max_step = self.step_sum - 1

        self.space_types: List[Space] = []

        for index, num in enumerate(self.space_numbers):
            self.space_types += [
                types[index] for _ in range(num)
            ]

        # 初始化: 图像、模型中的空间列表、目前布置的步数
        self.spaces: torch.Tensor = torch.tensor([])  # 布置了的空间将会存储在这里
        self.image: np.ndarray = np.array([])  # 栅格化成(size,size)大小的地图图像
        self.step_index: int = 0  # 当前智能体决策时间步
        self.action_history: List[torch.Tensor] = [] # 动作历史

        # step中会用到的public api
        self.current_to_others_distances: torch.Tensor = torch.tensor([])
        self.current_to_road_min_distance: torch.Tensor = torch.tensor([])
        self.space_param_min = torch.tensor([]) # (step_sum, 4), 每个代表(road_slice_index, t投影量, d圆心距离, ed边界距离)
        self.space_param_all = torch.tensor([]) # (step_sum, len(road_slice), 4), 表示空间对所有slice的param

        self.reset()  # 重置 已布置空间、观测图像、时间步


    def draw_roads_to_img(self, roads, image: np.ndarray, road_thickness=None):
        """
        将归一化坐标转换为像素坐标 并按照cv2的要求reshape
        这段代码只会在实例化的时候调用 性能影响不大
        """

        road_thickness = road_thickness or self.road_thickness

        def rebuild(points):
            for point in points:
                for i in range(2):
                    point[i] = int(point[i] * image.shape[0])

            return points.reshape((-1, 1, 2)).astype(np.int32)

        self.polylines = []
        for road in roads:
            self.polylines.append(
                rebuild(
                    np.array(road, np.float32)
                )
            )

        cv2.polylines(image, self.polylines, isClosed=False, color=(220, 220, 220), thickness=3)
        cv2.polylines(image, self.polylines, isClosed=False, color=(160, 160, 160), thickness=2)
        cv2.polylines(image, self.polylines, isClosed=False, color=self.road_color, thickness=road_thickness)


    @utils.count_runtime(track=ConfigProvider.track_time)
    def get_obs(self):
        """返回当前的 observation"""

        # 强制绘制道路中心！
        cv2.polylines(self.image, self.polylines, isClosed=False, color=self.road_color, thickness=self.road_thickness)

        image_flatten = torch.tensor(self.image, device=ConfigProvider.device).flatten()
        image_normalized = image_flatten / 255.0 # 0-255 映射到 0-1
        obs =  torch.cat((
            image_normalized, torch.tensor([self.step_index], device=ConfigProvider.device),
        )).to(torch.float32)

        return obs

    def reset(self):
        """重置图像为白色背景"""
        self.image = self.initial_img.copy()
        # self.spaces = torch.tensor([[0, 0, 0, 0] for _ in range(30)], device=ConfigProvider.device)
        self.spaces = torch.zeros([self.step_sum, 4], device=ConfigProvider.device)
        self.space_param_min = torch.zeros([self.step_sum, 4], device=ConfigProvider.device)
        self.space_param_all = torch.zeros([self.step_sum, self.road_slices.size(1), 4], device=ConfigProvider.device)
        self.step_index = 0
        self.action_history = []
        return self.get_obs()

    @utils.count_runtime(track=ConfigProvider.track_time)
    def show_plot(self, image: np.ndarray=None):
        """使用plt渲染当前的图像"""
        if image is None: image = self.image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # plt.imshow(image_rgb)
        plt.imshow(image_rgb, extent=(0, 1, 1, 0))
        plt.gca().xaxis.set_ticks_position('top')
        plt.show()

    @utils.count_runtime(track=ConfigProvider.track_time, threshold=2e5)
    def lay_space(self, space_type, x, y, radius) -> torch.Tensor:
        """存入 self.spaces 并绘制在image上"""
        # 构造空间 存入
        new_space = space_type(x, y, radius)
        self.spaces[self.step_index] = new_space.space

        # 将归一化的尺寸转换为图像像素尺寸
        new_x = int(x * self.image_size)
        new_y = int(y * self.image_size)
        new_radius = int(new_space.radius * self.image_size)

        cv2.circle(self.image, (new_x, new_y), new_radius, new_space.color, -1)

        return new_space.action


    def get_high_resolution_image(self, size, road_thickness=None) -> np.ndarray:
        """用于debug、evaluate的高清晰度图片"""
        image: np.ndarray = np.ones((size, size, 3), dtype=np.uint8) * 255
        self.draw_roads_to_img(self.road, image, road_thickness)
        valid_spaces: torch.Tensor = self.spaces[:self.step_index, :]

        for virtual_step, space in enumerate(valid_spaces):
            x = space[1]
            y = space[2]
            radius = space[3]

            new_x = int(x * size)
            new_y = int(y * size)
            new_radius = int(radius * size)

            type_to_lay = self.lay_type(virtual_step)

            cv2.circle(image, (new_x, new_y), new_radius, type_to_lay.color, -1)

        return image



    def nl2il(self, action:torch.tensor) -> torch.tensor:
        """
        normalized length to image length
        归一化的尺寸 映射回 图像像素尺寸
        """
        return (action * self.image_size).to(torch.int32).tolist()

    @utils.count_runtime(track=ConfigProvider.track_time)
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

        dprint("layout received: ", action)
        self.action_history.append(action)

        assert action.shape == (3,), "Action shape is not (3)."
        assert self.step_index < self.step_sum, "invalid time_step"

        truncated = torch.tensor([0], device=ConfigProvider.device)
        info = {'time_step': self.step_index}

        type_to_lay = self.lay_type(self.step_index)
        assert type_to_lay is not None, "The env is already terminated."

        px = action[0]
        py = action[1]
        r = action[2]

        remapped_action = self.lay_space(type_to_lay, px, py, r)
        dprint("my remapped action: ", remapped_action)
        dprint()


        # 提前计算一个 distance 然后共享给 reward 和 cost
        index_min, t_min, d_min, index, t, d = map_manager.build_point_param(remapped_action, self.road_slices)
        self.current_to_road_min_distance = (d_min - remapped_action[2])[0]
        self.space_param_min[self.step_index] = torch.tensor([index_min, t_min, d_min, self.current_to_road_min_distance], device=ConfigProvider.device)
        all_param = torch.stack([
            index, t, d, d - remapped_action[2]
        ], dim=1)
        self.space_param_all[self.step_index] = all_param
        self.current_to_others_distances = circle_to_circle_edge_distance(
            remapped_action,
            self.spaces[:self.step_index, 1:]
        )


        cost = constraints.cost_weighting(self, remapped_action)
        reward = rewards.reward_weighting(self, remapped_action)

        # get observation
        obs = self.get_obs()

        # determine if terminated 并且更改step_index
        if self.step_index == self.step_sum - 1:
            terminated = torch.tensor(True, device=ConfigProvider.device)
            self.step_index = self.step_sum
            info['reset'] = True
        else:
            self.step_index += 1
            terminated = torch.tensor(False, device=ConfigProvider.device)


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



def round_test(
    env: LayoutEnv,
    interrupt: bool = False,
    show_plot: bool = False,
    show_plot_after_round: bool = False,
):
    """
    一个只用于debug的测试函数
    Args:
        env: 传进布局环境实例
        interrupt: 每一个step是否打断？
        show_plot: 每一个step是否展示布局？
        show_plot_after_round: 每一回合完成后展示布局？
    Returns: null 啥也不返回
    """
    start = time.perf_counter_ns()
    # for _ in range(steps):
    while True:
        action = torch.tensor([random.random() for _ in range(3)], device=ConfigProvider.device)

        space_type = env.lay_type(env.step_index)

        # sample_action = action.clone()
        # sample_action[2] =  space_type.linear_radius(sample_action[2])
        # print(space_type)
        # layoutEnv.draw_reference_action(sample_action)
        # layoutEnv.show_plot()

        dprint(f"\nstep {env.step_index}:{space_type.__name__}, action: {action}")

        result = env.step(action)

        if show_plot:
            env.show_plot()
            env.show_plot(image=env.get_high_resolution_image(256))

        # dprint(f"R: {result[1]:.2f}, C: {result[2]:.2f}")

        if result[3].item():
            break

        if interrupt: input("continue > ")

    if show_plot_after_round: env.show_plot()
    env.reset()

    print(f"Round over in {time.perf_counter_ns() - start:,} ns.")

if __name__ == '__main__':

    layoutEnv = LayoutEnv(size=32)

    for _ in range(6):
        round_test(
            layoutEnv,
            interrupt=False,
            show_plot=False,
            show_plot_after_round=False,
        )




