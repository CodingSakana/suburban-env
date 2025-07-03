import cv2
import numpy as np
import time
import matplotlib.pyplot as plt


class CircleInImage:

    road_thickness:int = 1
    image: np.ndarray

    def __init__(self, image_size=128):
        self.image_size = image_size  # 图像大小，默认 128x128
        # self.image = np.zeros((image_size, image_size, 3), dtype=np.uint8)  # 创建一个黑色背景的图像
        self.reset()


    def draw_circle(self, x, y, radius=20):
        """根据归一化的坐标绘制一个实心圆"""
        # 将归一化的坐标 (x, y) 转换为图像坐标
        center_x = int(x * self.image_size)
        center_y = int(y * self.image_size)

        # 在指定位置绘制一个实心圆，颜色为蓝色 (255, 0, 0)
        cv2.circle(self.image, (center_x, center_y), radius, (255, 0, 0), -1)

    def get_image(self):
        """返回当前图像，作为智能体的状态"""
        return self.image

    def show(self, scale_factor=2):
        enlarged_image = cv2.resize(self.image, (self.image_size * scale_factor, self.image_size * scale_factor))
        cv2.imshow("Circle Image", enlarged_image)
        cv2.waitKey(1)  # 1毫秒后刷新

    def draw_roads(self):

        # 将归一化坐标转换为像素坐标 并按照cv2的要求reshape
        def rebuild(points):
            for point in points:
                for i in range(2):
                    point[i] = int(point[i] * self.image_size)

            return points.reshape((-1, 1, 2)).astype(np.int32)

        points1 = np.array([[0, 0.6], [0.2, 0.5], [0.6, 0.6], [1, 0.9]], np.float32)
        points2 = np.array([[0.56, 0], [0.5, 0.3], [0.6, 0.6]], np.float32)

        points1 = rebuild(points1)
        points2 = rebuild(points2)

        print(points1)
        print(points2)

        cv2.polylines(self.image, [points1, points2], isClosed=False, color=(0, 255, 0), thickness=self.road_thickness)

    def reset(self):
        """重置图像为黑色背景"""
        self.image = np.ones((self.image_size, self.image_size, 3), dtype=np.uint8) * 255

    def create_plot(self):
        image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        # plt.imshow(image_rgb)
        plt.imshow(image_rgb, extent=[0, 1, 1, 0])
        plt.gca().xaxis.set_ticks_position('top')
        plt.show()


if __name__ == '__main__':

    # 使用示例
    env = CircleInImage(image_size=256)  # 创建一个 128x128 的环境

    env.draw_roads()

    # 模拟智能体每次输出一个归一化的坐标
    actions = [(0.2, 0.3), (0.5, 0.5), (0.7, 0.8), (0.3, 0.7), (0.9, 0.1)]  # 示例智能体的动作序列

    # 每次根据智能体的动作绘制一个圆，并返回状态
    for action in actions:
        x, y = action  # 获取智能体输出的归一化坐标
        env.draw_circle(x, y)  # 在图像上绘制圆
        state = env.get_image()  # 获取当前图像状态
        env.create_plot()
        time.sleep(1)
        # input_str = input(" > ")
