import torch
from torch import nn

from omnisafe.typing import Activation, CriticType, InitFunction, OmnisafeSpace
from gymnasium.spaces import Box #, Discrete
from typing import Tuple
from utils.time_embedding import embedding_time_with_image

from utils.time_embedding import TimeEmbedding, ResBlock


class Extractor:

    # static field
    img_size = None
    omnisafe_obs_shape = None
    network = None

    @classmethod
    def set_img_size(cls, size: int):
        cls.img_size = size
        cls.omnisafe_obs_shape = size ** 2 * 3

        # 这个值暴露给了 actor 和 critic 网络
        cls.obs_dim = 128 # 128 * cls.img_size ** 2

    @classmethod
    def get_omnisafe_obs_shape(cls) -> int:
        return cls.omnisafe_obs_shape

    @classmethod
    def factory_state_preprocessor(cls, obs: Box) -> nn.Module:

        if cls.network is None:
            img_length = cls.get_omnisafe_obs_shape()

            assert cls.img_size is not None, 'img_size怕是没设置吧'
            assert obs.shape == (img_length + 1,), 'obs的形状不太对吧'

            cls.network = nn.Sequential(

                ObservationPreprocessor(), # => (1, 32, 128, 128)

                nn.Conv2d(
                    in_channels=32, out_channels=128, kernel_size=3, padding=1
                ), # => (1, 128, .., ..)

                nn.AdaptiveAvgPool2d((1, 1)), # => (1, 128, 1, 1)

                nn.Flatten(), # => (128,)

            ) # => mlp(128, 3)

        return cls.network

    @classmethod
    def factory_state_action_connector(cls, obs: Box, act: Box):
        # todo 如果是q-critic网络的话
        raise NotImplementedError


class ObservationPreprocessor(torch.nn.Module):
    def __init__(self):

        super().__init__()

        self.img_size = Extractor.img_size
        self.shape = Extractor.omnisafe_obs_shape

        self.time_embedding = TimeEmbedding(30, self.img_size, 512)
        self.conv2d = nn.Conv2d(3, 64, 1, stride=1, padding=0)
        self.resBlock = ResBlock(64, 32, 512, 0)


    def forward(self, x: torch.Tensor):

        # assert x.shape == self.shape, "形状不对"
        # 输入形状 (n, 49153)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        n = x.shape[0]

        image = x[:, :-1]
        time_index = x[:, -1].to(dtype=torch.int32)

        image = image.view(n, self.img_size, self.img_size, 3)
        image = image.permute(0, 3, 1, 2) # 把channels移到第一个

        step_value = self.time_embedding(time_index)

        image = self.conv2d(image)

        embedded = self.resBlock(image, step_value)

        return embedded # out: (1, 32, self.image_width, self.image_width)



class QCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.state_embed = nn.Sequential(...)
        self.action_embed = nn.Sequential(...)
        self.predictor = nn.Sequential(...)

    def forward(self, s, a):
        s_new = self.state_embed(s)
        a_new = self.action_embed(a)
        x = s_new + a_new
        return self.predictor(x)