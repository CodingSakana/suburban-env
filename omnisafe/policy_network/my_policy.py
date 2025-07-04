import torch
from torch import nn

from omnisafe.typing import Activation, CriticType, InitFunction, OmnisafeSpace
from gymnasium.spaces import Box #, Discrete
from typing import Tuple

from utils.time_embedding import TimeEmbedding, ResBlock


class PolicyProvider:

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
    def factory_state_preprocessor(cls, obs: "Box") -> nn.Module:

        if cls.network is None: # actor和critic应该共享这个 cls.network
            img_length = cls.get_omnisafe_obs_shape()

            assert cls.img_size is not None, 'img_size怕是没设置吧'
            assert obs.shape == (img_length + 1,), 'obs的形状不太对吧'

            cls.network = Policy()

        return cls.network

    @classmethod
    def factory_state_action_connector(cls, obs: "Box", act: "Box"):
        # todo 如果是q-critic网络的话
        raise NotImplementedError


class Policy(torch.nn.Module):
    def __init__(self):

        super().__init__()

        self.img_size = PolicyProvider.img_size
        self.shape = PolicyProvider.omnisafe_obs_shape

        self.time_embedding = TimeEmbedding(30, self.img_size, 64)

        self.conv2d_1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.conv2d_2 = nn.Conv2d(64, 128, 3, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()

        self.resBlock = ResBlock(128, 128, 64, 0)
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))  # => (1, 128, 1, 1)

        self.flat = nn.Flatten()  # => (128,)


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

        image = self.relu(self.bn1(self.conv2d_1(image)))
        image = self.relu(self.bn2(self.conv2d_2(image)))

        embedded = self.resBlock(image, step_value)

        embedded = self.flat(self.pooling(embedded))

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




if __name__ == '__main__':

    img_size = 256
    x = torch.randn((3, img_size, img_size))
    x = torch.cat((
        x.flatten(start_dim=0),torch.tensor([1])
    ))
    PolicyProvider.set_img_size(img_size)
    result = Policy().forward(x)
    print(result.shape)