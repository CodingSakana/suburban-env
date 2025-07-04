from __future__ import annotations



# import pandas
import torch
import numpy as np

from gymnasium.spaces import Box

import omnisafe
from omnisafe.envs.core import CMDP, env_register
from typing import Any

from my_env.layout_env import LayoutEnv
from omnisafe.typing import OmnisafeSpace
from config_provider import dprint, ConfigProvider


@env_register
class MyCMDP(CMDP):

    need_time_limit_wrapper = False
    need_auto_reset_wrapper = True

    _num_envs = 1 # todo 并行训练？
    _time_limit: int | None = None
    need_evaluation: bool = True

    # define what tasks the environment support.
    _support_envs = ['suburban_layout']

    @classmethod
    def support_envs(cls):
        return [
            "suburban_layout"
        ]

    def __init__(self, env_id: str, **kwargs: Any) -> None:
        super().__init__(env_id, **kwargs)

        # 默认size?
        self.size = kwargs['size'] if 'size' in kwargs else ConfigProvider.img_size

        omnisafe.PolicyProvider.set_img_size(self.size)

        self.layout_env = LayoutEnv(
            size=self.size,
        )

        self._action_space = Box(0, 1, (3,), dtype=np.float32)

    def step(self, action: torch.Tensor) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]:
        obs, reward, cost, terminated, truncated, info = self.layout_env.step(action)

        return obs, reward, cost, terminated, truncated, info

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[
        torch.Tensor, dict[str, Any]]:
        # self.layout_env = LayoutEnv(roads=self.roads)
        # return self.layout_env.get_obs(), {}
        obs = self.layout_env.reset()
        dprint("MyCMDP reset")
        return obs, {} # todo info没写

    def set_seed(self, seed: int) -> None:
        pass

    def render(self) -> Any:
        # self.layout_env.show_plot()
        self.layout_env.show_plot(
            image=self.layout_env.get_high_resolution_image(256)
        )

    def close(self) -> None:
        return

    # @property
    # def action_space(self) -> OmnisafeSpace:
    #     return Box(0, 1, (3,), dtype=np.float32)

    @property
    def observation_space(self) -> OmnisafeSpace:
        return Box(0, 255, (self.size**2 *3 + 1,), dtype=np.float32)
        # return Box(0, 255, (1, 32, self.size, self.size), dtype=np.float32)

if __name__ == '__main__':


    agent = omnisafe.Agent(
        'PPOLag', #
        # 'Ant-v4',
        'suburban_layout'
    )

    agent.learn()

