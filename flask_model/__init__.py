import torch
from flask import jsonify

import flask_model.history_action
import random

from config_provider import ConfigProvider
from my_env import LayoutEnv, Space

from typing import List


class DataModel(object):

    def __init__(self):
        self.layout_env = LayoutEnv()
        self.total_reward = torch.tensor(.0)
        self.total_cost = torch.tensor(.0)

        self.history_stack: List[torch.Tensor] = []

        self.commands = {
            "step": self.cmd_step,
            "random": self.cmd_random_step,
            "reset": self.cmd_reset,
            "undo": self.cmd_undo,
            "redo": self.cmd_redo,
            "save": self.cmd_save_history,
            "remove": self.cmd_remove_history,
            "restore": self.cmd_restore_history,
            "sinfo": self.cmd_save_info,
            "re_eval": self.cmd_re_evaluate,
        }

    def parse_command(self, command, args):
        print(f"command received: {command}")
        print(f"args received: {args}")

        if command in self.commands:
            return self.commands[command](args)
        else:
            return "Unknown Command"

    def cmd_step(self, args):
        assert len(args) == 3

        space_type_name = self.layout_env.lay_type(
            self.layout_env.step_index
        ).__name__
        reward, cost, terminated, truncated, info = self.util_step(
            torch.tensor([float(i) for i in args], device=ConfigProvider.device),
        )

        self.history_stack.clear()

        return f"reward: {reward}\ncost: {cost}\ninfo: {info}\nspace_type:{space_type_name}"

    def cmd_reset(self, args):
        assert len(args) == 0
        self.util_reset()
        return "reset success"

    def cmd_random_step(self, args):
        assert len(args) == 0

        space_type_name = self.layout_env.lay_type(
            self.layout_env.step_index
        ).__name__
        reward, cost, terminated, truncated, info = self.util_step(
            torch.tensor([random.random() for _ in range(3)], device=ConfigProvider.device),
        )

        self.history_stack.clear()

        return f"reward: {reward}\ncost: {cost}\ninfo: {info}\nspace_type:{space_type_name}"

    def cmd_undo(self, args):
        assert len(args) == 0

        if not self.layout_env.action_history:
            return "nothing to undo"

        *remained_actions, undo_action = self.layout_env.action_history
        self.history_stack.append(undo_action)

        self.util_reset()
        for action in remained_actions:
            self.util_step(action)

        return f"undo action: {undo_action}"

    def cmd_redo(self, args):
        assert len(args) == 0

        if not self.history_stack:
            return "nothing to redo"

        redo_action = self.history_stack.pop()
        reward, cost, terminated, truncated, info = self.util_step(redo_action)

        return f"redo action: {redo_action}\nreward: {reward}\ncost: {cost}\ninfo: {info}"



    def cmd_restore_history(self, args):
        assert len(args) == 1
        name = args[0]

        def wrapper(actions):
            self.util_reset()
            for action in actions:
                self.util_step(action)

        return history_action.load(name, wrapper)


    def cmd_save_history(self, args):
        assert len(args) == 1
        name = args[0]
        return history_action.save(
            name, self.layout_env.action_history,
            self.layout_env.get_high_resolution_image(256),
            info={
                "step_index": f"{self.layout_env.step_index}/{self.layout_env.step_sum}",
                "total_reward": float(self.total_reward),
                "total_cost": float(self.total_cost),
            }
        )

    def cmd_remove_history(self, args):
        assert len(args) == 1
        name = args[0]
        return history_action.remove(name)

    def cmd_save_info(self, args):
        assert len(args) == 1
        name = args[0]
        temp = jsonify(history_action.info(name))
        print(f"info: {temp.json}")
        return temp


    def cmd_re_evaluate(self, args):
        assert len(args) == 0

        names = history_action.get_saves()

        def outer(save_name):
            def wrapper(actions):
                env = LayoutEnv()
                total_reward = torch.tensor(.0)
                total_cost = torch.tensor(.0)
                for action in actions:
                    _, reward, cost, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    total_cost += cost

                history_action.update_info(save_name, "total_reward", float(total_reward))
                history_action.update_info(save_name, "total_cost", float(total_cost))

            return wrapper

        for name in names:
            history_action.load(name, outer(name))

        return "re-evaluation success"


    def get_all_save_info(self):
        return history_action.all_info()


    def get_specific_history_cover(self, name):
        return history_action.cover(name)


    def get_total_state(self):
        return f"{self.layout_env.step_index}/{self.layout_env.step_sum}_{self.total_reward:.2f}_{self.total_cost:.2f}"

    def get_current_radius_min_max(self):
        space_type: Space = self.layout_env.lay_type(
            self.layout_env.step_index
        )
        return f"{space_type.min_r}_{space_type.max_r}"

    def get_current_space_type(self):
        space_type_name: Space = self.layout_env.lay_type(
            self.layout_env.step_index
        ).__name__
        return space_type_name

    def get_obs(self):
        return self.layout_env.image

    def get_obs_plus(self, size:int):
        return self.layout_env.get_high_resolution_image(size)


    def get_all_actions(self):
        return [
            t.tolist() for t in self.layout_env.action_history
        ]


    def util_reset(self):
        self.layout_env.reset()
        self.total_reward = torch.tensor(.0)
        self.total_cost = torch.tensor(.0)

    def util_step(self, action):
        _, reward, cost, terminated, truncated, info = self.layout_env.step(action)
        self.total_reward += reward
        self.total_cost += cost
        return reward, cost, terminated, truncated, info