# todo evaluate_model

# import pdb
#
# def example_function():
#     x = 10
#     y = 20
#     pdb.set_trace()
#     z = x + y
#     print(z)
#
# example_function()


"""One example for evaluate saved policy."""

import os

import omnisafe
from config_provider import ConfigProvider


# Just fill your experiment's log directory in here.
# Such as: ~/omnisafe/examples/runs/PPOLag-{SafetyPointGoal1-v0}/seed-000-2023-03-07-20-25-48
LOG_DIR = r'C:\Users\ANASON\Desktop\seed-000-2025-04-11-01-59-53'
# LOG_DIR = r'D:\Desktop\图灵学术对接\学校集群\seed-000-2025-04-07-21-43-13'

# ConfigProvider.img_size = 32
ConfigProvider.img_size = 84
ConfigProvider.device = "cpu"

if __name__ == '__main__':

    from my_env import omnisafe_env

    evaluator = omnisafe.Evaluator(render_mode='rgb_array')
    scan_dir = os.scandir(os.path.join(LOG_DIR, 'torch_save'))
    for index, item in enumerate(scan_dir):
        if index!=2:continue
        if item.is_file() and item.name.split('.')[-1] == 'pt':
            evaluator.load_saved(
                save_dir=LOG_DIR,
                model_name=item.name,
                camera_name='track',
                width=256,
                height=256,
            )
            evaluator.render(num_episodes=1)
            evaluator.evaluate(num_episodes=1)
    scan_dir.close()