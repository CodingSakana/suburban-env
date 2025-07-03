
import omnisafe
from my_env import omnisafe_env

if __name__ == '__main__':
    agent = omnisafe.Agent(
        'PPOLag',  #
        # 'Ant-v4',
        'suburban_layout'
    )

    agent.learn()