
import os

path_prefix = "./runs/PPOLag-{suburban_layout}"
seed_dir = max(os.listdir(path_prefix))
relative_path = os.path.join(path_prefix, seed_dir)
print(f"opening {relative_path}")
os.system("tensorboard --logdir=" + relative_path)