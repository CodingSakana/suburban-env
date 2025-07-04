import time
import cv2
import numpy as np
import os
import shutil
import pickle
from PIL import Image

save_dir = "./flask_model/history_action_db/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


def get_saves():
    dirs = os.listdir(save_dir)
    return [
        i for i in dirs if os.path.isdir(
            os.path.join(save_dir, i)
        )
    ]


def save(name: str, actions: list, cover: np.ndarray, info):

    if name in get_saves():
        return f"'{name}' already exists."

    if not actions:
        return f"'{name}' no need to save because the actions is empty."

    temp_dir = os.path.join(save_dir, name)
    os.makedirs(temp_dir, exist_ok=True)

    info['name'] = name
    info['time'] = time.time()

    with open(os.path.join(temp_dir, "actions"), "wb") as f:
        pickle.dump(actions, f)

    with open(os.path.join(temp_dir, "info"), "wb") as f:
        pickle.dump(info, f)

    # 将 NumPy 数组转换为 PIL 图像
    cover = cv2.cvtColor(cover, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(cover)
    img_path = os.path.join(temp_dir, "cover.jpg")
    # 保存图像为 JPEG 文件
    image.save(img_path, 'JPEG')

    return f"'{name}' saved successfully."


def remove(name):

    if name not in get_saves():
        return f"'{name}' does not exist."

    shutil.rmtree(os.path.join(save_dir, name))

    return f"'{name}' removed successfully."


def load(name, wrapper):

    if name not in get_saves():
        return f"'{name}' does not exist."

    temp_dir = os.path.join(save_dir, name)

    with open(os.path.join(temp_dir, "actions"), "rb") as f:
        actions = pickle.load(f)

    wrapper(actions)

    return f"'{name}' loaded successfully."

def info(name):
    with open(os.path.join(save_dir, name, "info"), "rb") as f:
        return pickle.load(f)

def update_info(name, key, value):
    with open(os.path.join(save_dir, name, "info"), "rb") as f:
        info = pickle.load(f)

    assert key in info
    assert type(value) == type(info[key])
    info[key] = value

    print(f"new info to '{name}': {info}")
    with open(os.path.join(save_dir, name, "info"), "wb") as f:
        pickle.dump(info, f)

def all_info():
    return [info(i) for i in get_saves()]


def cover(name):
    if name not in get_saves():
        raise ValueError(f"'{name}' does not exist.")

    return os.path.join(save_dir, name, "cover.jpg")