
import sys
import torch
from config_provider import ConfigProvider

def parse_bool(string):
    if string == 'True' or string == "true":
        return True
    return False

def check_device():
    print("是否支持 CUDA:", torch.cuda.is_available())
    print("可用 GPU 数量:", torch.cuda.device_count())

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("当前只支持 CPU")

    print("\n")

def init_cuda_safety():
    import os
    import torch

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:16,garbage_collection_threshold:0.6"
    os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
    torch.cuda.empty_cache()
    # print("[CUDA] Memory summary before training:")
    # print(torch.cuda.memory_summary())

    # 让每次内存分配更为严格
    torch.cuda.set_per_process_memory_fraction(0.96, 0)


if __name__ == '__main__':
    # python -m run mode=debug size=128 device=cuda
    args = {
        "mode": "debug", # "train"
        "num_of_debug": "5",

        "device": "cpu",
        "size": "128",
        "use_count_time": "False",
        "track_time": "False",
        "debug_print": "False",
    }

    for arg in sys.argv[1:]:
        k, v = arg.split("=", 1)
        args[k] = v

    ConfigProvider.device = args["device"]
    ConfigProvider.img_size = int(args["size"])

    ConfigProvider.use_count_time = parse_bool(args["use_count_time"])
    ConfigProvider.track_time = parse_bool(args["track_time"])
    ConfigProvider.debug_print = parse_bool(args["debug_print"])


    print(f"Running args: ")
    ConfigProvider.print_args(prefix="\t")
    print("\n")

    check_device()


    if args['mode'] == 'debug':
        from my_env import layout_env
        env = layout_env.LayoutEnv(size=ConfigProvider.img_size)
        for _ in range(int(args['num_of_debug'])):
            layout_env.round_test(env)

    elif args['mode'] == 'train':

        # init_cuda_safety() 没啥用的东西！

        import omnisafe
        from my_env import omnisafe_env
        agent = omnisafe.Agent(
            'PPOLag',
            'suburban_layout'
        )
        agent.learn()

    else:
        raise ValueError(f"Unrecognized mode: {args['mode']}")

