from enum import Enum

class Device(Enum):
    cpu = 'cpu'
    cuda = 'cuda:0'

class ConfigProvider:
    device = Device.cpu.value
    track_time: bool = False
    debug_print: bool = False

    use_count_time: bool = False
    use_curve_debug: bool = False
    img_size: int = 64 # 训练时用的

    @classmethod
    def print_args(cls, prefix:str=""):
        print(f"{prefix}device: {cls.device}")
        print(f"{prefix}track_time: {cls.track_time}")
        print(f"{prefix}debug_print: {cls.debug_print}")
        print(f"{prefix}use_count_time: {cls.use_count_time}")
        print(f"{prefix}use_curve_debug: {cls.use_curve_debug}")
        print(f"{prefix}img_size: {cls.img_size}")

def dprint(*args, **kwargs):
    """
    debug print，在 ConfigProvider.debug_print 启用时 打印信息
    """
    if ConfigProvider.debug_print:
        print("[debug] ", end="")
        print(*args, **kwargs)