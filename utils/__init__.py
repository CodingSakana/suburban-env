
from utils._count_time import count_runtime
from utils import debug

import inspect

def where():
    # 获取当前调用栈
    frame = inspect.currentframe()
    # 获取调用栈的上一帧（即调用 print_call_location 的地方）
    caller_frame = frame.f_back
    # 获取文件名和行号
    filename = caller_frame.f_code.co_filename
    line_number = caller_frame.f_lineno

    print(f"..at {filename}:{line_number}")