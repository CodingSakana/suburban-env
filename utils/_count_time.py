
# 生产环境中请更换成 toggle -> False
def toggle():
    return True


import time
import atexit
import numpy as np
from typing import List, Dict, Tuple

function_runtime_track:Dict[str,Tuple[str, int, List[int]]] = {}
def on_exit():
    if function_runtime_track:
        print("\n\033[34mRuntime Summary:\033[0m")
        for k, v in function_runtime_track.items():
            th = v[1]
            thQ3 = th*0.75
            def w(value):
                value = int(value)
                color_sign=32
                if value <= thQ3:
                    color_sign = 32
                elif thQ3 < value <= th:
                    color_sign = 33
                elif value > th:
                    color_sign = 31
                return f"\033[{color_sign}m{split_every3(value)}\033[0m"

            print(f"{v[0]}  at {k}")
            print(f"\t  mean:{w(sum(v[2])/len(v[2]))} min:{w(min(v[2]))} max:{w(max(v[2]))} times:{len(v[2])}")
            # print(f"{v[0]}: mean:{sum(v[2])/len(v[2]):.0f} min:{min(v[2])} max:{max(v[2]):.0f}")
            # print(f"\t  at {k}")

def split_every3(data):
    # make "123456" to “123 456”
    data = list(str(data))
    length = len(data)
    times = (length - 1) // 3
    for i in range(times):
        data.insert(length-(i+1)*3, "'")
    return "".join(data)



# 注册on_exit函数在脚本退出时调用
if toggle():
    atexit.register(on_exit)

def _count_runtime_core(func=None, show_args=False, threshold=100000, warn_only=False, track=False):
    threshold_Q3 = threshold * 0.75
    def wrapper(func):
        def inner(*args, **kwargs):

            # 获取函数在代码中的信息
            code = func.__code__
            filename = code.co_filename
            first_lineno = code.co_firstlineno

            # 计算函数运行时间和返回值
            start_time = time.perf_counter_ns() #1e-9
            result = func(*args, **kwargs)
            time_taken = time.perf_counter_ns() - start_time

            # 追踪函数运行时间
            target = f"{filename}:{first_lineno+1}"
            if target not in function_runtime_track:
                function_runtime_track[target] = (func.__name__, threshold, [])
            function_runtime_track[target][2].append(time_taken)

            if track:
                # 计算运行时间是否超过阈值
                color_sign = 32
                time_warn = False
                if time_taken <= threshold_Q3:
                    color_sign = 32
                elif threshold_Q3 < time_taken <= threshold:
                    color_sign = 33
                    time_warn = True
                elif time_taken > threshold:
                    color_sign = 31
                    time_warn = True

                # 组装提示信息
                time_msg = f"Runtime of {func.__name__}(): \033[{color_sign}m{split_every3(time_taken)}\033[0m ns (th: {threshold})"
                if show_args:
                    time_msg += f"\n\t  with {args} {kwargs}"
                time_msg += f"\n\t  at {filename}:{first_lineno+1}"

                # 输出
                if warn_only:
                    if time_warn:
                        print("Warn "+time_msg)
                else:
                    print(time_msg)

            return result
        return inner

    if func is None:
        return wrapper
    else:
        return wrapper(func)

def _count_runtime_fake(func=None, *args, **kwargs):
    def wrapper(func):
        return func
    if func is None:
        return wrapper
    else:
        return wrapper(func)


count_runtime = _count_runtime_core if toggle() else _count_runtime_fake
if count_runtime == _count_runtime_core:
    print("\033[33m" + "utils.count_time: 生产环境中请更换成 _count_runtime_fake" + "\033[0m")
    print("\033[33m" + f"\t  at {count_runtime.__code__.co_filename}:{toggle.__code__.co_firstlineno}" + "\033[0m")
else:
    print("\033[32m" + "utils.count_time: 生产环境未启用" + "\033[0m")
    print("\033[32m" + f"\t  at {count_runtime.__code__.co_filename}:{toggle.__code__.co_firstlineno}" + "\033[0m")
