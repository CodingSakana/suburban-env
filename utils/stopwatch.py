import time

class Stopwatch:
    def __init__(self, name: str):
        self.time = time.perf_counter_ns()
        self.name = name

    def press(self, label: str=""):
        temp = time.perf_counter_ns()
        print(f"[{self.name}] {label} 用时 {temp - self.time:,} ns")
        self.time = time.perf_counter_ns()