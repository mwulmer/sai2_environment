"""
Adopted from https://github.com/stepjam/RLBench/blob/master/rlbench/utils.py
"""
import importlib
import time
from ipdb import set_trace
from collections import deque
import numpy as np


class InvalidTaskName(Exception):
    pass


def name_to_task_class(task_file: str):
    name = task_file.replace(".py", "")
    class_name = "".join([w[0].upper() + w[1:] for w in name.split("_")])
    try:
        mod = importlib.import_module("sai2_environment.tasks.%s" % name)
        mod = importlib.reload(mod)
    except ModuleNotFoundError as e:
        raise InvalidTaskName(
            "The task file '%s' does not exist or cannot be compiled." % name
        ) from e
    try:
        task_class = getattr(mod, class_name)
    except AttributeError as e:
        raise InvalidTaskName(
            "Cannot find the class name '%s' in the file '%s'."
            % (class_name, name)
        ) from e
    return task_class


class Timer(object):
    def __init__(self, frequency=20):
        self._frequency = frequency
        self._ns_update_interval = 1e9 / frequency
        self._update_counter = 0
        self._t_start = time.time_ns()
        self._t_curr = self._t_start
        self._t_next = self._t_curr + self._ns_update_interval

    def wait_for_next_loop(self):
        self._t_curr = time.time_ns()

        if self._t_curr < self._t_next:
            time.sleep((self._t_next - self._t_curr) / 1e9)

        self._t_curr = time.time_ns()
        self._t_next = self._t_curr + self._ns_update_interval
        self._update_counter += 1


class FrameStacker(object):
    def __init__(self, obs_shape, k):
        self._k = k
        self._frames = deque([], maxlen=k)
        for i in range(self._k):
            self._frames.append(np.empty((obs_shape), dtype=np.uint8))

    def get(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)

    def add(self, frame):
        self._frames.append(frame)

