import time

import torch

import xfuser.envs as envs

def gpu_timer_decorator(func):
    def wrapper(*args, **kwargs):
        envs.synchronize()
        start_time = time.time()
        result = func(*args, **kwargs)
        envs.synchronize()
        end_time = time.time()

        if torch.distributed.get_rank() == 0:
            print(
                f"{func.__name__} took {end_time - start_time} seconds to run on GPU."
            )
        return result

    return wrapper
