import builtins

import numpy as np
import torch
import torchvision
from torch import nn

import time

from torch import autocast


def speed_test(model : nn.Module,
               x,
               iterations : int = 1000,
               is_verbose : bool = True,
               auto_cast  : bool = False):

    model.to("cuda")
    x = x.to("cuda")

    times = []
    print(f"running speed test")
    for i in range(iterations):

        if auto_cast:
            with autocast("cuda"):
                start_time = time.time()
                model(x)
                times.append(time.time() - start_time)

        else:
            start_time = time.time()
            model(x)
            times.append(time.time() - start_time)

        if is_verbose and i % 50 == 0:
            print(f"[{i}] {times[-1]}")

    std  = np.std(times)
    mean = np.mean(times)

    print(f"mean iteration time: {mean}")
    print(f"mean iteration/sec : {1/mean} iter/s")
    print(f"std: {std}")


