from __future__ import annotations

from typing import Optional

import torch

import xfuser.envs as envs


def get_device_type() -> str:
    return envs.get_device_type()


def get_device(index: Optional[int] = None) -> torch.device:
    return envs.get_device(index)


def get_torch_device_module():
    return envs.get_torch_device_module()


def get_distributed_backend() -> str:
    return envs.get_distributed_backend()


def get_local_device(index: int) -> str:
    device = get_device(index)
    if device.index is None:
        return device.type
    return f"{device.type}:{device.index}"
