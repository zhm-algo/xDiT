from unittest.mock import patch

import torch

from xfuser.core import device_utils


@patch("xfuser.envs.get_distributed_backend", return_value="gloo")
@patch("xfuser.envs.get_device", return_value=torch.device("cpu"))
@patch("xfuser.envs.get_device_type", return_value="cpu")
def test_device_utils_forward_to_envs(_mock_device_type, mock_get_device, mock_get_backend):
    assert device_utils.get_device_type() == "cpu"
    assert device_utils.get_distributed_backend() == "gloo"
    assert device_utils.get_device() == torch.device("cpu")
    mock_get_device.assert_called_once_with(None)
    mock_get_backend.assert_called_once_with()


@patch("xfuser.core.device_utils.get_device", return_value=torch.device("xpu", 1))
def test_get_local_device_indexed(mock_get_device):
    assert device_utils.get_local_device(1) == "xpu:1"
    mock_get_device.assert_called_once_with(1)


@patch("xfuser.core.device_utils.get_device", return_value=torch.device("cpu"))
def test_get_local_device_indexless(mock_get_device):
    assert device_utils.get_local_device(0) == "cpu"
    mock_get_device.assert_called_once_with(0)
