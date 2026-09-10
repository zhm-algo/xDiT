from unittest.mock import patch

from xfuser.core import device_utils


@patch("xfuser.envs.get_device_type", return_value="cpu")
def test_device_utils_forward_to_envs(_mock_device_type):
    assert device_utils.get_device_type() == "cpu"
    assert device_utils.get_distributed_backend() == "gloo"
    assert str(device_utils.get_device()) == "cpu"
