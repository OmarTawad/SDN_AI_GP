#tests/test_seed.py
import numpy as np
import torch

from dos_detector.utils.seed import seed_everything


def test_seed_determinism():
    seed_everything(123)
    torch_vals1 = torch.randn(3)
    np_vals1 = np.random.rand(3)
    seed_everything(123)
    torch_vals2 = torch.randn(3)
    np_vals2 = np.random.rand(3)
    assert torch.allclose(torch_vals1, torch_vals2)
    assert np.allclose(np_vals1, np_vals2)