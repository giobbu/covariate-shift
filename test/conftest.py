import pytest
import numpy as np

@pytest.fixture
def identical_samples():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (50, 5))
    y = np.copy(x)
    return x, y
