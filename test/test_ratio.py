import numpy as np
from source.ratio import LLR_test


def test_LLR_test_kde_output(identical_samples):
    " Test likelihood ratio test output shape and types."
    x, y = identical_samples
    lr_statistic, p_value = LLR_test(x, y, bandwidth=0.5)
    assert isinstance(lr_statistic, float)
    assert isinstance(p_value, float)

def test_LLR_reproducibility(identical_samples):
    " Test likelihood ratio test function reproducibility with same inputs."
    x, y = identical_samples
    lr_statistic1, p_value1 = LLR_test(x, y, bandwidth=0.5)
    lr_statistic2, p_value2 = LLR_test(x, y, bandwidth=0.5)
    np.testing.assert_allclose(lr_statistic1, lr_statistic2)
    np.testing.assert_allclose(p_value1, p_value2)