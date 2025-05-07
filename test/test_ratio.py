import numpy as np
from source.ratio import likelihood_ratio_test_kde


def test_likelihood_ratio_test_kde_output(identical_samples):
    " Test likelihood ratio test output shape and types."
    x, y = identical_samples
    lr_statistic, p_value = likelihood_ratio_test_kde(x, y, bandwidth=0.5)
    assert isinstance(lr_statistic, float)
    assert isinstance(p_value, float)

def test_likelihood_reproducibility(identical_samples):
    " Test likelihood ratio test function reproducibility with same inputs."
    x, y = identical_samples
    lr_statistic1, p_value1 = likelihood_ratio_test_kde(x, y, bandwidth=0.5)
    lr_statistic2, p_value2 = likelihood_ratio_test_kde(x, y, bandwidth=0.5)
    np.testing.assert_allclose(lr_statistic1, lr_statistic2)
    np.testing.assert_allclose(p_value1, p_value2)