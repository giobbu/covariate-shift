import numpy as np
from source.mmd import rbf_kernel, MMD_test

def test_rbf_kernel_identity():
    " Test RBF kernel identity property."
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    k = rbf_kernel(x, x, sigma=1.0)
    assert np.allclose(np.diag(k), 1.0)

def test_rbf_kernel_symmetry():
    " Test RBF kernel symmetry."
    x = np.random.rand(5, 3)
    y = np.random.rand(4, 3)
    k1 = rbf_kernel(x, y)
    k2 = rbf_kernel(y, x)
    assert np.allclose(k1, k2.T)

def test_mmd_output_shape(identical_samples):
    " Test MMD function output shape and types."
    x, y = identical_samples
    mmd_val, mmd_perms, p_val = MMD_test(x, y, sigma=1.0, n_permutations=10)
    assert isinstance(mmd_val, float)
    assert isinstance(p_val, float)
    assert isinstance(mmd_perms, np.ndarray)
    assert len(mmd_perms) == 10

def test_mmd_reproducibility(identical_samples):
    " Test MMD function reproducibility with same inputs."
    x, y = identical_samples
    mmd1, _, _ = MMD_test(x, y, sigma=1.0, n_permutations=10)
    mmd2, _, _ = MMD_test(x, y, sigma=1.0, n_permutations=10)
    np.testing.assert_allclose(mmd1, mmd2)
