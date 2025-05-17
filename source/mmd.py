import numpy as np

def rbf_kernel(a, b, sigma=1.0):
    " Computes the RBF kernel between two sets of samples."
    dists = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    return np.exp(-dists**2 / (2 * sigma**2))

def MMD_test(x: np.ndarray, y: np.ndarray, sigma: float = 1.0, n_permutations: int = 1000) -> tuple:
    """
    Computes the Maximum Mean Discrepancy (MMD) between two samples x and y using a Gaussian kernel.
    Args:
        x (np.ndarray): First sample of shape (n, d).
        y (np.ndarray): Second sample of shape (m, d).
        n_permutations (int): Number of permutations for the permutation test.
        sigma (float): Bandwidth parameter for the Gaussian kernel.
    Returns:
        mmd_obs (float): Observed MMD value.
        mmd_perms (np.ndarray): MMD values from the permutations.
        p_value (float): p-value from the permutation test.
    """

    # Set random seed for reproducibility
    rng = np.random.default_rng(42)
    # Check input shapes
    n, d = x.shape
    m, d2 = y.shape
    assert d == d2
    # Original MMD
    k_xx = rbf_kernel(x, x, sigma) 
    k_yy = rbf_kernel(y, y, sigma)
    k_xy = rbf_kernel(x, y, sigma)
    # mmd = 1/n^2 * sum_i sum_j k(x_i, x_j) + 1/m^2 * sum_i sum_j k(y_i, y_j) - 2/(nm) * sum_i sum_j k(x_i, y_j)
    mmd_statistic = (k_xx.sum() / (n * (n - 1)) + k_yy.sum() / (m * (m - 1)) - 2 * k_xy.sum() / (n * m))
    # Permutation test
    z = np.vstack([x, y])
    total = n + m
    mmd_perms = []
    for _ in range(n_permutations):
        idx = rng.permutation(total)
        x_perm = z[idx[:n]]
        y_perm = z[idx[n:]]
        k_xx_perm = rbf_kernel(x_perm, x_perm)
        k_yy_perm = rbf_kernel(y_perm, y_perm)
        k_xy_perm = rbf_kernel(x_perm, y_perm)
        mmd_perm = (k_xx_perm.sum() / (n * (n - 1)) + k_yy_perm.sum() / (m * (m - 1)) - 2 * k_xy_perm.sum() / (n * m))
        mmd_perms.append(mmd_perm)
    # p-value: fraction of permutations with MMD >= observed MMD
    mmd_perms = np.array(mmd_perms)
    p_value = np.mean(mmd_perms >= mmd_statistic)
    return mmd_statistic, mmd_perms, p_value