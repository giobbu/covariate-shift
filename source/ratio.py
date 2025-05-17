import numpy as np
from scipy import stats
from sklearn.neighbors import KernelDensity

def log_likelihood_kde(data: np.ndarray,
                       kde_model: KernelDensity) -> float:
    """Compute log-likelihood of dataset under the model"""
    return np.sum(kde_model.score_samples(data.reshape(-1, 1)))

def LLR_test(reference_window: np.ndarray, 
                              detection_window: np.ndarray,
                              bandwidth: float = 0.5,
                              n_permutations: int = 1000) -> tuple:
    """
    Perform a likelihood ratio test using Kernel Density Estimation (KDE) to compare two distributions.
    Args:
        reference_window (np.ndarray): Reference data window.
        detection_window (np.ndarray): Detection data window.
        bandwidth (float): Bandwidth for the KDE.
        n_permutations (int): Number of permutations for the permutation test. Default is 1000.
    Returns:
        tuple: Likelihood ratio statistic and p-value. 
    """
    # Reshape data for KDE
    ref_data = reference_window.reshape(-1, 1)
    det_data = detection_window.reshape(-1, 1)
    # Alternative hypothesis
    kde_ref = KernelDensity(bandwidth=bandwidth).fit(ref_data)
    kde_det = KernelDensity(bandwidth=bandwidth).fit(det_data)
    ll_ref = log_likelihood_kde(ref_data, kde_ref)
    ll_det = log_likelihood_kde(det_data, kde_det)
    ll_alt = ll_ref + ll_det
    # Null hypothesis - same distribution
    combined_data = np.concatenate([ref_data, det_data])
    kde_null = KernelDensity(bandwidth=bandwidth).fit(combined_data)
    ll_null = log_likelihood_kde(combined_data, kde_null)
    # Likelihood ratio statistic
    lr_statistic = 2 * (ll_alt - ll_null)
    # Permutation test
    lr_perms = []
    for _ in range(n_permutations):
        permuted_data = np.random.permutation(combined_data)
        perm_ref = permuted_data[:len(ref_data)]
        perm_det = permuted_data[len(ref_data):]
        ll_alt_perm = log_likelihood_kde(perm_ref, kde_ref) + log_likelihood_kde(perm_det, kde_det)
        lr_statistic_perm = 2 * (ll_alt_perm - ll_null)
        lr_perms.append(lr_statistic_perm)
    lr_perms = np.array(lr_perms)
    # p-value: fraction of permutations with LR >= observed LR
    p_value = np.mean(lr_perms >= lr_statistic)
    return lr_statistic, p_value
