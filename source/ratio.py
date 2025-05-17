import numpy as np
from sklearn.neighbors import KernelDensity
from joblib import Parallel, delayed

def log_likelihood_kde(data: np.ndarray,
                       kde_model: KernelDensity) -> float:
    """Compute log-likelihood of dataset under the model"""
    return np.sum(kde_model.score_samples(data)) #.reshape(-1, 1)

def permute_and_compute_lr(i: int, ll_null: float,
                           combined_data: np.ndarray,
                           reference_window: np.ndarray,
                            detection_window: np.ndarray,
                           bandwidth: float) -> float:
    """
    Permute the data and compute the likelihood ratio statistic.
    Args:
        i (int): Permutation index.
        ll_null (float): Log-likelihood under the null hypothesis.
        reference_window (np.ndarray): Reference data window.
        bandwidth (float): Bandwidth for the KDE.
    Returns:
        float: Likelihood ratio statistic for the permuted data.
    """
    " Permute the data and compute the likelihood ratio statistic"
    np.random.shuffle(combined_data)
    perm_ref = combined_data[:len(reference_window)]
    perm_det = combined_data[len(detection_window):]
    kde_perm_ref = KernelDensity(bandwidth=bandwidth).fit(perm_ref)
    kde_perm_det = KernelDensity(bandwidth=bandwidth).fit(perm_det)
    ll_perm_ref = log_likelihood_kde(perm_ref, kde_perm_ref)
    ll_perm_det = log_likelihood_kde(perm_det, kde_perm_det)
    ll_perm_alt = ll_perm_ref + ll_perm_det
    return 2 * (ll_perm_alt - ll_null)

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
    # Alternative hypothesis
    kde_ref = KernelDensity(bandwidth=bandwidth).fit(reference_window)
    kde_det = KernelDensity(bandwidth=bandwidth).fit(detection_window)
    ll_ref = log_likelihood_kde(reference_window, kde_ref)
    ll_det = log_likelihood_kde(detection_window, kde_det)
    ll_alt = ll_ref + ll_det
    # Null hypothesis - same distribution
    combined_data = np.concatenate([reference_window, detection_window])
    kde_null = KernelDensity(bandwidth=bandwidth).fit(combined_data)
    ll_null = log_likelihood_kde(combined_data, kde_null)
    # Likelihood ratio statistic
    lr_statistic = 2 * (ll_alt - ll_null)
    # Use joblib to parallelize the permutation test
    lr_perms = Parallel(n_jobs=-1)(delayed(permute_and_compute_lr)(
                                    i, ll_null,combined_data, reference_window, detection_window, bandwidth)
                                    for i in range(n_permutations))
    lr_perms = np.array(lr_perms)
    # p-value: fraction of permutations with LR >= observed LR
    p_value = np.mean(lr_perms >= lr_statistic)
    return lr_statistic, p_value
