import numpy as np
from scipy import stats
from sklearn.neighbors import KernelDensity

def log_likelihood_kde(data: np.ndarray,
                       kde_model: KernelDensity) -> float:
    """Compute log-likelihood of dataset under the model"""
    return np.sum(kde_model.score_samples(data.reshape(-1, 1)))

def LLR_test(reference_window: np.ndarray, 
                              detection_window: np.ndarray,
                              bandwidth: float = 0.5) -> tuple:
    """
    Perform a likelihood ratio test using Kernel Density Estimation (KDE) to compare two distributions.
    Args:
        reference_window (np.ndarray): Reference data window.
        detection_window (np.ndarray): Detection data window.
        bandwidth (float): Bandwidth for the KDE.
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
    # A conservative approach is to use 2 degrees of freedom as in the Gaussian case
    df = 2
    p_value = 1 - stats.chi2.cdf(lr_statistic, df)
    return lr_statistic, p_value