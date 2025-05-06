# Multivariate Shift Detectors

![Covariate Shift](notebooks/imgs/mmd_pdf_drift_overlap.png?raw=true)


## **Covariate (Univariate/Multivariate) Drift**: 
We have a **Source** distribution, $P_{\text{source}}$, and a **Target** distribution, $P_{\text{target}}$:

* Source distribution: $(X^{\text{source}}, Y^{\text{source}}) ∼ P_{\text{source}}$

* Target distribution: $(X^{\text{target}}, Y^{\text{target}}) ∼ P_{\text{target}}$

Covariate drifts happen when:

  $$P_{\text{target}}(Y \mid X) = P_{\text{source}}(Y \mid X) \quad \text{but} \quad P_{\text{target}}(X) \ne P_{\text{source}}(X)$$


## Multivariate drift detectors

* ### Maximum Mean Discrepancy Two-Sample Test 

```python
from source.mmd import MMD
"""
x_before (np.ndarray): First sample of shape (n, d) from source distribution.
x_after (np.ndarray): Second sample of shape (m, d) from reference distribution.
n_permutations (int): Number of permutations for the permutation test.
sigma (float): Bandwidth parameter for the Gaussian kernel.
"""
sigma = 1.0
mmd, mmd_perms, pval = MMD(x_before, x_after, sigma, n_permutations=1000)
print(f"MMD Statistic: {mmd}, p-value: {pval}")
```

* ### Likelihood Ratio Test

```python
from source.ratio import likelihood_ratio_test_kde
"""
x_before (np.ndarray): First sample of shape (n, d) from source distribution.
x_after (np.ndarray): Second sample of shape (m, d) from reference distribution.
bandwidth (float): Bandwidth parameter for KDE.
"""
bandwidth = 0.5
lr_statistic, p_value = likelihood_ratio_test_kde(x_before, x_after, bandwidth=bandwidth)
print(f'Likelihood Ratio statistic: {lr_statistic}, p-value: {p_value}')
```



