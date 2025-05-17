[![Python Tests](https://github.com/giobbu/covariate-shift/actions/workflows/python-tests.yml/badge.svg)](https://github.com/giobbu/covariate-shift/actions/workflows/python-tests.yml)



# Multivariate Shift Detectors

![Covariate Shift](notebooks/imgs/mmd_pdf_drift_overlap.png?raw=true)


We have a **Source** distribution, $P_{\text{source}}$, and a **Target** distribution, $P_{\text{target}}$:

* Source distribution: $(X^{\text{source}}, Y^{\text{source}}) ∼ P_{\text{source}}$

* Target distribution: $(X^{\text{target}}, Y^{\text{target}}) ∼ P_{\text{target}}$

Covariate drifts happen when:

  $$P_{\text{target}}(Y \mid X) = P_{\text{source}}(Y \mid X) \quad \text{but} \quad P_{\text{target}}(X) \ne P_{\text{source}}(X)$$


## Multivariate drift detectors

* ### Maximum Mean Discrepancy Two-Sample Test - MMD Test 

```python
from source.mmd import MMD_test
"""
x_before (np.ndarray): First sample of shape (n, d) from source distribution.
x_after (np.ndarray): Second sample of shape (m, d) from reference distribution.
n_permutations (int): Number of permutations for the permutation test.
sigma (float): Bandwidth parameter for the Gaussian kernel.
"""
sigma = 1.0
n_permutations=1000
mmd_statistic, mmd_perms, pval = MMD_test(x_before, x_after, sigma, n_permutations=n_permutations)
print(f"MMD Statistic: {mmd}, p-value: {pval}")
```

* ### Log-Likelihood Ratio Test - LLR Test

```python
from source.ratio import LLR_test
"""
x_before (np.ndarray): First sample of shape (n, d) from source distribution.
x_after (np.ndarray): Second sample of shape (m, d) from reference distribution.
bandwidth (float): Bandwidth parameter for KDE.
n_permutations (int): Number of permutations for the permutation test. Default is 1000.
"""
bandwidth = 0.5
n_permutations=1000
llr_statistic, p_value = LLR_test(x_before, x_after, bandwidth=bandwidth, n_permutations=n_permutations)
print(f'LLR Statistic: {llr_statistic}, p-value: {p_value}')
```



