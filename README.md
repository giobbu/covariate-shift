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
llr_statistic, llr_perms, p_value = LLR_test(x_before, x_after, bandwidth=bandwidth, n_permutations=n_permutations)
print(f'LLR Statistic: {llr_statistic}, p-value: {p_value}')
```

## Streaming batch data simulator
* ### Data stream with simulated mean drifts
![Batch Streaming Animation](imgs/monitoring.gif?raw=true)

* ### Drifts detected with MOVING reference window with LLR-test
Useful to build adaptive learning models in streaming environments. The learning model is updated or rebuilt as soon as a drift-event is detected.
![Moving Window](imgs/drift_detection_moving_reference_window.png?raw=true)
![P-Value Moving Window](imgs/moving_window_llr_statistic.png?raw=true)

* ### Drifts detected with FIXED reference window with LLR-test
Useful to monitor automated systems over time either in an offline or online environments. The reference period should be representative.
![Fixed Window](imgs/drift_detection_fixed_reference_window.png?raw=true)
![P-Value Fixed Window](imgs/fixed_window_llr_statistic.png?raw=true)


## Lambda framework for near real-time covariate monitoring
### Offline layer: define the **Reference Component**
Using data collected offline, perform the following steps:
* 1. Define the Reference Distribution: select a fixed portion of the offline data to construct a stable covariate distribution representing normal condition.
* 2. Simulate streaming data via batch sampling: from the remaining offline data, draw multiple batches to simulate streaming behavior. For each batch, compute the statistic of interest.
* 3. Model Expected Statistical Variation: Aggregate the statistics to form a distribution that captures the natural variability of the statistic under normal conditions. This distribution serves as a reference and is passed to the streaming layer for real-time monitoring.
![Batch-Layer](imgs/lambda_batch.gif?raw=true)
### Streaming layer: define **Monitoring component**
