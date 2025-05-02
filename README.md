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
* ### Likelihood Ratio Test

## Kernels

## References

* https://jmlr.csail.mit.edu/papers/v13/gretton12a.html



