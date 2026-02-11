# Stage 1: Candidate CG Evaluation Metrics

## Overview

To evaluate coarse-grained (CG) models in a systematic and interpretable manner, we organize candidate metrics into three complementary categories:

1. Distribution-based fidelity  
2. Dynamics-based fidelity  
3. Sampling efficiency  

These dimensions capture distinct but interrelated aspects of model quality. Distribution-based metrics assess thermodynamic consistency with atomistic (AA) reference data when available. Dynamics-based metrics evaluate the preservation of slow processes and metastable behavior. Sampling efficiency metrics quantify how effectively the CG model explores configurational space, independent of AA reference.

---

## 1. Distribution-based Evaluation

### Core Question

Does the CG model reproduce the atomistic (AA) equilibrium distribution along physically meaningful degrees of freedom?

### (a) Pairwise Structural Distributions

- Radial distribution function (RDF, g(r))
- Bond, angle, and dihedral distributions (if defined)

Strengths:
- Classical and well-established
- Robust and interpretable

Limitations:
- Restricted to local, pairwise correlations

### (b) Physics-informed Collective Variable (CV) Distributions

Examples:
- Radius of gyration
- End-to-end distance
- Contact number or contact maps
- Order parameters (e.g., folding fraction)

This approach compares CG and AA distributions along physically meaningful collective variables.

### (c) Relative Entropy / KL Divergence (Optional)

- Theoretical gold standard for distribution comparison
- Typically approximated in low-dimensional CV spaces
- Requires AA reference data

---

## 2. Dynamics-based Evaluation

### Core Question

Does the CG model preserve the slow dynamical processes and metastable behavior?

### (a) Autocorrelation Times

- Computed for selected CVs (e.g., Rg(t), contact(t))
- Lightweight and broadly applicable

### (b) Implied Timescales / MSM-based Metrics

- Relaxation timescales t₂, t₃, ...
- Chapman–Kolmogorov (CK) test (optional)

Standard indicators of kinetic fidelity.

### (c) State Populations (Optional)

- Compare equilibrium populations of metastable states
- Can be defined using AA reference or CG-based discretization

---

## 3. Sampling Efficiency

### Core Question

Given a fixed computational budget, how efficiently does the CG model explore configurational space?

### (a) Effective Sample Size (ESS)

- Estimated from autocorrelation times
- AA-independent
- Measures statistical efficiency

### (b) Autocorrelation Decay Rate

- Evaluated across multiple CVs
- Faster decay implies more efficient exploration

### (c) Basin Discovery / Transition Counts

- Number of distinct basins visited
- Transition frequency between coarse states

### (d) Low-dimensional Free Energy Surface Coverage

- Projection onto 1–2 key CVs
- Comparison of explored area or configurational entropy

---

## Summary Table

| Category | Metric | Priority | AA Reference Needed? | Computational Cost |
|-----------|--------|----------|----------------------|-------------------|
| Distribution | RDF (g(r)) | High | Optional (for comparison) | Low |
|  | CV distributions | Essential | Optional | Low–Medium |
|  | Relative entropy | Low (advanced) | Yes | Medium |
| Dynamics | Autocorrelation times | Essential | Optional | Low |
|  | Implied timescales | High | Optional | Medium–High |
|  | CK test | Medium (optional) | Optional | Medium |
|  | State populations | Low (optional) | Optional | Low |
| Sampling Efficiency | Effective sample size | High | No | Low |
|  | Autocorrelation decay | Essential | No | Low |
|  | Basin discovery | High | No | Low–Medium |
|  | FES coverage | Medium | No | Medium |
