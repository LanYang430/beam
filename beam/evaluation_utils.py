"""
Evaluation Utilities - Pure Mathematical Functions

Contains all statistical, numerical, and mathematical tools
for CG model evaluation. No state, no I/O, no plotting.

Functions are organized by category:
1. Distribution comparison metrics
2. RDF computation
3. Autocorrelation and dynamics
4. Clustering and coverage
"""

import numpy as np


# ==================== DISTRIBUTION COMPARISON ====================

def compute_histogram(data, bins=50, range=None, density=True):
    """
    Compute normalized histogram (probability density).
    
    Parameters
    ----------
    data : np.ndarray
        1D data array
    bins : int or array_like
        Number of bins or bin edges
    range : tuple, optional
        (min, max) range for histogram
    density : bool
        If True, normalize to sum to 1
        
    Returns
    -------
    hist : np.ndarray
        Histogram (normalized if density=True)
    bin_edges : np.ndarray
        Bin edges
    """
    hist, bin_edges = np.histogram(data, bins=bins, range=range, density=False)
    
    if density:
        hist = hist.astype(float)
        if hist.sum() > 0:
            hist = hist / hist.sum()
    
    return hist, bin_edges


def kl_divergence(p, q):
    """
    Kullback-Leibler divergence: D_KL(P || Q).
    
    Measures information lost when Q is used to approximate P.
    
    Parameters
    ----------
    p, q : np.ndarray
        Probability distributions (must sum to 1)
        
    Returns
    -------
    kl : float
        KL divergence in nats (0 = identical, higher = more different)
    """
    epsilon = 1e-10
    p = p + epsilon
    q = q + epsilon
    
    # Normalize
    p = p / p.sum()
    q = q / q.sum()
    
    kl = np.sum(p * np.log(p / q))
    
    return kl


def js_divergence(p, q):
    """
    Jensen-Shannon divergence between two probability distributions.
    
    Symmetric version of KL divergence:
    JSD(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
    where M = 0.5 * (P + Q)
    
    Parameters
    ----------
    p, q : np.ndarray
        Probability distributions
        
    Returns
    -------
    jsd : float
        Jensen-Shannon divergence, range [0, 1]
        0 = identical, 1 = maximally different
    """
    epsilon = 1e-10
    p = p + epsilon
    q = q + epsilon
    
    # Normalize
    p = p / p.sum()
    q = q / q.sum()
    
    # Compute M
    m = 0.5 * (p + q)
    
    # KL divergences
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    
    # JSD (normalized to [0, 1])
    jsd = 0.5 * kl_pm + 0.5 * kl_qm
    jsd = jsd / np.log(2)
    
    return jsd


def overlap_coefficient(p, q):
    """
    Overlap coefficient (Bhattacharyya coefficient).
    
    Measures the amount of overlap between two distributions.
    
    Parameters
    ----------
    p, q : np.ndarray
        Probability distributions
        
    Returns
    -------
    overlap : float
        Overlap coefficient, range [0, 1]
        1 = perfect overlap, 0 = no overlap
    """
    epsilon = 1e-10
    p = p + epsilon
    q = q + epsilon
    
    # Normalize
    p = p / p.sum()
    q = q / q.sum()
    
    overlap = np.sum(np.sqrt(p * q))
    
    return overlap


def ks_test(data1, data2):
    """
    Kolmogorov-Smirnov test for two samples.
    
    Parameters
    ----------
    data1, data2 : np.ndarray
        1D data arrays to compare
        
    Returns
    -------
    statistic : float
        KS statistic (max difference in CDFs)
    pvalue : float
        Two-sided p-value
    """
    from scipy.stats import ks_2samp
    return ks_2samp(data1, data2)


def l2_error(p, q):
    """
    L2 norm (Euclidean distance) between two distributions.
    
    Parameters
    ----------
    p, q : np.ndarray
        Distributions (any normalization)
        
    Returns
    -------
    l2 : float
        L2 distance
    """
    return np.sqrt(np.sum((p - q)**2))


# ==================== RDF COMPUTATION ====================

def compute_rdf(positions, r_max=1.5, bins=100, box_size=None):
    """
    Compute radial distribution function g(r).
    
    For a set of particles, computes the probability of finding
    a particle at distance r from another particle, normalized
    by the ideal gas density.
    
    Parameters
    ----------
    positions : np.ndarray
        Particle positions, shape (n_atoms, 3) for single frame
        or (n_frames, n_atoms, 3) for trajectory
    r_max : float
        Maximum distance for g(r)
    bins : int
        Number of bins
    box_size : float or array_like, optional
        Simulation box size for PBC
        
    Returns
    -------
    g_r : np.ndarray
        Radial distribution function
    r_bins : np.ndarray
        Bin centers
    """
    # Handle single frame or trajectory
    if positions.ndim == 2:
        positions = positions[np.newaxis, ...]  # (1, n_atoms, 3)
    
    n_frames, n_atoms, _ = positions.shape
    
    # Bin setup
    r_bins = np.linspace(0, r_max, bins + 1)
    r_centers = 0.5 * (r_bins[1:] + r_bins[:-1])
    dr = r_bins[1] - r_bins[0]
    
    # Accumulate histogram
    hist = np.zeros(bins)
    
    for frame in range(n_frames):
        pos = positions[frame]
        
        # Compute all pairwise distances
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                vec = pos[i] - pos[j]
                
                # Apply PBC if box_size provided
                if box_size is not None:
                    vec = vec - box_size * np.round(vec / box_size)
                
                r = np.linalg.norm(vec)
                
                if r < r_max:
                    bin_idx = int(r / dr)
                    if bin_idx < bins:
                        hist[bin_idx] += 2  # Count both i-j and j-i
    
    # Normalize by number of frames and particles
    hist = hist / n_frames
    hist = hist / n_atoms
    
    # Normalize by shell volume and ideal gas density
    shell_volumes = 4.0 * np.pi * r_centers**2 * dr
    
    # Estimate density (number density)
    if box_size is not None:
        volume = box_size**3 if np.isscalar(box_size) else np.prod(box_size)
    else:
        # Estimate from positions
        volume = np.prod(positions.max(axis=(0,1)) - positions.min(axis=(0,1)))
    
    rho = n_atoms / volume
    
    # g(r) = histogram / (N * rho * shell_volume)
    g_r = hist / (rho * shell_volumes)
    
    return g_r, r_centers


# ==================== AUTOCORRELATION & DYNAMICS ====================

def compute_acf(time_series, max_lag=None):
    """
    Compute autocorrelation function.
    
    Parameters
    ----------
    time_series : np.ndarray
        1D time series
    max_lag : int, optional
        Maximum lag. If None, uses len(time_series) // 2
        
    Returns
    -------
    acf : np.ndarray
        Autocorrelation function
    """
    if time_series.ndim != 1:
        raise ValueError("time_series must be 1D")
    
    if max_lag is None:
        max_lag = len(time_series) // 2
    
    # Center
    ts = time_series - time_series.mean()
    variance = np.var(time_series)
    
    if variance == 0:
        return np.ones(max_lag)
    
    acf = np.zeros(max_lag)
    n = len(time_series)
    
    for lag in range(max_lag):
        if lag == 0:
            acf[lag] = 1.0
        else:
            acf[lag] = np.mean(ts[:n-lag] * ts[lag:]) / variance
    
    return acf


def integrate_acf(acf):
    """
    Compute integrated autocorrelation time.
    
    τ = 1 + 2 Σ ρ(k)
    
    Integrates until first negative value.
    
    Parameters
    ----------
    acf : np.ndarray
        Autocorrelation function
        
    Returns
    -------
    tau : float
        Integrated autocorrelation time
    """
    # Find first negative crossing
    negative_idx = np.where(acf < 0)[0]
    
    if len(negative_idx) > 0:
        cutoff = negative_idx[0]
    else:
        cutoff = len(acf)
    
    tau = 1.0 + 2.0 * np.sum(acf[1:cutoff])
    
    return max(tau, 1.0)


def compute_ess(n_samples, tau):
    """
    Effective sample size from autocorrelation time.
    
    ESS = N / (2τ - 1)
    
    Parameters
    ----------
    n_samples : int
        Total number of samples
    tau : float
        Integrated autocorrelation time
        
    Returns
    -------
    ess : float
        Effective sample size
    """
    return n_samples / (2.0 * tau - 1.0)


# ==================== CLUSTERING & COVERAGE ====================

def kmeans_cluster(data, n_clusters=10, max_iter=100, random_state=None):
    """
    Simple k-means clustering.
    
    Parameters
    ----------
    data : np.ndarray
        Data to cluster, shape (n_samples, n_features)
    n_clusters : int
        Number of clusters
    max_iter : int
        Maximum iterations
    random_state : int, optional
        Random seed
        
    Returns
    -------
    labels : np.ndarray
        Cluster labels for each sample
    centers : np.ndarray
        Cluster centers
    """
    from sklearn.cluster import KMeans
    
    kmeans = KMeans(
        n_clusters=n_clusters,
        max_iter=max_iter,
        random_state=random_state,
        n_init=10
    )
    
    labels = kmeans.fit_predict(data)
    centers = kmeans.cluster_centers_
    
    return labels, centers


def coverage_ratio(labels):
    """
    Compute fraction of clusters visited.
    
    Parameters
    ----------
    labels : np.ndarray
        Cluster labels
        
    Returns
    -------
    coverage : float
        Fraction of unique clusters (0-1)
    n_visited : int
        Number of unique clusters visited
    n_total : int
        Total number of possible clusters
    """
    n_total = labels.max() + 1  # Assumes labels are 0, 1, ..., K-1
    n_visited = len(np.unique(labels))
    coverage = n_visited / n_total
    
    return coverage, n_visited, n_total


def compute_cv_variance(cv_values):
    """
    Compute variance of CV (measure of space explored).
    
    Parameters
    ----------
    cv_values : np.ndarray
        1D or 2D array of CV values
        
    Returns
    -------
    variance : float or np.ndarray
        Variance (scalar for 1D, array for 2D)
    """
    return np.var(cv_values, axis=0)


# ==================== FREE ENERGY ====================

def compute_fes_1d(cv_values, temperature=300.0, bins=50):
    """
    Compute 1D free energy surface.
    
    F(x) = -kT ln P(x)
    
    Parameters
    ----------
    cv_values : np.ndarray
        1D collective variable values
    temperature : float
        Temperature in Kelvin
    bins : int
        Number of bins
        
    Returns
    -------
    fes : np.ndarray
        Free energy in kT units
    bin_centers : np.ndarray
        Bin centers
    """
    kB = 0.001987204  # kcal/(mol*K)
    kT = kB * temperature
    
    hist, bin_edges = np.histogram(cv_values, bins=bins)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    
    # Probability
    hist = hist.astype(float)
    hist = hist / hist.sum()
    
    # Avoid log(0)
    hist[hist == 0] = np.min(hist[hist > 0]) * 0.01
    
    # Free energy
    fes = -kT * np.log(hist)
    fes = fes - fes.min()  # Shift to zero
    
    return fes, bin_centers


def compute_fes_coverage(fes, threshold_kT=3.0):
    """
    Fraction of FES explored below energy threshold.
    
    Parameters
    ----------
    fes : np.ndarray
        Free energy in kT units
    threshold_kT : float
        Energy threshold
        
    Returns
    -------
    coverage : float
        Fraction of bins below threshold
    """
    accessible = fes <= threshold_kT
    return accessible.sum() / len(fes)
