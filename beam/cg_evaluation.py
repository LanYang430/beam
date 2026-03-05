"""
CG Model Evaluation - Stage 1

Complete evaluation framework for CG models:
1. Distribution-based (RDF + CV distributions)
2. Dynamics-based (autocorrelation, timescales)
3. Sampling efficiency (ESS, basin discovery, coverage)

This is the definitive quality assessment for BEAM CG models.
"""

import numpy as np
from pathlib import Path
import pickle

from .evaluation_utils import (
    # Distribution
    compute_histogram,
    kl_divergence,
    js_divergence,
    overlap_coefficient,
    ks_test,
    l2_error,
    # RDF
    compute_rdf,
    # Dynamics
    compute_acf,
    integrate_acf,
    compute_ess,
    # Clustering
    kmeans_cluster,
    coverage_ratio,
    compute_cv_variance,
    # FES
    compute_fes_1d,
    compute_fes_coverage
)


class CGEvaluator:
    """
    Comprehensive CG model evaluation framework.
    
    This class defines the quality assessment standard for BEAM.
    
    Parameters
    ----------
    cg_traj : np.ndarray or str
        CG trajectory features, shape (n_frames, n_features)
    aa_traj : np.ndarray or str, optional
        AA reference trajectory
    cg_positions : np.ndarray, optional
        CG positions for RDF, shape (n_frames, n_atoms, 3)
    aa_positions : np.ndarray, optional
        AA positions for RDF (must be mapped to CG resolution)
        
    Examples
    --------
    >>> # Full evaluation with AA reference
    >>> evaluator = CGEvaluator(
    ...     cg_traj=cg_features,
    ...     aa_traj=aa_features,
    ...     cg_positions=cg_pos,
    ...     aa_positions=aa_pos_mapped
    ... )
    >>> 
    >>> # Define CVs
    >>> cv_list = [('Rg', rg_func), ('end_to_end', e2e_func)]
    >>> 
    >>> # Run complete evaluation
    >>> report = evaluator.evaluate_all(cv_list)
    """
    
    def __init__(self, cg_traj, aa_traj=None, 
                 cg_positions=None, aa_positions=None):
        # Load trajectories (features)
        self.cg = self._load_array(cg_traj)
        self.aa = self._load_array(aa_traj) if aa_traj is not None else None
        
        # Load positions (for RDF)
        self.cg_pos = self._load_array(cg_positions) if cg_positions is not None else None
        self.aa_pos = self._load_array(aa_positions) if aa_positions is not None else None
        
        self._validate()
        
        print(f"CGEvaluator initialized:")
        print(f"  CG features: {self.cg.shape}")
        if self.aa is not None:
            print(f"  AA features: {self.aa.shape}")
        if self.cg_pos is not None:
            print(f"  CG positions: {self.cg_pos.shape}")
        if self.aa_pos is not None:
            print(f"  AA positions (mapped): {self.aa_pos.shape}")
    
    def _load_array(self, data):
        """Load from array or file"""
        if data is None:
            return None
        if isinstance(data, str):
            return np.load(data)
        elif isinstance(data, np.ndarray):
            return data
        else:
            raise TypeError(f"Data must be np.ndarray or str, got {type(data)}")
    
    def _validate(self):
        """Validate shapes"""
        # Features
        if self.cg.ndim != 2:
            raise ValueError(f"CG features must be 2D, got {self.cg.shape}")
        
        if self.aa is not None:
            if self.aa.ndim != 2:
                raise ValueError(f"AA features must be 2D")
            if self.aa.shape[1] != self.cg.shape[1]:
                raise ValueError(
                    f"CG and AA must have same n_features. "
                    f"CG: {self.cg.shape[1]}, AA: {self.aa.shape[1]}"
                )
        
        # Positions
        if self.cg_pos is not None:
            if self.cg_pos.ndim != 3:
                raise ValueError("CG positions must be 3D (n_frames, n_atoms, 3)")
        
        if self.aa_pos is not None:
            if self.aa_pos.ndim != 3:
                raise ValueError("AA positions must be 3D")
            if self.cg_pos is not None:
                if self.aa_pos.shape[1] != self.cg_pos.shape[1]:
                    raise ValueError(
                        "AA positions must be mapped to CG resolution. "
                        f"CG n_atoms: {self.cg_pos.shape[1]}, "
                        f"AA n_atoms: {self.aa_pos.shape[1]}"
                    )
    
    # ========== DISTRIBUTION-BASED: RDF ==========
    
    def compare_rdf(self, r_max=1.5, bins=100, box_size=None):
        """
        Compare radial distribution functions g(r).
        
        **Requires CG and AA positions.**
        
        Parameters
        ----------
        r_max : float
            Maximum distance for g(r) in nm
        bins : int
            Number of bins
        box_size : float, optional
            Box size for PBC
            
        Returns
        -------
        results : dict
            - g_cg : np.ndarray, CG g(r)
            - g_aa : np.ndarray, AA g(r)
            - r_bins : np.ndarray, bin centers
            - l2_error : float
            - kl_divergence : float
            - js_divergence : float
            
        Raises
        ------
        ValueError
            If positions not provided
        """
        if self.cg_pos is None or self.aa_pos is None:
            raise ValueError("compare_rdf requires both cg_positions and aa_positions")
        
        print("  Computing RDF for CG...")
        g_cg, r_bins = compute_rdf(self.cg_pos, r_max=r_max, bins=bins, box_size=box_size)
        
        print("  Computing RDF for AA...")
        g_aa, _ = compute_rdf(self.aa_pos, r_max=r_max, bins=bins, box_size=box_size)
        
        # Compute comparison metrics
        l2 = l2_error(g_cg, g_aa)
        
        # For KL and JS, treat g(r) as distributions (normalize)
        g_cg_norm = g_cg / g_cg.sum()
        g_aa_norm = g_aa / g_aa.sum()
        
        kl = kl_divergence(g_aa_norm, g_cg_norm)  # D_KL(AA || CG)
        js = js_divergence(g_cg_norm, g_aa_norm)
        
        return {
            'g_cg': g_cg,
            'g_aa': g_aa,
            'r_bins': r_bins,
            'l2_error': l2,
            'kl_divergence': kl,
            'js_divergence': js
        }
    
    # ========== DISTRIBUTION-BASED: CV DISTRIBUTIONS ==========
    
    def compare_cv_distribution(self, cv_func, bins=50):
        """
        Compare CG vs AA distribution of a collective variable.
        
        Computes multiple comparison metrics:
        - KS test
        - KL divergence
        - JS divergence
        - Overlap coefficient
        
        Parameters
        ----------
        cv_func : callable
            Function: cv_func(features) -> np.ndarray (n_frames,)
        bins : int
            Number of histogram bins
            
        Returns
        -------
        results : dict
            - cg_cv : np.ndarray
            - cg_hist : np.ndarray
            - bin_edges : np.ndarray
            - aa_cv : np.ndarray (if AA available)
            - aa_hist : np.ndarray (if AA available)
            - ks_statistic : float (if AA available)
            - ks_pvalue : float (if AA available)
            - kl_divergence : float (if AA available)
            - js_divergence : float (if AA available)
            - overlap_coefficient : float (if AA available)
        """
        # Compute CV for CG
        cg_cv = cv_func(self.cg)
        
        if cg_cv.ndim != 1:
            raise ValueError(f"cv_func must return 1D array, got shape {cg_cv.shape}")
        
        results = {'cg_cv': cg_cv}
        
        if self.aa is None:
            # No AA reference
            cg_hist, bin_edges = compute_histogram(cg_cv, bins=bins)
            results.update({
                'cg_hist': cg_hist,
                'bin_edges': bin_edges
            })
            return results
        
        # With AA reference
        aa_cv = cv_func(self.aa)
        
        if aa_cv.ndim != 1:
            raise ValueError(f"cv_func must return 1D array, got shape {aa_cv.shape}")
        
        # Common range
        cv_min = min(cg_cv.min(), aa_cv.min())
        cv_max = max(cg_cv.max(), aa_cv.max())
        cv_range = (cv_min, cv_max)
        
        # Histograms
        cg_hist, bin_edges = compute_histogram(cg_cv, bins=bins, range=cv_range)
        aa_hist, _ = compute_histogram(aa_cv, bins=bins, range=cv_range)
        
        # Comparison metrics
        ks_stat, ks_pval = ks_test(cg_cv, aa_cv)
        kl = kl_divergence(aa_hist, cg_hist)  # D_KL(AA || CG)
        js = js_divergence(cg_hist, aa_hist)
        overlap = overlap_coefficient(cg_hist, aa_hist)
        
        results.update({
            'aa_cv': aa_cv,
            'cg_hist': cg_hist,
            'aa_hist': aa_hist,
            'bin_edges': bin_edges,
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_pval,
            'kl_divergence': kl,
            'js_divergence': js,
            'overlap_coefficient': overlap
        })
        
        return results
    
    def evaluate_distribution(self, cv_list, rdf_params=None):
        """
        Evaluate all distribution-based metrics.
        
        Computes:
        1. RDF comparison (if positions provided)
        2. CV distribution comparisons for each CV
        
        Parameters
        ----------
        cv_list : list of (name, func) tuples
            CV definitions
        rdf_params : dict, optional
            Parameters for RDF computation
            
        Returns
        -------
        results : dict
            {
                'rdf': {...},  # if positions available
                'cv_name1': {...},
                'cv_name2': {...},
                ...
            }
        """
        print("\n" + "="*70)
        print("DISTRIBUTION-BASED EVALUATION")
        print("="*70)
        
        results = {}
        
        # RDF
        if self.cg_pos is not None and self.aa_pos is not None:
            print("\n[1] RDF comparison")
            rdf_params = rdf_params or {}
            results['rdf'] = self.compare_rdf(**rdf_params)
            print(f"    L2 error: {results['rdf']['l2_error']:.4f}")
            print(f"    JS divergence: {results['rdf']['js_divergence']:.4f}")
        else:
            print("\n[1] RDF comparison: SKIPPED (positions not provided)")
        
        # CV distributions
        print(f"\n[2] CV distributions ({len(cv_list)} CVs)")
        for cv_name, cv_func in cv_list:
            print(f"\n  Evaluating: {cv_name}")
            results[cv_name] = self.compare_cv_distribution(cv_func)
            
            if self.aa is not None:
                print(f"    KS statistic: {results[cv_name]['ks_statistic']:.4f}")
                print(f"    JS divergence: {results[cv_name]['js_divergence']:.4f}")
                print(f"    Overlap: {results[cv_name]['overlap_coefficient']:.4f}")
        
        return results
    
    # ========== DYNAMICS-BASED ==========
    
    def autocorrelation_time(self, cv_func, max_lag=None):
        """
        Compute integrated autocorrelation time.
        
        τ = 1 + 2 Σ ρ(k)
        
        Parameters
        ----------
        cv_func : callable
            CV function
        max_lag : int, optional
            Maximum lag
            
        Returns
        -------
        results : dict
            - tau_cg : float
            - acf_cg : np.ndarray
            - tau_aa : float (if AA available)
            - acf_aa : np.ndarray (if AA available)
            - speedup : float (if AA available), tau_aa / tau_cg
        """
        # CG
        cg_cv = cv_func(self.cg)
        acf_cg = compute_acf(cg_cv, max_lag=max_lag)
        tau_cg = integrate_acf(acf_cg)
        
        results = {
            'tau_cg': tau_cg,
            'acf_cg': acf_cg
        }
        
        if self.aa is not None:
            aa_cv = cv_func(self.aa)
            acf_aa = compute_acf(aa_cv, max_lag=max_lag)
            tau_aa = integrate_acf(acf_aa)
            
            results.update({
                'tau_aa': tau_aa,
                'acf_aa': acf_aa,
                'speedup': tau_aa / tau_cg
            })
        
        return results
    
    def evaluate_dynamics(self, cv_list):
        """
        Evaluate all dynamics-based metrics.
        
        Computes autocorrelation times for each CV.
        
        Parameters
        ----------
        cv_list : list of (name, func) tuples
            
        Returns
        -------
        results : dict
            {cv_name: {...}, ...}
        """
        print("\n" + "="*70)
        print("DYNAMICS-BASED EVALUATION")
        print("="*70)
        
        results = {}
        
        for cv_name, cv_func in cv_list:
            print(f"\n  Evaluating: {cv_name}")
            results[cv_name] = self.autocorrelation_time(cv_func)
            
            print(f"    tau_cg: {results[cv_name]['tau_cg']:.2f}")
            if self.aa is not None:
                print(f"    tau_aa: {results[cv_name]['tau_aa']:.2f}")
                print(f"    Speedup: {results[cv_name]['speedup']:.2f}x")
        
        return results
    
    # ========== SAMPLING EFFICIENCY ==========
    
    def effective_sample_size(self, cv_func):
        """
        Compute effective sample size.
        
        ESS = N / (2τ - 1)
        
        Parameters
        ----------
        cv_func : callable
            CV function
            
        Returns
        -------
        results : dict
            - n_frames_cg : int
            - tau_cg : float
            - ess_cg : float
            - efficiency_cg : float
            - (same for AA if available)
        """
        tau_results = self.autocorrelation_time(cv_func)
        
        # CG
        n_cg = len(self.cg)
        tau_cg = tau_results['tau_cg']
        ess_cg = compute_ess(n_cg, tau_cg)
        
        results = {
            'n_frames_cg': n_cg,
            'tau_cg': tau_cg,
            'ess_cg': ess_cg,
            'efficiency_cg': ess_cg / n_cg
        }
        
        if self.aa is not None:
            n_aa = len(self.aa)
            tau_aa = tau_results['tau_aa']
            ess_aa = compute_ess(n_aa, tau_aa)
            
            results.update({
                'n_frames_aa': n_aa,
                'tau_aa': tau_aa,
                'ess_aa': ess_aa,
                'efficiency_aa': ess_aa / n_aa
            })
        
        return results
    
    def basin_discovery(self, cv_func, n_clusters=10):
        """
        Basin discovery via k-means clustering.
        
        Measures how many distinct states are explored.
        
        Parameters
        ----------
        cv_func : callable
            CV function (can return 1D or 2D)
        n_clusters : int
            Number of clusters to define
            
        Returns
        -------
        results : dict
            - n_clusters_total : int
            - n_clusters_visited_cg : int
            - coverage_cg : float
            - labels_cg : np.ndarray
            - (same for AA if available)
        """
        # Get CV values
        cg_cv = cv_func(self.cg)
        
        # Reshape to 2D if needed
        if cg_cv.ndim == 1:
            cg_cv = cg_cv.reshape(-1, 1)
        
        # Cluster
        labels_cg, centers = kmeans_cluster(cg_cv, n_clusters=n_clusters)
        coverage_cg, n_visited_cg, n_total = coverage_ratio(labels_cg)
        
        results = {
            'n_clusters_total': n_total,
            'n_clusters_visited_cg': n_visited_cg,
            'coverage_cg': coverage_cg,
            'labels_cg': labels_cg,
            'centers': centers
        }
        
        if self.aa is not None:
            aa_cv = cv_func(self.aa)
            if aa_cv.ndim == 1:
                aa_cv = aa_cv.reshape(-1, 1)
            
            labels_aa, _ = kmeans_cluster(aa_cv, n_clusters=n_clusters)
            coverage_aa, n_visited_aa, _ = coverage_ratio(labels_aa)
            
            results.update({
                'n_clusters_visited_aa': n_visited_aa,
                'coverage_aa': coverage_aa,
                'labels_aa': labels_aa
            })
        
        return results
    
    def cv_variance(self, cv_func):
        """
        Compute variance of CV (measure of explored space).
        
        Parameters
        ----------
        cv_func : callable
            CV function
            
        Returns
        -------
        results : dict
            - variance_cg : float
            - variance_aa : float (if available)
        """
        cg_cv = cv_func(self.cg)
        var_cg = compute_cv_variance(cg_cv)
        
        results = {'variance_cg': var_cg}
        
        if self.aa is not None:
            aa_cv = cv_func(self.aa)
            var_aa = compute_cv_variance(aa_cv)
            results['variance_aa'] = var_aa
        
        return results
    
    def evaluate_sampling(self, cv_list, n_clusters=10):
        """
        Evaluate all sampling efficiency metrics.
        
        Computes for each CV:
        1. Effective sample size
        2. Basin discovery
        3. CV variance
        
        Parameters
        ----------
        cv_list : list of (name, func) tuples
        n_clusters : int
            Number of clusters for basin discovery
            
        Returns
        -------
        results : dict
            {cv_name: {'ess': {...}, 'basin': {...}, 'variance': {...}}, ...}
        """
        print("\n" + "="*70)
        print("SAMPLING EFFICIENCY EVALUATION")
        print("="*70)
        
        results = {}
        
        for cv_name, cv_func in cv_list:
            print(f"\n  Evaluating: {cv_name}")
            
            # ESS
            ess_results = self.effective_sample_size(cv_func)
            print(f"    ESS_cg: {ess_results['ess_cg']:.1f} ({ess_results['efficiency_cg']:.1%})")
            
            # Basin discovery
            basin_results = self.basin_discovery(cv_func, n_clusters=n_clusters)
            print(f"    Basins explored: {basin_results['n_clusters_visited_cg']}/{basin_results['n_clusters_total']}")
            
            # Variance
            var_results = self.cv_variance(cv_func)
            print(f"    CV variance: {var_results['variance_cg']:.4f}")
            
            results[cv_name] = {
                'ess': ess_results,
                'basin': basin_results,
                'variance': var_results
            }
        
        return results
    
    # ========== MAIN API ==========
    
    def evaluate_all(self, cv_list, rdf_params=None, n_clusters=10, 
                     output_dir='.', save_results=True):
        """
        Run complete evaluation pipeline.
        
        This is the definitive quality assessment for BEAM CG models.
        
        Parameters
        ----------
        cv_list : list of (name, func) tuples
            Collective variables to evaluate
        rdf_params : dict, optional
            Parameters for RDF (r_max, bins, box_size)
        n_clusters : int
            Number of clusters for basin discovery
        output_dir : str
            Output directory
        save_results : bool
            Whether to save results
            
        Returns
        -------
        report : dict
            Complete evaluation report:
            {
                'distribution': {...},
                'dynamics': {...},
                'sampling': {...}
            }
        """
        print("\n" + "="*70)
        print("BEAM CG MODEL EVALUATION - STAGE 1")
        print("="*70)
        print(f"\nEvaluating {len(cv_list)} collective variables")
        if self.aa is not None:
            print("Mode: COMPARISON (CG vs AA)")
        else:
            print("Mode: CG ONLY")
        
        # Run evaluations
        report = {}
        
        report['distribution'] = self.evaluate_distribution(cv_list, rdf_params=rdf_params)
        report['dynamics'] = self.evaluate_dynamics(cv_list)
        report['sampling'] = self.evaluate_sampling(cv_list, n_clusters=n_clusters)
        
        print("\n" + "="*70)
        print("EVALUATION COMPLETE")
        print("="*70)
        
        # Save
        if save_results:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)
            
            output_path = output_dir / 'evaluation_results.pkl'
            with open(output_path, 'wb') as f:
                pickle.dump(report, f)
            
            print(f"\nResults saved to: {output_path}")
        
        return report
    
    def generate_summary(self, report):
        """
        Generate text summary of evaluation results.
        
        Parameters
        ----------
        report : dict
            Report from evaluate_all()
            
        Returns
        -------
        summary : str
            Human-readable summary
        """
        lines = []
        lines.append("="*70)
        lines.append("BEAM CG EVALUATION SUMMARY")
        lines.append("="*70)
        
        # Distribution
        lines.append("\n[DISTRIBUTION]")
        if 'rdf' in report['distribution']:
            rdf = report['distribution']['rdf']
            lines.append(f"  RDF:")
            lines.append(f"    L2 error: {rdf['l2_error']:.4f}")
            lines.append(f"    JS divergence: {rdf['js_divergence']:.4f}")
        
        for cv_name in report['distribution']:
            if cv_name == 'rdf':
                continue
            dist = report['distribution'][cv_name]
            lines.append(f"  {cv_name}:")
            if 'ks_statistic' in dist:
                lines.append(f"    KS stat: {dist['ks_statistic']:.4f} (p={dist['ks_pvalue']:.4f})")
                lines.append(f"    JS div: {dist['js_divergence']:.4f}")
                lines.append(f"    Overlap: {dist['overlap_coefficient']:.4f}")
        
        # Dynamics
        lines.append("\n[DYNAMICS]")
        for cv_name, dyn in report['dynamics'].items():
            lines.append(f"  {cv_name}:")
            lines.append(f"    tau_cg: {dyn['tau_cg']:.2f}")
            if 'tau_aa' in dyn:
                lines.append(f"    tau_aa: {dyn['tau_aa']:.2f}")
                lines.append(f"    Speedup: {dyn['speedup']:.2f}x")
        
        # Sampling
        lines.append("\n[SAMPLING]")
        for cv_name, samp in report['sampling'].items():
            lines.append(f"  {cv_name}:")
            lines.append(f"    ESS: {samp['ess']['ess_cg']:.1f} ({samp['ess']['efficiency_cg']:.1%})")
            lines.append(f"    Basins: {samp['basin']['n_clusters_visited_cg']}/{samp['basin']['n_clusters_total']}")
            lines.append(f"    Variance: {samp['variance']['variance_cg']:.4f}")
        
        lines.append("\n" + "="*70)
        
        return "\n".join(lines)
