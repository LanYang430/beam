"""
AA Analysis: Stage 3 - All-Atom Trajectory Analysis and Comparison

This module handles:
1. Loading and preprocessing AA trajectories
2. Transforming AA data with CG-learned CVs
3. Training new TICA on AA data
4. Comparing CG vs AA CVs (to be enhanced)
5. Generating REUS window suggestions

Future enhancements:
- Quantitative CG/AA CV comparison metrics
- Cross-scale consistency analysis
- PMF calculation integration
"""

import numpy as np
import pickle
from pathlib import Path

try:
    import mdtraj as md
    MDTRAJ_AVAILABLE = True
except ImportError:
    MDTRAJ_AVAILABLE = False
    print("Warning: mdtraj not available")

try:
    from deeptime.decomposition import TICA
    DEEPTIME_AVAILABLE = True
except ImportError:
    DEEPTIME_AVAILABLE = False
    print("Warning: deeptime not available")


def load_and_preprocess_aa(
    dcd_path,
    topology_pdb,
    reference_pdb,
    align_selection="protein and backbone and resid 68 to 92",
    feature_selection="name CA or name N or name C"
):
    """
    Load and preprocess AA trajectory.
    
    Uses the same preprocessing pipeline as CG trajectories:
    1. Load from DCD
    2. Align to reference
    3. Extract backbone atoms
    4. Flatten to feature matrix
    
    Note: For direct CG/AA comparison, both should use backbone atoms only.
    
    Parameters
    ----------
    dcd_path : str
        Path to AA trajectory DCD file
    topology_pdb : str
        PDB file for trajectory topology
    reference_pdb : str
        Reference structure for alignment
    align_selection : str
        MDTraj selection string for alignment atoms
    feature_selection : str
        MDTraj selection string for feature extraction
        
    Returns
    -------
    features : np.ndarray
        Feature matrix of shape (n_frames, n_features)
    """
    if not MDTRAJ_AVAILABLE:
        raise ImportError("mdtraj is required")
    
    print(f"Loading AA trajectory: {dcd_path}")
    traj = md.load(dcd_path, top=topology_pdb)
    print(f"  Loaded {traj.n_frames} frames, {traj.n_atoms} atoms")
    
    # Load reference
    reference = md.load(reference_pdb)
    
    # Align
    print(f"Aligning on: {align_selection}")
    atom_indices = traj.topology.select(align_selection)
    traj_aligned = traj.superpose(reference, atom_indices=atom_indices,
                                  ref_atom_indices=atom_indices)
    
    # Extract features (backbone atoms)
    print(f"Extracting features: {feature_selection}")
    feature_atoms = traj_aligned.topology.select(feature_selection)
    traj_features = traj_aligned.atom_slice(feature_atoms)
    
    # Flatten
    xyz = traj_features.xyz
    n_frames, n_atoms, _ = xyz.shape
    features = xyz.reshape(n_frames, n_atoms * 3)
    
    print(f"Feature matrix shape: {features.shape}")
    
    return features


def transform_aa_with_cg_tica(aa_features, cg_tica_pkl_path):
    """
    Transform AA trajectory using CG-trained TICA model.
    
    This shows how well CG-learned CVs capture AA dynamics.
    Used for:
    - Visualizing AA sampling on CG CV space
    - Preparing AA data for enhanced sampling (REAP interface)
    
    Parameters
    ----------
    aa_features : np.ndarray
        AA feature matrix from load_and_preprocess_aa()
    cg_tica_pkl_path : str
        Path to CG-trained TICA model
        
    Returns
    -------
    aa_cv : np.ndarray
        AA trajectory projected onto CG-learned CV space
    """
    print(f"\nTransforming AA with CG TICA model...")
    
    # Load CG model
    with open(cg_tica_pkl_path, 'rb') as f:
        cg_tica_model = pickle.load(f)
    print(f"  Loaded CG TICA from: {cg_tica_pkl_path}")
    
    # Transform AA data
    aa_cv = cg_tica_model.transform(aa_features)
    print(f"  AA in CG CV space: {aa_cv.shape}")
    
    return aa_cv


def train_aa_tica(aa_features, lagtime=50, dim=2, save_path=None):
    """
    Train TICA model on AA trajectory data.
    
    This identifies slow modes specific to AA-level dynamics,
    which may differ from CG-learned CVs due to:
    - More detailed interactions (side chains, electrostatics)
    - Different energy landscape roughness
    - AA-specific metastable states
    
    The AA-learned CVs are typically used for:
    - PMF calculation (more accurate at AA level)
    - Comparison with CG CVs (identify CG artifacts)
    
    Parameters
    ----------
    aa_features : np.ndarray
        AA feature matrix
    lagtime : int
        Lag time for TICA
    dim : int
        Number of components
    save_path : str, optional
        Path to save trained model
        
    Returns
    -------
    tica_model : TICA
        Trained AA TICA model
    aa_cv : np.ndarray
        AA trajectory in AA CV space
    """
    if not DEEPTIME_AVAILABLE:
        raise ImportError("deeptime is required")
    
    print(f"\nTraining TICA on AA trajectory:")
    print(f"  Lag time: {lagtime} frames")
    print(f"  Dimensions: {dim}")
    
    # Train TICA
    tica = TICA(lagtime=lagtime, dim=dim)
    tica_model = tica.fit(aa_features)
    
    # Transform
    aa_cv = tica_model.transform(aa_features)
    
    print(f"  Training complete!")
    print(f"  Output shape: {aa_cv.shape}")
    print(f"  Eigenvalues (first {min(5, dim)}): {tica_model.eigenvalues[:min(5, dim)]}")
    
    # Save if requested
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(tica_model, f)
        print(f"  Model saved to: {save_path}")
    
    return tica_model, aa_cv


def compare_cg_aa_cvs(cg_tica_model, aa_tica_model, method='basic'):
    """
    Compare CG-learned vs AA-learned collective variables.
    
    **TO BE FULLY IMPLEMENTED**
    
    This will include:
    - Eigenvalue/timescale comparison
    - Eigenvector similarity (cosine similarity, overlap)
    - Residue-level weight correlation
    - CV space overlap metrics
    - Identification of CG artifacts vs AA-specific features
    
    For now, provides basic comparison and placeholder for future work.
    
    Parameters
    ----------
    cg_tica_model : TICA
        TICA model trained on CG data
    aa_tica_model : TICA
        TICA model trained on AA data
    method : str
        Comparison method (currently only 'basic' available)
        
    Returns
    -------
    comparison : dict
        Comparison metrics and notes
    """
    print("\n" + "="*60)
    print("CG vs AA CV Comparison")
    print("="*60)
    
    # Basic comparison: eigenvalues
    cg_eigenvalues = cg_tica_model.eigenvalues
    aa_eigenvalues = aa_tica_model.eigenvalues
    
    n_compare = min(len(cg_eigenvalues), len(aa_eigenvalues), 5)
    
    print("\nEigenvalue comparison (first {}):".format(n_compare))
    print("  Component  |  CG Value  |  AA Value  |  Ratio (AA/CG)")
    print("  " + "-"*56)
    for i in range(n_compare):
        ratio = aa_eigenvalues[i] / cg_eigenvalues[i] if cg_eigenvalues[i] > 0 else 0
        print(f"      {i+1}      |  {cg_eigenvalues[i]:.4f}   |  {aa_eigenvalues[i]:.4f}   |    {ratio:.4f}")
    
    comparison = {
        'cg_eigenvalues': cg_eigenvalues.tolist(),
        'aa_eigenvalues': aa_eigenvalues.tolist(),
        'status': 'basic_comparison_only',
        'note': 'Detailed comparison metrics to be implemented...',
        'future_features': [
            'Eigenvector similarity analysis',
            'Residue-level contribution correlation',
            'CV space overlap quantification',
            'CG artifact identification',
            'Statistical significance testing'
        ]
    }
    
    print("\n" + "="*60)
    print("Note: Detailed comparison to be implemented...")
    print("="*60)
    
    return comparison


def generate_reus_windows(aa_cv, n_windows=20, buffer=0.1, cv_dim=0):
    """
    Generate suggested REUS window centers based on AA CV range.
    
    This provides a starting point for REUS setup. Users should adjust:
    - n_windows: based on desired overlap and system characteristics
    - force_constant: based on barrier heights and system stiffness
    - exchange frequency: based on temperature and barrier crossing times
    
    For 2D REUS, call this function separately for each dimension.
    
    Parameters
    ----------
    aa_cv : np.ndarray
        AA trajectory in CV space, shape (n_frames, n_cv)
    n_windows : int
        Number of umbrella windows
    buffer : float
        Fraction to extend beyond min/max CV values (e.g., 0.1 = 10% extension)
    cv_dim : int
        Which CV dimension to use (0 for first, 1 for second, etc.)
        
    Returns
    -------
    window_info : dict
        Window centers and setup information
        
    Examples
    --------
    >>> # 1D REUS on first CV
    >>> windows_cv1 = generate_reus_windows(aa_cv, n_windows=20, cv_dim=0)
    >>> 
    >>> # 2D REUS on CV1 and CV2
    >>> windows_cv1 = generate_reus_windows(aa_cv, n_windows=20, cv_dim=0)
    >>> windows_cv2 = generate_reus_windows(aa_cv, n_windows=20, cv_dim=1)
    """
    if aa_cv.ndim == 1:
        cv_values = aa_cv
    else:
        cv_values = aa_cv[:, cv_dim]
    
    cv_min = np.min(cv_values)
    cv_max = np.max(cv_values)
    
    # Add buffer
    range_span = cv_max - cv_min
    cv_min_buffered = cv_min - buffer * range_span
    cv_max_buffered = cv_max + buffer * range_span
    
    # Generate window centers
    window_centers = np.linspace(cv_min_buffered, cv_max_buffered, n_windows)
    
    print(f"\nREUS Window Suggestion (CV dimension {cv_dim}):")
    print(f"  CV range (observed): [{cv_min:.3f}, {cv_max:.3f}]")
    print(f"  CV range (buffered): [{cv_min_buffered:.3f}, {cv_max_buffered:.3f}]")
    print(f"  Number of windows: {n_windows}")
    print(f"  Window spacing: {(cv_max_buffered - cv_min_buffered) / (n_windows - 1):.3f}")
    print(f"\nFirst 5 window centers: {window_centers[:5]}")
    print(f"Last 5 window centers: {window_centers[-5:]}")
    
    return {
        'cv_dimension': cv_dim,
        'cv_range_observed': (float(cv_min), float(cv_max)),
        'cv_range_buffered': (float(cv_min_buffered), float(cv_max_buffered)),
        'n_windows': n_windows,
        'window_centers': window_centers.tolist(),
        'window_spacing': float((cv_max_buffered - cv_min_buffered) / (n_windows - 1)),
        'note': 'These are suggested centers. Adjust force constants, exchange rates, '
                'and production times based on your system characteristics.'
    }


def run_full_aa_analysis(
    aa_dcd_path,
    topology_pdb,
    reference_pdb,
    cg_tica_pkl_path,
    output_dir=".",
    train_aa_tica_model=True,
    lagtime=50,
    dim=2,
    align_selection="protein and backbone and resid 68 to 92",
    feature_selection="name CA or name N or name C"
):
    """
    Run complete AA analysis pipeline.
    
    This is the main entry point for Stage 3.
    
    Steps:
    1. Load and preprocess AA trajectory
    2. Transform with CG TICA (for visualization and comparison)
    3. (Optional) Train new TICA on AA data
    4. (Optional) Compare CG vs AA CVs
    5. Generate REUS window suggestions
    
    Parameters
    ----------
    aa_dcd_path : str
        Path to AA trajectory
    topology_pdb : str
        Topology PDB
    reference_pdb : str
        Reference structure
    cg_tica_pkl_path : str
        Path to CG TICA model
    output_dir : str
        Output directory
    train_aa_tica_model : bool
        Whether to train new TICA on AA data
    lagtime : int
        Lag time for AA TICA (if training)
    dim : int
        Dimensions for AA TICA (if training)
    align_selection : str
        Alignment selection
    feature_selection : str
        Feature selection
        
    Returns
    -------
    results : dict
        Analysis results and output paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("\n" + "="*70)
    print("BEAM AA Analysis - Stage 3")
    print("="*70)
    
    # Step 1: Load AA trajectory
    print("\n[Stage 3.1] Loading and preprocessing AA trajectory...")
    aa_features = load_and_preprocess_aa(
        aa_dcd_path, topology_pdb, reference_pdb,
        align_selection, feature_selection
    )
    
    # Step 2: Transform with CG TICA
    print("\n[Stage 3.2] Transforming AA with CG-learned CVs...")
    aa_in_cg_cv = transform_aa_with_cg_tica(aa_features, cg_tica_pkl_path)
    
    # Save AA in CG CV space
    aa_cg_cv_path = output_dir / "aa_in_cg_cv_space.npy"
    np.save(aa_cg_cv_path, aa_in_cg_cv)
    print(f"  Saved AA in CG CV space: {aa_cg_cv_path}")
    
    results = {
        'aa_features': aa_features,
        'aa_in_cg_cv': aa_in_cg_cv,
        'paths': {
            'aa_in_cg_cv': str(aa_cg_cv_path)
        }
    }
    
    # Step 3: Train AA TICA (optional)
    if train_aa_tica_model:
        print("\n[Stage 3.3] Training TICA on AA data...")
        aa_tica_pkl_path = output_dir / "aa_tica_model.pkl"
        aa_tica_model, aa_in_aa_cv = train_aa_tica(
            aa_features,
            lagtime=lagtime,
            dim=dim,
            save_path=str(aa_tica_pkl_path)
        )
        
        # Save AA in AA CV space
        aa_aa_cv_path = output_dir / "aa_in_aa_cv_space.npy"
        np.save(aa_aa_cv_path, aa_in_aa_cv)
        print(f"  Saved AA in AA CV space: {aa_aa_cv_path}")
        
        results['aa_tica_model'] = aa_tica_model
        results['aa_in_aa_cv'] = aa_in_aa_cv
        results['paths']['aa_tica_model'] = str(aa_tica_pkl_path)
        results['paths']['aa_in_aa_cv'] = str(aa_aa_cv_path)
        
        # Step 4: Compare (if we have both models)
        print("\n[Stage 3.4] Comparing CG vs AA CVs...")
        with open(cg_tica_pkl_path, 'rb') as f:
            cg_tica_model = pickle.load(f)
        
        comparison = compare_cg_aa_cvs(cg_tica_model, aa_tica_model)
        results['comparison'] = comparison
    
    # Step 5: Generate REUS windows
    print("\n[Stage 3.5] Generating REUS window suggestions...")
    # Use AA in AA CV space if available, otherwise use AA in CG CV space
    cv_for_reus = results.get('aa_in_aa_cv', aa_in_cg_cv)
    
    reus_windows_cv1 = generate_reus_windows(cv_for_reus, n_windows=20, cv_dim=0)
    results['reus_windows_cv1'] = reus_windows_cv1
    
    if cv_for_reus.shape[1] > 1:
        reus_windows_cv2 = generate_reus_windows(cv_for_reus, n_windows=20, cv_dim=1)
        results['reus_windows_cv2'] = reus_windows_cv2
    
    print("\n" + "="*70)
    print("AA Analysis Complete!")
    print("="*70)
    print("\nKey outputs:")
    for key, path in results['paths'].items():
        print(f"  {key}: {path}")
    
    return results
