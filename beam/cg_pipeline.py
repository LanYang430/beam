"""
CG Pipeline: Stage 2 - From CG Trajectories to AA Enhanced Sampling Preparation

This module handles:
1. Loading and preprocessing CG trajectories
2. Training TICA on CG data
3. Preparing interfaces for AA enhanced sampling (REAP, etc.)

Future enhancements:
- Automatic lag time selection using VAMP-2 score, etc.
- Automatic dimension selection using kinetic variance
- Support for multiple ML methods (VAMP, PCA, autoencoders, etc.)
"""

import numpy as np
import pickle
from pathlib import Path

try:
    import mdtraj as md
    MDTRAJ_AVAILABLE = True
except ImportError:
    MDTRAJ_AVAILABLE = False
    print("Warning: mdtraj not available. Install with: conda install -c conda-forge mdtraj")

try:
    from deeptime.decomposition import TICA
    DEEPTIME_AVAILABLE = True
except ImportError:
    DEEPTIME_AVAILABLE = False
    print("Warning: deeptime not available. Install with: pip install deeptime")


def load_and_preprocess_cg(
    dcd_path,
    topology_pdb,
    reference_pdb,
    align_selection="protein and backbone and resid 68 to 92",
    feature_selection="name CA or name N or name C"
):
    """
    Load and preprocess CG trajectory from DCD file.
    
    This function:
    1. Loads trajectory using topology
    2. Aligns to reference structure
    3. Extracts backbone atoms from specified residue range
    4. Flattens to feature matrix for ML
    
    Parameters
    ----------
    dcd_path : str
        Path to CG trajectory DCD file
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
        where n_features = n_backbone_atoms * 3 (xyz coordinates)
    """
    if not MDTRAJ_AVAILABLE:
        raise ImportError("mdtraj is required. Install with: conda install -c conda-forge mdtraj")
    
    print(f"Loading CG trajectory: {dcd_path}")
    traj = md.load(dcd_path, top=topology_pdb)
    print(f"  Loaded {traj.n_frames} frames, {traj.n_atoms} atoms")
    
    # Load reference structure
    reference = md.load(reference_pdb)
    
    # Align trajectory
    print(f"Aligning on: {align_selection}")
    atom_indices = traj.topology.select(align_selection)
    traj_aligned = traj.superpose(reference, atom_indices=atom_indices, 
                                  ref_atom_indices=atom_indices)
    
    # Extract feature atoms
    print(f"Extracting features: {feature_selection}")
    feature_atoms = traj_aligned.topology.select(feature_selection)
    traj_features = traj_aligned.atom_slice(feature_atoms)
    
    # Get xyz coordinates and flatten
    xyz = traj_features.xyz  # shape: (n_frames, n_atoms, 3)
    n_frames, n_atoms, _ = xyz.shape
    features = xyz.reshape(n_frames, n_atoms * 3)
    
    print(f"Feature matrix shape: {features.shape}")
    print(f"  {n_frames} frames, {n_atoms} atoms, {n_atoms * 3} features")
    
    return features


def suggest_tica_params(traj_data, method='tica'):
    """
    Suggest optimal parameters for TICA or other ML methods.
    
    **TO BE IMPLEMENTED**
    
    This will include:
    - Lag time selection via VAMP-2 score cross-validation
    - Dimension selection via cumulative kinetic variance
    - ITS (implied timescales) convergence analysis
    - Method-specific parameter optimization
    
    For now, returns reasonable defaults based on common practice.
    
    Parameters
    ----------
    traj_data : np.ndarray
        Trajectory feature matrix
    method : str
        ML method ('tica', 'vamp', 'pca', 'ae')
        
    Returns
    -------
    params : dict
        Suggested parameters with explanatory notes
    """
    n_frames = traj_data.shape[0]
    
    # Default parameters (to be replaced with automatic selection)
    if method == 'tica':
        suggested_lagtime = min(50, n_frames // 20)
        suggested_dim = 2
        
        return {
            'lagtime': suggested_lagtime,
            'dim': suggested_dim,
            'method': method,
            'note': 'Using default values. Automatic parameter selection to be implemented ...',
            'future_features': [
                'VAMP-2 score optimization for lag time',
                'Kinetic variance analysis for dimension selection',
                'ITS convergence checking',
                'Cross-validation framework'
            ]
        }
    
    else:
        return {
            'note': f'Parameter suggestion for {method} not yet implemented',
            'recommended': 'Use TICA for now, other methods coming'
        }


def train_cg_tica(traj_data, lagtime=50, dim=2, save_path=None):
    """
    Train TICA model on CG trajectory data.
    
    Parameters
    ----------
    traj_data : np.ndarray
        Feature matrix from CG trajectory, shape (n_frames, n_features)
    lagtime : int
        Lag time in frames for TICA
    dim : int
        Number of TICA components to keep
    save_path : str, optional
        If provided, save trained model to this path (.pkl)
        
    Returns
    -------
    tica_model : deeptime.decomposition.TICA
        Trained TICA model
    tica_output : np.ndarray
        Trajectory projected onto TICA space, shape (n_frames, dim)
    """
    if not DEEPTIME_AVAILABLE:
        raise ImportError("deeptime is required. Install with: pip install deeptime")
    
    print(f"\nTraining TICA model:")
    print(f"  Lag time: {lagtime} frames")
    print(f"  Dimensions: {dim}")
    
    # Initialize and fit TICA
    tica = TICA(lagtime=lagtime, dim=dim)
    tica = TICA(lagtime=lagtime, dim=dim)
    tica.fit(traj_data)
    
    # Transform data
    tica_output = tica.transform(traj_data)
    
    print(f"  Training complete!")
    print(f"  Output shape: {tica_output.shape}")
    # Eigenvalues not available in fetch_model()
    
    # Save model if requested
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(tica, f)
        print(f"  Model saved to: {save_path}")
    
    return tica, tica_output


def prepare_reap_interface(
    aa_dcd_path,
    topology_pdb,
    reference_pdb,
    cg_tica_pkl_path,
    output_npy_path,
    align_selection="protein and backbone and resid 68 to 92",
    feature_selection="name CA or name N or name C"
):
    """
    Prepare AA trajectory for REAP using CG-trained TICA model.
    
    This is the key interface between BEAM and REAP enhanced sampling.
    
    Steps:
    1. Load and preprocess AA trajectory (same as CG)
    2. Transform using CG-trained TICA model
    3. Ensure C-contiguous array (REAP requirement)
    4. Save as .npy file for REAP input
    
    Parameters
    ----------
    aa_dcd_path : str
        Path to AA trajectory DCD file
    topology_pdb : str
        Topology PDB for AA trajectory
    reference_pdb : str
        Reference structure for alignment
    cg_tica_pkl_path : str
        Path to CG-trained TICA model (.pkl)
    output_npy_path : str
        Output path for REAP input file (.npy)
    align_selection : str
        MDTraj selection for alignment
    feature_selection : str
        MDTraj selection for feature extraction
        
    Returns
    -------
    aa_cv : np.ndarray
        AA trajectory in CV space, ready for REAP
    """
    print("\n" + "="*60)
    print("Preparing REAP Interface")
    print("="*60)
    
    # Step 1: Load and preprocess AA trajectory
    print("\n[1/4] Loading AA trajectory...")
    aa_features = load_and_preprocess_cg(  # Same preprocessing as CG
        aa_dcd_path, topology_pdb, reference_pdb,
        align_selection, feature_selection
    )
    
    # Step 2: Load CG TICA model
    print("\n[2/4] Loading CG TICA model...")
    with open(cg_tica_pkl_path, 'rb') as f:
        cg_tica_model = pickle.load(f)
    print(f"  Model loaded from: {cg_tica_pkl_path}")
    
    # Step 3: Transform AA data with CG TICA
    print("\n[3/4] Transforming AA trajectory with CG-learned CVs...")
    aa_cv = cg_tica_model.transform(aa_features)
    print(f"  Output shape: {aa_cv.shape}")
    
    # Step 4: Ensure C-contiguous and save
    print("\n[4/4] Saving for REAP...")
    print(f"  C-contiguous before: {aa_cv.flags['C_CONTIGUOUS']}")
    aa_cv = np.ascontiguousarray(aa_cv)
    print(f"  C-contiguous after: {aa_cv.flags['C_CONTIGUOUS']}")
    
    np.save(output_npy_path, aa_cv)
    print(f"  Saved to: {output_npy_path}")
    
    print("\n" + "="*60)
    print("REAP interface ready!")
    print("="*60)
    
    return aa_cv


def run_full_cg_pipeline(
    cg_dcd_path,
    topology_pdb,
    reference_pdb,
    output_dir=".",
    lagtime=None,
    dim=None,
    align_selection="protein and backbone and resid 68 to 92",
    feature_selection="name CA or name N or name C"
):
    """
    Run complete CG pipeline from trajectory to trained TICA model.
    
    This is the main entry point for Stage 2.
    
    Parameters
    ----------
    cg_dcd_path : str
        Path to CG trajectory
    topology_pdb : str
        Topology for trajectory
    reference_pdb : str
        Reference for alignment
    output_dir : str
        Directory for outputs
    lagtime : int, optional
        TICA lag time (if None, uses suggested value)
    dim : int, optional
        TICA dimensions (if None, uses suggested value)
    align_selection : str
        Alignment atom selection
    feature_selection : str
        Feature extraction selection
        
    Returns
    -------
    results : dict
        Dictionary containing:
        - tica_model: trained model
        - tica_output: projected trajectory
        - params: used parameters
        - paths: output file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("\n" + "="*70)
    print("BEAM CG Pipeline - Stage 2")
    print("="*70)
    
    # Step 1: Load and preprocess
    print("\n[Stage 2.1] Loading and preprocessing CG trajectory...")
    cg_features = load_and_preprocess_cg(
        cg_dcd_path, topology_pdb, reference_pdb,
        align_selection, feature_selection
    )
    
    # Step 2: Suggest parameters if not provided
    if lagtime is None or dim is None:
        print("\n[Stage 2.2] Suggesting TICA parameters...")
        params = suggest_tica_params(cg_features)
        lagtime = lagtime or params['lagtime']
        dim = dim or params['dim']
        print(f"  Using: lagtime={lagtime}, dim={dim}")
        print(f"  Note: {params['note']}")
    
    # Step 3: Train TICA
    print("\n[Stage 2.3] Training TICA model...")
    tica_pkl_path = output_dir / "cg_tica_model.pkl"
    tica_model, tica_output = train_cg_tica(
        cg_features, 
        lagtime=lagtime, 
        dim=dim,
        save_path=str(tica_pkl_path)
    )
    
    # Step 4: Save outputs
    print("\n[Stage 2.4] Saving outputs...")
    cv_output_path = output_dir / "cg_cv_trajectory.npy"
    np.save(cv_output_path, tica_output)
    print(f"  CV trajectory saved: {cv_output_path}")
    
    print("\n" + "="*70)
    print("CG Pipeline Complete!")
    print("="*70)
    print("\nOutputs:")
    print(f"  TICA model: {tica_pkl_path}")
    print(f"  CV trajectory: {cv_output_path}")
    print("\nNext steps:")
    print("  1. Use TICA model with prepare_reap_interface() for AA enhanced sampling")
    print("  2. Or use for other enhanced sampling methods (REUS, metadynamics, etc.)")
    
    return {
        'tica_model': tica_model,
        'tica_output': tica_output,
        'cg_features': cg_features,
        'params': {'lagtime': lagtime, 'dim': dim},
        'paths': {
            'tica_model': str(tica_pkl_path),
            'cv_trajectory': str(cv_output_path)
        }
    }
