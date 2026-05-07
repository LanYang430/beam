"""
CG model configurations for BEAM evaluation framework.

Each CG model defines:
- description: Human-readable description
- align_selection: MDAnalysis selection for trajectory alignment
- feature_selection: MDAnalysis selection for feature extraction
- expected_atoms_per_residue: Expected number of CG beads per residue
"""

CG_MODELS = {
    'upside': {
        'description': 'Upside force field (backbone N, CA, C)',
        'align_selection': 'name CA',
        'feature_selection': 'name N or name CA or name C',
        'expected_atoms_per_residue': 3
    },
    'sirah': {
        'description': 'SIRAH CG force field',
        'align_selection': 'name GN or name GC',
        'feature_selection': 'all',  # SIRAH uses all CG beads
        'expected_atoms_per_residue': None  # Variable per residue type in SIRAH
    }
}


def get_cg_model_config(model_name):
    """
    Get configuration for a specific CG model.
    
    Parameters
    ----------
    model_name : str
        Name of the CG model (e.g., 'upside', 'sirah')
    
    Returns
    -------
    dict
        Configuration dictionary for the model
    
    Raises
    ------
    ValueError
        If model_name is not recognized
    """
    if model_name not in CG_MODELS:
        available = ', '.join(CG_MODELS.keys())
        raise ValueError(
            f"Unknown CG model: '{model_name}'. "
            f"Available models: {available}"
        )
    
    return CG_MODELS[model_name].copy()


def register_custom_cg_model(model_name, config):
    """
    Register a custom CG model configuration.
    
    Parameters
    ----------
    model_name : str
        Name for the custom CG model
    config : dict
        Configuration dictionary with keys:
        - description: str
        - align_selection: str (MDAnalysis selection)
        - feature_selection: str (MDAnalysis selection)
        - expected_atoms_per_residue: int or None
    
    Raises
    ------
    ValueError
        If model_name already exists or config is invalid
    
    Examples
    --------
    >>> config = {
    ...     'description': 'My custom CG model',
    ...     'align_selection': 'name CA',
    ...     'feature_selection': 'all',
    ...     'expected_atoms_per_residue': 5
    ... }
    >>> register_custom_cg_model('my_model', config)
    """
    if model_name in CG_MODELS:
        raise ValueError(
            f"Model '{model_name}' already exists. "
            f"Use a different name or modify CG_MODELS directly."
        )
    
    required_keys = {
        'description', 
        'align_selection', 
        'feature_selection', 
        'expected_atoms_per_residue'
    }
    
    missing_keys = required_keys - set(config.keys())
    if missing_keys:
        raise ValueError(
            f"Config missing required keys: {missing_keys}"
        )
    
    CG_MODELS[model_name] = config.copy()
    print(f"✓ Registered custom CG model: {model_name}")


def list_available_models():
    """
    Get list of available CG models.
    
    Returns
    -------
    list of str
        Names of available CG models
    """
    return list(CG_MODELS.keys())


def print_model_info(model_name=None):
    """
    Print information about CG models.
    
    Parameters
    ----------
    model_name : str, optional
        If provided, print info for specific model.
        If None, print info for all models.
    """
    if model_name is not None:
        config = get_cg_model_config(model_name)
        print(f"\nCG Model: {model_name}")
        print(f"Description: {config['description']}")
        print(f"Align selection: {config['align_selection']}")
        print(f"Feature selection: {config['feature_selection']}")
        atoms_per_res = config['expected_atoms_per_residue']
        if atoms_per_res is not None:
            print(f"Atoms per residue: {atoms_per_res}")
        else:
            print(f"Atoms per residue: Variable")
    else:
        print("\nAvailable CG Models:")
        print("=" * 60)
        for name, config in CG_MODELS.items():
            print(f"\n{name}:")
            print(f"  {config['description']}")


# SIRAH-specific notes:
"""
SIRAH CG Model Notes:
---------------------
SIRAH uses a multi-resolution approach with different bead types:

- Backbone: GC (glycine), GN (non-glycine)
- Side chains: Different beads per residue type (1-3 beads)
- Total beads per residue: Variable (2-4 typically)

Common SIRAH bead names:
- GC: Glycine backbone
- GN: Non-glycine backbone  
- GO1, GO2: Oxygen beads
- GS1, GS2, GS3: Side chain beads (type-dependent)

For alignment:
- Use 'name GC or name GN' for backbone alignment
- Or 'name CA' if CA-like beads are defined

For features:
- Use 'all' to include all CG beads
- Or specify: 'name GC or name GN or name GS1 or name GS2'

Reference:
Machado & Pantano (2016) J. Chem. Theory Comput. 12, 3513-3527
Machado et al. (2019) J. Chem. Theory Comput. 15, 2719-2733
"""
