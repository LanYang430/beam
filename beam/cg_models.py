"""
CG Model Configurations

Registry of supported CG models with their atom selections.
Users can register custom models easily.
"""

CG_MODEL_CONFIGS = {
    'upside': {
        'description': 'Upside force field (heavy atom-only)',
        'align_selection': 'name N CA C',
        'feature_selection': 'name N CA C',
        'expected_atoms_per_residue': 3
    },
    
    'martini': {
        'description': 'MARTINI v3 force field',
        'align_selection': 'name BB',
        'feature_selection': 'name BB or name SC1 or name SC2 or name SC3 or name SC4',
        'expected_atoms_per_residue': 4
    },
    
    'go_model': {
        'description': 'Structure-based Go model (CA)',
        'align_selection': 'name CA',
        'feature_selection': 'name CA',
        'expected_atoms_per_residue': 1
    },
    
    'ca_only': {
        'description': 'Generic CA-only model',
        'align_selection': 'name CA',
        'feature_selection': 'name CA',
        'expected_atoms_per_residue': 1
    },
    
    'all_atom': {
        'description': 'All-atom MD (for AA trajectories)',
        'align_selection': 'protein and backbone',
        'feature_selection': 'name CA or name N or name C',
        'expected_atoms_per_residue': 3
    }
}


def get_cg_model_config(model_name):
    """
    Get configuration for a CG model.
    
    Parameters
    ----------
    model_name : str
        Name of CG model
        
    Returns
    -------
    config : dict
        Model configuration with keys:
        - description: str
        - align_selection: MDTraj selection string
        - feature_selection: MDTraj selection string
        - expected_atoms_per_residue: int
        
    Raises
    ------
    ValueError
        If model_name is not registered
        
    Examples
    --------
    >>> config = get_cg_model_config('upside')
    >>> print(config['feature_selection'])
    'name CA'
    """
    if model_name not in CG_MODEL_CONFIGS:
        available = ', '.join(CG_MODEL_CONFIGS.keys())
        raise ValueError(
            f"Unknown CG model '{model_name}'. "
            f"Available models: {available}\n"
            f"Use register_custom_cg_model() to add custom models."
        )
    
    return CG_MODEL_CONFIGS[model_name].copy()


def register_custom_cg_model(name, align_selection, feature_selection, 
                             description=None, expected_atoms_per_residue=None):
    """
    Register a custom CG model configuration.
    
    Parameters
    ----------
    name : str
        Model name (will be used in load_and_preprocess_cg)
    align_selection : str
        MDTraj selection string for alignment
    feature_selection : str
        MDTraj selection string for features
    description : str, optional
        Human-readable description
    expected_atoms_per_residue : int, optional
        Expected number of atoms/beads per residue
        
    Examples
    --------
    >>> register_custom_cg_model(
    ...     name='my_cg_model',
    ...     align_selection='name CA',
    ...     feature_selection='name CA',
    ...     description='My custom CG model',
    ...     expected_atoms_per_residue=1
    ... )
    >>> # Now can use it:
    >>> features = load_and_preprocess_cg(..., cg_model='my_cg_model')
    """
    config = {
        'align_selection': align_selection,
        'feature_selection': feature_selection,
        'description': description or f'Custom CG model: {name}',
    }
    
    if expected_atoms_per_residue is not None:
        config['expected_atoms_per_residue'] = expected_atoms_per_residue
    
    CG_MODEL_CONFIGS[name] = config
    print(f"✓ Registered custom CG model: {name}")


def list_available_models():
    """
    List all registered CG models.
    
    Returns
    -------
    models : list of str
        Available model names
    """
    return list(CG_MODEL_CONFIGS.keys())
