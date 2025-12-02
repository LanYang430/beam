"""
BEAM: Boosted Enhanced sampling through All-atom simulations 
guided by Machine-learned collective variables

A toolkit for learning collective variables from coarse-grained simulations
and applying them to all-atom enhanced sampling.

Main modules:
- cg_pipeline: CG trajectory → TICA training → AA enhanced sampling prep
- aa_analysis: AA trajectory analysis and comparison with CG
- visualize: Visualization tools for trajectories and CVs
"""

__version__ = "0.2.0"

# Core functionality
from .cg_pipeline import (
    load_and_preprocess_cg,
    train_cg_tica,
    suggest_tica_params,
    prepare_reap_interface
)

from .aa_analysis import (
    load_and_preprocess_aa,
    transform_aa_with_cg_tica,
    train_aa_tica,
    compare_cg_aa_cvs,
    generate_reus_windows
)

from .visualize import (
    plot_tica_projection,
    plot_free_energy_landscape,
    plot_cg_aa_overlay,
    plot_residue_contributions
)

__all__ = [
    # CG Pipeline
    'load_and_preprocess_cg',
    'train_cg_tica',
    'suggest_tica_params',
    'prepare_reap_interface',
    
    # AA Analysis
    'load_and_preprocess_aa',
    'transform_aa_with_cg_tica',
    'train_aa_tica',
    'compare_cg_aa_cvs',
    'generate_reus_windows',
    
    # Visualization
    'plot_tica_projection',
    'plot_free_energy_landscape',
    'plot_cg_aa_overlay',
    'plot_residue_contributions',
]
