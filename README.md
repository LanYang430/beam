# BEAM: Boosted Enhanced Sampling through All-Atom Simulations Guided by Machine-Learned Collective Variables

**Version:** 0.2.0 (CSSE@GT Fellowship - Milestone 2)  
**Author:** Lan Yang  
**Contact:** lyang430@gatech.edu  

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

BEAM is an open-source toolkit that accelerates biomolecular conformational sampling by learning collective variables (CVs) from fast coarse-grained (CG) simulations and applying them to guide all-atom (AA) enhanced sampling.

**The Challenge:** Selecting effective collective variables for enhanced sampling is a major bottleneck in computational biophysics. Traditional geometric CVs (RMSD, contacts, etc.) often fail to capture the true slow modes of complex molecular processes.

**The BEAM Solution:**  
1. **Evaluate CG models** to ensure quality before downstream use
2. Run fast CG simulations to explore conformational space
3. Use machine learning to automatically discover low dimensional CVs
4. Map learned CVs to AA systems for enhanced sampling
5. Analyze and validate results

---

## Features

### Current (v0.2.0)

**Stage 1: CG Model Evaluation** ✅ **NEW in v0.2.0**
- Multi-system, multi-CG-model evaluation framework
- Comprehensive metrics across three dimensions:
  - **Distribution-based:** RDF, CV distributions (KS test, JS divergence, overlap coefficient)
  - **Dynamics-based:** Autocorrelation times, decorrelation rates, speedup factors
  - **Sampling efficiency:** Effective sample size (ESS), basin discovery, conformational variance
- **In-support analysis:** Distinguish CG model errors from enhanced sampling effects
- Support for multiple CG force fields (Upside, SIRAH)
- Tested on multiple protein systems (Pertactin, ADK)
- Complete Jupyter notebook workflow with automated reporting

**Stage 2: CG → CV Pipeline**
- Load and preprocess CG trajectories (MDTraj)
- Train TICA to learn slow CVs
- Save trained models (.pkl)
- Basic REAP-compatible interface
- API placeholder for automatic parameter suggestion

**Stage 3: AA Analysis**
- Load and analyze AA trajectories
- Transform AA data into CG-learned CV space
- Visualization of CG/AA projections overlay
- Train TICA on AA data for refined CVs
- Generate REUS window suggestions
- API placeholder for Quantitative CG/AA comparison

**Visualization**
- TICA projections
- Free energy landscapes
- CG/AA overlay
- Residue-level contribution plots
- Timescale plots
- **NEW:** In-support distribution comparisons
- **NEW:** Multi-system comparison plots

### Planned (Fellowship Development)

**Enhanced Automation** 
- Automatic lag time selection (VAMP-2 cross-validation)
- Automatic dimensionality selection (kinetic variance)
- ITS-based convergence analysis
- Cross-validation frameworks

**Advanced Analysis**
- Quantitative CG/AA CV comparison
- Cross-scale consistency metrics
- Residue-level correlation analysis

**Extended Method Support**
- VAMP, PCA, autoencoders
- Additional CG force fields (MARTINI, AWSEM)
- Additional enhanced sampling interfaces (Weighted Ensemble, etc.)

**Production Quality**
- CI/CD pipeline (GitHub Actions)
- Unit tests and code coverage
- Additional examples and tutorials
- Full API documentation
- Performance optimization

---

## Installation

### Requirements

- Python >= 3.7
- numpy >= 1.19.0
- matplotlib >= 3.3.0
- mdtraj >= 1.9.0 (for trajectory handling)
- MDAnalysis >= 2.0.0 (for Stage 1 evaluation)
- deeptime >= 0.4.0 (for TICA/VAMP)

### Install BEAM

```bash
# Clone repository
git clone https://github.com/LanYang430/beam.git
cd beam

# Install dependencies
pip install -r requirements.txt

# Install BEAM in development mode
pip install -e .
```

---

## Quick Start

### 0. CG Model Evaluation (Stage 1)

```python
from beam import CGEvaluator

# Load trajectories (AA and CG)
# ... (see examples/stage1_cg_evaluation.ipynb)

# Initialize evaluator
evaluator = CGEvaluator(
    cg_features=cg_features,
    aa_features=aa_features,
    cg_positions=cg_positions,
    aa_positions=aa_positions
)

# Run complete evaluation
report = evaluator.evaluate_all(
    cv_list=[
        ('Rg', rg_function),
        ('end_to_end', e2e_function),
        ('contacts', contacts_function)
    ],
    n_clusters=10
)

# Generate summary
summary = evaluator.generate_summary(report)
print(summary)
```

For complete workflow, see `examples/stage1_cg_evaluation.ipynb`.

### 1. CG → CV Learning (Stage 2)

```python
from beam import load_and_preprocess_cg, train_cg_tica, plot_tica_projection

# Load CG trajectory
cg_features = load_and_preprocess_cg(
    'cg_traj.dcd',
    'topology.pdb',
    'reference.pdb'
)

# Train TICA
tica_model, cg_cv = train_cg_tica(
    cg_features,
    lagtime=50,
    dim=2,
    save_path='cg_tica_model.pkl'
)

# Visualize
plot_tica_projection(cg_cv, title="CG Collective Variables")
```

### 2. AA Analysis (Stage 3)

```python
from beam import (
    load_and_preprocess_aa,
    transform_aa_with_cg_tica,
    train_aa_tica,
    plot_cg_aa_overlay
)

# Load AA trajectory
aa_features = load_and_preprocess_aa(
    'aa_traj.dcd',
    'topology.pdb',
    'reference.pdb'
)

# Transform with CG model
aa_in_cg_cv = transform_aa_with_cg_tica(aa_features, 'cg_tica_model.pkl')

# Load CG CV for comparison
import pickle
with open('cg_tica_model.pkl', 'rb') as f:
    cg_model = pickle.load(f)
cg_cv = cg_model.transform(cg_features)

# Create key overlay figure
plot_cg_aa_overlay(cg_cv, aa_in_cg_cv, save_path='overlay.png')
```

For complete workflows, see `examples/` directory.

---

## Project Structure

```
beam/
├── beam/                    # Main package
│   ├── __init__.py
│   ├── cg_evaluation.py     # Stage 1: CG evaluation (NEW)
│   ├── evaluation_utils.py  # Stage 1: Metrics (NEW)
│   ├── cg_models.py         # CG model registry (NEW)
│   ├── cg_pipeline.py       # Stage 2: CG → CV
│   ├── aa_analysis.py       # Stage 3: AA analysis
│   └── visualize.py         # Plotting functions
│
├── examples/                # Example notebooks
│   ├── stage1_cg_evaluation.ipynb          # NEW
│   ├── demo_stage2_cg_pipeline.ipynb
│   └── demo_stage3_aa_analysis.ipynb
│
├── data/                    # Demo data
│   ├── pertactin/          # Pertactin test system
│   ├── adk/                # ADK test system (NEW)
│   └── (synthetic trajectories)
│
├── README.md                # This file
├── requirements.txt         # Python dependencies
├── setup.py                 # Installation script
└── LICENSE                  # MIT license
```

---

## Development Roadmap

### ✅ Completed (v0.2.0) - Milestone 2
- [x] **Stage 1: CG model evaluation framework**
- [x] Multi-system support (Pertactin, ADK)
- [x] Multi-CG-model support (Upside, SIRAH)
- [x] In-support analysis methodology
- [x] Complete evaluation metrics suite
- [x] Comprehensive documentation and examples
- [x] CG → CV learning pipeline
- [x] AA analysis workflow
- [x] Core visualization utilities
- [x] REAP interface
- [x] Demo notebooks

### Planned Work - Phase I (Milestone 3)
- [ ] CI/CD pipeline setup (GitHub Actions)
- [ ] Unit tests and code coverage
- [ ] Code modularization
- [ ] Automatic parameter selection (VAMP-2)
- [ ] Performance optimization
- [ ] Enhanced documentation

### Planned Work - Phase II
- [ ] Advanced CV comparison metrics
- [ ] Support for additional ML methods
- [ ] Additional CG force fields (MARTINI, AWSEM)
- [ ] Additional enhanced sampling interfaces
- [ ] Production-ready release
- [ ] Community-facing tutorials

---

## Key Publications and Methods

**In-Support Analysis** - A novel method for distinguishing CG model errors from enhanced sampling effects by comparing distributions only within the AA-sampled region. This allows:
- Clear assessment of CG force field accuracy
- Identification of valid enhanced sampling (CG exploring new conformations)
- Informed CG model selection for downstream workflows

See `examples/stage1_cg_evaluation.ipynb` for implementation details.

---

## Citation

If you use BEAM in your research, please cite:

```bibtex
@software{beam2025,
  title={BEAM: Boosted Enhanced sampling through Machine-learned CVs},
  author={Lan Yang},
  year={2025},
  url={https://github.com/LanYang430/beam.git}
}
```

---

## License

MIT License - See [LICENSE](LICENSE) file for details

---

## Acknowledgments

Supported by CSSE@GT Fellowship and Schmidt Sciences.

Special thanks to Robert Bates for software engineering guidance.

---

## Contact

For questions or issues, please contact:  
lyang430@gatech.edu

**BEAM: Making enhanced sampling accessible through data-driven collective variable discovery**
