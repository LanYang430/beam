# BEAM v0.2.0 - Complete Prototype Guide

## Quick Reference

**What's New:** Consolidated 3-stage architecture aligned with research workflow  
**Core Modules:** 3 files (~880 lines total)  
**Status:** Ready for fellowship application

---

## Three-Stage Architecture

### Stage 1: CG Evaluation ⏸️ **NOT IMPLEMENTED**  
→ Fellowship development priority

### Stage 2: CG→CV ✅ **IMPLEMENTED**  
→ `cg_pipeline.py`

### Stage 3: AA Analysis ✅ **IMPLEMENTED**  
→ `aa_analysis.py`

---

## What's Implemented

✅ Load CG/AA trajectories (DCD + PDB)  
✅ Align and extract features  
✅ Train TICA on CG data  
✅ REAP interface  
✅ Train TICA on AA data  
✅ **CG+AA overlay visualization**  
✅ REUS window generation  

⏸️ Automatic parameter selection (placeholder)  
⏸️ Detailed CG/AA comparison (placeholder)  

---

## File Structure

```
beam_v2/
├── beam/
│   ├── cg_pipeline.py    # Stage 2
│   ├── aa_analysis.py    # Stage 3
│   └── visualize.py      # All plots
├── examples/             # Demo notebooks
└── data/                 # Test data
```

---

## Usage

See `examples/` for complete workflows with your data!

---

## For Fellowship Proposal

**Current:** Stages 2 & 3 working  
**Goal:** Add Stage 1 + enhance automation  
**Timeline:** 2 semesters
