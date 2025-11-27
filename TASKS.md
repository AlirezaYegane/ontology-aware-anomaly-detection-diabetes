# Project Restructuring Tasks

## Phase 1: Initialization & Task Tracking
- [x] Create TASKS.md in project root
- [x] Populate TASKS.md with full checklist

## Phase 2: Directory Restructuring
- [x] Verify current directory structure
- [x] Ensure data files are in data/raw/
- [x] Clean archive/ (all old/unused files already moved)
- [x] Verify clean root directory structure

## Phase 3: Source Code Implementation
- [x] Review and update src/__init__.py
- [x] Implement/Update src/preprocessing.py (functions: `load_raw_data`, `build_feature_matrix`, `train_test_split_stratified`)
- [x] Implement/Update src/models.py (Isolation Forest and Autoencoder classes)
- [x] Implement/Update src/ontology.py (`compute_ontology_penalty`, `combine_scores`)
- [x] Implement/Update src/evaluation.py (ROC/PR metrics, plots, summaries)

## Phase 4: Notebook Creation
- [x] Create/Update notebooks/01_eda.ipynb with Colab path fix
- [x] Create/Update notebooks/02_baseline_if.ipynb with Colab path fix
- [x] Create/Update notebooks/03_autoencoder.ipynb with Colab path fix
- [x] Create/Update notebooks/04_ontology_eval.ipynb with Colab path fix

## Phase 5: Final Cleanup
- [x] Update requirements.txt with all dependencies
- [x] Update README.md with clear instructions
- [x] Final verification of all files
- [x] Mark all tasks complete

---
**Status**: ✅ ALL TASKS COMPLETE!
