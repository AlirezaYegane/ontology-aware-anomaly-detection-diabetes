# Refactoring Plan: Code Quality & Organization

This document outlines a systematic refactoring plan to improve code quality, remove legacy code, and enhance maintainability while preserving all functionality.

---

## Task 1: Identify Active Modules and Mark Legacy Files for Archival

**Description**: Perform a comprehensive audit of the codebase to distinguish between active production code and legacy/experimental files.

**Active Core Modules** (production pipeline):
- ✅ `run_pipeline_direct.py` - Main entrypoint for pipeline execution
- ✅ `src/config.py` - Configuration management
- ✅ `src/logger.py` - Structured logging
- ✅ `src/preprocessing.py` - Data preprocessing and filtering
- ✅ `src/models.py` - IsolationForest and Autoencoder implementations
- ✅ `src/ontology.py` - Ontology-aware rule layer
- ✅ `src/evaluation.py` - Metrics computation and plotting
- ✅ `src/filter_tracker.py` - Data filtering instrumentation
- ✅ `src/utils.py` - Path setup utilities (Colab/Local compatibility)
- ✅ `src/__init__.py` - Module initialization
- ✅ `tests/test_preprocessing.py` - Preprocessing unit tests
- ✅ `tests/test_models.py` - Model unit tests
- ✅ `tests/test_ontology.py` - Ontology rule unit tests
- ✅ `notebooks/01_eda.ipynb` - Exploratory Data Analysis
- ✅ `notebooks/02_baseline_if.ipynb` - Isolation Forest baseline
- ✅ `notebooks/03_autoencoder.ipynb` - Autoencoder experiments
- ✅ `notebooks/04_ontology_eval.ipynb` - Ontology evaluation

**Legacy/Auxiliary Files** (to be archived):
- ⚠️ `run_pipeline.py` - Old notebook-based pipeline runner (superseded by `run_pipeline_direct.py`)
- ⚠️ `setup_project.py` - Environment setup script (useful but not core pipeline)
- ⚠️ `update_notebooks.py` - Notebook updater utility (useful but not core pipeline)
- ⚠️ Multiple documentation files (analysis needed):
  - `GOOGLE_COLAB_SETUP.md`
  - `LOCAL_WINDOWS_SETUP.md`
  - `SETUP_GUIDE.md`
  - `TROUBLESHOOTING.md`
  - `TASKS_PREPROCESSING.md`
  - `EXPERIMENT_LOG.md` (keep this one - contains valuable experiment records)

**Notes**:
- `archive/` folder already exists and contains old scripts/notebooks
- Current `archive/` contains 25 items including old implementations
- Most legacy code has already been moved to archive in previous refactoring

**Actions**:
- [x] List all files in project root
- [x] Identify which scripts are actively used by the current pipeline
- [ ] Create a decision matrix: Keep vs Archive vs Delete
- [ ] Document dependencies between auxiliary scripts and core modules

---

## Task 2: Move Legacy Code, Notebooks, and Unused Scripts into `archive/`

**Description**: Systematically move non-core files to `archive/` to clean up the project root while preserving historical context.

**Files to Archive**:

### Scripts to Archive:
- [ ] `run_pipeline.py` → `archive/run_pipeline.py`
  - Reason: Superseded by `run_pipeline_direct.py`
  - Note: Uses Jupyter nbconvert; not part of main pipeline

- [ ] `setup_project.py` → `archive/setup_project.py`
  - Reason: Auxiliary setup utility, not core pipeline logic
  - Note: Keep a note in README about environment setup alternatives

- [ ] `update_notebooks.py` → `archive/update_notebooks.py`
  - Reason: One-time utility for notebook path fixes
  - Note: Already applied; no longer needed for future maintenance

### Documentation to Consolidate:
- [ ] Review and merge setup documentation:
  - Keep: `README.md` (main documentation)
  - Keep: `EXPERIMENT_LOG.md` (experiment results)
  - Archive or merge:
    - `GOOGLE_COLAB_SETUP.md` → Consider adding key points to README
    - `LOCAL_WINDOWS_SETUP.md` → Consider adding key points to README
    - `SETUP_GUIDE.md` → Possibly redundant with README
    - `TROUBLESHOOTING.md` → Archive or add key points to README
    - `TASKS_PREPROCESSING.md` → Archive (historical task tracking)

**Actions**:
- [ ] Create `archive/auxiliary_scripts/` subdirectory
- [ ] Move identified scripts to archive with git history preserved
- [ ] Create `archive/old_docs/` subdirectory
- [ ] Consolidate documentation (merge or archive)
- [ ] Update README with simplified setup instructions
- [ ] Add note in archive/README.md explaining what's archived and why

---

## Task 3: Replace `print` Statements with Structured Logging

**Description**: Replace all raw `print()` calls with proper logging using the `src/logger.py` module to enable better debugging, filtering, and production readiness.

**Files Affected**: 
- `src/preprocessing.py` - ~10 print statements
- `src/filter_tracker.py` - ~20 print statements
- `src/evaluation.py` - ~15 print statements
- `src/utils.py` - ~8 print statements
- `src/models.py` - ~1 commented print statement
- (Main runner `run_pipeline_direct.py` can keep some prints for user-facing output)

**Approach**:
1. Import logger: `from src.logger import get_logger`
2. Initialize logger: `logger = get_logger(__name__)`
3. Replace print patterns:
   - Informational messages → `logger.info(...)`
   - Success messages (✅) → `logger.info(...)`
   - Warning messages (⚠️) → `logger.warning(...)`
   - Error messages (❌) → `logger.error(...)`
   - Debug/verbose messages → `logger.debug(...)`
   - Headers/separators → Keep strategic ones, log others at INFO

**Specific Changes**:

### `src/preprocessing.py`:
- [ ] Lines 237-239: Header prints → `logger.info("DATA FILTERING & PREPROCESSING PIPELINE")`
- [ ] Lines 382, 391, 393: Status prints → logger calls
- [ ] Ensure all filtering messages go through logger

### `src/filter_tracker.py`:
- [ ] Lines 62, 86: Step logging → `logger.info(...)`
- [ ] Lines 103-131: Summary prints → `logger.info(...)` for structured summary
- [ ] Lines 146, 193: File save confirmations → `logger.info(...)`

### `src/evaluation.py`:
- [ ] Lines 37-56: Evaluation results → `logger.info(...)` with structured formatting
- [ ] Lines 170-174: Comparison table → `logger.info(...)`

### `src/utils.py`:
- [ ] Lines 27, 73: Environment detection → `logger.info(...)`
- [ ] Lines 95, 97, 102, 104-105: Path setup → `logger.debug(...)` or `logger.info(...)`
- [ ] Line 142: Results dir ready → `logger.info(...)`

**Actions**:
- [ ] Create a mapping document: print statement → appropriate log level
- [ ] Update `src/preprocessing.py` with logger
- [ ] Update `src/filter_tracker.py` with logger
- [ ] Update `src/evaluation.py` with logger
- [ ] Update `src/utils.py` with logger
- [ ] Verify `src/models.py` doesn't need changes (commented print found)
- [ ] Run tests to ensure no functionality broken: `pytest -q`

---

## Task 4: Clean Up Class and Module Organization within `src/`

**Description**: Refactor internal module organization for better code readability, maintainability, and adherence to Python conventions, while keeping public API stable for `run_pipeline_direct.py` and `tests/`.

**Files to Refactor**:

### `src/preprocessing.py` (~400 lines):
- [ ] Review function organization and grouping
- [ ] Extract constants to module-level (if any)
- [ ] Ensure docstrings follow consistent format (Google/NumPy style)
- [ ] Remove any unused imports
- [ ] Check for any dead code or commented-out sections

### `src/models.py` (~400 lines):
- [ ] Review IsolationForestModel class structure
- [ ] Review AutoencoderModel class structure
- [ ] Ensure consistent parameter naming and documentation
- [ ] Remove commented code (e.g., line 227 commented print)
- [ ] Verify private methods use leading underscore convention

### `src/ontology.py` (~250 lines):
- [ ] Review rule computation functions
- [ ] Ensure rule names are consistent and well-documented
- [ ] Check if any rules can be refactored for clarity
- [ ] Verify score combination logic is clearly explained

### `src/evaluation.py` (~350 lines):
- [ ] Review metrics computation functions
- [ ] Ensure plotting functions are well-organized
- [ ] Check for code duplication in plot generation
- [ ] Verify all metrics have proper docstrings

### `src/filter_tracker.py` (~200 lines):
- [ ] Review class organization
- [ ] Ensure method naming is consistent
- [ ] Check if summary generation can be simplified
- [ ] Verify JSON/Markdown export functions are robust

### `src/config.py`:
- [ ] Review configuration structure
- [ ] Ensure all constants are properly documented
- [ ] Check if any hardcoded values should be moved here

### `src/utils.py`:
- [ ] Review path setup logic
- [ ] Ensure Colab/Local detection is robust
- [ ] Check if any utility functions are unused

### `src/logger.py`:
- [ ] Review logger configuration
- [ ] Ensure log levels are appropriate
- [ ] Check if any custom handlers needed

**General Cleanup Actions**:
- [ ] Run import sorting: `isort src/ tests/`
- [ ] Check for unused imports across all modules
- [ ] Ensure consistent docstring style (recommend Google style)
- [ ] Verify all classes/functions have type hints
- [ ] Check for overly long functions (>100 lines) that should be split
- [ ] Ensure consistent naming conventions:
  - Functions: `snake_case`
  - Classes: `PascalCase`
  - Constants: `UPPER_SNAKE_CASE`
  - Private methods: `_leading_underscore`

**API Stability Check**:
- [ ] Document current public API from each module
- [ ] Ensure `run_pipeline_direct.py` imports will not break
- [ ] Ensure all test imports will not break
- [ ] Flag any breaking changes for user review

---

## Task 5: Run and Fix Tests (`pytest -q`)

**Description**: Ensure all unit tests pass after refactoring, and add any missing test coverage for core functionality.

**Test Files**:
- `tests/test_preprocessing.py`
- `tests/test_models.py`
- `tests/test_ontology.py`

**Testing Strategy**:

### Initial Test Run:
- [ ] Run full test suite: `pytest -v`
- [ ] Document current test coverage: `pytest --cov=src --cov-report=term-missing`
- [ ] Identify any failing tests from refactoring

### Fix Broken Tests:
- [ ] Fix any import errors from module refactoring
- [ ] Fix any API changes that broke tests
- [ ] Update test fixtures if needed
- [ ] Ensure all assertions still valid

### Enhance Test Coverage:
- [ ] Check if new functionality needs tests
- [ ] Verify edge cases are tested
- [ ] Add docstrings to test functions
- [ ] Consider adding integration tests for full pipeline

### Quality Checks:
- [ ] Ensure tests are deterministic (fix random seeds)
- [ ] Verify tests run quickly (flag slow tests)
- [ ] Check test naming follows conventions: `test_<functionality>`
- [ ] Ensure test files mirror source structure

**Test Execution Checklist**:
- [ ] Run quick tests: `pytest -q`
- [ ] Run verbose tests: `pytest -v`
- [ ] Run with coverage: `pytest --cov=src`
- [ ] Document coverage gaps
- [ ] All tests must pass before proceeding to Task 6

---

## Task 6: Final Pass - Remove Dead Code, Unused Imports, and Reformat Code (PEP8)

**Description**: Final cleanup pass to ensure codebase follows Python best practices, is PEP8 compliant, and contains no dead code or unused imports.

**Tools to Use**:
- `black` - Code formatting
- `isort` - Import sorting
- `flake8` or `ruff` - Linting
- `pylint` - Additional code quality checks
- `mypy` - Type checking (optional but recommended)

**Final Cleanup Actions**:

### Code Formatting:
- [ ] Install formatter: `pip install black isort`
- [ ] Format all source files: `black src/ tests/ *.py`
- [ ] Sort all imports: `isort src/ tests/ *.py`
- [ ] Verify line length <= 88 characters (black default) or 79 (PEP8 strict)

### Linting:
- [ ] Install linter: `pip install flake8` or `pip install ruff`
- [ ] Run linter: `flake8 src/ tests/` or `ruff check src/ tests/`
- [ ] Fix all linting errors
- [ ] Review and fix linting warnings
- [ ] Document any intentional violations (if any)

### Dead Code Removal:
- [ ] Search for commented-out code blocks and remove them
- [ ] Remove unused functions (check with IDE or coverage tools)
- [ ] Remove unused class methods
- [ ] Remove debug code and temporary hacks
- [ ] Check for unused variables in functions

### Import Cleanup:
- [ ] Remove unused imports from all files
- [ ] Organize imports in standard order:
  1. Standard library imports
  2. Third-party imports
  3. Local application imports
- [ ] Verify no circular import issues
- [ ] Use absolute imports where possible

### Documentation Review:
- [ ] Ensure all modules have module-level docstrings
- [ ] Ensure all public functions have docstrings
- [ ] Ensure all classes have docstrings
- [ ] Review README.md for accuracy after refactoring
- [ ] Update any outdated comments

### Type Hints (Optional but Recommended):
- [ ] Add type hints to function signatures where missing
- [ ] Run `mypy src/` to check type consistency
- [ ] Fix type-related issues

### Final Verification:
- [ ] Run full test suite: `pytest -v`
- [ ] Run full pipeline: `python run_pipeline_direct.py` (at least one quick run)
- [ ] Verify all results are reproducible
- [ ] Check that `results/` outputs are generated correctly
- [ ] Review git diff to ensure no unintended changes
- [ ] Create a summary of all changes made

---

## Success Criteria

By the end of this refactoring plan:

✅ **Organization**:
- Clear separation between active code and legacy files
- Clean project root with only essential files
- Well-organized `archive/` with proper documentation

✅ **Code Quality**:
- No raw `print()` calls in `src/` modules (use structured logging)
- All code follows PEP8 conventions
- No unused imports or dead code
- Consistent naming conventions throughout

✅ **Testing**:
- All tests pass: `pytest -q` shows all green
- Test coverage documented
- No broken imports or API changes

✅ **Documentation**:
- README.md is accurate and up-to-date
- All functions/classes have proper docstrings
- Archive is properly documented

✅ **Functionality**:
- Full pipeline runs successfully: `python run_pipeline_direct.py`
- All outputs generated correctly in `results/`
- No regressions in functionality

---

## Notes for Implementation

- **Git commits**: Make frequent, atomic commits for each subtask
- **Backup**: Ensure you have a backup before starting major changes
- **Incremental testing**: Run tests after each major change
- **Public API stability**: The refactoring should NOT break:
  - `run_pipeline_direct.py`
  - `tests/*` imports
  - Notebook imports (if they use `src.` modules)
- **Documentation**: Update inline comments as you refactor
- **Review**: Consider peer review before marking tasks complete

---

**Current Status**: 📋 READY TO START
**Last Updated**: 2025-11-30
