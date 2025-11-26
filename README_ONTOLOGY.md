# Ontology-Inspired Rule Layer

This module implements a domain-knowledge-based penalty system to enhance anomaly detection for early hospital readmission prediction. It combines machine learning anomaly scores (IsolationForest) with expert rule-based penalties derived from clinical domain knowledge.

## Features

- **Clinical Rule Implementation**: Encodes expert medical knowledge into executable rules.
- **Hybrid Scoring**: Combines statistical anomaly scores with rule-based penalties.
- **Comprehensive Evaluation**: Metrics for both baseline and enhanced methods.

## Clinical Rules

The system implements the following risk rules based on diabetes care guidelines:

1.  **High Risk (Penalty 0.9)**:
    *   **Condition**: Poor glycemic control (A1C > 7/8) AND No medication changes AND Patient on diabetes meds.
    *   **Rationale**: Indicates potential treatment non-compliance or inadequate management.

2.  **High Risk (Penalty 0.85)**:
    *   **Condition**: Very high glucose (>200/300) AND Insufficient lab monitoring (<40 procedures).
    *   **Rationale**: Suggests insufficient monitoring of severe hyperglycemia.

3.  **Medium Risk (Penalty 0.6)**:
    *   **Condition**: High medication burden (>20 meds) AND Short hospital stay (<3 days).
    *   **Rationale**: May indicate complex case with inadequate stabilization time.

4.  **Low Risk (Penalty 0.1)**:
    *   Default for cases not matching specific risk patterns.

## Usage

### Basic Usage

```python
from ontology_rule_layer import compute_ontology_penalty

# Apply to a single row
penalty = compute_ontology_penalty(patient_row)
```

### Full Pipeline Integration

```python
from ontology_rule_layer import compare_anomaly_methods

# df_test must contain: 'iforest_score', 'y', and original features
enhanced_df = compare_anomaly_methods(df_test)
```

## Testing

Unit tests are provided to verify the logic of the rules:

```bash
python -m unittest tests/test_ontology_rules.py
```

## Files

- `ontology_rule_layer.py`: Core implementation.
- `example_ontology_usage.py`: End-to-end demonstration script.
- `tests/test_ontology_rules.py`: Unit tests.
