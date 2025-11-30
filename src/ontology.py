"""
Ontology-inspired rule layer for anomaly detection.

This module implements a small set of transparent, clinically motivated rules
on top of the Diabetes 130-US hospitals dataset. It provides:

- A per-patient ontology penalty in [0, 1]
- Rule-level trigger statistics for analysis
- A combiner to mix ML anomaly scores with ontology penalties
"""

from __future__ import annotations

from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd


# -------------------------------------------------------------------------
# Rule definitions
# -------------------------------------------------------------------------

# Weights for each ontology rule.
# The actual firing logic is implemented in _evaluate_rules().
RULE_WEIGHTS: Dict[str, float] = {
    # Poor glycemic control and no adequate medication adjustment
    "poor_control_no_med_change": 0.9,
    # Very high glucose with short length of stay
    "high_glucose_short_stay": 0.85,
    # Multiple recent inpatient admissions
    "frequent_inpatient_admissions": 0.6,
    # Large number of concurrent medications
    "polypharmacy": 0.4,
    # Emergency + inpatient utilisation
    "er_and_inpatient_use": 0.5,
}


def _safe_get(row: pd.Series, key: str, default=None):
    """Safely access a value from a pandas row with a default."""
    return row[key] if key in row.index else default


def _evaluate_rules(row: pd.Series) -> Dict[str, int]:
    """
    Evaluate ontology rules on a single patient row.

    Returns
    -------
    Dict[str, int]
        Mapping rule_name -> 0/1 indicating whether each rule fired.
    """
    flags: Dict[str, int] = {name: 0 for name in RULE_WEIGHTS.keys()}

    # Extract commonly used fields, with safe defaults
    a1c = _safe_get(row, "A1Cresult")
    max_glu = _safe_get(row, "max_glu_serum")
    change = _safe_get(row, "change")
    diabetes_med = _safe_get(row, "diabetesMed")
    insulin = _safe_get(row, "insulin")  # Currently unused, kept for future rules

    num_inpatient = _safe_get(row, "number_inpatient", 0) or 0
    num_emergency = _safe_get(row, "number_emergency", 0) or 0
    num_meds = _safe_get(row, "num_medications", 0) or 0
    time_in_hosp = _safe_get(row, "time_in_hospital", 0) or 0

    # Normalise string values slightly
    if isinstance(a1c, str):
        a1c = a1c.strip()
    if isinstance(max_glu, str):
        max_glu = max_glu.strip()
    if isinstance(change, str):
        change = change.strip()
    if isinstance(diabetes_med, str):
        diabetes_med = diabetes_med.strip()
    if isinstance(insulin, str):
        insulin = insulin.strip()

    # ------------------------------------------------------------------
    # Rule 1: Poor glycemic control and no medication change
    #
    # Trigger if:
    #   - A1Cresult in {">7", ">8"}
    #   - AND (no medication change OR diabetesMed == "No")
    # ------------------------------------------------------------------
    if a1c in {">7", ">8"} and (change == "No" or diabetes_med == "No"):
        flags["poor_control_no_med_change"] = 1

    # ------------------------------------------------------------------
    # Rule 2: Very high glucose with short hospital stay
    #
    # Trigger if:
    #   - max_glu_serum in {">200", ">300"}
    #   - AND time_in_hospital <= 3 days
    # ------------------------------------------------------------------
    if max_glu in {">200", ">300"} and time_in_hosp is not None:
        try:
            if int(time_in_hosp) <= 3:
                flags["high_glucose_short_stay"] = 1
        except Exception:
            # If time_in_hospital is not numeric, ignore this rule
            pass

    # ------------------------------------------------------------------
    # Rule 3: Frequent inpatient admissions
    #
    # Trigger if:
    #   - number_inpatient >= 2
    # ------------------------------------------------------------------
    try:
        if int(num_inpatient) >= 2:
            flags["frequent_inpatient_admissions"] = 1
    except Exception:
        pass

    # ------------------------------------------------------------------
    # Rule 4: Polypharmacy (many medications)
    #
    # Trigger if:
    #   - num_medications >= 10
    # ------------------------------------------------------------------
    try:
        if int(num_meds) >= 10:
            flags["polypharmacy"] = 1
    except Exception:
        pass

    # ------------------------------------------------------------------
    # Rule 5: Emergency + inpatient use
    #
    # Trigger if:
    #   - number_emergency >= 1 AND number_inpatient >= 1
    # ------------------------------------------------------------------
    try:
        if int(num_emergency) >= 1 and int(num_inpatient) >= 1:
            flags["er_and_inpatient_use"] = 1
    except Exception:
        pass

    return flags


def _penalty_from_flags(flags: Dict[str, int]) -> float:
    """
    Aggregate rule fires into a single penalty in [0, 1].

    Parameters
    ----------
    flags : Dict[str, int]
        Mapping rule_name -> 0/1.

    Returns
    -------
    float
        Ontology penalty, clipped to [0, 1].
    """
    penalty = 0.0
    for rule_name, fired in flags.items():
        if fired:
            penalty += RULE_WEIGHTS.get(rule_name, 0.0)
    # Clip to [0, 1]
    return float(max(0.0, min(1.0, penalty)))


def compute_ontology_penalty(row: pd.Series) -> float:
    """
    Compute a scalar ontology penalty in [0, 1] for a single patient row.

    The penalty is a weighted sum over all fired rules, clipped to [0, 1].
    This function is intentionally simple so it can be used independently
    in notebooks or tests.

    Parameters
    ----------
    row : pd.Series
        Single patient record from the clinical feature table.

    Returns
    -------
    float
        Ontology penalty in [0, 1].
    """
    flags = _evaluate_rules(row)
    return _penalty_from_flags(flags)


def apply_ontology_rules(
    df: pd.DataFrame,
    y: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Dict[str, int]]]:
    """
    Apply ontology rules to an entire cohort DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least the columns used by the rules.
    y : np.ndarray, optional
        Optional binary target array (0/1) aligned with df rows. If provided,
        rule-level stats will include how often rules fired on positive cases.

    Returns
    -------
    penalties : np.ndarray
        Array of shape (n_samples,) with ontology penalties in [0, 1].
    rule_stats : Dict[str, Dict[str, int]]
        Mapping rule_name -> {"fired": int, "fired_positive": int}
    """
    penalties: List[float] = []
    rule_stats: Dict[str, Dict[str, int]] = {
        name: {"fired": 0, "fired_positive": 0} for name in RULE_WEIGHTS.keys()
    }

    # Ensure y is a simple 1D array if provided
    y_array: Optional[np.ndarray] = None
    if y is not None:
        y_array = np.asarray(y).astype(int)

    for idx, (_, row) in enumerate(df.iterrows()):
        flags = _evaluate_rules(row)
        penalty = _penalty_from_flags(flags)

        for rule_name, fired in flags.items():
            if fired:
                rule_stats[rule_name]["fired"] += 1
                if y_array is not None and y_array[idx] == 1:
                    rule_stats[rule_name]["fired_positive"] += 1

        penalties.append(penalty)

    return np.asarray(penalties, dtype=float), rule_stats


def combine_scores(
    ml_scores: np.ndarray,
    ontology_penalties: np.ndarray,
    alpha: float = 0.7,
    beta: float = 0.3,
    normalize_ml: bool = True,
) -> np.ndarray:
    """
    Combine ML anomaly scores with ontology penalties.

    Parameters
    ----------
    ml_scores : np.ndarray
        Raw anomaly scores from the ML model (larger = more anomalous).
    ontology_penalties : np.ndarray
        Penalties from apply_ontology_rules(), in [0, 1].
    alpha : float, default 0.7
        Weight for the ML component.
    beta : float, default 0.3
        Weight for the ontology component.
    normalize_ml : bool, default True
        If True, min-max normalise ml_scores to [0, 1] before combining.

    Returns
    -------
    np.ndarray
        Combined scores of shape (n_samples,).
    """
    ml_scores = np.asarray(ml_scores, dtype=float)
    ontology_penalties = np.asarray(ontology_penalties, dtype=float)

    if normalize_ml:
        ml_min = float(ml_scores.min())
        ml_max = float(ml_scores.max())
        if ml_max > ml_min:
            ml_norm = (ml_scores - ml_min) / (ml_max - ml_min)
        else:
            # Degenerate case: constant scores
            ml_norm = np.zeros_like(ml_scores)
    else:
        ml_norm = ml_scores

    combined = alpha * ml_norm + beta * ontology_penalties
    return combined
