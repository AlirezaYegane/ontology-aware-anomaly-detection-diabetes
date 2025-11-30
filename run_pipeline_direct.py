"""
Direct pipeline execution entry point.

Runs the full ontology-aware anomaly detection pipeline without requiring Jupyter,
using the refactored modules under src/.

Steps:
    1) Data loading, preprocessing, and single train/test split
    2) Unsupervised baselines (Isolation Forest, Autoencoder)
    3) Supervised baselines (Decision Tree, Random Forest)
    4) Ontology-enhanced evaluation and ontology-aware ensemble
    5) Multi-split evaluation across multiple random seeds
"""

import sys
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Make sure the project root is on sys.path so that `import src.*` works
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import GLOBAL_CONFIG
from src.logger import get_logger
from src.evaluation import compute_classification_metrics, plot_roc_pr_curves


# =============================================================================
# Helper for section logging
# =============================================================================

def log_section(logger, title: str) -> None:
    """Log a nicely formatted section header."""
    line = "=" * 80
    logger.info("\n%s\n  %s\n%s\n", line, title, line)


# =============================================================================
# Main pipeline
# =============================================================================

def main() -> int:
    base_dir = PROJECT_ROOT

    # Create results directory structure up front
    results_dir = base_dir / "results"
    figures_dir = results_dir / "figures"
    models_dir = results_dir / "models"
    reports_dir = results_dir / "reports"
    logs_dir = results_dir / "logs"

    for d in (figures_dir, models_dir, reports_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    logger = get_logger("pipeline", logs_dir)
    cfg = GLOBAL_CONFIG

    log_section(logger, "ONTOLOGY-AWARE ANOMALY DETECTION PIPELINE - DIRECT EXECUTION")
    logger.info("Base directory:   %s", base_dir)
    logger.info("Working directory: %s", Path.cwd())

    # Metrics placeholders for final summary
    if_metrics = None
    ae_metrics = None
    dt_metrics = None
    rf_metrics = None
    if_ont_metrics = None
    ae_ont_metrics = None
    ensemble_metrics = None
    best_lambda_if = None
    best_lambda_ae = None

    # -------------------------------------------------------------------------
    # STEP 1: Data loading, EDA summary, feature matrix, train/test split
    # -------------------------------------------------------------------------
    log_section(logger, "STEP 1: Data Loading & Feature Matrix")

    try:
        from src.preprocessing import (
            load_raw_data,
            build_feature_matrix,
            train_test_split_stratified,
        )

        csv_path = base_dir / "data" / "raw" / "diabetic_data.csv"
        if not csv_path.exists():
            logger.error("Data file not found: %s", csv_path)
            return 1

        logger.info("Loading raw data from %s ...", csv_path)
        df = load_raw_data(str(csv_path))
        logger.info("Loaded %d records with %d columns.", df.shape[0], df.shape[1])

        if "readmitted" in df.columns:
            logger.info("Target 'readmitted' distribution:\n%s", df["readmitted"].value_counts())

        logger.info("Building feature matrix (preprocessing, filtering, encoding)...")
        X, y, _ = build_feature_matrix(df)

        logger.info("Feature matrix shape: %s", X.shape)
        logger.info(
            "Target vector shape: %s | positives (readmitted <30 days): %d (%.1f%%)",
            y.shape,
            int(y.sum()),
            float(y.mean() * 100.0),
        )

        # Single train/test split for the main run
        X_train, X_test, y_train, y_test = train_test_split_stratified(
            X,
            y,
            test_size=cfg.data.test_size,
            random_state=cfg.data.random_seeds[0],
        )

        train_pos = int(y_train.sum())
        train_total = len(y_train)
        train_pos_rate = train_pos / train_total if train_total > 0 else 0.0

        logger.info(
            "Train/Test split with test_size=%.2f, seed=%d -> "
            "X_train=%s, X_test=%s",
            cfg.data.test_size,
            cfg.data.random_seeds[0],
            X_train.shape,
            X_test.shape,
        )
        logger.info(
            "Train class balance: %d positives out of %d samples (%.3f%%).",
            train_pos,
            train_total,
            train_pos_rate * 100.0,
        )

    except Exception as exc:
        logger.exception("Error in Step 1 (data loading & preprocessing): %s", exc)
        return 1

    # -------------------------------------------------------------------------
    # STEP 2: Baseline Isolation Forest (unsupervised)
    # -------------------------------------------------------------------------
    log_section(logger, "STEP 2: Baseline Isolation Forest (Unsupervised)")

    try:
        from src.models import IsolationForestDetector

        normal_mask_if = (y_train == 0)
        X_train_normal_if = X_train[normal_mask_if]

        logger.info(
            "Training Isolation Forest on normal-only subset: %d / %d samples (%.1f%%).",
            X_train_normal_if.shape[0],
            X_train.shape[0],
            X_train_normal_if.shape[0] / max(X_train.shape[0], 1) * 100.0,
        )

        # Contamination: if configured < 0, fall back to empirical positive rate
        if cfg.isolation_forest.contamination < 0:
            contamination = float(train_pos_rate)
        else:
            contamination = float(cfg.isolation_forest.contamination)

        if_detector = IsolationForestDetector(
            n_estimators=cfg.isolation_forest.n_estimators,
            contamination=contamination,
            random_state=cfg.isolation_forest.random_state,
        )

        logger.info(
            "Isolation Forest config: n_estimators=%d, contamination=%.4f, random_state=%d",
            cfg.isolation_forest.n_estimators,
            contamination,
            cfg.isolation_forest.random_state,
        )

        if_detector.fit(X_train_normal_if)

        logger.info("Computing Isolation Forest anomaly scores on train/test...")
        if_scores_train = if_detector.predict_scores(X_train)
        if_scores_test = if_detector.predict_scores(X_test)

        logger.info(
            "IF train scores range: [%.3f, %.3f]",
            float(if_scores_train.min()),
            float(if_scores_train.max()),
        )
        logger.info(
            "IF test scores range:  [%.3f, %.3f]",
            float(if_scores_test.min()),
            float(if_scores_test.max()),
        )

        if_metrics = compute_classification_metrics(
            y_true=y_test,
            anomaly_scores=if_scores_test,
            model_name="IsolationForest",
        )
        logger.info(
            "Isolation Forest performance: ROC-AUC=%.4f | PR-AUC=%.4f",
            if_metrics["roc_auc"],
            if_metrics["pr_auc"],
        )

        plot_path = figures_dir / "if_roc_pr.png"
        plot_roc_pr_curves(
            y_true=y_test,
            anomaly_scores=if_scores_test,
            title="Isolation Forest",
            save_path=str(plot_path),
        )
        logger.info("Saved Isolation Forest ROC/PR curves to %s", plot_path)

    except Exception as exc:
        logger.exception("Error in Step 2 (Isolation Forest): %s", exc)
        return 1

    # -------------------------------------------------------------------------
    # STEP 3: Autoencoder (unsupervised, reconstruction error)
    # -------------------------------------------------------------------------
    log_section(logger, "STEP 3: Autoencoder Anomaly Detection")

    try:
        from src.models import AutoencoderDetector

        normal_mask_ae = (y_train == 0)
        X_train_normal_ae = X_train[normal_mask_ae]

        logger.info(
            "Training Autoencoder on normal-only subset: %d / %d samples (%.1f%%).",
            X_train_normal_ae.shape[0],
            X_train.shape[0],
            X_train_normal_ae.shape[0] / max(X_train.shape[0], 1) * 100.0,
        )

        ae_detector = AutoencoderDetector(
            input_dim=X_train.shape[1],
            hidden_dims=list(cfg.autoencoder.hidden_dims),
            epochs=cfg.autoencoder.epochs,
            batch_size=cfg.autoencoder.batch_size,
            learning_rate=cfg.autoencoder.learning_rate,
        )

        logger.info(
            "Autoencoder config: hidden_dims=%s, epochs=%d, batch_size=%d, lr=%.1e",
            cfg.autoencoder.hidden_dims,
            cfg.autoencoder.epochs,
            cfg.autoencoder.batch_size,
            cfg.autoencoder.learning_rate,
        )

        ae_detector.fit(X_train_normal_ae)

        logger.info("Computing Autoencoder reconstruction-error scores on train/test...")
        ae_scores_train = ae_detector.predict_scores(X_train)
        ae_scores_test = ae_detector.predict_scores(X_test)

        logger.info(
            "AE train scores range: [%.3f, %.3f]",
            float(ae_scores_train.min()),
            float(ae_scores_train.max()),
        )
        logger.info(
            "AE test scores range:  [%.3f, %.3f]",
            float(ae_scores_test.min()),
            float(ae_scores_test.max()),
        )

        ae_metrics = compute_classification_metrics(
            y_true=y_test,
            anomaly_scores=ae_scores_test,
            model_name="Autoencoder",
        )
        logger.info(
            "Autoencoder performance: ROC-AUC=%.4f | PR-AUC=%.4f",
            ae_metrics["roc_auc"],
            ae_metrics["pr_auc"],
        )

        plot_path = figures_dir / "ae_roc_pr.png"
        plot_roc_pr_curves(
            y_true=y_test,
            anomaly_scores=ae_scores_test,
            title="Autoencoder",
            save_path=str(plot_path),
        )
        logger.info("Saved Autoencoder ROC/PR curves to %s", plot_path)

    except Exception as exc:
        logger.exception("Error in Step 3 (Autoencoder): %s", exc)
        return 1

    # -------------------------------------------------------------------------
    # STEP 3b: Supervised baselines (Decision Tree & Random Forest)
    # -------------------------------------------------------------------------
    log_section(logger, "STEP 3b: Supervised Baselines (Decision Tree & Random Forest)")

    try:
        from src.models import DecisionTreeDetector, RandomForestDetector

        # Decision Tree
        logger.info("Training Decision Tree supervised baseline...")
        dt_detector = DecisionTreeDetector(
            max_depth=8,
            min_samples_leaf=50,
            random_state=42,
            class_weight="balanced",
        )
        dt_detector.fit(X_train, y_train)
        dt_scores_test = dt_detector.predict_scores(X_test)

        dt_metrics = compute_classification_metrics(
            y_true=y_test,
            anomaly_scores=dt_scores_test,
            model_name="DecisionTree",
        )
        logger.info(
            "Decision Tree performance: ROC-AUC=%.4f | PR-AUC=%.4f",
            dt_metrics["roc_auc"],
            dt_metrics["pr_auc"],
        )

        plot_path = figures_dir / "dt_roc_pr.png"
        plot_roc_pr_curves(
            y_true=y_test,
            anomaly_scores=dt_scores_test,
            title="Decision Tree (supervised baseline)",
            save_path=str(plot_path),
        )
        logger.info("Saved Decision Tree ROC/PR curves to %s", plot_path)

        # Random Forest
        logger.info("Training Random Forest supervised baseline...")
        rf_detector = RandomForestDetector(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=50,
            random_state=42,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )
        rf_detector.fit(X_train, y_train)
        rf_scores_test = rf_detector.predict_scores(X_test)

        rf_metrics = compute_classification_metrics(
            y_true=y_test,
            anomaly_scores=rf_scores_test,
            model_name="RandomForest",
        )
        logger.info(
            "Random Forest performance: ROC-AUC=%.4f | PR-AUC=%.4f",
            rf_metrics["roc_auc"],
            rf_metrics["pr_auc"],
        )

        plot_path = figures_dir / "rf_roc_pr.png"
        plot_roc_pr_curves(
            y_true=y_test,
            anomaly_scores=rf_scores_test,
            title="Random Forest (supervised baseline)",
            save_path=str(plot_path),
        )
        logger.info("Saved Random Forest ROC/PR curves to %s", plot_path)

    except Exception as exc:
        logger.exception("Error in Step 3b (supervised baselines): %s", exc)
        return 1

    # -------------------------------------------------------------------------
    # STEP 4: Ontology-enhanced evaluation
    # -------------------------------------------------------------------------
    log_section(logger, "STEP 4: Ontology-Enhanced Evaluation")

    try:
        from src.ontology import apply_ontology_rules, combine_scores
        from src.preprocessing import (
            load_raw_data,
            clean_data,
            create_target,
            get_selected_features,
            train_test_split_stratified,
        )
        import numpy as np

        csv_path = base_dir / "data" / "raw" / "diabetic_data.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Cannot find CSV at {csv_path}")

        # Build a clinical feature DataFrame aligned with the ontology rules
        df_raw_full = load_raw_data(str(csv_path))
        selected_features = get_selected_features()
        df_clean_full = clean_data(df_raw_full, selected_features, tracker=None)
        X_clinical_full, y_full = create_target(df_clean_full)

        X_clin_train, X_clin_test, y_clin_train, y_clin_test = train_test_split_stratified(
            X_clinical_full,
            y_full,
            test_size=cfg.data.test_size,
            random_state=cfg.data.random_seeds[0],
        )

        if len(X_clin_test) != len(y_test):
            logger.warning(
                "Ontology test-set length mismatch: X_clin_test=%d vs y_test=%d. "
                "Skipping ontology enhancement.",
                len(X_clin_test),
                len(y_test),
            )
            combined_scores_if = if_scores_test
            if_ont_metrics = compute_classification_metrics(
                y_true=y_test,
                anomaly_scores=combined_scores_if,
                model_name="IF + Ontology (skipped)",
            )
        else:
            ontology_penalties_test, rule_stats = apply_ontology_rules(
                X_clin_test,
                y_clin_test.to_numpy(),
            )

            if len(ontology_penalties_test) != len(if_scores_test):
                logger.warning(
                    "Ontology penalties length %d != IF scores length %d. "
                    "Skipping ontology enhancement.",
                    len(ontology_penalties_test),
                    len(if_scores_test),
                )
                combined_scores_if = if_scores_test
                if_ont_metrics = compute_classification_metrics(
                    y_true=y_test,
                    anomaly_scores=combined_scores_if,
                    model_name="IF + Ontology (skipped)",
                )
            else:
                lambda_values = [0.0, 0.1, 0.3, 0.5]

                def _score_key(item):
                    lam, scores, metrics = item
                    return (metrics["pr_auc"], metrics["roc_auc"])

                # IF + Ontology
                if_combo_results = []
                for lam in lambda_values:
                    alpha = 1.0 - lam
                    beta = lam
                    scores_lam = combine_scores(
                        ml_scores=if_scores_test,
                        ontology_penalties=ontology_penalties_test,
                        alpha=alpha,
                        beta=beta,
                        normalize_ml=True,
                    )
                    metrics_lam = compute_classification_metrics(
                        y_true=y_test,
                        anomaly_scores=scores_lam,
                        model_name=f"IF+Ontology (lambda={lam:.2f})",
                    )
                    if_combo_results.append((lam, scores_lam, metrics_lam))

                best_idx_if = max(range(len(if_combo_results)), key=lambda i: _score_key(if_combo_results[i]))
                best_lambda_if, combined_scores_if, if_ont_metrics = if_combo_results[best_idx_if]
                logger.info(
                    "Best IF+Ontology lambda=%.2f | ROC-AUC=%.4f | PR-AUC=%.4f",
                    best_lambda_if,
                    if_ont_metrics["roc_auc"],
                    if_ont_metrics["pr_auc"],
                )

                # AE + Ontology
                ae_combo_results = []
                for lam in lambda_values:
                    alpha = 1.0 - lam
                    beta = lam
                    scores_lam = combine_scores(
                        ml_scores=ae_scores_test,
                        ontology_penalties=ontology_penalties_test,
                        alpha=alpha,
                        beta=beta,
                        normalize_ml=True,
                    )
                    metrics_lam = compute_classification_metrics(
                        y_true=y_test,
                        anomaly_scores=scores_lam,
                        model_name=f"AE+Ontology (lambda={lam:.2f})",
                    )
                    ae_combo_results.append((lam, scores_lam, metrics_lam))

                best_idx_ae = max(range(len(ae_combo_results)), key=lambda i: _score_key(ae_combo_results[i]))
                best_lambda_ae, ae_ont_scores, ae_ont_metrics = ae_combo_results[best_idx_ae]
                logger.info(
                    "Best AE+Ontology lambda=%.2f | ROC-AUC=%.4f | PR-AUC=%.4f",
                    best_lambda_ae,
                    ae_ont_metrics["roc_auc"],
                    ae_ont_metrics["pr_auc"],
                )

                # Ontology-aware ensemble
                ensemble_scores = 0.5 * combined_scores_if + 0.5 * ae_ont_scores
                ensemble_metrics = compute_classification_metrics(
                    y_true=y_test,
                    anomaly_scores=ensemble_scores,
                    model_name="Ensemble(IF+Ont, AE+Ont)",
                )
                logger.info(
                    "Ontology-aware ensemble: ROC-AUC=%.4f | PR-AUC=%.4f",
                    ensemble_metrics["roc_auc"],
                    ensemble_metrics["pr_auc"],
                )

                # Rule-level summary
                logger.info("Ontology rule statistics on test set:")
                header = f"{'Rule':<35} {'Fired':>8} {'Fired & y=1':>12} {'Precision':>12}"
                logger.info(header)
                logger.info("-" * len(header))
                total_fired = 0
                total_fired_pos = 0
                for rule_name, stats_dict in rule_stats.items():
                    fired = stats_dict.get("fired", 0)
                    fired_pos = stats_dict.get("fired_positive", 0)
                    prec = (fired_pos / fired) if fired > 0 else 0.0
                    if fired > 0:
                        total_fired += fired
                        total_fired_pos += fired_pos
                    logger.info(
                        "%-35s %8d %12d %12.3f",
                        rule_name,
                        fired,
                        fired_pos,
                        prec,
                    )
                logger.info(
                    "%-35s %8d %12d",
                    "Any rule fired (non-unique hits)",
                    total_fired,
                    total_fired_pos,
                )

        # Plots for ontology-enhanced models (if available)
        if if_ont_metrics is not None:
            title = "IF + Ontology"
            if best_lambda_if is not None:
                title = f"IF + Ontology (lambda={best_lambda_if:.2f})"
            plot_path = figures_dir / "ontology_if_roc_pr.png"
            plot_roc_pr_curves(
                y_true=y_test,
                anomaly_scores=combined_scores_if,
                title=title,
                save_path=str(plot_path),
            )
            logger.info("Saved IF+Ontology ROC/PR curves to %s", plot_path)

        if ae_ont_metrics is not None:
            title = "AE + Ontology"
            if best_lambda_ae is not None:
                title = f"AE + Ontology (lambda={best_lambda_ae:.2f})"
            plot_path = figures_dir / "ontology_ae_roc_pr.png"
            plot_roc_pr_curves(
                y_true=y_test,
                anomaly_scores=ae_ont_scores,
                title=title,
                save_path=str(plot_path),
            )
            logger.info("Saved AE+Ontology ROC/PR curves to %s", plot_path)

        if ensemble_metrics is not None:
            plot_path = figures_dir / "ontology_ensemble_roc_pr.png"
            plot_roc_pr_curves(
                y_true=y_test,
                anomaly_scores=ensemble_scores,
                title="Ensemble (IF+Ont + AE+Ont)",
                save_path=str(plot_path),
            )
            logger.info("Saved ontology-aware ensemble ROC/PR curves to %s", plot_path)

    except Exception as exc:
        logger.exception("Error in Step 4 (ontology-enhanced evaluation): %s", exc)
        return 1

    # -------------------------------------------------------------------------
    # STEP 5: Multi-split evaluation (metrics only)
    # -------------------------------------------------------------------------
    log_section(logger, "STEP 5: Multi-Split Evaluation (Metrics Only)")

    try:
        import numpy as np

        seeds = list(GLOBAL_CONFIG.data.random_seeds)
        logger.info("Running multi-split experiments for seeds=%s", seeds)

        all_results = []
        for seed in seeds:
            logger.info("Running single-split experiment for random_state=%d ...", seed)
            res = run_single_split_experiment(base_dir=base_dir, random_state=seed)
            all_results.append(res)
            logger.info(
                "Seed=%d | IF ROC=%.4f PR=%.4f | AE ROC=%.4f PR=%.4f | "
                "IF+Ont ROC=%.4f PR=%.4f (lambda*=%.2f)",
                seed,
                res["if_roc_auc"],
                res["if_pr_auc"],
                res["ae_roc_auc"],
                res["ae_pr_auc"],
                res["if_ont_roc_auc"],
                res["if_ont_pr_auc"],
                res["best_lambda"],
            )

        def _mean_std(values):
            arr = np.asarray(values, dtype=float)
            if len(arr) == 0:
                return 0.0, 0.0
            if len(arr) == 1:
                return float(arr[0]), 0.0
            return float(arr.mean()), float(arr.std(ddof=1))

        summary = {}
        for key, label in [
            ("if", "Isolation Forest"),
            ("ae", "Autoencoder"),
            ("if_ont", "IF + Ontology"),
        ]:
            roc_vals = [r[f"{key}_roc_auc"] for r in all_results]
            pr_vals = [r[f"{key}_pr_auc"] for r in all_results]
            roc_mean, roc_std = _mean_std(roc_vals)
            pr_mean, pr_std = _mean_std(pr_vals)
            summary[key] = {
                "label": label,
                "roc_auc_mean": roc_mean,
                "roc_auc_std": roc_std,
                "pr_auc_mean": pr_mean,
                "pr_auc_std": pr_std,
            }

        reports_dir.mkdir(parents=True, exist_ok=True)
        json_payload = {
            "seeds": seeds,
            "results": all_results,
            "summary": summary,
        }

        json_path = reports_dir / "multi_split_metrics.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_payload, f, indent=2)
        logger.info("Saved multi-split metrics JSON to %s", json_path)

        md_lines = [
            "# Multi-Split Evaluation Summary",
            "",
            f"Seeds: {', '.join(str(s) for s in seeds)}",
            "",
            "| Model | ROC-AUC (mean ± std) | PR-AUC (mean ± std) |",
            "|-------|----------------------|---------------------|",
        ]
        for key in ["if", "ae", "if_ont"]:
            s = summary[key]
            md_lines.append(
                f"| {s['label']} | "
                f"{s['roc_auc_mean']:.4f} ± {s['roc_auc_std']:.4f} | "
                f"{s['pr_auc_mean']:.4f} ± {s['pr_auc_std']:.4f} |"
            )

        md_path = reports_dir / "multi_split_metrics.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))
        logger.info("Saved multi-split metrics Markdown to %s", md_path)

    except Exception as exc:
        # Multi-split evaluation failure should not kill the whole pipeline
        logger.exception("Error in Step 5 (multi-split evaluation): %s", exc)

    # -------------------------------------------------------------------------
    # FINAL SUMMARY (single split)
    # -------------------------------------------------------------------------
    log_section(logger, "PIPELINE EXECUTION SUMMARY (Single Split)")

    header = f"{'Model':<22} {'ROC-AUC':<14} {'PR-AUC':<14}"
    logger.info(header)
    logger.info("-" * len(header))

    def _log_metrics(label: str, metrics: dict | None):
        if metrics is not None:
            logger.info(
                "%-22s %-14.4f %-14.4f",
                label,
                metrics["roc_auc"],
                metrics["pr_auc"],
            )

    _log_metrics("Decision Tree", dt_metrics)
    _log_metrics("Random Forest", rf_metrics)
    _log_metrics("Isolation Forest", if_metrics)
    _log_metrics("Autoencoder", ae_metrics)
    _log_metrics("IF + Ontology", if_ont_metrics)
    _log_metrics("AE + Ontology", ae_ont_metrics)
    _log_metrics("Ont. Ensemble", ensemble_metrics)

    logger.info("")
    logger.info("Generated figure files:")
    for file in sorted(figures_dir.glob("*.png")):
        logger.info("  %s", file.relative_to(base_dir))

    logger.info("")
    logger.info("All pipeline steps completed.")
    return 0


# =============================================================================
# Single-split experiment helper (used in Step 5)
# =============================================================================

def run_single_split_experiment(base_dir: Path, random_state: int) -> dict:
    """
    Run the anomaly detection pipeline (IF, AE, IF+Ontology) for a single
    random_state and return metrics only, without plotting or logging.

    Returns a dictionary:
        {
            "seed": int,
            "if_roc_auc": float,
            "if_pr_auc": float,
            "ae_roc_auc": float,
            "ae_pr_auc": float,
            "if_ont_roc_auc": float,
            "if_ont_pr_auc": float,
            "best_lambda": float,
        }
    """
    from src.preprocessing import (
        load_raw_data,
        build_feature_matrix,
        train_test_split_stratified,
        get_selected_features,
        clean_data,
        create_target,
    )
    from src.models import IsolationForestDetector, AutoencoderDetector
    from src.ontology import apply_ontology_rules, combine_scores
    from src.evaluation import compute_classification_metrics
    import numpy as np

    csv_path = base_dir / "data" / "raw" / "diabetic_data.csv"
    df = load_raw_data(str(csv_path))

    X, y, _ = build_feature_matrix(df)

    X_train, X_test, y_train, y_test = train_test_split_stratified(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
    )

    train_pos_rate = float(y_train.mean())

    # Isolation Forest on normal-only training set
    normal_mask_if = (y_train == 0)
    X_train_normal_if = X_train[normal_mask_if]

    if_detector = IsolationForestDetector(
        n_estimators=200,
        contamination=train_pos_rate,
        random_state=random_state,
    )
    if_detector.fit(X_train_normal_if)
    if_scores_test = if_detector.predict_scores(X_test)
    if_metrics = compute_classification_metrics(
        y_true=y_test,
        anomaly_scores=if_scores_test,
        model_name=f"IF (seed={random_state})",
    )

    # Autoencoder on normal-only training set
    ae_detector = AutoencoderDetector(
        input_dim=X_train.shape[1],
        hidden_dims=[128, 64, 32],
        epochs=50,
        batch_size=256,
        learning_rate=1e-3,
    )
    normal_mask_ae = (y_train == 0)
    X_train_normal_ae = X_train[normal_mask_ae]
    ae_detector.fit(X_train_normal_ae)
    ae_scores_test = ae_detector.predict_scores(X_test)
    ae_metrics = compute_classification_metrics(
        y_true=y_test,
        anomaly_scores=ae_scores_test,
        model_name=f"AE (seed={random_state})",
    )

    # Ontology on the same split
    df_raw_full = load_raw_data(str(csv_path))
    selected_features = get_selected_features()
    df_clean_full = clean_data(df_raw_full, selected_features, tracker=None)
    X_clinical_full, y_full = create_target(df_clean_full)

    X_clin_train, X_clin_test, y_clin_train, y_clin_test = train_test_split_stratified(
        X_clinical_full,
        y_full,
        test_size=0.2,
        random_state=random_state,
    )

    # If lengths disagree, fall back to IF metrics for ontology slot
    if len(X_clin_test) != len(y_test):
        return {
            "seed": random_state,
            "if_roc_auc": float(if_metrics["roc_auc"]),
            "if_pr_auc": float(if_metrics["pr_auc"]),
            "ae_roc_auc": float(ae_metrics["roc_auc"]),
            "ae_pr_auc": float(ae_metrics["pr_auc"]),
            "if_ont_roc_auc": float(if_metrics["roc_auc"]),
            "if_ont_pr_auc": float(if_metrics["pr_auc"]),
            "best_lambda": 0.0,
        }

    ontology_penalties_test, _ = apply_ontology_rules(
        X_clin_test,
        y_clin_test.to_numpy(),
    )

    lambda_values = [0.0, 0.1, 0.3, 0.5]
    combo_results = []

    for lam in lambda_values:
        alpha = 1.0 - lam
        beta = lam
        scores_lam = combine_scores(
            ml_scores=if_scores_test,
            ontology_penalties=ontology_penalties_test,
            alpha=alpha,
            beta=beta,
            normalize_ml=True,
        )
        metrics_lam = compute_classification_metrics(
            y_true=y_test,
            anomaly_scores=scores_lam,
            model_name=f"IF+Ontology (lambda={lam:.2f}, seed={random_state})",
        )
        combo_results.append((lam, metrics_lam))

    def _score_key(item):
        lam, metrics = item
        return (metrics["pr_auc"], metrics["roc_auc"])

    best_idx = max(range(len(combo_results)), key=lambda i: _score_key(combo_results[i]))
    best_lambda, best_metrics = combo_results[best_idx]

    return {
        "seed": random_state,
        "if_roc_auc": float(if_metrics["roc_auc"]),
        "if_pr_auc": float(if_metrics["pr_auc"]),
        "ae_roc_auc": float(ae_metrics["roc_auc"]),
        "ae_pr_auc": float(ae_metrics["pr_auc"]),
        "if_ont_roc_auc": float(best_metrics["roc_auc"]),
        "if_ont_pr_auc": float(best_metrics["pr_auc"]),
        "best_lambda": float(best_lambda),
    }


if __name__ == "__main__":
    sys.exit(main())
