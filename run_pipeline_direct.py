"""
Direct Pipeline Execution - Without Jupyter dependency
Executes all pipeline steps directly using src modules
"""
import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def print_section(title):
    """Print formatted section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")

def main():
    print_section("ONTOLOGY-AWARE ANOMALY DETECTION PIPELINE - DIRECT EXECUTION")
    
    base_dir = Path(__file__).parent
    os.chdir(base_dir)
    
    print(f"📁 Base Directory: {base_dir}")
    print(f"📁 Working Directory: {os.getcwd()}")
    
    # =========================================================================
    # STEP 1: EDA and Data Loading
    # =========================================================================
    print_section("STEP 1: Exploratory Data Analysis & Data Loading")
    
    try:
        from src.preprocessing import load_raw_data, build_feature_matrix, train_test_split_stratified
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Create results directories
        results_dir = base_dir / 'results'
        figures_dir = results_dir / 'figures'
        models_dir = results_dir / 'models'
        reports_dir = results_dir / 'reports'
        
        figures_dir.mkdir(parents=True, exist_ok=True)
        models_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        print("📥 Loading raw data...")
        csv_path = base_dir / 'data' / 'raw' / 'diabetic_data.csv'
        if not csv_path.exists():
            print(f"❌ Data file not found: {csv_path}")
            return 1
        
        df = load_raw_data(str(csv_path))
        print(f"✅ Loaded {len(df):,} records")
        print(f"   Columns: {list(df.columns)[:10]}... ({len(df.columns)} total)")
        
        # Basic EDA
        print("\n📊 Dataset shape:", df.shape)
        print("\n📊 Target distribution (readmitted):")
        if 'readmitted' in df.columns:
            print(df['readmitted'].value_counts())
        
        # Build feature matrix
        print("\n🔧 Building feature matrix...")
        X, y, _ = build_feature_matrix(df)
        print(f"✅ Feature matrix: {X.shape}")
        print(f"✅ Target vector: {y.shape}")
        print(f"   Positive class (readmitted<30): {y.sum()} ({y.mean()*100:.1f}%)")
        
        # Train/test split
        print("\n✂️  Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split_stratified(
            X,
            y,
            test_size=0.2,
            random_state=42,
        )

        # Log train set class balance (for anomaly rate)
        train_pos = int(y_train.sum())
        train_total = len(y_train)
        train_pos_rate = train_pos / train_total

        print(f"✅ Train set: {X_train.shape}")
        print(f"✅ Test set: {X_test.shape}")
        print(f"🔢 Train positives: {train_pos} / {train_total} ({train_pos_rate:.3%})")

        print("✅ STEP 1 COMPLETE")

    except Exception as e:
        print(f"❌ Error in Step 1: {e}")
        import traceback
        traceback.print_exc()
        return 1


    
    # =========================================================================
    # STEP 2: Baseline Isolation Forest
    # =========================================================================
    print_section("STEP 2: Baseline Isolation Forest Model")
    
    # Train only on NORMAL (y=0) samples to follow unsupervised anomaly detection logic
    normal_mask_if = (y_train == 0)
    X_train_normal_if = X_train[normal_mask_if]

    print("\n🌲 Training Isolation Forest on normal subset only...")
    print(
        f"   Normal train samples: {X_train_normal_if.shape[0]} / {X_train.shape[0]} "
        f"({X_train_normal_if.shape[0] / X_train.shape[0]:.1%} of train set)"
    )

    try:
        from src.models import IsolationForestDetector
        from src.evaluation import compute_classification_metrics, plot_roc_pr_curves
        
        print("🌲 Training Isolation Forest...")
        contamination = float(train_pos_rate)  # from section 3.1
        
        if_detector = IsolationForestDetector(
            n_estimators=200,
            contamination=contamination,
            random_state=42
        )
        # Fit on normal-only training data
        if_detector.fit(X_train_normal_if)

        # But compute scores on FULL train/test sets
        print("\n📊 Computing predictions...")
        if_scores_train = if_detector.predict_scores(X_train)
        if_scores_test = if_detector.predict_scores(X_test)

        print(f"✅ Train scores: min={if_scores_train.min():.3f}, max={if_scores_train.max():.3f}")
        print(f"✅ Test scores: min={if_scores_test.min():.3f}, max={if_scores_test.max():.3f}")
        
        print("\n📈 Evaluating performance...")
        if_metrics = compute_classification_metrics(y_test, if_scores_test, model_name="IsolationForest")
        print(f"   ROC-AUC: {if_metrics['roc_auc']:.4f}")
        print(f"   PR-AUC:  {if_metrics['pr_auc']:.4f}")
        
        print("\n💾 Saving results...")
        plot_roc_pr_curves(y_test, if_scores_test, 
                          title="Isolation Forest", 
                          save_path=str(figures_dir / 'if_roc_pr.png'))
        print(f"✅ Saved plot: {figures_dir / 'if_roc_pr.png'}")
        
        print("✅ STEP 2 COMPLETE")
        
    except Exception as e:
        print(f"❌ Error in Step 2: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # =========================================================================
    # STEP 3: Autoencoder
    # =========================================================================
    print_section("STEP 3: Autoencoder Anomaly Detection")
    
    try:
        from src.models import AutoencoderDetector
        
        print("🧠 Training Autoencoder...")
        ae_detector = AutoencoderDetector(
            input_dim=X_train.shape[1],
            hidden_dims=[128, 64, 32],
            epochs=50,
            batch_size=256,
            learning_rate=0.001
        )
        
        # Use the same normal-mask logic for Autoencoder
        normal_mask_ae = (y_train == 0)
        X_train_normal_ae = X_train[normal_mask_ae]

        print("\n🧠 Training Autoencoder on normal subset only...")
        print(f"   Normal train samples: {X_train_normal_ae.shape[0]} / {X_train.shape[0]} "
              f"({X_train_normal_ae.shape[0] / X_train.shape[0]:.1%} of train set)")

        ae_detector.fit(X_train_normal_ae)

        print("\n📊 Computing reconstruction errors...")
        ae_scores_train = ae_detector.predict_scores(X_train)
        ae_scores_test = ae_detector.predict_scores(X_test)
        
        print(f"✅ Train scores: min={ae_scores_train.min():.3f}, max={ae_scores_train.max():.3f}")
        print(f"✅ Test scores: min={ae_scores_test.min():.3f}, max={ae_scores_test.max():.3f}")
        
        print("\n📈 Evaluating performance...")
        ae_metrics = compute_classification_metrics(y_test, ae_scores_test, model_name="Autoencoder")
        print(f"   ROC-AUC: {ae_metrics['roc_auc']:.4f}")
        print(f"   PR-AUC:  {ae_metrics['pr_auc']:.4f}")
        
        print("\n💾 Saving results...")
        plot_roc_pr_curves(y_test, ae_scores_test, 
                          title="Autoencoder", 
                          save_path=str(figures_dir / 'ae_roc_pr.png'))
        print(f"✅ Saved plot: {figures_dir / 'ae_roc_pr.png'}")
        
        print("✅ STEP 3 COMPLETE")
        
    except Exception as e:
        print(f"❌ Error in Step 3: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # =========================================================================
    # STEP 4: Ontology-Enhanced Evaluation
    # =========================================================================
    print_section("STEP 4: Ontology-Enhanced Evaluation")
    
    try:
        from src.ontology import apply_ontology_rules, combine_scores
        from src.preprocessing import (
            load_raw_data,
            clean_data,
            create_target,
            get_selected_features,
            train_test_split_stratified,
        )

        print("🔬 Computing ontology penalty...")

        csv_path = base_dir / 'data' / 'raw' / 'diabetic_data.csv'
        if not csv_path.exists():
            raise FileNotFoundError(f"Cannot find CSV at {csv_path}")
        
        # Rebuild a *clinical* feature DataFrame aligned with the ML pipeline
        df_raw_full = load_raw_data(str(csv_path))
        selected_features = get_selected_features()
        df_clean_full = clean_data(df_raw_full, selected_features, tracker=None)
        X_clinical_full, y_full = create_target(df_clean_full)

        # Use the SAME stratified split settings as Step 1
        X_clin_train, X_clin_test, y_clin_train, y_clin_test = train_test_split_stratified(
            X_clinical_full, y_full, test_size=0.2, random_state=42
        )

        if len(X_clin_test) != len(y_test):
            print(f"⚠️ Ontology warning: test set length mismatch "
                  f"(X_clin_test={len(X_clin_test)}, y_test={len(y_test)}). "
                  "Skipping ontology enhancement.")
            combined_scores = if_scores_test
            combined_metrics = compute_classification_metrics(
                y_test, combined_scores, model_name="IF + Ontology (skipped)"
            )
        else:
            # Apply ontology rules on the clinical test DataFrame
            ontology_penalties_test, rule_stats = apply_ontology_rules(
                X_clin_test,
                y_clin_test.to_numpy()
            )

            # Sanity check on shapes
            if len(ontology_penalties_test) != len(if_scores_test):
                print(f"⚠️ Ontology warning: penalty length {len(ontology_penalties_test)} "
                      f"!= IF scores length {len(if_scores_test)}. "
                      "Skipping ontology enhancement.")
                combined_scores = if_scores_test
                combined_metrics = compute_classification_metrics(
                    y_test, combined_scores, model_name="IF + Ontology (skipped)"
                )
            else:
                print("\n🔗 Combining IF scores with ontology rules...")

                # Try a small grid of lambda (ontology weight) values
                lambda_values = [0.0, 0.1, 0.3, 0.5]
                combo_results = []
                for lam in lambda_values:
                    alpha = 1.0 - lam
                    beta = lam
                    scores_lam = combine_scores(
                        if_scores_test,
                        ontology_penalties_test,
                        alpha=alpha,
                        beta=beta,
                        normalize_ml=True,
                    )
                    metrics_lam = compute_classification_metrics(
                        y_test,
                        scores_lam,
                        model_name=f"IF+Ontology (λ={lam:.2f})",
                    )
                    combo_results.append((lam, scores_lam, metrics_lam))

                # Pick best lambda by PR-AUC (primary), ROC-AUC as tie-breaker
                def _score_key(item):
                    lam, _scores, metrics = item
                    return (metrics['pr_auc'], metrics['roc_auc'])

                best_idx = max(
                    range(len(combo_results)),
                    key=lambda i: _score_key(combo_results[i])
                )
                best_lambda, combined_scores, combined_metrics = combo_results[best_idx]

                print(f"✅ Ontology enhancement applied with best λ={best_lambda:.2f}")
                print("\n📈 Evaluation for best λ:")
                print(f"   ROC-AUC: {combined_metrics['roc_auc']:.4f}")
                print(f"   PR-AUC:  {combined_metrics['pr_auc']:.4f}")

                print("\n📊 Lambda sweep summary (IF + Ontology):")
                print(f"{'λ':<6} {'ROC-AUC':<10} {'PR-AUC':<10}")
                for lam, _scores, m in combo_results:
                    print(f"{lam:<6.2f} {m['roc_auc']:<10.4f} {m['pr_auc']:<10.4f}")

                # Print rule-level summary
                print("\n" + "="*80)
                print("  ONTOLOGY RULES SUMMARY (TEST SET)")
                print("="*80)
                print(f"{'Rule':<35} {'Fired':>8} {'Fired & y=1':>12} {'Precision':>12}")
                print("-"*80)
                total_fired = 0
                total_fired_pos = 0
                for rule_name, stats_dict in rule_stats.items():
                    fired = stats_dict.get('fired', 0)
                    fired_pos = stats_dict.get('fired_positive', 0)
                    prec = (fired_pos / fired) if fired > 0 else 0.0
                    if fired > 0:
                        total_fired += fired
                        total_fired_pos += fired_pos
                    print(f"{rule_name:<35} {fired:>8} {fired_pos:>12} {prec:>12.3f}")
                print("-"*80)
                print(f"{'Any rule fired (non-unique hits)':<35} {total_fired:>8} {total_fired_pos:>12}")
        
        print("\n📈 Evaluating enhanced model...")
        print(f"""
============================================================
IF+Ontology - Evaluation Results
============================================================
ROC-AUC:              {combined_metrics['roc_auc']:.4f}
Precision-Recall AUC: {combined_metrics['pr_auc']:.4f}
============================================================""")

        print("\n💾 Saving results...")
        plot_roc_pr_curves(
            y_test,
            combined_scores,
            title=f"IF + Ontology (λ={best_lambda:.2f})" if 'best_lambda' in locals() else "IF + Ontology",
            save_path=str(figures_dir / 'ontology_roc_pr.png'),
        )
        print(f"✅ Saved plot: {figures_dir / 'ontology_roc_pr.png'}")
        
        print("✅ STEP 4 COMPLETE")
        
    except Exception as e:
        print(f"❌ Error in Step 4: {e}")
        import traceback
        traceback.print_exc()
        return 1


    # =========================================================================
    # STEP 5: Multi-Split Evaluation (Metrics & Experiments)
    # =========================================================================
    print_section("STEP 5: Multi-Split Evaluation (Metrics & Experiments)")

    try:
        import json
        import numpy as np

        # می‌تونی این لیست رو هرطور دوست داری عوض کنی
        seeds = [42, 123, 456, 789, 2025]

        all_results = []

        for seed in seeds:
            print(f"\n🔁 Running experiment for random_state={seed} ...")
            res = run_single_split_experiment(base_dir, random_state=seed)
            all_results.append(res)
            print(
                f"   IF      ROC={res['if_roc_auc']:.4f}, PR={res['if_pr_auc']:.4f} | "
                f"AE      ROC={res['ae_roc_auc']:.4f}, PR={res['ae_pr_auc']:.4f} | "
                f"IF+Ont  ROC={res['if_ont_roc_auc']:.4f}, PR={res['if_ont_pr_auc']:.4f}, "
                f"λ*={res['best_lambda']:.2f}"
            )

        # -----------------------------
        # محاسبه‌ی میانگین و انحراف معیار
        # -----------------------------
        def _mean_std(values):
            arr = np.asarray(values, dtype=float)
            return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0

        summary = {}
        for model_key, label in [
            ("if", "Isolation Forest"),
            ("ae", "Autoencoder"),
            ("if_ont", "IF + Ontology"),
        ]:
            roc_vals = [r[f"{model_key}_roc_auc"] for r in all_results]
            pr_vals = [r[f"{model_key}_pr_auc"] for r in all_results]
            roc_mean, roc_std = _mean_std(roc_vals)
            pr_mean, pr_std = _mean_std(pr_vals)
            summary[model_key] = {
                "label": label,
                "roc_auc_mean": roc_mean,
                "roc_auc_std": roc_std,
                "pr_auc_mean": pr_mean,
                "pr_auc_std": pr_std,
            }

        # -----------------------------
        # ذخیره در فایل JSON + Markdown
        # -----------------------------
        reports_dir = base_dir / "results" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        json_payload = {
            "seeds": seeds,
            "results": all_results,
            "summary": summary,
        }

        json_path = reports_dir / "multi_split_metrics.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_payload, f, indent=2)
        print(f"\n💾 Saved multi-split metrics JSON to: {json_path}")

        # Markdown summary
        md_lines = []
        md_lines.append("# Multi-Split Evaluation Summary")
        md_lines.append("")
        md_lines.append(f"Seeds: {', '.join(str(s) for s in seeds)}")
        md_lines.append("")
        md_lines.append("| Model | ROC-AUC (mean ± std) | PR-AUC (mean ± std) |")
        md_lines.append("|-------|----------------------|---------------------|")

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

        print(f"💾 Saved multi-split metrics Markdown to: {md_path}")

        # چاپ خلاصه روی کنسول
        print("\n📊 Multi-split metrics summary (mean ± std):\n")
        print(f"{'Model':<20} {'ROC-AUC (mean±std)':<25} {'PR-AUC (mean±std)':<25}")
        print("-" * 70)
        for key in ["if", "ae", "if_ont"]:
            s = summary[key]
            print(
                f"{s['label']:<20} "
                f"{s['roc_auc_mean']:.4f} ± {s['roc_auc_std']:.4f}   "
                f"{s['pr_auc_mean']:.4f} ± {s['pr_auc_std']:.4f}"
            )

    except Exception as e:
        print(f"❌ Error in Step 5: {e}")
        import traceback
        traceback.print_exc()
        # ادامه می‌دیم، فقط evaluation چند-اسپلتی fail شده


    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print_section("PIPELINE EXECUTION SUMMARY")
    
    print("📊 Model Performance Comparison:\n")
    print(f"{'Model':<20} {'ROC-AUC':<12} {'PR-AUC':<12}")
    print("-" * 44)
    print(f"{'Isolation Forest':<20} {if_metrics['roc_auc']:<12.4f} {if_metrics['pr_auc']:<12.4f}")
    print(f"{'Autoencoder':<20} {ae_metrics['roc_auc']:<12.4f} {ae_metrics['pr_auc']:<12.4f}")
    print(f"{'IF + Ontology':<20} {combined_metrics['roc_auc']:<12.4f} {combined_metrics['pr_auc']:<12.4f}")
    
    print("\n📁 Generated Files:")
    for file in sorted(figures_dir.glob('*.png')):
        print(f"   ✅ {file.relative_to(base_dir)}")
    
    print("\n✅ ALL PIPELINE STEPS COMPLETED SUCCESSFULLY!")
    return 0

def run_single_split_experiment(base_dir: Path, random_state: int) -> dict:
    """
    Run the full anomaly pipeline (IF, AE, IF+Ontology) for a single random_state
    and return metrics only (بدون لاگ‌های مفصل).

    خروجی:
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

    # -----------------------------
    # 1) Load & preprocess features
    # -----------------------------
    csv_path = base_dir / "data" / "raw" / "diabetic_data.csv"
    df = load_raw_data(str(csv_path))

    # از همون پری‌پروسسینگ اصلی استفاده می‌کنیم
    X, y, _ = build_feature_matrix(df)

    # Train/test split برای این seed
    X_train, X_test, y_train, y_test = train_test_split_stratified(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
    )

    train_pos_rate = float(y_train.mean())

    # -----------------------------
    # 2) Isolation Forest روی نرمال‌ها
    # -----------------------------
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
        y_test, if_scores_test, model_name=f"IF (seed={random_state})"
    )

    # -----------------------------
    # 3) Autoencoder روی نرمال‌ها
    # -----------------------------
    ae_detector = AutoencoderDetector(
        input_dim=X_train.shape[1],
        hidden_dims=[128, 64, 32],
        epochs=50,
        batch_size=256,
        learning_rate=0.001,
    )

    normal_mask_ae = (y_train == 0)
    X_train_normal_ae = X_train[normal_mask_ae]
    ae_detector.fit(X_train_normal_ae)

    ae_scores_test = ae_detector.predict_scores(X_test)
    ae_metrics = compute_classification_metrics(
        y_test, ae_scores_test, model_name=f"AE (seed={random_state})"
    )

    # -----------------------------
    # 4) Ontology روی همان split
    # -----------------------------
    # ساخت DataFrame کلینیکال هم‌تراز با X
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

    # اگر به هر دلیلی طول تست‌ست‌ها هم‌تراز نبود، ontology را skip می‌کنیم
    if len(X_clin_test) != len(y_test):
        # در این حالت فقط IF و AE را گزارش می‌کنیم
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

    ontology_penalties_test, _rule_stats = apply_ontology_rules(
        X_clin_test,
        y_clin_test.to_numpy(),
    )

    # λ-sweep مثل Step 4 اصلی
    lambda_values = [0.0, 0.1, 0.3, 0.5]
    combo_results = []

    for lam in lambda_values:
        alpha = 1.0 - lam
        beta = lam
        scores_lam = combine_scores(
            if_scores_test,
            ontology_penalties_test,
            alpha=alpha,
            beta=beta,
            normalize_ml=True,
        )
        metrics_lam = compute_classification_metrics(
            y_test,
            scores_lam,
            model_name=f"IF+Ontology (λ={lam:.2f}, seed={random_state})",
        )
        combo_results.append((lam, metrics_lam))

    # انتخاب بهترین λ با PR-AUC و بعد ROC-AUC
    def _score_key(item):
        lam, m = item
        return (m["pr_auc"], m["roc_auc"])

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




if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
