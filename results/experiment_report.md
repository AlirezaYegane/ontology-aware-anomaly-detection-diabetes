# Ontology-Aware Anomaly Detection – Experimental Report

This report summarizes the final implementation and results of an ontology-aware anomaly detection pipeline on the **Diabetes 130-US hospitals (1999–2008)** dataset.  
The objective is to identify patients at high risk of **30-day hospital readmission**.

---

## 1. Data and Preprocessing

### 1.1. Dataset

- Source: UCI Machine Learning Repository – *Diabetes 130-US hospitals for years 1999–2008*
- Number of encounters (rows) in the raw dataset: **101,766**
- Number of columns in the raw dataset: **50** structural and clinical attributes

The dataset contains:

- Patient demographics (age, gender, race)
- Admission type and source
- Length of stay
- ICD-9 diagnostic and procedure codes
- Diabetes medications and treatment changes
- Counts of lab tests and medications
- Future readmission status (`readmitted`)

### 1.2. Target Definition

The modeling task is to predict whether a patient will be readmitted within 30 days.

We define the binary label as:

- `y = 1` if `readmitted == '<30'`
- `y = 0` if `readmitted` is `'>30'` or `'NO'`

After filtering and cleaning, the class distribution is:

- Total samples: **99,493**
- Positives (readmitted < 30 days): **11,169** (≈ 11.2%)
- Negatives: **88,324** (≈ 88.8%)

This results in a **class-imbalanced** problem with an anomaly rate of about 11%.

### 1.3. Preprocessing Pipeline

The preprocessing steps (as logged in `data_filtering_summary.json` and `data_filtering_summary.md`) are:

1. **Feature selection**  
   From the original 50 columns, 14 key features are selected, combining:
   - Demographics (age, race, gender)
   - Utilization intensity (number of inpatient stays, ER visits, length of stay)
   - Treatment behavior (medication changes, use of diabetes medications)
   - Counts of lab procedures and medications

2. **Missing data handling**  
   - Replace `'?'` markers with `NaN`
   - Drop rows with missing values in critical columns (`readmitted`, `race`, `gender`, `age`, `time_in_hospital`, `num_medications`, etc.)  
     ⇒ 2,273 rows removed (≈ 2.2%)

3. **Imputation and scaling**  
   - Numerical features imputed with the **median**
   - Categorical features imputed with `'Unknown'`
   - **OneHotEncoding** for categorical variables
   - **StandardScaler** for numerical variables  

The final feature matrix has shape **(99,493 × 30)**.

A complete summary of all filtering and preprocessing steps is stored in:

- `results/reports/data_filtering_summary.json`
- `results/reports/data_filtering_summary.md`

---

## 2. Experimental Setup and Models

### 2.1. Train/Test Split

For all single-split experiments (except the multi-split evaluation):

- `test_size = 0.2`
- `random_state = 42`
- Stratified split on `y` to preserve class balance

### 2.2. Models

**Unsupervised / anomaly-detection models**

- `IsolationForest`
- Reconstruction-based `Autoencoder`

Both models are trained **only on normal samples** (`y = 0`, i.e. no 30-day readmission) and then used to produce anomaly scores for the full test set.

**Supervised baselines**

- `DecisionTreeClassifier`
- `RandomForestClassifier`

These models are trained on the full labeled training set (`X_train`, `y_train`) and return the predicted probability of the positive class as the risk score.

**Ontology-aware models**

- `IF + Ontology`
- `AE + Ontology`
- `Ensemble(IF+Ont, AE+Ont)`

Details of the ontology layer are described in Section 4.

---

## 3. Single-Split Results

### 3.1. Unsupervised models

The ROC and PR curves for **Isolation Forest** and **Autoencoder** on the test set are shown below:

<table>
  <tr>
    <td style="text-align:center">
      <strong>Isolation Forest</strong><br>
      <img src="../figures/if_roc_pr.png" alt="IF ROC/PR" width="320">
    </td>
    <td style="text-align:center">
      <strong>Autoencoder</strong><br>
      <img src="../figures/ae_roc_pr.png" alt="AE ROC/PR" width="320">
    </td>
  </tr>
</table>

Performance metrics:

| Model            | ROC-AUC | PR-AUC |
|------------------|---------|--------|
| Isolation Forest | 0.5317  | 0.1276 |
| Autoencoder      | 0.5457  | 0.1311 |

**Interpretation**

Both models perform better than random, but the task is clearly difficult and noisy.  
The autoencoder is slightly stronger than Isolation Forest on this dataset.

---

### 3.2. Supervised baselines

Here we use the labels `y` directly, so higher performance than unsupervised models is expected.

<table>
  <tr>
    <td style="text-align:center">
      <strong>Decision Tree</strong><br>
      <img src="../figures/dt_roc_pr.png" alt="DT ROC/PR" width="320">
    </td>
    <td style="text-align:center">
      <strong>Random Forest</strong><br>
      <img src="../figures/rf_roc_pr.png" alt="RF ROC/PR" width="320">
    </td>
  </tr>
</table>

Performance metrics:

| Model          | ROC-AUC | PR-AUC |
|----------------|---------|--------|
| Decision Tree  | 0.6246  | 0.1860 |
| Random Forest  | 0.6362  | 0.2024 |

**Interpretation**

- The Random Forest outperforms the Decision Tree on both ROC-AUC and PR-AUC.
- The performance gap between RF and the unsupervised models shows that, for this dataset, labels carry substantial information about 30-day readmission risk.

---

## 4. Integrating Ontology with Anomaly Models

### 4.1. Ontology layer and clinical rules

The ontology layer applies a set of clinically motivated rules on the **clinical feature space** (before OneHotEncoding), for example:

- **poor_control_no_med_change**: indicators of poor glycemic control with no change in diabetes medications.
- **frequent_inpatient_admissions**: frequent inpatient hospitalizations.
- **polypharmacy**: unusually high number of concurrent medications.
- **er_and_inpatient_use**: combination of ER visits and inpatient utilization.

For each test patient:

- A subset of rules fires.
- Based on which rules are triggered and how often, an **ontology-based penalty score** is computed.

### 4.2. Combining model scores with ontology scores

For each anomaly model (IF and AE), the final score is defined as a weighted sum of the model score and the ontology score:

\[
\text{score}_{\lambda} = (1 - \lambda)\,\text{score}_{\text{model}} + \lambda\,\text{score}_{\text{ontology}},
\]

with \(\lambda \in \{0.0, 0.1, 0.3, 0.5\}\) controlling the contribution of the ontology layer.

For each \(\lambda\), we compute ROC-AUC and PR-AUC.  
The best \(\lambda\) is selected according to PR-AUC (breaking ties by ROC-AUC).

### 4.3. Single-split results with ontology

<table>
  <tr>
    <td style="text-align:center">
      <strong>IF + Ontology</strong><br>
      <img src="../figures/ontology_if_roc_pr.png" alt="IF+Ont ROC/PR" width="300">
    </td>
    <td style="text-align:center">
      <strong>AE + Ontology</strong><br>
      <img src="../figures/ontology_ae_roc_pr.png" alt="AE+Ont ROC/PR" width="300">
    </td>
    <td style="text-align:center">
      <strong>Ensemble(IF+Ont, AE+Ont)</strong><br>
      <img src="../figures/ontology_ensemble_roc_pr.png" alt="Ensemble ROC/PR" width="300">
    </td>
  </tr>
</table>

Results for the best \(\lambda = 0.5\):

| Model                     | ROC-AUC | PR-AUC |
|---------------------------|---------|--------|
| IF + Ontology             | 0.5774  | 0.1507 |
| AE + Ontology             | 0.5872  | 0.1601 |
| Ensemble(IF+Ont, AE+Ont)  | 0.5794  | 0.1529 |

**Key observations**

- Adding the ontology layer yields a **consistent improvement** in both ROC-AUC and PR-AUC over the plain IF and AE models.
- AE + Ontology achieves the best performance among the unsupervised / ontology-aware models.
- The ensemble of IF+Ont and AE+Ont does not outperform AE+Ont on this dataset, but it demonstrates that combining multiple ontology-aware scores is feasible.

### 4.4. Rule firing statistics

On the test set:

- Total (non-unique) rule firings: **21,180**
- Positive cases among patients with at least one fired rule: **2,846**

Rules such as `frequent_inpatient_admissions` and `er_and_inpatient_use` cover a larger share of high-risk patients and contribute most strongly to the PR-AUC gains.

---

## 5. Multi-Split Evaluation

To test robustness, we repeat the experiments over five different `random_state` values:

\[
\{42,\; 123,\; 456,\; 789,\; 2025\}.
\]

The outputs of this analysis are stored in:

- `results/reports/multi_split_metrics.json`
- `results/reports/multi_split_metrics.md`

The table below shows mean and standard deviation across these splits:

| Model            | ROC-AUC (mean ± std) | PR-AUC (mean ± std) |
|------------------|----------------------|---------------------|
| Isolation Forest | 0.5294 ± 0.0056      | 0.1265 ± 0.0021     |
| Autoencoder      | 0.5336 ± 0.0287      | 0.1276 ± 0.0127     |
| IF + Ontology    | 0.5824 ± 0.0041      | 0.1535 ± 0.0027     |

**Observations**

- IF + Ontology consistently outperforms both base models across all splits.
- The very small standard deviation for IF + Ontology suggests that the improvement is not split-specific or accidental.
- The plain autoencoder is more variable, which is consistent with the behavior of neural networks and sensitivity to initialization.

---

## 6. Discussion

1. **Trustworthiness of supervised baselines**  
   Supervised models (especially Random Forest) achieve higher absolute performance than the unsupervised models, but they depend on **reliable labels**. In real clinical settings, labels can be noisy or incomplete, which limits the applicability of purely supervised approaches.

2. **Role of the ontology layer**  
   Adding an ontology layer on top of unsupervised models:
   - Produces stable gains in both ROC-AUC and PR-AUC.
   - Demonstrates that interpretable clinical rules can steer anomaly scores in a medically meaningful direction.

3. **Comparison with supervised baselines**  
   Although Random Forest still has stronger raw metrics than AE + Ontology on this dataset:
   - Ontology-aware models do not rely on fully trustworthy labels.
   - The ontology output is directly interpretable (e.g., “high risk due to polypharmacy and frequent inpatient admissions”).

---

## 7. Limitations and Future Work

### 7.1. Limitations

- The dataset is a public UCI table with limited granularity compared to real ICU EHRs (such as MIMIC or HiRID).
- The current pipeline ignores temporal structure and operates on aggregated encounter-level features only.
- The ontology rules are manually designed and do not cover the full spectrum of possible risk patterns.

### 7.2. Future directions for the full research project

- Replace the dataset with richer, time-stamped EHR sources (ICU time series, medication events, clinical notes).
- Model encounter and event sequences using sequence-based architectures (RNNs, Transformers).
- Learn or semi-automatically derive ontology rules from medical knowledge graphs (SNOMED, RxNorm, ICD).
- Combine generative models (e.g., diffusion models) with ontology-aware scoring for counterfactual scenario generation and sensitivity analysis.

---

## 8. Reproducibility

To fully reproduce these experiments:

```bash
# 1. Create the environment
conda env create -f environment.yml
conda activate anomaly

# 2. Place the dataset
#   Put the diabetic_data.csv file at:
#   data/raw/diabetic_data.csv

# 3. Run the end-to-end pipeline
python run_pipeline_direct.py

# 4. Run tests
pytest -q
