# Ontology-aware Anomaly Detection Toy Pipeline

## Abstract
This project implements an anomaly detection pipeline to identify early hospital readmissions (<30 days) using the Diabetes 130-US Hospitals dataset. It explores a hybrid approach that combines unsupervised machine learning models (IsolationForest and Autoencoders) with an ontology-inspired rule layer. This "ontology layer" penalizes clinically high-risk patterns—such as poor glycemic control or medication mismanagement—to refine the statistical anomaly scores, aiming to improve detection of preventable readmissions.

## Dataset
**Diabetes 130-US Hospitals for years 1999–2008**  
This dataset represents 10 years (1999-2008) of clinical care at 130 US hospitals and integrated delivery networks. It includes over 50 features representing patient and hospital outcomes.

- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)
- **Target**: Early readmission (`<30` days).
- **Key Features**: Demographics, laboratory tests (A1C, glucose), medication administration, and hospital stay metrics.

## Method Overview

### 1. Baseline 1: IsolationForest
We use the standard **IsolationForest** algorithm (via scikit-learn) as a baseline unsupervised anomaly detector. It isolates observations by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature. Anomalies are susceptible to isolation and thus have shorter path lengths in the trees.

### 2. Baseline 2: Autoencoder
A feedforward **Autoencoder** is implemented using PyTorch.
- **Architecture**: Encoder compresses the input into a lower-dimensional bottleneck (latent space), and the Decoder attempts to reconstruct the original input.
- **Training**: The model is trained **only on normal samples** (non-readmitted patients).
- **Anomaly Score**: The reconstruction error (Mean Squared Error) between the input and the output. High error indicates the sample deviates from the "normal" patterns learned during training.

### 3. Ontology-Inspired Rule Layer
To bridge the gap between statistical anomalies and clinical risk, we introduce a rule-based penalty term derived from domain knowledge.

**Key Rules:**
1.  **Poor Glycemic Control**: High A1C (>7 or >8) + No change in medication + Currently on diabetes meds. (Suggests treatment inertia).
2.  **Severe Hyperglycemia**: Very high glucose (>200 or >300) + Low number of lab procedures. (Suggests insufficient monitoring).
3.  **Complex Case / Short Stay**: High medication burden (>20 meds) + Very short hospital stay (<3 days). (Suggests premature discharge).

**Combined Score:**
The final anomaly score ($S_{det}$) is a weighted combination of the model's statistical score ($S_{gen}$) and the ontology penalty ($S_{ont}$):

$$ S_{det} = \alpha \cdot S_{gen} + \beta \cdot S_{ont} $$

(e.g., $\alpha=0.7, \beta=0.3$)

## Project Structure

```
ontology-aware-anomaly-detection-diabetes/
├── data/
│   ├── raw/                # Original dataset
│   └── processed/          # Cleaned/Feature-engineered data
├── notebooks/
│   ├── 01_eda.ipynb        # Exploratory Data Analysis
│   ├── 02_baseline_if.ipynb # Isolation Forest Baseline
│   ├── 03_autoencoder.ipynb # Autoencoder Model
│   └── 04_ontology_eval.ipynb # Ontology Rules & Evaluation
├── src/
│   ├── preprocessing.py    # Data loading & cleaning functions
│   ├── models.py           # IsolationForest & Autoencoder classes
│   ├── ontology.py         # Ontology rule definitions
│   └── evaluation.py       # Metrics & plotting helpers
├── results/
├── tests/
├── README.md
└── requirements.txt
```

## How to Run

### Prerequisites
- Python 3.8+
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### Step-by-Step

1.  **Download Data**:
    - Download `diabetic_data.csv` from the [UCI Repository](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008).
    - Place the file in the `data/` directory: `data/diabetic_data.csv`.

2.  **Run Notebooks**:
    - Start Jupyter Lab or Notebook:
      ```bash
      jupyter lab
      ```
    - Open the notebooks in `notebooks/` in order:
        1.  `01_eda.ipynb`: Explore the data.
        2.  `02_baseline_if.ipynb`: Train and evaluate Isolation Forest.
        3.  `03_autoencoder.ipynb`: Train and evaluate Autoencoder.
        4.  `04_ontology_eval.ipynb`: Apply ontology rules and compare results.

3.  **Using the Source Code**:
    - The core logic is available in the `src/` package. You can import these modules in your own scripts:
      ```python
      from src.preprocessing import load_data, clean_data
      from src.models import train_autoencoder
      ```

## Results

| Model                           | ROC-AUC | PR-AUC | Notes                            |
|---------------------------------|--------:|-------:|----------------------------------|
| IsolationForest                 |   0.XX  |  0.XX  | Baseline unsupervised detector   |
| Autoencoder                     |   0.XX  |  0.XX  | Reconstruction-based baseline    |
| IF + ontology-inspired penalty  |   0.XX  |  0.XX  | Combined statistical + rule term |
| AE + ontology-inspired penalty  |   0.XX  |  0.XX  | Combined reconstruction + rules  |

(You can leave placeholder `0.XX` values where I will later fill in real numbers.)

## Future Work

- **Medical Ontologies**: Using real medical ontologies (**SNOMED CT**, **RxNorm**, **DOID**, **HPO**) for more robust rule generation.
- **Graph Representations**: Applying graph-based representations instead of simple rules.
- **Sequence Models**: Using sequence models (e.g. **GRU**/**LSTM**) over patient trajectories.
- **Generative Explanations**: Integrating diffusion-based generative models for counterfactual explanations.
