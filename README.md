# Neuro-Fuzzy Heart Health Risk Assessment

A college demonstration implementing the supplied `NeuroFuzzy_Heart_Health_Report.docx`:
custom NumPy triangular fuzzification, ten fixed Sugeno rules, a 16–8 MLP, three
baselines, leak-free evaluation, and a local Streamlit application.

**Current execution status:** dependencies and the 918-row dataset are installed.
All four models have been retrained on the corrected 624/110/184 split, five-fold
cross-validation completed, ten figures were generated, and model reloads verified.
**51 tests and 19 syntax subtests pass**, including all app pages and repeated patient
predictions. The tool sandbox blocks creating a localhost server socket; launch
Streamlit from your own terminal using the command below. See
[VERIFICATION.md](VERIFICATION.md) for measured results and this remaining limitation.

Academic demonstration only. Do not use for medical diagnosis or clinical decisions.
The target is the dataset's existing HeartDisease label; model scores and the three
risk bands are not calibrated clinical risk estimates.

## Install and run

Use Python 3.11 or newer (Python 3.12/3.13 recommended for broad wheel availability).
Run these commands from this directory on a machine with network access:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/download_data.py
python -m src.train
python -m pytest -q
python scripts/verify_e2e.py
python -m streamlit run app.py --server.address 127.0.0.1
```

If Kaggle rejects automated downloading, obtain `heart.csv` from the
[original dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
and place it at `data/heart.csv`. Expected shape is 918×12 with 508 positive labels.
No synthetic data is substituted. Training validates the schema, ranges, categories,
label counts and duplicates and saves a dataset fingerprint.

Training is a separate command. Opening the dashboard only loads saved models; if
artifacts are absent, the UI explains how to train and disables prediction.
No public deployment is configured. The verification script starts and stops a server
on localhost only; it also executes the test suite.

## Methodology and the report

Default: product AND, no FRI in the neural inputs, exactly as report §§3.5/4.4 specify.
Features: 15 memberships + 10 rule strengths + 13 encoded attributes = **38**. Raw
baselines see 5 standardized numeric + 13 encoded = **18**. Sex and ExerciseAngina
use binary coding; chest pain, ECG and slope use complete one-hot vectors.

The additional request for minimum AND and FRI input is supported in a separate run:

```bash
python -m src.train --tnorm minimum --include-fri --output experiments/requested
```

That configuration has **39** features; its artifacts stay separate from the default
run. The dashboard uses `models/heart_models.joblib` in the project root by default.
To explore that alternative in the app, set the artifact root:

```bash
HEART_ARTIFACT_ROOT=experiments/requested python -m streamlit run app.py --server.address 127.0.0.1
```

Imputation and all fitted transformations use training data only. Splits are stratified
624/110/184 with seed 42. Both ANNs use Adam, ReLU, hidden layers (16,8), learning rate
0.01, L2 alpha 0.001, binary cross entropy, up to 150 epochs, patience 15, and restoration
of the best validation-loss checkpoint. `partial_fit` preserves Adam state across epochs.
All learned models use the same 624 training rows. Five-fold CV repeats preprocessing
and internal validation independently; all 918 patients rotate through outer folds,
as in the report. This is not additional independent cohort validation.

FRI uses sum(strength×singleton)/sum(strength), defaulting to 0.5 for total strength
<=1e-12. The fuzzy baseline has no learned decision parameters; only its missing-value
imputer is fitted. Binary predictions use **p>=0.50**. Illustrative bands use **0.33**
and **0.66**. A Moderate-band result can have either binary class.

All discrepancies and reference metrics are in
[learning_notes/report_decisions.md](learning_notes/report_decisions.md). In particular,
R9 cannot activate for Oldpeak=3.2 despite the explanation in report §5.8: Table 4
sets Low Oldpeak to zero there. This implementation follows the table mathematics.

## Outputs after real training

- `models/heart_models.joblib`: all four models, preprocessing, metadata, metrics.
- `results/model_comparison.csv`, `metrics.json`: actual test metrics and confusion matrices.
- `results/cv_folds.csv`, `cv_summary.csv`: fold scores and mean/population standard deviation.
- `results/training_history.json`: loss and accuracy at every executed epoch.
- `results/test_predictions.csv`, `sample_patients.csv`: real test outcomes and examples.
- `results/metadata.json`: seed, split indices, medians, versions, best epochs, feature names and dataset hash.
- Ten PNG plots: class distribution, membership functions, feature distributions,
  correlation heatmap, learning curves, confusion matrices, ROC curves, rule heatmap,
  illustrative band distribution and model-performance heatmap.

The training script saves and reloads models and verifies probability equality. It
prints actual metrics; no reported accuracy is hardcoded into evaluation or the app.

Reevaluate saved models without training:

```bash
python -m src.evaluate
```

Predict from a JSON file containing all eleven attributes:

```bash
python -m src.predict path/to/patient.json
```

For the alternate experiment, evaluation accepts `--model` and `--output` paths and
prediction accepts `--model`. Reevaluation checks the exact dataset hash before using
saved test-row indices. Only load joblib files you trust.

## Code and learning guide

| Module | Purpose |
|---|---|
| `src/data_processing.py` | Validation, missing-value imputation, splits |
| `src/fuzzy_logic.py`, `fuzzy_rules.py` | Exact memberships, rule activation, Sugeno FRI |
| `src/feature_engineering.py` | Named 18/38/39 feature representations |
| `src/neural_network.py` | Explicit epochs, validation curves, best checkpoint |
| `src/baselines.py` | Fixed fuzzy estimator and logistic regression |
| `src/train.py` | Shared-split experiments, five-fold CV, persistence |
| `src/evaluate.py`, `predict.py` | Saved-model evaluation and patient inference |
| `src/visualization.py` | Real-data figures |
| `app.py` | Five interactive Streamlit pages and Plotly charts |

Read `learning_notes/01_system_and_preprocessing.md` through
`05_viva_questions.md` in order. They cover all eleven requested educational topics,
formulas, numerical examples, source-function mappings, and viva preparation.
Tests cover all 15 triangles, all 10 rules, both T-norms, Sugeno arithmetic, fallback,
feature dimensions, invalid inputs, train-only transformations, checkpoint restoration,
thresholds, persisted predictions and Streamlit page/form execution. Artifact-dependent
tests skip explicitly when no real data or models exist.

Dataset, models, results, local environments, caches and learning notes are ignored
by Git as requested. The Word report is unchanged.

Limitations: fixed expert-inspired rules without clinician review, one small merged
dataset, no external validation or calibration, and fuzzy traces explain only the
front end rather than the full neural network. Requirements use bounded compatible
versions; each training run records installed versions for reproducibility.
