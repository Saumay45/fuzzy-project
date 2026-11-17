# Current execution record

Verified on 2026-11-18 after the user installed dependencies and downloaded the dataset.
Python 3.14.4; NumPy 2.5.5; Pandas 3.1.1; scikit-learn 1.9.1;
Matplotlib 3.11.2; Seaborn 0.13.2; Plotly 6.9.0;
Streamlit 1.67.0; Joblib 1.6.0; Pytest 9.2.0.

## Problems found and corrected

1. Fractional validation size `.15` made sklearn reserve 111 of 734 patients.
   Use an explicit rounded integer count, reproducing Table 6's **624/110/184** split.
   Apply the same rounding policy in cross-validation. All four default models and
   five-fold results have been regenerated after this change.
2. Streamlit AppTest resolves relative script paths against the calling test file.
   Tests now use the absolute path to the repository's app.py.
3. Form key `patient` collided with `st.session_state['patient']` on submission.
   Store the submitted inputs as `patient_inputs`. A regression test submits two
   different profiles, checks updated inputs and predictions, and opens their traces.

## Checks actually completed

- Installed package imports succeeded; `python -m pip check` found no broken requirements.
- Real dataset passed schema, numeric range, category, class-count and duplicate validation:
  **918 rows, 508 positive and 410 negative labels**.
- Full training, independently preprocessed five-fold CV and held-out evaluation completed.
- Train-only imputation/scaling; default 38 neural features; actual split 624/110/184.
- Best validation checkpoints: Baseline ANN epoch **23**, Neuro-Fuzzy ANN epoch **30**.
- All four models saved, reloaded and checked for probability equality on the test set.
- Ten PNG figures and CSV/JSON evaluation files generated in results/.
- Saved models successfully predicted multiple actual held-out patients.
- `python -m src.evaluate` reproduced saved held-out metrics on the fingerprinted dataset.
- Complete test suite: **51 passed, 19 subtests passed in 3.08s**. No skips or failures.
  Includes all five Streamlit pages and repeated form submissions using different inputs.
- Exact Word-table parameters and numerical singletons checked by automated tests.
- The Word report remains unchanged. No Git commit or public deployment was performed.

## Actual held-out results (184 patients)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Fuzzy-Only | 0.826087 | 0.786885 | 0.941176 | 0.857143 | 0.844512 |
| Logistic Regression | 0.891304 | 0.894231 | 0.911765 | 0.902913 | 0.929579 |
| Baseline ANN | 0.869565 | 0.882353 | 0.882353 | 0.882353 | 0.925514 |
| Neuro-Fuzzy ANN | 0.864130 | 0.873786 | 0.882353 | 0.878049 | 0.916547 |

Five-fold accuracy (mean ± population standard deviation):
Baseline ANN **83.66% ± 2.18 percentage points**;
Neuro-Fuzzy ANN **85.18% ± 2.32 percentage points**.
Full scores are in results/cv_folds.csv and results/cv_summary.csv.
These are actual predictions, not hardcoded report values. Differences from the
report can follow from training-only imputation, encoding, library versions, checkpoint
behavior and data splitting. The model does not outperform every baseline.

## Remaining runtime limitation

`python scripts/verify_e2e.py` completed real-row prediction and the entire test suite,
then attempted to launch Streamlit at **127.0.0.1:8507**. The tool sandbox denied socket
creation: **PermissionError: [Errno 1] Operation not permitted** from socket.socket.
The server exited before HTTP health could be checked. No server was left running.
This is an environment restriction distinct from application execution: Streamlit
AppTest successfully executed every page and repeated patient predictions.
A live browser/HTTP check remains to be performed from the user's terminal.

Run:

```bash
cd /home/saumay/Documents/Fuzzy_Project
source .venv/bin/activate
python -m streamlit run app.py --server.address 127.0.0.1
```

Open http://localhost:8501. Select Heart Health Prediction, edit the eleven inputs,
then click Assess heart health. Repeat with changed inputs; retraining is unnecessary.
Stop the server with Ctrl+C. To repeat all verification outside the restricted tool
sandbox, run `python scripts/verify_e2e.py`.

## Earlier setup history

Initial attempts to install packages and download Kaggle data failed because external
DNS/network access was unavailable. Only three offline specification checks could run
then. That earlier dependency/data blocker is now resolved by the user's installation.
The minimum-AND-plus-FRI optional experiment is implemented and its components are
tested; only the report-default product-AND 38-feature configuration has been trained
and saved in the default models/ directory.
