"""Shared saved-model inference for Streamlit and command line."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from src.data_processing import validate
from src.fuzzy_logic import infer
from src.fuzzy_rules import RULES, CONSEQUENTS, rule_description
from src.evaluate import risk_band


def load_bundle(path='models/heart_models.joblib') -> dict:
    """Load locally trained trusted artifacts only (joblib can execute code)."""
    if not Path(path).exists():
        raise FileNotFoundError('No trained artifacts. Run python -m src.train first.')
    return joblib.load(path)


def predict_patient(patient: dict, bundle: dict) -> dict:
    frame = validate(pd.DataFrame([patient]))
    features = bundle['models']['Neuro-Fuzzy ANN'][0]
    clean = features.imputer_.transform(frame)
    memberships, strengths, fri = infer(clean, bundle['metadata']['tnorm'])
    strengths = strengths[0]
    total = strengths.sum()
    contribution = strengths * CONSEQUENTS / total if total > 1e-12 else np.zeros(10)
    predictions = {}
    for name, model in bundle['models'].items():
        p = float(model.predict_proba(frame)[0, 1])
        predictions[name] = {'probability': p, 'class': int(p >= .5), 'band': risk_band(p)}
    return {'predictions': predictions, 'fuzzy_risk_index': float(fri[0]),
            'no_rules_fired': bool(total <= 1e-12),
            'memberships': {k: float(v) for k, v in memberships.iloc[0].items()},
            'rules': [{'rule': r[0], 'antecedents': rule_description(r), 'consequent': r[2],
                       'strength': float(strengths[i]), 'fri_contribution': float(contribution[i])}
                      for i, r in enumerate(RULES)],
            'imputed_fields': [k for k in frame if frame[k].isna().iloc[0] or
                               (k in ['Cholesterol', 'RestingBP'] and frame[k].iloc[0] == 0)]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('patient_json', help='JSON object file containing the eleven patient attributes')
    parser.add_argument('--model', default='models/heart_models.joblib')
    args = parser.parse_args()
    print(json.dumps(predict_patient(json.loads(Path(args.patient_json).read_text()), load_bundle(args.model)), indent=2))


if __name__ == '__main__':
    main()
