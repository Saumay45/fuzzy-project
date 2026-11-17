"""Compute real binary classification metrics; never substitute report values."""
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve)


def evaluate(y, probability) -> dict:
    probability = np.asarray(probability)
    prediction = probability >= .5
    fpr, tpr, _ = roc_curve(y, probability)
    return {'accuracy': float(accuracy_score(y, prediction)),
            'precision': float(precision_score(y, prediction, zero_division=0)),
            'recall': float(recall_score(y, prediction, zero_division=0)),
            'f1': float(f1_score(y, prediction, zero_division=0)),
            'roc_auc': float(roc_auc_score(y, probability)),
            'confusion_matrix': confusion_matrix(y, prediction, labels=[0, 1]).tolist(),
            'roc': {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}}


def risk_band(probability: float) -> str:
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError('Probability must be in [0, 1]')
    return 'Low' if probability < .33 else 'Moderate' if probability < .66 else 'High'


def main():
    """Re-evaluate saved models on the recorded held-out rows without retraining."""
    import argparse
    import hashlib
    import json
    from pathlib import Path
    from src.data_processing import load_dataset
    from src.predict import load_bundle
    parser = argparse.ArgumentParser(description='Evaluate saved models without retraining')
    parser.add_argument('--data', default='data/heart.csv')
    parser.add_argument('--model', default='models/heart_models.joblib')
    parser.add_argument('--output', default='results/reevaluation.json')
    args = parser.parse_args()
    bundle = load_bundle(args.model)
    fingerprint = hashlib.sha256(Path(args.data).read_bytes()).hexdigest()
    if fingerprint != bundle['metadata']['dataset_sha256']:
        raise ValueError('Dataset bytes differ from the training dataset; held-out row indices are unsafe')
    frame = load_dataset(args.data)
    test = frame.loc[bundle['metadata']['split_indices']['test']]
    results = {name: evaluate(test.HeartDisease, model.predict_proba(test)[:, 1])
               for name, model in bundle['models'].items()}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
