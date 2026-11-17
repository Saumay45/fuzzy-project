"""Run: python -m src.train. Every transform is fitted inside its training split."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import importlib.metadata
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from src.baselines import FuzzyClassifier, logistic_regression
from src.data_processing import load_dataset, split_data
from src.feature_engineering import HeartFeatures
from src.neural_network import train_network
from src.evaluate import evaluate
from src.visualization import generate_figures


METRICS = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']


def fit_ann(train, validation, fuzzy, seed, tnorm, include_fri, max_epochs):
    features = HeartFeatures(fuzzy=fuzzy, tnorm=tnorm, include_fri=include_fri)
    x = features.fit_transform(train)
    xv = features.transform(validation)
    model, history = train_network(x, train.HeartDisease.to_numpy(), xv,
                                   validation.HeartDisease.to_numpy(), seed=seed,
                                   max_epochs=max_epochs)
    return Pipeline([('features', features), ('classifier', model)]), history


def cross_validate(frame, seed=42, tnorm='product', include_fri=False, max_epochs=150):
    """All-data CV as in report; no outer-fold test data enters preprocessing."""
    rows = []
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (development_index, test_index) in enumerate(folds.split(frame, frame.HeartDisease), 1):
        development, outer_test = frame.iloc[development_index], frame.iloc[test_index]
        train, validation = train_test_split(development, test_size=round(len(development) * .15),
                                            stratify=development.HeartDisease, random_state=seed)
        for name, fuzzy in [('Baseline ANN', False), ('Neuro-Fuzzy ANN', True)]:
            model, _ = fit_ann(train, validation, fuzzy, seed, tnorm, include_fri, max_epochs)
            score = evaluate(outer_test.HeartDisease, model.predict_proba(outer_test)[:, 1])
            rows.append({'fold': fold, 'model': name, 'best_epoch': model[-1].best_epoch_,
                         **{key: score[key] for key in METRICS}})
            print(f'CV fold {fold} {name}: accuracy={score["accuracy"]:.4f}', flush=True)
    folds_frame = pd.DataFrame(rows)
    summary = folds_frame.groupby('model')[METRICS].agg(['mean', lambda x: x.std(ddof=0)])
    summary.columns = [f'{metric}_{"mean" if stat == "mean" else "std"}'
                       for metric, stat in summary.columns]
    return folds_frame, summary.reset_index()


def run(data='data/heart.csv', output='.', seed=42, tnorm='product', include_fri=False,
        max_epochs=150):
    root = Path(output)
    model_dir, result_dir = root / 'models', root / 'results'
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    frame = load_dataset(data)
    train, validation, test = split_data(frame, seed)
    models = {'Fuzzy-Only': FuzzyClassifier(tnorm).fit(train)}
    models['Logistic Regression'] = Pipeline([
        ('features', HeartFeatures()), ('classifier', logistic_regression(seed))])
    models['Logistic Regression'].fit(train, train.HeartDisease)
    histories = {}
    for name, fuzzy in [('Baseline ANN', False), ('Neuro-Fuzzy ANN', True)]:
        models[name], histories[name] = fit_ann(train, validation, fuzzy, seed, tnorm,
                                               include_fri, max_epochs)
    probabilities = {name: model.predict_proba(test)[:, 1] for name, model in models.items()}
    metrics = {name: evaluate(test.HeartDisease, p) for name, p in probabilities.items()}
    folds, cv_summary = cross_validate(frame, seed, tnorm, include_fri, max_epochs)
    pre = models['Neuro-Fuzzy ANN'][0]
    metadata = {
        'dataset_sha256': hashlib.sha256(Path(data).read_bytes()).hexdigest(),
        'dataset_rows': len(frame), 'class_counts': {str(k): int(v) for k, v in frame.HeartDisease.value_counts().items()},
        'zero_readings': {k: int((frame[k] == 0).sum()) for k in ['Cholesterol', 'RestingBP']},
        'seed': seed, 'tnorm': tnorm, 'include_fri': include_fri, 'max_epochs': max_epochs,
        'patience': 15, 'binary_threshold': .5, 'risk_thresholds': [.33, .66],
        'split_sizes': {'train': len(train), 'validation': len(validation), 'test': len(test)},
        'split_indices': {'train': train.index.tolist(), 'validation': validation.index.tolist(), 'test': test.index.tolist()},
        'numeric_training_medians': {k: float(pre.imputer_.fill_[k]) for k in pre.scaler_.feature_names_in_},
        'neuro_feature_names': pre.get_feature_names_out().tolist(),
        'best_epochs': {k: models[k][-1].best_epoch_ for k in histories},
        'python_version': platform.python_version(),
        'packages': {k: importlib.metadata.version(k) for k in ['numpy', 'pandas', 'scikit-learn', 'joblib']},
        'fuzzy_training_metrics': evaluate(train.HeartDisease, models['Fuzzy-Only'].predict_proba(train)[:, 1]),
        'cv_scope': 'All 918 patients, independent within-fold train/validation preprocessing; not a second untouched holdout.',
    }
    clean = pre.imputer_.transform(frame)
    clean['HeartDisease'] = frame.HeartDisease
    clean_test = pre.imputer_.transform(test)
    generate_figures(clean, clean_test, probabilities, metrics, histories, tnorm, result_dir)
    bundle = {'models': models, 'metadata': metadata, 'metrics': metrics}
    joblib.dump(bundle, model_dir / 'heart_models.joblib')
    reloaded = joblib.load(model_dir / 'heart_models.joblib')
    for name in models:
        np.testing.assert_allclose(reloaded['models'][name].predict_proba(test)[:, 1], probabilities[name])
    (result_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    (result_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    (result_dir / 'training_history.json').write_text(json.dumps(histories, indent=2))
    table = pd.DataFrame({name: {k: values[k] for k in METRICS} for name, values in metrics.items()}).T
    table.index.name = 'model'
    table.to_csv(result_dir / 'model_comparison.csv')
    folds.to_csv(result_dir / 'cv_folds.csv', index=False)
    cv_summary.to_csv(result_dir / 'cv_summary.csv', index=False)
    pd.DataFrame({'patient_index': test.index, 'actual': test.HeartDisease.to_numpy(),
                  **probabilities}).to_csv(result_dir / 'test_predictions.csv', index=False)
    # Only real held-out examples are exported.
    test.head(4).to_csv(result_dir / 'sample_patients.csv', index=False)
    print(table.to_string(float_format=lambda x: f'{x:.4f}'))
    print('\n5-fold CV (population standard deviation):\n', cv_summary.to_string(index=False))
    print(f'Artifacts saved to {root.resolve()}; all model reloads verified.')
    return bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data/heart.csv')
    parser.add_argument('--output', default='.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--tnorm', choices=['product', 'minimum'], default='product')
    parser.add_argument('--include-fri', action='store_true')
    parser.add_argument('--max-epochs', type=int, default=150)
    args = parser.parse_args()
    run(**vars(args))


if __name__ == '__main__':
    main()
