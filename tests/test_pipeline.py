from copy import deepcopy
import joblib
import numpy as np
import pandas as pd
import pytest
from src.data_processing import ClinicalImputer, validate, load_dataset, split_data
from src.feature_engineering import HeartFeatures
from src.neural_network import train_network
from src.predict import load_bundle, predict_patient
from src.evaluate import risk_band, evaluate
from tests.test_fuzzy_logic import patient


def test_missing_and_zero_train_only_imputation():
    train = pd.concat([patient(Cholesterol=200), patient(Cholesterol=300)], ignore_index=True)
    imputer = ClinicalImputer().fit(train)
    assert imputer.transform(patient(Cholesterol=0)).Cholesterol.iloc[0] == 250
    assert imputer.transform(patient(Cholesterol=np.nan)).Cholesterol.iloc[0] == 250
    imputer.transform(patient(Cholesterol=999))
    assert imputer.fill_['Cholesterol'] == 250


def test_invalid_patient():
    for override in [dict(Age=-1), dict(Sex='Unknown'), dict(FastingBS=2), dict(MaxHR=np.inf)]:
        with pytest.raises(ValueError):
            validate(patient(**override))
    with pytest.raises(ValueError):
        validate(patient().drop(columns='Age'))


def test_feature_dimensions_and_scaler_isolation():
    train = pd.concat([patient(Age=30), patient(Age=70)], ignore_index=True)
    for fuzzy, fri, dimensions in [(False, False, 18), (True, False, 38), (True, True, 39)]:
        features = HeartFeatures(fuzzy, 'product', fri).fit(train)
        assert features.transform(patient()).shape == (1, dimensions)
        assert len(features.get_feature_names_out()) == dimensions
        before = features.scaler_.mean_.copy()
        features.transform(patient(Age=100))
        np.testing.assert_array_equal(features.scaler_.mean_, before)


@pytest.mark.parametrize('p,band', [(0, 'Low'), (.3299, 'Low'), (.33, 'Moderate'), (.5, 'Moderate'), (.6599, 'Moderate'), (.66, 'High'), (1, 'High')])
def test_band_thresholds(p, band):
    assert risk_band(p) == band


def test_measured_metrics():
    metrics = evaluate([0, 0, 1, 1], [.1, .9, .2, .8])
    assert metrics['accuracy'] == .5
    assert metrics['confusion_matrix'] == [[1, 1], [1, 1]]


def test_best_checkpoint_restoration(monkeypatch):
    import src.neural_network as module

    class ControlledNetwork:
        def __init__(self, **kwargs):
            self.epoch = 0

        def partial_fit(self, X, y, classes):
            self.epoch += 1
            return self

        def predict_proba(self, X):
            # Epoch two is best; later epochs degrade despite continued fitting.
            p = {1: .6, 2: .9}.get(self.epoch, .55)
            probabilities = np.array([1-p, p])
            return np.column_stack([1-probabilities, probabilities])

    monkeypatch.setattr(module, 'MLPClassifier', ControlledNetwork)
    model, history = module.train_network(np.zeros((2, 1)), [0, 1], np.zeros((2, 1)), [0, 1], patience=3)
    assert model.epoch == model.best_epoch_ == 2
    assert len(history) == 5


def test_dataset_and_saved_models_when_available(tmp_path):
    from pathlib import Path
    if not Path('data/heart.csv').exists() or not Path('models/heart_models.joblib').exists():
        pytest.skip('Requires real dataset and trained artifacts; run python -m src.train')
    frame = load_dataset()
    train, validation, test = split_data(frame)
    assert (len(train), len(validation), len(test)) == (624, 110, 184)
    assert not set(train.index) & set(test.index)
    bundle = load_bundle()
    result = predict_patient(test.iloc[0].to_dict(), bundle)
    assert set(result['predictions']) == set(bundle['models'])
    for prediction in result['predictions'].values():
        assert 0 <= prediction['probability'] <= 1
    path = tmp_path / 'reload.joblib'
    joblib.dump(bundle, path)
    loaded = joblib.load(path)
    assert predict_patient(test.iloc[0].to_dict(), loaded) == result
    for name, model in bundle['models'].items():
        np.testing.assert_allclose(model.predict_proba(test).sum(axis=1), 1)
