"""Validation, train-only imputation, and reproducible patient splits."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

NUMERIC = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
CATEGORIES = {'Sex': ['F', 'M'], 'ChestPainType': ['ASY', 'ATA', 'NAP', 'TA'],
              'RestingECG': ['LVH', 'Normal', 'ST'], 'ExerciseAngina': ['N', 'Y'],
              'ST_Slope': ['Down', 'Flat', 'Up']}
FEATURES = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS',
            'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']
# Broad input limits, not diagnostic thresholds. Zero BP/cholesterol means missing.
LIMITS = {'Age': (1, 120), 'RestingBP': (0, 300), 'Cholesterol': (0, 1000),
          'MaxHR': (1, 300), 'Oldpeak': (-10, 15)}


def validate(frame: pd.DataFrame, target: bool = False) -> pd.DataFrame:
    """Reject malformed, unknown or nonfinite data; allow missing predictors."""
    required = FEATURES + (['HeartDisease'] if target else [])
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    if frame.empty:
        raise ValueError('No patient records supplied')
    out = frame[required].copy()
    for key in NUMERIC + ['FastingBS'] + (['HeartDisease'] if target else []):
        out[key] = pd.to_numeric(out[key], errors='raise')
        if np.isinf(out[key].to_numpy(dtype=float)).any():
            raise ValueError(f'{key} must be finite')
    for key, values in CATEGORIES.items():
        if not out[key].dropna().isin(values).all():
            raise ValueError(f'{key} must be one of {values}')
    for key, (low, high) in LIMITS.items():
        if not out[key].dropna().between(low, high).all():
            raise ValueError(f'{key} must be between {low} and {high}')
    for key in ['FastingBS'] + (['HeartDisease'] if target else []):
        if not out[key].dropna().isin([0, 1]).all():
            raise ValueError(f'{key} must be 0 or 1')
    if target and out.HeartDisease.isna().any():
        raise ValueError('Target cannot be missing')
    return out


def load_dataset(path: str | Path = 'data/heart.csv') -> pd.DataFrame:
    """Load the specified real dataset and check its expected identity counts."""
    if not Path(path).exists():
        raise FileNotFoundError(f'{path} absent. Run python scripts/download_data.py; see data/README.md')
    frame = validate(pd.read_csv(path), target=True)
    counts = frame.HeartDisease.value_counts().to_dict()
    if len(frame) != 918 or counts != {1: 508, 0: 410}:
        raise ValueError(f'Expected 918 records and labels 508/410; got {len(frame)}, {counts}')
    if frame.duplicated().any():
        raise ValueError('Dataset contains duplicated patient records')
    return frame


class ClinicalImputer(TransformerMixin, BaseEstimator):
    """Train-only numeric medians and categorical/binary modes."""
    def fit(self, X: pd.DataFrame, y=None):
        frame = validate(X)
        frame[['RestingBP', 'Cholesterol']] = frame[['RestingBP', 'Cholesterol']].replace(0, np.nan)
        self.fill_ = {}
        for key in FEATURES:
            values = frame[key].dropna()
            if values.empty:
                raise ValueError(f'Cannot fit imputer: {key} has no observed training values')
            self.fill_[key] = values.median() if key in NUMERIC else values.mode().iloc[0]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = validate(X)
        frame[['RestingBP', 'Cholesterol']] = frame[['RestingBP', 'Cholesterol']].replace(0, np.nan)
        return frame.fillna(self.fill_)


def split_data(frame: pd.DataFrame, seed: int = 42) -> tuple:
    """624 training, 110 validation, 184 held-out test records."""
    development, test = train_test_split(frame, test_size=.2, stratify=frame.HeartDisease, random_state=seed)
    # sklearn rounds fractional test_size upward (734 * .15 -> 111).
    # Use an explicit rounded count to reproduce Table 6's 110 patients.
    train, validation = train_test_split(development, test_size=round(len(development) * .15),
                                        stratify=development.HeartDisease, random_state=seed)
    return train, validation, test
