"""Stable, named 18-dimensional raw and 38-dimensional fuzzy features."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.data_processing import ClinicalImputer, NUMERIC, CATEGORIES
from src.fuzzy_logic import infer, PARAMETERS, TERMS

ONEHOT = ['ChestPainType', 'RestingECG', 'ST_Slope']


class HeartFeatures(TransformerMixin, BaseEstimator):
    def __init__(self, fuzzy: bool = False, tnorm: str = 'product', include_fri: bool = False):
        self.fuzzy = fuzzy
        self.tnorm = tnorm
        self.include_fri = include_fri

    def fit(self, X: pd.DataFrame, y=None):
        self.imputer_ = ClinicalImputer().fit(X)
        clean = self.imputer_.transform(X)
        self.encoder_ = OneHotEncoder(categories=[CATEGORIES[k] for k in ONEHOT],
                                      handle_unknown='error', sparse_output=False)
        self.encoder_.fit(clean[ONEHOT])
        self.scaler_ = StandardScaler().fit(clean[NUMERIC])
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        clean = self.imputer_.transform(X)
        remaining = np.column_stack([clean.FastingBS, (clean.Sex == 'M').astype(float),
                                     (clean.ExerciseAngina == 'Y').astype(float),
                                     self.encoder_.transform(clean[ONEHOT])])
        if not self.fuzzy:
            return np.column_stack([self.scaler_.transform(clean[NUMERIC]), remaining])
        membership, strengths, fri = infer(clean, self.tnorm)
        arrays = [membership.to_numpy(), strengths]
        if self.include_fri:
            arrays.append(fri[:, None])
        return np.column_stack(arrays + [remaining])

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        names = ([f'{k}_{t}' for k in PARAMETERS for t in TERMS] +
                 [f'R{i}' for i in range(1, 11)] + (['FRI'] if self.include_fri else [])
                 if self.fuzzy else NUMERIC.copy())
        return np.array(names + ['FastingBS', 'Sex_M', 'ExerciseAngina_Y'] +
                        list(self.encoder_.get_feature_names_out(ONEHOT)))
