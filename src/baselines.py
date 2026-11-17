"""Zero-training fuzzy estimator and logistic-regression factory."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from src.data_processing import ClinicalImputer
from src.fuzzy_logic import infer


class FuzzyClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, tnorm: str = 'product'):
        self.tnorm = tnorm

    def fit(self, X, y=None):
        # Only missing-value preprocessing is fitted; rule parameters never learn.
        self.imputer_ = ClinicalImputer().fit(X)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = infer(self.imputer_.transform(X), self.tnorm)[2]
        return np.column_stack([1-p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= .5).astype(int)


def logistic_regression(seed: int = 42) -> LogisticRegression:
    return LogisticRegression(max_iter=1000, random_state=seed)
