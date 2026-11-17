"""Vectorized NumPy fuzzification and zero-order Sugeno inference."""
import numpy as np
import pandas as pd
from src.fuzzy_rules import RULES, CONSEQUENTS

PARAMETERS = {
    'Age': [(20, 30, 46), (36, 52, 62), (52, 68, 82)],
    'RestingBP': [(80, 100, 122), (112, 130, 150), (138, 160, 200)],
    'Cholesterol': [(100, 150, 205), (185, 225, 260), (245, 300, 400)],
    'MaxHR': [(60, 90, 112), (100, 130, 152), (140, 170, 202)],
    'Oldpeak': [(-3, 0, .6), (.3, 1.1, 2.1), (1.6, 3, 6.5)],
}
TERMS = ('Low', 'Medium', 'High')


def triangular(x, a: float, b: float, c: float) -> np.ndarray:
    """Triangle with zero endpoints, unit peak, and zero outside support."""
    if not a < b < c:
        raise ValueError('Triangle requires a < b < c')
    x = np.asarray(x, dtype=float)
    if not np.isfinite(x).all():
        raise ValueError('Membership inputs must be finite')
    return np.maximum(np.minimum((x-a)/(b-a), (c-x)/(c-b)), 0.)


def fuzzify(frame: pd.DataFrame) -> pd.DataFrame:
    """Return fifteen memberships in Table 4 order."""
    return pd.DataFrame({f'{key}_{term}': triangular(frame[key], *params)
                         for key, triples in PARAMETERS.items()
                         for term, params in zip(TERMS, triples)}, index=frame.index)


def firing_strengths(frame: pd.DataFrame, membership: pd.DataFrame,
                     tnorm: str = 'product') -> np.ndarray:
    """Apply Table 5 with report product AND or requested minimum AND."""
    if tnorm not in ('product', 'minimum'):
        raise ValueError('tnorm must be product or minimum')
    strengths = []
    for _, conditions, _ in RULES:
        values = [membership[f'{key}_{value}'].to_numpy() if key in PARAMETERS
                  else (frame[key].to_numpy() == value).astype(float)
                  for key, value in conditions]
        strengths.append(np.prod(values, axis=0) if tnorm == 'product'
                         else np.min(values, axis=0))
    return np.column_stack(strengths)


def defuzzify(strengths: np.ndarray) -> np.ndarray:
    """Weighted singleton average; total strength <= 1e-12 defaults to .5."""
    strengths = np.asarray(strengths, dtype=float)
    if strengths.ndim != 2 or strengths.shape[1] != 10:
        raise ValueError('Expected shape (patients, 10)')
    if not np.isfinite(strengths).all() or ((strengths < 0) | (strengths > 1)).any():
        raise ValueError('Strengths must be finite and within [0, 1]')
    total = strengths.sum(axis=1)
    return np.divide(strengths @ CONSEQUENTS, total,
                     out=np.full(len(total), .5), where=total > 1e-12)


def infer(frame: pd.DataFrame, tnorm: str = 'product') -> tuple:
    """Memberships, firing strengths, and FRI for already cleaned inputs."""
    membership = fuzzify(frame)
    strengths = firing_strengths(frame, membership, tnorm)
    return membership, strengths, defuzzify(strengths)
