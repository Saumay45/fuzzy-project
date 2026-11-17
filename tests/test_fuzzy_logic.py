import numpy as np
import pandas as pd
import pytest
from src.fuzzy_logic import triangular, fuzzify, firing_strengths, defuzzify, infer, PARAMETERS
from src.fuzzy_rules import RULES


def patient(**overrides):
    values = dict(Age=52, Sex='F', ChestPainType='NAP', RestingBP=130,
                  Cholesterol=225, FastingBS=0, RestingECG='Normal',
                  MaxHR=130, ExerciseAngina='N', Oldpeak=1.1, ST_Slope='Up')
    values.update(overrides)
    return pd.DataFrame([values])


def test_triangle_boundaries_and_midpoints():
    np.testing.assert_allclose(triangular([-1, 0, 1, 2, 3, 4, 5], 0, 2, 4), [0, 0, .5, 1, .5, 0, 0])
    assert triangular(58, 52, 68, 82) == pytest.approx(.375)


@pytest.mark.parametrize('params', [t for triples in PARAMETERS.values() for t in triples])
def test_every_report_triangle(params):
    a, b, c = params
    np.testing.assert_allclose(triangular([a, b, c], a, b, c), [0, 1, 0])


def test_invalid_triangle():
    with pytest.raises(ValueError):
        triangular(1, 0, 0, 2)
    with pytest.raises(ValueError):
        triangular(float('nan'), 0, 1, 2)


def test_no_rule_fallback():
    frame = patient()
    assert infer(frame)[1].sum() == 0
    assert infer(frame)[2][0] == .5


def test_product_and_minimum_differ():
    frame = patient(Age=60, Oldpeak=2.3)
    m = fuzzify(frame)
    product = firing_strengths(frame, m, 'product')[0, 0]
    minimum = firing_strengths(frame, m, 'minimum')[0, 0]
    assert product == pytest.approx(.5 * .5)
    assert minimum == pytest.approx(.5)


@pytest.mark.parametrize('index,rule', list(enumerate(RULES)))
def test_all_ten_rules_reach_unit_activation(index, rule):
    overrides = {}
    for key, value in rule[1]:
        overrides[key] = PARAMETERS[key][['Low', 'Medium', 'High'].index(value)][1] if key in PARAMETERS else value
    frame = patient(**overrides)
    assert infer(frame)[1][0, index] == pytest.approx(1)
    assert infer(frame, 'minimum')[1][0, index] == pytest.approx(1)


def test_defuzzification_solved_example():
    strengths = np.zeros((1, 10))
    strengths[0, 0], strengths[0, 8] = .2, .8
    assert defuzzify(strengths)[0] == pytest.approx(.23)


def test_actual_patient_solved_example():
    frame = patient(Age=60, Oldpeak=2.3, ST_Slope='Flat', FastingBS=1)
    _, strength, fri = infer(frame)
    np.testing.assert_allclose(strength[0], [.25, 0, 0, .5, 0, 0, .5, 0, 0, 0], atol=1e-12)
    assert fri[0] == pytest.approx(.85)


def test_crisp_rule_down_slope():
    _, strengths, fri = infer(patient(ST_Slope='Down'))
    assert strengths[0, 4] == 1
    assert fri[0] == .85


def test_invalid_strengths():
    with pytest.raises(ValueError):
        defuzzify(np.zeros((1, 9)))
    with pytest.raises(ValueError):
        defuzzify(np.full((1, 10), -1))
