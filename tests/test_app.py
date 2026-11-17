"""Execute Streamlit pages and, when available, real saved-model prediction."""
from pathlib import Path
from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / 'app.py'


def test_pages_render_without_exceptions():
    app = AppTest.from_file(APP_PATH, default_timeout=30).run()
    assert not app.exception
    for page in ['Heart Health Prediction', 'Fuzzy Logic Visualization', 'Model Comparison', 'Explain the Prediction']:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception


def test_prediction_form_when_trained():
    import pytest
    if not Path('models/heart_models.joblib').exists():
        pytest.skip('Requires trained model artifacts')
    app = AppTest.from_file(APP_PATH, default_timeout=30).run()
    app.sidebar.radio[0].set_value('Heart Health Prediction').run()
    app.button[0].click().run()
    assert not app.exception
    assert 'result' in app.session_state
    assert len(app.metric) == 4
    first_probability = app.session_state['result']['predictions']['Neuro-Fuzzy ANN']['probability']
    # Submit different inputs in the same session, then inspect its explanation.
    app.slider[0].set_value(70)
    app.number_input[2].set_value(3.0)
    app.button[0].click().run()
    assert not app.exception
    assert app.session_state['patient_inputs']['Age'] == 70
    assert app.session_state['patient_inputs']['Oldpeak'] == 3.0
    second_probability = app.session_state['result']['predictions']['Neuro-Fuzzy ANN']['probability']
    assert first_probability != second_probability
    for page in ['Fuzzy Logic Visualization', 'Explain the Prediction']:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception
