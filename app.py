"""Local Streamlit demonstration: streamlit run app.py --server.address 127.0.0.1"""
from pathlib import Path
import json
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from src.predict import load_bundle, predict_patient
from src.fuzzy_logic import PARAMETERS, TERMS, triangular

ROOT = Path(os.environ.get('HEART_ARTIFACT_ROOT', Path(__file__).resolve().parent)).resolve()
st.set_page_config(page_title='HeartScope • Neuro-Fuzzy Lab', page_icon='🫀', layout='wide')
st.markdown('''<style>
.stApp {background: linear-gradient(145deg,#071521,#112a38);color:#edf5f7;}
[data-testid="stMetric"] {background:#173544;border:1px solid #315361;border-radius:16px;padding:18px;}
h1,h2,h3 {letter-spacing:-.025em;}
[data-testid="stSidebar"] {background:#0a1d29;}
</style>''', unsafe_allow_html=True)
st.title('🫀 HeartScope')
st.caption('NEURO-FUZZY HEART HEALTH LAB · From clinical attributes to traceable fuzzy evidence')
st.warning('Academic demonstration only. Do not use for medical diagnosis or clinical decisions. '
           'Output scores are not clinically calibrated risks.')


@st.cache_resource
def cached_bundle(path, modified_ns):
    return load_bundle(path)


artifact = ROOT / 'models/heart_models.joblib'
bundle = None
if artifact.exists():
    try:
        bundle = cached_bundle(str(artifact), artifact.stat().st_mtime_ns)
    except Exception as exc:
        st.error(f'Unable to load saved model: {exc}. Retrain with the installed package versions.')
else:
    st.info('No trained models yet. Install requirements, place data/heart.csv, then run: python -m src.train')

page = st.sidebar.radio('Workspace', ['Dashboard', 'Heart Health Prediction',
                                      'Fuzzy Logic Visualization', 'Model Comparison', 'Explain the Prediction'])
st.sidebar.caption('Binary class: probability ≥ 0.50 → disease present.\n\n'
                   'Illustrative bands: Low < 0.33 · Moderate < 0.66 · High ≥ 0.66.')
if bundle:
    st.sidebar.caption(f"Saved configuration: {bundle['metadata']['tnorm']} AND · "
                       f"{len(bundle['metadata']['neuro_feature_names'])} neural inputs")


def show_trace(result):
    rules = pd.DataFrame(result['rules'])
    st.plotly_chart(px.bar(rules, x='rule', y='strength', color='consequent',
                          range_y=[0, 1], color_continuous_scale='Tealrose',
                          hover_data=['antecedents', 'fri_contribution'],
                          title='Which rules fired?'), width='stretch')
    st.dataframe(rules, hide_index=True, width='stretch')
    if result['no_rules_fired']:
        st.info('No rules fired: total strength ≤ 1e-12, so FRI defaults to 0.5. '
                'At the binary threshold this maps to class 1; it means uncertain fuzzy evidence.')
    else:
        numerator = sum(r['strength'] * r['consequent'] for r in result['rules'])
        denominator = sum(r['strength'] for r in result['rules'])
        st.latex(r'\mathrm{FRI}=\frac{\sum_r f_r c_r}{\sum_r f_r}'
                 + rf'=\frac{{{numerator:.5f}}}{{{denominator:.5f}}}={result["fuzzy_risk_index"]:.5f}')
    st.caption('FRI contributions sum to FRI when rules fire. These explain the fixed fuzzy front end; '
               'they do not establish the neural network’s complete reasoning or feature importance.')


if page == 'Dashboard':
    st.subheader('An interpretable front end. A learned classifier.')
    st.write('Five numeric measurements become fifteen overlapping membership degrees. '
             'Ten fixed rules extract fuzzy evidence. A 16 → 8 neural network combines that evidence '
             'with the encoded clinical attributes to classify the dataset’s HeartDisease outcome.')
    if bundle:
        meta = bundle['metadata']
        columns = st.columns(4)
        for col, label, value in zip(columns, ['Real patient records', 'Disease present', 'Held-out patients', 'Neural inputs'],
                                     [meta['dataset_rows'], meta['class_counts']['1'], meta['split_sizes']['test'], len(meta['neuro_feature_names'])]):
            col.metric(label, value)
        score = bundle['metrics']['Neuro-Fuzzy ANN']
        st.subheader('Proposed model · measured held-out results')
        for col, key in zip(st.columns(5), ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']):
            col.metric(key.replace('_', ' ').title(), f'{score[key]:.3f}')
        for name in ['class_distribution', 'feature_distributions', 'correlation_heatmap', 'training_curves', 'risk_distribution']:
            path = ROOT / 'results' / f'{name}.png'
            if path.exists():
                with st.expander(name.replace('_', ' ').title()):
                    st.image(str(path), width='stretch')
        with st.expander('Reproducibility and preprocessing'):
            st.json(meta)
    else:
        st.code('python -m pip install -r requirements.txt\npython scripts/download_data.py\npython -m src.train')
        st.write('No evaluation values are displayed until real training has completed.')

elif page == 'Heart Health Prediction':
    st.subheader('Explore a patient profile')
    st.caption('The initial form values are illustrative inputs, not a dataset record or evaluation evidence. '
               'BP or cholesterol set to zero will use the saved training median.')
    with st.form('patient'):
        left, middle, right = st.columns(3)
        with left:
            age = st.slider('Age (years)', 18, 100, 55)
            sex = st.selectbox('Sex (dataset coding)', ['F', 'M'])
            chest = st.selectbox('Chest pain type', ['ASY', 'ATA', 'NAP', 'TA'],
                                  format_func=lambda x: {'ASY': 'Asymptomatic', 'ATA': 'Atypical angina', 'NAP': 'Non-anginal pain', 'TA': 'Typical angina'}[x])
            bp = st.number_input('Resting BP (mm Hg)', min_value=0, max_value=300, value=130)
        with middle:
            cholesterol = st.number_input('Cholesterol (mg/dl)', min_value=0, max_value=1000, value=230)
            fasting = st.selectbox('Fasting blood sugar > 120 mg/dl', [0, 1], format_func=lambda x: 'Yes' if x else 'No')
            ecg = st.selectbox('Resting ECG', ['Normal', 'ST', 'LVH'])
            maxhr = st.slider('Maximum heart rate (bpm)', 40, 250, 140)
        with right:
            angina = st.selectbox('Exercise-induced angina', ['N', 'Y'])
            oldpeak = st.number_input('Oldpeak / ST depression (mm)', min_value=-10., max_value=15., value=1., step=.1)
            slope = st.selectbox('ST slope', ['Up', 'Flat', 'Down'])
        submitted = st.form_submit_button('Assess heart health', type='primary', disabled=bundle is None)
    if submitted and bundle:
        patient = dict(Age=age, Sex=sex, ChestPainType=chest, RestingBP=bp,
                       Cholesterol=cholesterol, FastingBS=fasting, RestingECG=ecg,
                       MaxHR=maxhr, ExerciseAngina=angina, Oldpeak=oldpeak, ST_Slope=slope)
        try:
            st.session_state['result'] = predict_patient(patient, bundle)
            st.session_state['patient_inputs'] = patient
        except (ValueError, TypeError) as exc:
            st.error(f'Please check the patient inputs: {exc}')
    if 'result' in st.session_state:
        result = st.session_state['result']
        prediction = result['predictions']['Neuro-Fuzzy ANN']
        cols = st.columns(4)
        cols[0].metric('Predicted class', 'Disease present' if prediction['class'] else 'Disease absent')
        cols[1].metric('Model probability', f"{prediction['probability']:.1%}")
        cols[2].metric('Illustrative band', prediction['band'])
        cols[3].metric('Fuzzy Risk Index', f"{result['fuzzy_risk_index']:.3f}")
        st.caption('Class uses 0.50; bands use 0.33 and 0.66. A Moderate band can therefore have either class.')
        if result['imputed_fields']:
            st.info('Used training imputation for: ' + ', '.join(result['imputed_fields']))
        show_trace(result)
        st.dataframe(pd.Series(result['memberships'], name='Membership degree'), width='stretch')
        comparison = pd.DataFrame(result['predictions']).T.reset_index(names='model')
        st.plotly_chart(px.bar(comparison, x='model', y='probability', color='band', range_y=[0, 1]), width='stretch')

elif page == 'Fuzzy Logic Visualization':
    st.subheader('Explore overlapping memberships')
    key = st.selectbox('Clinical attribute', list(PARAMETERS))
    triples = PARAMETERS[key]
    value = st.slider('Measurement', float(triples[0][0]), float(triples[-1][2]), float(triples[1][1]), step=.1)
    x = np.linspace(triples[0][0], triples[-1][2], 500)
    fig = go.Figure()
    for term, params in zip(TERMS, triples):
        fig.add_trace(go.Scatter(x=x, y=triangular(x, *params), name=term))
        fig.add_trace(go.Scatter(x=[value], y=[float(triangular(value, *params))],
                                 mode='markers', name=f'{term} at {value:.1f}'))
    fig.add_vline(x=value, line_dash='dash')
    fig.update_layout(xaxis_title=key, yaxis_title='Membership degree', yaxis_range=[0, 1.05])
    st.plotly_chart(fig, width='stretch')
    st.latex(r'\mu(x;a,b,c)=\max\left(\min\left(\frac{x-a}{b-a},\frac{c-x}{c-b}\right),0\right)')
    st.write('Exact Table 4 parameters:', dict(zip(TERMS, triples)))
    st.caption('Membership degree is compatibility with a fuzzy term, not a disease probability.')
    if 'result' in st.session_state:
        st.subheader('Current patient’s fuzzy evidence')
        show_trace(st.session_state['result'])
    else:
        st.info('Submit a patient on Heart Health Prediction to explore its rule activations.')

elif page == 'Model Comparison':
    st.subheader('Measured results on the same held-out patients')
    if bundle:
        metrics = bundle['metrics']
        table = pd.DataFrame({name: {k: v[k] for k in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']}
                              for name, v in metrics.items()}).T
        st.dataframe(table.style.format('{:.3f}'), width='stretch')
        metric = st.selectbox('Compare metric', list(table.columns))
        st.plotly_chart(px.bar(table.reset_index(names='model'), x='model', y=metric,
                              color='model', range_y=[0, 1]), width='stretch')
        name = st.selectbox('Confusion matrix', list(metrics))
        st.plotly_chart(px.imshow(metrics[name]['confusion_matrix'], text_auto=True,
                                  x=['Absent', 'Present'], y=['Absent', 'Present'],
                                  labels={'x': 'Predicted', 'y': 'Actual'}, color_continuous_scale='Blues'), width='stretch')
        fig = go.Figure()
        for name, score in metrics.items():
            fig.add_trace(go.Scatter(x=score['roc']['fpr'], y=score['roc']['tpr'],
                                    name=f"{name}: AUC {score['roc_auc']:.3f}"))
        fig.add_shape(type='line', x0=0, y0=0, x1=1, y1=1, line={'dash': 'dash'})
        fig.update_layout(xaxis_title='False positive rate', yaxis_title='True positive rate')
        st.plotly_chart(fig, width='stretch')
        cv_file = ROOT / 'results/cv_summary.csv'
        if cv_file.exists():
            st.subheader('Five-fold stratified cross-validation')
            st.dataframe(pd.read_csv(cv_file), hide_index=True, width='stretch')
            st.caption('Independent preprocessing and early stopping within each fold. '
                       'CV rotates all 918 records, as in the report; it is not another untouched test cohort. '
                       'Standard deviation uses ddof=0.')
        st.download_button('Download actual comparison CSV', table.to_csv(), 'model_comparison.csv', 'text/csv')
    else:
        st.info('Train the models to populate this view with actual measured results.')

elif page == 'Explain the Prediction':
    st.subheader('Trace the fuzzy front end')
    st.write('Each rule combines degrees of membership and crisp conditions. Its singleton consequent '
             'is fixed. Activated rules vote in a weighted average to form FRI. The neural network '
             'then processes the full feature vector, and may disagree with that fuzzy average.')
    if 'result' in st.session_state:
        result = st.session_state['result']
        activated = [r for r in result['rules'] if r['strength'] > 1e-12]
        for rule in sorted(activated, key=lambda r: r['strength'], reverse=True):
            st.write(f"**{rule['rule']}**: {rule['antecedents']}. Strength **{rule['strength']:.3f}**, "
                     f"singleton **{rule['consequent']:.2f}**, weighted FRI contribution **{rule['fri_contribution']:.3f}**.")
        show_trace(result)
        st.download_button('Download prediction trace', json.dumps(result, indent=2), 'prediction_trace.json', 'application/json')
    else:
        st.info('Submit a patient first to see its memberships and activated rules here.')
