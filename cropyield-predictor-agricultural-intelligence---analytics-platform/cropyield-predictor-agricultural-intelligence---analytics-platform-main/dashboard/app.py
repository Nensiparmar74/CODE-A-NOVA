"""CropYield Predictor - Streamlit analytics dashboard."""
import json
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/processed/clean_crop_yield.csv'
MODEL = ROOT / 'models/crop_yield_model.joblib'
COMPARISON = ROOT / 'models/model_comparison.json'
EVAL = ROOT / 'models/evaluation_metrics.json'
IMPORTANCE = ROOT / 'models/feature_importance.json'
RESIDUALS = ROOT / 'models/residuals.json'
EDA_SUMMARY = ROOT / 'models/eda_summary.json'
FIGURES = ROOT / 'reports/figures'

st.set_page_config(page_title='CropYield Predictor', page_icon='🌾', layout='wide')

@st.cache_data
def load_data(): return pd.read_csv(DATA)

@st.cache_resource
def load_model(): return joblib.load(MODEL)

def load_json(path):
    with open(path) as f: return json.load(f)

try:
    df = load_data(); model = load_model()
    comparison = load_json(COMPARISON); evaluation = load_json(EVAL); importance = load_json(IMPORTANCE); residuals = load_json(RESIDUALS); eda = load_json(EDA_SUMMARY)
    model_error = None
except Exception as exc:
    df = pd.DataFrame(); model = None; comparison=[]; evaluation={}; importance=[]; residuals=[]; eda={}; model_error=str(exc)

st.sidebar.title('🌱 CropYield Predictor')
st.sidebar.caption('Agricultural Intelligence & Analytics Platform')
st.sidebar.markdown('Leak-free IQR outlier capping + feature engineering + ColumnTransformer + model')

st.title('🌾 CropYield Predictor')
st.subheader('Agricultural Intelligence & Machine Learning Platform')
k1,k2,k3,k4=st.columns(4)
k1.metric('Validated Records',f'{len(df):,}')
k2.metric('Nations Monitored',str(df.Area.nunique()) if not df.empty else '-')
k3.metric('Crop Species',str(df.Item.nunique()) if not df.empty else '-')
k4.metric('Held-Out R²',f"{evaluation.get('test_r2',0):.4f}" if evaluation else '-')
if model_error: st.warning(f'Model loading error: {model_error}')

t1,t2,t3,t4=st.tabs(['📊 DATA & EDA','🔮 PREDICTION','⚖️ MODEL COMPARISON','🔬 FEATURE IMPORTANCE'])

with t1:
    st.header('Exploratory Data Analysis')
    if not df.empty:
        c1,c2=st.columns(2)
        with c1:
            st.subheader('Mean Yield by Crop')
            st.bar_chart(df.groupby('Item')['hg/ha_yield'].mean().sort_values(ascending=False))
        with c2:
            st.subheader('Annual Mean Yield')
            st.line_chart(df.groupby('Year')['hg/ha_yield'].mean())
        c3,c4=st.columns(2)
        with c3:
            st.subheader('Mean Yield by Rainfall Regime')
            cats=pd.cut(df.average_rain_fall_mm_per_year,[-1,600,1200,2000,10000],labels=['Low / Arid','Moderate','High','Tropical'])
            st.bar_chart(df.assign(rainfall_category=cats).groupby('rainfall_category',observed=True)['hg/ha_yield'].mean())
        with c4:
            st.subheader('Rainfall vs Yield')
            st.scatter_chart(df[['average_rain_fall_mm_per_year','hg/ha_yield']].sample(min(4000,len(df)),random_state=42), x='average_rain_fall_mm_per_year', y='hg/ha_yield')
        st.subheader('Correlation Matrix')
        numeric=['Year','average_rain_fall_mm_per_year','pesticides_tonnes','avg_temp','hg/ha_yield']
        st.dataframe(df[numeric].corr().round(3),use_container_width=True)
        st.caption('A complete 21-visualization EDA notebook and static figures are included in reports/figures/.')

with t2:
    st.header('Real-Time Crop Yield Prediction')
    areas=sorted(df.Area.unique()) if not df.empty else ['India']; items=sorted(df.Item.unique()) if not df.empty else ['Wheat']
    with st.form('prediction_form'):
        a,b=st.columns(2)
        with a:
            area=st.selectbox('Country / Area',areas,index=areas.index('India') if 'India' in areas else 0)
            item=st.selectbox('Crop Variety',items,index=items.index('Wheat') if 'Wheat' in items else 0)
            year=st.slider('Harvest Year',1990,2030,2024)
        with b:
            rain=st.number_input('Average annual rainfall (mm)',0.0,5000.0,1050.0,25.0)
            temp=st.slider('Average temperature (°C)',-10.0,45.0,24.0,0.5)
            pest=st.number_input('Pesticides (tonnes)',0.0,500000.0,45000.0,500.0)
        submit=st.form_submit_button('⚡ Predict Crop Yield',use_container_width=True)
    if submit:
        raw={'Area':area,'Item':item,'Year':int(year),'average_rain_fall_mm_per_year':float(rain),'pesticides_tonnes':float(pest),'avg_temp':float(temp)}
        if model is None: st.error('Saved model is unavailable.')
        else:
            pred=float(model.predict(raw)[0]); c1,c2,c3=st.columns(3); c1.metric('Yield (hg/ha)',f'{pred:,.1f}'); c2.metric('Yield (kg/ha)',f'{pred*0.1:,.1f}'); c3.metric('Yield (t/ha)',f'{pred*0.0001:.2f}')
            st.info(f'Prediction for {item} in {area}: {pred*0.0001:.2f} t/ha under the selected conditions.')

with t3:
    st.header('5-Fold Cross-Validation Benchmark')
    comp=pd.DataFrame(comparison)
    if not comp.empty:
        st.dataframe(comp,use_container_width=True)
        c1,c2=st.columns(2)
        with c1: st.bar_chart(comp.set_index('model')['cv_r2_mean'])
        with c2: st.bar_chart(comp.set_index('model')['cv_mae'])
        st.markdown('**Final model:** Tuned Random Forest Regressor')
        st.write(f"Pre-tuning CV R²: {evaluation.get('pre_tuning_cv_r2')} | Post-tuning CV R²: {evaluation.get('post_tuning_cv_r2')} | Held-out R²: {evaluation.get('test_r2')} | MAE: {evaluation.get('test_mae'):,} hg/ha")

with t4:
    st.header('Feature Importance & Residual Diagnostics')
    imp=pd.DataFrame(importance).head(15).sort_values('importance')
    if not imp.empty: st.bar_chart(imp.set_index('feature')['importance'])
    res=pd.DataFrame(residuals)
    if not res.empty:
        st.subheader('Actual vs Predicted (First 1000 Test Rows)')
        st.line_chart(res[['actual','predicted']].head(100))
        st.subheader('Residuals')
        st.line_chart(res[['residual']].head(100))
