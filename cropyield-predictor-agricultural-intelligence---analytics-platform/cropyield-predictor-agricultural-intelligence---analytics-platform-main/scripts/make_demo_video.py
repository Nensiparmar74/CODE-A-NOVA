from pathlib import Path
import json
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
SLIDE_DIR=ROOT/'reports'/'demo_slides'; SLIDE_DIR.mkdir(parents=True,exist_ok=True)
W,H=1280,720
try:
    FONT_BIG=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',42)
    FONT_MED=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',28)
    FONT_REG=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',22)
except Exception:
    FONT_BIG=FONT_MED=FONT_REG=None

def base(title, subtitle=''):
    im=Image.new('RGB',(W,H),'white'); d=ImageDraw.Draw(im); d.rectangle([0,0,W,90],fill='#16324F'); d.text((50,25),title,font=FONT_BIG,fill='white')
    if subtitle: d.text((52,110),subtitle,font=FONT_REG,fill='#334155')
    return im,d

def add_lines(d,lines,x=60,y=170,spacing=38):
    for i,line in enumerate(lines): d.text((x,y+i*spacing),line,font=FONT_REG,fill='#111827')

def add_chart(im,rel,x,y,w,h):
    chart=Image.open(ROOT/rel).convert('RGB'); chart.thumbnail((w,h)); im.paste(chart,(x,y))

# 1 Overview
im,d=base('CropYield Predictor','Agricultural Intelligence & Analytics Platform')
add_lines(d,['End-to-end crop yield prediction project','Data validation -> EDA -> feature engineering -> model benchmark -> tuning -> Streamlit dashboard','25,932 records | 101 areas | 10 crop categories | 1990-2013'],x=70,y=180,spacing=50)
im.save(SLIDE_DIR/'01_overview.png')
# 2 validation
im,d=base('Data Validation','Clean, leak-aware inputs for modeling')
add_lines(d,['Schema checked: 7 expected columns','Missing values after cleaning: 0','Duplicate rows after cleaning: 0','Range and category validation passed','Potential outliers handled with training-set IQR capping inside the sklearn pipeline'],y=180)
im.save(SLIDE_DIR/'02_validation.png')
# 3 crop EDA
im,d=base('EDA: Crop Yield Distribution','21 visualizations are included in the executed notebook')
add_chart(im,'reports/figures/01_mean_yield_by_crop.png',55,170,570,330); add_chart(im,'reports/figures/02_yield_distribution.png',650,170,570,330)
add_lines(d,['Crop identity creates large differences in baseline yield.','The pooled target is right-skewed.'],y=540,spacing=34); im.save(SLIDE_DIR/'03_eda_crop.png')
# 4 climate EDA
im,d=base('EDA: Climate and Inputs','Rainfall, temperature and pesticide patterns')
add_chart(im,'reports/figures/07_rainfall_distribution.png',55,170,570,300); add_chart(im,'reports/figures/09_temperature_distribution.png',650,170,570,300)
add_lines(d,['Rainfall spans multiple climatic regimes.','Temperature coverage is geographically broad.'],y=500,spacing=34); im.save(SLIDE_DIR/'04_eda_climate.png')
# 5 trends/correlation
im,d=base('EDA: Trends and Correlations','Descriptive pooled patterns')
add_chart(im,'reports/figures/05_annual_mean_trend.png',55,170,570,300); add_chart(im,'reports/figures/10_correlation_heatmap.png',650,170,570,300)
add_lines(d,['Pooled mean yield rises over 1990-2013 with a few local dips.','Raw linear climate correlations with yield are modest.'],y=500,spacing=34); im.save(SLIDE_DIR/'05_eda_trends.png')
# 6 FE
im,d=base('Feature Engineering','Four derived features are built inside the sklearn pipeline')
add_lines(d,['hydrothermal_index = rainfall / (temperature + 10)','rainfall_category = Low / Arid | Moderate | High | Tropical','pesticide_log = log1p(pesticides)','temp_rainfall_interaction = rainfall x temperature / 1000','Pipeline: IQRCapper -> FeatureEngineer -> ColumnTransformer -> Regressor'],y=175,spacing=48); im.save(SLIDE_DIR/'06_feature_engineering.png')
# 7 comparison
im,d=base('Model Comparison','Five distinct architectures evaluated with 5-fold cross-validation')
add_chart(im,'reports/figures/18_selected_crop_trends.png',50,155,500,300)
comp=json.load(open(ROOT/'models/model_comparison.json')); labels=[r['model'] for r in comp]; vals=[r['cv_r2_mean'] for r in comp]
fig=plt.figure(figsize=(6.8,3.8)); ax=fig.add_axes([.15,.2,.8,.7]); ax.bar(labels,vals); ax.set_ylabel('CV R2'); ax.set_ylim(0,1); ax.tick_params(axis='x',rotation=25,labelsize=8); ax.set_title('5-Fold CV R2 Comparison'); fig.savefig(SLIDE_DIR/'_modelcomp.png',dpi=140,bbox_inches='tight'); plt.close(fig)
add_chart(im,'reports/demo_slides/_modelcomp.png',590,155,640,300)
add_lines(d,['Final model: Tuned Random Forest Regressor','Held-out R2: 0.9813'],y=505,spacing=34); im.save(SLIDE_DIR/'07_model_comparison.png')
# 8 evaluation
im,d=base('Final Model Evaluation','Held-out test performance and residual diagnostics')
add_chart(im,'reports/figures/20_actual_vs_predicted.png',40,145,560,380); add_chart(im,'reports/figures/21_residual_distribution.png',640,145,560,320)
add_lines(d,['MAE: 4,589.79 hg/ha | RMSE: 11,654.74 hg/ha','Within +/-10% error: 75.03% | Within +/-20% error: 87.83%'],y=540,spacing=34); im.save(SLIDE_DIR/'08_evaluation.png')
# 9 dashboard preview
im,d=base('Streamlit Dashboard Preview','Four views: EDA | Prediction | Model Comparison | Feature Importance')
# light cards
for x,w in [(40,250),(305,250),(570,250),(835,385)]: d.rounded_rectangle([x,160,x+w,245],radius=12,outline='#CBD5E1',fill='#F8FAFC',width=2)
for x,txt in [(60,'25,932\nRecords'),(325,'101\nAreas'),(590,'10\nCrops'),(855,'R2\n0.9813')]: d.multiline_text((x,180),txt,font=FONT_MED,fill='#0F172A',spacing=5)
add_chart(im,'reports/figures/01_mean_yield_by_crop.png',50,285,540,300); add_chart(im,'reports/figures/19_feature_importance.png',640,285,570,300); im.save(SLIDE_DIR/'09_dashboard_preview.png')
# 10 close
im,d=base('Final Submission Package','Reproducibility and delivery artifacts')
add_lines(d,['Executed notebooks: validation, EDA, modeling','21 EDA figures with written interpretations','Working joblib model + metadata + diagnostics','Streamlit dashboard source','Local experiment-run metadata and MLflow API integration','12-page final PDF analysis report','Five-minute caption-led demo video','Pytest validation: 14 tests passed'],y=175,spacing=40)
im.save(SLIDE_DIR/'10_submission.png')

# generate concat list and MP4 externally with ffmpeg
print('slides',len(list(SLIDE_DIR.glob('0*.png'))))
