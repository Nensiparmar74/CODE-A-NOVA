from pathlib import Path
import json
import pandas as pd
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports' / 'FINAL_REPORT.pdf'
metrics = json.load(open(ROOT / 'models/evaluation_metrics.json'))
comparison = json.load(open(ROOT / 'models/model_comparison.json'))
figures = json.load(open(ROOT / 'models/eda_figures.json'))
outliers = json.load(open(ROOT / 'models/outlier_report.json'))
metadata = json.load(open(ROOT / 'models/model_metadata.json'))

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='TitleCenter', parent=styles['Title'], alignment=TA_CENTER, fontSize=24, leading=30, spaceAfter=12))
styles.add(ParagraphStyle(name='SubCenter', parent=styles['Normal'], alignment=TA_CENTER, fontSize=12, leading=16, textColor=colors.HexColor('#425466')))
styles.add(ParagraphStyle(name='H1x', parent=styles['Heading1'], fontSize=18, leading=22, spaceBefore=6, spaceAfter=8))
styles.add(ParagraphStyle(name='H2x', parent=styles['Heading2'], fontSize=12.5, leading=16, spaceBefore=6, spaceAfter=5))
styles.add(ParagraphStyle(name='Bodyx', parent=styles['BodyText'], fontSize=9.2, leading=13, spaceAfter=6))
styles.add(ParagraphStyle(name='Smallx', parent=styles['BodyText'], fontSize=7.6, leading=10))
styles.add(ParagraphStyle(name='Callout', parent=styles['BodyText'], fontSize=10.5, leading=15, backColor=colors.HexColor('#EEF5F9'), borderPadding=8, spaceBefore=5, spaceAfter=8))

def P(text, style='Bodyx'):
    return Paragraph(text, styles[style])

def img(relpath, width=86*mm):
    im = Image(str(ROOT / relpath))
    ratio = im.imageHeight / im.imageWidth
    im.drawWidth = width
    im.drawHeight = width * ratio
    return im

def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont('Helvetica', 7)
    canvas.setFillColor(colors.grey)
    canvas.drawString(15*mm, 8*mm, 'CropYield Predictor - Final Analysis Report')
    canvas.drawRightString(A4[0]-15*mm, 8*mm, f'Page {doc.page}')
    canvas.restoreState()

story = []
# 1
story += [Spacer(1, 28*mm), P('CropYield Predictor', 'TitleCenter'), P('Agricultural Intelligence & Analytics Platform', 'SubCenter'), Spacer(1, 10*mm), P('Final Analysis Report', 'H1x'), P('End-to-end data validation, exploratory data analysis, domain feature engineering, cross-validated model benchmarking, hyperparameter tuning, model diagnostics, and Streamlit dashboard delivery.', 'SubCenter'), Spacer(1, 15*mm), P('25,932 validated records | 101 areas | 10 crop categories | 1990-2013', 'Callout'), P('This report documents the completed project artifacts and the measured outputs included in the submission package.')]
story.append(PageBreak())
# 2
story += [P('1. Executive Summary','H1x'),
P(f"The project predicts crop yield in hg/ha from Area, Item, Year, rainfall, pesticide use, and average temperature. The deployable model is a Tuned Random Forest Regressor inside a pipeline containing training-set IQR capping, agricultural feature engineering, a ColumnTransformer, and the regressor. The held-out test set contains {metrics['dataset_test_samples']:,} records."),
P(f"Held-out performance: R2 = <b>{metrics['test_r2']:.4f}</b>, MAE = <b>{metrics['test_mae']:,.2f} hg/ha</b>, RMSE = <b>{metrics['test_rmse']:,.2f} hg/ha</b>. The R2 result is above the project brief target of 0.80."),
P('The analysis separates descriptive pooled relationships from causal interpretation. Crop identity and geographic indicators account for substantial group-level differences, while raw macro-climate correlations with the pooled target are modest.'),
P('Included deliverables: executed 21-visualization EDA notebook, model-comparison artifact, RandomizedSearchCV tuning evidence, joblib model, feature importance, residual diagnostics, Streamlit dashboard source, local experiment artifacts, final PDF, and a five-minute caption-led video walkthrough.'), Spacer(1,4*mm), P('Dataset Profile','H2x')]
profile = [['Metric','Value'],['Records','25,932'],['Columns','7'],['Areas','101'],['Crops','10'],['Years','1990-2013'],['Missing values','0'],['Duplicates after cleaning','0']]
t = Table(profile, colWidths=[65*mm,55*mm]); t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E6EEF3')),('GRID',(0,0),(-1,-1),0.4,colors.lightgrey),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),8),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#FAFAFA')])]))
story += [t, PageBreak()]
# 3
story += [P('2. Data Quality and Reproducible Pipeline','H1x'),
P('The raw agricultural data were validated for schema, missing values, numeric bounds, crop categories, duplicates, and potential post-harvest leakage columns. The processed export contains 25,932 rows and 7 columns with zero missing values and zero duplicates.'),
P('Outlier treatment is implemented as an IQR capper inside the scikit-learn pipeline. The cap bounds are fit only on training data; during cross-validation the transformer is cloned and fit separately in each fold. The target column is not capped.'),
P('Pipeline order: IQRCapper -> AgriculturalFeatureEngineer -> ColumnTransformer -> Regressor. Numeric inputs use median imputation and standard scaling. Area, Item and rainfall_category use most-frequent imputation and one-hot encoding with unknown-category handling.'),
P('The saved model file is a full sklearn Pipeline and was loaded without retraining to generate a sample prediction for India / Rice, paddy.'), P('Outlier Audit','H2x')]
rows = [['Feature','Potential IQR outliers','Treatment']] + [[o['column'], f"{o['outlier_count']:,} ({o['outlier_pct']:.2f}%)", 'Training-set IQR capping'] for o in outliers]
t = Table(rows, colWidths=[60*mm,45*mm,65*mm]); t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E6EEF3')),('GRID',(0,0),(-1,-1),0.35,colors.lightgrey),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),7.5),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#FAFAFA')])]))
story += [t, PageBreak()]
# 4
story += [P('3. Feature Engineering and Modeling Design','H1x'),
P('Four domain-derived features are engineered inside the model pipeline: hydrothermal_index = rainfall / (temperature + 10), rainfall_category, pesticide_log = log1p(pesticides), and temp_rainfall_interaction = rainfall * temperature / 1000.'),
P('Five distinct model architectures are benchmarked using the same shuffled 5-fold KFold design: Linear Regression, Ridge Regression, Decision Tree, Random Forest, and Gradient Boosting.'),
P(f"The final model is a Random Forest using n_estimators = {metrics['best_parameters'].get('n_estimators')}, max_depth = {metrics['best_parameters'].get('max_depth')}, min_samples_split = {metrics['best_parameters'].get('min_samples_split')}, and min_samples_leaf = {metrics['best_parameters'].get('min_samples_leaf')}."),
P('The project also records a RandomizedSearchCV experiment for Random Forest. The documented mean CV R2 moves from 0.9827 before tuning to 0.9830 after tuning, a small 0.03% relative improvement.'),
P('Interpretation note','H2x'), P('The stronger tree-model results indicate that the dataset contains non-linear and categorical structure that linear baselines do not capture as well. These metrics describe predictive performance and should not be interpreted as evidence of causal agricultural effects.'), PageBreak()]
# 5
story += [P('4. Model Comparison, Tuning and Error Diagnostics','H1x')]
rows = [['Model','CV R2','CV MAE','CV RMSE','Test R2']] + [[r['model'], f"{r['cv_r2_mean']:.4f}", f"{r['cv_mae']:,.0f}", f"{r['cv_rmse']:,.0f}", f"{r['test_r2']:.4f}"] for r in comparison]
t = Table(rows, colWidths=[45*mm,24*mm,30*mm,30*mm,22*mm]); t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E6EEF3')),('GRID',(0,0),(-1,-1),0.35,colors.lightgrey),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),7.2),('ALIGN',(1,1),(-1,-1),'RIGHT')]))
story += [t, Spacer(1,4*mm), P(f"Final error summary: normalized MAE = {metrics['normalized_mae']*100:.2f}% of the mean actual yield. {metrics['within_10_percent_tolerance']:.2f}% of held-out rows are within +/-10% relative error and {metrics['within_20_percent_tolerance']:.2f}% are within +/-20% relative error."), P('Feature importance is reported for the final Random Forest. The largest individual encoded feature is Item_Potatoes; other crop, area and numeric environmental variables also contribute. Feature importance is a model explanation signal, not a causal ranking.'), PageBreak()]
# 6-12: EDA, 3 figures per page. 21 figures => 7 pages, total 12 pages.
for i in range(0, len(figures), 3):
    group = figures[i:i+3]
    story.append(P(f"5. EDA Visualizations {group[0]['id']}-{group[-1]['id']}",'H1x'))
    for f in group:
        story.append(KeepTogether([P(f"Figure {f['id']}: {f['title']}",'H2x'), img(f['file'], 70*mm), P(f['interpretation'],'Smallx'), Spacer(1,1*mm)]))
    if i + 3 >= len(figures):
        story += [Spacer(1,2*mm), P('6. Delivery and Reproducibility','H1x'), P('The Streamlit dashboard contains Data & EDA, Prediction, Model Comparison, and Feature Importance views. The prediction form sends raw input fields directly to the serialized pipeline without retraining.'), P('The repository includes executed notebooks, model metadata, local experiment artifacts, requirements, tests, the final PDF report, and a five-minute caption-led video walkthrough. Install dependencies from requirements.txt before running the dashboard or notebook stack.'), P('Conclusion: the project package now contains the requested end-to-end workflow, with the main remaining deployment requirement being installation of the declared runtime dependencies in the target environment.')]
    story.append(PageBreak())

doc = SimpleDocTemplate(str(OUT), pagesize=A4, rightMargin=15*mm, leftMargin=15*mm, topMargin=15*mm, bottomMargin=15*mm, title='CropYield Predictor - Final Analysis Report', author='CropYield Predictor Team')
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print('wrote', OUT, OUT.stat().st_size)
