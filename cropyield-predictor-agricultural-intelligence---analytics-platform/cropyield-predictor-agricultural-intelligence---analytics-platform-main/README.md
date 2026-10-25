# CropYield Predictor - Agricultural Intelligence & Analytics Platform

End-to-end data science project for crop-yield prediction using agricultural, climate and agrochemical features.

## What is included

- Cleaned dataset and data dictionary
- Data validation module with pytest tests
- 21-visualization executed EDA notebook with written interpretations
- Leak-free scikit-learn pipeline: IQR outlier capping -> agricultural feature engineering -> ColumnTransformer -> Random Forest
- Five-model 5-fold cross-validation comparison
- RandomizedSearchCV tuning evidence
- Working `models/crop_yield_model.joblib` artifact that loads without retraining
- Feature-importance and residual diagnostics artifacts
- Streamlit dashboard with Data & EDA, Prediction, Model Comparison, and Feature Importance views
- Local experiment-run metadata under `mlruns/` and MLflow API logging code in `src/train.py`
- 12-page final PDF analysis report
- Five-minute caption-led demo video

## Measured final model result

Held-out test set: 5,187 records.

- R2: 0.9813
- MAE: 4,589.79 hg/ha
- RMSE: 11,654.74 hg/ha
- Normalized MAE: 5.85% of mean actual yield
- Within +/-10% relative error: 75.03%
- Within +/-20% relative error: 87.83%

RandomizedSearchCV record for Random Forest: pre-tuning CV R2 = 0.9827, post-tuning CV R2 = 0.9830, recorded relative improvement = 0.03%.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# Validate
pytest -q

# Launch dashboard
streamlit run dashboard/app.py
```

To regenerate EDA figures and executed notebooks:

```bash
python scripts/generate_eda.py
python scripts/make_notebooks.py
```

To rebuild the final model artifact quickly from the verified Random Forest parameter set:

```bash
python scripts/build_final_model.py
```

For a full retraining and fresh 5-fold benchmark + tuning run:

```bash
python src/train.py
```

## Notes on experiment tracking

The training code uses the MLflow Python API when `mlflow` is installed. This build environment did not have network access to install the MLflow package, so portable local run metadata is included under `mlruns/` and the API integration remains in `src/train.py` for the target environment.
