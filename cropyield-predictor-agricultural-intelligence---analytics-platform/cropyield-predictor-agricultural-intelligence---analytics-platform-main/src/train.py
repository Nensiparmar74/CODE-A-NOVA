"""CropYield Predictor - reproducible model training and experiment tracking."""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from src.feature_engineering import AgriculturalFeatureEngineer, get_feature_engineering_rationale
from src.pipeline_components import IQRCapper, iqr_outlier_report
from src.preprocessing import build_preprocessor, get_feature_column_names

CLEAN_DATA_PATH = 'data/processed/clean_crop_yield.csv'
MODEL_OUTPUT_PATH = 'models/crop_yield_model.joblib'
MODEL_COMPARISON_PATH = 'models/model_comparison.json'
EVALUATION_METRICS_PATH = 'models/evaluation_metrics.json'
FEATURE_IMPORTANCE_PATH = 'models/feature_importance.json'
RESIDUALS_PATH = 'models/residuals.json'
EDA_SUMMARY_PATH = 'models/eda_summary.json'
TUNING_RESULTS_PATH = 'models/tuning_results.json'
MODEL_METADATA_PATH = 'models/model_metadata.json'
OUTLIER_REPORT_PATH = 'models/outlier_report.json'
MLRUNS_PATH = Path('mlruns')

TARGET_COL = 'hg/ha_yield'
RAW_NUMERIC_FEATURES = [
    'Year',
    'average_rain_fall_mm_per_year',
    'pesticides_tonnes',
    'avg_temp',
]


def build_pipeline(estimator) -> Pipeline:
    """Build the full leak-free pipeline: IQR -> feature engineering -> preprocessing -> model."""
    return Pipeline([
        ('iqr_capper', IQRCapper(columns=RAW_NUMERIC_FEATURES, multiplier=1.5)),
        ('feature_engineer', AgriculturalFeatureEngineer()),
        ('preprocessor', build_preprocessor()),
        ('regressor', estimator),
    ])


def model_definitions() -> dict:
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=10.0),
        'Decision Tree': DecisionTreeRegressor(max_depth=20, random_state=42),
        'Random Forest': RandomForestRegressor(
            n_estimators=40, max_depth=20, random_state=42, n_jobs=1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=50, max_depth=4, learning_rate=0.08, random_state=42
        ),
    }
    return models


def generate_eda_summary(df: pd.DataFrame):
    crop_stats = df.groupby('Item')[TARGET_COL].agg(
        ['count', 'mean', 'median', 'std', 'min', 'max']
    ).reset_index().to_dict(orient='records')
    country_stats = df.groupby('Area')[TARGET_COL].agg(
        ['count', 'mean']
    ).sort_values('mean', ascending=False).reset_index().head(20).to_dict(orient='records')
    yearly = df.groupby('Year')[TARGET_COL].agg(['mean', 'count']).reset_index().to_dict(orient='records')
    numeric_cols = [
        'Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp', TARGET_COL
    ]
    corr = df[numeric_cols].corr().round(6).to_dict()
    summary = {
        'row_count': int(len(df)),
        'column_count': int(len(df.columns)),
        'unique_areas_count': int(df['Area'].nunique()),
        'unique_crops_count': int(df['Item'].nunique()),
        'year_min': int(df['Year'].min()),
        'year_max': int(df['Year'].max()),
        'missing_values_total': int(df.isna().sum().sum()),
        'duplicate_rows': int(df.duplicated().sum()),
        'crop_yield_stats': crop_stats,
        'top_countries_yield': country_stats,
        'yearly_trend': yearly,
        'correlation_matrix': corr,
    }
    with open(EDA_SUMMARY_PATH, 'w') as f:
        json.dump(summary, f, indent=2)
    return summary


def _start_mlflow_run(run_name: str):
    """Return a context manager when MLflow is available; otherwise return None."""
    try:
        import mlflow
        MLRUNS_PATH.mkdir(exist_ok=True)
        mlflow.set_tracking_uri(f"file://{MLRUNS_PATH.resolve()}")
        mlflow.set_experiment('CropYield Predictor')
        return mlflow.start_run(run_name=run_name), mlflow
    except Exception as exc:
        print(f'[MLflow] tracking unavailable: {exc}')
        return None, None


def _log_mlflow(run, mlflow, params: dict, metrics: dict):
    if run is None or mlflow is None:
        return
    mlflow.log_params(params)
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            mlflow.log_metric(key, float(value))
    mlflow.set_tag('project', 'CropYield Predictor')
    mlflow.set_tag('python_version', platform.python_version())


def tune_model(name, base_pipeline, X_train, y_train, cv):
    """Tune one candidate model using RandomizedSearchCV."""
    if name == 'Random Forest':
        params = {
            'regressor__n_estimators': [40, 60, 100],
            'regressor__max_depth': [15, 20, 25, None],
            'regressor__min_samples_split': [2, 5],
            'regressor__min_samples_leaf': [1, 2],
        }
        n_iter = 2
    elif name == 'Gradient Boosting':
        params = {
            'regressor__n_estimators': [40, 60, 100],
            'regressor__max_depth': [2, 3, 4, 5],
            'regressor__learning_rate': [0.03, 0.05, 0.08, 0.1],
            'regressor__min_samples_leaf': [1, 2, 4],
        }
        n_iter = 2
    else:
        raise ValueError(f'No tuning space defined for {name}')

    search = RandomizedSearchCV(
        base_pipeline,
        param_distributions=params,
        n_iter=n_iter,
        scoring='r2',
        cv=cv,
        random_state=42,
        n_jobs=1,
        refit=True,
        verbose=0,
    )
    search.fit(X_train, y_train)
    return search


def _feature_importance(final_pipeline):
    pre = final_pipeline.named_steps['preprocessor']
    reg = final_pipeline.named_steps['regressor']
    if not hasattr(reg, 'feature_importances_'):
        return []
    cat_names = pre.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(
        ['Area', 'Item', 'rainfall_category']
    )
    num_cols, _ = get_feature_column_names()
    names = list(num_cols) + list(cat_names)
    values = np.asarray(reg.feature_importances_)
    order = np.argsort(values)[::-1]
    return [
        {'feature': str(names[i]), 'importance': float(round(values[i], 6))}
        for i in order[:30]
    ]


def train_and_compare_models():
    os.makedirs('models', exist_ok=True)
    df = pd.read_csv(CLEAN_DATA_PATH)
    generate_eda_summary(df)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {'r2': 'r2', 'neg_mae': 'neg_mean_absolute_error', 'neg_rmse': 'neg_root_mean_squared_error'}
    results = []
    fitted = {}

    for name, estimator in model_definitions().items():
        pipe = build_pipeline(estimator)
        started = time.time()
        cv_res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        metrics = {
            'cv_r2_mean': round(float(np.mean(cv_res['test_r2'])), 4),
            'cv_r2_std': round(float(np.std(cv_res['test_r2'])), 4),
            'cv_mae': round(float(np.mean(-cv_res['test_neg_mae'])), 2),
            'cv_rmse': round(float(np.mean(-cv_res['test_neg_rmse'])), 2),
            'test_r2': round(float(r2_score(y_test, pred)), 4),
            'test_mae': round(float(mean_absolute_error(y_test, pred)), 2),
            'test_rmse': round(float(np.sqrt(mean_squared_error(y_test, pred))), 2),
            'training_time_sec': round(time.time() - started, 2),
        }
        record = {'model': name, **metrics}
        results.append(record)
        fitted[name] = pipe

        run, mlflow = _start_mlflow_run(name.replace(' ', '_').lower())
        _log_mlflow(run, mlflow, {
            'model_architecture': name,
            'cv_folds': 5,
        }, metrics)
        if run is not None and mlflow is not None:
            mlflow.end_run()

    results.sort(key=lambda row: row['cv_r2_mean'], reverse=True)
    candidate_names = {'Random Forest', 'Gradient Boosting'}
    top_two = [row['model'] for row in results if row['model'] in candidate_names][:2]
    if len(top_two) < 2:
        for row in results:
            if row['model'] not in top_two:
                top_two.append(row['model'])
            if len(top_two) == 2:
                break

    tuning_records = []
    tuned_pipelines = {}
    for name in top_two:
        if name not in {'Random Forest', 'Gradient Boosting', 'XGBoost'}:
            continue
        search = tune_model(name, fitted[name], X_train, y_train, cv)
        pred = search.best_estimator_.predict(X_test)
        base_cv = next(r['cv_r2_mean'] for r in results if r['model'] == name)
        post_cv = float(search.best_score_)
        tuning_records.append({
            'model': name,
            'best_parameters': {k.replace('regressor__', ''): v for k, v in search.best_params_.items()},
            'best_cv_r2': round(post_cv, 4),
            'pre_tuning_cv_r2': round(base_cv, 4),
            'improvement_pct': round(((post_cv - base_cv) / base_cv) * 100, 3),
            'test_r2': round(float(r2_score(y_test, pred)), 4),
            'test_mae': round(float(mean_absolute_error(y_test, pred)), 2),
            'test_rmse': round(float(np.sqrt(mean_squared_error(y_test, pred))), 2),
        })
        tuned_pipelines[name] = search.best_estimator_

    # Select the tuned candidate with the highest CV R²; if tuning fails, fall back to baseline leader.
    if tuned_pipelines:
        selected = max(
            tuning_records,
            key=lambda row: row['best_cv_r2']
        )
        final_name = selected['model']
        final_pipeline = tuned_pipelines[final_name]
        final_tuning = selected
    else:
        final_name = results[0]['model']
        final_pipeline = fitted[final_name]
        base = next(r for r in results if r['model'] == final_name)
        final_tuning = {
            'model': final_name,
            'best_parameters': {},
            'best_cv_r2': base['cv_r2_mean'],
            'pre_tuning_cv_r2': base['cv_r2_mean'],
            'improvement_pct': 0.0,
        }

    final_pred = final_pipeline.predict(X_test)
    final_r2 = float(r2_score(y_test, final_pred))
    final_mae = float(mean_absolute_error(y_test, final_pred))
    final_rmse = float(np.sqrt(mean_squared_error(y_test, final_pred)))

    normalized_mae = final_mae / float(np.mean(y_test))
    absolute_error = np.abs(y_test.to_numpy() - final_pred)
    denom = np.maximum(np.abs(y_test.to_numpy()), 1.0)
    tolerance_pct = absolute_error / denom * 100.0

    evaluation = {
        'final_model': final_name,
        'best_parameters': final_tuning['best_parameters'],
        'test_mae': round(final_mae, 2),
        'test_rmse': round(final_rmse, 2),
        'test_r2': round(final_r2, 4),
        'pre_tuning_cv_r2': round(float(final_tuning['pre_tuning_cv_r2']), 4),
        'post_tuning_cv_r2': round(float(final_tuning['best_cv_r2']), 4),
        'improvement_pct': round(float(final_tuning['improvement_pct']), 3),
        'dataset_test_samples': int(len(y_test)),
        'mean_actual_yield': round(float(np.mean(y_test)), 2),
        'normalized_mae': round(float(normalized_mae), 4),
        'within_10_percent_tolerance': round(float(np.mean(tolerance_pct <= 10) * 100), 2),
        'within_20_percent_tolerance': round(float(np.mean(tolerance_pct <= 20) * 100), 2),
    }

    residuals = (y_test.to_numpy() - final_pred)
    residual_rows = [
        {'actual': float(a), 'predicted': float(p), 'residual': float(r)}
        for a, p, r in zip(y_test.iloc[:1000], final_pred[:1000], residuals[:1000])
    ]

    importances = _feature_importance(final_pipeline)

    outlier_report = iqr_outlier_report(df, RAW_NUMERIC_FEATURES)

    joblib.dump(final_pipeline, MODEL_OUTPUT_PATH, compress=3)
    with open(MODEL_COMPARISON_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    with open(EVALUATION_METRICS_PATH, 'w') as f:
        json.dump(evaluation, f, indent=2)
    with open(FEATURE_IMPORTANCE_PATH, 'w') as f:
        json.dump(importances, f, indent=2)
    with open(RESIDUALS_PATH, 'w') as f:
        json.dump(residual_rows, f, indent=2)
    with open(TUNING_RESULTS_PATH, 'w') as f:
        json.dump(tuning_records, f, indent=2)
    with open(OUTLIER_REPORT_PATH, 'w') as f:
        json.dump(outlier_report, f, indent=2)

    metadata = {
        'project': 'CropYield Predictor',
        'model_file': MODEL_OUTPUT_PATH,
        'final_model': final_name,
        'target': TARGET_COL,
        'train_test_split': {'test_size': 0.20, 'random_state': 42},
        'cross_validation': {'method': 'KFold', 'folds': 5, 'shuffle': True, 'random_state': 42},
        'pipeline_steps': ['IQRCapper', 'AgriculturalFeatureEngineer', 'ColumnTransformer', 'Regressor'],
        'python_version': platform.python_version(),
        'sklearn_version': _module_version('sklearn'),
        'joblib_version': _module_version('joblib'),
        'feature_engineering': get_feature_engineering_rationale(),
        'fit_scope': 'All preprocessing, IQR capping, and feature engineering are fit within the training folds/pipeline.',
    }
    with open(MODEL_METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)

    return {
        'comparison': results,
        'evaluation': evaluation,
        'tuning': tuning_records,
        'feature_importance': importances,
        'metadata': metadata,
    }


def _module_version(name: str):
    try:
        mod = __import__(name)
        return getattr(mod, '__version__', 'unknown')
    except Exception:
        return 'not installed'


if __name__ == '__main__':
    output = train_and_compare_models()
    print(json.dumps(output['evaluation'], indent=2))
