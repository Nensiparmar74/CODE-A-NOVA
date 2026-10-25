"""CropYield Predictor - Domain feature engineering transformer."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class AgriculturalFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create domain-relevant agricultural features inside sklearn pipelines."""

    def __init__(self):
        self.engineered_feature_names = [
            'hydrothermal_index',
            'rainfall_category',
            'pesticide_log',
            'temp_rainfall_interaction',
        ]

    def fit(self, X, y=None):
        self.feature_names_in_ = list(X.columns) if isinstance(X, pd.DataFrame) else None
        return self

    def transform(self, X):
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        if {'average_rain_fall_mm_per_year', 'avg_temp'} <= set(df.columns):
            temp_adjusted = np.maximum(pd.to_numeric(df['avg_temp'], errors='coerce') + 10.0, 1.0)
            df['hydrothermal_index'] = pd.to_numeric(
                df['average_rain_fall_mm_per_year'], errors='coerce'
            ) / temp_adjusted

        if 'average_rain_fall_mm_per_year' in df.columns:
            bins = [-np.inf, 600.0, 1200.0, 2000.0, np.inf]
            labels = [
                'Low / Arid (<600mm)',
                'Moderate (600-1200mm)',
                'High (1200-2000mm)',
                'Tropical (>2000mm)',
            ]
            df['rainfall_category'] = pd.cut(
                pd.to_numeric(df['average_rain_fall_mm_per_year'], errors='coerce'),
                bins=bins, labels=labels
            ).astype(str)

        if 'pesticides_tonnes' in df.columns:
            pesticides = np.maximum(pd.to_numeric(df['pesticides_tonnes'], errors='coerce'), 0.0)
            df['pesticide_log'] = np.log1p(pesticides)

        if {'average_rain_fall_mm_per_year', 'avg_temp'} <= set(df.columns):
            df['temp_rainfall_interaction'] = (
                pd.to_numeric(df['average_rain_fall_mm_per_year'], errors='coerce')
                * pd.to_numeric(df['avg_temp'], errors='coerce')
            ) / 1000.0

        return df

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        if input_features is None:
            input_features = getattr(self, 'feature_names_in_', [])
        return np.asarray(
            list(input_features) + self.engineered_feature_names,
            dtype=object,
        )


def get_feature_engineering_rationale() -> List[Dict[str, str]]:
    return [
        {
            'name': 'hydrothermal_index',
            'formula': 'average_rain_fall_mm_per_year / (avg_temp + 10)',
            'source': 'average_rain_fall_mm_per_year, avg_temp',
            'domain_meaning': 'Water-availability proxy combining precipitation and temperature.',
            'impact': 'Provides a compact interaction between rainfall and thermal conditions.',
        },
        {
            'name': 'rainfall_category',
            'formula': 'Binned into Low, Moderate, High and Tropical rainfall regimes.',
            'source': 'average_rain_fall_mm_per_year',
            'domain_meaning': 'Non-linear rainfall regime indicator.',
            'impact': 'Lets models capture step-like differences between rainfall regimes.',
        },
        {
            'name': 'pesticide_log',
            'formula': 'log1p(pesticides_tonnes)',
            'source': 'pesticides_tonnes',
            'domain_meaning': 'Log transform for a strongly right-skewed input.',
            'impact': 'Reduces the influence of extreme input magnitudes.',
        },
        {
            'name': 'temp_rainfall_interaction',
            'formula': '(average_rain_fall_mm_per_year * avg_temp) / 1000',
            'source': 'average_rain_fall_mm_per_year, avg_temp',
            'domain_meaning': 'Joint climate interaction term.',
            'impact': 'Allows the model to represent combined thermal and rainfall conditions.',
        },
    ]
