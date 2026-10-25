"""Reusable scikit-learn transformers for the CropYield Predictor pipeline."""

from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class IQRCapper(BaseEstimator, TransformerMixin):
    """Cap numeric feature outliers using training-set IQR bounds.

    The transformer learns bounds only from the training data, preventing data leakage.
    The target is never modified. This is intentionally applied to raw numeric inputs
    before agricultural feature engineering.
    """

    def __init__(self, columns: Iterable[str] | None = None, multiplier: float = 1.5):
        self.columns = columns
        self.multiplier = multiplier

    def fit(self, X: pd.DataFrame, y=None):
        X_df = self._to_frame(X)
        self.feature_names_in_ = list(X_df.columns)
        cols = list(self.columns) if self.columns is not None else [
            c for c in X_df.columns
            if pd.api.types.is_numeric_dtype(X_df[c]) and c != 'hg/ha_yield'
        ]
        self.columns_ = [c for c in cols if c in X_df.columns]
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}
        for col in self.columns_:
            series = pd.to_numeric(X_df[col], errors='coerce')
            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1
            self.lower_bounds_[col] = q1 - self.multiplier * iqr
            self.upper_bounds_[col] = q3 + self.multiplier * iqr
        return self

    def transform(self, X: pd.DataFrame):
        self._check_is_fitted()
        X_df = self._to_frame(X).copy()
        for col in self.columns_:
            if col in X_df.columns:
                X_df[col] = pd.to_numeric(X_df[col], errors='coerce').clip(
                    lower=self.lower_bounds_[col], upper=self.upper_bounds_[col]
                )
        return X_df

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        names = input_features if input_features is not None else self.feature_names_in_
        return np.asarray(names, dtype=object)

    @staticmethod
    def _to_frame(X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        if isinstance(X, dict):
            return pd.DataFrame([X])
        if isinstance(X, list) and (not X or isinstance(X[0], dict)):
            return pd.DataFrame(X)
        return pd.DataFrame(X)

    def _check_is_fitted(self):
        required = ['columns_', 'lower_bounds_', 'upper_bounds_', 'feature_names_in_']
        missing = [name for name in required if not hasattr(self, name)]
        if missing:
            raise RuntimeError(f'IQRCapper is not fitted. Missing: {missing}')


def iqr_outlier_report(df: pd.DataFrame, columns: Iterable[str]) -> List[dict]:
    """Return a descriptive IQR outlier audit for the input dataframe."""
    rows = []
    for col in columns:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors='coerce')
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = int(((series < lower) | (series > upper)).sum())
        rows.append({
            'column': col,
            'q1': round(q1, 4),
            'q3': round(q3, 4),
            'iqr': round(iqr, 4),
            'lower_bound': round(lower, 4),
            'upper_bound': round(upper, 4),
            'outlier_count': count,
            'outlier_pct': round(100 * count / len(series), 3),
            'treatment': 'Training-set IQR capping in sklearn pipeline' if count else 'No IQR outliers detected',
        })
    return rows
