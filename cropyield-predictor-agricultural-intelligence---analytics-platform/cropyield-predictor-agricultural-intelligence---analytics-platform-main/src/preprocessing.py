"""Leak-free preprocessing for the CropYield Predictor."""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


def get_feature_column_names():
    numeric_features = [
        'Year',
        'average_rain_fall_mm_per_year',
        'pesticides_tonnes',
        'avg_temp',
        'hydrothermal_index',
        'pesticide_log',
        'temp_rainfall_interaction',
    ]
    categorical_features = ['Area', 'Item', 'rainfall_category']
    return numeric_features, categorical_features


def build_preprocessor() -> ColumnTransformer:
    numeric_features, categorical_features = get_feature_column_names()
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])
    return ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
    ], remainder='drop')
