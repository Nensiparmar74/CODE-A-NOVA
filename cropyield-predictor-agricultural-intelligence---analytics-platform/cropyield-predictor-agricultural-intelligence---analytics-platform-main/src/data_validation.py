"""
CropYield Predictor - Data Validation Module
Phase 3: Robust validation functions for agricultural crop yield data.
"""

from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np

EXPECTED_COLUMNS = [
    'Area',
    'Item',
    'Year',
    'hg/ha_yield',
    'average_rain_fall_mm_per_year',
    'pesticides_tonnes',
    'avg_temp'
]

VALID_CROPS = {
    'Cassava',
    'Maize',
    'Plantains and others',
    'Potatoes',
    'Rice, paddy',
    'Sorghum',
    'Soybeans',
    'Sweet potatoes',
    'Wheat',
    'Yams'
}

VALUE_BOUNDS = {
    'Year': (1900, 2030),
    'hg/ha_yield': (1.0, 1_000_000.0),
    'average_rain_fall_mm_per_year': (0.0, 10_000.0),
    'pesticides_tonnes': (0.0, 1_000_000.0),
    'avg_temp': (-40.0, 60.0)
}

FORBIDDEN_LEAKAGE_COLUMNS = {
    'production', 'production_tonnes', 'harvested_area', 'harvest_area_ha',
    'yield_kg_per_ha', 'yield_tonnes_per_ha'
}


def validate_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate that required domain columns exist in the DataFrame."""
    errors = []
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
    return len(errors) == 0, errors


def validate_missing_values(df: pd.DataFrame) -> Tuple[bool, Dict[str, int]]:
    """Check for null or empty values across all columns."""
    missing = df.isnull().sum().to_dict()
    has_missing = any(v > 0 for v in missing.values())
    return not has_missing, missing


def validate_value_ranges(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Check that continuous numeric features and target fall within valid physical bounds."""
    errors = []
    for col, (min_val, max_val) in VALUE_BOUNDS.items():
        if col in df.columns:
            invalid_low = (df[col] < min_val).sum()
            invalid_high = (df[col] > max_val).sum()
            if invalid_low > 0:
                errors.append(f"Column '{col}' has {invalid_low} values below minimum ({min_val})")
            if invalid_high > 0:
                errors.append(f"Column '{col}' has {invalid_high} values above maximum ({max_val})")
    return len(errors) == 0, errors


def validate_categories(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Verify that crop varieties in 'Item' belong to valid agricultural types."""
    errors = []
    if 'Item' in df.columns:
        items = set(df['Item'].dropna().unique())
        invalid_items = items - VALID_CROPS
        if invalid_items:
            errors.append(f"Unrecognized crop categories found: {invalid_items}")
    if 'Area' in df.columns:
        empty_areas = (df['Area'].astype(str).str.strip() == '').sum()
        if empty_areas > 0:
            errors.append(f"Found {empty_areas} blank or whitespace Area values.")
    return len(errors) == 0, errors


def check_duplicates(df: pd.DataFrame, subset: List[str] = None) -> int:
    """Identify duplicate observation count across domain features."""
    cols = subset if subset is not None else [c for c in EXPECTED_COLUMNS if c in df.columns]
    return int(df.duplicated(subset=cols).sum())


def detect_potential_leakage(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Detect presence of post-harvest or deterministic leakage features."""
    errors = []
    lower_cols = [c.lower() for c in df.columns]
    leakage = [c for c in lower_cols if c in FORBIDDEN_LEAKAGE_COLUMNS]
    if leakage:
        errors.append(f"Potential leakage columns detected: {leakage}")
    return len(errors) == 0, errors


def validate_full_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """Execute complete validation suite and return summary dictionary."""
    schema_ok, schema_err = validate_schema(df)
    missing_ok, missing_dict = validate_missing_values(df)
    ranges_ok, range_err = validate_value_ranges(df)
    cats_ok, cat_err = validate_categories(df)
    leakage_ok, leak_err = detect_potential_leakage(df)
    dup_count = check_duplicates(df)

    all_passed = schema_ok and missing_ok and ranges_ok and cats_ok and leakage_ok

    return {
        'is_valid': all_passed,
        'schema_valid': schema_ok,
        'schema_errors': schema_err,
        'missing_values': missing_dict,
        'ranges_valid': ranges_ok,
        'range_errors': range_err,
        'categories_valid': cats_ok,
        'category_errors': cat_err,
        'leakage_free': leakage_ok,
        'leakage_errors': leak_err,
        'duplicate_rows_count': dup_count,
        'row_count': len(df),
        'column_count': len(df.columns)
    }
