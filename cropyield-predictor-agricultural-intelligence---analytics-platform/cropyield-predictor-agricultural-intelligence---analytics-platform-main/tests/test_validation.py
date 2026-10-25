"""
Unit tests for data validation module using pytest.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_validation import (
    validate_schema,
    validate_missing_values,
    validate_value_ranges,
    validate_categories,
    check_duplicates,
    detect_potential_leakage,
    validate_full_dataset,
    EXPECTED_COLUMNS
)


@pytest.fixture
def valid_sample_df():
    """Return a pristine synthetic sample matching the actual dataset schema."""
    return pd.DataFrame({
        'Area': ['India', 'United States', 'Brazil', 'France'],
        'Item': ['Wheat', 'Maize', 'Soybeans', 'Potatoes'],
        'Year': [2000, 2005, 2010, 2012],
        'hg/ha_yield': [27700.0, 93000.0, 29000.0, 420000.0],
        'average_rain_fall_mm_per_year': [1083.0, 715.0, 1782.0, 867.0],
        'pesticides_tonnes': [45000.0, 350000.0, 280000.0, 65000.0],
        'avg_temp': [24.5, 12.8, 25.1, 11.2]
    })


def test_schema_valid(valid_sample_df):
    is_valid, errors = validate_schema(valid_sample_df)
    assert is_valid is True
    assert len(errors) == 0


def test_schema_missing_column(valid_sample_df):
    df_missing = valid_sample_df.drop(columns=['avg_temp'])
    is_valid, errors = validate_schema(df_missing)
    assert is_valid is False
    assert any("avg_temp" in err for err in errors)


def test_missing_values_clean(valid_sample_df):
    is_valid, missing_dict = validate_missing_values(valid_sample_df)
    assert is_valid is True
    assert all(count == 0 for count in missing_dict.values())


def test_missing_values_detected(valid_sample_df):
    df_with_nan = valid_sample_df.copy()
    df_with_nan.loc[0, 'hg/ha_yield'] = np.nan
    is_valid, missing_dict = validate_missing_values(df_with_nan)
    assert is_valid is False
    assert missing_dict['hg/ha_yield'] == 1


def test_range_validation_pass(valid_sample_df):
    is_valid, errors = validate_value_ranges(valid_sample_df)
    assert is_valid is True
    assert len(errors) == 0


def test_range_validation_fail(valid_sample_df):
    df_bad_range = valid_sample_df.copy()
    df_bad_range.loc[0, 'Year'] = 1850  # Below 1900
    df_bad_range.loc[1, 'avg_temp'] = 75.0  # Above 60 C
    is_valid, errors = validate_value_ranges(df_bad_range)
    assert is_valid is False
    assert len(errors) == 2


def test_categories_validation_pass(valid_sample_df):
    is_valid, errors = validate_categories(valid_sample_df)
    assert is_valid is True
    assert len(errors) == 0


def test_categories_validation_invalid_crop(valid_sample_df):
    df_bad_crop = valid_sample_df.copy()
    df_bad_crop.loc[0, 'Item'] = 'Golden Apples'
    is_valid, errors = validate_categories(df_bad_crop)
    assert is_valid is False
    assert any("Golden Apples" in err for err in errors)


def test_duplicate_detection(valid_sample_df):
    assert check_duplicates(valid_sample_df) == 0
    df_dup = pd.concat([valid_sample_df, valid_sample_df.iloc[[0]]], ignore_index=True)
    assert check_duplicates(df_dup) == 1


def test_leakage_detection():
    clean_df = pd.DataFrame({'Area': ['A'], 'Item': ['Wheat']})
    is_clean, _ = detect_potential_leakage(clean_df)
    assert is_clean is True

    leaky_df = pd.DataFrame({'Area': ['A'], 'production_tonnes': [100.0]})
    is_leaky_free, errors = detect_potential_leakage(leaky_df)
    assert is_leaky_free is False
    assert len(errors) > 0


def test_full_dataset_validation(valid_sample_df):
    result = validate_full_dataset(valid_sample_df)
    assert result['is_valid'] is True
    assert result['row_count'] == 4
    assert result['column_count'] == 7
    assert result['duplicate_rows_count'] == 0
