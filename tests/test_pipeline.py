"""
Tests: test_pipeline.py
Purpose: Unit tests for the insurance risk analytics pipeline.
"""
import sys
import os
import pytest
import pandas as pd
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.hypothesis_tests import run_provinces, run_zipcodes_risk, run_zipcodes_margin, run_gender
from src.data_loader import add_derived_metrics


@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 500
    df = pd.DataFrame({
        'PolicyID': range(n),
        'TotalPremium': np.random.uniform(10, 500, n),
        'TotalClaims': np.random.uniform(0, 300, n),
        'Province': np.random.choice(['Gauteng', 'Western Cape', 'KwaZulu-Natal'], n),
        'PostalCode': np.random.choice(['2000', '8000', '4000', '0001'], n),
        'Gender': np.random.choice(['Male', 'Female'], n),
        'VehicleType': np.random.choice(['Passenger', 'Commercial'], n),
        'make': np.random.choice(['TOYOTA', 'BMW', 'FORD'], n),
        'RegistrationYear': np.random.randint(2000, 2015, n),
    })
    df = add_derived_metrics(df)
    return df


def test_add_derived_metrics_columns(sample_df):
    assert 'LossRatio' in sample_df.columns
    assert 'Margin' in sample_df.columns
    assert 'HasClaim' in sample_df.columns


def test_margin_calculation(sample_df):
    expected = sample_df['TotalPremium'] - sample_df['TotalClaims']
    pd.testing.assert_series_equal(
        sample_df['Margin'].round(4), expected.round(4), check_names=False)


def test_has_claim_binary(sample_df):
    assert sample_df['HasClaim'].isin([0, 1]).all()


def test_loss_ratio_positive(sample_df):
    valid = sample_df[sample_df['TotalPremium'] > 0]
    assert (valid['LossRatio'] >= 0).all()


def test_dataset_shape(sample_df):
    assert len(sample_df) == 500


def test_no_null_in_derived(sample_df):
    assert sample_df['HasClaim'].isnull().sum() == 0


def test_provinces_returns_dict(sample_df):
    result = run_provinces(sample_df)
    assert isinstance(result, dict)
    assert 'p_value' in result
    assert 'Decision' in result
    assert 'Hypothesis' in result


def test_gender_returns_dict(sample_df):
    result = run_gender(sample_df)
    assert isinstance(result, dict)
    assert 'p_value' in result
    assert 'Decision' in result


def test_zipcodes_risk_returns_dict(sample_df):
    result = run_zipcodes_risk(sample_df)
    assert isinstance(result, dict)
    assert 'Decision' in result


def test_zipcodes_margin_returns_dict(sample_df):
    result = run_zipcodes_margin(sample_df)
    assert isinstance(result, dict)
    assert 'Decision' in result


def test_decision_values(sample_df):
    result = run_provinces(sample_df)
    assert result['Decision'] in ['Reject H0', 'Fail to Reject H0']


def test_p_value_range(sample_df):
    result = run_provinces(sample_df)
    assert 0 <= result['p_value'] <= 1


def test_overall_loss_ratio(sample_df):
    lr = sample_df['TotalClaims'].sum() / sample_df['TotalPremium'].sum()
    assert isinstance(lr, float)
    assert lr >= 0


def test_province_loss_ratio_varies(sample_df):
    prov_lr = sample_df.groupby('Province').apply(
        lambda x: x['TotalClaims'].sum() / x['TotalPremium'].sum()
    )
    assert len(prov_lr) > 1
    assert prov_lr.std() > 0