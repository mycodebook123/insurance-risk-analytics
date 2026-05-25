"""
Module: hypothesis_tests.py
Purpose: Statistical hypothesis testing functions for insurance risk analysis.
"""

import pandas as pd
import numpy as np
from scipy import stats


def test_provinces(df):
    """H1: No risk differences across provinces. Uses chi-squared on claim frequency."""
    contingency = pd.crosstab(df['Province'], df['HasClaim'])
    chi2, p, dof, _ = stats.chi2_contingency(contingency)
    decision = 'Reject H0' if p < 0.05 else 'Fail to Reject H0'
    interpretation = (
        "Significant risk differences exist across provinces. "
        "Provinces with higher claim frequency should have adjusted premiums."
        if p < 0.05 else
        "No significant risk differences across provinces at 5% significance level."
    )
    return {
        'Hypothesis': 'H1: No risk diff across provinces',
        'Test': 'Chi-Squared',
        'KPI': 'Claim Frequency',
        'p_value': round(p, 6),
        'Decision': decision,
        'Interpretation': interpretation
    }


def test_zipcodes_risk(df):
    """H2: No risk differences between zip codes. Uses t-test on claim severity."""
    df_claims = df[df['HasClaim'] == 1].copy()
    top_zips = df_claims['PostalCode'].value_counts().head(2).index.tolist()
    if len(top_zips) < 2:
        return {'Hypothesis': 'H2', 'Test': 'T-Test', 'p_value': None,
                'Decision': 'Insufficient data', 'Interpretation': 'Not enough zip codes'}
    g1 = df_claims[df_claims['PostalCode'] == top_zips[0]]['TotalClaims'].dropna()
    g2 = df_claims[df_claims['PostalCode'] == top_zips[1]]['TotalClaims'].dropna()
    t_stat, p = stats.ttest_ind(g1, g2, equal_var=False)
    decision = 'Reject H0' if p < 0.05 else 'Fail to Reject H0'
    interpretation = (
        f"Significant claim severity difference between zip codes {top_zips[0]} and {top_zips[1]}. "
        f"Zip-code level premium differentiation is statistically justified."
        if p < 0.05 else
        f"No significant claim severity difference between the two largest zip codes."
    )
    return {
        'Hypothesis': 'H2: No risk diff between zip codes',
        'Test': 'Welch T-Test',
        'KPI': 'Claim Severity',
        'p_value': round(p, 6),
        'Decision': decision,
        'Interpretation': interpretation
    }


def test_zipcodes_margin(df):
    """H3: No margin difference between zip codes. Uses t-test on Margin."""
    top_zips = df['PostalCode'].value_counts().head(2).index.tolist()
    if len(top_zips) < 2:
        return {'Hypothesis': 'H3', 'Test': 'T-Test', 'p_value': None,
                'Decision': 'Insufficient data', 'Interpretation': 'Not enough zip codes'}
    g1 = df[df['PostalCode'] == top_zips[0]]['Margin'].dropna()
    g2 = df[df['PostalCode'] == top_zips[1]]['Margin'].dropna()
    t_stat, p = stats.ttest_ind(g1, g2, equal_var=False)
    decision = 'Reject H0' if p < 0.05 else 'Fail to Reject H0'
    interpretation = (
        f"Significant margin difference between zip codes {top_zips[0]} and {top_zips[1]}. "
        "Premium structures should be reviewed at zip code level."
        if p < 0.05 else
        "No significant margin difference between the two largest zip codes."
    )
    return {
        'Hypothesis': 'H3: No margin diff between zip codes',
        'Test': 'Welch T-Test',
        'KPI': 'Margin',
        'p_value': round(p, 6),
        'Decision': decision,
        'Interpretation': interpretation
    }


def test_gender(df):
    """H4: No risk difference between Women and Men. Uses chi-squared on claim frequency."""
    df_gender = df[df['Gender'].isin(['Female', 'Male'])].copy()
    contingency = pd.crosstab(df_gender['Gender'], df_gender['HasClaim'])
    if contingency.shape != (2, 2):
        return {'Hypothesis': 'H4', 'Test': 'Chi-Squared', 'p_value': None,
                'Decision': 'Insufficient data', 'Interpretation': 'Gender data insufficient'}
    chi2, p, dof, _ = stats.chi2_contingency(contingency)
    decision = 'Reject H0' if p < 0.05 else 'Fail to Reject H0'
    interpretation = (
        "Significant risk difference between male and female policyholders. "
        "Gender may be a valid risk segmentation variable."
        if p < 0.05 else
        "No significant risk difference between male and female policyholders at 5% level."
    )
    return {
        'Hypothesis': 'H4: No risk diff between genders',
        'Test': 'Chi-Squared',
        'KPI': 'Claim Frequency',
        'p_value': round(p, 6),
        'Decision': decision,
        'Interpretation': interpretation
    }