"""
Module: data_loader.py
Purpose: Load and perform initial validation of the insurance dataset.
"""

import pandas as pd
import os


DATA_PATH = os.path.join("data", "MachineLearningRating_v3.txt")


def load_data(path=DATA_PATH):
    """Load the insurance dataset from a pipe-separated text file."""
    print(f"Loading data from {path} ...")
    df = pd.read_csv(path, sep="|", low_memory=False)
    print(f"  → Shape: {df.shape}")
    print(f"  → Columns: {len(df.columns)}")
    return df


def get_basic_info(df):
    """Print basic dataset information."""
    print("\n── Dataset Info ─────────────────────────")
    print(f"Rows:    {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")
    print(f"\nColumn dtypes:")
    print(df.dtypes.value_counts())
    print(f"\nMissing values (top 10):")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    print(missing.head(10))
    return missing


def add_derived_metrics(df):
    """Add Loss Ratio and Margin columns."""
    df = df.copy()
    df["LossRatio"] = df["TotalClaims"] / df["TotalPremium"].replace(0, float("nan"))
    df["Margin"] = df["TotalPremium"] - df["TotalClaims"]
    df["HasClaim"] = (df["TotalClaims"] > 0).astype(int)
    print("  → Added: LossRatio, Margin, HasClaim")
    return df