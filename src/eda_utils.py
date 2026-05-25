"""
Module: eda_utils.py
Purpose: Reusable EDA utility functions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_distributions(df, cols, title="Distributions"):
    """Plot histograms for a list of columns."""
    fig, axes = plt.subplots(1, len(cols), figsize=(6*len(cols), 4))
    if len(cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, cols):
        df[col].dropna().hist(ax=ax, bins=50, color='#3498db', alpha=0.7)
        ax.set_title(col)
        ax.set_xlabel('Value')
    fig.suptitle(title, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_correlation(df, cols):
    """Plot correlation heatmap."""
    corr = df[cols].dropna().corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax)
    ax.set_title('Correlation Matrix', fontweight='bold')
    plt.tight_layout()
    return fig


def plot_geographic(df, group_col, value_col):
    """Plot bar chart of a value grouped by a geographic column."""
    grouped = df.groupby(group_col)[value_col].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    grouped.plot(kind='bar', ax=ax, color='#2980b9', alpha=0.8)
    ax.set_title(f'{value_col} by {group_col}', fontweight='bold')
    ax.set_ylabel(value_col)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig


def plot_outliers(df, cols):
    """Plot boxplots for outlier detection."""
    fig, axes = plt.subplots(1, len(cols), figsize=(6*len(cols), 5))
    if len(cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, cols):
        data = df[col].dropna()
        ax.boxplot(data[data.between(data.quantile(0.01), data.quantile(0.99))],
                   patch_artist=True, boxprops=dict(facecolor='#3498db', alpha=0.6))
        ax.set_title(col)
    fig.suptitle('Outlier Detection', fontweight='bold')
    plt.tight_layout()
    return fig