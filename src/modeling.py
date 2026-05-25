"""
Module: modeling.py
Purpose: Feature engineering, model training, evaluation, and SHAP analysis.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def prepare_features(df, target='TotalClaims', test_size=0.2):
    """Prepare features for modeling claim severity."""
    df_model = df[df['TotalClaims'] > 0].copy()

    # Feature engineering
    df_model['VehicleAge'] = 2015 - pd.to_numeric(df_model['RegistrationYear'], errors='coerce')
    df_model['VehicleAge'] = df_model['VehicleAge'].clip(0, 50)

    features = [
        'TotalPremium', 'SumInsured', 'CalculatedPremiumPerTerm',
        'CustomValueEstimate', 'VehicleAge', 'Kilowatts', 'Cylinders',
        'Province', 'VehicleType', 'Make', 'Gender', 'MaritalStatus'
    ]

    df_feat = df_model[features + [target]].copy()

    # Handle missing values
    for col in df_feat.select_dtypes(include='number').columns:
        df_feat[col] = df_feat[col].fillna(df_feat[col].median())

    # Encode categoricals
    le = LabelEncoder()
    for col in df_feat.select_dtypes(include='object').columns:
        df_feat[col] = df_feat[col].fillna('Unknown')
        df_feat[col] = le.fit_transform(df_feat[col].astype(str))

    X = df_feat.drop(columns=[target])
    y = df_feat[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    return X_train, X_test, y_train, y_test, list(X.columns)


def train_models(X_train, y_train):
    """Train Linear Regression, Random Forest, and XGBoost."""
    models = {}

    print("Training Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    models['Linear Regression'] = lr

    print("Training Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf

    print("Training XGBoost...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=6,
        random_state=42, n_jobs=-1, verbosity=0
    )
    xgb_model.fit(X_train, y_train)
    models['XGBoost'] = xgb_model

    print("All models trained.")
    return models


def evaluate_models(models, X_test, y_test):
    """Evaluate all models and return comparison DataFrame."""
    results = []
    for name, model in models.items():
        preds = model.predict(X_test)
        preds = np.maximum(preds, 0)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        results.append({
            'Model': name,
            'RMSE': round(rmse, 2),
            'R2': round(r2, 4),
            'Recommendation': 'BEST' if r2 == max(r2_score(m.predict(X_test), y_test)
                                                   for m in models.values()) else ''
        })
    return pd.DataFrame(results).sort_values('R2', ascending=False)


def plot_shap(model, X_test, feature_names):
    """Generate SHAP summary plot for the best model."""
    print("Computing SHAP values (this may take 1-2 minutes)...")
    explainer = shap.TreeExplainer(model)
    sample = X_test.iloc[:500]
    shap_values = explainer.shap_values(sample)

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(shap_values, sample, feature_names=feature_names,
                      show=False, plot_size=(10, 7))
    plt.title('SHAP Feature Importance — XGBoost Model', fontweight='bold')
    plt.tight_layout()
    plt.savefig('data/plot_shap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("SHAP plot saved to data/plot_shap.png")