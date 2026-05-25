"""Script: run_modeling.py - ML Modeling Pipeline"""
import sys
sys.path.insert(0, '.')
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import shap
from src.data_loader import load_data, add_derived_metrics

print("="*50)
print("ALPHCARE INSURANCE - MODELING PIPELINE")
print("="*50)

df = load_data()
df = add_derived_metrics(df)
print(f"Dataset loaded: {df.shape}")

# ── Feature Engineering ──────────────────────────────
print("\nPreparing features...")
df_model = df[df['TotalClaims'] > 0].copy()
print(f"  → Policies with claims: {len(df_model):,}")

df_model['VehicleAge'] = 2015 - pd.to_numeric(
    df_model['RegistrationYear'], errors='coerce')
df_model['VehicleAge'] = df_model['VehicleAge'].clip(0, 50)
df_model['PremiumPerCover'] = df_model['TotalPremium'] / (
    df_model['SumInsured'].replace(0, np.nan))

features = [
    'TotalPremium', 'SumInsured', 'CalculatedPremiumPerTerm',
    'CustomValueEstimate', 'VehicleAge', 'kilowatts', 'Cylinders',
    'Province', 'VehicleType', 'make', 'Gender', 'MaritalStatus',
    'CoverType', 'CoverGroup', 'Product'
]

# Only keep features that exist
features = [f for f in features if f in df_model.columns]
print(f"  → Features used: {features}")

df_feat = df_model[features + ['TotalClaims']].copy()

# Handle missing values
for col in df_feat.select_dtypes(include='number').columns:
    df_feat[col] = df_feat[col].fillna(df_feat[col].median())

# Encode categoricals
le = LabelEncoder()
for col in df_feat.select_dtypes(include='object').columns:
    df_feat[col] = df_feat[col].fillna('Unknown')
    df_feat[col] = le.fit_transform(df_feat[col].astype(str))

X = df_feat.drop(columns=['TotalClaims'])
y = df_feat['TotalClaims']

print(f"  → Feature matrix shape: {X.shape}")
print(f"  → Target mean: {y.mean():.2f}, std: {y.std():.2f}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print(f"  → Train: {X_train.shape}, Test: {X_test.shape}")

# ── Train Models ─────────────────────────────────────
results = []

print("\nTraining Linear Regression...")
lr = LinearRegression()
lr.fit(X_train, y_train)
preds_lr = np.maximum(lr.predict(X_test), 0)
rmse_lr = np.sqrt(mean_squared_error(y_test, preds_lr))
r2_lr = r2_score(y_test, preds_lr)
results.append({'Model': 'Linear Regression', 'RMSE': round(rmse_lr,2), 'R2': round(r2_lr,4)})
print(f"  → RMSE: {rmse_lr:.2f}, R2: {r2_lr:.4f}")

print("Training Random Forest (100 trees)...")
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)
rf.fit(X_train, y_train)
preds_rf = np.maximum(rf.predict(X_test), 0)
rmse_rf = np.sqrt(mean_squared_error(y_test, preds_rf))
r2_rf = r2_score(y_test, preds_rf)
results.append({'Model': 'Random Forest', 'RMSE': round(rmse_rf,2), 'R2': round(r2_rf,4)})
print(f"  → RMSE: {rmse_rf:.2f}, R2: {r2_rf:.4f}")

print("Training XGBoost...")
xgb_model = xgb.XGBRegressor(
    n_estimators=200, learning_rate=0.05, max_depth=6,
    random_state=42, n_jobs=-1, verbosity=0)
xgb_model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)
preds_xgb = np.maximum(xgb_model.predict(X_test), 0)
rmse_xgb = np.sqrt(mean_squared_error(y_test, preds_xgb))
r2_xgb = r2_score(y_test, preds_xgb)
results.append({'Model': 'XGBoost', 'RMSE': round(rmse_xgb,2), 'R2': round(r2_xgb,4)})
print(f"  → RMSE: {rmse_xgb:.2f}, R2: {r2_xgb:.4f}")

# ── Model Comparison ─────────────────────────────────
print("\n" + "="*50)
print("MODEL COMPARISON TABLE")
print("="*50)
results_df = pd.DataFrame(results).sort_values('R2', ascending=False)
results_df['Best'] = results_df['R2'] == results_df['R2'].max()
print(results_df.to_string(index=False))

best_name = results_df.iloc[0]['Model']
print(f"\nBest model: {best_name}")

# ── SHAP Analysis ────────────────────────────────────
print("\nComputing SHAP values for XGBoost (this takes 1-2 min)...")
sample = X_test.iloc[:300]
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(sample)

# SHAP summary plot
fig, ax = plt.subplots(figsize=(10, 7))
shap.summary_plot(shap_values, sample,
                  feature_names=list(X.columns),
                  show=False, plot_size=None)
plt.title('SHAP Feature Importance — XGBoost Model\n(AlphaCare Insurance Risk Analytics)',
          fontweight='bold')
plt.tight_layout()
plt.savefig('data/plot_shap_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → Saved plot_shap_summary.png")

# SHAP bar plot
shap_mean = np.abs(shap_values).mean(axis=0)
shap_df = pd.DataFrame({
    'Feature': list(X.columns),
    'MeanSHAP': shap_mean
}).sort_values('MeanSHAP', ascending=True).tail(10)

fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(shap_df['Feature'], shap_df['MeanSHAP'], color='#2980b9', alpha=0.8)
ax.set_title('Top 10 Features by Mean |SHAP| Value\nXGBoost — Claim Severity Model',
             fontweight='bold')
ax.set_xlabel('Mean |SHAP Value|')
plt.tight_layout()
plt.savefig('data/plot_shap_bar.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → Saved plot_shap_bar.png")

print("\nTop 5 most influential features:")
top5 = pd.DataFrame({'Feature': list(X.columns), 'MeanSHAP': shap_mean})
top5 = top5.sort_values('MeanSHAP', ascending=False).head(5)
for _, row in top5.iterrows():
    print(f"  {row['Feature']}: {row['MeanSHAP']:.4f}")

print("\n" + "="*50)
print("MODELING COMPLETE")
print("="*50)