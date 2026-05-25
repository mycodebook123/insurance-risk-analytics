"""Script: run_eda.py - Full EDA pipeline"""
import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from src.data_loader import load_data, get_basic_info, add_derived_metrics

sns.set_theme(style='whitegrid', palette='muted')

print("="*50)
print("ALPHCARE INSURANCE - EDA PIPELINE")
print("="*50)

df = load_data()
df = add_derived_metrics(df)

# ── 1. Basic Info ────────────────────────────────────
missing = get_basic_info(df)

print("\nNumerical Summary:")
print(df[['TotalPremium','TotalClaims','LossRatio',
          'Margin','CustomValueEstimate']].describe().round(2))

# ── 2. Missing Values ────────────────────────────────
print("\nMissing values (top 15 columns):")
missing_pct = (df.isnull().sum()/len(df)*100).sort_values(ascending=False)
print(missing_pct[missing_pct > 0].head(15).round(2))

# ── 3. Guiding Q1: Loss Ratio ────────────────────────
overall_lr = df['TotalClaims'].sum() / df['TotalPremium'].sum()
print(f"\nOverall Loss Ratio: {overall_lr:.4f} ({overall_lr*100:.2f}%)")
print("Portfolio is", "UNPROFITABLE" if overall_lr > 1 else "PROFITABLE")

print("\nLoss Ratio by Province:")
prov_lr = df.groupby('Province').apply(
    lambda x: x['TotalClaims'].sum()/x['TotalPremium'].sum()
).sort_values(ascending=False)
print(prov_lr.round(4))

print("\nLoss Ratio by Gender:")
gen_lr = df.groupby('Gender').apply(
    lambda x: x['TotalClaims'].sum()/x['TotalPremium'].sum()
)
print(gen_lr.round(4))

# ── 4. Plot 1: Loss Ratio by Province ───────────────
print("\nGenerating Plot 1: Loss Ratio by Province...")
prov_data = df.groupby('Province').agg(
    TotalPremium=('TotalPremium','sum'),
    TotalClaims=('TotalClaims','sum'),
    LossRatio=('LossRatio','mean'),
    PolicyCount=('PolicyID','count')
).reset_index().sort_values('LossRatio', ascending=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Insurance Portfolio Risk Overview by Province',
             fontsize=14, fontweight='bold')

colors = ['#2ecc71' if lr < 0.5 else '#e74c3c' if lr > 1
          else '#f39c12' for lr in prov_data['LossRatio']]
bars = axes[0].barh(prov_data['Province'], prov_data['LossRatio'], color=colors)
axes[0].axvline(x=1.0, color='red', linestyle='--', alpha=0.7, label='Break-even')
axes[0].axvline(x=overall_lr, color='blue', linestyle='--', alpha=0.7,
                label=f'Portfolio avg ({overall_lr:.2f})')
axes[0].set_xlabel('Loss Ratio')
axes[0].set_title('Loss Ratio by Province')
axes[0].legend(fontsize=8)
for bar, val in zip(bars, prov_data['LossRatio']):
    axes[0].text(val+0.01, bar.get_y()+bar.get_height()/2,
                 f'{val:.2f}', va='center', fontsize=8)

axes[1].scatter(prov_data['TotalPremium']/1e6, prov_data['TotalClaims']/1e6,
                s=prov_data['PolicyCount']/100,
                c=prov_data['LossRatio'], cmap='RdYlGn_r', alpha=0.8)
for _, row in prov_data.iterrows():
    axes[1].annotate(row['Province'][:8],
                     (row['TotalPremium']/1e6, row['TotalClaims']/1e6), fontsize=7)
axes[1].set_xlabel('Total Premium (ZAR millions)')
axes[1].set_ylabel('Total Claims (ZAR millions)')
axes[1].set_title('Premium vs Claims by Province\n(bubble size = policy count)')
plt.tight_layout()
plt.savefig('data/plot1_loss_ratio_province.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → Saved plot1_loss_ratio_province.png")

# ── 5. Plot 2: Financial Distributions ──────────────
print("Generating Plot 2: Financial Distributions...")
cols = ['TotalPremium','TotalClaims','LossRatio',
        'Margin','CustomValueEstimate','SumInsured']
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Distribution of Key Financial Variables',
             fontsize=14, fontweight='bold')
for ax, col in zip(axes.flatten(), cols):
    data = df[col].dropna()
    clipped = data[data.between(data.quantile(0.01), data.quantile(0.99))]
    ax.hist(clipped, bins=50, color='#3498db', alpha=0.7, edgecolor='white')
    ax.set_title(f'{col}\n(mean={data.mean():.0f}, std={data.std():.0f})')
    ax.set_xlabel('Value (ZAR)')
    ax.set_ylabel('Count')
plt.tight_layout()
plt.savefig('data/plot2_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → Saved plot2_distributions.png")

print("\nOutlier check:")
for col in ['TotalPremium','TotalClaims','CustomValueEstimate']:
    p99 = df[col].quantile(0.99)
    outliers = (df[col] > p99).sum()
    print(f"  {col}: {outliers:,} outliers above {p99:,.0f}")

# ── 6. Plot 3: Temporal Trends ───────────────────────
print("Generating Plot 3: Temporal Trends...")
df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], errors='coerce')
monthly = df.groupby(df['TransactionMonth'].dt.to_period('M')).agg(
    AvgPremium=('TotalPremium','mean'),
    AvgClaims=('TotalClaims','mean'),
    ClaimFreq=('HasClaim','mean'),
).reset_index()
monthly['TransactionMonth'] = monthly['TransactionMonth'].astype(str)

fig, axes = plt.subplots(2, 1, figsize=(14, 8))
fig.suptitle('Temporal Trends: Feb 2014 – Aug 2015',
             fontsize=14, fontweight='bold')
axes[0].plot(monthly['TransactionMonth'], monthly['AvgPremium'],
             marker='o', label='Avg Premium', color='#2980b9')
axes[0].plot(monthly['TransactionMonth'], monthly['AvgClaims'],
             marker='s', label='Avg Claims', color='#e74c3c')
axes[0].set_ylabel('ZAR')
axes[0].legend()
axes[0].set_title('Average Premium and Claims Over Time')
axes[0].tick_params(axis='x', rotation=45)
axes[1].bar(monthly['TransactionMonth'], monthly['ClaimFreq']*100,
            color='#e67e22', alpha=0.8)
axes[1].set_ylabel('Claim Frequency (%)')
axes[1].set_title('Monthly Claim Frequency')
axes[1].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('data/plot3_temporal.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → Saved plot3_temporal.png")

# ── 7. Guiding Q4: Vehicle Makes ─────────────────────
print("\nTop 10 highest avg claim vehicle makes:")
make_col = 'make' if 'make' in df.columns else [c for c in df.columns if 'make' in c.lower()][0] if any('make' in c.lower() for c in df.columns) else None
if make_col:
    make_claims = df.groupby(make_col).agg(
        AvgClaims=('TotalClaims','mean'),
        Count=('PolicyID','count')
    ).query('Count >= 100').sort_values('AvgClaims', ascending=False)
    print("\nTop 10 highest avg claim vehicle makes:")
    print(make_claims.head(10).round(2))
    print("\nTop 10 lowest avg claim vehicle makes:")
    print(make_claims.tail(10).round(2))
else:
    print("\nMake column not found - skipping vehicle analysis")

# ── 8. Plot 4: Correlation Matrix ────────────────────
print("\nGenerating Plot 4: Correlation Matrix...")
num_cols = ['TotalPremium','TotalClaims','LossRatio','Margin',
            'CustomValueEstimate','SumInsured','CalculatedPremiumPerTerm',
            'kilowatts','Cylinders']
corr = df[num_cols].dropna().corr()
fig, ax = plt.subplots(figsize=(11, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, ax=ax, linewidths=0.5, annot_kws={'size': 8})
ax.set_title('Correlation Matrix — Key Financial & Vehicle Features',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('data/plot4_correlation.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → Saved plot4_correlation.png")

# ── 9. Plot 5: Outlier Boxplots ──────────────────────
print("Generating Plot 5: Outlier Detection...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Outlier Detection — Key Financial Variables',
             fontsize=13, fontweight='bold')
for ax, col in zip(axes, ['TotalPremium','TotalClaims','CustomValueEstimate']):
    data = df[col].dropna()
    ax.boxplot(data[data.between(data.quantile(0.01), data.quantile(0.99))],
               patch_artist=True,
               boxprops=dict(facecolor='#3498db', alpha=0.6))
    ax.set_title(col)
    ax.set_ylabel('ZAR')
plt.tight_layout()
plt.savefig('data/plot5_outliers.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → Saved plot5_outliers.png")

print("\n" + "="*50)
print("EDA COMPLETE. All 5 plots saved to data/")
print("="*50)