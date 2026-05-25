"""Script: run_hypothesis.py - A/B Hypothesis Testing"""
import sys
sys.path.insert(0, '.')
import warnings
warnings.filterwarnings('ignore')
from src.data_loader import load_data, add_derived_metrics
from src.hypothesis_tests import test_provinces, test_zipcodes_risk, test_zipcodes_margin, test_gender
import pandas as pd

print("="*50)
print("ALPHCARE INSURANCE - HYPOTHESIS TESTING")
print("="*50)

df = load_data()
df = add_derived_metrics(df)
print(f"Dataset loaded: {df.shape}\n")

print("Running H1: Province risk differences...")
h1 = test_provinces(df)
print(f"  p-value: {h1['p_value']} → {h1['Decision']}")
print(f"  {h1['Interpretation']}\n")

print("Running H2: Zip code risk differences...")
h2 = test_zipcodes_risk(df)
print(f"  p-value: {h2['p_value']} → {h2['Decision']}")
print(f"  {h2['Interpretation']}\n")

print("Running H3: Zip code margin differences...")
h3 = test_zipcodes_margin(df)
print(f"  p-value: {h3['p_value']} → {h3['Decision']}")
print(f"  {h3['Interpretation']}\n")

print("Running H4: Gender risk differences...")
h4 = test_gender(df)
print(f"  p-value: {h4['p_value']} → {h4['Decision']}")
print(f"  {h4['Interpretation']}\n")

results = pd.DataFrame([h1, h2, h3, h4])
print("="*50)
print("RESULTS SUMMARY TABLE")
print("="*50)
print(results[['Hypothesis','Test','KPI','p_value','Decision']].to_string(index=False))
print("\nHypothesis testing complete.")