# Insurance Risk Analytics

End-to-end insurance risk analytics and predictive modeling pipeline for AlphaCare Insurance Solutions (ACIS), analyzing 18 months of South African auto-insurance claim data (Feb 2014 – Aug 2015).

## Project Structure
insurance-risk-analytics/
├── .github/workflows/ci.yml     # CI/CD pipeline
├── data/                        # tracked by DVC (not committed to Git)
├── notebooks/                   # Jupyter notebooks
│   ├── 01_eda.ipynb
│   ├── 02_hypothesis_testing.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── data_loader.py           # Data loading and derived metrics
│   ├── eda_utils.py             # EDA utility functions
│   ├── hypothesis_tests.py      # Statistical testing functions
│   └── modeling.py              # ML modeling pipeline
├── scripts/
│   ├── run_eda.py               # EDA pipeline (5 plots)
│   ├── run_hypothesis.py        # Hypothesis testing pipeline
│   └── run_modeling.py          # Modeling pipeline (LR, RF, XGBoost, SHAP)
├── tests/
│   └── test_pipeline.py         # 14 unit tests (all passing)
├── reports/
│   └── final_report.md
├── requirements.txt
└── README.md

## Setup

```bash
pip install -r requirements.txt
```

## How to Run

```bash
# EDA (generates 5 plots)
python scripts/run_eda.py

# Hypothesis Testing
python scripts/run_hypothesis.py

# Modeling (Linear Regression, Random Forest, XGBoost + SHAP)
python scripts/run_modeling.py

# Unit Tests
python -m pytest tests/test_pipeline.py -v
```

## Data Pipeline (DVC)

```bash
dvc pull        # download tracked data from remote
dvc push        # push new data versions to remote
```

Two data versions tracked:
- `data/MachineLearningRating_v3.txt` — raw dataset (1M rows, 52 columns)
- `data/insurance_cleaned.csv` — cleaned dataset (49 columns, nulls filled)

## Key Metrics

- **Loss Ratio** = TotalClaims / TotalPremium
- **Margin** = TotalPremium − TotalClaims
- **Claim Frequency** = proportion of policies with at least one claim
- **Claim Severity** = average claim amount where claim > 0

## Key Findings

### EDA
- Overall portfolio Loss Ratio: **1.0477** (portfolio is unprofitable)
- Gauteng has the highest Loss Ratio (1.22), Northern Cape the lowest (0.28)
- TotalClaims and TotalPremium are heavily right-skewed with significant outliers
- SUZUKI has the highest average claim amount; FORD, HUMMER, VOLVO the lowest

### Hypothesis Testing

| Hypothesis | Test | p-value | Decision |
|---|---|---|---|
| H1: No risk diff across provinces | Chi-Squared | 0.000000 | **Reject H0** |
| H2: No risk diff between zip codes | Welch T-Test | 0.700208 | Fail to Reject H0 |
| H3: No margin diff between zip codes | Welch T-Test | 0.244462 | Fail to Reject H0 |
| H4: No risk diff between genders | Chi-Squared | 0.951464 | Fail to Reject H0 |

**Business insight**: Province is a statistically significant risk driver (p < 0.001). Gauteng and KwaZulu-Natal are high-risk provinces warranting premium adjustments. Gender and zip code do not show significant differences at the 5% level.

### Modeling Results

| Model | RMSE | R² |
|---|---|---|
| Linear Regression | 34,124 | 0.2759 |
| Random Forest | 35,780 | 0.2039 |
| XGBoost | 36,925 | 0.1522 |

**Best model**: Linear Regression (R² = 0.28). Low R² reflects inherent unpredictability of claim severity — a known challenge in insurance modeling.

**Top SHAP features**: SumInsured, CalculatedPremiumPerTerm, TotalPremium, CustomValueEstimate, VehicleAge

### Risk-Based Pricing Formula
Premium = (P(claim) × Predicted Severity) + Expense Loading + Profit Margin

## Unit Tests
14/14 tests passing. Covers: derived metrics, margin calculation, HasClaim binary validation, loss ratio, hypothesis test structure and output validation.

## Ethical Considerations
- Sampling bias: dataset covers Feb 2014–Aug 2015 only
- Gender and zip code showed no significant risk differences — using these for pricing may be discriminatory and is not recommended
- Low model R² indicates significant unexplained variance — pricing decisions should not rely solely on model output