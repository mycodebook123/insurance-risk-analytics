# Insurance Risk Analytics

End-to-end insurance risk analytics and predictive modeling pipeline for AlphaCare Insurance Solutions (ACIS), analyzing 18 months of South African auto-insurance claim data (Feb 2014 – Aug 2015).

## Project Structure

- `data/` — dataset tracked by DVC (not committed to Git)
- `notebooks/` — Jupyter notebooks for EDA, hypothesis testing, and modeling
- `src/` — reusable Python modules
- `tests/` — unit tests
- `reports/` — final report

## Setup

```bash
pip install -r requirements.txt
```

## How to Run

```bash
# EDA
jupyter notebook notebooks/01_eda.ipynb

# Hypothesis Testing
jupyter notebook notebooks/02_hypothesis_testing.ipynb

# Modeling
jupyter notebook notebooks/03_modeling.ipynb

# Tests
pytest tests/ -v
```

## Data Pipeline (DVC)

```bash
dvc pull        # download tracked data
dvc push        # push new data versions
```

## Key Metrics

- **Loss Ratio** = TotalClaims / TotalPremium
- **Margin** = TotalPremium − TotalClaims
- **Claim Frequency** = proportion of policies with at least one claim
- **Claim Severity** = average claim amount where claim > 0