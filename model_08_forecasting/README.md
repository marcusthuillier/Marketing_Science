# Model 08 — EPL Standings Forecasting

**Discipline:** Marketing DS + Growth DS
**Method:** Baseline extrapolation vs ARIMA vs Facebook Prophet — model horse race with RMSE comparison

---

## The Question

> Which forecasting model best predicts final Premier League standings from mid-season data — and does a simple baseline beat the complex models?

---

## Business Parallel

Each model maps directly to a core technique used in tech company data science roles. The sports domain makes the work memorable and shareable — the methods are identical to what you'd use on user or customer data.

---

## Method Summary

Baseline extrapolation vs ARIMA vs Facebook Prophet — model horse race with RMSE comparison

---

## Step-by-Step Plan

### 1. Data Collection
10 seasons of EPL results from Football-Data.org. Rolling standings table: team points after each matchweek (1-38).

### 2. Train/Test Setup
For each season: train on matchweeks 1-19, forecast 20-38. Run for 10 seasons = 10 evaluation sets.

### 3. Baseline Model
Points per game in first half × remaining games + home/away fixture adjustment. The benchmark everything must beat.

### 4. ARIMA Model
Fit ARIMA on each team's points time series. statsmodels auto_arima for parameter selection.

### 5. Prophet Model
Treat each team's trajectory as a time series. Add title race pressure as binary regressor.

### 6. Comparison
RMSE on final points + position accuracy. Does the baseline beat the complex models?


---

## Key Outputs

| File | Description |
|------|-------------|
| `outputs/forecasting_comparison.csv` | — |
| `outputs/model_accuracy.png` | — |
| `outputs/forecast_trajectories.html` | — |

---

## Results

<!-- Fill in after running the notebook -->

**Key finding:** _

**Surprising result:** _

---

## How to Run

```bash
conda activate ds_portfolio
jupyter notebook model_08_forecasting.ipynb
```

---

## Skill Mapping (for interviews)

| What you built | What interviewers call it |
|---------------|--------------------------|
| Three-model horse race with proper train/test setup | Forecasting model selection |
| ARIMA with auto parameter selection | Time series modeling |
| Facebook Prophet with custom regressors | Bayesian structural time series |
| RMSE + position accuracy comparison table | Forecast evaluation metrics |

---

## LinkedIn Post

See [linkedin_post.md](linkedin_post.md) for a ready-to-publish draft.
