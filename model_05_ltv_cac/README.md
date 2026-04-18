# Model 05 — Player LTV + CAC Modeling

**Discipline:** Marketing DS + Growth DS
**Method:** Gradient Boosting regression for LTV prediction + LTV:CAC ratio analysis by acquisition channel

---

## The Question

> Can we predict a player's total career value from just their first 3 seasons — and which acquisition channel (draft vs free agent vs trade) offers the best LTV:CAC ratio?

---

## Business Parallel

Each model maps directly to a core technique used in tech company data science roles. The sports domain makes the work memorable and shareable — the methods are identical to what you'd use on user or customer data.

---

## Method Summary

Gradient Boosting regression for LTV prediction + LTV:CAC ratio analysis by acquisition channel

---

## Step-by-Step Plan

### 1. Data Collection
NBA players 1990-2015 (full careers). Year 1-3 stats as features. Total career Win Shares as LTV target. First contract salary as CAC. Acquisition tag: draft / free agent / trade.

### 2. LTV Model
Features: Year 1-3 PER, Win Shares, TS%, USG%, Age, Draft position. Target: career Win Shares. Gradient Boosting Regressor. Evaluate: RMSE, R², residuals.

### 3. CAC Calculation
CAC = total salary in first 4 years (rookie contract). Free agents: first contract total. Trades: Spotrac trade value estimates.

### 4. LTV:CAC Analysis
Ratio per player. Group by acquisition channel. Which channel has best median LTV:CAC? Payback period: at what career year does player pay back their contract?

### 5. Insights
Late first-round picks with high LTV:CAC. Free agents with worst ratio. Best LTV:CAC by position.

### 6. Visualization
Scatter: predicted vs actual LTV. Box plot: LTV:CAC by channel. Payback curve: cumulative Win Shares vs salary.


---

## Key Outputs

| File | Description |
|------|-------------|
| `outputs/ltv_model.pkl` | — |
| `outputs/player_ltv_cac.csv` | — |
| `outputs/acquisition_channel_analysis.png` | — |

---

## Results

<!-- Fill in after running the notebook -->

**Key finding:** _

**Surprising result:** _

---

## How to Run

```bash
conda activate ds_portfolio
jupyter notebook model_05_ltv_cac.ipynb
```

---

## Skill Mapping (for interviews)

| What you built | What interviewers call it |
|---------------|--------------------------|
| Regression to predict career value from early indicators | LTV prediction modeling |
| LTV:CAC ratio calculation by acquisition channel | Unit economics / growth finance framing |
| Payback period curve per player | Payback period analysis |
| RMSE, R², residual plots for regression evaluation | Regression model diagnostics |

---

## LinkedIn Post

See [linkedin_post.md](linkedin_post.md) for a ready-to-publish draft.
