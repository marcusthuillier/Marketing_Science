# Model 11 — Rookie Activation Analysis

**Discipline:** Product DS
**Method:** Logistic regression + XGBoost + SHAP — find the single strongest rookie predictor

---

## The Question

> Which early-career stats in a player's first season best predict long-term success — what is the 'aha moment' for NBA rookies?

---

## Business Parallel

Each model maps directly to a core technique used in tech company data science roles. The sports domain makes the work memorable and shareable — the methods are identical to what you'd use on user or customer data.

---

## Method Summary

Logistic regression + XGBoost + SHAP — find the single strongest rookie predictor

---

## Step-by-Step Plan

### 1. Data Collection
All NBA rookies 1990-2015 via sportsipy. Rookie features: PTS, AST, REB, STL, BLK, TS%, USG%, MPG, draft position, age. Career outcome: >=5 seasons with >=1000 minutes.

### 2. Label Creation
Success = >=5 seasons significant contribution. Aim for ~30% success rate. Flag injury-shortened careers.

### 3. Feature Importance
Logistic regression (interpretable). XGBoost (higher accuracy). SHAP values: which rookie stat is #1?

### 4. Aha Moment
Find threshold: e.g. rookies with >1.5 AST/game have 3x higher success rate. Test multiple candidate thresholds. This is the Facebook '7 friends in 10 days' equivalent.

### 5. Draft Position Analysis
After controlling for rookie stats, does draft position still predict success?

### 6. Cohort Validation
Apply model to 2018-2020 rookies. Compare predictions to current reality.


---

## Key Outputs

| File | Description |
|------|-------------|
| `outputs/activation_model.pkl` | — |
| `outputs/rookie_predictions.csv` | — |
| `outputs/aha_moment_chart.png` | — |

---

## Results

<!-- Fill in after running the notebook -->

**Key finding:** _

**Surprising result:** _

---

## How to Run

```bash
conda activate ds_portfolio
jupyter notebook model_11_activation_analysis.ipynb
```

---

## Skill Mapping (for interviews)

| What you built | What interviewers call it |
|---------------|--------------------------|
| Binary classification: long-term career success from rookie season only | Activation / aha moment analysis |
| Logistic regression for interpretable coefficients | Interpretable ML for stakeholders |
| SHAP on XGBoost to find the single strongest predictor | Feature importance for business insight |
| Threshold analysis: where does the aha moment kick in? | Threshold / activation modeling |

---
