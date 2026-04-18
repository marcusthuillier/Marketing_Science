# Model 04 — NBA Player Churn Model

**Discipline:** Marketing DS
**Method:** XGBoost binary classifier + SHAP explainability + AUC-ROC evaluation

---

## The Question

> Can we predict which NBA players are about to churn out of a team's rotation — and which early warning signals are most predictive?

---

## Business Parallel

Each model maps directly to a core technique used in tech company data science roles. The sports domain makes the work memorable and shareable — the methods are identical to what you'd use on user or customer data.

---

## Method Summary

XGBoost binary classifier + SHAP explainability + AUC-ROC evaluation

---

## Step-by-Step Plan

### 1. Data Collection
20 seasons of NBA player stats (2003-2023) via sportsipy. Features: Age, MPG, PTS, AST, REB, TS%, USG%, PER, +/-.

### 2. Label Creation
Churn = minutes dropped >40% YoY OR left league. Retained = within 20% of prior season. Flag injury seasons separately.

### 3. Feature Engineering
Rolling 2-year averages. Quadratic age term. Efficiency metrics. USG% trend. Playoff team vs lottery team flag.

### 4. Modelling
Train on 2003-2018, test on 2019-2023. XGBoost classifier. GridSearchCV tuning. AUC-ROC, precision/recall, confusion matrix.

### 5. Explainability
SHAP summary beeswarm (global). SHAP waterfall (individual players). Identify #1 churn signal.

### 6. Validation
Backtest: 5 players flagged in 2022 — did they churn in 2023? Discuss false positives.


---

## Key Outputs

| File | Description |
|------|-------------|
| `outputs/churn_model.pkl` | — |
| `outputs/churn_predictions.csv` | — |
| `outputs/shap_plots.png` | — |

---

## Results

<!-- Fill in after running the notebook -->

**Key finding:** _

**Surprising result:** _

---

## How to Run

```bash
conda activate ds_portfolio
jupyter notebook model_04_churn_model.ipynb
```

---

## Skill Mapping (for interviews)

| What you built | What interviewers call it |
|---------------|--------------------------|
| Binary classification with temporal train/test split | Churn modeling with proper leakage prevention |
| SHAP summary + waterfall plots | ML explainability for stakeholder communication |
| AUC-ROC, precision/recall, confusion matrix | Classification model evaluation |
| Rolling 2-year averages as features | Feature engineering for trend signals |

---

## LinkedIn Post

See [linkedin_post.md](linkedin_post.md) for a ready-to-publish draft.
