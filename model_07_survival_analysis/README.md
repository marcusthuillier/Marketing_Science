# Model 07 — Player Career Survival Analysis

**Discipline:** Marketing DS
**Method:** Kaplan-Meier survival curves by position + Cox Proportional Hazard regression (lifelines)

---

## The Question

> At what point in their career do players at different positions hit irreversible decline — and what factors extend or shorten the survival curve?

---

## Business Parallel

Each model maps directly to a core technique used in tech company data science roles. The sports domain makes the work memorable and shareable — the methods are identical to what you'd use on user or customer data.

---

## Method Summary

Kaplan-Meier survival curves by position + Cox Proportional Hazard regression (lifelines)

---

## Step-by-Step Plan

### 1. Data Collection
20+ years of NBA player data via sportsipy. Event: season with <500 minutes (functional career end). Time: career age (years since debut). Features: position, peak PER, TS%, draft round.

### 2. Survival Data Setup
Duration = career length. Event = 1 if churned, 0 if still playing (censored). lifelines library format.

### 3. Kaplan-Meier Curves
KM curves by position (PG, SG, SF, PF, C). Log-rank test: are survival curves significantly different? Median survival per position.

### 4. Cox Regression
Features: position, peak PER, peak TS%, draft position, height. Interpret hazard ratios. Check proportional hazard assumption.

### 5. Analysis
Which position declines earliest? Most protective factor. Career survival probability for current players.

### 6. Visualization
KM curves by position. Cox forest plot (hazard ratios + CI). Individual survival curves for 5 famous players.


---

## Key Outputs

| File | Description |
|------|-------------|
| `outputs/survival_curves.png` | — |
| `outputs/cox_model.pkl` | — |
| `outputs/player_survival_probs.csv` | — |

---

## Results

1,757 players (1996–2025 debuts), 1,383 career endings, 374 right-censored (still active). Overall median career: 6.0 seasons.

Median survival by position: SG 7.0 / SF 7.0 / C 7.0 / PF 8.0 seasons. Log-rank test across positions: p = 0.083 — not significant.

Cox PH hazard ratios: Peak Performance (standardized) HR = 0.273, p < 0.005 — every SD of peak performance cuts churn risk ~73%. Draft pick HR = 1.000, p = 0.77. Height HR = 0.989, p = 0.82. Position dummies all p > 0.45. Model concordance = 0.82.

**Key finding:** Peak performance is the only statistically significant predictor of career survival. Position, draft pick, and height are all noise once you control for it.

**Surprising result:** The conventional wisdom that bigs decline fastest doesn't hold — PFs have the *longest* median survival (8 seasons), not the shortest, and the position differences aren't significant anyway.

---

## How to Run

```bash
conda activate ds_portfolio
jupyter notebook model_07_survival_analysis.ipynb
```

---

## Skill Mapping (for interviews)

| What you built | What interviewers call it |
|---------------|--------------------------|
| Kaplan-Meier curves by position with log-rank test | Survival analysis / time-to-event modeling |
| Cox PH regression with hazard ratios + CIs | Multivariate survival regression |
| Censoring handled correctly for active players | Proper survival data setup |
| Forest plot of hazard ratios | Communicating regression results visually |

---
