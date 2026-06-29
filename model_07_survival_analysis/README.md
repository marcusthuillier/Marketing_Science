# Model 07 — Player Career Survival Analysis

**Discipline:** Marketing DS
**Method:** Kaplan-Meier survival curves by position + Cox Proportional Hazard regression (lifelines)

---

## The Question

> At what point in their career do players at different positions hit irreversible decline — and what factors extend or shorten the survival curve? And once you fit a Cox model, does it actually satisfy the assumption the whole technique depends on?

---

## Business Parallel

Each model maps directly to a core technique used in tech company data science roles. The sports domain makes the work memorable and shareable — the methods are identical to what you'd use on user or customer churn data. The proportional hazards check below is the same diagnostic a churn model needs and frequently skips: does a feature's effect on churn risk actually stay constant over a customer's lifetime, or does it matter more early and less late (or vice versa)?

---

## Method Summary

Kaplan-Meier survival curves by position, plus Cox Proportional Hazard regression (`lifelines`) on peak performance, draft position, height, and position. Validated with a Schoenfeld-residuals test of the proportional hazards assumption and out-of-sample concordance across repeated train/test splits.

---

## Step-by-Step Plan

### 1. Data Collection
NBA player-season stats (1996–2025 debuts) via `nba_api`, merged with draft history and player position/height from the league's player index. Event: career ended (last active season before 2025). Time: seasons played.

### 2. Survival Data Setup
Duration = career length in seasons. Event = 1 if the career has ended, 0 if still active (right-censored). 1,757 players, `lifelines`-format dataframe.

### 3. Kaplan-Meier Curves
KM curves by position (PG, SG, SF, PF, C). Log-rank test: are survival curves significantly different by position? Median survival per position.

### 4. Cox Regression
Features: standardized peak performance (composite of peak points/rebounds/assists), draft pick, standardized height, position dummies. Hazard ratios with 95% confidence intervals.

### 5. Validation
Two checks the model wouldn't otherwise get for free. First, a Schoenfeld-residuals test of the proportional hazards assumption — Cox regression assumes each covariate's hazard ratio is constant across a career, and that's worth actually testing, not assuming. Second, out-of-sample concordance: refit the model on repeated random 80/20 splits and score concordance on the held-out 20% each time, since the in-sample 0.82 alone doesn't rule out overfitting.

### 6. Visualization
KM curves by position with confidence bands. Cox forest plot (hazard ratios + 95% CI). Individual survival curves for 8 spotlight players.

---

## Key Outputs

| File | Description |
|------|-------------|
| `outputs/survival_curves.png` | Kaplan-Meier survival curves by position, with 95% confidence bands |
| `outputs/cox_hazard_ratios.png` | Forest plot of Cox PH hazard ratios with 95% CIs |
| `outputs/proportional_hazards_test.csv` | Schoenfeld-residuals test results per covariate |
| `outputs/out_of_sample_concordance.csv` | Concordance scores across 5 held-out 80/20 splits |
| `outputs/player_positions.csv` | Cached player position/height/draft data from `nba_api` |

---

## Results

1,757 players (1996–2025 debuts), 1,383 career endings, 374 right-censored (still active). Overall median career: 6.0 seasons.

Median survival by position: SG 7.0 / SF 7.0 / C 7.0 / PF 8.0 seasons. Log-rank test across positions: p = 0.083 — not significant.

Cox PH hazard ratios (95% CI): Peak Performance (standardized) HR = 0.273 [0.250, 0.298], p < 0.0005 — every SD of peak performance cuts churn risk ~73%. Draft pick HR = 1.000 [0.998, 1.002], p = 0.77. Height HR = 0.989 [0.898, 1.089], p = 0.82. Position dummies all p > 0.45, with wide CIs that comfortably straddle 1.0. In-sample concordance = 0.824.

The two validation checks: a Schoenfeld-residuals test flags peak performance as violating the proportional hazards assumption (test statistic 190.4, p = 2.6e-43) — its effect on churn risk isn't actually constant across a career, which the single HR of 0.273 quietly averages over. Everything else (draft pick, height, position) passes the assumption cleanly. Out-of-sample concordance, averaged across 5 random 80/20 splits, is 0.822 — essentially identical to the in-sample 0.824, so the model isn't meaningfully overfit despite the strength of the peak-performance effect.

**Key finding:** Peak performance is the only statistically significant predictor of career survival. Position, draft pick, and height are all noise once you control for it, and the model's predictive power holds up out-of-sample.

**Surprising result:** The conventional wisdom that bigs decline fastest doesn't hold — PFs have the *longest* median survival (8 seasons), not the shortest, and the position differences aren't significant anyway. The more important surprise is methodological: the strongest, most significant predictor in the whole model is also the one whose hazard ratio isn't actually constant over time. A single number (HR = 0.273) is doing real work hiding that peak performance most likely matters differently in year 3 of a career than year 13 — the average effect is real and large, but it's an average over something that's genuinely changing shape.

---

## How to Run

```bash
conda activate ds_portfolio
python run_model_07.py
```

---

## Skill Mapping (for interviews)

| What you built | What interviewers call it |
|---------------|--------------------------|
| Kaplan-Meier curves by position with log-rank test | Survival analysis / time-to-event modeling |
| Cox PH regression with hazard ratios + 95% CIs | Multivariate survival regression |
| Schoenfeld-residuals test of the proportional hazards assumption | Validating a model's core statistical assumption, not just its fit |
| Out-of-sample concordance across repeated train/test splits | Checking a strong in-sample metric against overfitting |
| Censoring handled correctly for active players | Proper survival data setup |
| Forest plot of hazard ratios | Communicating regression results visually |

---
