# Model 06 — World Cup Winner Propensity Model

**Discipline:** Marketing DS
**Method:** LightGBM classifier + Platt scaling calibration + SHAP explainability

---

## The Question

> Which national teams have the profile of a World Cup winner — and how similar is this year's field to past champions?

---

## Business Parallel

Propensity models score leads by their similarity to customers who converted. This model does the same thing for national football teams: it learns what historical World Cup winners look like across a set of measurable features and scores every team in the field.

"This team has the profile of a World Cup winner" is the same sentence as "this lead has the profile of a customer who converts." Same technique. Different domain.

---

## Method Summary

Binary classification. Label: `won_world_cup` (1 = winner, 0 = eliminated at any stage). Features built from pre-tournament data only — ELO, form, historical WC record, continent, host flag. Output: a propensity score 0-100% representing how closely a team matches the historical winner profile.

Class imbalance is real (~3% positive rate, 1 winner per 32 teams per tournament) and honest. That is what a propensity model in marketing looks like too. Not corrected at training time — noted and monitored in calibration.

---

## Data Sources

| Source | What it provides |
|--------|-----------------|
| `martj42/international_results` (GitHub CSV) | All international match results since 1872 — used for ELO computation and form stats |
| Computed in-script | Rolling ELO ratings updated after every international match using standard Elo formula |
| Hardcoded | WC winner per year, WC host per year, tournament start dates |

No paid data. No API keys required. One CSV download.

---

## Features

| Feature | Description |
|---------|-------------|
| `elo` | Team ELO rating at tournament start date (computed from full match history) |
| `win_rate_12m` | Win rate in the 12 months before the tournament |
| `gf_pg_12m` | Goals scored per game in the 12 months before the tournament |
| `ga_pg_12m` | Goals conceded per game in the 12 months before the tournament |
| `wc_apps` | Number of previous World Cup appearances |
| `wc_wins` | Number of previous World Cup titles |
| `is_host` | Host nation flag |
| `continent_enc` | Continental confederation (UEFA, CONMEBOL, CONCACAF, CAF, AFC — encoded) |

---

## Step-by-Step Plan

### 1. Data Collection
Fetch all international match results from `martj42/international_results`. Cache locally.

### 2. ELO Computation
Compute rolling ELO for every national team updated after every match. K-factor weighted by match importance (World Cup > continental championship > qualifying > friendly). Take snapshot of each team's ELO at tournament start date.

### 3. Feature Engineering
Build a team-tournament feature matrix: one row per team per World Cup (1990-2018 for training, 2022 for testing). Features: ELO snapshot, form stats, historical WC record, host flag, continent.

### 4. Model Training
LightGBM binary classifier. No class rebalancing at training time. Evaluate: AUC-ROC, Brier score.

### 5. Calibration
Platt scaling. Plot calibration curve. Compare predicted probabilities to actual win rates.

### 6. SHAP Analysis
What features drive the winner profile? Is it ELO? Historical record? Continental dominance? Beeswarm plot.

### 7. Scoring
Score 2022 field (held-out test set). Score likely 2026 candidates. Rank by propensity.

### 8. Visualization
- Propensity bar chart: 2022 field ranked by score, actual winner highlighted
- SHAP beeswarm: feature importance
- Calibration curve
- 2026 candidate scores

---

## Key Outputs

| File | Description |
|------|-------------|
| `outputs/propensity_scores_2022.png` | 2022 field scored and ranked, actual winner marked |
| `outputs/shap_beeswarm.png` | What drives the winner profile |
| `outputs/calibration_curve.png` | Model calibration |
| `outputs/propensity_scores_2026.png` | 2026 candidate scores |

---

## Results

<!-- Fill in after running -->

**Key finding:** _

**Surprising result:** _

---

## How to Run

```bash
conda activate ds_portfolio
python run_model_06.py
```

---

## Skill Mapping (for interviews)

| What you built | What interviewers call it |
|---------------|--------------------------|
| LightGBM classifier trained on historical winner profiles | Propensity / lead scoring model |
| Platt scaling for probability output | Model calibration |
| Brier score + calibration curve | Probabilistic model evaluation |
| SHAP beeswarm on winner features | Feature attribution / explainability |
| Scored 2026 candidates | Out-of-sample scoring / model deployment |
| 3% positive rate, no rebalancing at training | Class imbalance handling |
