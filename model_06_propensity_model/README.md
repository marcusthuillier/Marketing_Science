# Model 06 — Title Propensity Model

**Discipline:** Marketing DS
**Method:** LightGBM classifier + Platt scaling calibration + weekly probability timelines

---

## The Question

> Can we build a model that gives a real-time probability of a team winning the title — updated weekly — and how does it compare to betting market implied probability?

---

## Business Parallel

Each model maps directly to a core technique used in tech company data science roles. The sports domain makes the work memorable and shareable — the methods are identical to what you'd use on user or customer data.

---

## Method Summary

LightGBM classifier + Platt scaling calibration + weekly probability timelines

---

## Step-by-Step Plan

### 1. Data Collection
10+ seasons of EPL data via Football-Data.org. Weekly snapshots: points, goal difference, xG, ELO, form last 5, injury flag. Label: did this team win the title?

### 2. Feature Engineering
Points pace (projected final points). xG differential. Remaining schedule difficulty. Historical base rate: how often do teams with X points at week Y win?

### 3. Modelling
Train: all seasons except last 2. LightGBM classifier. Calibrate: Platt scaling. Evaluate: Brier score, calibration curve, AUC-ROC.

### 4. Probability Timeline
For each historical season: plot weekly title probability for all teams. The drama arc of a season in data form.

### 5. Validation
Compare mid-season model probabilities to betting odds. Find systematic biases.

### 6. Visualization
Line chart: weekly title probability top 4-6 teams. Calibration plot. Season drama score: variance in probability over the season.


---

## Key Outputs

| File | Description |
|------|-------------|
| `outputs/propensity_model.pkl` | — |
| `outputs/weekly_probabilities.csv` | — |
| `outputs/probability_timeline.html` | — |

---

## Results

<!-- Fill in after running the notebook -->

**Key finding:** _

**Surprising result:** _

---

## How to Run

```bash
conda activate ds_portfolio
jupyter notebook model_06_propensity_model.ipynb
```

---

## Skill Mapping (for interviews)

| What you built | What interviewers call it |
|---------------|--------------------------|
| LightGBM classifier on weekly team snapshots | Propensity / lead scoring modeling |
| Platt scaling for probability calibration | Model calibration — critical for decision-making |
| Brier score + calibration curve evaluation | Probabilistic model evaluation |
| Compared model vs betting market probabilities | Ground truth benchmarking |

---
