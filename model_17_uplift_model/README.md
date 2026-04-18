# Model 17 — Player Uplift Modeling

**Discipline:** Growth DS
**Method:** T-Learner meta-learner uplift model + propensity score matching + Qini curve evaluation

---

## The Question

> Which players actually respond to coaching interventions — and which would have improved regardless of what the coach did?

---

## Business Parallel

Each model maps directly to a core technique used in tech company data science roles. The sports domain makes the work memorable and shareable — the methods are identical to what you'd use on user or customer data.

---

## Method Summary

T-Learner meta-learner uplift model + propensity score matching + Qini curve evaluation

---

## Step-by-Step Plan

### 1. Data Collection
Players who experienced a manager change (treatment) vs comparable players who did not (control). Match on: position, age, prior stats, team strength.

### 2. Treatment/Control Setup
Treatment: manager changed mid-season or between seasons. Control: same profile, no manager change. Propensity score matching.

### 3. T-Learner Uplift Model
Train separate models for treatment and control groups. Uplift = T-model prediction - C-model prediction.

### 4. Player Segmentation
Persuadables: improve because of intervention. Sure Things: improve regardless. Lost Causes: won't improve regardless. Sleeping Dogs: perform worse with intervention.

### 5. Qini Curve
Evaluate model: Qini curve vs random targeting baseline.

### 6. Insights
Which positions respond most to manager changes? Does age predict responsiveness?


---

## Key Outputs

| File | Description |
|------|-------------|
| `outputs/uplift_model.pkl` | — |
| `outputs/player_uplift_scores.csv` | — |
| `outputs/qini_curve.png` | — |

---

## Results

<!-- Fill in after running the notebook -->

**Key finding:** _

**Surprising result:** _

---

## How to Run

```bash
conda activate ds_portfolio
jupyter notebook model_17_uplift_model.ipynb
```

---

## Skill Mapping (for interviews)

| What you built | What interviewers call it |
|---------------|--------------------------|
| T-Learner: separate models for treatment and control | Meta-learner uplift modeling |
| Propensity score matching for comparable groups | Selection bias correction |
| Qini curve: uplift equivalent of ROC | Uplift model evaluation |
| Persuadables / Sure Things / Lost Causes segmentation | Uplift-based targeting strategy |

---

## LinkedIn Post

See [linkedin_post.md](linkedin_post.md) for a ready-to-publish draft.
