# Model 12 — Fan Engagement Model

**Discipline:** Product DS
**Method:** XGBoost regression on Google Trends engagement proxy + SHAP driver analysis

---

## The Question

> What factors actually drive fans to watch every match — and is winning the primary driver, or is drama/uncertainty more important?

---

## Business Parallel

Each model maps directly to a core technique used in tech company data science roles. The sports domain makes the work memorable and shareable — the methods are identical to what you'd use on user or customer data.

---

## Method Summary

XGBoost regression on Google Trends engagement proxy + SHAP driver analysis

---

## Step-by-Step Plan

### 1. Data Collection
pytrends: weekly Google search volume for 5 PL clubs (5 years). Football-Data.org: match results + standings. Build features: league position, points gap, opponent quality, rivalry flag, form last 5, match outcome.

### 2. Engagement Proxy
Relative Google search volume (100 = peak). Normalize within club. Lag: does a win drive engagement the week AFTER?

### 3. Feature Engineering
Match importance score: points gap × games remaining. Derby flag. Title race flag (within 5 points at matchweek N). Consecutive wins/losses streak.

### 4. Modelling
XGBoost regression: predict next week's search volume. SHAP analysis. Test: does losing occasionally INCREASE engagement?

### 5. Segmentation
Does the engagement model differ by club size? Big clubs: winning drives search. Smaller clubs: drama/survival drives search.

### 6. Visualization
SHAP beeswarm: top engagement drivers. Actual vs predicted line chart across 5 seasons. Scatter: match importance vs engagement with annotations.


---

## Key Outputs

| File | Description |
|------|-------------|
| `outputs/engagement_model.pkl` | — |
| `outputs/engagement_predictions.csv` | — |
| `outputs/shap_engagement_drivers.png` | — |

---

## Results

<!-- Fill in after running the notebook -->

**Key finding:** _

**Surprising result:** _

---

## How to Run

```bash
conda activate ds_portfolio
jupyter notebook model_12_engagement_model.ipynb
```

---

## Skill Mapping (for interviews)

| What you built | What interviewers call it |
|---------------|--------------------------|
| Google Trends as engagement proxy — creative measurement design | Proxy metric design |
| XGBoost regression on match-level weekly features | Engagement prediction modeling |
| SHAP to isolate winning vs drama as engagement driver | Causal feature attribution (observational) |
| Segmentation: does the model differ by club size? | Segment-level model interpretation |

---

## LinkedIn Post

See [linkedin_post.md](linkedin_post.md) for a ready-to-publish draft.
