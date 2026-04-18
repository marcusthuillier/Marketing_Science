# Model 16 — Transfer Incrementality Testing

**Discipline:** Growth DS
**Method:** Synthetic control + CausalImpact to quantify incremental impact of individual transfers

---

## The Question

> Did a major transfer signing cause team improvement — or were they already improving before the player arrived?

---

## Business Parallel

Each model maps directly to a core technique used in tech company data science roles. The sports domain makes the work memorable and shareable — the methods are identical to what you'd use on user or customer data.

---

## Method Summary

Synthetic control + CausalImpact to quantify incremental impact of individual transfers

---

## Step-by-Step Plan

### 1. Event Selection
5 high-profile transfers: clear successes, clear failures, ambiguous. Record exact signing date and debut.

### 2. Data Collection
Team xG per match for buying club. 20 matches pre-signing, 20 post-debut. Control: similar teams who didn't sign.

### 3. Synthetic Control
3-5 donor teams similar to treatment pre-signing. Optimally weight donors to replicate pre-signing trajectory.

### 4. CausalImpact
Actual post-debut vs synthetic control. Cumulative incremental xG. Confidence intervals.

### 5. Heterogeneity Analysis
Do expensive signings have higher incrementality? Do signings for struggling teams have higher uplift?

### 6. Findings
Many 'successful' transfers not as incremental as perceived. Some 'failed' transfers had positive incrementality.


---

## Key Outputs

| File | Description |
|------|-------------|
| `outputs/incrementality_results.csv` | — |
| `outputs/causal_impact_charts/` | — |
| `outputs/transfer_roi_scatter.png` | — |

---

## Results

<!-- Fill in after running the notebook -->

**Key finding:** _

**Surprising result:** _

---

## How to Run

```bash
conda activate ds_portfolio
jupyter notebook model_16_incrementality.ipynb
```

---

## Skill Mapping (for interviews)

| What you built | What interviewers call it |
|---------------|--------------------------|
| Synthetic control from donor teams pre-signing | Synthetic control method |
| CausalImpact for cumulative incremental xG | Incrementality measurement |
| Transfer ROI: incremental xG per £M transfer fee | Marketing ROI framing applied to sports |
| Heterogeneity: does fee size predict incrementality? | Treatment effect heterogeneity |

---

## LinkedIn Post

See [linkedin_post.md](linkedin_post.md) for a ready-to-publish draft.
