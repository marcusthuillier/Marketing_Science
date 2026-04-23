# Model 15 — A/B Testing + Causal Inference

**Discipline:** Growth DS + Product DS
**Method:** Interrupted time series + CausalImpact (Bayesian structural time series) + DiD

---

## The Question

> Did a manager change actually improve a team's results — or would they have improved anyway? Using causal inference to separate signal from noise.

---

## Business Parallel

Each model maps directly to a core technique used in tech company data science roles. The sports domain makes the work memorable and shareable — the methods are identical to what you'd use on user or customer data.

---

## Method Summary

Interrupted time series + CausalImpact (Bayesian structural time series) + DiD

---

## Step-by-Step Plan

### 1. Data Collection
10+ manager changes in top leagues 2015-2023. 15 matches before/after each. Metric: xG per match (removes luck). Adjust for opponent strength.

### 2. Naive Before/After
Mean xG before vs after. Simple t-test. Finding: ~60% show improvement, partly due to regression to the mean.

### 3. Confounder Control
Opponent strength adjustment. League position at time of change. Injury context.

### 4. CausalImpact
causalimpact Python package. Counterfactual: what would have happened without the change? Run for 10 manager changes.

### 5. Difference-in-Differences
Find comparable teams who did not change managers. DiD estimate = true treatment effect.

### 6. Meta-Analysis
Average causal effect across 10 changes. Separate: sacking effect (short-term) vs sustained improvement.


---

## Key Outputs

| File | Description |
|------|-------------|
| `outputs/causal_impact_results.csv` | — |
| `outputs/causal_plots/` | — |
| `outputs/meta_analysis_summary.png` | — |

---

## Results

<!-- Fill in after running the notebook -->

**Key finding:** _

**Surprising result:** _

---

## How to Run

```bash
conda activate ds_portfolio
jupyter notebook model_15_causal_inference.ipynb
```

---

## Skill Mapping (for interviews)

| What you built | What interviewers call it |
|---------------|--------------------------|
| CausalImpact: actual vs counterfactual trajectory | Bayesian causal inference |
| Difference-in-differences with comparable control teams | DiD / quasi-experimental design |
| Regression to the mean as an explicit confounder | Confounder identification and control |
| Meta-analysis across 10 interventions | Aggregating causal evidence |

---
