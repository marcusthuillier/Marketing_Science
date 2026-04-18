# Model 18 — Transfer Budget Allocation & Optimization

**Discipline:** Growth DS
**Method:** Hill function saturation curves per position + constrained optimization (scipy/PuLP)

---

## The Question

> Given a fixed transfer budget, how should a club optimally allocate spend across positions to maximize expected points — and where does marginal return diminish?

---

## Business Parallel

Each model maps directly to a core technique used in tech company data science roles. The sports domain makes the work memorable and shareable — the methods are identical to what you'd use on user or customer data.

---

## Method Summary

Hill function saturation curves per position + constrained optimization (scipy/PuLP)

---

## Step-by-Step Plan

### 1. Data Collection
Transfermarkt squad valuations for top 5 leagues. Group by position: GK, CB, FB, CM, AM, W, ST. FBref xG contribution per position.

### 2. Response Curve Fitting
For each position: squad value vs team xG contribution scatter. Fit Hill function: contribution = (investment^a) / (k^a + investment^a). Shows diminishing returns.

### 3. Marginal Return Analysis
Marginal xG per £1M at different investment levels. Find saturation point per position.

### 4. Optimization
Objective: maximize xG - xGA. Constraint: total budget = £X. Solver: scipy.optimize or PuLP. Output: optimal £ per position.

### 5. Scenario Analysis
Run for £50M, £100M, £200M. How does allocation strategy change with budget?

### 6. Validation
Back-test: teams closest to optimal distribution — did they outperform? Which clubs are most wasteful?


---

## Key Outputs

| File | Description |
|------|-------------|
| `outputs/response_curves.pkl` | — |
| `outputs/optimal_allocation.csv` | — |
| `outputs/budget_optimization.html` | — |

---

## Results

<!-- Fill in after running the notebook -->

**Key finding:** _

**Surprising result:** _

---

## How to Run

```bash
conda activate ds_portfolio
jupyter notebook model_18_budget_optimization.ipynb
```

---

## Skill Mapping (for interviews)

| What you built | What interviewers call it |
|---------------|--------------------------|
| Hill function saturation curve fitting per position | Response curve / diminishing returns modeling |
| Constrained optimization with scipy/PuLP | Budget allocation optimization |
| Marginal return analysis: where does spending stop helping? | Marginal ROI analysis |
| Scenario analysis: £50M / £100M / £200M budgets | Sensitivity / scenario analysis |

---

## LinkedIn Post

See [linkedin_post.md](linkedin_post.md) for a ready-to-publish draft.
