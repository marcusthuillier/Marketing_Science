# Model 14 — Marketing Mix Modeling (MMM)

**Discipline:** Marketing DS
**Method:** Bayesian MMM using PyMC-Marketing with adstock and saturation transformations

---

## The Question

> What combination of factors — rest days, travel, home advantage, form, and weather — best explains match outcomes, with saturation and diminishing returns modeled?

---

## Business Parallel

Each model maps directly to a core technique used in tech company data science roles. The sports domain makes the work memorable and shareable — the methods are identical to what you'd use on user or customer data.

---

## Method Summary

Bayesian MMM using PyMC-Marketing with adstock and saturation transformations

---

## Step-by-Step Plan

### 1. Data Collection
Football-Data.org: match results + dates. Calculate rest days. Stadium lat/long → haversine travel distance (geopy). Open-Meteo: historical weather per match. Form: rolling 5-game points. ClubElo: opponent strength.

### 2. Feature Engineering
Rest days (continuous). Travel km (0 for home). Weather severity score. Form. Opponent ELO. Season stage flag.

### 3. MMM Setup (PyMC-Marketing)
Target: goals scored or win probability. Saturation on rest days (diminishing returns >7 days). Adstock on form (recent form decays).

### 4. Model Fitting
PyMC-Marketing MMM class. MCMC sampling. Check: R-hat, trace plots. Posterior predictive checks.

### 5. Contribution Analysis
Decompose each match: baseline + rest + home advantage + weather + form. Response curves per factor.

### 6. Key Insights
At what rest level does benefit plateau? How much does home advantage contribute vs opponent strength? Is weather significant?


---

## Key Outputs

| File | Description |
|------|-------------|
| `outputs/mmm_model.pkl` | — |
| `outputs/contribution_data.csv` | — |
| `outputs/response_curves.png` | — |
| `outputs/waterfall_examples.html` | — |

---

## Results

<!-- Fill in after running the notebook -->

**Key finding:** _

**Surprising result:** _

---

## How to Run

```bash
conda activate ds_portfolio
jupyter notebook model_14_mmm.ipynb
```

---

## Skill Mapping (for interviews)

| What you built | What interviewers call it |
|---------------|--------------------------|
| Bayesian MMM with PyMC-Marketing (industry-standard framework) | Marketing Mix Modeling |
| Adstock + saturation transformations on input variables | Media transformation / diminishing returns |
| MCMC sampling with R-hat convergence diagnostics | Bayesian inference in practice |
| Contribution waterfall decomposition per match | Factor attribution / decomposition |

---

## LinkedIn Post

See [linkedin_post.md](linkedin_post.md) for a ready-to-publish draft.
