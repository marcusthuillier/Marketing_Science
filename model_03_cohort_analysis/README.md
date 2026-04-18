# Model 03 — NBA Draft Cohort Analysis

**Discipline:** Growth DS
**Method:** Cohort retention analysis + heatmap (SaaS retention grid format) + ANOVA

---

## The Question

> Do some NBA draft classes produce sustained value over time — and can cohort analysis reveal which draft years were genuinely exceptional vs just lucky?

---

## Business Parallel

Each model maps directly to a core technique used in tech company data science roles. The sports domain makes the work memorable and shareable — the methods are identical to what you'd use on user or customer data.

---

## Method Summary

Cohort retention analysis + heatmap (SaaS retention grid format) + ANOVA

---

## Step-by-Step Plan

### 1. Data Collection
Pull all NBA draft picks 1990-2018 via sportsipy. Season-by-season Win Shares per player. Define career year as years since draft, capped at 15.

### 2. Cohort Construction
Group by draft year. Calculate: total players, survival rates at Year 3/5/8/10, average and median Win Shares per career year.

### 3. Retention Heatmap
Matrix: rows = draft years, columns = career years. Values = avg Win Shares. Exact format of a growth cohort retention table.

### 4. Analysis
Identify best/worst draft classes. Calculate LTV equivalent. Analyze by draft position (lottery vs second round).

### 5. Statistical Testing
ANOVA across cohort LTVs. Trend analysis: are recent drafts better than historical?

### 6. Visualization
Seaborn cohort heatmap, line chart of avg career Win Shares by draft year, bar chart of top 10 draft classes by total value.


---

## Key Outputs

| File | Description |
|------|-------------|
| `outputs/cohort_matrix.csv` | — |
| `outputs/cohort_heatmap.png` | — |
| `outputs/draft_class_ltv.csv` | — |

---

## Results

<!-- Fill in after running the notebook -->

**Key finding:** _

**Surprising result:** _

---

## How to Run

```bash
conda activate ds_portfolio
jupyter notebook model_03_cohort_analysis.ipynb
```

---

## Skill Mapping (for interviews)

| What you built | What interviewers call it |
|---------------|--------------------------|
| Grouped players by draft year, tracked value over career years | Cohort retention analysis |
| Built cohort heatmap identical to SaaS retention grids | Growth analytics / retention reporting |
| LTV equivalent: total career Win Shares per cohort | LTV framing applied to non-standard domain |
| ANOVA on cohort value differences | Statistical significance testing |

---

## LinkedIn Post

See [linkedin_post.md](linkedin_post.md) for a ready-to-publish draft.
