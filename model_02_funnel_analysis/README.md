# Model 02 — Fan Funnel Analysis

**Discipline:** Growth DS
**Method:** Stage-to-stage funnel conversion rates + Sankey diagram + Chi-square testing

---

## The Question

> Where do sports fans drop off on their journey from casual viewer to committed season ticket holder — and what is the biggest leak in the funnel?

---

## Business Parallel

Each model maps directly to a core technique used in tech company data science roles. The sports domain makes the work memorable and shareable — the methods are identical to what you'd use on user or customer data.

---

## Method Summary

Stage-to-stage funnel conversion rates + Sankey diagram + Chi-square testing

---

## Step-by-Step Plan

### 1. Data Collection
Pull Google Trends (pytrends) for 5-10 teams as awareness proxy. Pull Reddit subscriber counts via PRAW. Collect public attendance figures and season ticket estimates.

### 2. Funnel Construction
Map stages: Awareness (search volume) → Interest (Reddit followers) → Attendance (games/year) → Loyalty (season ticket holder). Estimate population at each stage per team.

### 3. Conversion Rate Analysis
Calculate stage-to-stage conversion %. Compare across teams. Find the biggest drop-off stage.

### 4. Statistical Testing
Chi-square test: are conversion differences between teams significant? Pearson correlation: does league position predict funnel efficiency?

### 5. Visualization
Sankey diagram (Plotly), conversion rate bar chart by team, scatter of league position vs funnel efficiency.

### 6. Outputs
Save funnel_data.csv, sankey_diagram.html, conversion_analysis.png


---

## Key Outputs

| File | Description |
|------|-------------|
| `outputs/funnel_data.csv` | — |
| `outputs/sankey_diagram.html` | — |
| `outputs/conversion_analysis.png` | — |

---

## Results

<!-- Fill in after running the notebook -->

**Key finding:** _

**Surprising result:** _

---

## How to Run

```bash
conda activate ds_portfolio
jupyter notebook model_02_funnel_analysis.ipynb
```

---

## Skill Mapping (for interviews)

| What you built | What interviewers call it |
|---------------|--------------------------|
| Built stage-to-stage conversion rates from proxies | Funnel / conversion analysis |
| Sankey diagram for drop-off visualization | Stakeholder-ready funnel reporting |
| Chi-square test on team conversion differences | Statistical hypothesis testing |
| Aggregated multiple public data sources as proxies | Proxy metric design / measurement |

---

## LinkedIn Post

See [linkedin_post.md](linkedin_post.md) for a ready-to-publish draft.
