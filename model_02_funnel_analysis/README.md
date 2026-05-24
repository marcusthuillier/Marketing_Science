# Model 02 — Fan Funnel Analysis

**Discipline:** Growth DS  
**Method:** Stage-to-stage funnel conversion rates + Sankey diagram + Pearson correlation + Chi-square testing  
**Data:** Instagram followers (official club profiles) · PL official attendance 2023-24 · UK Companies House annual reports

---

## The Question

> Where do Premier League fans drop off on their journey from digital follower to committed season ticket holder — and which clubs convert global audiences into local commitment most efficiently?

---

## Business Parallel

In product DS, funnel analysis tracks users dropping off between acquisition → activation → retention. The same logic applies to sports fans: each stage (Instagram follower → season ticket holder → matchday appearance) has a conversion rate. The technique is identical to what you'd use on SaaS or e-commerce data — the domain makes it memorable.

---

## Method

| Step | Technique | Why |
|------|-----------|-----|
| Data collection | Three published sources only | No proxies — Instagram exact, PL attendance official, STH from annual reports |
| Funnel construction | Stage-to-stage conversion rates | Standard funnel metric: what % move to the next stage |
| Statistical testing | Chi-square (proportion differences) + Pearson r (scale vs conversion) | Two questions: do clubs differ? does brand size predict conversion? |
| Visualization | Sankey diagram + horizontal bar charts + scatter | Stakeholder-readable funnel + club-level comparison |

---

## Key Outputs

| File | Description |
|------|-------------|
| `outputs/funnel_data.csv` | Per-club funnel metrics: Instagram, STH, attendance, conversion rates |
| `outputs/sankey_diagram.html` | Interactive Plotly Sankey: Instagram → STH → matchday split (aggregate) |
| `outputs/conversion_analysis.png` | Instagram→STH conversion + STH% of crowd per club |
| `outputs/brand_vs_commitment.png` | Scatter: Instagram size vs conversion + STH attendance share by club |
| `outputs/awareness_vs_sth_conversion.png` | Supporting conversion chart |

---

## Results

**Key finding:** Instagram → STH conversion is strongly negatively correlated with follower count (Pearson r = -0.631, p = 0.003). Smaller clubs convert a meaningfully higher share of their digital audience into committed, paying locals. Non-Big-Six clubs average 1.18% conversion vs 0.13% for the Big Six — a 9.2x gap. Chi-square confirms STH proportions differ significantly across clubs (p < 0.0001).

**Surprising result:** Chelsea has the highest STH share of matchday crowd at 93.2% — nearly their entire home crowd holds a season ticket. Stamford Bridge is full of committed holders, not casual visitors. At the other end, Spurs (44.3%) and West Ham (41.7%) fill over half their stadium with non-STH fans each week, suggesting either a less captive local market or more flexible inventory management. Sheffield United's 2.857% Instagram→STH conversion rate dwarfs Liverpool's 0.070% — a 41x difference that reflects supply constraint, not weak demand. Arsenal and Newcastle both have documented waitlists in the tens of thousands.

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
