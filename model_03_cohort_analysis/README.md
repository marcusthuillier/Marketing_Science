# Model 03 — NBA Draft Cohort Analysis

**Discipline:** Growth DS  
**Method:** Cohort retention analysis + heatmap (SaaS retention grid format) + ANOVA + survival curves  
**Data:** NBA stats via `nba_api` — draft classes 1990–2018, seasons 1990–91 through 2022–23

---

## The Question

> Do some NBA draft classes produce sustained value over time — and can cohort analysis reveal which draft years were genuinely exceptional vs just lucky?

---

## Business Parallel

Cohort retention tables are a growth DS staple. Every SaaS company builds them: group users by signup month, track what % are still active at month 1, 3, 6, 12. The grid tells you whether your product retains users and which acquisition cohorts perform best.

This model applies identical logic to NBA draft classes. Each draft year is a cohort. Each career year is a time period. The cell value is % of the original cohort still playing meaningful minutes. The format is indistinguishable from a SaaS retention table — only the labels change.

---

## Method

| Step | Technique | Why |
|------|-----------|-----|
| Data collection | `DraftHistory` + `leaguedashplayerstats` via nba_api | Draft year as cohort ID; season stats for activity tracking |
| Cohort construction | Group by draft year, track career year 1–15 | Direct analog to user cohort + time since signup |
| Survival matrix | % still active (≥20 GP) per (draft_year, career_year) | SaaS retention table format |
| Value matrix | Avg performance score per (draft_year, career_year) | Quality of active players, not just count |
| LTV | Total career contribution per cohort ÷ class size | Equivalent to customer LTV across acquisition cohorts |
| Statistical testing | ANOVA + Kruskal-Wallis on cohort LTVs | Are quality differences real or random variation? |
| Draft position survival | Survival curves by pick tier | Which "acquisition channel" retains best long-term? |

**Performance score:** `PTS + 1.2×REB + 1.5×AST + STL + 1.5×BLK − TOV` per game (composite value metric from basic stats, available for all seasons)

---

## Key Outputs

| File | Description |
|------|-------------|
| `outputs/cohort_matrix.csv` | Survival % matrix: rows = draft years, cols = career years |
| `outputs/cohort_heatmap.png` | Main visual — retention grid identical to SaaS cohort table |
| `outputs/value_heatmap.png` | Same grid but average performance score per cell |
| `outputs/draft_class_ltv.csv` | Per-class LTV metrics: total perf, per-player LTV, trend |
| `outputs/draft_class_ltv.png` | LTV bar chart + time trend scatter |
| `outputs/survival_by_draft_position.png` | Survival curves by draft tier (Top 5 / Lottery / Late 1st / 2nd Round) |

---

## Results

**Key finding:** 2003 is the best class in the 23-class window by every metric — retention (43% at year 10), LTV per player (177, vs 19-class avg of 130), and normalized LTV per player-year (27.1). ANOVA confirms class quality differences are statistically real (F=5.47, p<0.001).

**Draft position survival (year 10):** Top 5 = 55.7%, Lottery = 41.5%, Late 1st = 32.9%, 2nd Round = 19.9%. Top-5 picks survive at 2.8x the rate of second-rounders.

**Average cohort survival:** Year 5 = 49%, Year 10 = 30%. Half the class is gone by year 5.

**Worst class:** 2000 (LTV/player = 96, vs avg 130). Topped by Kenyon Martin; produced virtually no long-term contributors.

**Surprising result:** No significant linear trend in draft quality across 1996-2014. The draft isn't getting better or worse — it's just noisy.

---

## How to Run

```bash
conda activate ds_portfolio
jupyter notebook model_03_cohort_analysis.ipynb
```

First run pulls data from nba_api and caches to `outputs/raw_season_stats.csv`. Subsequent runs load from cache instantly.

---

## Skill Mapping (for interviews)

| What you built | What interviewers call it |
|----------------|--------------------------|
| Grouped players by draft year, tracked activity per career year | Cohort retention analysis |
| Built survival/value grid identical to SaaS retention table | Growth analytics / cohort reporting |
| Total career contribution per class ÷ class size | LTV per acquisition cohort |
| ANOVA on cohort quality differences | Statistical significance testing |
| Survival curves by draft position tier | Acquisition channel retention comparison |

---
