# Model 19 — Player Search & Ranking Model

**Discipline:** Product DS
**Method:** LightGBM LambdaRank with Ballon d'Or votes as relevance labels + NDCG evaluation

---

## The Question

> Can we build a learning-to-rank model that orders players by true impact — going beyond box scores — and does it agree with expert consensus?

---

## Business Parallel

Each model maps directly to a core technique used in tech company data science roles. The sports domain makes the work memorable and shareable — the methods are identical to what you'd use on user or customer data.

---

## Method Summary

LightGBM LambdaRank with Ballon d'Or votes as relevance labels + NDCG evaluation

---

## Step-by-Step Plan

### 1. Data Collection
5 seasons FBref advanced stats. Ranking labels: Ballon d'Or top-30 votes, PFA award shortlists. Convert to relevance scores: winner=5, top5=4, top10=3, top30=2, nominated=1, none=0.

### 2. Ranking Label Creation
Group by season (each season = one ranking query). Season 2015-2021 = train, 2022-2023 = test.

### 3. Feature Engineering
Per-90: xG, xA, progressive carries, pressures, tackles. Positional normalization. Minutes played. Team xG share.

### 4. Learning-to-Rank Model
LightGBM objective='lambdarank'. Query group = season. Eval: NDCG@10, NDCG@30. Train/test split by season.

### 5. Ranking Evaluation
NDCG score. Kendall's tau vs expert consensus. Where does model most disagree with experts?

### 6. Model Rankings
Apply to current season. Top-50 model rankings. Who is overrated? Who is underrated?


---

## Key Outputs

| File | Description |
|------|-------------|
| `outputs/ranking_model.pkl` | — |
| `outputs/player_rankings.csv` | — |
| `outputs/ranking_comparison.html` | — |

---

## Results

<!-- Fill in after running the notebook -->

**Key finding:** _

**Surprising result:** _

---

## How to Run

```bash
conda activate ds_portfolio
jupyter notebook model_19_search_ranking.ipynb
```

---

## Skill Mapping (for interviews)

| What you built | What interviewers call it |
|---------------|--------------------------|
| LightGBM LambdaRank objective for learning-to-rank | Learning-to-rank (LTR) modeling |
| NDCG@10 and NDCG@30 as evaluation metrics | Ranking evaluation metrics |
| Ballon d'Or votes converted to graded relevance labels | Label engineering from noisy human judgments |
| Kendall's tau rank correlation vs expert consensus | Rank correlation analysis |

---

## LinkedIn Post

See [linkedin_post.md](linkedin_post.md) for a ready-to-publish draft.
