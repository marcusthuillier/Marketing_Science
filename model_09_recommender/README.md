# Model 09 — Player Similarity Recommender

**Discipline:** Product DS
**Method:** Cosine similarity on normalized per-90 stats + FAISS nearest neighbor search + Streamlit app

---

## The Question

> Can we build a 'players like this' recommender that finds statistically similar players across different eras and leagues?

---

## Business Parallel

Each model maps directly to a core technique used in tech company data science roles. The sports domain makes the work memorable and shareable — the methods are identical to what you'd use on user or customer data.

---

## Method Summary

Cosine similarity on normalized per-90 stats + FAISS nearest neighbor search + Streamlit app

---

## Step-by-Step Plan

### 1. Data Collection
5 seasons of FBref data for PL, La Liga, Bundesliga via soccerdata. Per-90 stats. Min 900 minutes. Most recent season per player.

### 2. Feature Engineering
Normalize all stats to per-90 (FBref already does this). StandardScaler. Option: position-adjusted comparison.

### 3. Similarity Engine
Cosine similarity matrix for <5000 players. FAISS index for efficient search at scale. Store top-10 similar players + scores per player.

### 4. Validation
Manual check: does Haaland return elite strikers? Cross-league test: does Bundesliga striker ≈ PL striker?

### 5. Streamlit App
Input: player name with autocomplete. Output: top 5 similar + similarity score + radar chart. Deploy to Streamlit Cloud (free).

### 6. Outputs
player_similarity_matrix.pkl, streamlit_app.py, radar_chart_examples.png


---

## Key Outputs

| File | Description |
|------|-------------|
| `outputs/player_similarity_matrix.pkl` | — |
| `outputs/streamlit_app.py` | — |
| `outputs/radar_chart_examples.png` | — |

---

## Results

<!-- Fill in after running the notebook -->

**Key finding:** _

**Surprising result:** _

---

## How to Run

```bash
conda activate ds_portfolio
jupyter notebook model_09_recommender.ipynb
```

---

## Skill Mapping (for interviews)

| What you built | What interviewers call it |
|---------------|--------------------------|
| Cosine similarity matrix on normalized per-90 stats | Item-item collaborative filtering |
| FAISS index for efficient nearest neighbor search | Scalable similarity search |
| Streamlit app with autocomplete + radar chart | Deploying ML as interactive product |
| Cross-league and cross-era comparison | Feature normalization for domain shift |

---

## LinkedIn Post

See [linkedin_post.md](linkedin_post.md) for a ready-to-publish draft.
