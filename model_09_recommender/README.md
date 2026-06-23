# Model 09 — Player Similarity Recommender

**Discipline:** Product DS
**Method:** Cosine similarity on normalized per-90 stats, indexed with FAISS

---

## Scope Note

The original plan covered 5 seasons across 3 leagues with a live Streamlit deployment. Scoped down to one league-season (Premier League 2024-25) to limit FBref scraping time/risk, and shipped as static outputs (charts + CSVs) instead of a deployed app — the recommender logic is identical, it's just not wrapped in a web UI.

---

## The Question

> Can we build a "players like this" recommender that finds statistically similar players from their per-90 stat profile — and does it surface sensible comparisons?

---

## Business Parallel

This is item-item collaborative filtering, the same technique behind "customers like this one" in marketing personalization and "products similar to this" in e-commerce recommendations. The only thing that changes between domains is the feature vector — purchase behavior becomes per-90 stat output.

---

## Method Summary

Pull one season of Premier League player stats from FBref (standard, shooting, misc). Build an 11-feature per-90 profile per player (goals, assists, shots, shots on target, fouls, crosses, interceptions, tackles won, etc.), filtered to players with 900+ minutes. Standardize, L2-normalize, and index with FAISS (`IndexFlatIP` on normalized vectors = cosine similarity). For a set of well-known players, retrieve their top-5 nearest neighbors.

---

## Data Sources

| Source | What it provides |
|--------|-------------------|
| FBref via `soccerdata` | Premier League 2024-25 standard, shooting, and misc per-player stats |

No paid data, no API key. One scrape, cached locally.

---

## Step-by-Step Plan

### 1. Data Collection
Premier League 2024-25 season, three stat tables (standard, shooting, misc) via `soccerdata.FBref`, merged on player/team.

### 2. Feature Engineering
Filter to players with ≥10 "90s" (900+ minutes). Convert raw counting stats to per-90: goals, assists, G+A, shots, shots on target, fouls committed/drawn, offsides, crosses, interceptions, tackles won. 321 players, 11 features.

### 3. Similarity Engine
StandardScaler → L2-normalize → FAISS `IndexFlatIP` (cosine similarity via inner product on normalized vectors).

### 4. Validation
Manual sanity check on 6 well-known players spanning forward, midfield, and defensive profiles — does a center-back return other center-backs? Does a striker return other attackers?

### 5. Outputs
Spotlight comparison table, PCA scatter of the full player population colored by position, bar chart of one player's top-5 matches.

---

## Key Outputs

| File | Description |
|------|-------------|
| `outputs/spotlight_comparisons.csv` | Top-5 matches + similarity scores for 6 spotlight players |
| `outputs/player_features.csv` | Full per-90 feature matrix, 321 players |
| `outputs/player_pca_scatter.png` | 2D PCA of the player population, colored by position, spotlight players annotated |
| `outputs/similarity_top5_example.png` | Bar chart of Mohamed Salah's top-5 matches and scores |

---

## Results

321 Premier League players (2024-25, 900+ minutes) indexed on 11 per-90 features.

Validation check passed cleanly: Virgil van Dijk's top-5 matches are all center-backs (Luke Woolfenden, Ethan Pinnock, Matthijs de Ligt, Joachim Andersen, Leny Yoro), similarity scores 0.90–0.96. The model separates positions correctly without being told what a position is.

Mohamed Salah's closest match is Ollie Watkins (sim 0.953), followed by Harvey Barnes, Jarrod Bowen, Cody Gakpo, Alexander Isak — all high-output wide forwards/strikers.

Erling Haaland is the most interesting case: his nearest neighbor (Danny Welbeck, sim 0.924) is a good Premier League striker, but a clear tier below him. Nobody in this dataset actually plays like Haaland; the model just returns the closest available approximation.

Kevin De Bruyne's matches (Pedro Neto, Bukayo Saka, Dwight McNeil) skew toward wingers rather than creative midfielders. This is a real limitation, not a hidden one: the feature set is built from standard/shooting/misc stat tables, which don't include passing or progressive-carry metrics. The model is matching attacking output, not playmaking style. A passing-stats feature set would likely place De Bruyne closer to other creators.

**Key finding:** The similarity engine correctly separates positions (the van Dijk result) using only attacking/disciplinary stats, no position label fed in.

**Surprising result:** The model is more useful for "who approximates this player" than "who plays like this player" — Haaland's and De Bruyne's matches expose the difference between similar output and similar style, and that gap is exactly where this feature set is weakest.

---

## How to Run

```bash
conda activate ds_portfolio
python run_model_09.py
```

---

## Skill Mapping (for interviews)

| What you built | What interviewers call it |
|---------------|--------------------------|
| Cosine similarity on normalized per-90 stats | Item-item collaborative filtering |
| FAISS `IndexFlatIP` for nearest-neighbor search | Vector similarity search / embeddings retrieval |
| PCA projection for visual validation | Dimensionality reduction for QA |
| Honest discussion of feature-set limitations (output vs style) | Knowing what your model can't see |

---
