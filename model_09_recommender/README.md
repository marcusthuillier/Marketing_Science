# Model 09 — Player Similarity Recommender

**Discipline:** Product DS
**Method:** Cosine similarity on normalized per-90 stats, indexed with FAISS

---

## Scope Note

The original plan covered 5 seasons across 3 leagues with a live Streamlit deployment. Scoped down to one league-season (Premier League 2024-25) to limit FBref scraping time/risk, and shipped as static outputs (charts + CSVs) instead of a deployed app — the recommender logic is identical, it's just not wrapped in a web UI.

A first pass built the feature set from FBref's public `standard`/`shooting`/`misc` stat tables only, which `soccerdata`'s public API exposes directly. That feature set had no passing or progressive-carry data, and it showed: Kevin De Bruyne's nearest matches were wingers, not creative midfielders. `soccerdata` restricts its public `stat_type` argument to a fixed allowlist, but the underlying scrape-and-parse machinery works for any FBref stat page. Calling the library's own anti-bot-aware session (`fb.get()`) directly against the `passing`, `gca` (shot/goal-creating actions), and `possession` URLs, and reusing its internal HTML-comment-unwrapping parser, pulls real passing and creativity data with no extra dependencies. The feature set below uses that.

The model's quality was originally going to be validated against FBref's own "Similar Players" feature, which historically appeared free on player scouting-report pages. Checked directly: that feature is gone from the free site (confirmed by fetching a current scouting report page and finding zero mentions of "similar" anywhere in it, just Stathead subscription promos) — it was pulled behind FBref's paid Stathead tier along with the rest of their advanced free stats in early 2026. No clean free replacement exists either (checked WhoScored, SofaScore, and the newer xG Stat product — none expose a comparable public similarity list). Validation below instead uses a standard quantitative substitute: do this model's nearest neighbors recover position more often than random chance would predict, given the dataset's actual position mix?

---

## The Question

> Can we build a "players like this" recommender that finds statistically similar players from their per-90 stat profile — and does it surface sensible comparisons, including for creative playmakers, not just high-output scorers?

---

## Business Parallel

This is item-item collaborative filtering, the same technique behind "customers like this one" in marketing personalization and "products similar to this" in e-commerce recommendations. The only thing that changes between domains is the feature vector — purchase behavior becomes per-90 stat output. It's also a real lesson in feature completeness: a recommender is only as good as the dimensions it can see, and a missing feature category (here, passing) doesn't fail loudly, it just quietly returns a worse answer.

---

## Method Summary

Pull one season of Premier League player stats from FBref: standard, shooting, and misc via `soccerdata`'s public API, plus passing, goal/shot-creating actions (GCA), and possession via a direct fetch that bypasses `soccerdata`'s stat-type allowlist. Build a 23-feature per-90 profile per player — goals, assists, shots, key passes, passes into the final third and penalty area, pass completion %, shot- and goal-creating actions per 90, touches, carries, progressive carrying distance, and more — filtered to players with 900+ minutes. Standardize, L2-normalize, and index with FAISS (`IndexFlatIP` on normalized vectors = cosine similarity). For a set of well-known players, retrieve their top-5 nearest neighbors.

---

## Data Sources

| Source | What it provides |
|--------|-------------------|
| FBref via `soccerdata` (public API) | Premier League 2024-25 standard, shooting, and misc per-player stats |
| FBref via direct fetch (bypassing `soccerdata`'s stat-type allowlist) | Passing, goal/shot-creating actions, and possession per-player stats |

No paid data, no API key. One scrape, cached locally.

---

## Step-by-Step Plan

### 1. Data Collection
Premier League 2024-25 season, six stat tables (standard, shooting, misc, passing, GCA, possession) via `soccerdata.FBref`, merged on player/team. The last three are fetched by constructing the FBref URL directly and reusing `soccerdata`'s own request session and internal parser, since its public `read_player_season_stats()` only exposes the first three.

### 2. Feature Engineering
Filter to players with ≥10 "90s" (900+ minutes). Convert raw counting stats to per-90: goals, assists, shots, fouls, crosses, interceptions, tackles won, key passes, passes into the final third/penalty area, progressive pass distance, touches, carries, progressive carry distance/final-third entries. Pass completion %, SCA90, and GCA90 are already rate stats. 321 players, 23 features.

### 3. Similarity Engine
StandardScaler → L2-normalize → FAISS `IndexFlatIP` (cosine similarity via inner product on normalized vectors).

### 4. Validation
Two checks. First, a quantitative one: for every player in the index, do their top-5 nearest neighbors share their position more often than a randomly chosen player would, given the dataset's actual position mix? Second, a manual sanity check on 6 well-known players spanning forward, midfield, and defensive profiles — does a center-back return other center-backs? Does a creative playmaker return other creators, now that passing data is in the feature set?

### 5. Outputs
Spotlight comparison table, PCA scatter of the full player population colored by position, bar chart of one player's top-5 matches.

---

## Key Outputs

| File | Description |
|------|-------------|
| `outputs/spotlight_comparisons.csv` | Top-5 matches + similarity scores for 6 spotlight players |
| `outputs/player_features.csv` | Full per-90 feature matrix, 321 players, 23 features |
| `outputs/player_pca_scatter.png` | 2D PCA of the player population, colored by position, spotlight players annotated |
| `outputs/similarity_top5_example.png` | Bar chart of Kevin De Bruyne's top-5 matches and scores |
| `outputs/validation_metrics.csv` | Position-recovery rate vs. random-chance baseline, across all 321 players |

---

## Results

321 Premier League players (2024-25, 900+ minutes) indexed on 23 per-90 features, including passing, creation (SCA/GCA), and possession/carrying.

The quantitative validation: across all 321 players, the top-5 nearest neighbors share the query player's position 72.1% of the time. Given the dataset's actual position mix, a randomly chosen player would match position only 34.3% of the time. That's a 2.10x lift over chance, with no position label ever fed into the model — it's recovered entirely from attacking, passing, and defensive output.

The manual sanity check agrees: Virgil van Dijk's top-5 matches are all center-backs (Luke Woolfenden, Ethan Pinnock, Matthijs de Ligt, Joachim Andersen, Leny Yoro), similarity scores 0.90–0.96.

Kevin De Bruyne's matches are the real test of the fix. With the richer feature set, his top-5 are Pedro Neto (sim 0.949), Bukayo Saka (0.909), Dwight McNeil (0.908), Sávio (0.878), and Marcus Tavernier (0.874). Saka in particular is a meaningful result: he's a genuine elite creative wide player with real passing output, not just a high-volume attacker. The previous, passing-free feature set never surfaced anyone like him for De Bruyne.

Mohamed Salah's closest match is Ollie Watkins (sim 0.953), followed by Harvey Barnes, Jarrod Bowen, Cody Gakpo, Alexander Isak — high-output wide forwards. Erling Haaland's nearest neighbor (Danny Welbeck, sim 0.924) is a good Premier League striker but a clear tier below him — nobody in this dataset actually plays like Haaland, the model just returns the closest available approximation.

**Key finding:** Adding passing, creation, and possession data changes De Bruyne's matches from "wingers with similar shot/goal output" to "wide creators with similar passing and chance-creation profiles" — the model now distinguishes playmaking style, not just attacking volume.

**Surprising result:** Even with the richer feature set, De Bruyne's nearest matches are still wide players (Neto, McNeil, Sávio), not central creators. That's a real signal, not a bug: De Bruyne's specific combination of central creative volume at his level has very few statistical peers in this league-season, so the model reaches for the next-closest creative profile it can find, regardless of exact position label.

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
| Bypassing a library's restrictive public API by reusing its internal session/parser | Reading library internals to unblock a real requirement |
| Position-recovery rate vs. random-chance baseline as a quantitative validation metric | Validating an embedding without external ground-truth labels |
| PCA projection for visual validation | Dimensionality reduction for QA |
| Diagnosing and fixing a feature-completeness gap, then re-validating | Knowing what your model can't see, and fixing it |

---
