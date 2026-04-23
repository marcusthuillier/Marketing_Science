# Model 20 — Network Effects Modeling

**Discipline:** Product DS
**Method:** Passing network graph analysis (networkx) + NBA on/off lineup analysis + network multiplier score

---

## The Question

> Do some players make everyone around them better — and can we quantify the network effect of adding a specific player to a lineup?

---

## Business Parallel

Each model maps directly to a core technique used in tech company data science roles. The sports domain makes the work memorable and shareable — the methods are identical to what you'd use on user or customer data.

---

## Method Summary

Passing network graph analysis (networkx) + NBA on/off lineup analysis + network multiplier score

---

## Step-by-Step Plan

### 1. Data Collection
StatsBomb free match data via statsbombpy. Extract all passes: passer, recipient, timestamp. Build adjacency matrix. Also: NBA lineup data via nba_api for on/off analysis.

### 2. Passing Network Construction
Per match: directed weighted graph (networkx). Nodes = players, edges = passes, weight = pass count. Normalize by time on pitch.

### 3. Graph Metrics
Degree centrality. Betweenness centrality (sits on path between others). PageRank. Clustering coefficient.

### 4. On/Off Analysis (NBA)
Team offensive rating WITH vs WITHOUT each player. Network effect = change in teammates' stats when player is on court.

### 5. Network Effect Score
Combine: passing centrality + teammate performance lift. Normalize for team quality. Rank all players.

### 6. Key Finding
Players with highest network effect vs highest individual stats. Find the 'hidden connector': low individual stats, high network effect.


---

## Key Outputs

| File | Description |
|------|-------------|
| `outputs/passing_networks/` | — |
| `outputs/network_effect_scores.csv` | — |
| `outputs/network_graph.html` | — |
| `outputs/individual_vs_network_scatter.png` | — |

---

## Results

<!-- Fill in after running the notebook -->

**Key finding:** _

**Surprising result:** _

---

## How to Run

```bash
conda activate ds_portfolio
jupyter notebook model_20_network_effects.ipynb
```

---

## Skill Mapping (for interviews)

| What you built | What interviewers call it |
|---------------|--------------------------|
| Directed weighted passing network graph per match | Graph construction and analysis |
| Betweenness centrality + PageRank on passing networks | Network centrality metrics |
| NBA on/off: teammate stats with/without player | On/off analysis / network effect quantification |
| Network effect score: combined centrality + teammate lift | Network multiplier metric design |

---
