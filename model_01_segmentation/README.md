# Model 01 — NBA Player Segmentation

**Discipline:** Marketing DS  
**Method:** K-Means Clustering + UMAP Dimensionality Reduction  
**Data:** NBA Advanced Stats via `nba_api` — three eras: Early 2000s (2003–06), Mid 2010s (2013–16), Modern (2023–26)

---

## The Question

> Do traditional basketball positions actually capture how players play — or are there natural statistical archetypes that tell a different story?

Traditional positions (PG, SG, SF, PF, C) were invented decades ago and don't reflect modern basketball. This project applies customer segmentation — one of the most common Marketing DS tasks — to find the *real* player archetypes hiding in the data.

---

## Business Parallel

In marketing DS, customer segmentation groups users by behavior rather than demographics. The same logic applies here: instead of segmenting by assigned label (position), we segment by what players actually *do* on the court. The technique is identical — the domain makes it memorable.

---

## Method

| Step | Technique | Why |
|------|-----------|-----|
| Feature scaling | StandardScaler | Required for distance-based algorithms |
| Dimensionality reduction (viz) | UMAP | Preserves local structure better than PCA for clustering visualization |
| Dimensionality reduction (interp) | PCA | Interpretable variance decomposition |
| Clustering | K-Means | Standard, interpretable, maps directly to industry use |
| K selection | Elbow curve + Silhouette score | Two complementary signals to avoid arbitrary K |
| Explainability | Cluster centroid profiles + radar charts | Makes results stakeholder-readable |

---

## Features Used

| Feature | What It Captures |
|---------|-----------------|
| `USG_PCT` | How often this player is involved in possessions |
| `AST_PCT` | Playmaking — pass creation |
| `REB_PCT` | Rebounding presence (both ends) |
| `OREB_PCT` | Offensive rebounding |
| `DREB_PCT` | Defensive rebounding |
| `TS_PCT` | Scoring efficiency (accounts for FTs and 3s) |
| `E_TOV_PCT` | Ball security |
| `PACE` | Team context proxy |

---

## Key Outputs

| File | Description |
|------|-------------|
| `outputs/clustered_players.csv` | Every player with their archetype label and distance to centroid |
| `outputs/umap_plot.html` | Interactive scatter — hover to see player names |
| `outputs/cluster_profiles.png` | Radar charts for each archetype |
| `outputs/elbow_silhouette.png` | K selection diagnostic |

---

## Results

**Model:** Era comparison across three 3-season windows. K selected by silhouette score for each era independently.

| Era | Players | K |
|-----|---------|---|
| Early 2000s (2003–06) | 423 | 4 |
| Mid 2010s (2013–16) | 450 | 4 |
| Modern (2023–26) | 494 | 5 |

**Archetypes by era:**

| Archetype | Early 2000s | Mid 2010s | Modern |
|-----------|:-----------:|:---------:|:------:|
| 3-and-D / Role Player | 104 | 179 | 165 |
| Ball-Dominant Playmaker | 122 | 104 | 70 |
| Scoring Big / Power Forward | 96 | 87 | — |
| Glass Anchor / Traditional Center | 101 | 80 | 54 |
| Pass-First Guard | — | — | 104 |
| Rim Protector / Defensive Big | — | — | 101 |

**Most surprising finding:**  
K=5 is required for the modern era — the model independently carved out a Pass-First Guard cluster (high AST, lower USG) that didn't exist in either prior era. Chris Paul and Russell Westbrook land there. The math is picking up the decline arc without being told to look for it.

---

## How to Run

**Prerequisites:** conda environment `ds_portfolio` (see [environment.yml](environment.yml))

```bash
# Option A — Jupyter
conda activate ds_portfolio
jupyter notebook model_01_segmentation.ipynb

# Option B — VS Code
# Open model_01_segmentation.ipynb
# Select kernel: "DS Portfolio (3.11)"
# Run All
```

**After running:**
1. Check `elbow_silhouette.png` — pick K at the elbow and/or silhouette peak
2. Read the cluster centroid profiles in Section 5
3. Edit `CLUSTER_LABELS` dict in the notebook with meaningful archetype names
4. Re-run visualization cells
5. Fill in the Results table above

---

## Skill Mapping (for interviews)

| What you built | What interviewers call it |
|---------------|--------------------------|
| Grouped players by stats, not labels | Behavioral segmentation |
| UMAP for visualization | Dimensionality reduction for EDA |
| Elbow + silhouette to pick K | Model selection with multiple criteria |
| Centroid profiles + radar charts | Communicating clusters to non-technical stakeholders |
| Identified representative examples | Persona creation |

---
