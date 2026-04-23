# Model 13 — Goal Attribution Modeling

**Discipline:** Marketing DS
**Method:** Last-touch vs Linear vs Shapley value attribution on StatsBomb event data

---

## The Question

> When a goal is scored, who really deserves credit — and how radically does the answer change depending on which attribution model you use?

---

## Business Parallel

Each model maps directly to a core technique used in tech company data science roles. The sports domain makes the work memorable and shareable — the methods are identical to what you'd use on user or customer data.

---

## Method Summary

Last-touch vs Linear vs Shapley value attribution on StatsBomb event data

---

## Step-by-Step Plan

### 1. Data Collection
StatsBomb free open data via statsbombpy. Focus: passes, carries, shots, goals. Reconstruct possession sequences leading to each goal.

### 2. Possession Chain Reconstruction
For each goal: trace back 10 events. Identify all players who touched the ball. Build: [player_a: 2 passes, player_b: 1 carry, player_c: assist, player_d: goal].

### 3. Last-Touch Attribution
100% credit to goal scorer. Baseline — represents last-click.

### 4. Linear Attribution
Equal credit across all players in the possession chain, weighted by touches.

### 5. Shapley Value Attribution
Shapley value calculation per goal sequence. Treat each player's inclusion/exclusion as a coalition.

### 6. Comparison
For 100 goals: how does credit distribution change? Which player is most undervalued by last-touch vs Shapley?


---

## Key Outputs

| File | Description |
|------|-------------|
| `outputs/attribution_comparison.csv` | — |
| `outputs/attribution_plots.html` | — |
| `outputs/shapley_implementation.py` | — |

---

## Results

<!-- Fill in after running the notebook -->

**Key finding:** _

**Surprising result:** _

---

## How to Run

```bash
conda activate ds_portfolio
jupyter notebook model_13_attribution_model.ipynb
```

---

## Skill Mapping (for interviews)

| What you built | What interviewers call it |
|---------------|--------------------------|
| Last-touch, linear, and Shapley value attribution built from scratch | Multi-touch attribution modeling |
| Possession chain reconstruction from event-level data | Event data wrangling |
| Shapley value as game-theoretic fair credit allocation | Shapley / cooperative game theory |
| Found most undervalued players by last-touch vs Shapley gap | Attribution audit / bias detection |

---
