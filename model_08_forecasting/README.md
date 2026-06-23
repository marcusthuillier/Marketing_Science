# Model 08 — NBA Standings Forecasting

**Discipline:** Marketing DS + Growth DS
**Method:** Mid-season pace extrapolation vs ARIMA — model horse race with RMSE comparison

---

## Scope Note

The original plan targeted EPL standings via Football-Data.org + Facebook Prophet. Swapped for NBA standings via `nba_api` (no API key required) and dropped Prophet (build issues on Windows, heavy dependency for marginal benefit over ARIMA for this comparison). The question is identical — does a dumb baseline beat a real time series model — just on a different league and without the signup friction.

---

## The Question

> Given mid-season data, which forecasting model best predicts an NBA team's final win total — and does a simple baseline beat a real time series model?

---

## Business Parallel

This is the same horse race a growth or marketing forecasting team runs constantly: a naive run-rate extrapolation versus a "real" statistical model, before committing a quarterly forecast to leadership. The sports version removes the political pressure to pick the fancier model and just asks which one is actually more accurate.

---

## Method Summary

For every team-season, cumulative win totals are tracked game-by-game. Each team-season is split at the midpoint (game 41 of 82). Two methods then forecast the final win total using only the first half: (1) a baseline that extrapolates the team's current win rate over the remaining 41 games, and (2) an ARIMA model fit on the cumulative win series, forecast forward. Whichever has lower error against the real final win total wins.

---

## Data Sources

| Source | What it provides |
|--------|-------------------|
| `nba_api.stats.endpoints.leaguegamelog` | Team-level game logs (win/loss, date) for every regular-season game |

No paid data. No API keys required.

---

## Step-by-Step Plan

### 1. Data Collection
Team game logs for 8 full, non-shortened 82-game seasons: 2014-15 through 2018-19, and 2021-22 through 2023-24. Explicitly excludes 2019-20 (bubble), 2020-21 (72-game COVID season), and the 2011-12 lockout season.

### 2. Series Construction
Sort each team-season's games by date, compute cumulative win total after each game (game number 1–82). Drop any team-season without a clean 82-game record.

### 3. Baseline Model
`wins_at_game_41 / 41 * 82` — straight-line pace extrapolation. The benchmark everything must beat.

### 4. ARIMA Model
Fit `statsmodels` ARIMA on the cumulative win series through game 41. Small grid search over orders (1,1,0), (0,1,1), (1,1,1), (2,1,0), pick by AIC, forecast to game 82.

### 5. Comparison
RMSE and MAE across all 240 team-seasons. % of team-seasons where ARIMA beats the baseline. Notable best/worst examples for each model.

---

## Key Outputs

| File | Description |
|------|-------------|
| `outputs/forecast_trajectories.png` | Actual vs baseline vs ARIMA trajectory for the team-seasons where ARIMA helped most and hurt most |
| `outputs/model_error_comparison.png` | RMSE/MAE bar comparison, baseline vs ARIMA |
| `outputs/forecast_results.csv` | Per-team-season predictions and errors for both models |
| `outputs/team_season_cumulative_wins.csv` | Full game-by-game cumulative win series for every team-season |
| `outputs/summary_metrics.csv` | Aggregate RMSE/MAE/win-rate summary |

---

## Results

240 team-seasons across 8 full 82-game seasons (2014-15 to 2023-24, excluding shortened seasons).

Baseline (pace extrapolation): RMSE 4.97 wins, MAE 3.98 wins.
ARIMA: RMSE 8.29 wins, MAE 6.01 wins.

The baseline beats ARIMA outright — ARIMA only had lower error in 35.8% of team-seasons. ARIMA's worst miss: 2017-18 Cleveland, predicted ~26 final wins against an actual 50 (baseline was off by only 2 wins on the same team). Its best win: 2021-22 Golden State, where ARIMA caught a downward momentum shift the baseline missed (ARIMA error 3.3 vs baseline error 7.0). Many ARIMA fits also failed to fully converge on this series shape — a cumulative, monotonically non-decreasing count isn't a great fit for ARIMA's assumptions, and that's part of why it misfires so badly in the worst cases.

**Key finding:** The dumb baseline wins on every aggregate metric. ARIMA isn't just slightly worse, it's catastrophically wrong in a meaningful minority of cases because the underlying series (cumulative wins) violates ARIMA's stationarity assumptions.

**Surprising result:** ARIMA's failures aren't random noise, they're concentrated in team-seasons with a real mid-season inflection (a trade, an injury return, a coaching change) — exactly the situations a "smarter" model is supposed to handle better. It doesn't.

---

## How to Run

```bash
conda activate ds_portfolio
python run_model_08.py
```

---

## Skill Mapping (for interviews)

| What you built | What interviewers call it |
|---------------|--------------------------|
| Baseline vs ARIMA horse race with proper train/test split | Forecasting model selection |
| ARIMA with small grid order search by AIC | Time series modeling |
| RMSE/MAE comparison across 240 evaluation sets | Forecast evaluation at scale |
| Diagnosing why a "smarter" model underperforms | Model selection judgment, not just model fitting |

---
