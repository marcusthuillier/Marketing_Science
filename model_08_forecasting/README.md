# Model 08 — NBA Standings Forecasting

**Discipline:** Marketing DS + Growth DS
**Method:** Baseline vs ARIMA vs Pythagorean win expectation (leave-one-season-out regression)

---

## Scope Note

The original plan targeted EPL standings via Football-Data.org + Facebook Prophet. Swapped for NBA standings via `nba_api` (no API key required) and dropped Prophet (build issues on Windows). After an initial pass found ARIMA losing badly to a dumb baseline, added a third, genuinely competitive model: Pythagorean win expectation, the standard sports-analytics technique (Bill James → Daryl Morey/John Hollinger for basketball) for projecting team performance from scoring margin instead of noisy win-loss record.

---

## The Question

> Given mid-season data, which forecasting model best predicts an NBA team's final win total — and can a model actually beat a simple baseline, or does sophistication just add error?

---

## Business Parallel

This is the same horse race a growth or marketing forecasting team runs constantly: a naive run-rate extrapolation versus a "real" statistical model, before committing a quarterly forecast to leadership. The twist here is that the obvious "smarter" model (ARIMA) loses badly, while the model that actually wins isn't more complex, it's just built on a better underlying signal (scoring margin instead of win-loss record) with proper regression-to-the-mean.

---

## Method Summary

For every team-season, cumulative wins and points for/against are tracked game-by-game. Each team-season is split at the midpoint (game 41 of 82). Three methods forecast the final win total from the first half only:

1. **Baseline** — extrapolate the team's win rate through game 41 over the remaining 41 games.
2. **ARIMA** — fit on the cumulative win series through game 41, forecast forward.
3. **Pythagorean** — compute Pythagorean win expectation (`PF^13.91 / (PF^13.91 + PA^13.91)`) from points for/against through game 41, then map it to a second-half win rate using a linear regression fit on every *other* season (leave-one-season-out, so the model is never trained on the season it predicts).

---

## Data Sources

| Source | What it provides |
|--------|-------------------|
| `nba_api.stats.endpoints.leaguegamelog` | Team-level game logs (win/loss, points, plus/minus, date) for every regular-season game |

No paid data. No API keys required.

---

## Step-by-Step Plan

### 1. Data Collection
Team game logs for 8 full, non-shortened 82-game seasons: 2014-15 through 2018-19, and 2021-22 through 2023-24. Explicitly excludes 2019-20 (bubble), 2020-21 (72-game COVID season), and the 2011-12 lockout season.

### 2. Series Construction
Cumulative wins, points for, and points against per team-season, game-by-game. Points against derived as `PTS - PLUS_MINUS`.

### 3. Baseline Model
`wins_at_game_41 / 41 * 82` — straight-line pace extrapolation.

### 4. ARIMA Model
Fit `statsmodels` ARIMA on the cumulative win series through game 41. Small grid search over orders, pick by AIC, forecast to game 82.

### 5. Pythagorean Model
Pythagorean win% from cumulative PF/PA through game 41. Leave-one-season-out linear regression maps first-half Pythagorean win% to actual second-half win%, fit on the other 7 seasons each time. Final wins = first-half actual wins + predicted second-half win% × 41.

### 6. Comparison
RMSE and MAE across all 240 team-seasons for all three models. % of team-seasons each smarter model beats the baseline.

---

## Key Outputs

| File | Description |
|------|-------------|
| `outputs/forecast_trajectories.png` | Actual vs baseline vs ARIMA vs Pythagorean trajectory for instructive team-seasons |
| `outputs/model_error_comparison.png` | RMSE/MAE bar comparison across all three models |
| `outputs/forecast_results.csv` | Per-team-season predictions and errors for all three models |
| `outputs/team_season_cumulative_wins.csv` | Full game-by-game cumulative win/points series for every team-season |
| `outputs/summary_metrics.csv` | Aggregate RMSE/MAE/win-rate summary |

---

## Results

240 team-seasons across 8 full 82-game seasons (2014-15 to 2023-24, excluding shortened seasons).

| Model | RMSE (wins) | MAE (wins) | Beats baseline |
|-------|------------|-----------|-----------------|
| Baseline (pace extrapolation) | 4.97 | 3.98 | — |
| ARIMA | 8.29 | 6.01 | 35.8% |
| Pythagorean (LOSO regression) | **4.47** | **3.60** | **60.4%** |

ARIMA loses badly, as before — its worst miss is still 2017-18 Cleveland, predicted ~26 final wins against an actual 50. A meaningful share of ARIMA fits failed to fully converge, because a cumulative, monotonically non-decreasing win count violates ARIMA's stationarity assumptions.

Pythagorean expectation is the real upgrade. It beats the baseline outright on both RMSE and MAE, and wins in 60.4% of team-seasons — not because it's more complex, but because it's built on a better signal. Scoring margin through the first half is a more stable predictor of true team strength than win-loss record, which is noisy in close games. Its biggest win: 2014-15 New York, baseline predicted 10 final wins (error 7.0), Pythagorean predicted 15.5 (error 1.5) against an actual 17. Its worst miss: 2021-22 Indiana, where the team's underlying point differential pointed toward more wins than they actually banked, likely due to a second-half roster teardown that pace-based and points-based methods alike couldn't see coming.

**Key finding:** A genuinely better forecasting model exists for this problem, and it isn't the obvious "use a real time series model" answer. It's a domain-specific signal (scoring margin) combined with simple, properly cross-validated regression to handle the regression-to-the-mean.

**Surprising result:** ARIMA and Pythagorean are both "smarter than the baseline" in complexity, but only one of them actually understands the problem. Model sophistication without the right underlying signal is worse than no sophistication at all.

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
| Three-way model horse race with proper train/test split | Forecasting model selection |
| ARIMA with small grid order search by AIC | Time series modeling |
| Pythagorean win expectation from scoring margin | Domain-specific feature engineering |
| Leave-one-season-out regression for regression-to-the-mean | Proper cross-validation, avoiding leakage |
| Diagnosing why ARIMA underperforms and why Pythagorean doesn't | Model selection judgment, not just model fitting |

---
