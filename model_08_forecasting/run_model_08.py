"""
Model 08 -- NBA Standings Forecasting
Mid-season win-total extrapolation vs ARIMA -- does the dumb baseline beat the time series model?
Run: python run_model_08.py
"""
import os, time, warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

OUTPUTS_DIR = "outputs/"
os.makedirs(OUTPUTS_DIR, exist_ok=True)

CACHE_GAMELOG = OUTPUTS_DIR + "team_gamelog_raw.csv"

BG, RED, BLUE = "#f5f0eb", "#9b3d36", "#4c72b0"

# Full, non-shortened 82-game seasons only.
# Explicitly excludes 2011-12 (lockout, 66 games), 2019-20 (COVID-shortened/bubble),
# and 2020-21 (72-game COVID season).
SEASONS = [
    "2014-15", "2015-16", "2016-17", "2017-18", "2018-19",
    "2021-22", "2022-23", "2023-24",
]

HALF = 41   # games 1-41 = "mid-season" train window
FULL = 82   # full season length


# ── Data loading ────────────────────────────────────────────────────────────

def _fetch_season_gamelog(season):
    from nba_api.stats.endpoints import leaguegamelog
    print(f"  Fetching {season} team game log from nba_api...", end=" ", flush=True)
    time.sleep(0.6)
    r = leaguegamelog.LeagueGameLog(season=season, season_type_all_star="Regular Season",
                                     player_or_team_abbreviation="T")
    df = r.get_data_frames()[0]
    print(f"{df['TEAM_ID'].nunique()} teams, {len(df)} team-games")
    return df


def load_data():
    if os.path.exists(CACHE_GAMELOG):
        print(f"Loading cached game logs from {CACHE_GAMELOG}...")
        raw = pd.read_csv(CACHE_GAMELOG, parse_dates=["GAME_DATE"])
        print(f"  {len(raw):,} team-games | {raw['SEASON'].nunique()} seasons")
        return raw

    print("Fetching team game logs from nba_api (no cache found)...")
    frames = []
    for season in SEASONS:
        df = _fetch_season_gamelog(season)
        df["SEASON"] = season
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    raw["GAME_DATE"] = pd.to_datetime(raw["GAME_DATE"])
    raw.to_csv(CACHE_GAMELOG, index=False)
    print(f"  Cached {len(raw):,} team-games to {CACHE_GAMELOG}")
    return raw


# ── Build cumulative win series per team-season ─────────────────────────────

PYTH_EXP = 13.91  # standard NBA Pythagorean exponent (Hollinger/Morey)


def build_team_season_series(raw):
    print("\nBuilding cumulative win series per team-season...")
    df = raw.copy()
    df["WIN"] = (df["WL"] == "W").astype(int)
    df["PTS_AGAINST"] = df["PTS"] - df["PLUS_MINUS"]
    df = df.sort_values(["SEASON", "TEAM_ID", "GAME_DATE"])
    df["GAME_NUM"] = df.groupby(["SEASON", "TEAM_ID"]).cumcount() + 1
    df["CUM_WINS"] = df.groupby(["SEASON", "TEAM_ID"])["WIN"].cumsum()
    df["CUM_PF"] = df.groupby(["SEASON", "TEAM_ID"])["PTS"].cumsum()
    df["CUM_PA"] = df.groupby(["SEASON", "TEAM_ID"])["PTS_AGAINST"].cumsum()

    # Keep only team-seasons with a full 82 games (drops mid-season-traded
    # franchise oddities / any data gaps so the 41/82 split is clean)
    counts = df.groupby(["SEASON", "TEAM_ID"])["GAME_NUM"].max()
    full_season_keys = counts[counts == FULL].index
    df = df.set_index(["SEASON", "TEAM_ID"])
    df = df.loc[df.index.isin(full_season_keys)].reset_index()

    n_team_seasons = df.groupby(["SEASON", "TEAM_ID"]).ngroups
    print(f"  {n_team_seasons} team-seasons with a full {FULL}-game record "
          f"across {df['SEASON'].nunique()} seasons")
    return df


def pythagorean_pct(pf, pa, exponent=PYTH_EXP):
    return pf ** exponent / (pf ** exponent + pa ** exponent)


def fit_loso_regression(team_season_df):
    """
    Leave-one-season-out: for each season, fit a linear regression mapping
    first-half Pythagorean win% -> second-half actual win% using every OTHER
    season's team-seasons, then apply it to the held-out season. This is the
    standard sports-analytics regression-to-the-mean correction on top of
    Pythagorean win expectation -- avoids in-sample overfitting.
    """
    from sklearn.linear_model import LinearRegression

    rows = []
    for (season, team_id), g in team_season_df.groupby(["SEASON", "TEAM_ID"]):
        g = g.sort_values("GAME_NUM")
        half = g[g["GAME_NUM"] == HALF].iloc[0]
        second_half_wins = g[g["GAME_NUM"] > HALF]["WIN"].sum()
        pyth_pct_half = pythagorean_pct(half["CUM_PF"], half["CUM_PA"])
        rows.append({
            "SEASON": season, "TEAM_ID": team_id,
            "PYTH_PCT_HALF": pyth_pct_half,
            "SECOND_HALF_WIN_PCT": second_half_wins / (FULL - HALF),
        })
    fit_df = pd.DataFrame(rows)

    models = {}
    for season in fit_df["SEASON"].unique():
        train = fit_df[fit_df["SEASON"] != season]
        reg = LinearRegression()
        reg.fit(train[["PYTH_PCT_HALF"]].values, train["SECOND_HALF_WIN_PCT"].values)
        models[season] = reg
    return fit_df, models


def pythagorean_forecast(season, pyth_pct_half, wins_at_half, loso_models):
    reg = loso_models[season]
    pred_second_half_pct = reg.predict([[pyth_pct_half]])[0]
    pred_second_half_pct = min(max(pred_second_half_pct, 0.0), 1.0)
    return wins_at_half + pred_second_half_pct * (FULL - HALF)


# ── Forecasting models ──────────────────────────────────────────────────────

def baseline_forecast(cum_wins_at_half):
    """Linear extrapolation: pace through game 41, projected over 82."""
    return cum_wins_at_half / HALF * FULL


def arima_forecast(cum_series_through_half):
    """
    Fit ARIMA on the cumulative win series through game 41, forecast to game 82,
    return the final forecasted cumulative win total.
    Tries a small grid of orders, picks the lowest AIC.
    """
    from statsmodels.tsa.arima.model import ARIMA

    y = cum_series_through_half.astype(float).values
    candidate_orders = [(1, 1, 0), (0, 1, 1), (1, 1, 1), (2, 1, 0)]

    best_aic = np.inf
    best_fc = None
    for order in candidate_orders:
        try:
            model = ARIMA(y, order=order)
            fit = model.fit()
            if fit.aic < best_aic:
                best_aic = fit.aic
                fc = fit.forecast(steps=FULL - HALF)
                best_fc = fc[-1]
        except Exception:
            continue

    if best_fc is None:
        # Fallback: if every ARIMA order failed to converge, fall back to baseline pace
        return y[-1] / HALF * FULL
    return best_fc


def run_forecasts(team_season_df):
    print("\nFitting leave-one-season-out Pythagorean regression...")
    fit_df, loso_models = fit_loso_regression(team_season_df)
    for season, reg in loso_models.items():
        print(f"  {season}: second_half_win_pct = {reg.intercept_:.3f} + {reg.coef_[0]:.3f} * pyth_pct_half")

    print("\nRunning baseline + ARIMA + Pythagorean forecasts for each team-season...")
    results = []
    groups = list(team_season_df.groupby(["SEASON", "TEAM_ID"]))
    for i, ((season, team_id), g) in enumerate(groups):
        g = g.sort_values("GAME_NUM")
        team_abbr = g["TEAM_ABBREVIATION"].iloc[0]
        actual_final_wins = g["CUM_WINS"].iloc[-1]
        half_row = g.loc[g["GAME_NUM"] == HALF].iloc[0]
        wins_at_half = half_row["CUM_WINS"]

        cum_through_half = g.loc[g["GAME_NUM"] <= HALF, "CUM_WINS"]
        pyth_pct_half = pythagorean_pct(half_row["CUM_PF"], half_row["CUM_PA"])

        baseline_pred = baseline_forecast(wins_at_half)
        arima_pred = arima_forecast(cum_through_half)
        pyth_pred = pythagorean_forecast(season, pyth_pct_half, wins_at_half, loso_models)

        results.append({
            "SEASON": season,
            "TEAM_ID": team_id,
            "TEAM_ABBR": team_abbr,
            "WINS_AT_HALF": wins_at_half,
            "PYTH_PCT_HALF": pyth_pct_half,
            "ACTUAL_FINAL_WINS": actual_final_wins,
            "BASELINE_PRED": baseline_pred,
            "ARIMA_PRED": arima_pred,
            "PYTH_PRED": pyth_pred,
            "BASELINE_ERROR": abs(baseline_pred - actual_final_wins),
            "ARIMA_ERROR": abs(arima_pred - actual_final_wins),
            "PYTH_ERROR": abs(pyth_pred - actual_final_wins),
        })

        if (i + 1) % 20 == 0 or (i + 1) == len(groups):
            print(f"  {i+1}/{len(groups)} team-seasons forecasted")

    results_df = pd.DataFrame(results)
    results_df["ARIMA_BEATS_BASELINE"] = results_df["ARIMA_ERROR"] < results_df["BASELINE_ERROR"]
    results_df["PYTH_BEATS_BASELINE"] = results_df["PYTH_ERROR"] < results_df["BASELINE_ERROR"]
    return results_df


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(results_df):
    print("\nEvaluating forecast accuracy...")
    n = len(results_df)

    rmse_baseline = np.sqrt((results_df["BASELINE_ERROR"] ** 2).mean())
    rmse_arima = np.sqrt((results_df["ARIMA_ERROR"] ** 2).mean())
    rmse_pyth = np.sqrt((results_df["PYTH_ERROR"] ** 2).mean())
    mae_baseline = results_df["BASELINE_ERROR"].mean()
    mae_arima = results_df["ARIMA_ERROR"].mean()
    mae_pyth = results_df["PYTH_ERROR"].mean()
    pct_arima_better = results_df["ARIMA_BEATS_BASELINE"].mean()
    pct_pyth_better = results_df["PYTH_BEATS_BASELINE"].mean()

    print(f"  n = {n} team-seasons")
    print(f"  Baseline    RMSE = {rmse_baseline:.3f} wins | MAE = {mae_baseline:.3f} wins")
    print(f"  ARIMA       RMSE = {rmse_arima:.3f} wins | MAE = {mae_arima:.3f} wins")
    print(f"  Pythagorean RMSE = {rmse_pyth:.3f} wins | MAE = {mae_pyth:.3f} wins")
    print(f"  ARIMA beats baseline in {pct_arima_better:.1%} of team-seasons")
    print(f"  Pythagorean beats baseline in {pct_pyth_better:.1%} of team-seasons")

    summary = {
        "n_team_seasons": n,
        "rmse_baseline": rmse_baseline,
        "rmse_arima": rmse_arima,
        "rmse_pyth": rmse_pyth,
        "mae_baseline": mae_baseline,
        "mae_arima": mae_arima,
        "mae_pyth": mae_pyth,
        "pct_arima_better": pct_arima_better,
        "pct_pyth_better": pct_pyth_better,
    }
    return summary


def find_notable_examples(results_df):
    print("\nFinding notable team-season examples...")
    df = results_df.copy()
    df["ARIMA_ADVANTAGE"] = df["BASELINE_ERROR"] - df["ARIMA_ERROR"]  # positive = ARIMA better
    df["PYTH_ADVANTAGE"] = df["BASELINE_ERROR"] - df["PYTH_ERROR"]    # positive = Pythagorean better

    best_for_arima = df.sort_values("ARIMA_ADVANTAGE", ascending=False).head(3)
    worst_for_arima = df.sort_values("ARIMA_ADVANTAGE", ascending=True).head(3)
    best_for_pyth = df.sort_values("PYTH_ADVANTAGE", ascending=False).head(3)
    worst_for_pyth = df.sort_values("PYTH_ADVANTAGE", ascending=True).head(3)

    print("  Where ARIMA beat the baseline by the widest margin:")
    for _, r in best_for_arima.iterrows():
        print(f"    {r['SEASON']} {r['TEAM_ABBR']}: actual={r['ACTUAL_FINAL_WINS']:.0f}  "
              f"baseline={r['BASELINE_PRED']:.1f} (err {r['BASELINE_ERROR']:.1f})  "
              f"arima={r['ARIMA_PRED']:.1f} (err {r['ARIMA_ERROR']:.1f})")

    print("  Where ARIMA lost to the baseline by the widest margin:")
    for _, r in worst_for_arima.iterrows():
        print(f"    {r['SEASON']} {r['TEAM_ABBR']}: actual={r['ACTUAL_FINAL_WINS']:.0f}  "
              f"baseline={r['BASELINE_PRED']:.1f} (err {r['BASELINE_ERROR']:.1f})  "
              f"arima={r['ARIMA_PRED']:.1f} (err {r['ARIMA_ERROR']:.1f})")

    print("  Where Pythagorean beat the baseline by the widest margin:")
    for _, r in best_for_pyth.iterrows():
        print(f"    {r['SEASON']} {r['TEAM_ABBR']}: actual={r['ACTUAL_FINAL_WINS']:.0f}  "
              f"baseline={r['BASELINE_PRED']:.1f} (err {r['BASELINE_ERROR']:.1f})  "
              f"pyth={r['PYTH_PRED']:.1f} (err {r['PYTH_ERROR']:.1f})")

    print("  Where Pythagorean lost to the baseline by the widest margin:")
    for _, r in worst_for_pyth.iterrows():
        print(f"    {r['SEASON']} {r['TEAM_ABBR']}: actual={r['ACTUAL_FINAL_WINS']:.0f}  "
              f"baseline={r['BASELINE_PRED']:.1f} (err {r['BASELINE_ERROR']:.1f})  "
              f"pyth={r['PYTH_PRED']:.1f} (err {r['PYTH_ERROR']:.1f})")

    return best_for_arima, worst_for_arima, best_for_pyth, worst_for_pyth


# ── Visualizations ───────────────────────────────────────────────────────────

def generate_plots(team_season_df, results_df, best_for_arima, worst_for_arima,
                    best_for_pyth, summary, show=False):
    matplotlib.use("Agg")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

    GREEN = "#5a965a"

    # ── Plot 1: trajectory chart for illustrative teams ─────────────────────
    # Pick the team-season where ARIMA hurt most, and where Pythagorean helped
    # most -- these are the most instructive trajectories to show.
    illustrative = pd.concat([worst_for_arima.head(1), best_for_pyth.head(1)])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor=BG)
    for ax, (_, row) in zip(axes, illustrative.iterrows()):
        ax.set_facecolor(BG)
        season, team_id, team_abbr = row["SEASON"], row["TEAM_ID"], row["TEAM_ABBR"]
        g = team_season_df[(team_season_df["SEASON"] == season) &
                            (team_season_df["TEAM_ID"] == team_id)].sort_values("GAME_NUM")

        ax.plot(g["GAME_NUM"], g["CUM_WINS"], color="#333", lw=2, label="Actual cumulative wins")
        ax.axvline(HALF, color="#aaa", lw=1, ls="--")

        # Baseline trajectory: straight-line pace from game 41 to game 82
        x_proj = np.array([HALF, FULL])
        y_baseline = np.array([row["WINS_AT_HALF"], row["BASELINE_PRED"]])
        ax.plot(x_proj, y_baseline, color=BLUE, lw=2, ls="--", label=f"Baseline forecast ({row['BASELINE_PRED']:.1f})")

        ax.scatter([FULL], [row["ARIMA_PRED"]], color=RED, s=60, zorder=5,
                   label=f"ARIMA forecast ({row['ARIMA_PRED']:.1f})")
        ax.scatter([FULL], [row["PYTH_PRED"]], color=GREEN, s=60, zorder=5, marker="^",
                   label=f"Pythagorean forecast ({row['PYTH_PRED']:.1f})")
        ax.scatter([FULL], [row["ACTUAL_FINAL_WINS"]], color="#333", s=60, zorder=5, marker="D",
                   label=f"Actual final ({row['ACTUAL_FINAL_WINS']:.0f})")

        ax.set_xlabel("Game Number", fontsize=10)
        ax.set_ylabel("Cumulative Wins", fontsize=10)
        ax.set_title(f"{season} {team_abbr}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(axis="y", color="#e0dbd5", lw=0.7)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Mid-Season Forecast vs Actual Finish -- Baseline vs ARIMA vs Pythagorean",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR + "forecast_trajectories.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: forecast_trajectories.png")

    # ── Plot 2: RMSE/MAE comparison bar chart ───────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor=BG)
    ax.set_facecolor(BG)
    labels = ["RMSE", "MAE"]
    baseline_vals = [summary["rmse_baseline"], summary["mae_baseline"]]
    arima_vals = [summary["rmse_arima"], summary["mae_arima"]]
    pyth_vals = [summary["rmse_pyth"], summary["mae_pyth"]]

    x = np.arange(len(labels))
    width = 0.26
    bars1 = ax.bar(x - width, baseline_vals, width, color=BLUE, alpha=0.85, label="Baseline (pace extrapolation)")
    bars2 = ax.bar(x, arima_vals, width, color=RED, alpha=0.85, label="ARIMA")
    bars3 = ax.bar(x + width, pyth_vals, width, color=GREEN, alpha=0.85, label="Pythagorean (LOSO regression)")

    for bars in (bars1, bars2, bars3):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.05, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Error in Final Win Total (wins)", fontsize=11)
    ax.set_title(
        f"Forecast Error by Model\n"
        f"n={summary['n_team_seasons']} team-seasons  |  Pythagorean beats baseline in {summary['pct_pyth_better']:.1%} of cases",
        fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", color="#e0dbd5", lw=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR + "model_error_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: model_error_comparison.png")

    if show:
        plt.show()


# ── Print findings ──────────────────────────────────────────────────────────

def print_findings(summary, best_for_arima, worst_for_arima, best_for_pyth, worst_for_pyth):
    print(f"\n{'='*60}")
    print("  KEY FINDINGS -- NBA Standings Forecasting")
    print(f"{'='*60}")
    print(f"  Team-seasons analyzed: {summary['n_team_seasons']}")
    print(f"  Baseline    RMSE:      {summary['rmse_baseline']:.3f} wins | MAE {summary['mae_baseline']:.3f}")
    print(f"  ARIMA       RMSE:      {summary['rmse_arima']:.3f} wins | MAE {summary['mae_arima']:.3f}")
    print(f"  Pythagorean RMSE:      {summary['rmse_pyth']:.3f} wins | MAE {summary['mae_pyth']:.3f}")
    print(f"  ARIMA beats baseline:       {summary['pct_arima_better']:.1%} of team-seasons")
    print(f"  Pythagorean beats baseline: {summary['pct_pyth_better']:.1%} of team-seasons")

    rmses = {"Baseline": summary["rmse_baseline"], "ARIMA": summary["rmse_arima"], "Pythagorean": summary["rmse_pyth"]}
    winner = min(rmses, key=rmses.get)
    print(f"\n  Overall RMSE winner: {winner}")

    print("\n  Best ARIMA win:")
    r = best_for_arima.iloc[0]
    print(f"    {r['SEASON']} {r['TEAM_ABBR']}: baseline error {r['BASELINE_ERROR']:.1f} vs ARIMA error {r['ARIMA_ERROR']:.1f}")

    print("\n  Worst ARIMA miss:")
    r = worst_for_arima.iloc[0]
    print(f"    {r['SEASON']} {r['TEAM_ABBR']}: baseline error {r['BASELINE_ERROR']:.1f} vs ARIMA error {r['ARIMA_ERROR']:.1f}")

    print("\n  Best Pythagorean win:")
    r = best_for_pyth.iloc[0]
    print(f"    {r['SEASON']} {r['TEAM_ABBR']}: baseline error {r['BASELINE_ERROR']:.1f} vs Pythagorean error {r['PYTH_ERROR']:.1f}")

    print("\n  Worst Pythagorean miss:")
    r = worst_for_pyth.iloc[0]
    print(f"    {r['SEASON']} {r['TEAM_ABBR']}: baseline error {r['BASELINE_ERROR']:.1f} vs Pythagorean error {r['PYTH_ERROR']:.1f}")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    matplotlib.use("Agg")

    raw = load_data()
    team_season_df = build_team_season_series(raw)
    results_df = run_forecasts(team_season_df)
    summary = evaluate(results_df)
    best_for_arima, worst_for_arima, best_for_pyth, worst_for_pyth = find_notable_examples(results_df)
    generate_plots(team_season_df, results_df, best_for_arima, worst_for_arima, best_for_pyth, summary, show=False)
    print_findings(summary, best_for_arima, worst_for_arima, best_for_pyth, worst_for_pyth)

    results_df.to_csv(OUTPUTS_DIR + "forecast_results.csv", index=False)
    team_season_df.to_csv(OUTPUTS_DIR + "team_season_cumulative_wins.csv", index=False)
    pd.DataFrame([summary]).to_csv(OUTPUTS_DIR + "summary_metrics.csv", index=False)
    print(f"\nAll outputs saved to {OUTPUTS_DIR}")
