"""
Model 12 - Fan Engagement Model
What drives Wikipedia pageviews for Premier League clubs:
winning, or just having a match?
"""

import os
import time
import warnings
import requests
import numpy as np
import pandas as pd
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import xgboost as xgb
import shap

warnings.filterwarnings("ignore")

# ─── Config ───────────────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLUBS = {
    "Arsenal":          "Arsenal_F.C.",
    "Chelsea":          "Chelsea_F.C.",
    "Liverpool":        "Liverpool_F.C.",
    "Manchester City":  "Manchester_City_F.C.",
    "Tottenham":        "Tottenham_Hotspur_F.C.",
}

SEASONS = ["1920", "2021", "2122", "2223"]

WIKI_START = "20190801"
WIKI_END   = "20230630"

HEADERS = {"User-Agent": "portfolio-project/1.0 (marcusbulls123@gmail.com)"}

# ─── STEP 1: Match data from football-data.co.uk ─────────────────────────────
def fetch_match_data(seasons, clubs):
    frames = []
    for season in seasons:
        url = f"https://www.football-data.co.uk/mmz4281/{season}/E0.csv"
        print(f"  Fetching match data: {url}")
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            r.raise_for_status()
            df = pd.read_csv(StringIO(r.text), encoding="latin-1")
            print(f"    Columns: {list(df.columns[:10])}")
            df["season"] = season
            frames.append(df)
        except Exception as e:
            print(f"    ERROR fetching {url}: {e}")
    if not frames:
        raise RuntimeError("No match data fetched.")
    raw = pd.concat(frames, ignore_index=True)

    # Normalise date column (sometimes 'Date', sometimes 'date')
    date_col = next((c for c in raw.columns if c.lower() == "date"), None)
    if date_col is None:
        raise RuntimeError(f"No date column found. Columns: {list(raw.columns)}")
    raw["date"] = pd.to_datetime(raw[date_col], dayfirst=True, errors="coerce")
    raw = raw.dropna(subset=["date"])

    # Goal columns
    home_goals_col = next((c for c in raw.columns if c in ("FTHG", "HG")), None)
    away_goals_col = next((c for c in raw.columns if c in ("FTAG", "AG")), None)
    result_col     = next((c for c in raw.columns if c in ("FTR", "Res")),  None)
    if not all([home_goals_col, away_goals_col, result_col]):
        raise RuntimeError(
            f"Could not find goal/result columns. Columns: {list(raw.columns)}"
        )

    rows = []
    club_set = set(clubs.keys())
    for _, r in raw.iterrows():
        home = r.get("HomeTeam", "")
        away = r.get("AwayTeam", "")
        fthg = r[home_goals_col]
        ftag = r[away_goals_col]
        ftr  = r[result_col]
        date = r["date"]
        # Map fuzzy names
        home_club = _match_club(home, club_set)
        away_club = _match_club(away, club_set)

        if home_club:
            rows.append({
                "club": home_club, "date": date,
                "home_match": 1,
                "goals_scored": fthg, "goals_conceded": ftag,
                "result": ftr,  # H=win, A=loss, D=draw from home perspective
            })
        if away_club:
            rows.append({
                "club": away_club, "date": date,
                "home_match": 0,
                "goals_scored": ftag, "goals_conceded": fthg,
                "result": "H" if ftr == "A" else ("A" if ftr == "H" else "D"),
            })

    matches = pd.DataFrame(rows)
    matches["win"]  = (matches["result"] == "H").astype(int)
    matches["draw"] = (matches["result"] == "D").astype(int)
    matches["loss"] = (matches["result"] == "A").astype(int)
    matches["margin"] = matches["goals_scored"] - matches["goals_conceded"]
    matches["match_day"] = 1
    matches["date"] = pd.to_datetime(matches["date"])
    return matches


def _match_club(name, club_set):
    """Fuzzy match a team name string to our club set."""
    name = str(name).strip()
    for club in club_set:
        # Simple substring check both ways
        if club.lower() in name.lower() or name.lower() in club.lower():
            return club
        # Special cases
        if club == "Manchester City" and "man city" in name.lower():
            return club
        if club == "Tottenham" and ("spurs" in name.lower() or "hotspur" in name.lower()):
            return club
    return None


# ─── STEP 2: Wikipedia pageviews ─────────────────────────────────────────────
def fetch_wiki_views(clubs, start, end):
    frames = []
    for club, article in clubs.items():
        url = (
            f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
            f"en.wikipedia/all-access/all-agents/{article}/daily/{start}/{end}"
        )
        print(f"  Fetching Wikipedia views: {club}")
        try:
            r = requests.get(url, headers=HEADERS, timeout=60)
            r.raise_for_status()
            items = r.json().get("items", [])
            rows = []
            for item in items:
                ts = item["timestamp"]  # "YYYYMMDD00"
                date = pd.to_datetime(ts[:8], format="%Y%m%d")
                rows.append({"club": club, "date": date, "views": item["views"]})
            df = pd.DataFrame(rows)
            frames.append(df)
            print(f"    {club}: {len(df)} days, avg views {df['views'].mean():.0f}")
        except Exception as e:
            print(f"    ERROR for {club}: {e}")
        time.sleep(0.3)  # be polite to the API

    if not frames:
        raise RuntimeError("No Wikipedia data fetched.")
    return pd.concat(frames, ignore_index=True)


# ─── STEP 3: Merge & feature engineering ─────────────────────────────────────
def build_panel(views_df, matches_df):
    """
    Create one row per (club, date). Merge match features onto view data.
    """
    views_df = views_df.copy()
    views_df["date"] = pd.to_datetime(views_df["date"])

    match_cols = [
        "club", "date", "match_day", "home_match",
        "goals_scored", "goals_conceded", "margin",
        "win", "draw", "loss",
    ]
    match_agg = (
        matches_df[match_cols]
        .copy()
        .assign(date=lambda d: pd.to_datetime(d["date"]))
    )
    # If a club played two matches on the same day (very rare) just keep first
    match_agg = match_agg.drop_duplicates(subset=["club", "date"])

    panel = views_df.merge(match_agg, on=["club", "date"], how="left")
    # Fill non-match days
    fill_zero = ["match_day", "home_match", "goals_scored", "goals_conceded",
                 "margin", "win", "draw", "loss"]
    panel[fill_zero] = panel[fill_zero].fillna(0)

    # Day-after features (shift per club)
    panel = panel.sort_values(["club", "date"]).reset_index(drop=True)
    for feat in ["match_day", "win", "draw", "loss", "margin", "home_match"]:
        panel[f"prev_{feat}"] = panel.groupby("club")[feat].shift(1).fillna(0)

    # Rename day-after features for clarity
    panel.rename(columns={
        "prev_match_day": "day_after_match",
        "prev_win":       "day_after_win",
        "prev_draw":      "day_after_draw",
        "prev_loss":      "day_after_loss",
        "prev_margin":    "prev_margin",
        "prev_home_match":"day_after_home",
    }, inplace=True)

    # Rolling 5-match form (points: win=3, draw=1, loss=0)
    panel["match_points"] = panel["win"] * 3 + panel["draw"] * 1
    panel["cum_form"] = (
        panel.groupby("club")["match_points"]
        .transform(lambda x: x.rolling(5, min_periods=1).sum())
    )

    # Log pageviews
    panel["log_views"] = np.log1p(panel["views"])

    # Day of week seasonality
    panel["dow"] = panel["date"].dt.dayofweek
    panel["month"] = panel["date"].dt.month

    # Club as category code
    panel["club_code"] = pd.Categorical(panel["club"]).codes

    return panel


# ─── STEP 4: Analysis ─────────────────────────────────────────────────────────
def run_analysis(panel):
    """
    Compute summary stats and fit XGBoost + SHAP.
    Returns (summary_df, shap_df, metrics_dict).
    """
    # ── Engagement summary by event type ──────────────────────────────────────
    conditions = {
        "non_match_day":  (panel["match_day"] == 0) & (panel["day_after_match"] == 0),
        "match_day":      panel["match_day"] == 1,
        "day_after_match":panel["day_after_match"] == 1,
        "day_after_win":  panel["day_after_win"] == 1,
        "day_after_draw": panel["day_after_draw"] == 1,
        "day_after_loss": panel["day_after_loss"] == 1,
    }
    rows = []
    for label, mask in conditions.items():
        subset = panel.loc[mask, "views"]
        rows.append({
            "event_type":   label,
            "mean_views":   round(subset.mean(), 1),
            "median_views": round(subset.median(), 1),
            "n_days":       int(mask.sum()),
        })
    summary_df = pd.DataFrame(rows)
    print("\n-- Engagement Summary --")
    print(summary_df.to_string(index=False))

    # ── Compute lift metrics ───────────────────────────────────────────────────
    base = summary_df.loc[summary_df["event_type"] == "non_match_day", "mean_views"].values[0]
    match_mean = summary_df.loc[summary_df["event_type"] == "match_day", "mean_views"].values[0]
    win_mean   = summary_df.loc[summary_df["event_type"] == "day_after_win", "mean_views"].values[0]
    loss_mean  = summary_df.loc[summary_df["event_type"] == "day_after_loss", "mean_views"].values[0]

    match_lift = (match_mean - base) / base * 100
    win_lift   = (win_mean   - base) / base * 100
    loss_lift  = (loss_mean  - base) / base * 100

    print(f"\nMatch-day lift vs baseline:    {match_lift:.1f}%")
    print(f"Day-after-win lift vs baseline: {win_lift:.1f}%")
    print(f"Day-after-loss lift vs baseline:{loss_lift:.1f}%")

    # Big win vs narrow win
    big_win  = panel.loc[(panel["day_after_win"] == 1) & (panel["prev_margin"] >= 3), "views"]
    narr_win = panel.loc[(panel["day_after_win"] == 1) & (panel["prev_margin"] == 1), "views"]
    print(f"\nDay-after big win (margin≥3): mean views {big_win.mean():.0f}")
    print(f"Day-after narrow win (margin=1): mean views {narr_win.mean():.0f}")
    big_vs_narrow_pct = (big_win.mean() - narr_win.mean()) / narr_win.mean() * 100 if narr_win.mean() > 0 else 0

    # ── XGBoost regression ────────────────────────────────────────────────────
    features = [
        "match_day", "day_after_match",
        "win", "draw", "loss", "margin",
        "day_after_win", "day_after_draw", "day_after_loss",
        "home_match", "day_after_home",
        "cum_form", "prev_margin",
        "dow", "month", "club_code",
    ]
    target = "log_views"

    model_data = panel[features + [target]].dropna()
    X = model_data[features]
    y = model_data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"\nXGBoost R²: {r2:.4f}")

    # ── SHAP ──────────────────────────────────────────────────────────────────
    print("  Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(X_test)
    mean_abs   = np.abs(shap_vals).mean(axis=0)
    shap_df = pd.DataFrame({
        "feature":    features,
        "importance": mean_abs,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    top_feature = shap_df.iloc[0]["feature"]
    print(f"  Top SHAP feature: {top_feature} ({shap_df.iloc[0]['importance']:.4f})")
    print(shap_df.head(8).to_string(index=False))

    metrics = {
        "n_clubs":                  len(panel["club"].unique()),
        "date_range":               f"{panel['date'].min().date()} to {panel['date'].max().date()}",
        "n_obs":                    len(panel),
        "avg_match_day_lift_pct":   round(match_lift, 1),
        "avg_win_day_lift_pct":     round(win_lift, 1),
        "avg_loss_day_lift_pct":    round(loss_lift, 1),
        "big_win_vs_narrow_pct":    round(big_vs_narrow_pct, 1),
        "top_feature":              top_feature,
        "xgb_r2":                   round(r2, 4),
        "base_views":               round(base, 0),
        "match_day_mean_views":     round(match_mean, 0),
        "win_day_mean_views":       round(win_mean, 0),
        "loss_day_mean_views":      round(loss_mean, 0),
        "big_win_mean_views":       round(big_win.mean(), 0),
        "narrow_win_mean_views":    round(narr_win.mean(), 0),
    }

    return summary_df, shap_df, metrics


# ─── STEP 5: Club-level breakdown ────────────────────────────────────────────
def club_breakdown(panel):
    print("\n-- Club-level: match-day lift vs win lift --")
    rows = []
    base_all = panel.loc[
        (panel["match_day"] == 0) & (panel["day_after_match"] == 0), "views"
    ].mean()

    for club in sorted(panel["club"].unique()):
        sub = panel[panel["club"] == club]
        base = sub.loc[
            (sub["match_day"] == 0) & (sub["day_after_match"] == 0), "views"
        ].mean()
        match_mean = sub.loc[sub["match_day"] == 1, "views"].mean()
        win_mean   = sub.loc[sub["day_after_win"] == 1, "views"].mean()
        loss_mean  = sub.loc[sub["day_after_loss"] == 1, "views"].mean()

        match_lift = (match_mean - base) / base * 100 if base > 0 else 0
        win_lift   = (win_mean   - base) / base * 100 if base > 0 else 0
        loss_lift  = (loss_mean  - base) / base * 100 if base > 0 else 0
        rows.append({
            "club":             club,
            "base_views":       round(base, 0),
            "match_day_lift_%": round(match_lift, 1),
            "win_lift_%":       round(win_lift, 1),
            "loss_lift_%":      round(loss_lift, 1),
        })
    club_df = pd.DataFrame(rows)
    print(club_df.to_string(index=False))
    return club_df


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Model 12 - Fan Engagement: Winning vs Drama")
    print("=" * 60)

    # Step 1
    print("\n[1/5] Fetching match data...")
    matches = fetch_match_data(SEASONS, CLUBS)
    print(f"  Total match appearances: {len(matches)}")
    print(f"  Clubs found: {sorted(matches['club'].unique())}")

    # Step 2
    print("\n[2/5] Fetching Wikipedia pageviews...")
    views = fetch_wiki_views(CLUBS, WIKI_START, WIKI_END)

    # Step 3
    print("\n[3/5] Building panel dataset...")
    panel = build_panel(views, matches)
    print(f"  Panel shape: {panel.shape}")
    print(f"  Date range: {panel['date'].min().date()} — {panel['date'].max().date()}")
    print(f"  Match days: {panel['match_day'].sum():.0f}")

    # Step 4
    print("\n[4/5] Running analysis...")
    summary_df, shap_df, metrics = run_analysis(panel)

    # Step 5
    print("\n[5/5] Club-level breakdown...")
    club_df = club_breakdown(panel)

    # ── Save outputs ──────────────────────────────────────────────────────────
    print("\n-- Saving outputs --")
    panel_out = panel[[
        "club", "date", "views", "log_views",
        "match_day", "home_match", "goals_scored", "goals_conceded",
        "margin", "win", "draw", "loss",
        "day_after_match", "day_after_win", "day_after_draw", "day_after_loss",
        "cum_form", "dow", "month",
    ]].copy()
    panel_out.to_csv(os.path.join(OUTPUT_DIR, "pageviews_merged.csv"), index=False)
    print("  Saved pageviews_merged.csv")

    summary_df.to_csv(os.path.join(OUTPUT_DIR, "engagement_summary.csv"), index=False)
    print("  Saved engagement_summary.csv")

    shap_df.to_csv(os.path.join(OUTPUT_DIR, "shap_importance.csv"), index=False)
    print("  Saved shap_importance.csv")

    club_df.to_csv(os.path.join(OUTPUT_DIR, "club_breakdown.csv"), index=False)
    print("  Saved club_breakdown.csv")

    metrics_path = os.path.join(OUTPUT_DIR, "key_metrics.txt")
    with open(metrics_path, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
    print("  Saved key_metrics.txt")

    print("\n-- KEY METRICS --")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print("\nDone.")
    return metrics


if __name__ == "__main__":
    main()
