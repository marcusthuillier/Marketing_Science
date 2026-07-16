"""
Model 14 - Premier League MMM (Marketing Mix Model)
Question: What combination of rest, travel, home advantage, form, and opponent
strength best explains Premier League match outcomes?
Method: Manual adstock + saturation transformations + Ridge/OLS regression
Data: football-data.co.uk, seasons 2015-16 through 2022-23
"""

import os
import io
import warnings
import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
os.makedirs(OUTPUTS_DIR, exist_ok=True)

RANDOM_STATE = 42
SEASONS = ["1516", "1617", "1718", "1819", "1920", "2021", "2122", "2223"]
BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/E0.csv"

# ---------------------------------------------------------------------------
# Travel distances (km) — hardcoded approximate values
# ---------------------------------------------------------------------------

TRAVEL_PAIRS = {
    frozenset(["Arsenal", "Chelsea"]): 12,
    frozenset(["Arsenal", "Liverpool"]): 290,
    frozenset(["Arsenal", "Man City"]): 295,
    frozenset(["Arsenal", "Man United"]): 295,
    frozenset(["Arsenal", "Tottenham"]): 10,
    frozenset(["Chelsea", "Liverpool"]): 295,
    frozenset(["Chelsea", "Man City"]): 300,
    frozenset(["Chelsea", "Man United"]): 305,
    frozenset(["Chelsea", "Tottenham"]): 11,
    frozenset(["Liverpool", "Man City"]): 55,
    frozenset(["Liverpool", "Man United"]): 55,
    frozenset(["Liverpool", "Tottenham"]): 295,
    frozenset(["Man City", "Man United"]): 10,
    frozenset(["Man City", "Tottenham"]): 295,
    frozenset(["Man United", "Tottenham"]): 300,
}

TEAM_NAME_MAP = {
    "Man City": "Man City", "Manchester City": "Man City",
    "Man United": "Man United", "Manchester United": "Man United",
    "Tottenham": "Tottenham", "Spurs": "Tottenham",
    "Arsenal": "Arsenal", "Chelsea": "Chelsea", "Liverpool": "Liverpool",
}


def normalise_team(name):
    return TEAM_NAME_MAP.get(name, name)


def get_travel_km(home_team, away_team):
    h = normalise_team(home_team)
    a = normalise_team(away_team)
    return TRAVEL_PAIRS.get(frozenset([h, a]), 150)


# ---------------------------------------------------------------------------
# 1. Fetch data
# ---------------------------------------------------------------------------

def fetch_season(season):
    url = BASE_URL.format(season=season)
    print(f"  Fetching {url} ...", end=" ")
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), encoding='latin-1')
        print(f"{len(df)} rows")
        df['season'] = season
        return df
    except Exception as e:
        print(f"FAILED: {e}")
        return None


def fetch_all_data():
    cache = os.path.join(OUTPUTS_DIR, '_raw_pl_data.csv')
    if os.path.exists(cache):
        print(f"Loading cached raw data from {cache}")
        df = pd.read_csv(cache, low_memory=False)
        print(f"  Loaded {len(df)} rows")
        return df
    frames = []
    for s in SEASONS:
        df_s = fetch_season(s)
        if df_s is not None:
            frames.append(df_s)
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(cache, index=False)
    print(f"Cached {len(combined)} rows -> {cache}")
    return combined


# ---------------------------------------------------------------------------
# 2. Parse & clean
# ---------------------------------------------------------------------------

REQUIRED_COLS = ['HomeTeam', 'AwayTeam', 'Date', 'FTHG', 'FTAG', 'FTR']


def parse_raw(df_raw):
    keep = REQUIRED_COLS + ['season']
    df = df_raw[df_raw.columns.intersection(keep)].copy()
    df = df.dropna(subset=REQUIRED_COLS)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['Date'])
    df['FTHG'] = pd.to_numeric(df['FTHG'], errors='coerce')
    df['FTAG'] = pd.to_numeric(df['FTAG'], errors='coerce')
    df = df.dropna(subset=['FTHG', 'FTAG'])
    df['FTHG'] = df['FTHG'].astype(int)
    df['FTAG'] = df['FTAG'].astype(int)
    df = df.sort_values('Date').reset_index(drop=True)
    print(f"Parsed {len(df)} valid matches across {df['season'].nunique()} seasons")
    return df


# ---------------------------------------------------------------------------
# 3. Per-team feature engineering (chronological order)
# ---------------------------------------------------------------------------

def build_team_histories(df):
    """
    Walk matches in date order, tracking per-team:
      - last match date (for rest_days)
      - rolling form (last 5 match points)
    Returns df with columns added.
    """
    last_date = {}   # team -> last match date
    form_pts = {}    # team -> list of points (last 5 kept)

    rest_home_list, rest_away_list = [], []
    form_home_list, form_away_list = [], []

    for _, row in df.iterrows():
        d = row['Date']
        h = row['HomeTeam']
        a = row['AwayTeam']
        result = row['FTR']

        # Rest days BEFORE this match
        rest_h = (d - last_date[h]).days if h in last_date else 14
        rest_a = (d - last_date[a]).days if a in last_date else 14
        rest_home_list.append(min(int(rest_h), 21))
        rest_away_list.append(min(int(rest_a), 21))

        # Form BEFORE this match (last 5)
        form_home_list.append(sum(form_pts.get(h, [])))
        form_away_list.append(sum(form_pts.get(a, [])))

        # Update histories
        last_date[h] = d
        last_date[a] = d

        home_pts = 3 if result == 'H' else (1 if result == 'D' else 0)
        away_pts = 3 if result == 'A' else (1 if result == 'D' else 0)

        form_pts.setdefault(h, [])
        form_pts.setdefault(a, [])
        form_pts[h].append(home_pts)
        form_pts[a].append(away_pts)
        form_pts[h] = form_pts[h][-5:]
        form_pts[a] = form_pts[a][-5:]

    df['rest_days_home'] = rest_home_list
    df['rest_days_away'] = rest_away_list
    df['form_home'] = form_home_list
    df['form_away'] = form_away_list
    return df


def add_travel_and_stage(df):
    df['travel_km'] = df.apply(
        lambda r: get_travel_km(r['HomeTeam'], r['AwayTeam']), axis=1
    )
    # Season stage: each team's match number within the season
    match_count = {}
    stages = []
    for _, row in df.iterrows():
        key = (row['HomeTeam'], row['season'])
        match_count[key] = match_count.get(key, 0) + 1
        stages.append(match_count[key])
    df['season_stage'] = stages
    return df


# ---------------------------------------------------------------------------
# 4. MMM transforms: adstock (carryover) + saturation (diminishing returns)
# ---------------------------------------------------------------------------

def adstock_per_team(df, col_home, col_away, decay=0.5):
    """
    Apply adstock per team, in chronological order.
    Returns two series: adstocked home and adstocked away values.
    """
    adstock_vals = {}     # team -> current adstock value
    result_home, result_away = [], []

    for _, row in df.iterrows():
        h = row['HomeTeam']
        a = row['AwayTeam']

        # Current adstock = raw_value + decay * prev_adstock
        adstock_h = row[col_home] + decay * adstock_vals.get(h, 0.0)
        adstock_a = row[col_away] + decay * adstock_vals.get(a, 0.0)

        result_home.append(adstock_h)
        result_away.append(adstock_a)

        adstock_vals[h] = adstock_h
        adstock_vals[a] = adstock_a

    return result_home, result_away


def saturation(series, alpha):
    arr = np.array(series, dtype=float)
    return arr / (arr + alpha)


def apply_mmm_transforms(df):
    # Saturation: rest (alpha=4 means 4-day rest = 50% of max benefit)
    df['sat_rest_home'] = saturation(df['rest_days_home'], alpha=4)
    df['sat_rest_away'] = saturation(df['rest_days_away'], alpha=4)

    # Saturation: travel (alpha=100 km)
    df['sat_travel'] = saturation(df['travel_km'], alpha=100)

    # Adstock on form per team
    ads_form_h, ads_form_a = adstock_per_team(df, 'form_home', 'form_away', decay=0.5)
    df['adstock_form_home'] = ads_form_h
    df['adstock_form_away'] = ads_form_a

    return df


# ---------------------------------------------------------------------------
# 5. Build two-row-per-match dataset
#    (home + away perspective so home_advantage has variance)
# ---------------------------------------------------------------------------

def build_model_df(df):
    """
    Each match -> 2 rows:
      - row 1 (home perspective): team=HomeTeam, home_advantage=1,
                                  sat_rest=sat_rest_home, form=adstock_form_home,
                                  opp_form=adstock_form_away, sat_travel=0
      - row 2 (away perspective): team=AwayTeam, home_advantage=0,
                                  sat_rest=sat_rest_away, form=adstock_form_away,
                                  opp_form=adstock_form_home, sat_travel=sat_travel
    Target: goals scored by the team in that row.
    """
    rows = []
    for i, r in df.iterrows():
        # Home row
        rows.append({
            'match_id': i,
            'team': r['HomeTeam'],
            'opponent': r['AwayTeam'],
            'Date': r['Date'],
            'season': r['season'],
            'home_advantage': 1,
            'sat_rest': r['sat_rest_home'],
            'adstock_form': r['adstock_form_home'],
            'adstock_opp_form': r['adstock_form_away'],
            'sat_travel': 0.0,
            'season_stage': r['season_stage'],
            'goals_scored': r['FTHG'],
        })
        # Away row
        rows.append({
            'match_id': i,
            'team': r['AwayTeam'],
            'opponent': r['HomeTeam'],
            'Date': r['Date'],
            'season': r['season'],
            'home_advantage': 0,
            'sat_rest': r['sat_rest_away'],
            'adstock_form': r['adstock_form_away'],
            'adstock_opp_form': r['adstock_form_home'],
            'sat_travel': r['sat_travel'],
            'season_stage': r['season_stage'],
            'goals_scored': r['FTAG'],
        })

    model_df = pd.DataFrame(rows).reset_index(drop=True)
    print(f"Model dataset: {len(model_df)} rows (2 per match), 6 features")
    return model_df


FEATURES = [
    'sat_rest',
    'home_advantage',
    'adstock_form',
    'adstock_opp_form',
    'season_stage',
    'sat_travel',
]


# ---------------------------------------------------------------------------
# 6. Ridge regression + CV
# ---------------------------------------------------------------------------

def run_regression(model_df):
    X = model_df[FEATURES].values
    y = model_df['goals_scored'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    ridge = Ridge(alpha=1.0)
    ridge_r2 = cross_val_score(ridge, X_scaled, y, cv=kf, scoring='r2')
    ridge_mae = -cross_val_score(ridge, X_scaled, y, cv=kf, scoring='neg_mean_absolute_error')

    print(f"\nRidge CV R2:  {ridge_r2.mean():.4f} (+/- {ridge_r2.std():.4f})")
    print(f"Ridge CV MAE: {ridge_mae.mean():.4f} (+/- {ridge_mae.std():.4f})")

    ols = LinearRegression()
    ols_r2 = cross_val_score(ols, X_scaled, y, cv=kf, scoring='r2')
    ols_mae = -cross_val_score(ols, X_scaled, y, cv=kf, scoring='neg_mean_absolute_error')
    print(f"OLS  CV R2:   {ols_r2.mean():.4f} (+/- {ols_r2.std():.4f})")
    print(f"OLS  CV MAE:  {ols_mae.mean():.4f} (+/- {ols_mae.std():.4f})")

    ridge.fit(X_scaled, y)
    coefs_scaled = dict(zip(FEATURES, ridge.coef_))
    intercept_scaled = ridge.intercept_

    # Convert to original-space coefficients for interpretable decomposition
    # coef_orig = coef_scaled / std(X)
    # intercept_orig = intercept_scaled - sum(coef_scaled * mean(X) / std(X))
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    stds[stds == 0] = 1  # avoid div by zero for constant features

    coefs_orig = ridge.coef_ / stds
    intercept_orig = ridge.intercept_ - np.sum(ridge.coef_ * means / stds)
    coefs_orig_dict = dict(zip(FEATURES, coefs_orig))

    print("\nRidge coefficients (original scale, goals/unit):")
    for feat, coef in sorted(coefs_orig_dict.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {feat:30s}: {coef:+.6f}")

    print(f"  {'intercept':30s}: {intercept_orig:+.4f}")

    return ridge, scaler, coefs_scaled, coefs_orig_dict, intercept_orig, \
           ridge_r2.mean(), ridge_mae.mean()


# ---------------------------------------------------------------------------
# 7. Contribution decomposition (original scale)
# ---------------------------------------------------------------------------

def decompose_contributions(model_df, coefs_orig, intercept_orig):
    contrib_df = pd.DataFrame()
    contrib_df['match_id'] = model_df['match_id']
    contrib_df['team'] = model_df['team'].values
    contrib_df['Date'] = model_df['Date'].values
    contrib_df['baseline'] = intercept_orig
    contrib_df['goals_scored'] = model_df['goals_scored'].values

    label_map = {
        'sat_rest': 'rest',
        'home_advantage': 'home_adv',
        'adstock_form': 'form',
        'adstock_opp_form': 'opp_form',
        'season_stage': 'season_stage',
        'sat_travel': 'travel',
    }

    for feat in FEATURES:
        col = label_map[feat]
        contrib_df[col] = model_df[feat].values * coefs_orig[feat]

    contrib_df['predicted'] = (
        contrib_df['baseline']
        + contrib_df['rest']
        + contrib_df['home_adv']
        + contrib_df['form']
        + contrib_df['opp_form']
        + contrib_df['season_stage']
        + contrib_df['travel']
    )

    print("\nAverage contributions across all team-match rows:")
    avg_pred = contrib_df['predicted'].mean()
    print(f"  {'baseline':20s}: {intercept_orig:+.4f}")
    for col in ['rest', 'home_adv', 'form', 'opp_form', 'season_stage', 'travel']:
        avg = contrib_df[col].mean()
        pct = (abs(avg) / avg_pred * 100) if avg_pred else 0
        print(f"  {col:20s}: {avg:+.4f}  ({pct:.1f}% of avg predicted goals)")

    return contrib_df


# ---------------------------------------------------------------------------
# 8. Saturation curve for rest_days
# ---------------------------------------------------------------------------

def compute_saturation_curves():
    alpha = 4
    rows = [{'rest_days': d, 'sat_rest': round(d / (d + alpha), 4)} for d in range(1, 15)]
    df_sat = pd.DataFrame(rows)
    print("\nSaturation curve for rest_days (alpha=4):")
    print(df_sat.to_string(index=False))
    return df_sat


def find_saturation_elbow(sat_df):
    """First day where marginal gain drops below 0.05."""
    gains = sat_df['sat_rest'].diff().fillna(1.0)
    for i, gain in enumerate(gains):
        if i == 0:
            continue
        if gain < 0.05:
            return int(sat_df.iloc[i]['rest_days'])
    return 7


# ---------------------------------------------------------------------------
# 9. Save outputs & key metrics
# ---------------------------------------------------------------------------

def save_all_outputs(df, model_df, contrib_df, sat_df,
                     coefs_orig, intercept_orig, ridge_r2, ridge_mae):

    # match_features.csv (from original df — one row per match)
    feat_cols = ['HomeTeam', 'AwayTeam', 'Date', 'season', 'FTHG', 'FTAG', 'FTR',
                 'rest_days_home', 'rest_days_away', 'travel_km', 'form_home', 'form_away',
                 'season_stage', 'sat_rest_home', 'sat_rest_away', 'sat_travel',
                 'adstock_form_home', 'adstock_form_away']
    out_cols = [c for c in feat_cols if c in df.columns]
    df[out_cols].to_csv(os.path.join(OUTPUTS_DIR, 'match_features.csv'), index=False)
    print(f"\nSaved match_features.csv ({len(df)} rows)")

    # contribution_decomposition.csv
    contrib_df.to_csv(os.path.join(OUTPUTS_DIR, 'contribution_decomposition.csv'), index=False)
    print(f"Saved contribution_decomposition.csv")

    # saturation_curves.csv
    sat_df.to_csv(os.path.join(OUTPUTS_DIR, 'saturation_curves.csv'), index=False)
    print(f"Saved saturation_curves.csv")

    # Key metrics
    n_matches = len(df)
    n_seasons = df['season'].nunique()
    home_adv_avg = contrib_df['home_adv'].mean()
    form_coef = coefs_orig['adstock_form']
    travel_coef = coefs_orig['sat_travel']
    rest_optimal = find_saturation_elbow(sat_df)

    contrib_cols = ['rest', 'home_adv', 'form', 'opp_form', 'travel']
    avg_contribs = {c: abs(contrib_df[c].mean()) for c in contrib_cols}
    top_contributor = max(avg_contribs, key=avg_contribs.get)

    lines = [
        f"n_matches={n_matches}",
        f"n_seasons={n_seasons}",
        f"ridge_cv_r2={ridge_r2:.4f}",
        f"ridge_cv_mae={ridge_mae:.4f}",
        f"top_contributor={top_contributor}",
        f"home_adv_avg_goals={home_adv_avg:.4f}",
        f"rest_optimal_days={rest_optimal}",
        f"form_coefficient={form_coef:.6f}",
        f"travel_coefficient={travel_coef:.6f}",
    ]

    with open(os.path.join(OUTPUTS_DIR, 'key_metrics.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print(f"Saved key_metrics.txt")
    print("\n=== KEY METRICS ===")
    for line in lines:
        print(f"  {line}")

    return {k: v for k, v in [l.split('=', 1) for l in lines]}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Model 14 - Premier League Marketing Mix Model")
    print("=" * 60)

    print("\n[1/6] Fetching data...")
    df_raw = fetch_all_data()

    print("\n[2/6] Parsing & cleaning...")
    df = parse_raw(df_raw)

    print("\n[3/6] Engineering features per-team...")
    df = build_team_histories(df)
    df = add_travel_and_stage(df)

    print("\n[4/6] Applying MMM transformations (adstock + saturation)...")
    df = apply_mmm_transforms(df)

    print("\n[5/6] Building model dataset & running regression...")
    model_df = build_model_df(df)
    (ridge, scaler, coefs_scaled, coefs_orig, intercept_orig,
     ridge_r2, ridge_mae) = run_regression(model_df)

    print("\n[6/6] Contribution decomposition & saving outputs...")
    contrib_df = decompose_contributions(model_df, coefs_orig, intercept_orig)
    sat_df = compute_saturation_curves()
    metrics = save_all_outputs(df, model_df, contrib_df, sat_df,
                               coefs_orig, intercept_orig, ridge_r2, ridge_mae)

    print("\nModel 14 complete. Outputs saved to outputs/")
    return metrics


if __name__ == '__main__':
    main()
