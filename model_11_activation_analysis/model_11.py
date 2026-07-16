"""
Model 11 - NBA Rookie Activation Analysis
Question: Which first-season stat best predicts NBA career longevity?
Method: Logistic Regression + XGBoost + SHAP, threshold aha-moment analysis
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
import xgboost as xgb
import shap

warnings.filterwarnings('ignore')

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
os.makedirs(OUTPUTS_DIR, exist_ok=True)

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# 1. Pull season data from nba_api
# ---------------------------------------------------------------------------

def fetch_all_seasons():
    from nba_api.stats.endpoints import LeagueDashPlayerStats

    all_frames = []
    seasons = [f"{y}-{str(y+1)[-2:]}" for y in range(1987, 2024)]

    print(f"Fetching {len(seasons)} seasons from nba_api...")

    for season in seasons:
        try:
            result = LeagueDashPlayerStats(
                season=season,
                per_mode_detailed='Totals',
                season_type_all_star='Regular Season',
                measure_type_detailed_defense='Base',
            )
            df = result.get_data_frames()[0]
            df['SEASON'] = season
            df['SEASON_YEAR'] = int(season[:4])
            all_frames.append(df)
            print(f"  {season}: {len(df)} players")
        except Exception as e:
            print(f"  {season}: ERROR - {e}")

        time.sleep(0.65)

    combined = pd.concat(all_frames, ignore_index=True)
    print(f"\nTotal rows fetched: {len(combined)}")
    return combined


# ---------------------------------------------------------------------------
# 2. Identify rookie seasons and build features
# ---------------------------------------------------------------------------

def build_features(df_all):
    # Each player's first season in the dataset = their rookie season
    # Use PLAYER_ID to track across seasons
    df_all = df_all.copy()

    first_season = (
        df_all.groupby('PLAYER_ID')['SEASON_YEAR']
        .min()
        .reset_index()
        .rename(columns={'SEASON_YEAR': 'FIRST_SEASON_YEAR'})
    )
    df_all = df_all.merge(first_season, on='PLAYER_ID')

    # Rookie row: player's first season
    rookies = df_all[df_all['SEASON_YEAR'] == df_all['FIRST_SEASON_YEAR']].copy()

    # Filter: first season between 1990 and 2015 (inclusive)
    rookies = rookies[(rookies['FIRST_SEASON_YEAR'] >= 1990) & (rookies['FIRST_SEASON_YEAR'] <= 2015)]

    print(f"\nRookies after first-season year filter (1990-2015): {len(rookies)}")

    # ---------------------------------------------------------------------------
    # 3. Success label: >= 5 seasons with MIN >= 1000
    # ---------------------------------------------------------------------------
    qualified_seasons = df_all[df_all['MIN'] >= 1000]
    career_seasons = (
        qualified_seasons.groupby('PLAYER_ID')['SEASON_YEAR']
        .nunique()
        .reset_index()
        .rename(columns={'SEASON_YEAR': 'QUAL_SEASONS'})
    )
    rookies = rookies.merge(career_seasons, on='PLAYER_ID', how='left')
    rookies['QUAL_SEASONS'] = rookies['QUAL_SEASONS'].fillna(0)
    rookies['SUCCESS'] = (rookies['QUAL_SEASONS'] >= 5).astype(int)

    # ---------------------------------------------------------------------------
    # 4. Compute per-game features from rookie season row
    # ---------------------------------------------------------------------------
    rookies['GP'] = rookies['GP'].clip(lower=1)

    rookies['PTS_PG']  = rookies['PTS']  / rookies['GP']
    rookies['AST_PG']  = rookies['AST']  / rookies['GP']
    rookies['REB_PG']  = rookies['REB']  / rookies['GP']
    rookies['STL_PG']  = rookies['STL']  / rookies['GP']
    rookies['BLK_PG']  = rookies['BLK']  / rookies['GP']
    rookies['MIN_PG']  = rookies['MIN']  / rookies['GP']
    rookies['TOV_PG']  = rookies['TOV']  / rookies['GP']

    # True Shooting %: PTS / (2 * (FGA + 0.44 * FTA))
    denom = 2 * (rookies['FGA'] + 0.44 * rookies['FTA'])
    safe_denom = denom.replace(0, np.nan)
    rookies['TS_PCT'] = (rookies['PTS'] / safe_denom).fillna(0.0)

    # Age
    rookies['AGE'] = rookies['AGE'].fillna(rookies['AGE'].median())

    # ---------------------------------------------------------------------------
    # 5. Filter: MIN_PG >= 5 (played meaningful minutes)
    # ---------------------------------------------------------------------------
    rookies = rookies[rookies['MIN_PG'] >= 5].copy()
    print(f"After MIN_PG >= 5 filter: {len(rookies)} rookies")
    print(f"Success rate: {rookies['SUCCESS'].mean():.3f}")

    return rookies


# ---------------------------------------------------------------------------
# 6. Modelling
# ---------------------------------------------------------------------------

FEATURES = ['PTS_PG', 'AST_PG', 'REB_PG', 'STL_PG', 'BLK_PG',
            'MIN_PG', 'TS_PCT', 'TOV_PG', 'AGE']


def run_models(rookies):
    X = rookies[FEATURES].values
    y = rookies['SUCCESS'].values

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # Logistic Regression with scaling
    lr_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
    ])
    lr_aucs = cross_val_score(lr_pipe, X, y, cv=cv, scoring='roc_auc')
    lr_auc = lr_aucs.mean()
    print(f"\nLogistic Regression CV AUC: {lr_auc:.4f} (+/- {lr_aucs.std():.4f})")

    # XGBoost
    xgb_clf = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    xgb_aucs = cross_val_score(xgb_clf, X, y, cv=cv, scoring='roc_auc')
    xgb_auc = xgb_aucs.mean()
    print(f"XGBoost CV AUC:             {xgb_auc:.4f} (+/- {xgb_aucs.std():.4f})")

    # Fit XGBoost on full data for SHAP
    xgb_clf.fit(X, y)

    return lr_auc, xgb_auc, xgb_clf


# ---------------------------------------------------------------------------
# 7. SHAP feature importance
# ---------------------------------------------------------------------------

def compute_shap(rookies, xgb_clf):
    X = rookies[FEATURES].values

    explainer = shap.TreeExplainer(xgb_clf)
    shap_values = explainer.shap_values(X)

    # Mean absolute SHAP per feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    shap_df = pd.DataFrame({
        'feature': FEATURES,
        'shap_importance': mean_abs_shap
    }).sort_values('shap_importance', ascending=False).reset_index(drop=True)

    print("\nSHAP Feature Importances:")
    print(shap_df.to_string(index=False))

    top_feature = shap_df.iloc[0]['feature']
    print(f"\nTop predictor: {top_feature}")

    out_path = os.path.join(OUTPUTS_DIR, 'shap_importance.csv')
    shap_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    return shap_df, top_feature


# ---------------------------------------------------------------------------
# 8. Aha moment threshold analysis
# ---------------------------------------------------------------------------

def aha_moment(rookies, top_feature):
    values = rookies[top_feature].values
    success = rookies['SUCCESS'].values

    # Scan 20th to 80th percentile, step 5
    percentiles = np.arange(20, 85, 5)
    thresholds = np.percentile(values, percentiles)

    results = []
    for pct, thresh in zip(percentiles, thresholds):
        above_mask = values >= thresh
        below_mask = values < thresh
        n_above = above_mask.sum()
        n_below = below_mask.sum()

        if n_above < 30 or n_below < 30:
            continue

        above_rate = success[above_mask].mean()
        below_rate = success[below_mask].mean()

        # Compute lift in both directions; pick the one with higher contrast
        if below_rate > 0 and above_rate > 0:
            lift_above = above_rate / below_rate
            lift_below = below_rate / above_rate
        elif above_rate > 0:
            lift_above = np.inf
            lift_below = np.nan
        elif below_rate > 0:
            lift_above = np.nan
            lift_below = np.inf
        else:
            lift_above = np.nan
            lift_below = np.nan

        # Best directional lift
        best_lift = max(
            lift_above if not np.isnan(lift_above) else 0,
            lift_below if not np.isnan(lift_below) else 0,
        )
        direction = 'above' if (not np.isnan(lift_above) and lift_above >= (lift_below if not np.isnan(lift_below) else 0)) else 'below'

        results.append({
            'percentile': pct,
            'threshold': round(thresh, 4),
            'above_rate': round(above_rate, 4),
            'below_rate': round(below_rate, 4),
            'lift': round(best_lift, 4),
            'direction': direction,
            'n_above': int(n_above),
            'n_below': int(n_below),
        })

    aha_df = pd.DataFrame(results)
    print("\nAha Moment Threshold Analysis:")
    print(aha_df.to_string(index=False))

    # Best lift (already maximised over both directions)
    best_row = aha_df.loc[aha_df['lift'].idxmax()]
    direction = best_row['direction']
    if direction == 'above':
        high_rate = best_row['above_rate']
        low_rate  = best_row['below_rate']
        print(f"\nBest threshold: {top_feature} >= {best_row['threshold']:.2f}  (direction: above)")
    else:
        high_rate = best_row['below_rate']
        low_rate  = best_row['above_rate']
        print(f"\nBest threshold: {top_feature} < {best_row['threshold']:.2f}  (direction: below)")
    print(f"  High-success group rate: {high_rate:.3f}, Low-success group rate: {low_rate:.3f}, Lift: {best_row['lift']:.2f}x")

    out_path = os.path.join(OUTPUTS_DIR, 'aha_threshold_analysis.csv')
    aha_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    return aha_df, best_row


# ---------------------------------------------------------------------------
# 9. Save key metrics
# ---------------------------------------------------------------------------

def save_metrics(rookies, lr_auc, xgb_auc, shap_df, top_feature, best_row):
    n_rookies = len(rookies)
    success_rate = rookies['SUCCESS'].mean()
    top_shap = shap_df.iloc[0]['shap_importance']

    direction = best_row['direction']
    if direction == 'above':
        high_rate = best_row['above_rate']
        low_rate  = best_row['below_rate']
    else:
        high_rate = best_row['below_rate']
        low_rate  = best_row['above_rate']

    lines = [
        f"n_rookies={n_rookies}",
        f"success_rate={success_rate:.4f}",
        f"lr_auc={lr_auc:.4f}",
        f"xgb_auc={xgb_auc:.4f}",
        f"top_feature={top_feature}",
        f"top_feature_shap={top_shap:.4f}",
        f"aha_threshold={best_row['threshold']:.4f}",
        f"aha_percentile={best_row['percentile']:.0f}",
        f"aha_direction={direction}",
        f"aha_above_rate={best_row['above_rate']:.4f}",
        f"aha_below_rate={best_row['below_rate']:.4f}",
        f"aha_high_rate={high_rate:.4f}",
        f"aha_low_rate={low_rate:.4f}",
        f"aha_lift={best_row['lift']:.4f}",
        f"aha_n_above={best_row['n_above']}",
        f"aha_n_below={best_row['n_below']}",
    ]

    out_path = os.path.join(OUTPUTS_DIR, 'key_metrics.txt')
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"\nSaved: {out_path}")

    print("\n=== KEY METRICS ===")
    for line in lines:
        print(f"  {line}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Model 11 - NBA Rookie Activation Analysis")
    print("=" * 60)

    # Check for cached data to speed up re-runs
    cache_path = os.path.join(OUTPUTS_DIR, '_raw_all_seasons.parquet')
    if os.path.exists(cache_path):
        print(f"Loading cached season data from {cache_path}")
        df_all = pd.read_parquet(cache_path)
        print(f"Loaded {len(df_all)} rows from cache.")
    else:
        df_all = fetch_all_seasons()
        df_all.to_parquet(cache_path, index=False)
        print(f"Cached to {cache_path}")

    rookies = build_features(df_all)

    lr_auc, xgb_auc, xgb_clf = run_models(rookies)

    shap_df, top_feature = compute_shap(rookies, xgb_clf)

    aha_df, best_row = aha_moment(rookies, top_feature)

    save_metrics(rookies, lr_auc, xgb_auc, shap_df, top_feature, best_row)

    print("\nModel 11 complete. All outputs saved to outputs/")


if __name__ == '__main__':
    main()
