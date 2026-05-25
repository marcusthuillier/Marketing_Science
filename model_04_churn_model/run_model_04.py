import time, warnings, os, sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
import xgboost as xgb
import shap

warnings.filterwarnings('ignore')
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    try: sys.stdout.reconfigure(encoding='utf-8')
    except: pass

BASE_CACHE     = '../model_03_cohort_analysis/outputs/raw_season_stats.csv'
EXTENDED_CACHE = 'outputs/raw_season_stats_extended.csv'
OUTPUTS_DIR    = 'outputs/'
os.makedirs(OUTPUTS_DIR, exist_ok=True)

MIN_GP           = 20
ALL_SEASONS      = list(range(1996, 2026))
MAX_TRAIN_SEASON = 2024
PREDICT_SEASON   = 2025
VAL_CUTOFF       = 2019

FEATURES = [
    'AGE', 'CAREER_YEAR', 'ROUND_NUMBER', 'OVERALL_PICK',
    'GP', 'MIN_PG', 'PTS_PG', 'REB_PG', 'AST_PG',
    'STL_PG', 'BLK_PG', 'TOV_PG',
    'FG_PCT', 'FG3_PCT', 'FT_PCT', 'PERF',
    'PERF_DELTA', 'MIN_DELTA',
]

FEATURE_LABELS = [
    'Age', 'Career Year', 'Draft Round', 'Pick #', 'Games Played', 'Min/G',
    'Pts/G', 'Reb/G', 'Ast/G', 'Stl/G', 'Blk/G', 'Tov/G',
    'FG%', '3P%', 'FT%', 'Perf Score', 'Perf Trend', 'Min Trend',
]


def load_data():
    print('Loading season stats...')
    if os.path.exists(EXTENDED_CACHE):
        print(f'  Using extended cache: {EXTENDED_CACHE}')
        base_df = pd.read_csv(EXTENDED_CACHE)
    else:
        base_df = pd.read_csv(BASE_CACHE)
    base_df['PLAYER_ID']    = base_df['PLAYER_ID'].astype('int64')
    base_df['SEASON_START'] = base_df['SEASON_START'].astype(int)

    existing = set(base_df['SEASON_START'].unique())
    missing  = [s for s in ALL_SEASONS if s not in existing]
    print(f'  Cached seasons: {min(existing)}-{max(existing)} | Missing: {missing}')

    new_chunks = []
    if missing:
        from nba_api.stats.endpoints import leaguedashplayerstats
        for yr in missing:
            season_str = f'{yr}-{str(yr + 1)[-2:]}'
            print(f'  Fetching {season_str}...')
            for attempt in range(3):
                try:
                    time.sleep(1.5)
                    df = leaguedashplayerstats.LeagueDashPlayerStats(
                        season=season_str,
                        season_type_all_star='Regular Season',
                        per_mode_detailed='Totals'
                    ).get_data_frames()[0]
                    df['SEASON']       = season_str
                    df['SEASON_START'] = yr
                    df['PLAYER_ID']    = df['PLAYER_ID'].astype('int64')
                    new_chunks.append(df)
                    print(f'    {len(df)} players')
                    break
                except Exception as e:
                    print(f'    Attempt {attempt+1} failed: {e}')
                    time.sleep(3 * (attempt + 1))

    all_stats = pd.concat([base_df] + new_chunks, ignore_index=True) if new_chunks else base_df.copy()
    all_stats.to_csv(EXTENDED_CACHE, index=False)
    print(f'  Total: {len(all_stats):,} rows | {all_stats["SEASON"].nunique()} seasons')

    print('Loading draft history...')
    from nba_api.stats.endpoints import drafthistory
    time.sleep(1)
    dh_raw = drafthistory.DraftHistory(league_id='00').get_data_frames()[0]
    dh_raw['DRAFT_YEAR'] = dh_raw['SEASON'].astype(int)
    draft = dh_raw[dh_raw['DRAFT_YEAR'].between(1996, 2024)][[
        'PERSON_ID', 'DRAFT_YEAR', 'ROUND_NUMBER', 'OVERALL_PICK'
    ]].copy()
    draft['PERSON_ID']    = draft['PERSON_ID'].astype('int64')
    draft['ROUND_NUMBER'] = pd.to_numeric(draft['ROUND_NUMBER'], errors='coerce').fillna(3).astype(int)
    draft['OVERALL_PICK'] = pd.to_numeric(draft['OVERALL_PICK'], errors='coerce').fillna(99).astype(int)
    print(f'  {len(draft)} picks | {draft["DRAFT_YEAR"].nunique()} classes')

    return all_stats, draft


def engineer_features(all_stats, draft):
    print('Engineering features...')
    merged = all_stats.merge(
        draft[['PERSON_ID', 'DRAFT_YEAR', 'ROUND_NUMBER', 'OVERALL_PICK']],
        left_on='PLAYER_ID', right_on='PERSON_ID', how='left'
    )

    first_season = all_stats.groupby('PLAYER_ID')['SEASON_START'].min().rename('FIRST_SEASON')
    merged = merged.merge(first_season, on='PLAYER_ID', how='left')
    merged['DRAFT_YEAR']   = merged['DRAFT_YEAR'].fillna(merged['FIRST_SEASON']).astype(int)
    merged['ROUND_NUMBER'] = merged['ROUND_NUMBER'].fillna(3).astype(int)
    merged['OVERALL_PICK'] = merged['OVERALL_PICK'].fillna(99).astype(int)
    merged['CAREER_YEAR']  = (merged['SEASON_START'] - merged['DRAFT_YEAR'] + 1).clip(lower=1)

    for col in ['MIN', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV']:
        if col in merged.columns:
            merged[f'{col}_PG'] = (merged[col] / merged['GP'].replace(0, np.nan)).round(3)

    merged['PERF'] = (
        merged['PTS_PG'] + 1.2 * merged['REB_PG'] + 1.5 * merged['AST_PG']
        + merged['STL_PG'] + 1.5 * merged['BLK_PG'] - merged['TOV_PG']
    ).round(3)

    for col in ['FG_PCT', 'FG3_PCT', 'FT_PCT']:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0.0)

    merged = merged.sort_values(['PLAYER_ID', 'SEASON_START'])
    merged['PERF_DELTA'] = merged.groupby('PLAYER_ID')['PERF'].diff().round(3).fillna(0)
    merged['MIN_DELTA']  = merged.groupby('PLAYER_ID')['MIN_PG'].diff().round(3).fillna(0)

    print('Building churn labels...')
    active_next = (
        all_stats[all_stats['GP'] >= MIN_GP][['PLAYER_ID', 'SEASON_START']]
        .assign(ACTIVE_NEXT=1).copy()
    )
    active_next['SEASON_START'] -= 1
    merged = merged.merge(active_next, on=['PLAYER_ID', 'SEASON_START'], how='left')
    merged['ACTIVE_NEXT'] = merged['ACTIVE_NEXT'].fillna(0).astype(int)
    merged['CHURNED']     = 1 - merged['ACTIVE_NEXT']

    non_delta = [f for f in FEATURES if f not in ('PERF_DELTA', 'MIN_DELTA')]
    df_model  = merged[merged['GP'] >= MIN_GP][
        ['PLAYER_ID', 'PLAYER_NAME', 'SEASON_START', 'CHURNED'] + FEATURES
    ].dropna(subset=non_delta).copy()

    available_seasons = set(all_stats['SEASON_START'].unique())
    df_model = df_model[df_model['SEASON_START'].apply(lambda s: (s + 1) in available_seasons)]

    predict = merged[
        (merged['SEASON_START'] == PREDICT_SEASON) & (merged['GP'] >= MIN_GP)
    ][['PLAYER_ID', 'PLAYER_NAME', 'SEASON_START'] + FEATURES].dropna(subset=non_delta).copy()
    predict['PERF_DELTA'] = predict['PERF_DELTA'].fillna(0)
    predict['MIN_DELTA']  = predict['MIN_DELTA'].fillna(0)

    print(f'  Labeled player-seasons: {len(df_model):,}')
    print(f'  Overall churn rate:     {df_model["CHURNED"].mean():.1%}')
    print(f'  {PREDICT_SEASON}-26 predict pool: {len(predict):,} players')

    train   = df_model[df_model['SEASON_START'] <= MAX_TRAIN_SEASON].copy()
    X_train = train[train['SEASON_START'] <= VAL_CUTOFF][FEATURES]
    y_train = train[train['SEASON_START'] <= VAL_CUTOFF]['CHURNED']
    X_test  = train[train['SEASON_START']  > VAL_CUTOFF][FEATURES]
    y_test  = train[train['SEASON_START']  > VAL_CUTOFF]['CHURNED']

    print(f'  Train: {len(X_train):,} | Val: {len(X_test):,} | Predict: {len(predict):,}')
    return df_model, predict, train, X_train, y_train, X_test, y_test


def train_models(X_train, y_train, X_test, y_test, train):
    print('\nTraining Logistic Regression...')
    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=1000, C=0.5, random_state=42)
    lr.fit(X_train_sc, y_train)
    lr_probs = lr.predict_proba(X_test_sc)[:, 1]
    lr_auc   = roc_auc_score(y_test, lr_probs)
    print(f'  AUC: {lr_auc:.3f}')
    print(classification_report(y_test, lr.predict(X_test_sc), target_names=['Retain', 'Churn']))

    print('Training XGBoost...')
    scale_pos = float((y_train == 0).sum()) / float((y_train == 1).sum())
    xgb_model = xgb.XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos, min_child_weight=5,
        random_state=42, verbosity=0,
        early_stopping_rounds=30, eval_metric='auc'
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    xgb_auc   = roc_auc_score(y_test, xgb_probs)
    print(f'  AUC: {xgb_auc:.3f}')
    print(classification_report(y_test, xgb_model.predict(X_test), target_names=['Retain', 'Churn']))

    fi_df = pd.DataFrame({
        'Feature':        FEATURES,
        'XGB_Importance': xgb_model.feature_importances_,
        'LR_Coef':        lr.coef_[0],
    }).sort_values('XGB_Importance', ascending=False).reset_index(drop=True)
    fi_df.to_csv(OUTPUTS_DIR + 'feature_importance.csv', index=False)

    churn_by_yr = (
        train.groupby('CAREER_YEAR')['CHURNED']
        .agg(CHURN_RATE='mean', N='count')
        .reset_index()
        .query('N >= 30 and CAREER_YEAR <= 15')
    )
    churn_by_yr['CHURN_RATE'] = churn_by_yr['CHURN_RATE'].round(3)
    churn_by_yr.to_csv(OUTPUTS_DIR + 'churn_by_career_year.csv', index=False)

    pd.DataFrame([
        {'Model': 'Logistic Regression', 'AUC': round(lr_auc, 3)},
        {'Model': 'XGBoost',             'AUC': round(xgb_auc, 3)},
    ]).to_csv(OUTPUTS_DIR + 'model_metrics.csv', index=False)

    return lr, scaler, xgb_model, lr_auc, xgb_auc, fi_df, churn_by_yr


def compute_shap(xgb_model, X_test):
    print('Computing SHAP values...')
    explainer = shap.TreeExplainer(xgb_model)
    shap_vals = explainer.shap_values(X_test)
    return explainer, shap_vals


def generate_predictions(xgb_model, predict):
    print('\nGenerating predictions...')
    if len(predict) == 0:
        print('  No predict-season data available.')
        return pd.DataFrame()
    pred_probs = xgb_model.predict_proba(predict[FEATURES])[:, 1]
    predict    = predict.copy()
    predict['CHURN_PROB'] = pred_probs.round(3)
    predict['CHURN_PRED'] = (pred_probs >= 0.5).astype(int)
    out_cols    = ['PLAYER_NAME', 'CAREER_YEAR', 'AGE', 'MIN_PG', 'PTS_PG', 'PERF', 'CHURN_PROB', 'CHURN_PRED']
    predict_out = predict[out_cols].sort_values('CHURN_PROB', ascending=False).reset_index(drop=True)
    predict_out.to_csv(OUTPUTS_DIR + 'churn_predictions_2026.csv', index=False)
    print(f'  Predicted churners: {predict["CHURN_PRED"].sum()} / {len(predict)}')
    print('\n  Top 20 at-risk players:')
    print(predict_out[predict_out['CHURN_PRED'] == 1].head(20).to_string(index=False))
    return predict_out


def generate_plots(fi_df, churn_by_yr, shap_vals, X_test, predict_out, predict, explainer, show=False):
    import matplotlib.pyplot as plt
    print('\nPlotting...')

    fi_top = fi_df.sort_values('XGB_Importance', ascending=True).tail(12)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(fi_top['Feature'], fi_top['XGB_Importance'], color='#4C72B0', alpha=0.82)
    ax.set_title('XGBoost Feature Importance — NBA Player Churn', fontsize=13)
    ax.set_xlabel('Importance Score')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR + 'feature_importance.png', dpi=150, bbox_inches='tight')
    if show: plt.show()
    else: plt.close()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(churn_by_yr['CAREER_YEAR'], churn_by_yr['CHURN_RATE'] * 100, color='#9B3D36', alpha=0.80)
    ax.set_title('Churn Rate by Career Year — NBA Players with >=20 GP', fontsize=13)
    ax.set_xlabel('Career Year')
    ax.set_ylabel('% Who Do Not Return at >=20 GP Next Season')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR + 'churn_by_career_year.png', dpi=150, bbox_inches='tight')
    if show: plt.show()
    else: plt.close()

    shap.summary_plot(shap_vals, X_test.values, feature_names=FEATURE_LABELS,
                      plot_type='dot', max_display=15, show=False)
    plt.title('SHAP — What Drives NBA Player Churn', fontsize=13, pad=10)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR + 'shap_beeswarm.png', dpi=150, bbox_inches='tight')
    if show: plt.show()
    else: plt.close()

    if len(predict_out) > 0:
        top_player = predict_out.iloc[0]
        top_row    = predict[predict['PLAYER_NAME'] == top_player['PLAYER_NAME']][FEATURES].iloc[[0]]
        sv_top     = explainer.shap_values(top_row)
        shap_exp   = shap.Explanation(
            values=sv_top[0],
            base_values=float(explainer.expected_value),
            data=top_row.values[0],
            feature_names=FEATURE_LABELS,
        )
        shap.waterfall_plot(shap_exp, max_display=12, show=False)
        plt.title(f'SHAP Waterfall — {top_player["PLAYER_NAME"]} (Churn Prob: {top_player["CHURN_PROB"]:.0%})',
                  fontsize=11, pad=10)
        plt.tight_layout()
        plt.savefig(OUTPUTS_DIR + 'shap_waterfall.png', dpi=150, bbox_inches='tight')
        if show: plt.show()
        else: plt.close()
        print(f'  Saved: shap_waterfall.png ({top_player["PLAYER_NAME"]})')

    print('  All plots saved.')


def print_findings(train, lr_auc, xgb_auc, fi_df, churn_by_yr):
    print('\n' + '=' * 60)
    print('KEY FINDINGS — NBA Player Churn Prediction')
    print('=' * 60)
    print(f'Training player-seasons:  {len(train):,}')
    print(f'Overall churn rate:       {train["CHURNED"].mean():.1%}')
    print(f'Logistic Regression AUC:  {lr_auc:.3f}')
    print(f'XGBoost AUC:              {xgb_auc:.3f}')
    print(f'\nTop 5 predictors (XGB importance):')
    for _, row in fi_df.head(5).iterrows():
        print(f'  {row["Feature"]:<18}  {row["XGB_Importance"]:.4f}')
    print(f'\nChurn rate by career year:')
    for _, row in churn_by_yr.iterrows():
        print(f'  Year {int(row["CAREER_YEAR"]):>2}: {row["CHURN_RATE"]*100:.1f}%  (n={int(row["N"])})')
    print('\nAll outputs saved to outputs/')


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    all_stats, draft = load_data()
    df_model, predict, train, X_train, y_train, X_test, y_test = engineer_features(all_stats, draft)
    lr, scaler, xgb_model, lr_auc, xgb_auc, fi_df, churn_by_yr = train_models(X_train, y_train, X_test, y_test, train)
    explainer, shap_vals = compute_shap(xgb_model, X_test)
    predict_out = generate_predictions(xgb_model, predict)
    generate_plots(fi_df, churn_by_yr, shap_vals, X_test, predict_out, predict, explainer, show=False)
    print_findings(train, lr_auc, xgb_auc, fi_df, churn_by_yr)
