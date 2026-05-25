"""
Model 06 -- World Cup Winner Propensity Model
Which national teams have the profile of a World Cup winner?
Logistic regression (L2) + SHAP explainability
"""

import os, warnings
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve

warnings.filterwarnings('ignore')

# ── CONSTANTS ─────────────────────────────────────────────────────────
OUTPUTS      = 'outputs/'
CACHE        = OUTPUTS + 'intl_results.csv'
BG, RED, BLUE = '#f5f0eb', '#9b3d36', '#4c72b0'
RANDOM_STATE = 42
CUTOFF_2026  = pd.to_datetime('2026-06-01')
HOST_2026    = {'United States', 'Canada', 'Mexico'}

FEATURES = [
    'elo_z', 'win_rate_12m', 'gf_pg_12m', 'ga_pg_12m',
    'wc_apps_z', 'wc_wins_z', 'wc_wins_last3', 'wc_wins_weighted',
    'prev_wc_depth', 'is_host', 'continent_enc',
]

WC_YEARS = [1990, 1994, 1998, 2002, 2006, 2010, 2014, 2018, 2022]

WC_START = {
    1990: '1990-06-08', 1994: '1994-06-17', 1998: '1998-06-10',
    2002: '2002-05-31', 2006: '2006-06-09', 2010: '2010-06-11',
    2014: '2014-06-12', 2018: '2018-06-14', 2022: '2022-11-20',
}
WC_WINNER = {
    1990: 'Germany',   1994: 'Brazil',  1998: 'France',  2002: 'Brazil',
    2006: 'Italy',     2010: 'Spain',   2014: 'Germany', 2018: 'France',
    2022: 'Argentina',
}
WC_HOST = {
    1990: 'Italy',        1994: 'United States', 1998: 'France',
    2002: 'South Korea',  2006: 'Germany',       2010: 'South Africa',
    2014: 'Brazil',       2018: 'Russia',        2022: 'Qatar',
}

WC_ALL_YEARS = [
    1930, 1934, 1938, 1950, 1954, 1958, 1962, 1966,
    1970, 1974, 1978, 1982, 1986,
] + WC_YEARS

WC_ALL_STARTS = {
    1930: '1930-07-13', 1934: '1934-05-27', 1938: '1938-06-04',
    1950: '1950-06-24', 1954: '1954-06-16', 1958: '1958-06-08',
    1962: '1962-05-30', 1966: '1966-07-11', 1970: '1970-05-30',
    1974: '1974-06-13', 1978: '1978-06-01', 1982: '1982-06-13',
    1986: '1986-05-31',
    **WC_START,
}
WC_ALL_WINNERS = {
    1930: 'Uruguay',      1934: 'Italy',        1938: 'Italy',
    1950: 'Uruguay',      1954: 'West Germany', 1958: 'Brazil',
    1962: 'Brazil',       1966: 'England',      1970: 'Brazil',
    1974: 'West Germany', 1978: 'Argentina',    1982: 'Italy',
    1986: 'Argentina',
    **WC_WINNER,
}
HIST_NAME_MAP = {
    'West Germany':               'Germany',
    'German Democratic Republic': 'Germany',
    'Soviet Union':               'Russia',
    'Czechoslovakia':             'Czech Republic',
    'Yugoslavia':                 'Serbia',
}
CONTINENT = {
    'Argentina': 'CONMEBOL', 'Brazil': 'CONMEBOL', 'Uruguay': 'CONMEBOL',
    'Colombia': 'CONMEBOL', 'Chile': 'CONMEBOL', 'Ecuador': 'CONMEBOL',
    'Paraguay': 'CONMEBOL', 'Peru': 'CONMEBOL', 'Bolivia': 'CONMEBOL',
    'Germany': 'UEFA', 'France': 'UEFA', 'Italy': 'UEFA', 'Spain': 'UEFA',
    'England': 'UEFA', 'Netherlands': 'UEFA', 'Portugal': 'UEFA',
    'Belgium': 'UEFA', 'Croatia': 'UEFA', 'Poland': 'UEFA', 'Denmark': 'UEFA',
    'Sweden': 'UEFA', 'Switzerland': 'UEFA', 'Austria': 'UEFA',
    'Czech Republic': 'UEFA', 'Slovakia': 'UEFA', 'Hungary': 'UEFA',
    'Romania': 'UEFA', 'Serbia': 'UEFA', 'Russia': 'UEFA', 'Ukraine': 'UEFA',
    'Turkey': 'UEFA', 'Greece': 'UEFA', 'Scotland': 'UEFA', 'Wales': 'UEFA',
    'Norway': 'UEFA', 'Republic of Ireland': 'UEFA', 'Slovenia': 'UEFA',
    'Bulgaria': 'UEFA', 'North Macedonia': 'UEFA', 'Albania': 'UEFA',
    'Yugoslavia': 'UEFA',
    'Mexico': 'CONCACAF', 'United States': 'CONCACAF', 'Costa Rica': 'CONCACAF',
    'Honduras': 'CONCACAF', 'Jamaica': 'CONCACAF', 'Panama': 'CONCACAF',
    'Canada': 'CONCACAF', 'El Salvador': 'CONCACAF',
    'Trinidad and Tobago': 'CONCACAF', 'Cuba': 'CONCACAF',
    'Nigeria': 'CAF', 'Cameroon': 'CAF', 'Senegal': 'CAF', 'Ghana': 'CAF',
    'Ivory Coast': 'CAF', 'Morocco': 'CAF', 'Tunisia': 'CAF', 'Algeria': 'CAF',
    'Egypt': 'CAF', 'South Africa': 'CAF', 'Angola': 'CAF', 'Togo': 'CAF',
    'Zambia': 'CAF', 'Zimbabwe': 'CAF',
    'Japan': 'AFC', 'South Korea': 'AFC', 'Iran': 'AFC', 'Saudi Arabia': 'AFC',
    'Australia': 'AFC', 'China PR': 'AFC', 'Iraq': 'AFC', 'Qatar': 'AFC',
    'United Arab Emirates': 'AFC', 'North Korea': 'AFC',
    'New Zealand': 'OFC',
}
CANDIDATES_2026 = [
    'Germany', 'France', 'Spain', 'England', 'Portugal', 'Netherlands',
    'Belgium', 'Croatia', 'Denmark', 'Poland', 'Switzerland', 'Austria',
    'Serbia', 'Turkey', 'Romania', 'Scotland',
    'Brazil', 'Argentina', 'Uruguay', 'Colombia', 'Ecuador', 'Chile',
    'United States', 'Canada', 'Mexico', 'Costa Rica', 'Honduras', 'Panama',
    'Morocco', 'Senegal', 'Nigeria', 'Cameroon', 'South Africa',
    'Japan', 'South Korea', 'Iran', 'Australia', 'Saudi Arabia',
    'New Zealand',
]


def k_factor(tournament):
    t = str(tournament).lower()
    if 'world cup' in t and 'qualif' not in t: return 60
    if any(x in t for x in ['euro', 'copa america', 'africa cup', 'asian cup', 'gold cup', 'nations']): return 50
    if 'qualif' in t: return 40
    return 30


def zscore(x):
    s = x.std()
    return (x - x.mean()) / (s if s > 1e-9 else 1e-9)


# ── PIPELINE FUNCTIONS ────────────────────────────────────────────────

def load_data():
    """Load or download international match results. Returns raw DataFrame."""
    os.makedirs(OUTPUTS, exist_ok=True)
    if os.path.exists(CACHE):
        raw = pd.read_csv(CACHE)
        print(f'  Loaded from cache: {CACHE}')
    else:
        url = 'https://raw.githubusercontent.com/martj42/international_results/master/results.csv'
        raw = pd.read_csv(url)
        raw.to_csv(CACHE, index=False)
        print(f'  Downloaded and cached: {len(raw):,} matches')
    raw['date'] = pd.to_datetime(raw['date'])
    print(f'  {len(raw):,} matches  |  {raw["date"].min().year}-{raw["date"].max().year}')
    print(f'  {raw["home_team"].nunique()} unique teams')
    return raw


def compute_elo(raw):
    """
    Rolling ELO ratings from all historical matches.
    Returns (elo_df, elo_at, form_stats) where elo_at and form_stats
    are closures capturing elo_df and raw for downstream lookup.
    """
    print('\n[ELO] Computing rolling ELO ratings...')
    elo  = {}
    hist = []

    for _, row in raw.sort_values('date').iterrows():
        h = row['home_team']
        a = row['away_team']
        if h not in elo: elo[h] = 1500
        if a not in elo: elo[a] = 1500

        exp_h = 1 / (1 + 10 ** ((elo[a] - elo[h]) / 400))
        exp_a = 1 - exp_h

        hs, as_ = row['home_score'], row['away_score']
        if   hs > as_: act_h, act_a = 1.0, 0.0
        elif hs < as_: act_h, act_a = 0.0, 1.0
        else:          act_h, act_a = 0.5, 0.5

        k = k_factor(row.get('tournament', ''))
        hist.append({'date': row['date'], 'team': h, 'elo': elo[h]})
        hist.append({'date': row['date'], 'team': a, 'elo': elo[a]})
        elo[h] += k * (act_h - exp_h)
        elo[a] += k * (act_a - exp_a)

    elo_df = pd.DataFrame(hist)
    print(f'  ELO tracked for {elo_df["team"].nunique()} teams')

    def elo_at(team, date):
        sub = elo_df[(elo_df['team'] == team) & (elo_df['date'] < date)]
        return float(sub['elo'].iloc[-1]) if len(sub) else 1500.0

    def form_stats(team, before_date, months=12):
        cutoff = before_date - pd.DateOffset(months=months)
        m = raw[
            ((raw['home_team'] == team) | (raw['away_team'] == team)) &
            (raw['date'] >= cutoff) & (raw['date'] < before_date)
        ]
        m = m[m['home_score'].notna() & m['away_score'].notna()]
        if len(m) == 0:
            return 0.5, 1.5, 1.5
        wins = gf = ga = 0
        for _, r in m.iterrows():
            if r['home_team'] == team:
                gf += r['home_score']; ga += r['away_score']
                wins += int(r['home_score'] > r['away_score'])
            else:
                gf += r['away_score']; ga += r['home_score']
                wins += int(r['away_score'] > r['home_score'])
        n = len(m)
        return wins / n, gf / n, ga / n

    return elo_df, elo_at, form_stats


def build_wc_history(raw):
    """
    All-time WC participation records from 1930 onward.
    Returns (wc_hist_teams, wc_hist_matches, wc_all_winners_norm).
    """
    print('\n[WC History] Building all-time participation records...')
    wc_all = raw[raw['tournament'] == 'FIFA World Cup'].copy()
    wc_hist_teams   = {}
    wc_hist_matches = {}

    for yr in WC_ALL_YEARS:
        start = pd.to_datetime(WC_ALL_STARTS[yr])
        end   = start + pd.DateOffset(days=70)
        yr_m  = wc_all[(wc_all['date'] >= start) & (wc_all['date'] <= end)]

        def norm(t): return HIST_NAME_MAP.get(t, t)
        teams_raw = set(yr_m['home_team'].tolist() + yr_m['away_team'].tolist())
        wc_hist_teams[yr] = {norm(t) for t in teams_raw}
        mc = {}
        for _, r in yr_m.iterrows():
            h, a = norm(r['home_team']), norm(r['away_team'])
            mc[h] = mc.get(h, 0) + 1
            mc[a] = mc.get(a, 0) + 1
        wc_hist_matches[yr] = mc

    wc_all_winners_norm = {yr: HIST_NAME_MAP.get(w, w) for yr, w in WC_ALL_WINNERS.items()}
    brazil_wins  = sum(1 for y in WC_ALL_YEARS if wc_all_winners_norm.get(y) == 'Brazil')
    germany_wins = sum(1 for y in WC_ALL_YEARS if wc_all_winners_norm.get(y) == 'Germany')
    print(f'  All-time WC wins  --  Brazil: {brazil_wins}  Germany: {germany_wins}')
    return wc_hist_teams, wc_hist_matches, wc_all_winners_norm


def build_feature_matrix(elo_at, form_stats, wc_hist_teams, wc_hist_matches, wc_all_winners_norm):
    """
    Build team-tournament feature DataFrame for WC_YEARS (1990-2022).
    Returns (df, le) where le is the fitted LabelEncoder for continent.
    """
    print('\n[Features] Building team-tournament feature matrix...')
    _decay = np.log(2) / 16
    rows = []

    for yr in WC_YEARS:
        start  = pd.to_datetime(WC_START[yr])
        winner = WC_WINNER[yr]
        host   = WC_HOST[yr]
        teams  = wc_hist_teams.get(yr, set())

        prev_yrs = [y for y in WC_YEARS if y < yr]
        prev_yr  = prev_yrs[-1] if prev_yrs else None

        print(f'  {yr}: {len(teams)} teams found')

        for team in teams:
            apps = sum(1 for y in WC_ALL_YEARS if y < yr and team in wc_hist_teams.get(y, set()))
            wins = sum(1 for y in WC_ALL_YEARS if y < yr and wc_all_winners_norm.get(y) == team)

            last3 = [y for y in WC_ALL_YEARS if y < yr][-3:]
            wins_last3 = sum(1 for y in last3 if wc_all_winners_norm.get(y) == team)

            wins_weighted = sum(
                np.exp(-_decay * (yr - y))
                for y in WC_ALL_YEARS
                if y < yr and wc_all_winners_norm.get(y) == team
            )
            prev_depth = wc_hist_matches.get(prev_yr, {}).get(team, 0) if prev_yr else 0
            wr, gf, ga = form_stats(team, start)

            rows.append({
                'year':             yr,
                'team':             team,
                'elo':              elo_at(team, start),
                'win_rate_12m':     wr,
                'gf_pg_12m':        gf,
                'ga_pg_12m':        ga,
                'wc_apps':          apps,
                'wc_wins':          wins,
                'wc_wins_last3':    wins_last3,
                'wc_wins_weighted': wins_weighted,
                'prev_wc_depth':    prev_depth,
                'is_host':          int(team == host),
                'continent':        CONTINENT.get(team, 'OTHER'),
                'won_wc':           int(team == winner),
            })

    df = pd.DataFrame(rows)
    le = LabelEncoder()
    df['continent_enc'] = le.fit_transform(df['continent'])
    df['elo_z']     = df.groupby('year')['elo'].transform(zscore)
    df['wc_wins_z'] = df.groupby('year')['wc_wins'].transform(zscore)
    df['wc_apps_z'] = df.groupby('year')['wc_apps'].transform(zscore)

    print(f'\n  Total rows : {len(df)}')
    print(f'  Winners    : {df["won_wc"].sum()}  (positive rate: {df["won_wc"].mean():.1%})')
    print('\n  Winner check:')
    for yr in WC_YEARS:
        row = df[(df['year'] == yr) & (df['won_wc'] == 1)]
        if len(row):
            r = row.iloc[0]
            print(f'    {yr}: {r["team"]:15s}  wc_apps={r["wc_apps"]:2.0f}  wc_wins={r["wc_wins"]:2.0f}  elo_z={r["elo_z"]:+.2f}')
    return df, le


def train_model(df):
    """
    Train logistic regression on 1990-2018, evaluate on 2022.
    Returns (model, train, test) where test has a 'propensity' column.
    """
    print('\n[Model] Training logistic regression...')
    train = df[df['year'] <= 2018].copy()
    test  = df[df['year'] == 2022].copy()

    X_tr, y_tr = train[FEATURES].values, train['won_wc'].values
    X_te, y_te = test[FEATURES].values,  test['won_wc'].values

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('lr',     LogisticRegression(C=0.5, max_iter=1000, random_state=RANDOM_STATE)),
    ])
    model.fit(X_tr, y_tr)

    raw_tr = model.predict_proba(X_tr)[:, 1]
    raw_te = model.predict_proba(X_te)[:, 1]

    print(f'  Train AUC  : {roc_auc_score(y_tr, raw_tr):.3f}')
    print(f'  Test  AUC  : {roc_auc_score(y_te, raw_te):.3f}')
    print(f'  Train Brier: {brier_score_loss(y_tr, raw_tr):.4f}')
    print(f'  Test  Brier: {brier_score_loss(y_te, raw_te):.4f}')

    test = test.copy()
    test['propensity'] = raw_te
    print('\n  2022 field (top 12 by propensity):')
    print(test.sort_values('propensity', ascending=False).head(12)
          [['team', 'elo', 'wc_wins', 'propensity', 'won_wc']].to_string(index=False))
    return model, train, test


def compute_shap(model, train):
    """
    SHAP LinearExplainer on scaled training data.
    Returns (shap_vals, X_tr_sc).
    """
    print('\n[SHAP] Computing feature attributions...')
    scaler    = model.named_steps['scaler']
    lr        = model.named_steps['lr']
    X_tr_sc   = scaler.transform(train[FEATURES].values)
    explainer = shap.LinearExplainer(lr, X_tr_sc, feature_perturbation='interventional')
    shap_vals = explainer.shap_values(X_tr_sc)
    return shap_vals, X_tr_sc


def score_2026(elo_at, form_stats, wc_hist_teams, wc_hist_matches, wc_all_winners_norm, le, model):
    """
    Score 2026 World Cup candidates using pre-tournament features.
    Returns df_2026 sorted by propensity score.
    """
    print('\n[2026] Scoring candidates...')
    _decay   = np.log(2) / 16
    last3_26 = [y for y in WC_ALL_YEARS if y < 2026][-3:]  # 2014, 2018, 2022

    rows = []
    for team in CANDIDATES_2026:
        apps = sum(1 for y in WC_ALL_YEARS if team in wc_hist_teams.get(y, set()))
        wins = sum(1 for y in WC_ALL_YEARS if wc_all_winners_norm.get(y) == team)
        wins_last3 = sum(1 for y in last3_26 if wc_all_winners_norm.get(y) == team)
        wins_weighted = sum(
            np.exp(-_decay * (2026 - y))
            for y in WC_ALL_YEARS if wc_all_winners_norm.get(y) == team
        )
        prev_depth = wc_hist_matches.get(2022, {}).get(team, 0)
        wr, gf, ga = form_stats(team, CUTOFF_2026)
        rows.append({
            'team':             team,
            'elo':              elo_at(team, CUTOFF_2026),
            'win_rate_12m':     wr,
            'gf_pg_12m':        gf,
            'ga_pg_12m':        ga,
            'wc_apps':          apps,
            'wc_wins':          wins,
            'wc_wins_last3':    wins_last3,
            'wc_wins_weighted': wins_weighted,
            'prev_wc_depth':    prev_depth,
            'is_host':          int(team in HOST_2026),
            'continent':        CONTINENT.get(team, 'OTHER'),
        })

    df_2026 = pd.DataFrame(rows)
    df_2026['continent_enc'] = df_2026['continent'].apply(
        lambda c: int(le.transform([c])[0]) if c in le.classes_ else 0
    )
    for col, zcol in [('elo', 'elo_z'), ('wc_wins', 'wc_wins_z'), ('wc_apps', 'wc_apps_z')]:
        s = df_2026[col]
        df_2026[zcol] = (s - s.mean()) / max(s.std(), 1e-9)
    df_2026[FEATURES] = df_2026[FEATURES].fillna(0)

    df_2026['propensity'] = model.predict_proba(df_2026[FEATURES].values)[:, 1]
    df_2026 = df_2026.sort_values('propensity', ascending=False).reset_index(drop=True)
    print(df_2026[['team', 'elo', 'elo_z', 'wc_wins', 'prev_wc_depth', 'propensity']]
          .head(20).to_string(index=False))
    return df_2026


def generate_plots(test, df_2026, shap_vals, X_tr_sc, model, df, show=False):
    """
    Generate and save all 4 output plots.
    Pass show=True when running in a notebook for inline display.
    """
    import matplotlib.pyplot as plt
    sns.set_theme(style='whitegrid')

    def _finish(path):
        plt.savefig(path, dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
        print(f'  Saved {path}')

    # -- 2022 propensity scores ------------------------------------------------
    top16  = test.sort_values('propensity', ascending=False).head(16).reset_index(drop=True)
    colors = [RED if w else BLUE for w in top16['won_wc']]
    fig, ax = plt.subplots(figsize=(10, 7), facecolor=BG)
    ax.set_facecolor(BG)
    bars = ax.barh(top16['team'][::-1], top16['propensity'][::-1],
                   color=colors[::-1], alpha=0.82, edgecolor='none')
    ax.axvline(1/32, color='black', lw=1.2, ls='--', alpha=0.5, label='Random (1/32)')
    for bar, val in zip(bars, top16['propensity'][::-1]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.1%}', va='center', fontsize=9)
    ax.set_xlabel('Winner Propensity Score', fontsize=11)
    ax.set_title('2022 World Cup: Who Had the Winner Profile?\nRed = actual winner (Argentina)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    plt.tight_layout()
    _finish(OUTPUTS + 'propensity_scores_2022.png')

    # -- SHAP beeswarm --------------------------------------------------------
    shap.summary_plot(shap_vals, X_tr_sc, feature_names=FEATURES,
                      plot_type='dot', show=False, max_display=8)
    plt.gcf().set_facecolor(BG)
    plt.title('What drives the World Cup winner profile?', fontsize=12, fontweight='bold')
    plt.tight_layout()
    _finish(OUTPUTS + 'shap_beeswarm.png')

    # -- 2026 candidates ------------------------------------------------------
    top_2026  = df_2026.head(20).reset_index(drop=True)
    colors_26 = [RED if t in HOST_2026 else BLUE for t in top_2026['team']]
    fig, ax = plt.subplots(figsize=(10, 7), facecolor=BG)
    ax.set_facecolor(BG)
    bars = ax.barh(top_2026['team'][::-1], top_2026['propensity'][::-1],
                   color=colors_26[::-1], alpha=0.82, edgecolor='none')
    ax.axvline(1/48, color='black', lw=1.2, ls='--', alpha=0.5, label='Random (1/48)')
    for bar, val in zip(bars, top_2026['propensity'][::-1]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.1%}', va='center', fontsize=9)
    ax.set_xlabel('Winner Propensity Score', fontsize=11)
    ax.set_title('2026 World Cup: Which Teams Have the Winner Profile?\nRed = host nations',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    plt.tight_layout()
    _finish(OUTPUTS + 'propensity_scores_2026.png')

    # -- Calibration curve ----------------------------------------------------
    all_probs  = model.predict_proba(df[FEATURES].values)[:, 1]
    all_labels = df['won_wc'].values
    frac_pos, mean_pred = calibration_curve(all_labels, all_probs, n_bins=8, strategy='quantile')
    fig, ax = plt.subplots(figsize=(7, 6), facecolor=BG)
    ax.set_facecolor(BG)
    ax.plot(mean_pred, frac_pos, 'o-', color=BLUE, lw=2, markersize=7, label='Model')
    ax.plot([0, 1], [0, 1], 'k--', lw=1.2, alpha=0.5, label='Perfect calibration')
    ax.set_xlabel('Mean Predicted Probability', fontsize=11)
    ax.set_ylabel('Fraction of Positives', fontsize=11)
    ax.set_title('Calibration Curve', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    plt.tight_layout()
    _finish(OUTPUTS + 'calibration_curve.png')

    print('\n[Done] All outputs in', OUTPUTS)


# ── MAIN ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')

    print('=' * 60)
    print('Model 06 -- World Cup Winner Propensity Model')
    print('=' * 60)

    print('\n[Data] Loading international match results...')
    raw = load_data()
    elo_df, elo_at, form_stats = compute_elo(raw)
    wc_hist_teams, wc_hist_matches, wc_all_winners_norm = build_wc_history(raw)
    df, le = build_feature_matrix(elo_at, form_stats, wc_hist_teams, wc_hist_matches, wc_all_winners_norm)
    model, train, test = train_model(df)
    shap_vals, X_tr_sc = compute_shap(model, train)
    df_2026 = score_2026(elo_at, form_stats, wc_hist_teams, wc_hist_matches, wc_all_winners_norm, le, model)
    print('\n[Plots] Generating visualizations...')
    generate_plots(test, df_2026, shap_vals, X_tr_sc, model, df, show=False)
