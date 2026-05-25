import time, warnings, os, sys
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import f_oneway, kruskal, pearsonr, spearmanr

warnings.filterwarnings('ignore')
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    try: sys.stdout.reconfigure(encoding='utf-8')
    except: pass

OUTPUTS_DIR = 'outputs/'
os.makedirs(OUTPUTS_DIR, exist_ok=True)
MAX_CAREER = 15
MIN_GP     = 20
HIGHLIGHT  = [2003, 1996, 2001, 2008, 1999]
PICK_ORDER  = ['Top 5', 'Lottery (6-14)', 'Late 1st (15-30)', '2nd Round']
PICK_COLORS = {
    'Top 5':            '#9B3D36',
    'Lottery (6-14)':   '#4C72B0',
    'Late 1st (15-30)': '#55A868',
    '2nd Round':        '#DD8452',
}


def _pick_bucket(p):
    if p <= 5:  return 'Top 5'
    if p <= 14: return 'Lottery (6-14)'
    if p <= 30: return 'Late 1st (15-30)'
    return '2nd Round'


def load_data():
    print('Loading draft history...')
    from nba_api.stats.endpoints import drafthistory
    time.sleep(1)
    dh_raw = drafthistory.DraftHistory(league_id='00').get_data_frames()[0]
    dh_raw['DRAFT_YEAR'] = dh_raw['SEASON'].astype(int)
    draft = dh_raw[dh_raw['DRAFT_YEAR'].between(1996, 2018)][
        ['PERSON_ID', 'PLAYER_NAME', 'DRAFT_YEAR', 'ROUND_NUMBER', 'OVERALL_PICK']
    ].copy()
    draft['PERSON_ID']    = draft['PERSON_ID'].astype('int64')
    draft['ROUND_NUMBER'] = pd.to_numeric(draft['ROUND_NUMBER'], errors='coerce').fillna(3).astype(int)
    draft['OVERALL_PICK'] = pd.to_numeric(draft['OVERALL_PICK'], errors='coerce').fillna(99).astype(int)
    cohort_sizes = draft.groupby('DRAFT_YEAR').size().rename('COHORT_SIZE')
    print(f'  Draft picks 1996-2018: {len(draft)} | Classes: {cohort_sizes.shape[0]} | Avg size: {cohort_sizes.mean():.0f}')

    print('Loading season stats from cache...')
    all_df = pd.read_csv(OUTPUTS_DIR + 'raw_season_stats.csv')
    all_df['PLAYER_ID']    = all_df['PLAYER_ID'].astype('int64')
    all_df['SEASON_START'] = all_df['SEASON_START'].astype(int)
    print(f'  {len(all_df):,} rows | {all_df["SEASON"].nunique()} seasons')

    return draft, cohort_sizes, all_df


def build_cohorts(all_df, draft, cohort_sizes):
    print('Building cohorts...')
    merged = all_df.merge(
        draft[['PERSON_ID', 'DRAFT_YEAR', 'ROUND_NUMBER', 'OVERALL_PICK']],
        left_on='PLAYER_ID', right_on='PERSON_ID', how='inner'
    )
    merged['CAREER_YEAR'] = merged['SEASON_START'] - merged['DRAFT_YEAR'] + 1
    merged = merged[
        (merged['CAREER_YEAR'] >= 1) &
        (merged['CAREER_YEAR'] <= MAX_CAREER) &
        (merged['GP'] >= MIN_GP)
    ].copy()

    merged['PICK_BUCKET'] = merged['OVERALL_PICK'].apply(_pick_bucket)
    for col in ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV']:
        merged[col + '_PG'] = merged[col] / merged['GP']
    merged['PERF'] = (
        merged['PTS_PG'] + 1.2 * merged['REB_PG'] + 1.5 * merged['AST_PG']
        + merged['STL_PG'] + 1.5 * merged['BLK_PG'] - merged['TOV_PG']
    ).round(3)
    print(f'  {len(merged):,} player-seasons | {merged["DRAFT_YEAR"].nunique()} classes | {merged["PLAYER_ID"].nunique():,} players')
    return merged


def build_matrices(merged, cohort_sizes):
    print('Building cohort matrices...')
    active_counts = (
        merged.groupby(['DRAFT_YEAR', 'CAREER_YEAR'])['PLAYER_ID']
        .nunique().reset_index(name='N_ACTIVE')
    )
    active_counts = active_counts.merge(cohort_sizes.reset_index(), on='DRAFT_YEAR')
    active_counts['PCT_ACTIVE'] = (active_counts['N_ACTIVE'] / active_counts['COHORT_SIZE'] * 100).round(1)
    survival_matrix = active_counts.pivot(
        index='DRAFT_YEAR', columns='CAREER_YEAR', values='PCT_ACTIVE'
    ).sort_index()

    value_stats = (
        merged.groupby(['DRAFT_YEAR', 'CAREER_YEAR'])['PERF']
        .mean().reset_index(name='AVG_PERF')
    )
    value_matrix = value_stats.pivot(
        index='DRAFT_YEAR', columns='CAREER_YEAR', values='AVG_PERF'
    ).sort_index().round(2)

    ltv_df = (
        merged.groupby('DRAFT_YEAR').agg(
            N_PLAYERS=('PLAYER_ID', 'nunique'),
            TOTAL_PERF=('PERF', 'sum'),
            AVG_CAREER_YRS=('CAREER_YEAR', 'mean'),
        ).reset_index()
    )
    ltv_df['LTV_PER_PLAYER']    = (ltv_df['TOTAL_PERF'] / ltv_df['N_PLAYERS']).round(2)
    ltv_df['LTV_PER_PLAYER_YR'] = (ltv_df['TOTAL_PERF'] / (ltv_df['N_PLAYERS'] * ltv_df['AVG_CAREER_YRS'])).round(3)
    ltv_df['CAREER_COMPLETE']   = ltv_df['DRAFT_YEAR'] <= 2014
    ltv_df = ltv_df.sort_values('DRAFT_YEAR').reset_index(drop=True)

    survival_matrix.to_csv(OUTPUTS_DIR + 'cohort_matrix.csv')
    ltv_df.to_csv(OUTPUTS_DIR + 'draft_class_ltv.csv', index=False)
    print(f'  Survival matrix: {survival_matrix.shape}')

    complete = ltv_df[ltv_df['CAREER_COMPLETE']]
    return survival_matrix, value_matrix, ltv_df, complete


def run_stats(merged, ltv_df):
    print('Running statistical tests...')
    groups = [
        merged[merged['DRAFT_YEAR'] == yr]['PERF'].dropna().values
        for yr in sorted(merged['DRAFT_YEAR'].unique())
        if len(merged[merged['DRAFT_YEAR'] == yr]) >= 15
    ]
    f_stat,  p_anova   = f_oneway(*groups)
    h_stat,  p_kruskal = kruskal(*groups)
    r_p, p_p           = pearsonr(ltv_df['DRAFT_YEAR'], ltv_df['LTV_PER_PLAYER'])
    r_s, p_s           = spearmanr(ltv_df['DRAFT_YEAR'], ltv_df['LTV_PER_PLAYER'])
    print(f'  ANOVA:      F={f_stat:.2f}  p={p_anova:.6f}')
    print(f'  Kruskal-W:  H={h_stat:.2f}  p={p_kruskal:.6f}')
    print(f'  LTV trend:  Pearson r={r_p:.3f} p={p_p:.4f} | Spearman r={r_s:.3f} p={p_s:.4f}')
    return f_stat, p_anova, h_stat, p_kruskal, r_p, p_p, r_s, p_s


def compute_draft_survival(merged):
    print('Computing draft position survival...')
    reach_yr10 = {}
    for bucket in PICK_ORDER:
        sub    = merged[merged['PICK_BUCKET'] == bucket]
        total  = sub['PLAYER_ID'].nunique()
        active = sub[sub['CAREER_YEAR'] >= 10]['PLAYER_ID'].nunique()
        reach_yr10[bucket] = round(active / total * 100, 1) if total > 0 else 0
    return reach_yr10


def compute_top_players(merged, complete):
    print('Computing top players per class...')
    player_career = (
        merged.groupby(['PLAYER_ID', 'PLAYER_NAME', 'DRAFT_YEAR'])['PERF']
        .sum().reset_index(name='CAREER_PERF')
        .sort_values('CAREER_PERF', ascending=False)
    )
    top3 = (
        player_career.groupby('DRAFT_YEAR').head(3)
        .sort_values(['DRAFT_YEAR', 'CAREER_PERF'], ascending=[True, False])
    )
    top5_classes = complete.nlargest(5, 'LTV_PER_PLAYER_YR')['DRAFT_YEAR'].tolist()
    bot5_classes = complete.nsmallest(5, 'LTV_PER_PLAYER_YR')['DRAFT_YEAR'].tolist()

    print('\n=== Top 5 Draft Classes by LTV (career-complete only) ===')
    for yr in sorted(top5_classes):
        val   = complete.loc[complete['DRAFT_YEAR'] == yr, 'LTV_PER_PLAYER_YR'].values[0]
        names = top3[top3['DRAFT_YEAR'] == yr]['PLAYER_NAME'].tolist()
        print(f'  {yr}  LTV={val:.1f}  |  {" / ".join(names)}')

    print('\n=== Bottom 5 Draft Classes by LTV (career-complete only) ===')
    for yr in sorted(bot5_classes):
        val   = complete.loc[complete['DRAFT_YEAR'] == yr, 'LTV_PER_PLAYER_YR'].values[0]
        names = top3[top3['DRAFT_YEAR'] == yr]['PLAYER_NAME'].tolist()
        print(f'  {yr}  LTV={val:.1f}  |  {" / ".join(names)}')

    print('\n=== 2003 Class Top 10 ===')
    c2003 = player_career[player_career['DRAFT_YEAR'] == 2003].head(10)
    for _, row in c2003.iterrows():
        print(f'  {row["PLAYER_NAME"]:<26} {row["CAREER_PERF"]:.1f}')

    return player_career, top3


def generate_plots(merged, survival_matrix, value_matrix, ltv_df, reach_yr10, show=False):
    import matplotlib.pyplot as plt

    print('Plotting cohort heatmap...')
    fig, ax = plt.subplots(figsize=(18, 10))
    sns.heatmap(
        survival_matrix, cmap='YlOrRd', annot=True, fmt='.0f',
        linewidths=0.4, linecolor='white', ax=ax,
        cbar_kws={'label': '% of draft class still active (>=20 GP)', 'shrink': 0.55},
        annot_kws={'size': 7.5}, vmin=0, vmax=80
    )
    ax.set_title(
        'NBA Draft Class Retention Grid (1996-2018)\n'
        'Each cell = % of that draft class still playing >=20 games in career year N\n'
        'Identical format to a SaaS monthly cohort retention table',
        fontsize=12, fontweight='bold', pad=15
    )
    ax.set_xlabel('Career Year  (1 = Rookie Season)', fontsize=11)
    ax.set_ylabel('Draft Year  (Cohort)', fontsize=11)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR + 'cohort_heatmap.png', dpi=150, bbox_inches='tight')
    if show: plt.show()
    else: plt.close()
    print('  Saved: cohort_heatmap.png')

    print('Plotting LTV chart...')
    ltv_sorted  = ltv_df.sort_values('LTV_PER_PLAYER_YR', ascending=True).reset_index(drop=True)
    median_norm = ltv_df['LTV_PER_PLAYER_YR'].median()
    colors_bar  = [
        '#9B3D36' if yr in HIGHLIGHT
        else '#CCAA55' if not ltv_sorted.loc[ltv_sorted['DRAFT_YEAR'] == yr, 'CAREER_COMPLETE'].values[0]
        else '#4C72B0' if ltv_sorted.loc[ltv_sorted['DRAFT_YEAR'] == yr, 'LTV_PER_PLAYER_YR'].values[0] >= median_norm
        else '#B0B8C4'
        for yr in ltv_sorted['DRAFT_YEAR']
    ]
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    fig.suptitle('NBA Draft Class LTV Analysis (1996-2018)', fontsize=13, fontweight='bold')

    ax = axes[0]
    bars = ax.barh(ltv_sorted['DRAFT_YEAR'].astype(str), ltv_sorted['LTV_PER_PLAYER_YR'],
                   color=colors_bar, alpha=0.88)
    ax.axvline(median_norm, color='black', linestyle='--', linewidth=1.2, alpha=0.5, label='Median')
    ax.set_title('LTV Per Player Per Career Year\n(Normalized for career completeness — gold = still active)', fontsize=10)
    ax.set_xlabel('Avg Performance Score Per Player Per Year')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25, axis='x')
    for bar, val in zip(bars, ltv_sorted['LTV_PER_PLAYER_YR']):
        ax.text(val * 1.005, bar.get_y() + bar.get_height() / 2, f'{val:.2f}', va='center', fontsize=7)

    complete = ltv_df[ltv_df['CAREER_COMPLETE']]
    ax2 = axes[1]
    sc_colors = ['#9B3D36' if yr in HIGHLIGHT else '#4C72B0' for yr in complete['DRAFT_YEAR']]
    ax2.scatter(complete['DRAFT_YEAR'], complete['LTV_PER_PLAYER_YR'], color=sc_colors, s=65, zorder=3, alpha=0.85)
    m, b = np.polyfit(complete['DRAFT_YEAR'], complete['LTV_PER_PLAYER_YR'], 1)
    x_line = np.linspace(complete['DRAFT_YEAR'].min(), complete['DRAFT_YEAR'].max(), 100)
    ax2.plot(x_line, m * x_line + b, color='#DD8452', linestyle='--', linewidth=1.5, alpha=0.8, label='Trend')
    for _, row in complete.iterrows():
        if int(row['DRAFT_YEAR']) in HIGHLIGHT:
            ax2.annotate(str(int(row['DRAFT_YEAR'])),
                         (row['DRAFT_YEAR'], row['LTV_PER_PLAYER_YR']),
                         textcoords='offset points', xytext=(5, 3), fontsize=8)
    ax2.set_title('LTV Per Player Per Year — Career-Complete Classes (1996-2014)', fontsize=10)
    ax2.set_xlabel('Draft Year')
    ax2.set_ylabel('LTV Per Player Per Year')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR + 'draft_class_ltv.png', dpi=150, bbox_inches='tight')
    if show: plt.show()
    else: plt.close()
    print('  Saved: draft_class_ltv.png')

    print('Plotting draft position survival...')
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('Career Survival by Draft Position', fontsize=13, fontweight='bold')

    ax = axes[0]
    for bucket in PICK_ORDER:
        sub   = merged[merged['PICK_BUCKET'] == bucket]
        total = sub['PLAYER_ID'].nunique()
        if total == 0: continue
        survival = sub.groupby('CAREER_YEAR')['PLAYER_ID'].nunique() / total * 100
        ax.plot(survival.index, survival.values, marker='o', markersize=4, linewidth=2,
                color=PICK_COLORS[bucket], label=f'{bucket} (n={total})', alpha=0.85)
    ax.set_title('% of group still active by career year', fontsize=10)
    ax.set_xlabel('Career Year')
    ax.set_ylabel('% Still Active (>=20 GP)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.bar(reach_yr10.keys(), reach_yr10.values(),
            color=[PICK_COLORS[b] for b in reach_yr10], alpha=0.85, width=0.5)
    ax2.set_title('% Reaching Career Year 10', fontsize=10)
    ax2.set_ylabel('%')
    ax2.grid(True, alpha=0.3, axis='y')
    for i, (bucket, val) in enumerate(reach_yr10.items()):
        ax2.text(i, val + 0.5, f'{val}%', ha='center', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR + 'survival_by_draft_position.png', dpi=150, bbox_inches='tight')
    if show: plt.show()
    else: plt.close()
    print('  Saved: survival_by_draft_position.png')

    print('Plotting value heatmap...')
    fig, ax = plt.subplots(figsize=(18, 10))
    sns.heatmap(
        value_matrix, cmap='Blues', annot=True, fmt='.1f',
        linewidths=0.4, linecolor='white', ax=ax,
        cbar_kws={
            'label': 'Avg Perf Score/game (PTS+1.2*REB+1.5*AST+STL+1.5*BLK-TOV)',
            'shrink': 0.55
        },
        annot_kws={'size': 7.5}
    )
    ax.set_title(
        'NBA Draft Class — Average Player Performance by Career Year (1996-2018)\n'
        'Darker = higher avg contribution among players still active that year',
        fontsize=12, fontweight='bold', pad=15
    )
    ax.set_xlabel('Career Year', fontsize=11)
    ax.set_ylabel('Draft Year (Cohort)', fontsize=11)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR + 'value_heatmap.png', dpi=150, bbox_inches='tight')
    if show: plt.show()
    else: plt.close()
    print('  Saved: value_heatmap.png')


def print_findings(merged, ltv_df, complete, f_stat, p_anova, h_stat, p_kruskal,
                   r_p, p_p, r_s, p_s, reach_yr10, survival_matrix):
    best  = complete.nlargest(1,  'LTV_PER_PLAYER_YR').iloc[0]
    worst = complete.nsmallest(1, 'LTV_PER_PLAYER_YR').iloc[0]

    print('\n' + '=' * 60)
    print('KEY FINDINGS — NBA Draft Cohort Analysis 1996-2018')
    print('=' * 60)
    print(f'Draft classes:            {ltv_df["DRAFT_YEAR"].nunique()} (1996-2018)')
    print(f'Career-complete (<=2014): {complete["DRAFT_YEAR"].nunique()} classes')
    print(f'Player-seasons:           {len(merged):,}')
    print(f'Unique players:           {merged["PLAYER_ID"].nunique():,}')
    print(f'Best class (normalized):  {int(best.DRAFT_YEAR)}  LTV/yr={best.LTV_PER_PLAYER_YR:.2f}')
    print(f'Worst class (normalized): {int(worst.DRAFT_YEAR)}  LTV/yr={worst.LTV_PER_PLAYER_YR:.2f}')
    print(f'Gap best/worst:           {best.LTV_PER_PLAYER_YR / worst.LTV_PER_PLAYER_YR:.1f}x')
    print(f'ANOVA:                    F={f_stat:.2f}  p={p_anova:.6f}  -> {"SIGNIFICANT" if p_anova < 0.05 else "not sig"}')
    print(f'Kruskal-Wallis:           H={h_stat:.2f}  p={p_kruskal:.6f}')
    print(f'LTV trend Pearson:        r={r_p:.3f}  p={p_p:.4f}')
    print(f'LTV trend Spearman:       r={r_s:.3f}  p={p_s:.4f}')
    print('\nDraft position -> % reaching career year 10:')
    for bucket, val in reach_yr10.items():
        print(f'  {bucket:<22} {val:.1f}%')
    print('\nAvg cohort survival:')
    for yr in [3, 5, 8, 10]:
        if yr in survival_matrix.columns:
            print(f'  Career Year {yr:>2}: {survival_matrix[yr].mean():.1f}% still active')
    print('\nAll outputs saved.')


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    draft, cohort_sizes, all_df = load_data()
    merged = build_cohorts(all_df, draft, cohort_sizes)
    survival_matrix, value_matrix, ltv_df, complete = build_matrices(merged, cohort_sizes)
    f_stat, p_anova, h_stat, p_kruskal, r_p, p_p, r_s, p_s = run_stats(merged, ltv_df)
    reach_yr10 = compute_draft_survival(merged)
    player_career, top3 = compute_top_players(merged, complete)
    generate_plots(merged, survival_matrix, value_matrix, ltv_df, reach_yr10, show=False)
    print_findings(merged, ltv_df, complete, f_stat, p_anova, h_stat, p_kruskal,
                   r_p, p_p, r_s, p_s, reach_yr10, survival_matrix)
