"""
Model 05 — MLB Player LTV:CAC Analysis
Which players produced more value than they cost, and when does the
average player pay back their contract?
"""

import os
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from pybaseball import bwar_bat, bwar_pitch, amateur_draft

warnings.filterwarnings('ignore')

OUTPUTS_DIR  = 'outputs/'
os.makedirs(OUTPUTS_DIR, exist_ok=True)

WAR_DOLLARS  = 8_000_000
DEBUT_START  = 1990
DEBUT_END    = 2015
MIN_PA       = 300
MIN_IPOUTS   = 300

YEARLY_CACHE = OUTPUTS_DIR + 'mlb_yearly.csv'
DRAFT_CACHE  = OUTPUTS_DIR + 'mlb_draft_r1.csv'

BG   = '#f5f0eb'
RED  = '#9b3d36'
BLUE = '#4c72b0'

LEAGUE_MIN = {
    1990: 100_000, 1991: 100_000, 1992: 109_000, 1993: 109_000, 1994: 109_000,
    1995: 109_000, 1996: 150_000, 1997: 150_000, 1998: 170_000, 1999: 200_000,
    2000: 200_000, 2001: 200_000, 2002: 200_000, 2003: 300_000, 2004: 300_000,
    2005: 316_000, 2006: 327_000, 2007: 380_000, 2008: 390_000, 2009: 400_000,
    2010: 400_000, 2011: 414_000, 2012: 414_000, 2013: 490_000, 2014: 500_000,
    2015: 507_500, 2016: 507_500, 2017: 535_000, 2018: 545_000, 2019: 555_000,
    2020: 563_500, 2021: 570_500, 2022: 700_000, 2023: 720_000,
}

TEAM_NAMES = {
    'OAK': "Oakland A's", 'MON': 'Montreal Expos', 'CLE': 'Cleveland',
    'FLA': 'Florida Marlins', 'MIN': 'Minnesota', 'CHW': 'Chicago White Sox',
    'TBD': 'Tampa Bay', 'TBR': 'Tampa Bay', 'BAL': 'Baltimore', 'PIT': 'Pittsburgh',
    'SEA': 'Seattle', 'MIL': 'Milwaukee', 'TOR': 'Toronto', 'HOU': 'Houston',
    'CIN': 'Cincinnati', 'PHI': 'Philadelphia', 'ATL': 'Atlanta', 'DET': 'Detroit',
    'STL': 'St. Louis', 'TEX': 'Texas', 'BOS': 'Boston', 'KCR': 'Kansas City',
    'SDP': 'San Diego', 'COL': 'Colorado', 'SFG': 'San Francisco',
    'NYY': 'NY Yankees', 'CHC': 'Chicago Cubs', 'LAD': 'LA Dodgers',
    'NYM': 'NY Mets', 'MIA': 'Miami', 'ARI': 'Arizona', 'WSN': 'Washington',
    'LAA': 'LA Angels',
}


def _fill_salary(df):
    df['salary_filled'] = df.apply(
        lambda r: r['salary'] if pd.notna(r['salary']) else LEAGUE_MIN.get(int(r['season']), 0),
        axis=1
    )
    return df


def load_yearly_stats():
    if os.path.exists(YEARLY_CACHE):
        print('[Cache] Loading year-by-year stats...')
        yearly = pd.read_csv(YEARLY_CACHE)
    else:
        print('[Data] Fetching Baseball Reference batting WAR...')
        bat = bwar_bat(return_all=True)
        bat = bat[['player_ID', 'name_common', 'year_ID', 'team_ID', 'age', 'PA', 'WAR', 'salary']].copy()
        bat['player_type'] = 'batter'
        bat.rename(columns={'PA': 'volume'}, inplace=True)

        print('[Data] Fetching Baseball Reference pitching WAR...')
        pit = bwar_pitch(return_all=True)
        pit = pit[['player_ID', 'name_common', 'year_ID', 'team_ID', 'age', 'IPouts', 'WAR', 'salary']].copy()
        pit['player_type'] = 'pitcher'
        pit.rename(columns={'IPouts': 'volume'}, inplace=True)

        yearly = pd.concat([bat, pit], ignore_index=True)
        yearly.rename(columns={'year_ID': 'season', 'name_common': 'Name'}, inplace=True)
        yearly = yearly[yearly['season'] >= DEBUT_START]
        yearly = _fill_salary(yearly)
        yearly.to_csv(YEARLY_CACHE, index=False)
        print(f'[Cache] Saved {len(yearly):,} player-seasons to {YEARLY_CACHE}')

    if 'salary_filled' not in yearly.columns:
        yearly = _fill_salary(yearly)
    return yearly


def aggregate_careers(yearly):
    career = (
        yearly.groupby(['player_ID', 'Name', 'player_type'])
        .agg(
            career_war    = ('WAR',           'sum'),
            career_volume = ('volume',         'sum'),
            career_salary = ('salary_filled',  'sum'),
            seasons       = ('season',         'count'),
            debut_season  = ('season',         'min'),
            final_season  = ('season',         'max'),
        )
        .reset_index()
    )
    career = career[
        (career['debut_season'] >= DEBUT_START) &
        (career['debut_season'] <= DEBUT_END)
    ]
    career = career[
        ((career['player_type'] == 'batter')  & (career['career_volume'] >= MIN_PA)) |
        ((career['player_type'] == 'pitcher') & (career['career_volume'] >= MIN_IPOUTS))
    ]
    print(f'\n[Filter] Players after debut/volume filter: {len(career):,}')

    career['ltv']     = career['career_war'] * WAR_DOLLARS
    career['cac']     = career['career_salary']
    career['ltv_cac'] = career['ltv'] / career['cac']
    merged = career[(career['ltv_cac'] > 0) & (career['ltv_cac'] <= 50)].copy()

    print(f'\n[Results] Players in analysis: {len(merged):,}')
    print(f'  Median career WAR:    {merged["career_war"].median():.1f}')
    print(f'  Median career salary: ${merged["cac"].median()/1e6:.1f}M')
    print(f'  Median LTV:           ${merged["ltv"].median()/1e6:.1f}M')
    print(f'  Median LTV:CAC:       {merged["ltv_cac"].median():.2f}x')
    print(f'  LTV > CAC (ROI+):     {(merged["ltv_cac"] > 1).mean():.0%}')
    print('\n[Results] By player type:')
    print(merged.groupby('player_type')[['career_war', 'cac', 'ltv_cac']]
          .median().round(2).rename(columns={
              'career_war': 'Median WAR', 'cac': 'Median Salary', 'ltv_cac': 'Median LTV:CAC'}))

    return career, merged


def compute_payback_curve(merged, yearly):
    print('\n[Analysis] Building payback curve...')
    valid_ids    = set(merged['player_ID'].unique())
    yearly_valid = yearly[yearly['player_ID'].isin(valid_ids)].copy()
    yearly_valid = yearly_valid.sort_values(['player_ID', 'season'])
    yearly_valid['career_year']   = yearly_valid.groupby('player_ID').cumcount() + 1
    yearly_valid['cum_war_value'] = yearly_valid.groupby('player_ID')['WAR'].cumsum() * WAR_DOLLARS
    yearly_valid['cum_salary']    = yearly_valid.groupby('player_ID')['salary_filled'].cumsum()
    yearly_valid = yearly_valid[yearly_valid['cum_salary'] > 0]
    yearly_valid['cum_ltv_cac']   = yearly_valid['cum_war_value'] / yearly_valid['cum_salary']

    payback_curve = (
        yearly_valid[yearly_valid['career_year'] <= 16]
        .groupby('career_year')
        .agg(median_cum_ltv_cac=('cum_ltv_cac', 'median'), n_players=('player_ID', 'nunique'))
        .reset_index()
    )
    print(payback_curve[['career_year', 'median_cum_ltv_cac', 'n_players']].to_string(index=False))

    crossover    = payback_curve[payback_curve['median_cum_ltv_cac'] >= 1.0]
    payback_year = int(crossover['career_year'].iloc[0]) if not crossover.empty else None
    if payback_year:
        print(f'\n  Median payback: Career Year {payback_year}')

    return payback_curve, payback_year


def analyze_draft_slots():
    if os.path.exists(DRAFT_CACHE):
        print('\n[Cache] Loading draft data...')
        draft_r1 = pd.read_csv(DRAFT_CACHE)
    else:
        print('\n[Data] Fetching Round 1 draft data (1990-2015)...')
        frames = []
        for yr in range(1990, 2016):
            try:
                d = amateur_draft(yr, 1)
                d['draft_year'] = yr
                frames.append(d)
                if yr % 5 == 0:
                    print(f'  ...through {yr}')
            except Exception as e:
                print(f'  [skip] {yr}: {e}')
        if frames:
            draft_r1 = pd.concat(frames, ignore_index=True)
            draft_r1.to_csv(DRAFT_CACHE, index=False)
            print(f'[Cache] Saved {len(draft_r1):,} draft picks')
        else:
            draft_r1 = pd.DataFrame()

    has_draft     = False
    pick_analysis = pd.DataFrame()

    if not draft_r1.empty:
        draft_r1['Name']  = draft_r1['Name'].str.strip()
        draft_r1['OvPck'] = pd.to_numeric(draft_r1['OvPck'], errors='coerce')
        draft_r1['WAR']   = pd.to_numeric(draft_r1['WAR'],   errors='coerce')
        draft_r1['Bonus'] = (draft_r1['Bonus'].astype(str)
                             .str.replace(r'[\$,]', '', regex=True).str.strip())
        draft_r1['Bonus'] = pd.to_numeric(draft_r1['Bonus'], errors='coerce')

        d30 = draft_r1[(draft_r1['OvPck'] >= 1) & (draft_r1['OvPck'] <= 30)
                       & draft_r1['WAR'].notna()].copy()

        if len(d30) >= 30:
            has_draft = True
            d30['pick_bin'] = pd.cut(
                d30['OvPck'],
                bins=[0, 5, 10, 15, 20, 30],
                labels=['Picks\n1-5', 'Picks\n6-10', 'Picks\n11-15', 'Picks\n16-20', 'Picks\n21-30']
            )
            d30['ltv_draft']     = d30['WAR'] * WAR_DOLLARS
            d30['ltv_cac_draft'] = d30['ltv_draft'] / d30['Bonus']
            pick_analysis = (
                d30.groupby('pick_bin', observed=True)
                .agg(median_war=('WAR', 'median'), median_bonus=('Bonus', 'median'),
                     median_ltv_cac=('ltv_cac_draft', 'median'), n=('Name', 'count'))
                .reset_index()
            )
            print('\n[Results] LTV:CAC by Round 1 pick slot:')
            print(pick_analysis.to_string(index=False))

    return draft_r1, pick_analysis, has_draft


def analyze_teams(yearly):
    print('\n[Analysis] Team payroll efficiency...')
    _bat = bwar_bat(return_all=True)[['player_ID', 'year_ID', 'team_ID', 'PA', 'WAR', 'salary']].copy()
    _bat.rename(columns={'PA': 'volume'}, inplace=True)
    _pit = bwar_pitch(return_all=True)[['player_ID', 'year_ID', 'team_ID', 'IPouts', 'WAR', 'salary']].copy()
    _pit.rename(columns={'IPouts': 'volume'}, inplace=True)
    _df = pd.concat([_bat, _pit], ignore_index=True)
    _df = _df[_df['year_ID'] >= 1990].copy()
    _df.rename(columns={'year_ID': 'season'}, inplace=True)
    _df['salary_filled'] = _df.apply(
        lambda r: r['salary'] if pd.notna(r['salary']) else LEAGUE_MIN.get(int(r['season']), 0), axis=1
    )

    team_yr = _df.groupby(['team_ID', 'season']).agg(
        total_war    = ('WAR',           'sum'),
        total_salary = ('salary_filled', 'sum'),
    ).reset_index()
    team_yr = team_yr[team_yr['total_salary'] > 1_000_000]
    team_yr['ltv_cac'] = (team_yr['total_war'] * WAR_DOLLARS) / team_yr['total_salary']

    moneyball_era = team_yr[(team_yr['season'] >= 2000) & (team_yr['season'] <= 2004)]
    mb_rank = (moneyball_era.groupby('team_ID')['ltv_cac']
               .mean().sort_values(ascending=False).reset_index())
    print('\n[Results] Moneyball era (2000-2004) team efficiency:')
    print(mb_rank.head(10).to_string(index=False))

    all_time = (team_yr.groupby('team_ID')
                .agg(avg_ltv_cac=('ltv_cac', 'mean'), seasons=('season', 'count'))
                .reset_index())
    all_time = all_time[all_time['seasons'] >= 15].sort_values('avg_ltv_cac', ascending=False)
    print('\n[Results] All-time team efficiency (min 15 seasons):')
    print(all_time.to_string(index=False))

    return mb_rank, all_time, team_yr


def generate_plots(merged, payback_curve, payback_year, pick_analysis, has_draft,
                   mb_rank, all_time, team_yr, show=False):
    import matplotlib.pyplot as plt
    sns.set_theme(style='whitegrid')
    print('\n[Plots] Generating visualizations...')

    fig, ax = plt.subplots(figsize=(10, 7), facecolor=BG)
    ax.set_facecolor(BG)
    ax.scatter(merged['cac'] / 1e6, merged['ltv'] / 1e6, alpha=0.28, s=14, color=BLUE, linewidths=0)
    cap = max(merged['cac'].max(), merged['ltv'].max()) / 1e6
    ax.plot([0, cap], [0, cap], 'k--', lw=1.2, label='Break even (LTV = CAC)')
    ax.set_xlabel('Career Salary — CAC ($M)', fontsize=11)
    ax.set_ylabel('Career Value (WAR x $8M) — LTV ($M)', fontsize=11)
    ax.set_title('MLB Player LTV vs CAC  |  1990-2015 Debuts', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR + 'ltv_vs_cac.png', dpi=150, bbox_inches='tight')
    if show: plt.show()
    else: plt.close()
    print('  Saved ltv_vs_cac.png')

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)
    ax.set_facecolor(BG)
    plot_data = merged[merged['ltv_cac'] <= 10]
    ax.hist(plot_data['ltv_cac'], bins=60, color=BLUE, alpha=0.75, edgecolor='none')
    ax.axvline(1.0, color=RED, lw=2.0, ls='--', label='Break even')
    ax.axvline(merged['ltv_cac'].median(), color='black', lw=1.5, ls=':',
               label=f'Median ({merged["ltv_cac"].median():.1f}x)')
    ax.set_xlabel('LTV:CAC Ratio', fontsize=11)
    ax.set_ylabel('Number of Players', fontsize=11)
    ax.set_title('Distribution of MLB Player LTV:CAC Ratios', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR + 'ltv_cac_distribution.png', dpi=150, bbox_inches='tight')
    if show: plt.show()
    else: plt.close()
    print('  Saved ltv_cac_distribution.png')

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)
    ax.set_facecolor(BG)
    ax.plot(payback_curve['career_year'], payback_curve['median_cum_ltv_cac'],
            color=BLUE, lw=2.5, marker='o', markersize=6)
    ax.axhline(1.0, color=RED, lw=1.5, ls='--', label='Break even')
    if payback_year:
        ax.axvline(payback_year, color='black', lw=1.2, ls=':', alpha=0.7,
                   label=f'Median payback: Year {payback_year}')
    ax.set_xlabel('Career Year', fontsize=11)
    ax.set_ylabel('Cumulative LTV:CAC (median)', fontsize=11)
    ax.set_title('Payback Curve: When Does a Player Pay Back Their Contract?', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR + 'payback_curve.png', dpi=150, bbox_inches='tight')
    if show: plt.show()
    else: plt.close()
    print('  Saved payback_curve.png')

    if has_draft and not pick_analysis.empty:
        fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
        ax.set_facecolor(BG)
        best_war = pick_analysis['median_war'].max()
        colors   = [RED if v == best_war else BLUE for v in pick_analysis['median_war']]
        bars     = ax.bar(pick_analysis['pick_bin'].astype(str), pick_analysis['median_war'],
                          color=colors, alpha=0.82, edgecolor='none')
        for bar, val in zip(bars, pick_analysis['median_war']):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set_xlabel('Round 1 Pick Range', fontsize=11)
        ax.set_ylabel('Median Career WAR', fontsize=11)
        ax.set_title('First Round Pick Efficiency: Career WAR by Pick Slot\n1990-2015 MLB Drafts',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(OUTPUTS_DIR + 'pick_slot_war.png', dpi=150, bbox_inches='tight')
        if show: plt.show()
        else: plt.close()
        print('  Saved pick_slot_war.png')

        pick_ltv = pick_analysis.dropna(subset=['median_ltv_cac'])
        if len(pick_ltv) >= 3:
            fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
            ax.set_facecolor(BG)
            colors2 = [RED if str(b) == 'Picks\n21-30' else BLUE for b in pick_ltv['pick_bin']]
            bars2   = ax.bar(pick_ltv['pick_bin'].astype(str), pick_ltv['median_ltv_cac'],
                             color=colors2, alpha=0.82, edgecolor='none')
            for bar, val in zip(bars2, pick_ltv['median_ltv_cac']):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f'{val:.0f}x', ha='center', va='bottom', fontsize=10, fontweight='bold')
            ax.set_xlabel('Round 1 Pick Range', fontsize=11)
            ax.set_ylabel('Median LTV:CAC (Signing Bonus as CAC)', fontsize=11)
            ax.set_title('Pick Slot ROI: LTV:CAC by Pick Range\n(LTV = WAR x $8M, CAC = Signing Bonus)',
                         fontsize=12, fontweight='bold')
            plt.tight_layout()
            plt.savefig(OUTPUTS_DIR + 'pick_slot_ltv_cac.png', dpi=150, bbox_inches='tight')
            if show: plt.show()
            else: plt.close()
            print('  Saved pick_slot_ltv_cac.png')

    mb_plot = mb_rank.head(12).copy()
    mb_plot['label'] = mb_plot['team_ID'].map(TEAM_NAMES).fillna(mb_plot['team_ID'])
    fig, ax = plt.subplots(figsize=(11, 5), facecolor=BG)
    ax.set_facecolor(BG)
    colors = [RED if t == 'OAK' else BLUE for t in mb_plot['team_ID']]
    bars   = ax.bar(mb_plot['label'], mb_plot['ltv_cac'], color=colors, alpha=0.82, edgecolor='none')
    for bar, val in zip(bars, mb_plot['ltv_cac']):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f'{val:.1f}x', ha='center', va='bottom', fontsize=9)
    plt.xticks(rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Avg Seasonal LTV:CAC', fontsize=11)
    ax.set_title('Moneyball Era (2000-2004): Payroll Efficiency by Team', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR + 'team_moneyball_era.png', dpi=150, bbox_inches='tight')
    if show: plt.show()
    else: plt.close()
    print('  Saved team_moneyball_era.png')

    top8 = all_time.head(8).copy()
    bot8 = all_time.tail(8).copy()
    at_plot = pd.concat([top8, bot8]).drop_duplicates()
    at_plot['label'] = at_plot['team_ID'].map(TEAM_NAMES).fillna(at_plot['team_ID'])
    league_median = all_time['avg_ltv_cac'].median()
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=BG)
    ax.set_facecolor(BG)
    at_colors = [RED if t == 'OAK' else (BLUE if v > league_median else '#aaaaaa')
                 for t, v in zip(at_plot['team_ID'], at_plot['avg_ltv_cac'])]
    ax.barh(at_plot['label'], at_plot['avg_ltv_cac'], color=at_colors, alpha=0.82, edgecolor='none')
    ax.axvline(league_median, color='black', lw=1.2, ls='--', alpha=0.6,
               label=f'Median ({league_median:.1f}x)')
    ax.set_xlabel('Avg Seasonal LTV:CAC (1990-2023)', fontsize=11)
    ax.set_title('Best and Worst Teams: Payroll Efficiency 1990-2023', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR + 'team_all_time.png', dpi=150, bbox_inches='tight')
    if show: plt.show()
    else: plt.close()
    print('  Saved team_all_time.png')

    oak_yr = team_yr[team_yr['team_ID'] == 'OAK'].copy()
    lg_avg = team_yr.groupby('season')['ltv_cac'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(11, 5), facecolor=BG)
    ax.set_facecolor(BG)
    ax.fill_between(oak_yr['season'], oak_yr['ltv_cac'], alpha=0.2, color=BLUE)
    ax.plot(oak_yr['season'], oak_yr['ltv_cac'], color=BLUE, lw=2, label="Oakland A's")
    ax.plot(lg_avg['season'], lg_avg['ltv_cac'], color='black', lw=1.2, ls='--', alpha=0.6, label='League avg')
    ax.axvspan(2000, 2004, alpha=0.12, color=RED, label='Moneyball era')
    ax.set_xlabel('Season', fontsize=11)
    ax.set_ylabel('Seasonal LTV:CAC', fontsize=11)
    ax.set_title("Oakland A's: Payroll Efficiency vs League Average (1990-2023)",
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR + 'team_oak_trend.png', dpi=150, bbox_inches='tight')
    if show: plt.show()
    else: plt.close()
    print('  Saved team_oak_trend.png')


def print_findings(merged):
    print('\n[Results] Best LTV:CAC (min 10 career WAR):')
    top10 = (merged[merged['career_war'] >= 10]
             .nlargest(10, 'ltv_cac')
             [['Name', 'player_type', 'career_war', 'cac', 'ltv_cac', 'debut_season']]
             .copy())
    top10['cac'] = (top10['cac'] / 1e6).round(1)
    top10.columns = ['Player', 'Type', 'Career WAR', 'Career Salary ($M)', 'LTV:CAC', 'Debut']
    print(top10.to_string(index=False))

    print('\n[Results] Worst LTV:CAC (min 10 career WAR):')
    bot10 = (merged[merged['career_war'] >= 10]
             .nsmallest(10, 'ltv_cac')
             [['Name', 'player_type', 'career_war', 'cac', 'ltv_cac', 'debut_season']]
             .copy())
    bot10['cac'] = (bot10['cac'] / 1e6).round(1)
    bot10.columns = ['Player', 'Type', 'Career WAR', 'Career Salary ($M)', 'LTV:CAC', 'Debut']
    print(bot10.to_string(index=False))

    print(f'\n[Done] All outputs saved to {OUTPUTS_DIR}')


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    print('=' * 60)
    print('Model 05 — MLB Player LTV:CAC Analysis')
    print('=' * 60)
    yearly = load_yearly_stats()
    career, merged = aggregate_careers(yearly)
    payback_curve, payback_year = compute_payback_curve(merged, yearly)
    draft_r1, pick_analysis, has_draft = analyze_draft_slots()
    mb_rank, all_time, team_yr = analyze_teams(yearly)
    generate_plots(merged, payback_curve, payback_year, pick_analysis, has_draft,
                   mb_rank, all_time, team_yr, show=False)
    print_findings(merged)
