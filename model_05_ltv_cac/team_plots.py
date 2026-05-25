import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pybaseball import bwar_bat, bwar_pitch

WAR_DOLLARS = 8_000_000
LEAGUE_MIN = {
    1990:100000,1991:100000,1992:109000,1993:109000,1994:109000,
    1995:109000,1996:150000,1997:150000,1998:170000,1999:200000,
    2000:200000,2001:200000,2002:200000,2003:300000,2004:300000,
    2005:316000,2006:327000,2007:380000,2008:390000,2009:400000,
    2010:400000,2011:414000,2012:414000,2013:490000,2014:500000,
    2015:507500,2016:507500,2017:535000,2018:545000,2019:555000,
    2020:563500,2021:570500,2022:700000,2023:720000,
}

print("Loading data...")
bat = bwar_bat(return_all=True)[['player_ID','year_ID','team_ID','WAR','salary']].copy()
pit = bwar_pitch(return_all=True)[['player_ID','year_ID','team_ID','WAR','salary']].copy()
df = pd.concat([bat, pit], ignore_index=True)
df = df[(df['year_ID'] >= 1990) & (df['year_ID'] <= 2023)].copy()
df['salary_filled'] = df.apply(
    lambda r: r['salary'] if pd.notna(r['salary'])
    else LEAGUE_MIN.get(int(r['year_ID']), 0), axis=1
)

team_yr = df.groupby(['team_ID','year_ID']).agg(
    total_war    = ('WAR',           'sum'),
    total_salary = ('salary_filled', 'sum'),
).reset_index()
team_yr = team_yr[team_yr['total_salary'] > 1_000_000]
team_yr['ltv_cac'] = (team_yr['total_war'] * WAR_DOLLARS) / team_yr['total_salary']
team_yr.rename(columns={'year_ID': 'season'}, inplace=True)

league_avg = team_yr.groupby('season')['ltv_cac'].mean().reset_index(name='lg_avg')

BG   = '#f5f0eb'
RED  = '#9b3d36'
BLUE = '#4c72b0'
OUTPUTS = 'outputs/'
sns.set_theme(style='whitegrid')

# best/worst single seasons for annotation
best_seasons  = team_yr.nlargest(10, 'ltv_cac')
worst_seasons = team_yr[team_yr['total_war'] > 5].nsmallest(10, 'ltv_cac')
print("Best seasons:")
print(best_seasons[['team_ID','season','ltv_cac','total_war','total_salary']].to_string(index=False))
print("\nWorst seasons:")
print(worst_seasons[['team_ID','season','ltv_cac','total_war','total_salary']].to_string(index=False))

# ── Plot 1: Line chart — notable teams over time ──────────────────
# OAK = Moneyball efficiency story
# NYY = big-spend cautionary tale
# LAA = worst all-time (Trout + bad FAs)
# MIN = quiet small-market success
# SEA = mid-pack contrast
highlight = {
    'OAK': (RED,       "Oakland A's",      2.4, '-'),
    'NYY': ('#2c5f8a', 'New York Yankees',  1.6, '-'),
    'LAA': ('#888888', 'LA Angels',         1.4, '-'),
    'MIN': ('#e07b39', 'Minnesota Twins',   1.4, '-'),
}

fig, ax = plt.subplots(figsize=(13, 6), facecolor=BG)
ax.set_facecolor(BG)

for tid, (color, label, lw, ls) in highlight.items():
    d = team_yr[team_yr['team_ID'] == tid].sort_values('season')
    ax.plot(d['season'], d['ltv_cac'], color=color, lw=lw, ls=ls, label=label, alpha=0.9)

ax.plot(league_avg['season'], league_avg['lg_avg'],
        color='black', lw=1.0, ls='--', alpha=0.45, label='League average')

ax.axvspan(2000, 2004, alpha=0.07, color=RED)
ax.text(2002, team_yr['ltv_cac'].quantile(0.92),
        'Moneyball', fontsize=8, color=RED, ha='center', alpha=0.65)

ax.set_xlim(1990, 2023)
ax.set_xlabel('Season', fontsize=11)
ax.set_ylabel('Seasonal LTV:CAC  (WAR value / payroll)', fontsize=11)
ax.set_title('Payroll Efficiency Over Time: Selected Teams  |  1990-2023',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=9, loc='upper right')
plt.tight_layout()
plt.savefig(OUTPUTS + 'team_lines_selected.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved team_lines_selected.png")

# ── Plot 2: Heatmap — all teams x season ─────────────────────────
pivot = team_yr.pivot(index='team_ID', columns='season', values='ltv_cac')
pivot = pivot[pivot.notna().sum(axis=1) >= 20]
pivot = pivot.reindex(pivot.mean(axis=1).sort_values(ascending=False).index)

NAMES = {
    'OAK':"OAK  A's",  'MON':'MON  Expos',    'CLE':'CLE  Guardians',
    'FLA':'FLA  Marlins','MIN':'MIN  Twins',    'CHW':'CHW  White Sox',
    'TBD':'TBD  Devil Rays','TBR':'TBR  Rays', 'BAL':'BAL  Orioles',
    'PIT':'PIT  Pirates','SEA':'SEA  Mariners','MIL':'MIL  Brewers',
    'TOR':'TOR  Blue Jays','HOU':'HOU  Astros','CIN':'CIN  Reds',
    'PHI':'PHI  Phillies','ATL':'ATL  Braves', 'DET':'DET  Tigers',
    'STL':'STL  Cardinals','TEX':'TEX  Rangers','BOS':'BOS  Red Sox',
    'KCR':'KCR  Royals', 'SDP':'SDP  Padres', 'COL':'COL  Rockies',
    'SFG':'SFG  Giants', 'NYY':'NYY  Yankees', 'CHC':'CHC  Cubs',
    'LAD':'LAD  Dodgers','NYM':'NYM  Mets',    'MIA':'MIA  Marlins',
    'ARI':'ARI  D-backs','WSN':'WSN  Nationals','LAA':'LAA  Angels',
}
pivot.index = [NAMES.get(t, t) for t in pivot.index]

fig, ax = plt.subplots(figsize=(18, 10), facecolor=BG)
ax.set_facecolor(BG)
sns.heatmap(
    pivot, ax=ax,
    cmap='RdYlGn',
    center=team_yr['ltv_cac'].median(),
    vmin=0, vmax=12,
    linewidths=0.25, linecolor='#dddddd',
    cbar_kws={'label': 'LTV:CAC', 'shrink': 0.55},
    annot=False,
)
ax.set_title(
    'MLB Team Payroll Efficiency by Season  |  1990-2023\n'
    'Green = high WAR per dollar    Red = overpaid relative to production',
    fontsize=12, fontweight='bold', pad=12
)
ax.set_xlabel('Season', fontsize=10)
ax.set_ylabel('')
ax.tick_params(axis='x', labelsize=7, rotation=45)
ax.tick_params(axis='y', labelsize=8)
plt.tight_layout()
plt.savefig(OUTPUTS + 'team_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved team_heatmap.png")
