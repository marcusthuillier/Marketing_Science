import pandas as pd, numpy as np
import matplotlib, matplotlib.pyplot as plt
matplotlib.use('Agg')
import seaborn as sns
from pybaseball import bwar_bat, bwar_pitch

LEAGUE_MIN = {
    1990:100000,1991:100000,1992:109000,1993:109000,1994:109000,
    1995:109000,1996:150000,1997:150000,1998:170000,1999:200000,
    2000:200000,2001:200000,2002:200000,2003:300000,2004:300000,
    2005:316000,2006:327000,2007:380000,2008:390000,2009:400000,
    2010:400000,2011:414000,2012:414000,2013:490000,2014:500000,
    2015:507500,2016:507500,2017:535000,2018:545000,2019:555000,
    2020:563500,2021:570500,2022:700000,2023:720000,
}
NAMES = {
    'OAK':"Oakland A's",'MON':'Montreal Expos','CLE':'Cleveland',
    'FLA':'Florida Marlins','MIN':'Minnesota Twins','CHW':'Chicago White Sox',
    'TBD':'Tampa Bay (pre-2008)','TBR':'Tampa Bay Rays','BAL':'Baltimore Orioles',
    'PIT':'Pittsburgh Pirates','SEA':'Seattle Mariners','MIL':'Milwaukee Brewers',
    'TOR':'Toronto Blue Jays','HOU':'Houston Astros','CIN':'Cincinnati Reds',
    'PHI':'Philadelphia Phillies','ATL':'Atlanta Braves','DET':'Detroit Tigers',
    'STL':'St. Louis Cardinals','TEX':'Texas Rangers','BOS':'Boston Red Sox',
    'KCR':'Kansas City Royals','SDP':'San Diego Padres','COL':'Colorado Rockies',
    'SFG':'San Francisco Giants','NYY':'New York Yankees','CHC':'Chicago Cubs',
    'LAD':'Los Angeles Dodgers','NYM':'New York Mets','MIA':'Miami Marlins',
    'ARI':'Arizona Diamondbacks','WSN':'Washington Nationals','LAA':'LA Angels',
}
BG, RED, BLUE, GREY = '#f5f0eb', '#9b3d36', '#4c72b0', '#aaaaaa'
OUTPUTS = 'outputs/'
sns.set_theme(style='whitegrid')

print('Loading data...')
bat = bwar_bat(return_all=True)[['player_ID','year_ID','team_ID','WAR','salary']].copy()
pit = bwar_pitch(return_all=True)[['player_ID','year_ID','team_ID','WAR','salary']].copy()
df  = pd.concat([bat, pit], ignore_index=True)
df  = df[(df['year_ID'] >= 1990) & (df['year_ID'] <= 2023)].copy()
df['salary_filled'] = df.apply(
    lambda r: r['salary'] if pd.notna(r['salary'])
    else LEAGUE_MIN.get(int(r['year_ID']), 0), axis=1
)

mkt = df.groupby('year_ID').agg(
    total_war=('WAR','sum'), total_sal=('salary_filled','sum')
).reset_index()
mkt['dol_per_war'] = mkt['total_sal'] / mkt['total_war']

team_yr = (df.groupby(['team_ID','year_ID'])
    .agg(total_war=('WAR','sum'), total_salary=('salary_filled','sum'))
    .reset_index()
    .merge(mkt[['year_ID','dol_per_war']], on='year_ID'))
team_yr = team_yr[(team_yr['total_salary'] > 1_000_000) & (team_yr['year_ID'] != 2020)]
team_yr['ltv_cac'] = (team_yr['total_war'] * team_yr['dol_per_war']) / team_yr['total_salary']

all_time = (team_yr.groupby('team_ID')
    .agg(avg_ltv_cac=('ltv_cac','mean'), seasons=('year_ID','count'))
    .reset_index())
all_time = all_time[all_time['seasons'] >= 15].sort_values('avg_ltv_cac', ascending=False)
all_time['label'] = all_time['team_ID'].map(NAMES).fillna(all_time['team_ID'])

league_avg = team_yr.groupby('year_ID')['ltv_cac'].mean().reset_index(name='lg_avg')

# ── Plot 1: Normalized all-time best & worst (horizontal bar) ────
top8 = all_time.head(8)
bot8 = all_time.tail(8)
plot_df = pd.concat([top8, bot8]).drop_duplicates()
highlight = {'OAK','TBR','TBD'}

fig, ax = plt.subplots(figsize=(11, 7), facecolor=BG)
ax.set_facecolor(BG)
colors = [RED if t in highlight else (BLUE if v > 1.0 else GREY)
          for t, v in zip(plot_df['team_ID'], plot_df['avg_ltv_cac'])]
bars = ax.barh(plot_df['label'], plot_df['avg_ltv_cac'],
               color=colors, alpha=0.82, edgecolor='none')
ax.axvline(1.0, color='black', lw=1.3, ls='--', alpha=0.6, label='Break even (1.0x)')
for bar, val in zip(bars, plot_df['avg_ltv_cac']):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
            f'{val:.2f}x', va='center', fontsize=9)
ax.set_xlabel('Avg Normalized LTV:CAC  (year-specific market rate)', fontsize=11)
ax.set_title('Best and Worst Teams: Payroll Efficiency 1990-2023\n'
             'Normalized to each season\'s market rate per WAR  |  2020 excluded',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(OUTPUTS + 'team_normalized_bar.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved team_normalized_bar.png')

# ── Plot 2: Selected teams over time (normalized) ─────────────────
# TBR = modern efficiency, OAK = classic, NYY = overspender, MIN = quiet
sel = {
    'OAK': (RED,       "Oakland A's",     2.2, '-'),
    'TBR': ('#2ca02c', 'Tampa Bay Rays',  2.2, '-'),
    'MIN': ('#e07b39', 'Minnesota Twins', 1.4, '-'),
    'NYY': ('#2c5f8a', 'New York Yankees',1.4, '-'),
}

fig, ax = plt.subplots(figsize=(13, 6), facecolor=BG)
ax.set_facecolor(BG)
for tid, (color, label, lw, ls) in sel.items():
    d = team_yr[team_yr['team_ID'].isin([tid,'TBD'] if tid=='TBR' else [tid])].sort_values('year_ID')
    ax.plot(d['year_ID'], d['ltv_cac'], color=color, lw=lw, ls=ls, label=label, alpha=0.9)

ax.plot(league_avg['year_ID'], league_avg['lg_avg'],
        color='black', lw=1.0, ls='--', alpha=0.4, label='League average')
ax.axhline(1.0, color='black', lw=0.8, ls=':', alpha=0.3)
ax.axvspan(2000, 2004, alpha=0.07, color=RED)
ax.text(2002, team_yr['ltv_cac'].quantile(0.94),
        'Moneyball', fontsize=8, color=RED, ha='center', alpha=0.65)
ax.set_xlim(1990, 2023)
ax.set_xlabel('Season', fontsize=11)
ax.set_ylabel('Normalized LTV:CAC  (year-specific market rate)', fontsize=11)
ax.set_title('Payroll Efficiency Over Time: Selected Teams  |  Normalized  |  2020 excluded',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9, loc='upper right')
plt.tight_layout()
plt.savefig(OUTPUTS + 'team_normalized_lines.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved team_normalized_lines.png')

# ── Plot 3: Heatmap normalized ────────────────────────────────────
pivot = team_yr.pivot(index='team_ID', columns='year_ID', values='ltv_cac')
pivot = pivot[pivot.notna().sum(axis=1) >= 20]
pivot = pivot.reindex(pivot.mean(axis=1).sort_values(ascending=False).index)
pivot.index = [NAMES.get(t, t) for t in pivot.index]

fig, ax = plt.subplots(figsize=(18, 10), facecolor=BG)
ax.set_facecolor(BG)
sns.heatmap(pivot, ax=ax, cmap='RdYlGn', center=1.0, vmin=0, vmax=3,
            linewidths=0.25, linecolor='#dddddd',
            cbar_kws={'label': 'LTV:CAC (normalized)', 'shrink': 0.55})
ax.set_title(
    'MLB Team Payroll Efficiency by Season  |  Normalized to market rate  |  1990-2023\n'
    'Green = got more WAR than payroll implies    Red = overpaid relative to market',
    fontsize=12, fontweight='bold', pad=12)
ax.set_xlabel('Season', fontsize=10)
ax.set_ylabel('')
ax.tick_params(axis='x', labelsize=7, rotation=45)
ax.tick_params(axis='y', labelsize=8)
plt.tight_layout()
plt.savefig(OUTPUTS + 'team_normalized_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved team_normalized_heatmap.png')

# ── Plot 4: Is market efficiency improving? (team spread over time) ─
yr_spread = team_yr.groupby('year_ID').agg(
    iqr = ('ltv_cac', lambda x: x.quantile(0.75) - x.quantile(0.25))
).reset_index()
z = np.polyfit(yr_spread['year_ID'], yr_spread['iqr'], 1)
trend = np.poly1d(z)

fig, ax = plt.subplots(figsize=(11, 5), facecolor=BG)
ax.set_facecolor(BG)
ax.fill_between(yr_spread['year_ID'], yr_spread['iqr'], alpha=0.2, color=BLUE)
ax.plot(yr_spread['year_ID'], yr_spread['iqr'], color=BLUE, lw=2.2, label='IQR of team LTV:CAC')
ax.plot(yr_spread['year_ID'], trend(yr_spread['year_ID']),
        color=RED, lw=1.5, ls='--',
        label=f'Trend ({z[0]:+.4f}/yr — essentially flat)')
ax.set_xlabel('Season', fontsize=11)
ax.set_ylabel('IQR of Normalized Team LTV:CAC', fontsize=11)
ax.set_title('Is the Market Getting More Efficient?\n'
             'Spread between best and worst teams over time (smaller = more efficient)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(OUTPUTS + 'market_efficiency_teams.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved market_efficiency_teams.png')
print(f'\nTrend slope: {z[0]:+.4f} per year (flat)')
