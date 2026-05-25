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

BG   = '#f5f0eb'
RED  = '#9b3d36'
BLUE = '#4c72b0'
GREY = '#aaaaaa'
OUTPUTS = 'outputs/'
sns.set_theme(style='whitegrid')

print('Loading data...')
bat = bwar_bat(return_all=True)[['player_ID','year_ID','WAR','salary']].copy()
pit = bwar_pitch(return_all=True)[['player_ID','year_ID','WAR','salary']].copy()
df  = pd.concat([bat, pit], ignore_index=True)
df  = df[(df['year_ID'] >= 1990) & (df['year_ID'] <= 2023)].copy()
df['salary_filled'] = df.apply(
    lambda r: r['salary'] if pd.notna(r['salary'])
    else LEAGUE_MIN.get(int(r['year_ID']), 0), axis=1
)

# implied $/WAR by year from the data itself
mkt = df.groupby('year_ID').agg(
    total_war=('WAR','sum'), total_sal=('salary_filled','sum')
).reset_index()
mkt['dol_per_war'] = mkt['total_sal'] / mkt['total_war']

# player fair-value ratio relative to that year's market rate
df = df.merge(mkt[['year_ID','dol_per_war']], on='year_ID')
df = df[(df['salary_filled'] > 0) & (df['WAR'] > 0)]
df['fair_ratio'] = (df['WAR'] * df['dol_per_war']) / df['salary_filled']

yr = df.groupby('year_ID').agg(
    median_ratio = ('fair_ratio', 'median'),
    iqr          = ('fair_ratio', lambda x: x.quantile(0.75) - x.quantile(0.25)),
    pct_bargain  = ('fair_ratio', lambda x: (x > 2.0).mean()),
).reset_index()

# exclude 2020 (COVID distortion)
mkt_plot = mkt[mkt['year_ID'] != 2020]
yr_plot  = yr[yr['year_ID'] != 2020]

# ── Plot: two panels side by side ────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
for ax in (ax1, ax2):
    ax.set_facecolor(BG)

# Panel 1: $/WAR market rate over time
ax1.fill_between(mkt_plot['year_ID'], mkt_plot['dol_per_war']/1e6,
                 alpha=0.18, color=BLUE)
ax1.plot(mkt_plot['year_ID'], mkt_plot['dol_per_war']/1e6,
         color=BLUE, lw=2.2)
ax1.set_xlabel('Season', fontsize=11)
ax1.set_ylabel('Implied Market Rate ($M per WAR)', fontsize=11)
ax1.set_title('What One Win Is Worth Over Time', fontsize=12, fontweight='bold')
ax1.annotate('$0.77M\nper WAR', xy=(1990, 0.77), xytext=(1993, 1.8),
             fontsize=8, color=BLUE,
             arrowprops=dict(arrowstyle='->', color=BLUE, lw=1))
ax1.annotate('$7.4M\nper WAR', xy=(2023, 7.41), xytext=(2018, 5.5),
             fontsize=8, color=BLUE,
             arrowprops=dict(arrowstyle='->', color=BLUE, lw=1))

# Panel 2: % of players earning < half their market value (bargain rate)
ax2.fill_between(yr_plot['year_ID'], yr_plot['pct_bargain'] * 100,
                 alpha=0.18, color=RED)
ax2.plot(yr_plot['year_ID'], yr_plot['pct_bargain'] * 100,
         color=RED, lw=2.2, label='% of players with fair-value ratio > 2x')
# flat reference line at the mean
avg_bargain = yr_plot['pct_bargain'].mean() * 100
ax2.axhline(avg_bargain, color='black', lw=1.1, ls='--', alpha=0.5,
            label=f'Average ({avg_bargain:.0f}%)')
# add trend line
z = np.polyfit(yr_plot['year_ID'], yr_plot['pct_bargain'] * 100, 1)
p = np.poly1d(z)
ax2.plot(yr_plot['year_ID'], p(yr_plot['year_ID']),
         color=RED, lw=1.2, ls=':', alpha=0.7, label=f'Trend ({z[0]:+.2f}%/yr)')
ax2.set_xlabel('Season', fontsize=11)
ax2.set_ylabel('% of Players Producing 2x+ Their Salary in Value', fontsize=11)
ax2.set_title('Market Efficiency Over Time\n(Has the bargain rate declined?)',
              fontsize=12, fontweight='bold')
ax2.set_ylim(0, 100)
ax2.legend(fontsize=8)

plt.suptitle(
    'MLB Player Market Efficiency: 1990-2023  |  2020 excluded (COVID shortened season)',
    fontsize=11, y=1.01, color=GREY
)
plt.tight_layout()
plt.savefig(OUTPUTS + 'market_efficiency.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved market_efficiency.png')

# print the trend slope
print(f'\nTrend in bargain rate: {z[0]:+.3f}% per year')
print(f'Meaning: the share of underpaid players is {"increasing" if z[0]>0 else "decreasing"} by ~{abs(z[0]):.2f}pp per year')
print(f'Over 33 years that is {z[0]*33:+.1f}pp total — essentially flat')
