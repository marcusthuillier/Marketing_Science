import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WAR_DOLLARS = 8_000_000
OUTPUTS = 'outputs/'
BG   = '#f5f0eb'
RED  = '#9b3d36'
BLUE = '#4c72b0'

draft_r1 = pd.read_csv(OUTPUTS + 'mlb_draft_r1.csv')
draft_r1['Name']  = draft_r1['Name'].str.strip()
draft_r1['OvPck'] = pd.to_numeric(draft_r1['OvPck'], errors='coerce')
draft_r1['WAR']   = pd.to_numeric(draft_r1['WAR'],   errors='coerce')
draft_r1['Bonus'] = (draft_r1['Bonus'].astype(str)
                     .str.replace(r'[\$,]', '', regex=True).str.strip())
draft_r1['Bonus'] = pd.to_numeric(draft_r1['Bonus'], errors='coerce')

d30 = draft_r1[(draft_r1['OvPck'] >= 1) & (draft_r1['OvPck'] <= 30)
               & draft_r1['WAR'].notna()].copy()
d30['pick_bin'] = pd.cut(
    d30['OvPck'],
    bins=[0, 5, 10, 15, 20, 30],
    labels=['Picks\n1-5', 'Picks\n6-10', 'Picks\n11-15', 'Picks\n16-20', 'Picks\n21-30']
)
d30['ltv_draft']     = d30['WAR'] * WAR_DOLLARS
d30['ltv_cac_draft'] = d30['ltv_draft'] / d30['Bonus']

pick_analysis = (d30.groupby('pick_bin', observed=True)
                 .agg(median_war     = ('WAR',          'median'),
                      median_bonus   = ('Bonus',        'median'),
                      median_ltv_cac = ('ltv_cac_draft','median'),
                      n              = ('Name',         'count'))
                 .reset_index())

pick_ltv = pick_analysis.dropna(subset=['median_ltv_cac'])

# WAR chart
fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
ax.set_facecolor(BG)
best_war = pick_analysis['median_war'].max()
colors   = [RED if v == best_war else BLUE for v in pick_analysis['median_war']]
bars     = ax.bar(pick_analysis['pick_bin'].astype(str),
                  pick_analysis['median_war'],
                  color=colors, alpha=0.82, edgecolor='none')
for bar, val in zip(bars, pick_analysis['median_war']):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
            f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_xlabel('Round 1 Pick Range', fontsize=11)
ax.set_ylabel('Median Career WAR', fontsize=11)
ax.set_title('First Round Pick Efficiency: Career WAR by Pick Slot\n1990-2015 MLB Drafts',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUTS + 'pick_slot_war.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved pick_slot_war.png')

# LTV:CAC chart — picks 21-30 highlighted in RED to show the cliff
fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
ax.set_facecolor(BG)
colors2 = [RED if str(b) == 'Picks\n21-30' else BLUE for b in pick_ltv['pick_bin']]
bars2   = ax.bar(pick_ltv['pick_bin'].astype(str),
                 pick_ltv['median_ltv_cac'],
                 color=colors2, alpha=0.82, edgecolor='none')
for bar, val in zip(bars2, pick_ltv['median_ltv_cac']):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f'{val:.0f}x', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_xlabel('Round 1 Pick Range', fontsize=11)
ax.set_ylabel('Median LTV:CAC (Signing Bonus as CAC)', fontsize=11)
ax.set_title('Pick Slot ROI: LTV:CAC by Pick Range\n(LTV = WAR x $8M, CAC = Signing Bonus)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUTS + 'pick_slot_ltv_cac.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved pick_slot_ltv_cac.png')
print('\nPick analysis:')
print(pick_ltv[['pick_bin','median_war','median_ltv_cac','n']].to_string(index=False))
