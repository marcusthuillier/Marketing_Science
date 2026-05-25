import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUTS_DIR = 'outputs/'

# Crosstab data from notebook output (position rows, cluster cols, row-normalized)
crosstab = pd.DataFrame({
    'Cluster 0': [0.000, 0.040, 0.068, 0.000, 0.267, 0.435, 0.152],
    'Cluster 1': [0.148, 0.400, 0.154, 0.276, 0.000, 0.000, 0.030],
    'Cluster 2': [0.019, 0.080, 0.239, 0.069, 0.533, 0.214, 0.576],
    'Cluster 3': [0.426, 0.280, 0.385, 0.310, 0.000, 0.015, 0.152],
    'Cluster 4': [0.000, 0.000, 0.085, 0.000, 0.200, 0.336, 0.091],
    'Cluster 5': [0.407, 0.200, 0.068, 0.345, 0.000, 0.000, 0.000],
}, index=['C', 'C-F', 'F', 'F-C', 'F-G', 'G', 'G-F'])
crosstab.index.name = 'POSITION'
crosstab.columns.name = 'ARCHETYPE'

# Reverse: archetype rows, position cols, row-normalized (computed from counts)
pos_counts  = {'C': 54, 'C-F': 25, 'F': 117, 'F-C': 29, 'F-G': 15, 'G': 131, 'G-F': 33}
clust_names = ['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5']
arch_data = {}
for pos, n in pos_counts.items():
    arch_data[pos] = {c: round(crosstab.loc[pos, c] * n) for c in clust_names}

arch_df = pd.DataFrame(arch_data, index=clust_names)
arch_df.index.name = 'ARCHETYPE'
arch_df_norm = arch_df.div(arch_df.sum(axis=1), axis=0)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

sns.heatmap(
    crosstab,
    annot=True, fmt='.0%',
    cmap='YlOrRd',
    linewidths=0.5,
    ax=axes[0]
)
axes[0].set_title('Where does each position end up? (each position sums to 100%)', fontsize=11)
axes[0].set_xlabel('Archetype')
axes[0].set_ylabel('Traditional Position')

arch_df_norm.plot(
    kind='bar', stacked=True, colormap='tab10',
    ax=axes[1], edgecolor='white', linewidth=0.5
)
axes[1].set_title('Position mix inside each archetype\n(col %, each cluster sums to 100%)', fontsize=11)
axes[1].set_xlabel('Archetype')
axes[1].set_ylabel('Proportion')
axes[1].legend(title='Position', bbox_to_anchor=(1.01, 1), loc='upper left')
axes[1].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig(OUTPUTS_DIR + 'position_vs_cluster.png', dpi=150, bbox_inches='tight')
print('Saved: outputs/position_vs_cluster.png')
