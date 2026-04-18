"""
Standalone runner for Model 01 — NBA Player Segmentation.
Equivalent to running all notebook cells top-to-bottom.
Run with: python run_model_01.py
"""
import asyncio
import sys
# Windows fix for asyncio + nba_api
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import time
import warnings
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — saves files without displaying
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, pairwise_distances
import umap

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 50)
pd.set_option("display.float_format", "{:.3f}".format)

RANDOM_STATE = 42
OUTPUTS_DIR = "outputs/"
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ── 1. DATA COLLECTION ────────────────────────────────────────────────────────
print("\n=== 1. Data Collection ===")
from nba_api.stats.endpoints import leaguedashplayerstats

SEASONS = ["2022-23", "2023-24", "2024-25"]
MIN_MINUTES = 500

frames = []
for season in SEASONS:
    print(f"  Fetching {season}...", end=" ", flush=True)
    time.sleep(1)
    resp = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        per_mode_simple="PerGame",
        measure_type_simple="Advanced"
    )
    df = resp.get_data_frames()[0]
    df["SEASON"] = season
    frames.append(df)
    print(f"{len(df)} players")

raw = pd.concat(frames, ignore_index=True)
print(f"  Total rows: {len(raw)}")

# ── 2. PREPROCESSING ──────────────────────────────────────────────────────────
print("\n=== 2. Preprocessing ===")
df = raw[raw["MIN"] >= MIN_MINUTES].copy()
print(f"  Players >= {MIN_MINUTES} min: {len(df)}")

FEATURES = [
    "USG_PCT", "AST_PCT", "REB_PCT", "OREB_PCT", "DREB_PCT",
    "STL_PCT", "BLK_PCT", "TS_PCT", "E_TOV_PCT", "PACE",
]
available = [f for f in FEATURES if f in df.columns]
missing   = [f for f in FEATURES if f not in df.columns]
if missing:
    print(f"  Skipping missing columns: {missing}")
FEATURES = available

df = df.sort_values("SEASON", ascending=False)
df = df.drop_duplicates(subset="PLAYER_NAME", keep="first").reset_index(drop=True)
print(f"  Unique players (most recent season): {len(df)}")

df_clean = df.dropna(subset=FEATURES).copy()
print(f"  After dropping nulls: {len(df_clean)}")
print(f"  Features: {FEATURES}")

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(df_clean[FEATURES])

# ── 3. DIMENSIONALITY REDUCTION ───────────────────────────────────────────────
print("\n=== 3. Dimensionality Reduction ===")
print("  Fitting UMAP...", end=" ", flush=True)
reducer = umap.UMAP(n_components=2, random_state=RANDOM_STATE, n_neighbors=15, min_dist=0.1)
X_umap  = reducer.fit_transform(X_scaled)
df_clean = df_clean.copy()
df_clean["UMAP_1"] = X_umap[:, 0]
df_clean["UMAP_2"] = X_umap[:, 1]
print("done")

pca   = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)
df_clean["PCA_1"] = X_pca[:, 0]
df_clean["PCA_2"] = X_pca[:, 1]
print(f"  PCA variance explained (2 components): {pca.explained_variance_ratio_.cumsum()[1]:.1%}")

# ── 4. K SELECTION ────────────────────────────────────────────────────────────
print("\n=== 4. K Selection ===")
K_RANGE     = range(4, 11)
inertias    = []
silhouettes = []

for k in K_RANGE:
    km     = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))
    print(f"  K={k}: inertia={km.inertia_:.1f}, silhouette={silhouettes[-1]:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(list(K_RANGE), inertias, "bo-")
axes[0].set_xlabel("K"); axes[0].set_ylabel("Inertia")
axes[0].set_title("Elbow Curve"); axes[0].grid(True, alpha=0.3)
axes[1].plot(list(K_RANGE), silhouettes, "ro-")
axes[1].set_xlabel("K"); axes[1].set_ylabel("Silhouette Score")
axes[1].set_title("Silhouette Score by K"); axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUTS_DIR + "elbow_silhouette.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUTPUTS_DIR}elbow_silhouette.png")

best_k = list(K_RANGE)[silhouettes.index(max(silhouettes))]
print(f"  Best K by silhouette: {best_k}")

K = best_k
km_final = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=20)
df_clean["CLUSTER"] = km_final.fit_predict(X_scaled)
print(f"\n  Final K={K}")
print(df_clean["CLUSTER"].value_counts().sort_index().to_string())

# ── 5. ANALYSIS ───────────────────────────────────────────────────────────────
print("\n=== 5. Cluster Analysis ===")
cluster_profiles = df_clean.groupby("CLUSTER")[FEATURES].mean().round(3)
print("\nCluster Profiles (mean stats):")
print(cluster_profiles.to_string())

centroids = km_final.cluster_centers_
dists     = pairwise_distances(X_scaled, centroids)
df_clean["DIST_TO_CENTROID"] = [dists[i, c] for i, c in enumerate(df_clean["CLUSTER"])]

print("\nTop 5 representative players per cluster:")
for c in range(K):
    top5 = (
        df_clean[df_clean["CLUSTER"] == c]
        .nsmallest(5, "DIST_TO_CENTROID")[["PLAYER_NAME", "SEASON", "DIST_TO_CENTROID"]]
    )
    print(f"\n  --- Cluster {c} ---")
    print(top5.to_string(index=False))

# ── CLUSTER LABELS (auto-assigned, edit as needed) ────────────────────────────
# Auto-label by dominant stat in each cluster's profile
def auto_label(row):
    dominant = row[["USG_PCT","AST_PCT","REB_PCT","STL_PCT","BLK_PCT"]].idxmax()
    label_map = {
        "USG_PCT":  "High-Usage Scorer",
        "AST_PCT":  "Playmaker",
        "REB_PCT":  "Rebounder / Big",
        "STL_PCT":  "Perimeter Defender",
        "BLK_PCT":  "Rim Protector",
    }
    return label_map.get(dominant, f"Cluster {row.name}")

# Normalize profiles for comparison
profile_norm = cluster_profiles.copy()
profile_norm = (profile_norm - profile_norm.min()) / (profile_norm.max() - profile_norm.min() + 1e-9)
CLUSTER_LABELS = {i: auto_label(profile_norm.loc[i]) for i in range(K)}

# Deduplicate labels if same name assigned twice
seen = {}
for k_idx, label in CLUSTER_LABELS.items():
    if label in seen:
        seen[label] += 1
        CLUSTER_LABELS[k_idx] = f"{label} {seen[label]}"
    else:
        seen[label] = 1

print("\nAuto-assigned cluster labels (edit CLUSTER_LABELS in notebook to refine):")
for k_idx, label in CLUSTER_LABELS.items():
    print(f"  Cluster {k_idx}: {label}")

df_clean["ARCHETYPE"] = df_clean["CLUSTER"].map(CLUSTER_LABELS)
print("\nArchetype counts:")
print(df_clean["ARCHETYPE"].value_counts().to_string())

# ── 6. VISUALIZATIONS ─────────────────────────────────────────────────────────
print("\n=== 6. Visualizations ===")

# UMAP scatter (Plotly — saves to HTML)
fig = px.scatter(
    df_clean,
    x="UMAP_1", y="UMAP_2",
    color="ARCHETYPE",
    hover_name="PLAYER_NAME",
    hover_data={"SEASON": True, "UMAP_1": False, "UMAP_2": False},
    title=f"NBA Player Archetypes — UMAP Projection (K={K})",
    width=900, height=650,
    template="plotly_white"
)
fig.update_traces(marker=dict(size=7, opacity=0.8))
fig.write_html(OUTPUTS_DIR + "umap_plot.html")
print(f"  Saved: {OUTPUTS_DIR}umap_plot.html")

# Radar charts
RADAR_FEATURES = [f for f in ["USG_PCT","AST_PCT","REB_PCT","STL_PCT","BLK_PCT","TS_PCT"] if f in FEATURES]
profile_norm_radar = cluster_profiles[RADAR_FEATURES].copy()
profile_norm_radar = (profile_norm_radar - profile_norm_radar.min()) / (profile_norm_radar.max() - profile_norm_radar.min() + 1e-9)

cols = min(4, K)
rows = (K + cols - 1) // cols
fig2, axes2 = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), subplot_kw=dict(polar=True))
axes2 = np.array(axes2).flatten()

N      = len(RADAR_FEATURES)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]
colors  = plt.cm.tab10(np.linspace(0, 1, K))

for i, (cluster_id, row) in enumerate(profile_norm_radar.iterrows()):
    ax     = axes2[i]
    values = row.tolist() + row.tolist()[:1]
    ax.plot(angles, values, "o-", linewidth=2, color=colors[i])
    ax.fill(angles, values, alpha=0.25, color=colors[i])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f.replace("_PCT","%%").replace("_"," ") for f in RADAR_FEATURES], size=8)
    ax.set_ylim(0, 1)
    ax.set_title(CLUSTER_LABELS[cluster_id], size=10, fontweight="bold", pad=15)
    ax.grid(True, alpha=0.3)

for j in range(K, len(axes2)):
    axes2[j].set_visible(False)

plt.suptitle("NBA Player Archetype Profiles", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUTPUTS_DIR + "cluster_profiles.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUTPUTS_DIR}cluster_profiles.png")

# ── 7. OUTPUT CSV ─────────────────────────────────────────────────────────────
print("\n=== 7. Outputs ===")
output_cols = ["PLAYER_NAME","SEASON","CLUSTER","ARCHETYPE","DIST_TO_CENTROID","UMAP_1","UMAP_2"] + FEATURES
output_cols = [c for c in output_cols if c in df_clean.columns]
df_clean[output_cols].to_csv(OUTPUTS_DIR + "clustered_players.csv", index=False)
print(f"  Saved: {OUTPUTS_DIR}clustered_players.csv")

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("DONE — Model 01 Complete")
print("="*55)
print(f"  Players clustered : {len(df_clean)}")
print(f"  Seasons           : {SEASONS}")
print(f"  K                 : {K}")
print(f"\n  Archetype distribution:")
print(df_clean.groupby("ARCHETYPE").size().sort_values(ascending=False).to_string())
print(f"\n  Outputs saved to: {OUTPUTS_DIR}")
print("  - elbow_silhouette.png")
print("  - umap_plot.html  (open in browser)")
print("  - cluster_profiles.png")
print("  - clustered_players.csv")

# ── PLAYER LOOKUP ─────────────────────────────────────────────────────────────
print("\n  Sample player lookups:")
for name in ["Jokic", "Curry", "Giannis", "Gobert", "Haliburton"]:
    match = df_clean[df_clean["PLAYER_NAME"].str.contains(name, case=False)]
    if not match.empty:
        row = match.iloc[0]
        print(f"  {row['PLAYER_NAME']:25s} → {row['ARCHETYPE']} (cluster {row['CLUSTER']})")
