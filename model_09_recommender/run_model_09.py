"""
Model 09 -- Player Similarity Recommender (Premier League, FBref)
Cosine similarity over per-90 stat profiles, indexed with FAISS
Run: python run_model_09.py
"""
import os, time, warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

OUTPUTS_DIR = "outputs/"
os.makedirs(OUTPUTS_DIR, exist_ok=True)

CACHE_RAW = OUTPUTS_DIR + "player_stats_raw.csv"

BG, RED, BLUE = "#f5f0eb", "#9b3d36", "#4c72b0"
LEAGUE = "ENG-Premier League"
SEASON = "2425"   # 2024-25, most recent completed PL season
MIN_90S = 10.0     # minimum "90s played" to qualify (~900 minutes)

SPOTLIGHT = [
    "Mohamed Salah", "Erling Haaland", "Bukayo Saka",
    "Virgil van Dijk", "Kevin De Bruyne", "Cole Palmer",
]

POS_COLORS = {
    "FW": "#9b3d36", "MF": "#4c72b0", "DF": "#5a965a", "GK": "#c89b32",
}


# ── Data loading ─────────────────────────────────────────────────────────────

def _flatten_cols(df):
    df = df.copy()
    df.columns = ["_".join([c for c in col if c]).strip("_") if isinstance(col, tuple) else col
                  for col in df.columns]
    return df.reset_index()


def fetch_fbref_stats():
    import soccerdata as sd
    print(f"Fetching FBref player stats: {LEAGUE} {SEASON}...")
    fb = sd.FBref(leagues=LEAGUE, seasons=SEASON)

    frames = {}
    for stat_type in ["standard", "shooting", "misc"]:
        print(f"  Pulling stat_type={stat_type}...", end=" ", flush=True)
        df = fb.read_player_season_stats(stat_type=stat_type)
        df = _flatten_cols(df)
        frames[stat_type] = df
        print(f"{df.shape[0]} rows")
        time.sleep(2)

    return frames


def load_data():
    if os.path.exists(CACHE_RAW):
        print(f"Loading cached player stats from {CACHE_RAW}...")
        df = pd.read_csv(CACHE_RAW)
        print(f"  {len(df)} players loaded from cache")
        return df

    print("No cache found -- scraping FBref via soccerdata (this hits FBref's site, be patient)...")
    frames = fetch_fbref_stats()

    std = frames["standard"][["league", "season", "team", "player", "nation", "pos", "age",
                               "Playing Time_90s", "Performance_Gls", "Performance_Ast",
                               "Performance_G+A", "Performance_CrdY", "Performance_CrdR"]].copy()
    std = std.rename(columns={"Playing Time_90s": "NINETIES"})

    sho = frames["shooting"][["player", "team", "Standard_Sh", "Standard_SoT",
                               "Standard_Sh/90", "Standard_SoT/90"]].copy()

    misc = frames["misc"][["player", "team", "Performance_Fls", "Performance_Fld",
                            "Performance_Off", "Performance_Crs", "Performance_Int",
                            "Performance_TklW"]].copy()

    df = std.merge(sho, on=["player", "team"], how="left").merge(misc, on=["player", "team"], how="left")
    df.to_csv(CACHE_RAW, index=False)
    print(f"  Cached {len(df)} players to {CACHE_RAW}")
    return df


# ── Feature engineering ───────────────────────────────────────────────────────

def build_features(df):
    print("\nBuilding per-90 feature profiles...")
    d = df.copy()
    d["NINETIES"] = pd.to_numeric(d["NINETIES"], errors="coerce")
    d = d[d["NINETIES"] >= MIN_90S].copy()
    print(f"  {len(d)} players with >= {MIN_90S} 90s played ({MIN_90S*90:.0f}+ minutes)")

    raw_cols = ["Performance_Gls", "Performance_Ast", "Performance_G+A",
                "Standard_Sh", "Standard_SoT", "Performance_Fls", "Performance_Fld",
                "Performance_Off", "Performance_Crs", "Performance_Int", "Performance_TklW"]
    for c in raw_cols:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0)

    feature_cols = []
    for c in raw_cols:
        per90_col = c + "_p90"
        d[per90_col] = d[c] / d["NINETIES"]
        feature_cols.append(per90_col)

    d["pos_simple"] = d["pos"].astype(str).str.split(",").str[0].str.strip()
    d = d.dropna(subset=feature_cols)
    d = d.drop_duplicates(subset=["player"], keep="first").reset_index(drop=True)

    print(f"  Feature matrix: {len(d)} players x {len(feature_cols)} features")
    return d, feature_cols


# ── Similarity engine (FAISS) ─────────────────────────────────────────────────

def build_similarity_index(d, feature_cols):
    import faiss
    from sklearn.preprocessing import StandardScaler

    print("\nBuilding FAISS similarity index...")
    X = d[feature_cols].values.astype("float32")
    X = StandardScaler().fit_transform(X).astype("float32")

    # L2-normalize so inner product == cosine similarity
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    X_norm = (X / norms).astype("float32")

    index = faiss.IndexFlatIP(X_norm.shape[1])
    index.add(X_norm)
    print(f"  Index built: {index.ntotal} vectors, dim={X_norm.shape[1]}")
    return index, X_norm


def find_similar(d, index, X_norm, player_name, k=6):
    matches = d.index[d["player"] == player_name]
    if len(matches) == 0:
        return None
    idx = matches[0]
    query = X_norm[idx:idx+1]
    sims, ids = index.search(query, k + 1)  # +1 because the player matches itself
    rows = []
    for sim, i in zip(sims[0], ids[0]):
        if i == idx:
            continue
        rows.append({
            "player": d.iloc[i]["player"],
            "team": d.iloc[i]["team"],
            "pos": d.iloc[i]["pos_simple"],
            "similarity": float(sim),
        })
        if len(rows) == k:
            break
    return rows


def run_spotlight_comparisons(d, index, X_norm):
    print("\nRunning spotlight player comparisons...")
    all_rows = []
    for name in SPOTLIGHT:
        matches = find_similar(d, index, X_norm, name, k=5)
        if matches is None:
            print(f"  {name}: not found in dataset (filtered out or name mismatch)")
            continue
        print(f"  {name}:")
        for m in matches:
            print(f"    {m['player']:<22} ({m['team']:<18}) sim={m['similarity']:.3f}")
            all_rows.append({"query_player": name, **m})
    return pd.DataFrame(all_rows)


# ── Visualizations ────────────────────────────────────────────────────────────

def generate_plots(d, feature_cols, comparisons_df, show=False):
    matplotlib.use("Agg")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    X = StandardScaler().fit_transform(d[feature_cols].values)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    d = d.copy()
    d["PCA1"], d["PCA2"] = coords[:, 0], coords[:, 1]

    fig, ax = plt.subplots(figsize=(9, 7), facecolor=BG)
    ax.set_facecolor(BG)
    for pos, color in POS_COLORS.items():
        sub = d[d["pos_simple"] == pos]
        ax.scatter(sub["PCA1"], sub["PCA2"], color=color, alpha=0.5, s=28, label=pos)

    spotlight_rows = d[d["player"].isin(SPOTLIGHT)]
    ax.scatter(spotlight_rows["PCA1"], spotlight_rows["PCA2"], color="#222", s=90,
               marker="*", zorder=5, label="Spotlight")
    for _, r in spotlight_rows.iterrows():
        ax.annotate(r["player"], (r["PCA1"], r["PCA2"]), fontsize=8, fontweight="bold",
                    xytext=(5, 5), textcoords="offset points")

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)", fontsize=10)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)", fontsize=10)
    ax.set_title(f"Premier League {SEASON[:2]}-{SEASON[2:]} Player Stat Profiles (PCA)\n"
                 f"n={len(d)} players, min {MIN_90S*90:.0f} minutes",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="best")
    ax.grid(color="#e0dbd5", lw=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR + "player_pca_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: player_pca_scatter.png")

    if len(comparisons_df) > 0:
        focus_player = SPOTLIGHT[0]
        sub = comparisons_df[comparisons_df["query_player"] == focus_player].sort_values("similarity")
        if len(sub) > 0:
            fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=BG)
            ax.set_facecolor(BG)
            ax.barh(sub["player"], sub["similarity"], color=RED, alpha=0.85)
            ax.set_xlabel("Cosine Similarity", fontsize=10)
            ax.set_title(f"Top-5 Most Similar Players to {focus_player}\n"
                         f"Premier League {SEASON[:2]}-{SEASON[2:]}, per-90 stat profile",
                         fontsize=11, fontweight="bold")
            ax.grid(axis="x", color="#e0dbd5", lw=0.7)
            ax.spines[["top", "right"]].set_visible(False)
            plt.tight_layout()
            plt.savefig(OUTPUTS_DIR + "similarity_top5_example.png", dpi=150, bbox_inches="tight")
            plt.close()
            print("  Saved: similarity_top5_example.png")

    if show:
        plt.show()


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    matplotlib.use("Agg")

    raw = load_data()
    d, feature_cols = build_features(raw)
    index, X_norm = build_similarity_index(d, feature_cols)
    comparisons_df = run_spotlight_comparisons(d, index, X_norm)
    generate_plots(d, feature_cols, comparisons_df, show=False)

    comparisons_df.to_csv(OUTPUTS_DIR + "spotlight_comparisons.csv", index=False)
    d.to_csv(OUTPUTS_DIR + "player_features.csv", index=False)

    print(f"\n{'='*60}")
    print("  KEY FINDINGS -- Player Similarity Recommender")
    print(f"{'='*60}")
    print(f"  Players in index: {len(d)}")
    print(f"  Spotlight comparisons run: {comparisons_df['query_player'].nunique() if len(comparisons_df) else 0}")
    print(f"\nAll outputs saved to {OUTPUTS_DIR}")
