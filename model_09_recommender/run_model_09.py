"""
Model 09 -- Player Similarity Recommender (Premier League, FBref)
Cosine similarity over per-90 stat profiles (incl. passing, creation, possession),
indexed with FAISS
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

# FBref pages that soccerdata's public API doesn't expose for read_player_season_stats
# but are fetched directly using the library's own (anti-bot-aware) session + the
# same comment-unwrapping parse logic it uses internally.
EXTRA_STAT_PAGES = {"passing": "passing", "gca": "gca", "possession": "possession"}


# ── Data loading ─────────────────────────────────────────────────────────────

def _flatten_cols(df):
    df = df.copy()
    df.columns = ["_".join([c for c in col if c]).strip("_") if isinstance(col, tuple) else col
                  for col in df.columns]
    return df.reset_index()


def _fetch_extra_stat_table(fb, stat_type, page, season_url):
    """
    Fetch an FBref player-season stat table that soccerdata's public
    read_player_season_stats() doesn't support (passing, gca, possession).
    Reuses soccerdata's own request/cache session (which handles FBref's
    anti-bot protection) and its internal HTML-comment-unwrapping parser.
    """
    from soccerdata.fbref import _parse_table, _fix_nation_col
    from lxml import html, etree

    url = "https://fbref.com" + "/".join(season_url.split("/")[:-1]) + f"/{page}/" + season_url.split("/")[-1]
    filepath = fb.data_dir / f"players_{LEAGUE}_{SEASON}_{stat_type}.html"
    reader = fb.get(url, filepath)
    tree = html.parse(reader)
    for elem in tree.xpath("//td[@data-stat='comp_level']//span"):
        elem.getparent().remove(elem)
    (el,) = tree.xpath(f"//comment()[contains(.,'div_stats_{stat_type}')]")
    parser = etree.HTMLParser(recover=True)
    (html_table,) = etree.fromstring(el.text, parser).xpath(f"//table[contains(@id, 'stats_{stat_type}')]")
    df = _parse_table(html_table)
    df = _fix_nation_col(df)
    df = _flatten_cols(df)
    return df


def fetch_fbref_stats():
    import soccerdata as sd
    print(f"Fetching FBref player stats: {LEAGUE} {SEASON}...")
    fb = sd.FBref(leagues=LEAGUE, seasons=SEASON)

    frames = {}
    for stat_type in ["standard", "shooting", "misc"]:
        print(f"  Pulling stat_type={stat_type} (public API)...", end=" ", flush=True)
        df = fb.read_player_season_stats(stat_type=stat_type)
        df = _flatten_cols(df)
        frames[stat_type] = df
        print(f"{df.shape[0]} rows")
        time.sleep(2)

    season_url = fb.read_seasons().iloc[0]["url"]
    for stat_type, page in EXTRA_STAT_PAGES.items():
        print(f"  Pulling stat_type={stat_type} (direct fetch, passing/creation data)...", end=" ", flush=True)
        df = _fetch_extra_stat_table(fb, stat_type, page, season_url)
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

    passing = frames["passing"][["Unnamed: 1_level_0_Player", "Unnamed: 4_level_0_Squad",
                                  "Total_Cmp", "Total_Att", "Total_Cmp%", "Total_PrgDist",
                                  "Unnamed: 22_level_0_Ast", "Unnamed: 24_level_0_KP",
                                  "Unnamed: 25_level_0_1/3", "Unnamed: 26_level_0_PPA"]].copy()
    passing.columns = ["player", "team", "Pass_Cmp", "Pass_Att", "Pass_CmpPct", "Pass_PrgDist",
                        "Pass_Ast", "Pass_KP", "Pass_Final3rd", "Pass_PPA"]

    gca = frames["gca"][["Unnamed: 1_level_0_Player", "Unnamed: 4_level_0_Squad",
                          "SCA_SCA90", "GCA_GCA90"]].copy()
    gca.columns = ["player", "team", "SCA90", "GCA90"]

    poss = frames["possession"][["Unnamed: 1_level_0_Player", "Unnamed: 4_level_0_Squad",
                                  "Touches_Touches", "Carries_Carries", "Carries_PrgDist",
                                  "Carries_1/3", "Carries_CPA"]].copy()
    poss.columns = ["player", "team", "Touches", "Carries", "Carry_PrgDist", "Carry_Final3rd", "Carry_PPA"]

    df = (std
          .merge(sho, on=["player", "team"], how="left")
          .merge(misc, on=["player", "team"], how="left")
          .merge(passing, on=["player", "team"], how="left")
          .merge(gca, on=["player", "team"], how="left")
          .merge(poss, on=["player", "team"], how="left"))
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

    per90_raw_cols = [
        "Performance_Gls", "Performance_Ast", "Performance_G+A",
        "Standard_Sh", "Standard_SoT",
        "Performance_Fls", "Performance_Fld", "Performance_Off", "Performance_Crs",
        "Performance_Int", "Performance_TklW",
        "Pass_KP", "Pass_Final3rd", "Pass_PPA", "Pass_PrgDist",
        "Touches", "Carries", "Carry_PrgDist", "Carry_Final3rd", "Carry_PPA",
    ]
    native_cols = ["Pass_CmpPct", "SCA90", "GCA90"]

    for c in per90_raw_cols + native_cols:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0)

    feature_cols = []
    for c in per90_raw_cols:
        per90_col = c + "_p90"
        d[per90_col] = d[c] / d["NINETIES"]
        feature_cols.append(per90_col)
    feature_cols += native_cols  # already per-90 or percentage-based

    d["pos_simple"] = d["pos"].astype(str).str.split(",").str[0].str.strip()
    d = d.dropna(subset=feature_cols)
    d = d.drop_duplicates(subset=["player"], keep="first").reset_index(drop=True)

    print(f"  Feature matrix: {len(d)} players x {len(feature_cols)} features "
          f"(incl. passing, creation, possession)")
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


def validate_position_recovery(d, index, X_norm, k=5):
    """
    Quantitative validation substitute for FBref's own "Similar Players" feature,
    which was removed from the free site and is now Stathead-only (confirmed by
    direct check: zero mentions of "similar" on a player's scouting report page
    as of mid-2026, just Stathead subscription promos). Instead: for every player,
    check whether their top-k nearest neighbors share their position more often
    than a randomly chosen player would, by chance, given the dataset's actual
    position mix. This is the standard way to validate a similarity embedding
    when no external ground-truth labels exist.
    """
    print(f"\nValidating: do top-{k} matches share position more than chance? (n={len(d)})")
    n_total = len(d)
    pos_counts = d["pos_simple"].value_counts().to_dict()

    match_rates, baseline_rates = [], []
    for idx in range(n_total):
        query = X_norm[idx:idx+1]
        sims, ids = index.search(query, k + 1)
        own_pos = d.iloc[idx]["pos_simple"]
        neighbor_ids = [i for i in ids[0] if i != idx][:k]
        matches = sum(1 for i in neighbor_ids if d.iloc[i]["pos_simple"] == own_pos)
        match_rates.append(matches / k)
        n_p = pos_counts.get(own_pos, 0)
        baseline_rates.append((n_p - 1) / (n_total - 1))

    observed = float(np.mean(match_rates))
    baseline = float(np.mean(baseline_rates))
    print(f"  Observed top-{k} same-position rate: {observed:.1%}")
    print(f"  Random-chance baseline (by position mix): {baseline:.1%}")
    print(f"  Lift over random chance: {observed / baseline:.2f}x")
    return {"observed_rate": observed, "baseline_rate": baseline, "lift": observed / baseline}


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
                 f"n={len(d)} players, min {MIN_90S*90:.0f} minutes, incl. passing/creation/possession",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="best")
    ax.grid(color="#e0dbd5", lw=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR + "player_pca_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: player_pca_scatter.png")

    if len(comparisons_df) > 0:
        focus_player = "Kevin De Bruyne"
        sub = comparisons_df[comparisons_df["query_player"] == focus_player].sort_values("similarity")
        if len(sub) == 0:
            focus_player = SPOTLIGHT[0]
            sub = comparisons_df[comparisons_df["query_player"] == focus_player].sort_values("similarity")
        if len(sub) > 0:
            fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=BG)
            ax.set_facecolor(BG)
            ax.barh(sub["player"], sub["similarity"], color=RED, alpha=0.85)
            ax.set_xlabel("Cosine Similarity", fontsize=10)
            ax.set_title(f"Top-5 Most Similar Players to {focus_player}\n"
                         f"Premier League {SEASON[:2]}-{SEASON[2:]}, per-90 stat profile incl. passing",
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
    validation = validate_position_recovery(d, index, X_norm, k=5)
    comparisons_df = run_spotlight_comparisons(d, index, X_norm)
    generate_plots(d, feature_cols, comparisons_df, show=False)

    comparisons_df.to_csv(OUTPUTS_DIR + "spotlight_comparisons.csv", index=False)
    d.to_csv(OUTPUTS_DIR + "player_features.csv", index=False)
    pd.DataFrame([validation]).to_csv(OUTPUTS_DIR + "validation_metrics.csv", index=False)

    print(f"\n{'='*60}")
    print("  KEY FINDINGS -- Player Similarity Recommender")
    print(f"{'='*60}")
    print(f"  Players in index: {len(d)}")
    print(f"  Features used: {len(feature_cols)} (incl. passing, creation, possession)")
    print(f"  Top-5 same-position rate: {validation['observed_rate']:.1%} "
          f"vs {validation['baseline_rate']:.1%} random chance ({validation['lift']:.2f}x lift)")
    print(f"  Spotlight comparisons run: {comparisons_df['query_player'].nunique() if len(comparisons_df) else 0}")
    print(f"\nAll outputs saved to {OUTPUTS_DIR}")
