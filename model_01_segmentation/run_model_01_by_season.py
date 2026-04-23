"""
Run Model 01 for three NBA eras, each averaged across 3 seasons.
Usage: python run_model_01_by_season.py
"""
import asyncio, sys, time, warnings, os
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, pairwise_distances

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
MIN_MINUTES  = 750           # total minutes across all seasons in the era window
K_RANGE      = range(4, 11)

ERAS = {
    "Early 2000s (03-06)": ["2003-04", "2004-05", "2005-06"],
    "Mid 2010s  (13-16)":  ["2013-14", "2014-15", "2015-16"],
    "Modern     (23-26)":  ["2023-24", "2024-25", "2025-26"],
}

FEATURES = [
    "USG_PCT", "AST_PCT", "REB_PCT", "OREB_PCT", "DREB_PCT",
    "TS_PCT", "E_TOV_PCT", "PACE",
]

SPOTLIGHT_PLAYERS = [
    "Nikola Jokic", "Stephen Curry", "Giannis Antetokounmpo", "Rudy Gobert",
    "LeBron James", "Kevin Durant", "Joel Embiid", "Jayson Tatum",
    "Dwyane Wade", "Kobe Bryant", "Shaquille O'Neal",
    "Tim Duncan", "Dirk Nowitzki", "James Harden", "Russell Westbrook",
    "Allen Iverson", "Paul Pierce", "Carmelo Anthony", "Chris Paul",
]

def smart_label(norm_row, raw_row):
    """
    Multi-stat rule-based labeler.
    norm_row  — 0-1 normalized profile (relative position vs other clusters)
    raw_row   — actual mean stats (for absolute grounding)
    """
    usg  = norm_row.get("USG_PCT",  0)
    ast  = norm_row.get("AST_PCT",  0)
    dreb = norm_row.get("DREB_PCT", 0)
    oreb = norm_row.get("OREB_PCT", 0)
    reb  = norm_row.get("REB_PCT",  0)
    ts   = norm_row.get("TS_PCT",   0)

    # Raw absolute anchors — prevents normalisation artefacts
    r_dreb = raw_row.get("DREB_PCT", 0)   # >0.17 = clear big-man territory
    r_ast  = raw_row.get("AST_PCT",  0)   # >0.20 = genuine playmaking
    r_usg  = raw_row.get("USG_PCT",  0)   # >0.24 = high usage

    is_big       = dreb > 0.60 or (r_dreb > 0.16)
    is_playmaker = ast  > 0.60 or (r_ast  > 0.18)
    is_scorer    = usg  > 0.60 or (r_usg  > 0.23)
    is_role      = (not is_big) and (not is_playmaker) and (not is_scorer)

    if is_big and is_scorer and not is_playmaker:
        return "Scoring Big / Power Forward"
    if is_big and is_playmaker:
        return "Versatile Big / Modern Center"
    if is_big and oreb > 0.55:
        return "Glass Anchor / Traditional Center"
    if is_big:
        return "Rim Protector / Defensive Big"
    if is_playmaker and is_scorer:
        return "Ball-Dominant Playmaker"
    if is_playmaker:
        return "Pass-First Guard"
    if is_scorer and ts > 0.55:
        return "Efficient Primary Scorer"
    if is_scorer:
        return "High-Volume Wing"
    if is_role:
        return "3-and-D / Role Player"
    # fallback: look at relative peaks
    peaks = {"USG": usg, "AST": ast, "REB": reb}
    top = max(peaks, key=peaks.get)
    return {"USG": "Versatile Scorer", "AST": "Combo Guard", "REB": "Stretch Big"}.get(top, "Versatile Wing")

def fetch_season(season):
    from nba_api.stats.endpoints import leaguedashplayerstats
    print(f"    Fetching {season}...", end=" ", flush=True)
    time.sleep(1.2)
    resp = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        per_mode_detailed="Totals",
        measure_type_detailed_defense="Advanced"
    )
    df = resp.get_data_frames()[0]
    df["SEASON"] = season
    df["MIN_TOTAL"] = df["MIN"] * df["GP"]
    print(f"{len(df)} players")
    return df

def run_era(era_name, seasons):
    print(f"\n{'='*55}")
    print(f"  ERA: {era_name}")
    print(f"  Seasons: {', '.join(seasons)}")
    print(f"{'='*55}")

    out_dir = f"outputs/{era_name.split('(')[0].strip().replace(' ','_')}/"
    os.makedirs(out_dir, exist_ok=True)

    # ── Fetch all seasons ──────────────────────────────────
    frames = [fetch_season(s) for s in seasons]
    raw = pd.concat(frames, ignore_index=True)

    # ── Average rate stats per player across seasons ───────
    feats = [f for f in FEATURES if f in raw.columns]
    agg   = {f: "mean" for f in feats}
    agg["MIN_TOTAL"] = "sum"
    agg["GP"]        = "sum"

    df = (
        raw.groupby("PLAYER_NAME", as_index=False)
           .agg(agg)
    )
    print(f"\n  Unique players across era: {len(df)}")

    df = df[df["MIN_TOTAL"] >= MIN_MINUTES].dropna(subset=feats).reset_index(drop=True)
    print(f"  After {MIN_MINUTES}+ total min filter + dropna: {len(df)} players")

    X = StandardScaler().fit_transform(df[feats])

    # ── K selection ────────────────────────────────────────
    silhouettes = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        silhouettes.append(silhouette_score(X, km.fit_predict(X)))
    best_k   = list(K_RANGE)[silhouettes.index(max(silhouettes))]
    best_sil = max(silhouettes)
    print(f"  Best K={best_k}  (silhouette={best_sil:.3f})")

    # ── Final clustering ───────────────────────────────────
    km_final     = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=20)
    df["CLUSTER"] = km_final.fit_predict(X)

    profiles     = df.groupby("CLUSTER")[feats].mean().round(3)
    profile_norm = (profiles - profiles.min()) / (profiles.max() - profiles.min() + 1e-9)
    labels       = {i: smart_label(profile_norm.loc[i], profiles.loc[i]) for i in range(best_k)}

    seen = {}
    for idx, lbl in labels.items():
        if lbl in seen:
            seen[lbl] += 1
            labels[idx] = f"{lbl} {seen[lbl]}"
        else:
            seen[lbl] = 1

    df["ARCHETYPE"] = df["CLUSTER"].map(labels)

    centroids              = km_final.cluster_centers_
    dists                  = pairwise_distances(X, centroids)
    df["DIST_TO_CENTROID"] = [dists[i, c] for i, c in enumerate(df["CLUSTER"])]

    # ── Print archetypes with stats + top 5 players ────────
    print(f"\n  Archetypes ({best_k} clusters):")
    stat_cols = ["USG_PCT", "AST_PCT", "DREB_PCT", "OREB_PCT", "TS_PCT"]
    stat_cols = [s for s in stat_cols if s in feats]
    cluster_summary = []
    for c in range(best_k):
        sub  = df[df["CLUSTER"] == c]
        top5 = sub.nsmallest(5, "DIST_TO_CENTROID")["PLAYER_NAME"].tolist()
        size = len(sub)
        arch = labels[c]
        raw  = profiles.loc[c]
        stat_str = "  ".join(f"{s}: {raw[s]:.3f}" for s in stat_cols)
        print(f"\n    [{arch}] ({size} players)")
        print(f"      Stats : {stat_str}")
        print(f"      Top 5 : {', '.join(top5)}")
        cluster_summary.append({
            "archetype": arch, "size": size, "top3": top5[:3],
            "profile": profiles.loc[c, feats].to_dict()
        })

    # ── Spotlight players ──────────────────────────────────
    print(f"\n  Spotlight players:")
    spotlight_results = {}
    for name in SPOTLIGHT_PLAYERS:
        match = df[df["PLAYER_NAME"] == name]
        if not match.empty:
            row = match.iloc[0]
            print(f"    {row['PLAYER_NAME']:28s} -> {row['ARCHETYPE']}")
            spotlight_results[row["PLAYER_NAME"]] = row["ARCHETYPE"]

    df.to_csv(out_dir + "clustered_players.csv", index=False)

    return {
        "era": era_name, "seasons": seasons,
        "n_players": len(df), "K": best_k, "silhouette": round(best_sil, 3),
        "clusters": cluster_summary, "spotlight": spotlight_results,
        "profiles": profiles, "labels": labels,
    }

# ── Run all three eras ─────────────────────────────────────────────────────────
results = {}
for era_name, seasons in ERAS.items():
    results[era_name] = run_era(era_name, seasons)

# ── Cross-era comparison ───────────────────────────────────────────────────────
print(f"\n\n{'='*55}")
print("  CROSS-ERA COMPARISON")
print(f"{'='*55}\n")

print(f"{'Era':<28} {'Players':>8} {'K':>4} {'Silhouette':>12}")
print("-" * 56)
for era, r in results.items():
    print(f"{era:<28} {r['n_players']:>8} {r['K']:>4} {r['silhouette']:>12.3f}")

print(f"\n  Archetypes by era:")
for era, r in results.items():
    archs = [c["archetype"] for c in r["clusters"]]
    print(f"  {era}: {', '.join(archs)}")

print(f"\n  Spotlight player archetype shifts:")
all_players = set()
for r in results.values():
    all_players.update(r["spotlight"].keys())

era_names = list(ERAS.keys())
header = f"  {'Player':<28}" + "".join(f" | {e[:14]:<14}" for e in era_names)
print(header)
print("  " + "-" * (28 + 18 * len(era_names)))
for player in sorted(all_players):
    row_parts = [f"{player:<28}"]
    for era in era_names:
        arch = results[era]["spotlight"].get(player, "--")
        row_parts.append(f"{arch:<16}")
    print("  " + " | ".join(row_parts))

print(f"\n  Archetype size by era:")
for era, r in results.items():
    print(f"\n  {era}:")
    for c in sorted(r["clusters"], key=lambda x: -x["size"]):
        print(f"    {c['archetype']:<28} {c['size']:>3} players  | {', '.join(c['top3'])}")

print(f"\n{'='*55}")
print("  DONE")
print(f"{'='*55}")
