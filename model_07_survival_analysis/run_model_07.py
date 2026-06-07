"""
Model 07 — NBA Player Career Survival Analysis
Kaplan-Meier curves by position + Cox Proportional Hazard regression
Run: python run_model_07.py
"""
import asyncio, sys, os, time, warnings
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

OUTPUTS_DIR = "outputs/"
os.makedirs(OUTPUTS_DIR, exist_ok=True)

CACHE_STATS    = "../model_04_churn_model/outputs/raw_season_stats_extended.csv"
CACHE_DRAFT    = "../model_03_cohort_analysis/outputs/draft_history.csv"
CACHE_POS      = OUTPUTS_DIR + "player_positions.csv"
MIN_GP         = 20          # games to count as an active season
MIN_SEASONS    = 2           # minimum seasons to include a player
DEBUT_START    = 1996
CENSOR_SEASON  = 2025        # players still active — right-censored

POSITIONS = ["PG", "SG", "SF", "PF", "C"]

SPOTLIGHT = [
    "LeBron James", "Kobe Bryant", "Tim Duncan",
    "Dirk Nowitzki", "Shaquille O'Neal", "Kevin Durant",
    "Stephen Curry", "Dwyane Wade",
]


# ── Data loading ───────────────────────────────────────────────────────────────

def _fetch_player_index():
    from nba_api.stats.endpoints import playerindex
    print("  Fetching player index from nba_api...", end=" ", flush=True)
    time.sleep(1)
    r = playerindex.PlayerIndex(historical_nullable=1)
    df = r.get_data_frames()[0]
    df["PLAYER_NAME"] = df["PLAYER_FIRST_NAME"] + " " + df["PLAYER_LAST_NAME"]
    df = df[["PLAYER_NAME", "POSITION", "HEIGHT", "DRAFT_ROUND"]].copy()
    # Normalize position to primary (C-F -> C, G-F -> SF, etc.)
    def norm_pos(p):
        if pd.isna(p) or p == "":
            return "Unknown"
        p = str(p).strip()
        if p.startswith("G"):  return "SG"
        if p.startswith("F-C") or p.startswith("C-F"): return "PF"
        if p.startswith("F"): return "SF"
        if p.startswith("C"): return "C"
        return p
    df["PLAYER_POSITION"] = df["POSITION"].apply(norm_pos)
    df["DRAFT_ROUND"] = pd.to_numeric(df["DRAFT_ROUND"], errors="coerce").fillna(3).astype(int)
    # Height in inches
    def parse_height(h):
        try:
            ft, inch = str(h).split("-")
            return int(ft) * 12 + int(inch)
        except Exception:
            return np.nan
    df["HEIGHT_IN"] = df["HEIGHT"].apply(parse_height)
    print(f"{len(df)} players loaded")
    return df[["PLAYER_NAME", "PLAYER_POSITION", "HEIGHT_IN", "DRAFT_ROUND"]]


def load_data():
    print("Loading season stats...")
    stats = pd.read_csv(CACHE_STATS)
    stats["SEASON_YR"] = stats["SEASON"].str[:4].astype(int)
    stats = stats[stats["SEASON_YR"] >= DEBUT_START]
    print(f"  {len(stats):,} player-seasons | {stats['SEASON'].nunique()} seasons")

    print("Loading draft data...")
    try:
        draft = pd.read_csv(CACHE_DRAFT)
        draft = draft[["PLAYER_NAME", "SEASON", "ROUND_NUMBER"]].copy()
        draft.columns = ["PLAYER_NAME", "DRAFT_SEASON", "DRAFT_ROUND"]
        draft["DRAFT_ROUND"] = pd.to_numeric(draft["DRAFT_ROUND"], errors="coerce").fillna(3).astype(int)
    except Exception:
        draft = pd.DataFrame(columns=["PLAYER_NAME", "DRAFT_SEASON", "DRAFT_ROUND"])
        print("  Draft cache not found — round will default to 3 (undrafted)")

    print("Loading/fetching player index...")
    if os.path.exists(CACHE_POS):
        pos_df = pd.read_csv(CACHE_POS)
        print(f"  Loaded {len(pos_df)} players from cache")
    else:
        pos_df = _fetch_player_index()
        pos_df.to_csv(CACHE_POS, index=False)

    stats = stats.merge(pos_df[["PLAYER_NAME", "PLAYER_POSITION", "HEIGHT_IN", "DRAFT_ROUND"]], on="PLAYER_NAME", how="left")
    stats["DRAFT_ROUND"] = stats["DRAFT_ROUND"].fillna(3).astype(int)
    stats["HEIGHT_IN"]   = stats["HEIGHT_IN"].fillna(stats["HEIGHT_IN"].median())
    stats["POSITION"]    = stats["PLAYER_POSITION"].fillna("Unknown")

    print(f"  Position coverage: {stats['POSITION'].ne('Unknown').mean():.1%}")
    return stats


# ── Survival data construction ─────────────────────────────────────────────────

def build_survival_df(stats):
    print("\nBuilding survival dataset...")

    active = stats[stats["GP"] >= MIN_GP].copy()

    careers = (
        active.groupby("PLAYER_NAME")
        .agg(
            DEBUT_YR   = ("SEASON_YR", "min"),
            LAST_YR    = ("SEASON_YR", "max"),
            N_SEASONS  = ("SEASON_YR", "count"),
            PEAK_PTS   = ("PTS", "max"),
            PEAK_REB   = ("REB", "max"),
            PEAK_AST   = ("AST", "max"),
            DRAFT_ROUND= ("DRAFT_ROUND", "first"),
            POSITION   = ("POSITION", "first"),
            HEIGHT_IN  = ("HEIGHT_IN", "first"),
        )
        .reset_index()
    )

    careers = careers[careers["N_SEASONS"] >= MIN_SEASONS].copy()

    # Duration = seasons played
    # Event = 1 if career ended, 0 if still active (censored)
    careers["DURATION"] = careers["N_SEASONS"]
    careers["EVENT"]    = (careers["LAST_YR"] < CENSOR_SEASON).astype(int)

    # Composite peak performance (reuse model 04 formula)
    careers["PEAK_PERF"] = (
        careers["PEAK_PTS"]
        + 1.2 * careers["PEAK_REB"]
        + 1.5 * careers["PEAK_AST"]
    )

    careers = careers[careers["POSITION"].isin(POSITIONS)].copy()

    print(f"  Players: {len(careers)} | Events (career ended): {careers['EVENT'].sum()} | Censored: {(careers['EVENT']==0).sum()}")
    print(f"  Median career length: {careers['DURATION'].median():.1f} seasons")
    print(f"\n  By position:")
    print(careers.groupby("POSITION")[["DURATION","EVENT"]].agg({"DURATION":"median","EVENT":"mean"}).round(2).to_string())

    return careers


# ── Kaplan-Meier analysis ──────────────────────────────────────────────────────

def run_km_analysis(survival_df):
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import multivariate_logrank_test

    print("\nRunning Kaplan-Meier analysis...")
    kmf = {}
    for pos in POSITIONS:
        sub = survival_df[survival_df["POSITION"] == pos]
        if len(sub) < 10:
            continue
        kmf[pos] = KaplanMeierFitter()
        kmf[pos].fit(sub["DURATION"], sub["EVENT"], label=pos)
        med = kmf[pos].median_survival_time_
        print(f"  {pos}: n={len(sub):>3} | median survival={med:.1f} seasons")

    results = multivariate_logrank_test(
        survival_df["DURATION"],
        survival_df["POSITION"],
        survival_df["EVENT"],
    )
    print(f"\n  Log-rank test: p={results.p_value:.4f} {'(SIGNIFICANT)' if results.p_value < 0.05 else ''}")

    return kmf, results


# ── Cox regression ─────────────────────────────────────────────────────────────

def run_cox_regression(survival_df):
    from lifelines import CoxPHFitter

    print("\nRunning Cox Proportional Hazard regression...")

    df = survival_df.copy()
    pos_dummies = pd.get_dummies(df["POSITION"], prefix="POS", drop_first=True)
    df = pd.concat([df, pos_dummies], axis=1)

    features = ["DURATION", "EVENT", "PEAK_PERF", "DRAFT_ROUND", "HEIGHT_IN"] + list(pos_dummies.columns)
    df_cox = df[features].dropna()

    from sklearn.preprocessing import StandardScaler
    df_cox = df_cox.copy()
    scaler = StandardScaler()
    df_cox["PEAK_PERF"]  = scaler.fit_transform(df_cox[["PEAK_PERF"]])
    df_cox["HEIGHT_IN"]  = scaler.fit_transform(df_cox[["HEIGHT_IN"]])

    cph = CoxPHFitter()
    cph.fit(df_cox, duration_col="DURATION", event_col="EVENT")
    cph.print_summary()

    return cph, df_cox


# ── Spotlight player survival curves ──────────────────────────────────────────

def compute_spotlight_curves(survival_df, cph, df_cox):
    print("\nComputing spotlight player survival curves...")
    found = []
    for name in SPOTLIGHT:
        row = survival_df[survival_df["PLAYER_NAME"] == name]
        if not row.empty:
            r = row.iloc[0]
            found.append({
                "name": name,
                "duration": r["DURATION"],
                "event": r["EVENT"],
                "position": r["POSITION"],
            })
            status = "retired" if r["EVENT"] == 1 else f"active ({r['LAST_YR']+1 if 'LAST_YR' in r else 'current'})"
            print(f"  {name:28s} | {r['POSITION']} | {int(r['DURATION'])} seasons | {status}")
    return found


# ── Visualizations ────────────────────────────────────────────────────────────

def generate_plots(survival_df, kmf, results, cph, show=False):
    matplotlib.use("Agg")

    COLORS = {
        "PG": "#4C72B0",
        "SG": "#9B3D36",
        "SF": "#5a965a",
        "PF": "#c89b32",
        "C":  "#8A6BAE",
    }
    FONT = {"family": "DejaVu Sans", "size": 10}
    plt.rcParams.update({"font.family": FONT["family"], "font.size": FONT["size"]})

    # ── Plot 1: KM survival curves by position ─────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for pos, km in kmf.items():
        color = COLORS.get(pos, "#888")
        t = km.survival_function_.index
        s = km.survival_function_[pos]
        ci_lo = km.confidence_interval_[f"{pos}_lower_0.95"]
        ci_hi = km.confidence_interval_[f"{pos}_upper_0.95"]
        ax.step(t, s, where="post", color=color, linewidth=2, label=pos)
        ax.fill_between(t, ci_lo, ci_hi, step="post", alpha=0.08, color=color)

    ax.set_xlabel("Career Year", fontsize=11)
    ax.set_ylabel("Survival Probability", fontsize=11)
    ax.set_title(
        "NBA Career Survival by Position\n"
        f"Log-rank p={results.p_value:.4f}  ·  1996–2025  ·  n={len(survival_df)}",
        fontsize=12, fontweight="bold"
    )
    ax.set_xlim(1, 18)
    ax.set_ylim(0, 1.02)
    ax.axhline(0.5, color="#aaa", lw=1, ls="--", zorder=0)
    ax.text(17.2, 0.505, "50%", fontsize=8, color="#888", va="center")
    ax.legend(title="Position", fontsize=9, title_fontsize=9)
    ax.grid(axis="y", color="#e0dbd5", lw=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR + "survival_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: survival_curves.png")

    # ── Plot 2: Cox hazard ratio forest plot ──────────────────────────────
    summary = cph.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].copy()
    summary = summary.sort_values("exp(coef)")

    label_map = {
        "PEAK_PERF":   "Peak Performance (std)",
        "DRAFT_ROUND": "Draft Round",
        "POS_PF":      "Position: PF vs PG",
        "POS_SF":      "Position: SF vs PG",
        "POS_SG":      "Position: SG vs PG",
        "POS_C":       "Position: C vs PG",
    }
    summary.index = [label_map.get(i, i) for i in summary.index]

    fig, ax = plt.subplots(figsize=(8, max(4, len(summary) * 0.6 + 1)))
    y = range(len(summary))
    for i, (idx, row) in enumerate(summary.iterrows()):
        color = "#9B3D36" if row["p"] < 0.05 else "#6b7a8d"
        ax.plot([row["exp(coef) lower 95%"], row["exp(coef) upper 95%"]], [i, i],
                color=color, lw=2, solid_capstyle="round")
        ax.scatter(row["exp(coef)"], i, color=color, s=60, zorder=3)

    ax.axvline(1.0, color="#aaa", lw=1.2, ls="--", zorder=0)
    ax.set_yticks(list(y))
    ax.set_yticklabels(summary.index, fontsize=9)
    ax.set_xlabel("Hazard Ratio (exp coef)  —  >1 = higher churn risk", fontsize=10)
    ax.set_title("Cox PH Model — Hazard Ratios\nRed = significant (p < 0.05)", fontsize=12, fontweight="bold")
    ax.grid(axis="x", color="#e0dbd5", lw=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR + "cox_hazard_ratios.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: cox_hazard_ratios.png")

    if show:
        plt.show()


# ── Print findings ─────────────────────────────────────────────────────────────

def print_findings(survival_df, kmf, results, cph):
    print(f"\n{'='*60}")
    print("  KEY FINDINGS — NBA Career Survival Analysis")
    print(f"{'='*60}")
    print(f"  Players analyzed:      {len(survival_df)}")
    print(f"  Careers ended:         {survival_df['EVENT'].sum()}")
    print(f"  Still active:          {(survival_df['EVENT']==0).sum()}")
    print(f"  Overall median career: {survival_df['DURATION'].median():.1f} seasons")
    print(f"  Log-rank p-value:      {results.p_value:.4f}")

    print(f"\n  Median survival by position:")
    for pos, km in sorted(kmf.items(), key=lambda x: x[1].median_survival_time_):
        print(f"    {pos}: {km.median_survival_time_:.1f} seasons")

    print(f"\n  Cox regression top hazard ratios:")
    hr = cph.summary["exp(coef)"].sort_values(ascending=False)
    for feat, val in hr.items():
        p = cph.summary.loc[feat, "p"]
        sig = " *" if p < 0.05 else ""
        print(f"    {feat:<22} HR={val:.3f}  p={p:.3f}{sig}")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    matplotlib.use("Agg")

    stats       = load_data()
    survival_df = build_survival_df(stats)
    kmf, results = run_km_analysis(survival_df)
    cph, df_cox  = run_cox_regression(survival_df)
    spotlight    = compute_spotlight_curves(survival_df, cph, df_cox)
    generate_plots(survival_df, kmf, results, cph, show=False)
    print_findings(survival_df, kmf, results, cph)
    print(f"\nAll outputs saved to {OUTPUTS_DIR}")
