"""
Model 12: MLB Ticket Price Elasticity
======================================
Estimates price elasticity of demand for MLB games using a log-log OLS
regression. Identification comes from cross-sectional price variation between
teams within the same year (anchor years where actual TMR prices exist).
Derives revenue-maximizing pricing direction by segment.

Data sources:
  - Attendance: Baseball Reference (scraped 2011, 2013, 2015)
  - Win %: pybaseball.standings() via Baseball Reference
  - Ticket prices: Team Marketing Report / Fan Cost Index (actual TMR data,
    2011/2013/2015 from ESPN annual FCI reports citing TMR)

Author: Marcus Thuillier
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. ATTENDANCE (Baseball Reference home attendance) ───────────────────────
# Scraped from baseball-reference.com/leagues/MLB/{year}-misc.shtml
ATTENDANCE = {
    (2011, "ARI"): 2105432, (2011, "ATL"): 2372940, (2011, "BAL"): 1755461,
    (2011, "BOS"): 3054001, (2011, "CHN"): 3017966, (2011, "CHA"): 2001117,
    (2011, "CIN"): 2213588, (2011, "CLE"): 1840835, (2011, "COL"): 2909777,
    (2011, "DET"): 2642045, (2011, "FLO"): 1520562, (2011, "HOU"): 2067016,
    (2011, "KCA"): 1724450, (2011, "ANA"): 3166321, (2011, "LAN"): 2935139,
    (2011, "MIL"): 3071373, (2011, "MIN"): 3168116, (2011, "NYN"): 2352596,
    (2011, "NYA"): 3653680, (2011, "OAK"): 1476791, (2011, "PHI"): 3680718,
    (2011, "PIT"): 1940429, (2011, "SDN"): 2143018, (2011, "SFN"): 3387303,
    (2011, "SEA"): 1896321, (2011, "SLN"): 3093954, (2011, "TBA"): 1529188,
    (2011, "TEX"): 2946949, (2011, "TOR"): 1818103, (2011, "WAS"): 1940478,

    (2013, "ARI"): 2134895, (2013, "ATL"): 2548679, (2013, "BAL"): 2357561,
    (2013, "BOS"): 2833333, (2013, "CHN"): 2642682, (2013, "CHA"): 1768413,
    (2013, "CIN"): 2492101, (2013, "CLE"): 1572926, (2013, "COL"): 2793828,
    (2013, "DET"): 3083397, (2013, "HOU"): 1651883, (2013, "KCA"): 1750754,
    (2013, "ANA"): 3019505, (2013, "LAN"): 3743527, (2013, "MIA"): 1586322,
    (2013, "MIL"): 2531105, (2013, "MIN"): 2477644, (2013, "NYN"): 2135657,
    (2013, "NYA"): 3279589, (2013, "OAK"): 1809302, (2013, "PHI"): 3012403,
    (2013, "PIT"): 2256862, (2013, "SDN"): 2166691, (2013, "SFN"): 3369106,
    (2013, "SEA"): 1761546, (2013, "SLN"): 3369769, (2013, "TBA"): 1510300,
    (2013, "TEX"): 3178273, (2013, "TOR"): 2536562, (2013, "WAS"): 2652422,

    (2015, "ARI"): 2080145, (2015, "ATL"): 2001392, (2015, "BAL"): 2281202,
    (2015, "BOS"): 2880694, (2015, "CHN"): 2919122, (2015, "CHA"): 1755810,
    (2015, "CIN"): 2419506, (2015, "CLE"): 1388905, (2015, "COL"): 2506789,
    (2015, "DET"): 2726048, (2015, "HOU"): 2153585, (2015, "KCA"): 2708549,
    (2015, "ANA"): 3012765, (2015, "LAN"): 3764815, (2015, "MIA"): 1752235,
    (2015, "MIL"): 2542558, (2015, "MIN"): 2220054, (2015, "NYN"): 2569753,
    (2015, "NYA"): 3193795, (2015, "OAK"): 1768175, (2015, "PHI"): 1831080,
    (2015, "PIT"): 2498596, (2015, "SDN"): 2459752, (2015, "SFN"): 3375882,
    (2015, "SEA"): 2193581, (2015, "SLN"): 3520889, (2015, "TBA"): 1287054,
    (2015, "TEX"): 2491875, (2015, "TOR"): 2794891, (2015, "WAS"): 2619843,
}

# ── 2. WIN PCT (pybaseball.standings() via Baseball Reference) ────────────────
WIN_PCT = {
    (2011, "NYA"): 0.599, (2011, "TBA"): 0.562, (2011, "BOS"): 0.556,
    (2011, "TOR"): 0.500, (2011, "BAL"): 0.426, (2011, "DET"): 0.586,
    (2011, "CLE"): 0.494, (2011, "CHA"): 0.488, (2011, "KCA"): 0.438,
    (2011, "MIN"): 0.389, (2011, "TEX"): 0.593, (2011, "ANA"): 0.531,
    (2011, "OAK"): 0.457, (2011, "SEA"): 0.414, (2011, "PHI"): 0.630,
    (2011, "ATL"): 0.549, (2011, "WAS"): 0.497, (2011, "NYN"): 0.475,
    (2011, "FLO"): 0.444, (2011, "MIL"): 0.593, (2011, "SLN"): 0.556,
    (2011, "CIN"): 0.488, (2011, "PIT"): 0.444, (2011, "CHN"): 0.438,
    (2011, "HOU"): 0.346, (2011, "ARI"): 0.580, (2011, "SFN"): 0.531,
    (2011, "LAN"): 0.509, (2011, "COL"): 0.451, (2011, "SDN"): 0.438,

    (2013, "BOS"): 0.599, (2013, "TBA"): 0.564, (2013, "BAL"): 0.525,
    (2013, "NYA"): 0.525, (2013, "TOR"): 0.457, (2013, "DET"): 0.574,
    (2013, "CLE"): 0.568, (2013, "KCA"): 0.531, (2013, "MIN"): 0.407,
    (2013, "CHA"): 0.389, (2013, "OAK"): 0.593, (2013, "TEX"): 0.558,
    (2013, "ANA"): 0.481, (2013, "SEA"): 0.438, (2013, "HOU"): 0.315,
    (2013, "ATL"): 0.593, (2013, "WAS"): 0.531, (2013, "NYN"): 0.457,
    (2013, "PHI"): 0.451, (2013, "MIA"): 0.383, (2013, "SLN"): 0.599,
    (2013, "PIT"): 0.580, (2013, "CIN"): 0.556, (2013, "MIL"): 0.457,
    (2013, "CHN"): 0.407, (2013, "LAN"): 0.568, (2013, "ARI"): 0.500,
    (2013, "SDN"): 0.469, (2013, "SFN"): 0.469, (2013, "COL"): 0.457,

    (2015, "TOR"): 0.574, (2015, "NYA"): 0.537, (2015, "BAL"): 0.500,
    (2015, "TBA"): 0.494, (2015, "BOS"): 0.481, (2015, "KCA"): 0.586,
    (2015, "MIN"): 0.512, (2015, "CLE"): 0.503, (2015, "CHA"): 0.469,
    (2015, "DET"): 0.460, (2015, "TEX"): 0.543, (2015, "HOU"): 0.531,
    (2015, "ANA"): 0.525, (2015, "SEA"): 0.469, (2015, "OAK"): 0.420,
    (2015, "NYN"): 0.556, (2015, "WAS"): 0.512, (2015, "MIA"): 0.438,
    (2015, "ATL"): 0.414, (2015, "PHI"): 0.389, (2015, "SLN"): 0.617,
    (2015, "PIT"): 0.605, (2015, "CHN"): 0.599, (2015, "MIL"): 0.420,
    (2015, "CIN"): 0.395, (2015, "LAN"): 0.568, (2015, "SFN"): 0.519,
    (2015, "ARI"): 0.488, (2015, "SDN"): 0.457, (2015, "COL"): 0.420,
}

# ── 3. TICKET PRICES (Team Marketing Report / Fan Cost Index, actual data) ───
ANCHOR_PRICES = {
    2011: {
        "ARI": 15.74, "ATL": 19.38, "BAL": 23.90, "BOS": 53.38,
        "CHN": 46.90, "CHA": 40.67, "CIN": 20.56, "CLE": 18.49,
        "COL": 19.50, "DET": 29.32, "FLO": 19.06, "HOU": 30.84,
        "KCA": 18.95, "ANA": 17.13, "LAN": 30.59, "MIL": 22.10,
        "MIN": 33.04, "NYN": 31.81, "NYA": 51.83, "OAK": 21.52,
        "PHI": 36.29, "PIT": 15.30, "SDN": 15.45, "SFN": 25.04,
        "SEA": 26.40, "SLN": 31.17, "TBA": 19.42, "TEX": 18.60,
        "TOR": 24.35, "WAS": 30.54,
    },
    2013: {
        "ARI": 16.89, "ATL": 17.32, "BAL": 23.89, "BOS": 53.38,
        "CHN": 44.55, "CHA": 26.05, "CIN": 21.35, "CLE": 19.59,
        "COL": 23.65, "DET": 26.36, "MIA": 29.27, "HOU": 30.09,
        "KCA": 19.83, "ANA": 27.54, "LAN": 22.37, "MIL": 24.95,
        "MIN": 32.59, "NYN": 25.30, "NYA": 51.55, "OAK": 22.12,
        "PHI": 37.42, "PIT": 17.21, "SDN": 15.99, "SFN": 30.09,
        "SEA": 28.45, "SLN": 33.11, "TBA": 20.39, "TEX": 22.54,
        "TOR": 32.98, "WAS": 35.24,
    },
    2015: {
        "ARI": 17.98, "ATL": 19.14, "BAL": 24.97, "BOS": 52.34,
        "CHN": 44.81, "CHA": 26.05, "CIN": 22.03, "CLE": 22.38,
        "COL": 23.65, "DET": 29.01, "MIA": 28.96, "HOU": 31.82,
        "KCA": 29.76, "ANA": 27.54, "LAN": 28.61, "MIL": 26.32,
        "MIN": 32.59, "NYN": 25.30, "NYA": 51.55, "OAK": 24.00,
        "PHI": 37.42, "PIT": 19.99, "SDN": 16.37, "SFN": 33.78,
        "SEA": 31.00, "SLN": 34.20, "TBA": 21.90, "TEX": 23.64,
        "TOR": 25.14, "WAS": 36.02,
    },
}

# ── 4. PLAYOFF FLAGS ──────────────────────────────────────────────────────────
PLAYOFF_TEAMS = {
    2011: {"NYA", "DET", "TEX", "TBA", "PHI", "MIL", "ARI", "ATL"},
    2013: {"BOS", "DET", "OAK", "TBA", "CLE", "ATL", "SLN", "LAN", "PIT", "CIN"},
    2015: {"TOR", "KCA", "TEX", "HOU", "NYA", "NYN", "SLN", "LAN", "PIT", "CHN"},
}

NEW_STADIUM = {
    ("MIN", 2010): 1, ("MIA", 2012): 1, ("ATL", 2017): 1,
}

# ── 5. BUILD DATASET ──────────────────────────────────────────────────────────
def build_dataset():
    records = []
    for year, prices in ANCHOR_PRICES.items():
        for team, price in prices.items():
            att     = ATTENDANCE.get((year, team))
            win_pct = WIN_PCT.get((year, team))
            if att is None or win_pct is None:
                continue
            records.append({
                "team":             team,
                "year":             year,
                "attendance":       att,
                "win_pct":          win_pct,
                "avg_ticket_price": price,
                "playoff":          1 if team in PLAYOFF_TEAMS.get(year, set()) else 0,
                "new_stadium":      NEW_STADIUM.get((team, year), 0),
            })

    df = pd.DataFrame(records)
    df["log_attendance"] = np.log(df["attendance"])
    df["log_price"]      = np.log(df["avg_ticket_price"])
    df["log_win_pct"]    = np.log(df["win_pct"])
    df["is_playoff_caliber"] = (
        (df["playoff"] == 1) | (df["win_pct"] > 0.550)
    ).astype(int)

    print(f"Dataset: {len(df)} obs  |  {df['team'].nunique()} teams  |  years: {sorted(df['year'].unique())}")
    print(f"Price range: ${df['avg_ticket_price'].min():.2f} - ${df['avg_ticket_price'].max():.2f}")
    print(f"Attendance range: {df['attendance'].min():,} - {df['attendance'].max():,}")
    return df

# ── 6. REGRESSION ─────────────────────────────────────────────────────────────
def run_ols(data, label=""):
    formula = "log_attendance ~ log_price + log_win_pct + playoff + new_stadium + C(year)"
    try:
        model = smf.ols(formula, data=data).fit(cov_type="HC3")
        return model
    except Exception as e:
        print(f"  OLS error ({label}): {e}")
        return None

# ── 7. OUTPUTS ────────────────────────────────────────────────────────────────
def save_outputs(df, model_all, model_po, model_npo):
    df_out = df.copy()
    df_out["predicted_log_att"] = model_all.predict(df_out)
    df_out["residual"]          = df_out["log_attendance"] - df_out["predicted_log_att"]
    df_out["segment"] = df_out["is_playoff_caliber"].map({1: "playoff_caliber", 0: "non_playoff"})
    cols = ["team","year","segment","attendance","avg_ticket_price","win_pct",
            "playoff","new_stadium","log_attendance","log_price","predicted_log_att","residual"]
    df_out[cols].to_csv(os.path.join(OUT_DIR, "elasticity_results.csv"), index=False)

    seg_rows = []
    for model, lbl in [(model_all,"overall"),(model_po,"playoff_caliber"),(model_npo,"non_playoff")]:
        if model is None:
            continue
        e    = model.params.get("log_price", np.nan)
        se   = model.bse.get("log_price", np.nan)
        pval = model.pvalues.get("log_price", np.nan)
        rec  = ("underpricing - inelastic demand (raise price to grow revenue)"
                if abs(e) < 1 else
                "overpricing risk - elastic demand (lower price may grow revenue)")
        seg_rows.append({
            "segment": lbl, "elasticity": round(e,4), "std_error": round(se,4),
            "p_value": round(pval,4), "r_squared": round(model.rsquared,4),
            "n_obs": int(model.nobs), "pricing_recommendation": rec,
        })
    pd.DataFrame(seg_rows).to_csv(os.path.join(OUT_DIR, "segment_elasticity.csv"), index=False)

    e_all = round(model_all.params.get("log_price", np.nan), 4)
    e_po  = round(model_po.params.get("log_price",  np.nan), 4) if model_po  else np.nan
    e_npo = round(model_npo.params.get("log_price", np.nan), 4) if model_npo else np.nan

    lines = [
        f"n_observations={len(df)}",
        f"n_teams={df['team'].nunique()}",
        f"n_seasons={df['year'].nunique()}",
        f"anchor_years=2011,2013,2015",
        f"overall_elasticity={e_all}",
        f"playoff_elasticity={e_po}",
        f"non_playoff_elasticity={e_npo}",
        f"r_squared={round(model_all.rsquared, 4)}",
        f"pricing_recommendation_playoff={'underpricing - inelastic demand' if (not np.isnan(e_po) and abs(e_po)<1) else 'overpricing risk - elastic demand'}",
        f"pricing_recommendation_non_playoff={'underpricing - inelastic demand' if (not np.isnan(e_npo) and abs(e_npo)<1) else 'overpricing risk - elastic demand'}",
    ]
    with open(os.path.join(OUT_DIR, "key_metrics.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    return e_all, e_po, e_npo

# ── 8. CHART ──────────────────────────────────────────────────────────────────
def save_chart(df, e_all, e_po, e_npo):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.patch.set_facecolor("#0f1117")
    for ax in axes:
        ax.set_facecolor("#1a1d27")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    ax = axes[0]
    palette = {1: "#4fc3f7", 0: "#ef9a9a"}
    labels  = {1: "Playoff-caliber", 0: "Non-playoff"}
    for seg, grp in df.groupby("is_playoff_caliber"):
        ax.scatter(grp["log_price"], grp["log_attendance"],
                   alpha=0.55, s=30, color=palette[seg], label=labels[seg])
    for seg, grp in df.groupby("is_playoff_caliber"):
        x = grp["log_price"].values
        y = grp["log_attendance"].values
        m, b = np.polyfit(x, y, 1)
        xr = np.linspace(x.min(), x.max(), 50)
        ax.plot(xr, m*xr+b, color=palette[seg], linewidth=2.2)
    ax.set_xlabel("log(Avg Ticket Price $)", color="white", fontsize=11)
    ax.set_ylabel("log(Home Attendance)", color="white", fontsize=11)
    ax.set_title("Price vs. Attendance  (log-log)", color="white", fontsize=12, fontweight="bold")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#1a1d27", edgecolor="#444", labelcolor="white", fontsize=9)
    ax.text(0.04, 0.07, f"Overall elasticity = {e_all:.3f}",
            transform=ax.transAxes, color="white", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#333", alpha=0.8))

    ax2 = axes[1]
    segs  = ["Playoff-caliber", "Non-playoff", "Overall"]
    vals  = [e_po if not np.isnan(e_po) else 0,
             e_npo if not np.isnan(e_npo) else 0,
             e_all]
    bars = ax2.barh(segs, vals, color=["#4fc3f7","#ef9a9a","#aaaaaa"], edgecolor="#444", height=0.5)
    ax2.axvline(x=-1, color="#ff6b6b", linestyle="--", linewidth=1.5, label="|e|=1 threshold")
    ax2.axvline(x=0,  color="#888",    linestyle="-",  linewidth=0.8)
    ax2.set_xlabel("Price Elasticity of Demand", color="white", fontsize=11)
    ax2.set_title("Elasticity by Segment", color="white", fontsize=12, fontweight="bold")
    ax2.tick_params(colors="white")
    ax2.legend(facecolor="#1a1d27", edgecolor="#444", labelcolor="white", fontsize=9)
    for bar, val in zip(bars, vals):
        x_pos = val - 0.05 if val < 0 else val + 0.02
        ax2.text(x_pos, bar.get_y()+bar.get_height()/2, f"{val:.3f}",
                 va="center", ha="right" if val<0 else "left",
                 color="white", fontsize=10, fontweight="bold")
    xlim = ax2.get_xlim()
    for val, y_pos in zip(vals, range(len(segs))):
        lbl   = "Inelastic" if abs(val) < 1 else "Elastic"
        color = "#4caf50"   if abs(val) < 1 else "#ff5722"
        ax2.text(xlim[1], y_pos, f"  {lbl}", va="center", ha="left", fontsize=9, color=color)

    plt.tight_layout(pad=2.0)
    plt.savefig(os.path.join(OUT_DIR, "elasticity_by_segment.png"),
                dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print("  Saved elasticity_by_segment.png")

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Model 12: MLB Ticket Price Elasticity")
    print("=" * 60)

    df = build_dataset()
    df.to_csv(os.path.join(OUT_DIR, "raw_panel_data.csv"), index=False)

    print("\nRunning regressions...")
    model_all = run_ols(df,                                 "overall")
    model_po  = run_ols(df[df["is_playoff_caliber"]==1],   "playoff")
    model_npo = run_ols(df[df["is_playoff_caliber"]==0],   "non-playoff")

    print("Saving outputs...")
    e_all, e_po, e_npo = save_outputs(df, model_all, model_po, model_npo)
    save_chart(df, e_all, e_po, e_npo)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    def fmt(e, model):
        if np.isnan(e) or model is None:
            return "n/a"
        pval  = model.pvalues.get("log_price", np.nan)
        stars = "***" if pval<0.01 else "**" if pval<0.05 else "*" if pval<0.10 else "(ns)"
        tag   = "INELASTIC" if abs(e)<1 else "ELASTIC"
        r2    = model.rsquared
        return f"e={e:+.4f}  [{tag}]  p={pval:.3f}{stars}  R2={r2:.3f}"

    print(f"  Overall:     {fmt(e_all, model_all)}")
    print(f"  Playoff:     {fmt(e_po,  model_po)}")
    print(f"  Non-playoff: {fmt(e_npo, model_npo)}")
    print(f"\n  N={len(df)}  |  30 teams  |  3 anchor years (2011, 2013, 2015)")

    if model_all:
        print("\n  Coefficient summary:")
        for var in ["log_price", "log_win_pct", "playoff", "new_stadium"]:
            if var in model_all.params:
                c  = model_all.params[var]
                pv = model_all.pvalues[var]
                s  = "***" if pv<0.01 else "**" if pv<0.05 else "*" if pv<0.10 else "  "
                print(f"    {var:15s}  coef={c:+.4f}  p={pv:.4f} {s}")

    print("\nDone. Outputs:", OUT_DIR)

if __name__ == "__main__":
    main()
