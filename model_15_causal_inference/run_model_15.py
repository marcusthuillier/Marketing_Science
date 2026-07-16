"""
Model 15: Causal Inference on Manager Changes in La Liga
Did a manager change actually improve a team's results, or was it regression to the mean?
Uses StatsBomb free data + ITS regression + CausalImpact (Bayesian structural time series)
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
from scipy import stats

# Force UTF-8 output to handle accented manager names
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Monkey-patch for pandas 2.x compatibility with causalimpact
if not hasattr(pd.core.dtypes.common, "is_datetime_or_timedelta_dtype"):
    def _is_dt_td_dtype(x):
        try:
            return (pd.api.types.is_datetime64_any_dtype(x)
                    or pd.api.types.is_timedelta64_dtype(x))
        except Exception:
            return False
    pd.core.dtypes.common.is_datetime_or_timedelta_dtype = _is_dt_td_dtype

OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
CACHE_PATH = os.path.join(OUTPUTS_DIR, "_xg_cache.pkl")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# STEP 1: Load match data from StatsBomb
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading StatsBomb La Liga match data")
print("=" * 60)

from statsbombpy import sb

# 2015/16 (season_id=27): FULL 380-match season — all 20 teams
# Others: Barcelona-only data (~33-36 matches each)
SEASON_MAP = {
    27: "2015/16",
    42: "2019/20",
    4:  "2018/19",
    1:  "2017/18",
    90: "2020/21",
}

all_matches = {}
for sid, sname in SEASON_MAP.items():
    m = sb.matches(competition_id=11, season_id=sid)
    all_matches[sid] = m
    print(f"  Season {sname} (id={sid}): {len(m)} matches")


# ─────────────────────────────────────────────────────────────
# STEP 2: Auto-detect manager changes per team per season
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Detecting manager changes")
print("=" * 60)


def get_team_matches(matches_df, team):
    tm = matches_df[
        (matches_df["home_team"] == team) | (matches_df["away_team"] == team)
    ].copy().sort_values("match_date").reset_index(drop=True)
    mgrs = []
    for _, row in tm.iterrows():
        if row["home_team"] == team:
            mgrs.append(str(row.get("home_manager_name", "Unknown")))
        else:
            mgrs.append(str(row.get("away_manager_name", "Unknown")))
    tm["team_manager"] = mgrs
    tm["team"] = team
    return tm


all_changes = []

for sid, matches_df in all_matches.items():
    season_name = SEASON_MAP[sid]
    teams = list(set(matches_df["home_team"].tolist() + matches_df["away_team"].tolist()))
    for team in sorted(teams):
        tm = get_team_matches(matches_df, team)
        for i in range(1, len(tm)):
            prev_mgr = tm.iloc[i - 1]["team_manager"]
            curr_mgr = tm.iloc[i]["team_manager"]
            if curr_mgr == prev_mgr:
                continue
            # Filter: skip if either manager is null/nan/unknown/caretaker combo
            if (prev_mgr in ("nan", "None", "Unknown", "")
                    or curr_mgr in ("nan", "None", "Unknown", "")
                    or "," in prev_mgr or "," in curr_mgr):
                continue
            n_before = i
            n_after = len(tm) - i
            if n_before >= 10 and n_after >= 10:
                all_changes.append({
                    "season_id": sid,
                    "season_name": season_name,
                    "team": team,
                    "manager_before": prev_mgr,
                    "manager_after": curr_mgr,
                    "change_date": tm.iloc[i]["match_date"],
                    "change_index": i,
                    "n_before": n_before,
                    "n_after": n_after,
                    "team_matches": tm,
                })

print(f"  Found {len(all_changes)} manager changes with >=10 matches before and after")
for c in all_changes:
    mb = c["manager_before"].split()[-1]
    ma = c["manager_after"].split()[-1]
    print(f"    {c['team']} ({c['season_name']}): {mb} -> {ma} "
          f"| {c['change_date']} | before={c['n_before']} after={c['n_after']}")


# ─────────────────────────────────────────────────────────────
# STEP 3: Compute xGD per match (with disk cache)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Fetching xG from StatsBomb events (disk-cached)")
print("=" * 60)

# Load existing cache from disk if available
if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, "rb") as f:
        XG_CACHE = pickle.load(f)
    print(f"  Loaded {len(XG_CACHE)} cached match xG records")
else:
    XG_CACHE = {}
    print("  No cache found — will fetch all events")


def get_match_xgd(match_id, team):
    if match_id not in XG_CACHE:
        try:
            evts = sb.events(match_id=match_id)
            shots = evts[evts["type"] == "Shot"][["team", "shot_statsbomb_xg"]].copy()
            shots["shot_statsbomb_xg"] = pd.to_numeric(
                shots["shot_statsbomb_xg"], errors="coerce").fillna(0)
            XG_CACHE[match_id] = shots.groupby("team")["shot_statsbomb_xg"].sum().to_dict()
        except Exception:
            XG_CACHE[match_id] = {}
    xg_dict = XG_CACHE[match_id]
    team_xg = float(xg_dict.get(team, 0.0))
    opp_xg = float(sum(v for k, v in xg_dict.items() if k != team))
    return team_xg, opp_xg


def build_xgd_series(tm_df):
    result = []
    for _, row in tm_df.iterrows():
        t_xg, o_xg = get_match_xgd(row["match_id"], row["team"])
        result.append(t_xg - o_xg)
    return np.array(result)


# Identify unique (season_id, team) pairs needed
needed = {}
for c in all_changes:
    key = (c["season_id"], c["team"])
    if key not in needed:
        needed[key] = c["team_matches"]

total_matches = sum(len(v) for v in needed.values())
cached_matches = sum(
    1 for (_, tm) in needed.items()
    for _, row in tm.iterrows()
    if row["match_id"] in XG_CACHE
)
print(f"  Need {total_matches} matches for {len(needed)} team-seasons "
      f"({cached_matches} already cached)")

xgd_store = {}
t0 = time.time()
done = 0
for (sid, team), tm_df in needed.items():
    new_fetches = sum(1 for _, r in tm_df.iterrows() if r["match_id"] not in XG_CACHE)
    if new_fetches > 0:
        print(f"  Fetching {SEASON_MAP[sid]} {team} "
              f"({new_fetches} new + {len(tm_df)-new_fetches} cached)...",
              end=" ", flush=True)
    xgd = build_xgd_series(tm_df)
    xgd_store[(sid, team)] = xgd
    done += len(tm_df)
    if new_fetches > 0:
        print(f"done. ({done}/{total_matches}, {time.time()-t0:.0f}s)")

# Save updated cache
with open(CACHE_PATH, "wb") as f:
    pickle.dump(XG_CACHE, f)
print(f"  Cache saved ({len(XG_CACHE)} records). Total time: {time.time()-t0:.0f}s")


# ─────────────────────────────────────────────────────────────
# STEP 4: Naive before/after analysis
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Naive before/after analysis")
print("=" * 60)

results = []

for c in all_changes:
    sid = c["season_id"]
    team = c["team"]
    ci_idx = c["change_index"]
    xgd = xgd_store.get((sid, team), np.array([]))
    if len(xgd) == 0:
        continue

    xgd_before = xgd[:ci_idx]
    xgd_after = xgd[ci_idx:]
    mean_before = float(np.mean(xgd_before))
    mean_after = float(np.mean(xgd_after))
    naive_pct_change = (
        (mean_after - mean_before) / abs(mean_before) * 100
        if mean_before != 0 else 0.0
    )
    _, p_val_ttest = stats.ttest_ind(xgd_after, xgd_before)
    naive_improved = mean_after > mean_before
    season_avg_xgd = float(np.mean(xgd))
    rtm_gap = mean_before - season_avg_xgd  # negative = below season avg before change

    mgr_label = c["manager_after"].split()[-1]
    mb_last = c["manager_before"].split()[-1]

    res = {
        "season_name": c["season_name"],
        "team": team,
        "manager_before": c["manager_before"],
        "manager_after": c["manager_after"],
        "mgr_label": mgr_label,
        "change_date": c["change_date"],
        "n_before": int(c["n_before"]),
        "n_after": int(c["n_after"]),
        "naive_before_xgd": round(mean_before, 4),
        "naive_after_xgd": round(mean_after, 4),
        "naive_pct_change": round(naive_pct_change, 2),
        "naive_improved": bool(naive_improved),
        "ttest_p": round(float(p_val_ttest), 4),
        "season_avg_xgd": round(season_avg_xgd, 4),
        "rtm_gap": round(float(rtm_gap), 4),
        "xgd_series": xgd.tolist(),
        "change_index": ci_idx,
    }
    results.append(res)
    print(f"  {team} ({c['season_name']}): {mb_last} -> {mgr_label}")
    print(f"    Before: {mean_before:+.3f} | After: {mean_after:+.3f} "
          f"| Change: {naive_pct_change:+.1f}% | t-test p={p_val_ttest:.3f}")
    print(f"    Season avg: {season_avg_xgd:+.3f} | RTM gap: {rtm_gap:+.3f}")

print(f"\n  Analyzed {len(results)} manager changes")
if len(results) == 0:
    print("ERROR: No results.")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────
# STEP 5: Causal analysis — ITS regression + CausalImpact (UCM)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Causal analysis — ITS + CausalImpact (UCM)")
print("=" * 60)

import statsmodels.api as sm
from statsmodels.tsa.statespace.structural import UnobservedComponents

try:
    from causalimpact import CausalImpact
    HAS_CI = True
except ImportError:
    HAS_CI = False
    print("  WARNING: causalimpact not available.")


def run_its_regression(xgd_series, change_index):
    """
    Interrupted Time Series (ITS) regression.
    xGD = b0 + b1*time + b2*intervention + b3*(time-T)*intervention + error
    b2 = immediate level change (primary causal estimate, pre-intervention trend removed)
    b3 = slope change after the intervention
    """
    n = len(xgd_series)
    t = np.arange(n, dtype=float)
    D = (t >= change_index).astype(float)           # intervention dummy
    P = np.maximum(t - change_index, 0) * D         # post-intervention time trend
    X = sm.add_constant(np.column_stack([t, D, P]))
    try:
        fit = sm.OLS(xgd_series, X).fit()
        return {
            "its_level_change": round(float(fit.params[2]), 4),
            "its_level_pvalue": round(float(fit.pvalues[2]), 4),
            "its_slope_change": round(float(fit.params[3]), 4),
            "its_slope_pvalue": round(float(fit.pvalues[3]), 4),
            "its_r2": round(float(fit.rsquared), 4),
        }
    except Exception as e:
        print(f"    ITS error: {e}")
        return None


def run_causal_impact_ucm(xgd_series, change_index, alpha=0.1):
    """
    CausalImpact (Bayesian structural time series, local level).
    Trains a structural model on the pre-period, extrapolates a counterfactual
    into the post-period, then measures actual vs counterfactual.

    NOTE on this causalimpact library's CI column naming:
    'point_effect_lower' and 'point_effect_upper' appear reversed from convention.
    We use min/max to get the true CI bounds.
    """
    if not HAS_CI:
        return None
    pre = np.array(xgd_series[:change_index], dtype=float)
    post = np.array(xgd_series[change_index:], dtype=float)
    if len(pre) < 5 or len(post) < 3:
        return None
    try:
        y_full = pd.Series(
            np.concatenate([pre, np.full(len(post), np.nan)]), dtype=float)
        ucm = UnobservedComponents(y_full, level="local level")
        ci_model = CausalImpact(
            ucm_model=ucm,
            post_period_response=pd.Series(post, dtype=float),
            alpha=alpha,
        )
        ci_model.run()
        inf = ci_model.inferences
        if inf is None:
            return None

        post_rows = inf.iloc[change_index:]
        avg_effect = float(post_rows["point_effect"].mean())

        # Columns are named 'lower' and 'upper' but in this library
        # point_effect_lower is the HIGH end and _upper is the LOW end.
        # We use min/max to get the real CI bounds.
        raw_lower = float(post_rows["point_effect_lower"].mean())
        raw_upper = float(post_rows["point_effect_upper"].mean())
        real_ci_low = min(raw_lower, raw_upper)
        real_ci_high = max(raw_lower, raw_upper)

        predicted_xgd = float(post_rows["point_pred"].mean())
        actual_xgd = float(post_rows["response"].mean())

        # Significant if CI excludes zero
        causal_significant = bool(real_ci_low > 0 or real_ci_high < 0)

        return {
            "causal_estimate": round(avg_effect, 4),
            "causal_ci_low": round(real_ci_low, 4),
            "causal_ci_high": round(real_ci_high, 4),
            "causal_significant": causal_significant,
            "predicted_xgd": round(predicted_xgd, 4),
            "actual_xgd": round(actual_xgd, 4),
        }
    except Exception as e:
        print(f"    CausalImpact error: {e}")
        return None


for res in results:
    xgd = np.array(res["xgd_series"])
    ci_idx = res["change_index"]
    print(f"\n  {res['team']} ({res['season_name']}) -> {res['mgr_label']}")

    its = run_its_regression(xgd, ci_idx)
    if its:
        res.update(its)
        print(f"    ITS level change: {its['its_level_change']:+.3f} xGD "
              f"(p={its['its_level_pvalue']:.3f}), "
              f"slope: {its['its_slope_change']:+.3f} (p={its['its_slope_pvalue']:.3f})")
    else:
        for k in ["its_level_change", "its_level_pvalue",
                  "its_slope_change", "its_slope_pvalue", "its_r2"]:
            res[k] = np.nan

    ci_r = run_causal_impact_ucm(xgd, ci_idx)
    if ci_r:
        res.update(ci_r)
        print(f"    CausalImpact: avg effect={ci_r['causal_estimate']:+.3f} "
              f"(90% CI: [{ci_r['causal_ci_low']:+.3f}, {ci_r['causal_ci_high']:+.3f}]) "
              f"significant={ci_r['causal_significant']}")
    else:
        for k in ["causal_estimate", "causal_ci_low", "causal_ci_high",
                  "predicted_xgd", "actual_xgd"]:
            res[k] = np.nan
        res["causal_significant"] = False

    # Primary causal significance = ITS level-change p < 0.1
    its_p = res.get("its_level_pvalue", 1.0)
    res["causal_significant_its"] = bool(
        not np.isnan(its_p) and its_p < 0.1)


# ─────────────────────────────────────────────────────────────
# STEP 6: Regression to mean analysis
# ─────────────────────────────────────────────────────────────
print("\n\n" + "=" * 60)
print("STEP 6: Regression to mean analysis")
print("=" * 60)

# RTM risk: team's pre-change xGD was well below their seasonal average
rtm_threshold = -0.15  # 0.15 xGD/match below seasonal avg

rtm_cases = [r for r in results if r["rtm_gap"] < rtm_threshold]
non_rtm_cases = [r for r in results if r["rtm_gap"] >= rtm_threshold]

print(f"  RTM-risk cases (pre-change xGD > {abs(rtm_threshold):.2f} below season avg): {len(rtm_cases)}")
print(f"  Non-RTM cases: {len(non_rtm_cases)}")

for r in rtm_cases:
    print(f"    {r['team']} ({r['season_name']}): "
          f"season avg={r['season_avg_xgd']:+.3f}, "
          f"pre-change avg={r['naive_before_xgd']:+.3f} "
          f"(gap={r['rtm_gap']:+.3f})")

rtm_misleading = [r for r in rtm_cases
                  if r["naive_improved"] and not r.get("causal_significant_its", False)]
print(f"\n  RTM-risk where naive showed improvement but ITS was NOT significant: "
      f"{len(rtm_misleading)}/{len(rtm_cases)}")
for r in rtm_misleading:
    print(f"    {r['team']} ({r['season_name']}): "
          f"naive +{r['naive_pct_change']:.1f}% | ITS p={r.get('its_level_pvalue',float('nan')):.3f}")


# ─────────────────────────────────────────────────────────────
# STEP 7: Meta-analysis
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: Meta-analysis")
print("=" * 60)

n_total = len(results)
n_naive_improved = sum(1 for r in results if r["naive_improved"])
n_its_sig = sum(1 for r in results if r.get("causal_significant_its", False))
n_ci_sig = sum(1 for r in results if r.get("causal_significant", False))

pct_naive = n_naive_improved / n_total * 100 if n_total else 0.0
pct_its = n_its_sig / n_total * 100 if n_total else 0.0
pct_ci = n_ci_sig / n_total * 100 if n_total else 0.0

avg_naive_pct = float(np.mean([r["naive_pct_change"] for r in results]))
valid_its = [r["its_level_change"] for r in results
             if not np.isnan(r.get("its_level_change", float("nan")))]
avg_its = float(np.mean(valid_its)) if valid_its else 0.0
valid_ci = [r["causal_estimate"] for r in results
            if not np.isnan(r.get("causal_estimate", float("nan")))]
avg_ci = float(np.mean(valid_ci)) if valid_ci else 0.0

most_improved = max(results, key=lambda r: r["naive_pct_change"])
most_disappoint = min(results, key=lambda r: r["naive_pct_change"])

ci_valid = [r for r in results if not np.isnan(r.get("causal_estimate", float("nan")))]
most_causal = max(ci_valid, key=lambda r: r["causal_estimate"]) if ci_valid else None

print(f"  Manager changes analyzed: {n_total}")
print(f"  Naive before/after showed improvement: {n_naive_improved}/{n_total} ({pct_naive:.0f}%)")
print(f"  ITS found significant effect (p<0.1): {n_its_sig}/{n_total} ({pct_its:.0f}%)")
print(f"  CausalImpact (UCM) found significant: {n_ci_sig}/{n_total} ({pct_ci:.0f}%)")
print(f"  Average naive % change: {avg_naive_pct:+.1f}%")
print(f"  Average ITS level change: {avg_its:+.4f} xGD/match")
print(f"  Average CausalImpact effect: {avg_ci:+.4f} xGD/match")
print(f"  Most improved (naive): {most_improved['team']} -> "
      f"{most_improved['mgr_label']} ({most_improved['naive_pct_change']:+.1f}%)")
print(f"  Most disappointing (naive): {most_disappoint['team']} -> "
      f"{most_disappoint['mgr_label']} ({most_disappoint['naive_pct_change']:+.1f}%)")
if most_causal:
    print(f"  Strongest causal effect: {most_causal['team']} -> "
          f"{most_causal['mgr_label']} ({most_causal['causal_estimate']:+.4f} xGD)")
print(f"  RTM-risk misleading cases: {len(rtm_misleading)}/{max(len(rtm_cases),1)} "
      f"({len(rtm_misleading)/max(len(rtm_cases),1)*100:.0f}%)")


# ─────────────────────────────────────────────────────────────
# STEP 8: Save outputs
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8: Saving outputs")
print("=" * 60)

# 8a. manager_changes_results.csv
csv_rows = []
for r in results:
    csv_rows.append({
        "season": r["season_name"],
        "team": r["team"],
        "manager_before": r["manager_before"],
        "manager_after": r["manager_after"],
        "change_date": r["change_date"],
        "n_before": r["n_before"],
        "n_after": r["n_after"],
        "naive_before_xgd": r["naive_before_xgd"],
        "naive_after_xgd": r["naive_after_xgd"],
        "naive_pct_change": r["naive_pct_change"],
        "naive_improved": r["naive_improved"],
        "ttest_p": r["ttest_p"],
        "season_avg_xgd": r["season_avg_xgd"],
        "rtm_gap": r["rtm_gap"],
        "its_level_change": r.get("its_level_change", np.nan),
        "its_level_pvalue": r.get("its_level_pvalue", np.nan),
        "its_slope_change": r.get("its_slope_change", np.nan),
        "its_slope_pvalue": r.get("its_slope_pvalue", np.nan),
        "causal_significant_its": r.get("causal_significant_its", False),
        "causal_estimate_ucm": r.get("causal_estimate", np.nan),
        "causal_ci_low_ucm": r.get("causal_ci_low", np.nan),
        "causal_ci_high_ucm": r.get("causal_ci_high", np.nan),
        "causal_significant_ucm": r.get("causal_significant", False),
    })

df_results = pd.DataFrame(csv_rows)
path_results = os.path.join(OUTPUTS_DIR, "manager_changes_results.csv")
df_results.to_csv(path_results, index=False, encoding="utf-8")
print(f"  Saved: {path_results}")

# 8b. naive_vs_causal.csv
comp_rows = []
for r in results:
    comp_rows.append({
        "team": r["team"],
        "manager": r["mgr_label"],
        "season": r["season_name"],
        "change_date": r["change_date"],
        "naive_improved": r["naive_improved"],
        "naive_pct_change": r["naive_pct_change"],
        "its_level_change": r.get("its_level_change", np.nan),
        "its_p_value": r.get("its_level_pvalue", np.nan),
        "its_significant": r.get("causal_significant_its", False),
        "causal_impact_estimate": r.get("causal_estimate", np.nan),
        "causal_impact_significant": r.get("causal_significant", False),
        "rtm_risk": r["rtm_gap"] < rtm_threshold,
    })

df_comp = pd.DataFrame(comp_rows)
path_comp = os.path.join(OUTPUTS_DIR, "naive_vs_causal.csv")
df_comp.to_csv(path_comp, index=False, encoding="utf-8")
print(f"  Saved: {path_comp}")

# 8c. key_metrics.txt
metrics = {
    "n_changes_analyzed": n_total,
    "pct_naive_improvement": round(pct_naive, 1),
    "pct_its_significant": round(pct_its, 1),
    "pct_causal_impact_significant": round(pct_ci, 1),
    "avg_naive_pct_change": round(avg_naive_pct, 2),
    "avg_its_level_change_xgd": round(avg_its, 4),
    "avg_causal_effect_xgd": round(avg_ci, 4),
    "most_improved_team": most_improved["team"],
    "most_improved_manager": most_improved["mgr_label"],
    "most_improved_pct": most_improved["naive_pct_change"],
    "most_disappointing_team": most_disappoint["team"],
    "most_disappointing_manager": most_disappoint["mgr_label"],
    "most_disappointing_pct": most_disappoint["naive_pct_change"],
    "n_rtm_risk_cases": len(rtm_cases),
    "n_rtm_misleading": len(rtm_misleading),
    "rtm_mislead_pct": round(len(rtm_misleading) / max(len(rtm_cases), 1) * 100, 1),
    "rtm_risk_teams": ",".join(r["team"] for r in rtm_cases),
}
if most_causal:
    metrics["strongest_causal_team"] = most_causal["team"]
    metrics["strongest_causal_manager"] = most_causal["mgr_label"]
    metrics["strongest_causal_estimate"] = most_causal["causal_estimate"]

path_metrics = os.path.join(OUTPUTS_DIR, "key_metrics.txt")
with open(path_metrics, "w", encoding="utf-8") as f:
    for k, v in metrics.items():
        f.write(f"{k}={v}\n")
print(f"  Saved: {path_metrics}")

# 8d. Final summary
print("\n" + "=" * 60)
print("FINAL RESULTS SUMMARY")
print("=" * 60)
print(df_comp.to_string(index=False))

print("\n" + "=" * 60)
print("KEY METRICS")
print("=" * 60)
for k, v in metrics.items():
    print(f"  {k} = {v}")

print("\nDone.")
