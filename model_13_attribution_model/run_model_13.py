"""
Model 13: Football Attribution Model
Question: When a goal is scored, who really deserves credit?
How much does the answer change depending on which attribution model you use?
Marketing parallel: last-click vs linear vs Shapley attribution.

Data: StatsBomb free open data - La Liga 2015/2016 (Messi's last Barca title)
"""

import pandas as pd
import numpy as np
from statsbombpy import sb
import warnings
import time
from pathlib import Path

warnings.filterwarnings('ignore')

# ─── SETUP ────────────────────────────────────────────────────
OUTPUT_DIR = Path(r'C:\Users\mthui\OneDrive\Desktop\Code\Marketing_Science\model_13_attribution_model\outputs')
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 65)
print("MODEL 13: FOOTBALL ATTRIBUTION MODEL")
print("La Liga 2015/2016 - Messi's Last Barca Title")
print("Last-touch vs Linear vs Shapley attribution on goal sequences")
print("=" * 65)

# ─── STEP 1: DATA PULL ────────────────────────────────────────
print("\n[STEP 1] Loading match list...")

matches = sb.matches(competition_id=11, season_id=27)
n_total = len(matches)
print(f"Total available matches: {n_total}")

# Sample up to 100 matches for speed
SAMPLE_SIZE = 100
sampled = matches.sample(n=min(SAMPLE_SIZE, n_total), random_state=42)
print(f"Sampled {len(sampled)} matches for analysis")

# ─── STEP 2: POSSESSION CHAIN RECONSTRUCTION ──────────────────
print("\n[STEP 2] Extracting possession chains leading to goals...")
t0 = time.time()

all_chains = []
n_processed = 0
n_errors = 0

for i, (_, match_row) in enumerate(sampled.iterrows()):
    match_id = int(match_row['match_id'])
    try:
        events = sb.events(match_id=match_id)

        # Identify goals: Shot events with outcome = Goal
        goals = events[
            (events['type'] == 'Shot') &
            (events['shot_outcome'] == 'Goal')
        ]

        for _, goal_event in goals.iterrows():
            possession_num = goal_event['possession']
            scorer = goal_event['player']
            if pd.isna(scorer):
                continue

            # Get the full possession chain, sorted by StatsBomb event index
            chain = (
                events[events['possession'] == possession_num]
                .sort_values('index')
                .reset_index(drop=True)
            )

            # Find where this shot sits in the chain
            shot_mask = chain['id'] == goal_event['id']
            shot_positions = chain.index[shot_mask].tolist()
            if not shot_positions:
                continue
            shot_pos = shot_positions[0]  # integer position after reset_index

            # Keep last 10 events up to and including the shot
            window_start = max(0, shot_pos - 9)
            window = chain.iloc[window_start: shot_pos + 1]

            # Players in window (drop NaN player rows — tactical events, etc.)
            players_in_window = window[window['player'].notna()]['player'].tolist()

            if not players_in_window:
                continue

            # Touch counts
            touch_counts = {}
            for p in players_in_window:
                touch_counts[p] = touch_counts.get(p, 0) + 1

            # Unique players (insertion order = first appearance)
            seen: set = set()
            unique_players: list = []
            for p in players_in_window:
                if p not in seen:
                    seen.add(p)
                    unique_players.append(p)

            # Ensure scorer is present
            if scorer not in seen:
                unique_players.append(scorer)
                touch_counts[scorer] = touch_counts.get(scorer, 0) + 1
                seen.add(scorer)

            # ── Identify assister ──────────────────────────────
            assister = None

            # Method 1: follow shot_key_pass_id to the pass event
            kp_id = goal_event.get('shot_key_pass_id', None)
            if kp_id is not None and pd.notna(kp_id):
                kp_row = events[events['id'] == kp_id]
                if len(kp_row) > 0:
                    ap = kp_row.iloc[0]['player']
                    if pd.notna(ap) and ap != scorer:
                        assister = ap

            # Method 2: pass_goal_assist flag within the window
            if assister is None and 'pass_goal_assist' in window.columns:
                assist_rows = window[window['pass_goal_assist'] == True]
                if len(assist_rows) > 0:
                    ap = assist_rows.iloc[-1]['player']
                    if pd.notna(ap) and ap != scorer:
                        assister = ap

            # If assister not in chain window players, ignore them
            if assister and assister not in seen:
                assister = None

            all_chains.append({
                'goal_id': str(goal_event['id']),
                'match_id': match_id,
                'possession': int(possession_num),
                'scorer': scorer,
                'assister': assister or '',
                'n_players': len(unique_players),
                'chain_length': len(window),
                'unique_players': unique_players,
                'touch_counts': touch_counts,
            })

        n_processed += 1
        if (i + 1) % 25 == 0 or (i + 1) == len(sampled):
            elapsed = time.time() - t0
            print(f"  {i+1:3d}/{len(sampled)} matches | {len(all_chains):4d} goals | "
                  f"{elapsed:.0f}s elapsed")

    except Exception as e:
        n_errors += 1
        if n_errors <= 5:
            print(f"  Warning: match {match_id}: {e}")

print(f"\nGoals collected:  {len(all_chains)}")
print(f"Matches processed:{n_processed}  |  Errors: {n_errors}")

if not all_chains:
    print("ERROR: No goals found. Exiting.")
    raise SystemExit(1)

# ─── STEP 3: ATTRIBUTION MODELS ───────────────────────────────
print("\n[STEP 3] Computing three attribution models per goal...")

attr_rows = []

for chain in all_chains:
    players      = chain['unique_players']
    touches      = chain['touch_counts']
    scorer       = chain['scorer']
    assister     = chain['assister'] or None
    n            = len(players)
    goal_id      = chain['goal_id']

    # ── Last-touch: scorer gets 100% ──────────────────────────
    lt = {p: 0.0 for p in players}
    lt[scorer] = 1.0

    # ── Linear: proportional to touch count ───────────────────
    total_touches = sum(touches.get(p, 0) for p in players)
    if total_touches > 0:
        lin = {p: touches.get(p, 0) / total_touches for p in players}
    else:
        lin = {p: 1.0 / n for p in players}

    # ── Shapley (role-weighted approximation) ─────────────────
    shap = {p: 0.0 for p in players}

    if n == 1:
        shap[scorer] = 1.0

    elif n == 2:
        other = [p for p in players if p != scorer]
        shap[scorer] = 0.55
        if other:
            shap[other[0]] = 0.45

    else:  # n >= 3
        buildup = [p for p in players if p != scorer and p != assister]

        if assister and assister in players:
            # scorer 40%, assister 30%, build-up share 30%
            shap[scorer]   = 0.40
            shap[assister] = 0.30
            bu_share = 0.30 / len(buildup) if buildup else 0.0
            for p in buildup:
                shap[p] = bu_share
        else:
            # No identified assister: scorer 55%, build-up 45%
            shap[scorer] = 0.55
            bu_share = 0.45 / (n - 1)
            for p in players:
                if p != scorer:
                    shap[p] = bu_share

    # Normalise (guard against floating-point drift)
    for credit_dict in [lt, lin, shap]:
        s = sum(credit_dict.values())
        if s > 0 and abs(s - 1.0) > 1e-6:
            for p in credit_dict:
                credit_dict[p] /= s

    for p in players:
        attr_rows.append({
            'goal_id':      goal_id,
            'match_id':     chain['match_id'],
            'player':       p,
            'is_scorer':    p == scorer,
            'is_assister':  p == assister if assister else False,
            'last_touch':   lt.get(p, 0.0),
            'linear':       lin.get(p, 0.0),
            'shapley':      shap.get(p, 0.0),
            'n_players':    n,
            'chain_length': chain['chain_length'],
        })

attr_df = pd.DataFrame(attr_rows)
print(f"Attribution records created: {len(attr_df):,}")

# ─── STEP 4: AGGREGATE BY PLAYER ──────────────────────────────
print("\n[STEP 4] Aggregating credit by player across all goals...")

n_goals = len(all_chains)

player_agg = (
    attr_df
    .groupby('player')
    .agg(
        last_touch_credit =('last_touch', 'sum'),
        linear_credit     =('linear',     'sum'),
        shapley_credit    =('shapley',    'sum'),
        goals_scored      =('is_scorer',  'sum'),
        goals_assisted    =('is_assister','sum'),
        chains_involved   =('goal_id',    'nunique'),
    )
    .reset_index()
)

# Percentage of total goals (normalized credit)
player_agg['lt_pct']   = player_agg['last_touch_credit'] / n_goals * 100
player_agg['lin_pct']  = player_agg['linear_credit']     / n_goals * 100
player_agg['shap_pct'] = player_agg['shapley_credit']    / n_goals * 100

# Rankings (1 = best)
player_agg['last_touch_rank'] = (
    player_agg['last_touch_credit']
    .rank(ascending=False, method='min')
    .astype(int)
)
player_agg['shapley_rank'] = (
    player_agg['shapley_credit']
    .rank(ascending=False, method='min')
    .astype(int)
)
# positive rank_change = rose under Shapley vs Last-touch
player_agg['rank_change'] = player_agg['last_touch_rank'] - player_agg['shapley_rank']

# Shapley-to-last-touch ratio (how much more/less Shapley rewards vs last-touch)
player_agg['shap_to_lt_ratio'] = player_agg.apply(
    lambda r: (r['shapley_credit'] / r['last_touch_credit'])
              if r['last_touch_credit'] > 0 else np.inf,
    axis=1
)

# ── Top-10 tables ──────────────────────────────────────────────
top10_lt   = player_agg.nlargest(10, 'last_touch_credit')[
    ['player', 'last_touch_credit', 'goals_scored']
].reset_index(drop=True)

top10_lin  = player_agg.nlargest(10, 'linear_credit')[
    ['player', 'linear_credit', 'chains_involved']
].reset_index(drop=True)

top10_shap = player_agg.nlargest(10, 'shapley_credit')[
    ['player', 'shapley_credit', 'chains_involved']
].reset_index(drop=True)

print("\nTop 10 by Last-touch:")
print(top10_lt.to_string(index=False))
print("\nTop 10 by Linear (touch-weighted):")
print(top10_lin.to_string(index=False))
print("\nTop 10 by Shapley:")
print(top10_shap.to_string(index=False))

# ── Most undervalued by last-touch ────────────────────────────
# = player with high Shapley credit but low last-touch credit
# (high Shapley rank, low last-touch rank => involved a lot but rarely scored)
eligible = player_agg[
    (player_agg['chains_involved'] >= 4) &
    (player_agg['shapley_credit'] > 0)
].copy()

# Undervalued: biggest improvement in rank under Shapley
if len(eligible) > 0:
    undervalued_row = eligible.loc[eligible['rank_change'].idxmax()]
    most_undervalued = undervalued_row['player']
    uv_lt_rank   = int(undervalued_row['last_touch_rank'])
    uv_shap_rank = int(undervalued_row['shapley_rank'])
else:
    most_undervalued = player_agg.loc[player_agg['shapley_credit'].idxmax(), 'player']
    uv_lt_rank, uv_shap_rank = 0, 0

# Overvalued: biggest drop in rank under Shapley (scored lots, lower Shapley)
eligible_ov = player_agg[player_agg['goals_scored'] >= 3].copy()
if len(eligible_ov) > 0:
    overvalued_row = eligible_ov.loc[eligible_ov['rank_change'].idxmin()]
    most_overvalued = overvalued_row['player']
    ov_lt_rank   = int(overvalued_row['last_touch_rank'])
    ov_shap_rank = int(overvalued_row['shapley_rank'])
else:
    most_overvalued = "N/A"
    ov_lt_rank, ov_shap_rank = 0, 0

print(f"\nMost UNDERVALUED by last-touch vs Shapley: {most_undervalued}")
print(f"  Last-touch rank: {uv_lt_rank}  |  Shapley rank: {uv_shap_rank}")
print(f"Most OVERVALUED by last-touch vs Shapley: {most_overvalued}")
print(f"  Last-touch rank: {ov_lt_rank}  |  Shapley rank: {ov_shap_rank}")

# ─── STEP 5: METHOD DISAGREEMENT ──────────────────────────────
print("\n[STEP 5] Quantifying disagreement between attribution methods...")

scorer_rows = attr_df[attr_df['is_scorer'] == True]
scorer_lt_avg   = scorer_rows['last_touch'].mean() * 100
scorer_lin_avg  = scorer_rows['linear'].mean()     * 100
scorer_shap_avg = scorer_rows['shapley'].mean()    * 100

lt_shap_swing = scorer_lt_avg - scorer_shap_avg

print(f"Average credit to scorer per goal:")
print(f"  Last-touch : {scorer_lt_avg:.1f}%")
print(f"  Linear     : {scorer_lin_avg:.1f}%")
print(f"  Shapley    : {scorer_shap_avg:.1f}%")
print(f"Swing (last-touch vs Shapley): {lt_shap_swing:.1f} percentage points")

avg_chain_length = np.mean([c['chain_length'] for c in all_chains])
avg_n_players    = np.mean([c['n_players']    for c in all_chains])
n_with_assist    = sum(1 for c in all_chains if c['assister'])

# Top player by each method
top_lt_player   = player_agg.loc[player_agg['last_touch_credit'].idxmax(), 'player']
top_lt_credit   = player_agg['last_touch_credit'].max()
top_lt_pct      = top_lt_credit / n_goals * 100

top_lin_player  = player_agg.loc[player_agg['linear_credit'].idxmax(), 'player']
top_lin_credit  = player_agg['linear_credit'].max()
top_lin_pct     = top_lin_credit / n_goals * 100

top_shap_player = player_agg.loc[player_agg['shapley_credit'].idxmax(), 'player']
top_shap_credit = player_agg['shapley_credit'].max()
top_shap_pct    = top_shap_credit / n_goals * 100

print(f"\nTop player by last-touch : {top_lt_player} "
      f"({top_lt_credit:.1f} credits = {top_lt_pct:.1f}% of goals)")
print(f"Top player by Shapley    : {top_shap_player} "
      f"({top_shap_credit:.1f} credits = {top_shap_pct:.1f}% of goals)")

# ─── SAVE OUTPUTS ─────────────────────────────────────────────
print("\n[SAVING] Writing output files...")

# 1. goal_chains.csv
chains_df = pd.DataFrame([{
    'goal_id':      c['goal_id'],
    'match_id':     c['match_id'],
    'n_players':    c['n_players'],
    'scorer':       c['scorer'],
    'assister':     c['assister'],
    'chain_length': c['chain_length'],
} for c in all_chains])
chains_df.to_csv(OUTPUT_DIR / 'goal_chains.csv', index=False)
print(f"  goal_chains.csv         ({len(chains_df)} rows)")

# 2. attribution_comparison.csv  (all players, sorted by Shapley)
attr_out = player_agg.sort_values('shapley_credit', ascending=False)[[
    'player', 'last_touch_credit', 'linear_credit', 'shapley_credit',
    'last_touch_rank', 'shapley_rank', 'rank_change',
    'chains_involved', 'goals_scored', 'goals_assisted',
]].copy()
attr_out.to_csv(OUTPUT_DIR / 'attribution_comparison.csv', index=False)
print(f"  attribution_comparison.csv ({len(attr_out)} rows)")

# 3. method_stats.csv
method_stats = pd.DataFrame([
    {
        'method':          'Last-touch',
        'scorer_avg_pct':  round(scorer_lt_avg,   2),
        'top_player':      top_lt_player,
        'top_player_pct':  round(top_lt_pct,      2),
    },
    {
        'method':          'Linear',
        'scorer_avg_pct':  round(scorer_lin_avg,  2),
        'top_player':      top_lin_player,
        'top_player_pct':  round(top_lin_pct,     2),
    },
    {
        'method':          'Shapley',
        'scorer_avg_pct':  round(scorer_shap_avg, 2),
        'top_player':      top_shap_player,
        'top_player_pct':  round(top_shap_pct,    2),
    },
])
method_stats.to_csv(OUTPUT_DIR / 'method_stats.csv', index=False)
print(f"  method_stats.csv")

# 4. key_metrics.txt
with open(OUTPUT_DIR / 'key_metrics.txt', 'w') as f:
    f.write(f"n_goals:                    {n_goals}\n")
    f.write(f"n_matches:                  {n_processed}\n")
    f.write(f"avg_chain_length:           {avg_chain_length:.2f}\n")
    f.write(f"avg_players_per_chain:      {avg_n_players:.2f}\n")
    f.write(f"pct_goals_with_assist:      {n_with_assist / n_goals * 100:.1f}%\n")
    f.write(f"\n--- Attribution to scorer (avg per goal) ---\n")
    f.write(f"scorer_lastouch_pct:        100.0%\n")
    f.write(f"scorer_linear_pct:          {scorer_lin_avg:.1f}%\n")
    f.write(f"scorer_shapley_pct:         {scorer_shap_avg:.1f}%\n")
    f.write(f"lastouch_vs_shapley_swing:  {lt_shap_swing:.1f} percentage points\n")
    f.write(f"\n--- Top players ---\n")
    f.write(f"top_player_lastouch:        {top_lt_player}\n")
    f.write(f"top_player_lastouch_pct:    {top_lt_pct:.1f}%\n")
    f.write(f"top_player_shapley:         {top_shap_player}\n")
    f.write(f"top_player_shapley_pct:     {top_shap_pct:.1f}%\n")
    f.write(f"\n--- Valuation findings ---\n")
    f.write(f"most_undervalued_player:    {most_undervalued}\n")
    f.write(f"undervalued_lt_rank:        {uv_lt_rank}\n")
    f.write(f"undervalued_shap_rank:      {uv_shap_rank}\n")
    f.write(f"most_overvalued_player:     {most_overvalued}\n")
    f.write(f"overvalued_lt_rank:         {ov_lt_rank}\n")
    f.write(f"overvalued_shap_rank:       {ov_shap_rank}\n")
print(f"  key_metrics.txt")

# ─── FINAL SUMMARY ────────────────────────────────────────────
print("\n" + "=" * 65)
print("MODEL 13 COMPLETE")
print(f"Competition  : La Liga 2015/2016")
print(f"Matches      : {n_processed}")
print(f"Goals        : {n_goals}")
print(f"Avg chain    : {avg_chain_length:.1f} events, {avg_n_players:.1f} unique players")
print(f"Goals with identified assister: {n_with_assist} ({n_with_assist/n_goals*100:.0f}%)")
print()
print(f"HEADLINE STAT:")
print(f"  Last-touch gives the scorer: 100.0% of credit")
print(f"  Shapley gives the scorer  : {scorer_shap_avg:.1f}% of credit")
print(f"  Swing                     : {lt_shap_swing:.1f} percentage points")
print()
print(f"Top scorer (last-touch): {top_lt_player} — {top_lt_credit:.1f} credits ({top_lt_pct:.1f}% of goals)")
print(f"Top scorer (Shapley)   : {top_shap_player} — {top_shap_credit:.1f} credits ({top_shap_pct:.1f}% of goals)")
print()
print(f"Most undervalued by last-touch: {most_undervalued}")
print(f"  (Last-touch rank {uv_lt_rank} -> Shapley rank {uv_shap_rank})")
print(f"Most overvalued by last-touch : {most_overvalued}")
print(f"  (Last-touch rank {ov_lt_rank} -> Shapley rank {ov_shap_rank})")
print("=" * 65)
