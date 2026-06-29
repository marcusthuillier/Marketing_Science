"""
Model 10 -- Sentiment Shift Around Shocking NBA Trades (Stacked Event Study)
Daily news-headline sentiment (VADER + DistilBERT) before/after 3 major trades
Run: python run_model_10.py
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # torch/numpy OpenMP DLL conflict workaround

# torch must be imported before numpy/pandas/scipy on this Windows env, or its
# bundled DLLs (shm.dll) conflict with the ones numpy already loaded.
import torch  # noqa: F401
from transformers import pipeline  # noqa: F401

import time, warnings
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

OUTPUTS_DIR = "outputs/"
os.makedirs(OUTPUTS_DIR, exist_ok=True)

CACHE_RAW = OUTPUTS_DIR + "headlines_raw.csv"

BG, RED, BLUE = "#f5f0eb", "#9b3d36", "#4c72b0"

WINDOW_BEFORE_DAYS = 30
WINDOW_AFTER_DAYS  = 14

# A single trade (Doncic) showed a finding that didn't survive widening the
# pre-trade window. Stacking more major in-season superstar trades tests
# whether that null result is specific to Doncic or general to "shocking
# trade" as a category. Durant and Harden had a few days of rumor lead time
# (unlike Doncic's zero-rumor surprise), so this isn't a perfectly matched
# set -- flagged honestly in the README -- but all three are major,
# well-documented, in-season blockbusters with real before/after structure.
EVENTS = [
    {"name": "Doncic to Lakers",  "query": '"Luka Doncic"',  "trade_date": dt.date(2025, 2, 1)},
    {"name": "Durant to Suns",    "query": '"Kevin Durant"', "trade_date": dt.date(2023, 2, 9)},
    {"name": "Harden to Nets",    "query": '"James Harden"', "trade_date": dt.date(2021, 1, 13)},
]


# ── Data loading ─────────────────────────────────────────────────────────────

def fetch_day(query, day):
    import requests
    import xml.etree.ElementTree as ET

    next_day = day + dt.timedelta(days=1)
    url = "https://news.google.com/rss/search"
    params = {
        "q": f"{query} after:{day.isoformat()} before:{next_day.isoformat()}",
        "hl": "en-US", "gl": "US", "ceid": "US:en",
    }
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    r = requests.get(url, params=params, headers=headers, timeout=30)
    r.raise_for_status()
    root = ET.fromstring(r.text)
    rows = []
    for item in root.findall(".//item"):
        title_el = item.find("title")
        if title_el is None or not title_el.text:
            continue
        title = title_el.text
        # Title format is "Headline - Source"; split off the source if present
        source = title.rsplit(" - ", 1)[-1] if " - " in title else ""
        rows.append({"date": day.isoformat(), "title": title, "source": source})
    return rows


def fetch_event(event):
    trade_date = event["trade_date"]
    window_start = trade_date - dt.timedelta(days=WINDOW_BEFORE_DAYS)
    window_end = trade_date + dt.timedelta(days=WINDOW_AFTER_DAYS)
    print(f"Fetching daily Google News headlines for {event['name']} "
          f"({window_start} to {window_end})...")
    all_rows = []
    day = window_start
    while day <= window_end:
        print(f"  {day} ...", end=" ", flush=True)
        try:
            rows = fetch_day(event["query"], day)
        except Exception as e:
            print(f"failed ({e}), retrying once...")
            time.sleep(3)
            try:
                rows = fetch_day(event["query"], day)
            except Exception as e2:
                print(f"  still failed: {e2}, skipping day")
                rows = []
        print(f"{len(rows)} headlines")
        for row in rows:
            row["event"] = event["name"]
            row["trade_date"] = trade_date.isoformat()
        all_rows.extend(rows)
        time.sleep(1.5)
        day += dt.timedelta(days=1)
    return all_rows


def load_data():
    if os.path.exists(CACHE_RAW):
        print(f"Loading cached headlines from {CACHE_RAW}...")
        df = pd.read_csv(CACHE_RAW, parse_dates=["date", "trade_date"])
        print(f"  {len(df):,} headlines across {df['event'].nunique()} events")
        return df

    all_rows = []
    for event in EVENTS:
        all_rows.extend(fetch_event(event))

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"])
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.drop_duplicates(subset=["event", "date", "title"]).reset_index(drop=True)
    df["relative_day"] = (df["date"] - df["trade_date"]).dt.days
    df.to_csv(CACHE_RAW, index=False)
    print(f"  Cached {len(df):,} headlines to {CACHE_RAW}")
    return df


# ── Sentiment scoring ─────────────────────────────────────────────────────────

def score_sentiment(df):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    print("\nScoring headlines with VADER...")
    analyzer = SentimentIntensityAnalyzer()
    df = df.copy()
    df["compound"] = df["title"].apply(lambda t: analyzer.polarity_scores(str(t))["compound"])
    print(f"  VADER: scored {len(df)} headlines")

    print("\nScoring headlines with DistilBERT (transformer model)...")
    clf = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    titles = df["title"].astype(str).tolist()
    results = clf(titles, batch_size=32, truncation=True)
    df["bert_label"] = [r["label"] for r in results]
    df["bert_score"] = [r["score"] if r["label"] == "POSITIVE" else -r["score"] for r in results]
    print(f"  DistilBERT: scored {len(df)} headlines")
    return df


def aggregate_daily(df):
    daily = df.groupby(["event", "relative_day"]).agg(
        n_articles=("title", "count"),
        mean_sentiment=("compound", "mean"),
        mean_bert=("bert_score", "mean"),
        pct_bert_positive=("bert_label", lambda s: (s == "POSITIVE").mean()),
    ).reset_index()
    return daily


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_event(df, event_name):
    sub = df[df["event"] == event_name]
    before = sub[sub["relative_day"] < 0]
    after = sub[sub["relative_day"] >= 0]
    from scipy import stats

    t_v, p_v = stats.ttest_ind(before["compound"], after["compound"], equal_var=False)
    t_b, p_b = stats.ttest_ind(before["bert_score"], after["bert_score"], equal_var=False)

    return {
        "event": event_name,
        "n_before": len(before), "n_after": len(after),
        "vader_before": before["compound"].mean(), "vader_after": after["compound"].mean(),
        "vader_t": t_v, "vader_p": p_v,
        "bert_before": before["bert_score"].mean(), "bert_after": after["bert_score"].mean(),
        "bert_t": t_b, "bert_p": p_b,
    }


def evaluate_pooled(df):
    before = df[df["relative_day"] < 0]
    after = df[df["relative_day"] >= 0]
    from scipy import stats

    t_v, p_v = stats.ttest_ind(before["compound"], after["compound"], equal_var=False)
    t_b, p_b = stats.ttest_ind(before["bert_score"], after["bert_score"], equal_var=False)

    return {
        "event": "POOLED (all 3 trades)",
        "n_before": len(before), "n_after": len(after),
        "vader_before": before["compound"].mean(), "vader_after": after["compound"].mean(),
        "vader_t": t_v, "vader_p": p_v,
        "bert_before": before["bert_score"].mean(), "bert_after": after["bert_score"].mean(),
        "bert_t": t_b, "bert_p": p_b,
    }


def run_evaluation(df):
    print("\nEvaluating before/after each trade, individually and pooled...")
    rows = [evaluate_event(df, e["name"]) for e in EVENTS]
    rows.append(evaluate_pooled(df))
    summary_df = pd.DataFrame(rows)
    for _, r in summary_df.iterrows():
        sig_v = "(SIGNIFICANT)" if r["vader_p"] < 0.05 else "(not significant)"
        sig_b = "(SIGNIFICANT)" if r["bert_p"] < 0.05 else "(not significant)"
        print(f"  {r['event']:<24} n_before={r['n_before']:>4} n_after={r['n_after']:>5}  "
              f"VADER p={r['vader_p']:.4f} {sig_v}  |  DistilBERT p={r['bert_p']:.4f} {sig_b}")
    return summary_df


# ── Visualizations ───────────────────────────────────────────────────────────

def generate_plots(df, daily, show=False):
    matplotlib.use("Agg")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

    # ── Plot 1: per-event small multiples, VADER + DistilBERT + volume ──────
    n_events = len(EVENTS)
    fig, axes = plt.subplots(n_events, 1, figsize=(10, 3.4 * n_events), facecolor=BG, sharex=True)
    for i, event in enumerate(EVENTS):
        ax = axes[i]
        ax.set_facecolor(BG)
        sub = daily[daily["event"] == event["name"]].sort_values("relative_day")
        colors = [RED if d >= 0 else BLUE for d in sub["relative_day"]]
        ax2 = ax.twinx()
        ax2.bar(sub["relative_day"], sub["n_articles"], color=colors, alpha=0.18, width=0.9)
        ax2.set_ylabel("Volume", fontsize=8)
        ax.plot(sub["relative_day"], sub["mean_sentiment"], color="#222", lw=1.8, label="VADER")
        ax.plot(sub["relative_day"], sub["mean_bert"], color=RED, lw=1.8, ls="--", label="DistilBERT")
        ax.axhline(0, color="#888", lw=0.8)
        ax.axvline(0, color="#222", lw=1.2, ls=":")
        ax.set_title(event["name"], fontsize=10, fontweight="bold", loc="left")
        ax.set_ylabel("Sentiment", fontsize=8)
        ax.grid(axis="y", color="#e0dbd5", lw=0.5)
        ax.spines[["top", "left"]].set_visible(False)
        if i == 0:
            ax.legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("Days relative to trade announcement", fontsize=9)
    fig.suptitle("Sentiment + Volume Around 3 Shocking NBA Trades", fontsize=12, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR + "sentiment_volume_timeline.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: sentiment_volume_timeline.png")

    # ── Plot 2: pooled before/after summary bars ─────────────────────────────
    before = df[df["relative_day"] < 0]
    after = df[df["relative_day"] >= 0]
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=BG)
    ax.set_facecolor(BG)
    labels = ["VADER", "DistilBERT"]
    before_means = [before["compound"].mean(), before["bert_score"].mean()]
    after_means = [after["compound"].mean(), after["bert_score"].mean()]
    before_sem = [before["compound"].sem(), before["bert_score"].sem()]
    after_sem = [after["compound"].sem(), after["bert_score"].sem()]
    x = np.arange(len(labels))
    width = 0.32
    ax.bar(x - width/2, before_means, width, yerr=before_sem, color=BLUE, alpha=0.8, label="Before (pooled, 3 trades)", capsize=4)
    ax.bar(x + width/2, after_means, width, yerr=after_sem, color=RED, alpha=0.8, label="After (pooled, 3 trades)", capsize=4)
    ax.axhline(0, color="#888", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Mean sentiment score (+/- SEM)", fontsize=10)
    ax.set_title("Pooled Before/After Sentiment Across 3 Trades\n(Doncic, Durant, Harden)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", color="#e0dbd5", lw=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR + "pooled_sentiment_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: pooled_sentiment_comparison.png")

    if show:
        plt.show()


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    matplotlib.use("Agg")

    raw = load_data()
    scored = score_sentiment(raw)
    daily = aggregate_daily(scored)
    summary_df = run_evaluation(scored)
    generate_plots(scored, daily, show=False)

    scored.to_csv(OUTPUTS_DIR + "headlines_scored.csv", index=False)
    daily.to_csv(OUTPUTS_DIR + "daily_sentiment.csv", index=False)
    summary_df.to_csv(OUTPUTS_DIR + "summary_metrics.csv", index=False)

    pooled = summary_df.iloc[-1]
    print(f"\n{'='*60}")
    print("  KEY FINDINGS -- Stacked Trade Sentiment Analysis")
    print(f"{'='*60}")
    print(f"  Total headlines (3 events): {len(scored)}")
    for _, r in summary_df.iterrows():
        print(f"  {r['event']}: VADER {r['vader_before']:.3f} -> {r['vader_after']:.3f} (p={r['vader_p']:.4f})  |  "
              f"DistilBERT {r['bert_before']:.3f} -> {r['bert_after']:.3f} (p={r['bert_p']:.4f})")
    print(f"\n  POOLED across all 3 trades (n_before={pooled['n_before']}, n_after={pooled['n_after']}):")
    print(f"  VADER:      p={pooled['vader_p']:.4f} {'(SIGNIFICANT)' if pooled['vader_p'] < 0.05 else '(not significant)'}")
    print(f"  DistilBERT: p={pooled['bert_p']:.4f} {'(SIGNIFICANT)' if pooled['bert_p'] < 0.05 else '(not significant)'}")
    print(f"\nAll outputs saved to {OUTPUTS_DIR}")
