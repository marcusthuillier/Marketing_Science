"""
Model 10 -- Sentiment Shift Around the Luka Doncic Trade
Daily news-headline sentiment (VADER + DistilBERT) before/after the Feb 1, 2025 trade to the Lakers
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

TRADE_DATE = dt.date(2025, 2, 1)
WINDOW_START = TRADE_DATE - dt.timedelta(days=7)   # 2025-01-25
WINDOW_END   = TRADE_DATE + dt.timedelta(days=14)  # 2025-02-15
QUERY = '"Luka Doncic"'


# ── Data loading ─────────────────────────────────────────────────────────────

def fetch_day(day):
    import requests
    import xml.etree.ElementTree as ET

    next_day = day + dt.timedelta(days=1)
    url = "https://news.google.com/rss/search"
    params = {
        "q": f"{QUERY} after:{day.isoformat()} before:{next_day.isoformat()}",
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


def load_data():
    if os.path.exists(CACHE_RAW):
        print(f"Loading cached headlines from {CACHE_RAW}...")
        df = pd.read_csv(CACHE_RAW, parse_dates=["date"])
        print(f"  {len(df):,} headlines across {df['date'].nunique()} days")
        return df

    print(f"Fetching daily Google News headlines for {QUERY} "
          f"({WINDOW_START} to {WINDOW_END})...")
    all_rows = []
    day = WINDOW_START
    while day <= WINDOW_END:
        print(f"  {day} ...", end=" ", flush=True)
        try:
            rows = fetch_day(day)
        except Exception as e:
            print(f"failed ({e}), retrying once...")
            time.sleep(3)
            try:
                rows = fetch_day(day)
            except Exception as e2:
                print(f"  still failed: {e2}, skipping day")
                rows = []
        print(f"{len(rows)} headlines")
        all_rows.extend(rows)
        time.sleep(1.5)
        day += dt.timedelta(days=1)

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset=["date", "title"]).reset_index(drop=True)
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
    daily = df.groupby(df["date"].dt.date).agg(
        n_articles=("title", "count"),
        mean_sentiment=("compound", "mean"),
        median_sentiment=("compound", "median"),
        mean_bert=("bert_score", "mean"),
        pct_bert_positive=("bert_label", lambda s: (s == "POSITIVE").mean()),
    ).reset_index()
    daily.columns = ["date", "n_articles", "mean_sentiment", "median_sentiment",
                      "mean_bert", "pct_bert_positive"]
    return daily


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(df, daily):
    print("\nEvaluating before/after the trade...")
    before = df[df["date"].dt.date < TRADE_DATE]
    after = df[df["date"].dt.date >= TRADE_DATE]
    from scipy import stats

    n_before, n_after = len(before), len(after)

    # VADER
    mean_before = before["compound"].mean()
    mean_after = after["compound"].mean()
    t_stat, p_val = stats.ttest_ind(before["compound"], after["compound"], equal_var=False)

    # DistilBERT
    bert_mean_before = before["bert_score"].mean()
    bert_mean_after = after["bert_score"].mean()
    bert_t_stat, bert_p_val = stats.ttest_ind(before["bert_score"], after["bert_score"], equal_var=False)
    pct_pos_before = (before["bert_label"] == "POSITIVE").mean()
    pct_pos_after = (after["bert_label"] == "POSITIVE").mean()

    trade_day = daily[daily["date"] == TRADE_DATE]
    trade_day_volume = int(trade_day["n_articles"].iloc[0]) if len(trade_day) else 0
    baseline_volume = daily[daily["date"] < TRADE_DATE]["n_articles"].mean()

    print(f"  VADER      before (n={n_before}): mean = {mean_before:.4f}  |  after (n={n_after}): mean = {mean_after:.4f}")
    print(f"  VADER      t-test: t={t_stat:.3f}, p={p_val:.4f} {'(SIGNIFICANT)' if p_val < 0.05 else '(not significant)'}")
    print(f"  DistilBERT before: mean = {bert_mean_before:.4f} ({pct_pos_before:.1%} positive)  |  "
          f"after: mean = {bert_mean_after:.4f} ({pct_pos_after:.1%} positive)")
    print(f"  DistilBERT t-test: t={bert_t_stat:.3f}, p={bert_p_val:.4f} "
          f"{'(SIGNIFICANT)' if bert_p_val < 0.05 else '(not significant)'}")
    print(f"  Trade-day volume: {trade_day_volume} headlines vs {baseline_volume:.1f}/day baseline before")

    return {
        "n_before": n_before, "n_after": n_after,
        "mean_before": mean_before, "mean_after": mean_after,
        "t_stat": t_stat, "p_value": p_val,
        "bert_mean_before": bert_mean_before, "bert_mean_after": bert_mean_after,
        "bert_t_stat": bert_t_stat, "bert_p_value": bert_p_val,
        "pct_pos_before": pct_pos_before, "pct_pos_after": pct_pos_after,
        "trade_day_volume": trade_day_volume, "baseline_daily_volume": baseline_volume,
    }


# ── Visualizations ───────────────────────────────────────────────────────────

def generate_plots(daily, show=False):
    matplotlib.use("Agg")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

    fig, axes = plt.subplots(3, 1, figsize=(10, 10.5), facecolor=BG, sharex=True)

    ax = axes[0]
    ax.set_facecolor(BG)
    colors = [RED if d >= TRADE_DATE else BLUE for d in daily["date"]]
    ax.bar(daily["date"], daily["mean_sentiment"], color=colors, alpha=0.85, width=0.8)
    ax.axhline(0, color="#888", lw=1)
    ax.axvline(TRADE_DATE, color="#222", lw=1.5, ls="--")
    ax.text(TRADE_DATE, ax.get_ylim()[1] * 0.92, "  Trade announced", fontsize=9, fontweight="bold")
    ax.set_ylabel("Mean VADER Sentiment", fontsize=10)
    ax.set_title("Daily Headline Sentiment — \"Luka Doncic\" News\nBlue = before trade, Red = after",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", color="#e0dbd5", lw=0.7)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    ax.set_facecolor(BG)
    colors = [RED if d >= TRADE_DATE else BLUE for d in daily["date"]]
    ax.bar(daily["date"], daily["mean_bert"], color=colors, alpha=0.85, width=0.8)
    ax.axhline(0, color="#888", lw=1)
    ax.axvline(TRADE_DATE, color="#222", lw=1.5, ls="--")
    ax.set_ylabel("Mean DistilBERT Score", fontsize=10)
    ax.set_title("Daily Headline Sentiment — DistilBERT (transformer model)", fontsize=11, fontweight="bold")
    ax.grid(axis="y", color="#e0dbd5", lw=0.7)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[2]
    ax.set_facecolor(BG)
    ax.bar(daily["date"], daily["n_articles"], color=BLUE, alpha=0.7, width=0.8)
    ax.axvline(TRADE_DATE, color="#222", lw=1.5, ls="--")
    ax.set_ylabel("Headline Volume", fontsize=10)
    ax.set_xlabel("Date", fontsize=10)
    ax.set_title("Daily Headline Volume", fontsize=11, fontweight="bold")
    ax.grid(axis="y", color="#e0dbd5", lw=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.autofmt_xdate(rotation=45)

    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR + "sentiment_volume_timeline.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: sentiment_volume_timeline.png")

    if show:
        plt.show()


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    matplotlib.use("Agg")

    raw = load_data()
    scored = score_sentiment(raw)
    daily = aggregate_daily(scored)
    summary = evaluate(scored, daily)
    generate_plots(daily, show=False)

    scored.to_csv(OUTPUTS_DIR + "headlines_scored.csv", index=False)
    daily.to_csv(OUTPUTS_DIR + "daily_sentiment.csv", index=False)
    pd.DataFrame([summary]).to_csv(OUTPUTS_DIR + "summary_metrics.csv", index=False)

    print(f"\n{'='*60}")
    print("  KEY FINDINGS -- Doncic Trade Sentiment Analysis")
    print(f"{'='*60}")
    print(f"  Total headlines:       {len(scored)}")
    print(f"  VADER      before (n={summary['n_before']}): {summary['mean_before']:.4f}  |  after (n={summary['n_after']}): {summary['mean_after']:.4f}  |  p={summary['p_value']:.4f}")
    print(f"  DistilBERT before: {summary['bert_mean_before']:.4f} ({summary['pct_pos_before']:.1%} pos)  |  "
          f"after: {summary['bert_mean_after']:.4f} ({summary['pct_pos_after']:.1%} pos)  |  p={summary['bert_p_value']:.4f}")
    print(f"  Trade-day volume:      {summary['trade_day_volume']} vs {summary['baseline_daily_volume']:.1f}/day baseline")
    print(f"\nAll outputs saved to {OUTPUTS_DIR}")
