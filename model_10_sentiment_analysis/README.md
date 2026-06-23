# Model 10 — Sentiment Shift Around the Luka Doncic Trade

**Discipline:** Marketing DS + Product DS
**Method:** VADER sentiment scoring on daily news headline volume, before/after a major trade

---

## Scope Note

The original plan used Reddit via PRAW, which requires registering a Reddit API app — not set up, and intentionally skipped. Substituted with Google News RSS, which supports historical date-windowed search (`after:`/`before:` query operators) with no signup or API key. GDELT's DOC 2.0 API was tried first (also free/keyless and built for exactly this) but returned a persistent 429 rate limit across 15 retries over 7+ minutes from this environment, so Google News RSS became the actual data source. Heavy NLP (DistilBERT, BERTopic) was dropped in favor of VADER alone — it's the method the original plan itself recommends trying first, and it was sufficient here.

---

## The Question

> How does media coverage shift, in volume and in tone, around a genuinely shocking trade — and does the sentiment move as much as the volume does?

---

## Business Parallel

This is the same before/after analysis a marketing or comms team runs after a major announcement: did sentiment move, or did volume move? They're not the same question, and conflating them leads to the wrong read on whether an announcement "landed well."

---

## Method Summary

Event: Luka Doncic traded from the Mavericks to the Lakers, announced February 1, 2025 — one of the most shocking trades in NBA history. Daily Google News headlines for `"Luka Doncic"` were pulled day-by-day from January 25 to February 15, 2025 (7 days before to 14 days after), scored with VADER, and compared before vs. after the trade date.

---

## Data Sources

| Source | What it provides |
|--------|-------------------|
| Google News RSS (`news.google.com/rss/search`) | Daily headline text, date-windowed via `after:`/`before:` query operators |

No paid data, no API key, no app registration.

---

## Step-by-Step Plan

### 1. Data Collection
Daily query for `"Luka Doncic"`, January 25 – February 15, 2025, one Google News RSS call per day with that day's date window.

### 2. Sentiment Scoring
VADER compound score per headline title.

### 3. Aggregation
Daily mean/median sentiment, daily headline volume.

### 4. Evaluation
Before/after comparison split at the trade date. Two-sample t-test (Welch's, unequal variance) on sentiment. Trade-day volume vs. pre-trade daily baseline.

### 5. Visualization
Two-panel chart: daily mean sentiment (colored by before/after) and daily headline volume, both with the trade date marked.

---

## Key Outputs

| File | Description |
|------|-------------|
| `outputs/sentiment_volume_timeline.png` | Daily sentiment and daily volume, before/after the trade |
| `outputs/headlines_scored.csv` | Every headline with its VADER compound score |
| `outputs/daily_sentiment.csv` | Daily aggregates (volume, mean/median sentiment) |
| `outputs/summary_metrics.csv` | Before/after summary stats |

---

## Results

1,211 headlines, January 25 – February 15, 2025. 64 before the trade date, 1,147 after — a count that already tells most of the story.

Daily volume baseline before the trade: 9.1 headlines/day. Volume on trade day: 100 headlines/day — the maximum Google News RSS returns per query, meaning the real spike is a floor, not a ceiling; actual coverage was almost certainly higher. Volume stayed at or near that 100-headline cap for four straight days after the trade and didn't fully settle back toward baseline within the 14-day window observed.

Sentiment barely moved. Mean VADER compound score before: 0.0419. After: 0.0574. Welch's t-test: p = 0.7249 — not significant. Both before and after sit close to neutral-to-slightly-positive, the normal register of NBA news coverage.

**Key finding:** The trade produced an enormous volume shock (at least 11x baseline, likely more) and essentially no sentiment shock. Coverage exploded; tone didn't move.

**Surprising result:** A trade widely described as shocking, controversial, and emotionally charged for fans generated almost no measurable change in headline sentiment. The story was covered as basketball news, not as a positive or negative event. Volume and sentiment are different axes, and this is a case where only one of them moved.

---

## How to Run

```bash
conda activate ds_portfolio
python run_model_10.py
```

---

## Skill Mapping (for interviews)

| What you built | What interviewers call it |
|---------------|--------------------------|
| VADER lexicon-based sentiment scoring on headline text | Rule-based NLP for short text |
| Before/after event analysis with a two-sample t-test | Pre/post intervention testing |
| Separating volume signal from sentiment signal | Distinguishing reach from tone in measurement |
| Source substitution under a real rate-limit failure (GDELT → Google News RSS), documented rather than hidden | Handling source breakage gracefully |

---
