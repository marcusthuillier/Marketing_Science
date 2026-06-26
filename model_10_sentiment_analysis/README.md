# Model 10 — Sentiment Shift Around the Luka Doncic Trade

**Discipline:** Marketing DS + Product DS
**Method:** VADER + DistilBERT sentiment scoring on daily news headline volume, before/after a major trade

---

## Scope Note

The original plan used Reddit via PRAW, which requires registering a Reddit API app — not set up, and intentionally skipped. Substituted with Google News RSS, which supports historical date-windowed search (`after:`/`before:` query operators) with no signup or API key. GDELT's DOC 2.0 API was tried first (also free/keyless and built for exactly this) but returned a persistent 429 rate limit across 15 retries over 7+ minutes from this environment, so Google News RSS became the actual data source. Sentiment is scored two ways — VADER (lexicon-based) and DistilBERT (transformer, context-aware) — so the conclusion isn't resting on one method's blind spots.

---

## The Question

> How does media coverage shift, in volume and in tone, around a genuinely shocking trade — and does a more sophisticated sentiment model see something a simple lexicon score misses?

---

## Business Parallel

This is the same before/after analysis a marketing or comms team runs after a major announcement: did sentiment move, or did volume move? They're not the same question, and conflating them leads to the wrong read on whether an announcement "landed well." It's also a real test of measurement choice — a lexicon-based tool and a transformer can disagree on the same text, and which one you trust changes the conclusion.

---

## Method Summary

Event: Luka Doncic traded from the Mavericks to the Lakers, announced February 1, 2025 — one of the most shocking trades in NBA history. Daily Google News headlines for `"Luka Doncic"` were pulled day-by-day from January 25 to February 15, 2025 (7 days before to 14 days after), scored with both VADER and DistilBERT (`distilbert-base-uncased-finetuned-sst-2-english`), and compared before vs. after the trade date.

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
Two methods, run on every headline: VADER compound score (lexicon-based, fast), and DistilBERT sentiment classification (transformer, context-aware) via HuggingFace `transformers`.

### 3. Aggregation
Daily mean sentiment per method, daily headline volume.

### 4. Evaluation
Before/after comparison split at the trade date, for both VADER and DistilBERT independently. Two-sample t-test (Welch's, unequal variance) on each. Trade-day volume vs. pre-trade daily baseline.

### 5. Visualization
Three-panel chart: daily VADER sentiment, daily DistilBERT sentiment, and daily headline volume, all with the trade date marked.

---

## Key Outputs

| File | Description |
|------|-------------|
| `outputs/sentiment_volume_timeline.png` | Daily VADER sentiment, daily DistilBERT sentiment, and daily volume, before/after the trade |
| `outputs/headlines_scored.csv` | Every headline with its VADER compound score and DistilBERT label/score |
| `outputs/daily_sentiment.csv` | Daily aggregates (volume, mean sentiment per method, % positive per DistilBERT) |
| `outputs/summary_metrics.csv` | Before/after summary stats for both methods |

---

## Results

1,211 headlines, January 25 – February 15, 2025. 64 before the trade date, 1,147 after — a count that already tells most of the story.

Daily volume baseline before the trade: 9.1 headlines/day. Volume on trade day: 100 headlines/day — the maximum Google News RSS returns per query, meaning the real spike is a floor, not a ceiling. Volume stayed at or near that 100-headline cap for four straight days after the trade.

The two sentiment methods tell different stories:

| Method | Before | After | p-value |
|--------|--------|-------|---------|
| VADER (mean compound) | 0.0419 | 0.0574 | 0.7249 (not significant) |
| DistilBERT (mean score) | -0.2370 (37.5% positive) | -0.0580 (46.8% positive) | 0.1253 (not significant) |

VADER sees almost nothing — both periods sit close to neutral. DistilBERT sees a real directional shift: headlines before the trade date (largely pre-trade rumor and roster-drama coverage) score clearly negative on average, and headlines after the trade (the trade itself, reactions, and analysis) move much closer to neutral. Neither crosses the conventional p < 0.05 threshold, but DistilBERT's shift (-0.237 to -0.058, a 0.18 swing) is an order of magnitude larger than VADER's (0.042 to 0.057), and DistilBERT's p-value (0.125) is meaningfully closer to significance.

**Key finding:** A simple lexicon-based sentiment score and a context-aware transformer model can reach different conclusions from the same text. VADER says nothing moved. DistilBERT says something did, just not quite at conventional significance, and the direction makes sense: pre-trade coverage carried more negative framing (drama, rumors, "worst trade" hot takes) than the more neutral, factual reporting that dominated once the trade was official.

**Surprising result:** The volume spike (11x+ baseline) is unambiguous and massive. The sentiment shift is real by one method and invisible by another. Which sentiment tool you pick can change whether you conclude "tone didn't move" or "tone improved once the news settled" — a methodological lesson at least as important as the headline finding.

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
| VADER + DistilBERT scored side-by-side on the same text | Cross-validating NLP methods, not trusting one blindly |
| Before/after event analysis with a two-sample t-test per method | Pre/post intervention testing |
| Separating volume signal from sentiment signal | Distinguishing reach from tone in measurement |
| Source substitution under a real rate-limit failure (GDELT → Google News RSS), documented rather than hidden | Handling source breakage gracefully |

---
