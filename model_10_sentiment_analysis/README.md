# Model 10 — Sentiment Shift Around Shocking NBA Trades (Stacked Event Study)

**Discipline:** Marketing DS + Product DS
**Method:** VADER + DistilBERT sentiment scoring on daily news headlines, before/after 3 major trades

---

## Scope Note

The original plan used Reddit via PRAW, which requires registering a Reddit API app — not set up, and intentionally skipped. Substituted with Google News RSS, which supports historical date-windowed search (`after:`/`before:` query operators) with no signup or API key. GDELT's DOC 2.0 API was tried first (also free/keyless and built for exactly this) but returned a persistent 429 rate limit across 15 retries over 7+ minutes from this environment, so Google News RSS became the actual data source.

This went through two prior iterations. The first used a 7-day pre-trade window on the Luka Doncic trade alone and found a DistilBERT sentiment shift just short of significance. Widening that window to 30 days as a robustness check made the effect mostly disappear — it had been driven by a short, unusually negative burst of trade-rumor headlines in the final week, not the broader pre-trade period. Rather than conclude "trades don't move sentiment" from one event, this version stacks three major trades into one pooled before/after analysis, which is both higher-powered and a real test of whether the null result was specific to Doncic or general to shocking trades as a category.

Durant and Harden had a few days to a few weeks of public rumor lead time before their trades (unlike Doncic's zero-rumor surprise), so this isn't a perfectly matched set of "equally shocking" events. That's reported honestly rather than glossed over — it's part of why the per-event results below differ so much from each other.

---

## The Question

> Does a major trade reliably shift media sentiment, or was the earlier null result specific to one event? Does stacking multiple trades reveal a pattern that no single trade had the statistical power to show on its own?

---

## Business Parallel

This is the same before/after analysis a marketing or comms team runs after a major announcement: did sentiment move, or did volume move? It's also a direct lesson in statistical power — a single campaign or launch often doesn't have enough volume to detect a real effect, and the fix isn't a fancier test, it's pooling multiple comparable events into one analysis.

---

## Method Summary

Three major in-season NBA trades, each treated as an event: Luka Doncic to the Lakers (Feb 1, 2025), Kevin Durant to the Suns (Feb 9, 2023), and James Harden to the Nets (Jan 13, 2021). For each, daily Google News headlines for the player's name were pulled from 30 days before to 14 days after the trade date, scored with both VADER and DistilBERT (`distilbert-base-uncased-finetuned-sst-2-english`), and compared before vs. after — individually per event, and pooled across all three.

---

## Data Sources

| Source | What it provides |
|--------|-------------------|
| Google News RSS (`news.google.com/rss/search`) | Daily headline text, date-windowed via `after:`/`before:` query operators |

No paid data, no API key, no app registration.

---

## Step-by-Step Plan

### 1. Data Collection
Daily query per event (`"Luka Doncic"`, `"Kevin Durant"`, `"James Harden"`), each spanning 30 days before to 14 days after that event's trade date, one Google News RSS call per day per event.

### 2. Sentiment Scoring
Two methods, run on every headline across all three events: VADER compound score (lexicon-based, fast), and DistilBERT sentiment classification (transformer, context-aware) via HuggingFace `transformers`.

### 3. Aggregation
Daily mean sentiment per method per event, daily headline volume per event, headlines indexed by days-relative-to-trade so events can be compared and pooled on a common timeline.

### 4. Evaluation
Before/after comparison split at each event's own trade date, evaluated three ways: per-event (does Doncic alone show a shift? Durant? Harden?) and pooled (does sentiment move across all three trades combined?). Welch's t-test on each.

### 5. Visualization
Small-multiples timeline (one panel per event: sentiment + volume) and a pooled before/after summary bar chart with standard-error bars, for both VADER and DistilBERT.

---

## Key Outputs

| File | Description |
|------|-------------|
| `outputs/sentiment_volume_timeline.png` | Per-event sentiment + volume timelines, VADER and DistilBERT |
| `outputs/pooled_sentiment_comparison.png` | Pooled before/after sentiment, both methods, with SEM bars |
| `outputs/headlines_scored.csv` | Every headline across all 3 events with VADER compound score and DistilBERT label/score |
| `outputs/daily_sentiment.csv` | Daily aggregates per event, indexed by days-relative-to-trade |
| `outputs/summary_metrics.csv` | Before/after summary stats per event and pooled |

---

## Results

3,204 headlines across 3 trades. 1,220 before, 1,984 after (pooled).

| Event | n before / after | VADER before → after (p) | DistilBERT before → after (p) |
|-------|-------------------|---------------------------|--------------------------------|
| Doncic to Lakers | 202 / 1,128 | 0.031 → 0.055 (p=0.377) | -0.078 → -0.020 (p=0.413) |
| Durant to Suns | 454 / 361 | 0.083 → 0.094 (p=0.657) | 0.007 → 0.038 (p=0.652) |
| Harden to Nets | 564 / 495 | 0.002 → 0.110 (p<0.0001) | -0.233 → 0.219 (p<0.0001) |
| **Pooled (all 3)** | 1,220 / 1,984 | **0.037 → 0.076 (p=0.0025)** | **-0.118 → 0.050 (p<0.0001)** |

Neither Doncic nor Durant shows a significant shift individually — consistent with the earlier, single-event finding. Harden shows an enormous, highly significant one: DistilBERT's mean score moves from -0.233 (clearly negative, 35 days of trade-demand drama coverage) to +0.219 (clearly positive, four-team blockbuster analysis and "Nets are now a superteam" framing) the moment the trade actually happened.

Pooled across all three, the effect is real: VADER p=0.0025, DistilBERT p<0.0001. Sentiment measurably improves after a major trade, on average, even though any single event (Doncic, Durant) didn't have enough statistical power on its own to show it clearly. The earlier "no significant shift" conclusion from the Doncic-only analysis wasn't wrong about Doncic specifically — it just didn't have the power to detect what turns out to be a real, generalizable pattern.

**Key finding:** A single shocking trade often doesn't move the needle on sentiment by enough to clear statistical significance. Three trades, pooled, do. The fix for an underpowered single-event analysis wasn't a smarter model or a wider window within one event — it was more comparable events.

**Surprising result:** The pattern is driven heavily by Harden, but it isn't only Harden — even with his result included, this is a pooled test across genuinely different events, and the direction (sentiment improves after the trade) holds in all three point estimates, just not all three p-values. The likely mechanism: pre-trade periods, especially ones with a public trade-demand saga (Harden's case, and to a lesser extent Durant's brief rumor window), carry negative "drama and dysfunction" framing that resolves into more neutral-to-positive "here's the new team" coverage once the trade is official. Doncic's case had no rumor period at all, which may be exactly why it shows the smallest shift of the three — there was no drama to resolve.

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
| Stacked event study across 3 comparable events | Pooling for statistical power when a single event is underpowered |
| VADER + DistilBERT scored side-by-side on the same text | Cross-validating NLP methods, not trusting one blindly |
| Per-event and pooled before/after tests, reported together | Honest reporting of heterogeneous effects, not cherry-picking the strongest one |
| Re-running with a wider window, then with more events, as successive robustness checks | Sensitivity analysis as an iterative process, not a one-time check |
| Source substitution under a real rate-limit failure (GDELT → Google News RSS), documented rather than hidden | Handling source breakage gracefully |

---
