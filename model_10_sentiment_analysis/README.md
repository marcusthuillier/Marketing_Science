# Model 10 — Sentiment Shift Around the Luka Doncic Trade

**Discipline:** Marketing DS + Product DS
**Method:** VADER + DistilBERT sentiment scoring on daily news headline volume, before/after a major trade

---

## Scope Note

The original plan used Reddit via PRAW, which requires registering a Reddit API app — not set up, and intentionally skipped. Substituted with Google News RSS, which supports historical date-windowed search (`after:`/`before:` query operators) with no signup or API key. GDELT's DOC 2.0 API was tried first (also free/keyless and built for exactly this) but returned a persistent 429 rate limit across 15 retries over 7+ minutes from this environment, so Google News RSS became the actual data source. Sentiment is scored two ways — VADER (lexicon-based) and DistilBERT (transformer, context-aware) — so the conclusion isn't resting on one method's blind spots.

A first pass used a 7-day pre-trade window (64 headlines) and found a sizeable DistilBERT sentiment shift, just short of p < 0.05. Widening the pre-trade window to 30 days (209 headlines) as a robustness check changed the answer: the apparent effect shrinks and the p-value gets worse, not better. That result — a finding that doesn't survive a wider window — is reported below instead of the original, because it's the honest one.

---

## The Question

> How does media coverage shift, in volume and in tone, around a genuinely shocking trade — and does a more sophisticated sentiment model see something a simple lexicon score misses, robustly, regardless of how wide a pre-trade window you choose?

---

## Business Parallel

This is the same before/after analysis a marketing or comms team runs after a major announcement: did sentiment move, or did volume move? They're not the same question, and conflating them leads to the wrong read on whether an announcement "landed well." It's also a real test of measurement choice — a lexicon-based tool and a transformer can disagree on the same text, and a real test of robustness — a result that only survives with a narrow baseline window deserves the same skepticism as one method disagreeing with another.

---

## Method Summary

Event: Luka Doncic traded from the Mavericks to the Lakers, announced February 1, 2025 — one of the most shocking trades in NBA history. Daily Google News headlines for `"Luka Doncic"` were pulled day-by-day from January 2 to February 15, 2025 (30 days before to 14 days after), scored with both VADER and DistilBERT (`distilbert-base-uncased-finetuned-sst-2-english`), and compared before vs. after the trade date.

---

## Data Sources

| Source | What it provides |
|--------|-------------------|
| Google News RSS (`news.google.com/rss/search`) | Daily headline text, date-windowed via `after:`/`before:` query operators |

No paid data, no API key, no app registration.

---

## Step-by-Step Plan

### 1. Data Collection
Daily query for `"Luka Doncic"`, January 2 – February 15, 2025, one Google News RSS call per day with that day's date window. The pre-trade window was widened from an initial 7 days to 30 days specifically to stress-test whether the sentiment-shift finding held up with a larger, less cherry-picked baseline.

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

1,289 headlines, January 2 – February 15, 2025. 209 before the trade date, 1,080 after.

Daily volume baseline before the trade: 7.0 headlines/day. Volume on trade day: 100 headlines/day — the maximum Google News RSS returns per query, meaning the real spike is a floor, not a ceiling. Volume stayed at or near that 100-headline cap for four straight days after the trade.

| Method | Before | After | p-value |
|--------|--------|-------|---------|
| VADER (mean compound) | 0.0324 | 0.0439 | 0.6620 (not significant) |
| DistilBERT (mean score) | -0.0804 (45.9% positive) | -0.0376 (48.1% positive) | 0.5420 (not significant) |

This is the robustness check that mattered. With only a 7-day pre-trade window (64 headlines), DistilBERT showed a large apparent shift (-0.237 before vs -0.058 after, p = 0.125) that looked like it might be a real, if marginal, effect. Widening the pre-trade window to 30 days more than triples the pre-trade sample (64 → 209) and the effect mostly disappears: DistilBERT's before-mean moves from -0.237 to -0.080, and the p-value gets worse, not better (0.125 → 0.542). VADER's read stays flat and unremarkable either way.

The 7-day window wasn't wrong, exactly — it was real coverage. But it captured a specific, unusually negative stretch of last-week trade-rumor headlines (Jan 26–30, several days scoring below -0.5 on DistilBERT) that isn't representative of the broader pre-trade news cycle. Once diluted by three more weeks of ordinary coverage, that stretch stops looking like a meaningful "before" baseline and starts looking like what it actually was: a short burst of drama in the final days before the trade broke, not a sustained negative tone across the whole pre-trade period.

**Key finding:** The headline-grabbing "DistilBERT sees something VADER misses" result from the narrow window didn't survive a basic robustness check. Both methods now agree: no significant sentiment shift, before or after, by either measure.

**Surprising result:** A result that looks real and looks defensible — different from a competing method, in a sensible direction, theoretically plausible — can still be a window-selection artifact. The fix wasn't a better model, it was checking whether the finding survived a wider, less convenient choice of baseline. It didn't, and that's worth reporting as plainly as the original finding would have been.

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
| Re-running the analysis with a wider baseline window as a robustness check | Sensitivity analysis, not trusting the first result |
| Separating volume signal from sentiment signal | Distinguishing reach from tone in measurement |
| Source substitution under a real rate-limit failure (GDELT → Google News RSS), documented rather than hidden | Handling source breakage gracefully |

---
