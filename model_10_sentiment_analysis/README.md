# Model 10 — Transfer Sentiment Analysis

**Discipline:** Marketing DS + Product DS
**Method:** VADER + DistilBERT sentiment scoring + BERTopic topic modeling + correlation analysis

---

## The Question

> How does fan sentiment on Reddit shift after major transfer announcements — and does sentiment predict short-term match performance?

---

## Business Parallel

Each model maps directly to a core technique used in tech company data science roles. The sports domain makes the work memorable and shareable — the methods are identical to what you'd use on user or customer data.

---

## Method Summary

VADER + DistilBERT sentiment scoring + BERTopic topic modeling + correlation analysis

---

## Step-by-Step Plan

### 1. Event Selection
Choose 3-5 major transfers or match results. Define time windows: 48h before and 72h after each event.

### 2. Data Collection
Register PRAW app at reddit.com/prefs/apps. Pull r/soccer or team subreddit: title, body, upvotes, timestamp. ~500-2000 posts per event window.

### 3. Preprocessing
Remove URLs, usernames. Lowercase. Keep timestamp, text, upvote_score.

### 4. Sentiment Scoring
VADER compound score per post, hourly aggregate. DistilBERT (HuggingFace) for comparison. Create sentiment timeline.

### 5. Topic Modeling
BERTopic on all posts. Find: do topics shift before/after event? What do fans discuss most?

### 6. Correlation Analysis
Does pre-match sentiment predict match outcome? Pearson/Spearman — be honest about weak signal.


---

## Key Outputs

| File | Description |
|------|-------------|
| `outputs/sentiment_data.csv` | — |
| `outputs/sentiment_timeline.html` | — |
| `outputs/topic_model/` | — |

---

## Results

<!-- Fill in after running the notebook -->

**Key finding:** _

**Surprising result:** _

---

## How to Run

```bash
conda activate ds_portfolio
jupyter notebook model_10_sentiment_analysis.ipynb
```

---

## Skill Mapping (for interviews)

| What you built | What interviewers call it |
|---------------|--------------------------|
| VADER lexicon-based sentiment (fast, social-media optimized) | Rule-based NLP for social data |
| DistilBERT for higher-accuracy sentiment scoring | Transformer-based NLP |
| BERTopic for unsupervised topic discovery | Topic modeling |
| Correlation of sentiment with match outcomes | Cross-domain signal correlation |

---
