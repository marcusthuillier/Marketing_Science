# 20-Model Data Science Portfolio: Full Planning Document
**Disciplines:** Marketing DS · Growth DS · Product DS  
**Theme:** Sports Analytics  
**Timeline:** ~11 months · 3–5 hrs/week  
**Target Roles:** Marketing DS, Growth DS, Product DS at tech companies

---

## How to Use This Document

Each model follows the same structure:
- **The Question** — the specific business/sports question you're answering
- **Data Sources** — 2–3 free options to choose from
- **Approach** — the method and why
- **Step-by-Step Plan** — exactly what Claude Code needs to build it
- **Key Output** — what you ship
- **LinkedIn Hook** — the post angle

Pick one primary sport per project where possible — consistency builds a portfolio narrative.

---

## Global Data Sources Reference

| Source | What It Has | Access |
|--------|------------|--------|
| `nba_api` | NBA stats, play-by-play, lineups, advanced | `pip install nba_api` — free, no key |
| `Basketball-Reference` via `sportsipy` | Historical NBA/NFL/MLB/NHL stats | `pip install sportsipy` — free scraper |
| `FBref` via `soccerdata` | Soccer per-90 stats, advanced metrics | `pip install soccerdata` — free |
| `StatsBomb Open Data` | Free event-level soccer data (JSON) | `github.com/statsbomb/open-data` — free |
| `Understat` | xG, xA, shot maps for top leagues | `pip install understat` — free |
| `Football-Data.org` | Match results, standings, top leagues | Free API key — no credit card |
| `Transfermarkt` via `transfermarkt-scraper` | Player valuations, transfer history | `pip install transfermarkt-scraper` — free |
| `ClubElo` | Historical ELO ratings for soccer | `clubelo.com/api` — free REST API |
| `Open-Meteo` | Historical weather API | `api.open-meteo.com` — completely free |
| `PRAW (Reddit API)` | Reddit posts, comments, sentiment | Free with account — `pip install praw` |
| `pytrends` | Google Trends data | `pip install pytrends` — free |
| `Kaggle NBA datasets` | Pre-cleaned NBA datasets | Free download after login |
| `balldontlie.io` | NBA stats API | Free tier — no credit card |

---

## Phase 1: Customer Fundamentals
*Months 1–3 · Models 1–6*

---

### Model 01 — Segmentation

**Discipline:** Marketing DS  
**The Question:** *Do traditional basketball positions actually capture how players play — or are there natural statistical archetypes that tell a different story?*

#### Data Sources (choose one)
| Option | Source | What to pull | How |
|--------|--------|-------------|-----|
| ⭐ NBA Advanced Stats | `nba_api` | Per-game advanced stats: PER, TS%, USG%, DRTG, ORTG, AST%, REB%, STL%, BLK% | `from nba_api.stats.endpoints import leaguedashplayerstats` |
| Soccer Alternative | `FBref` via `soccerdata` | Per-90 stats: xG, xA, progressive passes, pressures, tackles, aerials | `import soccerdata as sd; sd.FBref()` |
| Kaggle fallback | Kaggle NBA Dataset | Pre-cleaned multi-season player stats | Download `nba_games.csv` from Kaggle |

#### Approach
K-Means clustering with UMAP dimensionality reduction for visualization. Use the elbow method and silhouette scores to choose K. Scale features before clustering. Label clusters manually based on stat profiles.

#### Step-by-Step Plan
```
1. DATA COLLECTION
   - Pull 3 seasons of NBA player advanced stats via nba_api
   - Filter: min 500 minutes played per season to remove noise
   - Features to use: USG%, AST%, REB%, STL%, BLK%, 3PAr, FTr, TS%, DRTG contribution

2. PREPROCESSING
   - StandardScaler on all features
   - Handle missing values (drop or impute with median)
   - Remove duplicate player-seasons, keep most recent season per player

3. DIMENSIONALITY REDUCTION
   - Apply UMAP (n_components=2) for visualization
   - Apply PCA separately for interpretability comparison

4. CLUSTERING
   - Run K-Means for K=4 to K=10
   - Plot elbow curve (inertia) and silhouette scores
   - Choose optimal K (expect ~6–8 meaningful clusters)
   - Assign cluster labels to each player

5. ANALYSIS
   - Compute mean stats per cluster
   - Name each cluster (e.g., "3&D Wing", "Pass-First PG", "Rim Protector")
   - Identify 5 most representative players per cluster
   - Compare cluster assignment vs traditional position

6. VISUALIZATION
   - UMAP scatter plot colored by cluster (interactive Plotly)
   - Radar/spider chart per cluster showing stat profiles
   - Table of top 10 players per cluster

7. OUTPUT
   - Save: clustered_players.csv with cluster assignments
   - Save: umap_plot.html (interactive)
   - Save: cluster_profiles.png (radar charts)
```

**Key Output:** Interactive UMAP plot showing ~6–8 player archetypes. The post hook is which famous player ends up in a surprising cluster.  
**LinkedIn Hook:** *"I clustered every NBA player by stats. The results completely broke traditional positions."*

---

### Model 02 — Funnel Analysis

**Discipline:** Growth DS  
**The Question:** *Where do sports fans drop off on their journey from casual viewer to committed season ticket holder — and what's the biggest leak in the funnel?*

#### Data Sources (choose one)
| Option | Source | What to pull | How |
|--------|--------|-------------|-----|
| ⭐ Reddit + Google Trends | `PRAW` + `pytrends` | Subreddit member counts, post frequency, Google search volume as proxies for fan engagement stages | `pip install praw pytrends` |
| Soccer attendance | `Football-Data.org` + Wikipedia | Match attendance over seasons, league position, ticket price proxies | Free API key at football-data.org |
| Synthetic + real | Build synthetic funnel from public attendance/TV ratings | Model stages: TV viewer → social follower → match attendee → season holder | Public broadcast data + club annual reports |

#### Approach
Funnel analysis using stage-to-stage conversion rates. Since fan-level data isn't public, use aggregate proxies (TV ratings → social followers → attendance → season tickets) to estimate funnel shape. Sankey diagram to visualize flow. Chi-square tests on conversion rate differences.

#### Step-by-Step Plan
```
1. DATA COLLECTION
   - Pull Google Trends data for 5–10 teams via pytrends (search volume = awareness proxy)
   - Pull Reddit subscriber counts for team subreddits via PRAW
   - Find public attendance figures (Wikipedia, club sites, ESPN)
   - Find season ticket holder estimates (press releases, news articles)
   - Map to funnel stages: Awareness → Interest → Attendance → Loyalty

2. FUNNEL CONSTRUCTION
   - Define stage thresholds:
     Stage 1: Aware (Google search impressions proxy)
     Stage 2: Engaged (social/Reddit follower)
     Stage 3: Attendee (1+ games/year)
     Stage 4: Committed (season ticket holder)
   - Estimate population at each stage per team

3. CONVERSION RATE ANALYSIS
   - Calculate stage-to-stage conversion %
   - Compare across teams: which teams convert fans better?
   - Identify the biggest drop-off stage (the "leaky bucket")

4. STATISTICAL TESTING
   - Chi-square test: are conversion differences between teams significant?
   - Correlation: does league position affect funnel conversion?

5. VISUALIZATION
   - Sankey diagram (plotly) showing fan flow through stages
   - Bar chart: conversion rates by team
   - Scatter: league position vs funnel efficiency

6. OUTPUT
   - Save: funnel_data.csv
   - Save: sankey_diagram.html (interactive)
   - Save: conversion_analysis.png
```

**Key Output:** Sankey diagram showing where fans are lost. Find which stage has the worst conversion and what predicts it.  
**LinkedIn Hook:** *"Growth teams obsess over funnel drop-off. I mapped one for football fans — the biggest leak isn't where you'd expect."*

---

### Model 03 — Cohort Analysis

**Discipline:** Growth DS  
**The Question:** *Do some NBA draft classes produce sustained value over time — and can cohort analysis reveal which draft years were genuinely exceptional vs just lucky?*

#### Data Sources (choose one)
| Option | Source | What to pull | How |
|--------|--------|-------------|-----|
| ⭐ Basketball-Reference | `sportsipy` | Draft class by year, career stats per season, Win Shares per season | `from sportsipy.nba.players import Players` |
| NFL Alternative | Pro Football Reference via `sportsipy` | Draft class career approximate value (AV) by season | `from sportsipy.nfl.players import Players` |
| Kaggle | Kaggle NBA Dataset | Pre-built draft history with career stats | Download from Kaggle |

#### Approach
Classic cohort retention analysis: group players by draft year (cohort), track a value metric (Win Shares, WAR, or Games Played) across their career years. Build a cohort heatmap identical to what growth teams use for user retention. Visualize as a retention grid.

#### Step-by-Step Plan
```
1. DATA COLLECTION
   - Pull all NBA draft picks 1990–2018 with draft year and position
   - For each player, pull season-by-season Win Shares
   - Define "career year" as years since draft (Year 1 = rookie season)
   - Cap at 15 career years

2. COHORT CONSTRUCTION
   - Group players by draft year (cohort = year drafted)
   - For each cohort, calculate:
     - Total players drafted
     - Players still active at Year 3, 5, 8, 10 (survival rate)
     - Average Win Shares at each career year
     - Median Win Shares at each career year

3. RETENTION HEATMAP
   - Build matrix: rows = draft years, columns = career years
   - Values = average Win Shares OR % of players still contributing
   - This is the exact format of a growth cohort retention table

4. ANALYSIS
   - Identify best and worst draft classes
   - Calculate "LTV equivalent" — total career Win Shares per cohort
   - Find: which draft position (1st pick, lottery, second round) retains value longest?

5. STATISTICAL TESTING
   - ANOVA: are differences between cohort LTVs statistically significant?
   - Trend analysis: are recent drafts better or worse than historical ones?

6. VISUALIZATION
   - Cohort heatmap (seaborn heatmap — classic growth analytics format)
   - Line chart: average career Win Shares by draft year over career time
   - Bar chart: top 10 draft classes by total cohort value

7. OUTPUT
   - Save: cohort_matrix.csv
   - Save: cohort_heatmap.png
   - Save: draft_class_ltv.csv
```

**Key Output:** Cohort heatmap that looks like a SaaS retention grid — but for NBA draft classes.  
**LinkedIn Hook:** *"Cohort analysis is growth DS 101. I ran it on 30 years of NBA drafts. The 2003 class is an outlier by every metric."*

---

### Model 04 — Churn / Retention

**Discipline:** Marketing DS  
**The Question:** *Can we predict which NBA players are about to "churn" out of a team's rotation — and which early warning signals are most predictive?*

#### Data Sources (choose one)
| Option | Source | What to pull | How |
|--------|--------|-------------|-----|
| ⭐ Basketball-Reference | `sportsipy` | Season-by-season player stats, minutes played, age, team | `from sportsipy.nba.players import Players` |
| Soccer Alternative | `FBref` via `soccerdata` | Minutes played per season, age, club changes | `import soccerdata as sd` |
| Kaggle | Kaggle NBA Dataset | Multi-season player stats CSV | Download from Kaggle |

#### Approach
Binary classification: label a player as "churned" if their minutes dropped >50% the following season OR they left the league. Train XGBoost on prior-season stats. Use SHAP for explainability. Validate against held-out seasons.

#### Step-by-Step Plan
```
1. DATA COLLECTION
   - Pull 20 seasons of NBA player stats (2003–2023)
   - Features: Age, MPG, PTS, AST, REB, TS%, USG%, PER, +/-, contract year flag
   - Target: did this player's role significantly decline next season?

2. LABEL CREATION
   - Define churn: minutes played dropped >40% season-over-season OR retired/left league
   - Define retained: minutes within 20% of prior season
   - Remove ambiguous cases (injury seasons) or flag separately

3. FEATURE ENGINEERING
   - Rolling 2-year average for each stat (trend, not just snapshot)
   - Age curve: quadratic age term
   - Efficiency metrics: TS%, PER
   - Usage trend: is USG% declining?
   - Season type: playoff team vs lottery team

4. MODELLING
   - Train/test split: train on 2003–2018, test on 2019–2023
   - Model: XGBoost classifier
   - Tune: GridSearchCV on max_depth, learning_rate, n_estimators
   - Evaluate: AUC-ROC, precision/recall, confusion matrix

5. EXPLAINABILITY
   - SHAP summary plot: global feature importance
   - SHAP waterfall plot: explain individual player predictions
   - Find: what's the #1 signal of impending churn?

6. VALIDATION
   - Pick 5 players the model flagged in 2022 — did they churn in 2023?
   - Discuss false positives (injury vs genuine decline)

7. VISUALIZATION
   - Feature importance bar chart
   - SHAP summary beeswarm plot
   - ROC curve
   - "Churn risk leaderboard" for current season players

8. OUTPUT
   - Save: churn_model.pkl
   - Save: churn_predictions.csv (player, churn_prob, top_shap_features)
   - Save: shap_plots.png
```

**Key Output:** Churn probability for all current NBA players + SHAP explanation of why each was flagged.  
**LinkedIn Hook:** *"I built a customer churn model — but the customers were NBA players. SHAP revealed the #1 warning sign."*

---

### Model 05 — LTV + CAC / Payback

**Discipline:** Marketing DS + Growth DS  
**The Question:** *Can we predict a player's total career value (LTV) from just their first 3 seasons — and when we factor in salary (CAC), which acquisition channel (draft vs free agent vs trade) offers the best LTV:CAC ratio?*

#### Data Sources (choose one)
| Option | Source | What to pull | How |
|--------|--------|-------------|-----|
| ⭐ Basketball-Reference + Spotrac | `sportsipy` + manual scrape | Career Win Shares, salary history | `sportsipy` for stats; scrape Spotrac for salaries |
| Baseball Alternative | `sportsipy` MLB | Career WAR by season, contract values | `from sportsipy.mlb.players import Players` |
| Kaggle | Kaggle NBA salaries dataset | Pre-compiled salary + stats | Search Kaggle for "NBA salaries" |

#### Approach
Two-part model. Part 1: regression to predict career Win Shares from Year 1–3 stats (LTV model). Part 2: join with salary data, calculate LTV:CAC ratio by acquisition type. Use BG/NBD-style thinking adapted to sports careers.

#### Step-by-Step Plan
```
1. DATA COLLECTION
   - Pull all NBA players 1990–2015 (need full career data to label LTV)
   - Collect: Year 1, 2, 3 stats as features
   - Collect: total career Win Shares as target (LTV)
   - Collect: first contract salary as CAC proxy
   - Tag: how was player acquired? (Draft pick # / Free agent / Trade)

2. LTV MODEL
   - Features: Year 1–3 PER, Win Shares, TS%, USG%, Age, Draft position
   - Target: Total career Win Shares (or Win Shares per 48 * career minutes)
   - Model: Gradient Boosting Regressor
   - Evaluate: RMSE, R², residual plot
   - Key question: does draft position matter after controlling for early stats?

3. CAC CALCULATION
   - CAC = total salary paid in first 4 years (rookie contract)
   - For free agents: first contract total value
   - For trades: use Spotrac trade value estimates

4. LTV:CAC ANALYSIS
   - Calculate LTV:CAC ratio per player
   - Group by acquisition channel (draft round, free agent tier, trade)
   - Which channel has best median LTV:CAC?
   - Plot payback period: at what career year does player "pay back" their contract?

5. INSIGHTS
   - Late first-round picks with high LTV:CAC (hidden value)
   - Free agents with worst LTV:CAC (overpaid)
   - Best LTV:CAC by position

6. VISUALIZATION
   - Scatter: predicted vs actual career LTV
   - Box plot: LTV:CAC by acquisition channel
   - Payback curve: cumulative Win Shares vs cumulative salary over career years

7. OUTPUT
   - Save: ltv_model.pkl
   - Save: player_ltv_cac.csv
   - Save: acquisition_channel_analysis.png
```

**Key Output:** LTV:CAC comparison by acquisition type. The post: draft picks have ~3x better LTV:CAC than free agents.  
**LinkedIn Hook:** *"GMs obsess over player value. Growth teams obsess over LTV:CAC. I built both for the NBA — and the results surprised me."*

---

### Model 06 — Propensity Modeling

**Discipline:** Marketing DS  
**The Question:** *Can we build a model that gives a real-time probability of a team winning the title — updated weekly throughout the season — and how does it compare to betting market implied probability?*

#### Data Sources (choose one)
| Option | Source | What to pull | How |
|--------|--------|-------------|-----|
| ⭐ FiveThirtyEight + FBref | `ClubElo API` + `soccerdata` | Historical ELO, match results, xG | `requests.get("http://api.clubelo.com/Arsenal")` |
| NBA Alternative | `nba_api` + Basketball-Reference | Season records, point differential, SRS | `nba_api` endpoints |
| Football-Data.org | `football-data.org` | Match results, standings, historical | Free API key |

#### Approach
Logistic regression / LightGBM classifier. Target = did this team win the title that season? Features = mid-season stats at various week marks. Train on historical seasons, predict current. Calibrate probabilities using Platt scaling. Compare to betting odds as ground truth.

#### Step-by-Step Plan
```
1. DATA COLLECTION
   - Pull 10+ seasons of EPL or NBA data
   - For each team-season, record weekly snapshots of:
     Points/wins, goal difference/point differential, xG for/against,
     ELO rating, injuries (binary: star player injured?), form (last 5 games)
   - Label: did this team win the championship? (binary)

2. FEATURE ENGINEERING
   - Week-in-season flag (week 10 vs week 30 features mean different things)
   - Points pace: projected final points at current rate
   - xG differential: are they outperforming or underperforming?
   - Remaining schedule difficulty
   - Historical: how often do teams with X points at week Y win the title?

3. MODELLING
   - Train: all seasons except last 2
   - Model: LightGBM classifier (handles non-linearity well)
   - Calibrate: Platt scaling / isotonic regression for probability calibration
   - Evaluate: Brier score, calibration curve, AUC-ROC

4. PROBABILITY TIMELINE
   - For each historical season: plot weekly title probability for all teams
   - Creates the "drama arc" of a season in data form

5. VALIDATION
   - Compare model probabilities at mid-season to betting odds
   - Are there systematic differences? (model overconfident on leaders?)

6. VISUALIZATION
   - Line chart: weekly title probability for top 4–6 teams across a season
   - Calibration plot: predicted vs actual probability
   - "Season drama score": variance in probability over the season

7. OUTPUT
   - Save: propensity_model.pkl
   - Save: weekly_probabilities.csv
   - Save: probability_timeline.html (interactive Plotly)
```

**Key Output:** Animated probability chart showing a full title race. The post angle is about model calibration.  
**LinkedIn Hook:** *"Propensity models score leads. I built one that scores title chances — updated every week. Model calibration is the key nobody talks about."*

---

## Phase 2: Prediction & Discovery
*Months 4–6 · Models 7–12*

---

### Model 07 — Survival Analysis

**Discipline:** Marketing DS  
**The Question:** *At what point in their career do players at different positions hit irreversible decline — and what factors extend or shorten the survival curve?*

#### Data Sources (choose one)
| Option | Source | What to pull | How |
|--------|--------|-------------|-----|
| ⭐ Basketball-Reference | `sportsipy` | Season-by-season stats, age, position, career length | `from sportsipy.nba.players import Players` |
| Soccer Alternative | `FBref` via `soccerdata` | Career logs, position, minutes per season | `soccerdata` |
| Baseball | `sportsipy` MLB | Career WAR by age, position, career length | `from sportsipy.mlb.players import Players` |

#### Approach
Kaplan-Meier survival curves by position. Cox Proportional Hazard regression to find what extends careers. "Death event" = career decline (minutes <500/season or retirement). Time = career age.

#### Step-by-Step Plan
```
1. DATA COLLECTION
   - Pull 20+ years of NBA player data
   - Define "survival event": season where minutes < 500 (functional career end)
   - Define time variable: career age (not calendar age — years since debut)
   - Features for Cox model: position, peak PER, peak TS%, draft round, peak USG%

2. SURVIVAL DATA SETUP
   - For each player: duration = career length in seasons
   - Event = 1 if they "churned", 0 if still playing (censored)
   - Use lifelines library format

3. KAPLAN-MEIER CURVES
   - Plot KM curves by position (PG, SG, SF, PF, C)
   - Log-rank test: are position survival curves significantly different?
   - Median survival time per position

4. COX REGRESSION
   - Features: position, peak PER, peak TS%, draft position, body type proxy (height/weight)
   - Interpret hazard ratios: what doubles/halves the risk of decline?
   - Check proportional hazard assumption

5. ANALYSIS
   - Which position declines earliest?
   - What's the most protective factor (extends career)?
   - Build a "career survival probability" for current players

6. VISUALIZATION
   - KM curves by position (survival probability vs career year)
   - Cox regression forest plot (hazard ratios with CI)
   - Individual player survival curves for 5 famous players

7. OUTPUT
   - Save: survival_curves.png
   - Save: cox_model.pkl
   - Save: player_survival_probs.csv
```

**Key Output:** KM curves by position — visually compelling, immediately shareable.  
**LinkedIn Hook:** *"Survival analysis tells you when customers leave. I used it to find when NBA players peak — by position. The curves are beautiful and brutal."*

---

### Model 08 — Forecasting

**Discipline:** Marketing DS + Growth DS  
**The Question:** *Given mid-season data, which forecasting model best predicts final Premier League standings — and does a simple baseline beat the complex models?*

#### Data Sources (choose one)
| Option | Source | What to pull | How |
|--------|--------|-------------|-----|
| ⭐ Football-Data.org | Free API key | Match results, standings by matchweek, home/away | `requests` with API key |
| NBA Alternative | `nba_api` | Season game log, standings by week | `nba_api.stats.endpoints` |
| FBref | `soccerdata` | Historical season standings, xG per match | `import soccerdata as sd` |

#### Approach
Compare 3 forecasting approaches: (1) Simple extrapolation baseline, (2) ARIMA/SARIMA, (3) Facebook Prophet. Train on first half of season, forecast second half. Evaluate with RMSE on final points. Key learning: simpler often wins.

#### Step-by-Step Plan
```
1. DATA COLLECTION
   - Pull 10 seasons of EPL results from Football-Data.org
   - Build rolling standings table: after each matchweek, what are team points?
   - This creates time series: team_name → weekly_points (matchweek 1 to 38)

2. TRAIN/TEST SETUP
   - For each season: train on matchweeks 1–19, forecast matchweeks 20–38
   - Evaluate: how well does mid-season data predict final table?
   - Run this for each of 10 seasons = 10 evaluation sets

3. MODEL 1: BASELINE (SIMPLE EXTRAPOLATION)
   - Take points per game in first half, multiply by remaining games
   - Add home/away fixture difficulty adjustment
   - This is the "dumb" benchmark — everything must beat this

4. MODEL 2: ARIMA
   - For each team: fit ARIMA on points time series
   - Forecast remaining matchweeks
   - statsmodels auto_arima for parameter selection

5. MODEL 3: PROPHET
   - Treat each team's points trajectory as a time series
   - Add "playoff pressure" as a regressor (binary: in title race?)
   - Facebook Prophet for each team

6. COMPARISON
   - Evaluate each model: RMSE on final points, position accuracy (did they get top 4 right?)
   - Create comparison table
   - Key finding: does the baseline beat the complex models?

7. VISUALIZATION
   - Line chart: actual vs forecast points trajectories (best/worst performing teams)
   - Bar chart: model RMSE comparison
   - Table: final position predicted vs actual for each model

8. OUTPUT
   - Save: forecasting_comparison.csv
   - Save: model_accuracy.png
   - Save: forecast_trajectories.html (interactive)
```

**Key Output:** Model comparison table showing simple baseline vs ARIMA vs Prophet. The insight that simple wins is the LinkedIn hook.  
**LinkedIn Hook:** *"I tested 3 forecasting models on 10 seasons of EPL data. The simplest one won 7 out of 10 times. Here's why that matters for marketing forecasts."*

---

### Model 09 — Recommendation Models

**Discipline:** Product DS  
**The Question:** *Can we build a "players like this" recommender that finds statistically similar players across different eras and leagues — and does it surface non-obvious comparisons?*

#### Data Sources (choose one)
| Option | Source | What to pull | How |
|--------|--------|-------------|-----|
| ⭐ FBref | `soccerdata` | Per-90 stats for players across 5 top leagues, 5 seasons | `import soccerdata as sd; sd.FBref()` |
| NBA Alternative | `nba_api` + Basketball-Reference | Advanced stats, per-36 stats, multiple seasons | `nba_api` |
| Combined | Both NBA + Soccer | Cross-sport similarity ("this NBA player plays like this footballer") | Both above |

#### Approach
Item-item collaborative filtering using cosine similarity on normalized per-90 stats. Build player embeddings, use FAISS for efficient nearest-neighbor search. Wrap in a Streamlit app for interactive querying.

#### Step-by-Step Plan
```
1. DATA COLLECTION
   - Pull 5 seasons of FBref data for Premier League, La Liga, Bundesliga
   - Stats: per-90 xG, xA, progressive carries, passes, pressures, tackles, aerials, dribbles
   - Minimum 900 minutes per season
   - Join multiple seasons per player (use most recent season)

2. FEATURE ENGINEERING
   - Normalize all stats to per-90 (FBref already does this)
   - Create position-adjusted scores: compare strikers to strikers, etc.
   - Option: also include cross-position comparison (flatten all together)
   - StandardScaler on all features

3. SIMILARITY ENGINE
   - Method 1: cosine similarity matrix (all vs all — works for <5000 players)
   - Method 2: FAISS index for faster search (better for scale)
   - For each player: store top-10 most similar players + similarity scores

4. VALIDATION
   - Manual sanity check: does "Erling Haaland" return other elite strikers?
   - Interesting test: does it find cross-league equivalents? (Bundesliga striker ≈ PL striker)
   - Find the most surprising/non-obvious comparisons

5. STREAMLIT APP
   - Input: player name (autocomplete)
   - Output: top 5 similar players with similarity score, radar chart comparison
   - Radar chart: show stat profiles of input player vs recommendations
   - Deploy: Streamlit Cloud (free)

6. OUTPUT
   - Save: player_similarity_matrix.pkl
   - Save: streamlit_app.py
   - Deploy live Streamlit app
   - Save: radar_chart_examples.png (for LinkedIn post)
```

**Key Output:** Live Streamlit app — type a player, get 5 similar ones. This is the most shareable output in the portfolio.  
**LinkedIn Hook:** *"I built a 'players like this' recommender. It found comparisons nobody expected. [Link to live app]"*

---

### Model 10 — Sentiment Analysis

**Discipline:** Marketing DS + Product DS  
**The Question:** *How does fan sentiment on Reddit shift after major transfer announcements — and does sentiment predict short-term match performance or ticket demand?*

#### Data Sources (choose one)
| Option | Source | What to pull | How |
|--------|--------|-------------|-----|
| ⭐ Reddit via PRAW | `PRAW` | Posts and comments from r/soccer, r/PremierLeague, or team subreddits | `pip install praw` + free API key |
| Twitter/X Alternative | `snscrape` or Kaggle Twitter dataset | Tweets about team/player before and after events | `pip install snscrape` (scraper) |
| News headlines | `newsapi.org` | Sports headlines free tier | Free API key at newsapi.org |

#### Approach
Two-stage: (1) Sentiment scoring with VADER (fast, good for social) + HuggingFace DistilBERT (more accurate). (2) Topic modeling with BERTopic to find what fans actually discuss. Correlate sentiment with subsequent match results.

#### Step-by-Step Plan
```
1. EVENT SELECTION
   - Choose 3–5 major transfer events or match results to analyze
   - Example: a surprise signing, a shock defeat, a title win
   - Define time windows: 48h before and 72h after each event

2. DATA COLLECTION (Reddit)
   - Register app at reddit.com/prefs/apps (free)
   - Use PRAW to pull posts from r/soccer or team subreddit
   - Pull: title, body text, score (upvotes), num_comments, timestamp
   - Target: ~500–2000 posts per event window

3. PREPROCESSING
   - Remove URLs, emojis (or encode them), usernames
   - Lowercase, basic cleaning
   - Keep structure: timestamp, text, upvote_score

4. SENTIMENT SCORING
   - Method 1: VADER (lexicon-based, fast, good for social media slang)
   - Method 2: DistilBERT fine-tuned on Twitter (via HuggingFace transformers)
   - Create: compound sentiment score per post, hourly aggregate

5. TOPIC MODELING
   - Apply BERTopic to identify what fans are discussing
   - Find: do topics shift before/after the event?
   - Key finding: do fans discuss tactics, players, or emotions most?

6. CORRELATION ANALYSIS
   - Does pre-match sentiment predict match outcome?
   - Does post-signing sentiment predict next game attendance?
   - Use Pearson/Spearman correlation; be honest about weak signal

7. VISUALIZATION
   - Timeline: sentiment score over 7-day window around event
   - BERTopic cluster visualization
   - Word cloud per topic
   - Correlation scatter: sentiment vs next match result

8. OUTPUT
   - Save: sentiment_data.csv
   - Save: sentiment_timeline.html (interactive)
   - Save: topic_model/ (BERTopic saved model)
```

**Key Output:** Sentiment timeline around a major event + surprising correlation with on-pitch performance.  
**LinkedIn Hook:** *"I analyzed Reddit after a blockbuster transfer announcement. Fan sentiment collapsed — then the team won 3 straight. The data is more interesting than the narrative."*

---

### Model 11 — Onboarding / Activation Analysis

**Discipline:** Product DS  
**The Question:** *Which early-career actions and stats in a player's first season are most predictive of long-term success — what is the "aha moment" for NBA rookies?*

#### Data Sources (choose one)
| Option | Source | What to pull | How |
|--------|--------|-------------|-----|
| ⭐ Basketball-Reference | `sportsipy` | Rookie season stats + career outcomes | `sportsipy` |
| Soccer Alternative | `FBref` via `soccerdata` | First season per-90 stats, age, subsequent career | `soccerdata` |
| Kaggle | Kaggle NBA draft dataset | Pre-cleaned draft + career stats | Download from Kaggle |

#### Approach
Binary classification: did a player have a "successful" career (>5 seasons of significant contribution)? Features = rookie season stats only. Use logistic regression for interpretability + feature importance. The goal is to find the single strongest predictor — the "aha moment" equivalent.

#### Step-by-Step Plan
```
1. DATA COLLECTION
   - Pull all NBA rookies 1990–2015 (need full career to label outcome)
   - Rookie season features: PTS, AST, REB, STL, BLK, TS%, USG%, MPG, draft position, age
   - Career outcome label: did player have ≥5 seasons with ≥1000 minutes? (binary success)

2. LABEL CREATION
   - Success = played ≥5 seasons with ≥1000 minutes AND had ≥1 All-Star/significant role
   - Adjust threshold — aim for ~30% success rate (realistic)
   - Flag: remove injury-shortened careers from labeling

3. FEATURE IMPORTANCE ANALYSIS
   - Logistic regression (interpretable coefficients)
   - XGBoost (non-linear, higher accuracy)
   - SHAP values on XGBoost
   - Key question: which single rookie stat best predicts success?

4. THE "AHA MOMENT" EQUIVALENT
   - Find the threshold: e.g., "rookies who average >1.5 AST/game in Year 1 have 3x higher success rate"
   - This is the sports equivalent of Facebook's "7 friends in 10 days"
   - Test multiple potential thresholds

5. DRAFT POSITION ANALYSIS
   - After controlling for rookie stats, does draft position still matter?
   - Does high draft pick + poor rookie stats still predict failure?

6. COHORT VALIDATION
   - Apply model to 2018–2020 rookies: who does it predict will succeed?
   - Check against current reality

7. VISUALIZATION
   - Feature importance bar chart
   - Decision boundary: rookie stat X vs outcome (scatter with logistic curve)
   - "Aha moment" threshold visualization

8. OUTPUT
   - Save: activation_model.pkl
   - Save: rookie_predictions.csv
   - Save: aha_moment_chart.png
```

**Key Output:** The single rookie stat that best predicts career success. The "aha moment" framing makes it immediately relatable to product DS.  
**LinkedIn Hook:** *"Facebook found '7 friends in 10 days' predicts retention. I found the NBA rookie equivalent. One stat predicts long-term success better than draft position."*

---

### Model 12 — Engagement Modeling

**Discipline:** Product DS  
**The Question:** *What factors actually drive a fan to watch every match — and is winning the primary driver, or is there something more interesting going on?*

#### Data Sources (choose one)
| Option | Source | What to pull | How |
|--------|--------|-------------|-----|
| ⭐ Google Trends + Football-Data.org | `pytrends` + free API key | Search volume as engagement proxy, match results, standings | `pytrends` + `requests` |
| TV Ratings | Public broadcast data + Wikipedia | Match TV ratings, league position, star players | Manual collection |
| Reddit engagement | `PRAW` | Post volume, comment activity on team subreddits by week | `praw` |

#### Approach
Regression model predicting engagement proxy (Google search volume or Reddit post volume) from match importance, rivalry status, league position, recent results, star player presence. SHAP to find the primary driver. Test hypothesis: is it winning, or is it drama/uncertainty?

#### Step-by-Step Plan
```
1. DATA COLLECTION
   - Pull Google Trends for 5 Premier League clubs (5-year weekly data) via pytrends
   - Pull match results + standings from Football-Data.org
   - Build features per matchweek:
     League position, points gap to leaders, opponent quality,
     Is it a rivalry match?, Recent form (W/D/L last 5),
     Star player availability (injury data from news headlines),
     Match outcome (won/drew/lost)

2. ENGAGEMENT PROXY
   - Use relative Google search volume as engagement metric (100 = peak interest)
   - Normalise within each club (to control for club size)
   - Lag: does a big win drive engagement the WEEK AFTER?

3. FEATURE ENGINEERING
   - Match importance score: points gap × games remaining
   - Derby/rivalry flag (binary)
   - Title race involvement (within 5 points of leaders at matchweek N)
   - Consecutive wins/losses streak
   - Days since last match (fatigue proxy)

4. MODELLING
   - XGBoost regression: predict next week's search volume
   - SHAP analysis: what drives engagement most?
   - Test key hypothesis: does losing occasionally INCREASE engagement? (drama effect)

5. SEGMENTATION
   - Does the engagement model differ by club size?
   - Big clubs: winning drives search
   - Smaller clubs: drama/survival race drives search

6. VISUALIZATION
   - SHAP beeswarm: top engagement drivers
   - Line chart: actual engagement vs predicted across 5 seasons
   - Scatter: match importance vs engagement (with annotations for famous games)

7. OUTPUT
   - Save: engagement_model.pkl
   - Save: engagement_predictions.csv
   - Save: shap_engagement_drivers.png
```

**Key Output:** SHAP plot revealing that "drama" (close title race, relegation battle) drives more engagement than simply winning.  
**LinkedIn Hook:** *"Product teams optimize for engagement. I built an engagement model for football fans. Winning isn't the biggest driver — drama is. Here's the data."*

---

## Phase 3: Measurement & Causality
*Months 7–9 · Models 13–18*

---

### Model 13 — Attribution Modeling

**Discipline:** Marketing DS  
**The Question:** *When a goal is scored, who really deserves credit — and how radically does the answer change depending on which attribution model you use?*

#### Data Sources (choose one)
| Option | Source | What to pull | How |
|--------|--------|-------------|-----|
| ⭐ StatsBomb Open Data | `statsbombpy` | Event-level data: passes, carries, shots, assists for free matches | `pip install statsbombpy` |
| NBA Play-by-Play | `nba_api` play-by-play | Assist chains, screen assists, secondary assists | `nba_api.stats.endpoints.playbyplayv3` |
| Kaggle Soccer Events | Kaggle "Soccer match event dataset" | Spatio-temporal events for 7 competitions | Download from Kaggle |

#### Approach
Build 3 attribution models on the same data: (1) Last-touch (just the scorer), (2) Linear (equal credit across pass chain), (3) Shapley value (game-theoretic fair credit). Compare results dramatically — this is the "attribution is broken" post.

#### Step-by-Step Plan
```
1. DATA COLLECTION (StatsBomb)
   - Install: pip install statsbombpy
   - Load free data: from statsbombpy import sb; events = sb.events(match_id=...)
   - Focus on: passes, carries, shots, goals
   - Reconstruct possession sequences leading to each goal

2. POSSESSION CHAIN RECONSTRUCTION
   - For each goal: trace back through events to find the full sequence
   - Identify all players who touched the ball in the 10 events before the goal
   - Build: [player_a: 2 passes, player_b: 1 carry, player_c: assist, player_d: goal]

3. MODEL 1: LAST-TOUCH ATTRIBUTION
   - 100% credit to goal scorer
   - Baseline — represents "everyone uses last-click"

4. MODEL 2: LINEAR ATTRIBUTION
   - Equal credit across all players in the possession chain
   - Adjust for number of touches per player

5. MODEL 3: SHAPLEY VALUE ATTRIBUTION
   - Implement Shapley value calculation for each goal sequence
   - Treat each player's inclusion/exclusion as a coalition
   - Computationally intensive — may need to simplify for long chains

6. COMPARISON
   - For 100 goals: how does credit distribution change across models?
   - Find: which model most credits "invisible" players (set-piece specialists, press-causers)?
   - Find: which player is most undervalued by last-touch vs Shapley?

7. VISUALIZATION
   - Stacked bar: credit by player per model for 5 example goals
   - Scatter: last-touch credit vs Shapley credit per player (season total)
   - Annotate outliers: "this player contributes far more than their goals suggest"

8. OUTPUT
   - Save: attribution_comparison.csv
   - Save: attribution_plots.html (interactive)
   - Save: shapley_implementation.py
```

**Key Output:** Side-by-side attribution model comparison showing dramatic disagreement. The Shapley result surfaces undervalued players.  
**LinkedIn Hook:** *"Marketing attribution is broken. Basketball attribution has the exact same problem. I ran 3 models on the same goals — they disagreed by 400%. Here's why."*

---

### Model 14 — Marketing Mix Modeling (MMM)

**Discipline:** Marketing DS  
**The Question:** *What combination of factors — rest days, travel distance, home advantage, recent form, and weather — best explains match outcomes, and how do we model saturation and diminishing returns?*

#### Data Sources (choose one)
| Option | Source | What to pull | How |
|--------|--------|-------------|-----|
| ⭐ Football-Data.org + Open-Meteo | Free API key + free weather API | Match results, home/away, stadium locations → weather lookup | `football-data.org` + `open-meteo.com` |
| FBref | `soccerdata` | xG, match stats, team form | `soccerdata` |
| Combined | Football-Data.org + FBref + Open-Meteo | Full feature set | All three |

#### Approach
Bayesian MMM using PyMC-Marketing. This is the same framework used in professional marketing analytics. Target = goals scored (continuous) or match result (ordered). Include adstock/saturation transformations on variables like "rest days" and "form".

#### Step-by-Step Plan
```
1. DATA COLLECTION
   - Match results + dates from Football-Data.org (5 seasons)
   - Calculate rest days: days since last match (scrape fixture list)
   - Calculate travel distance: stadium lat/long → haversine distance (geopy)
   - Pull weather: Open-Meteo historical API for match date/location
     Variables: temperature, precipitation, wind speed
   - Form: rolling 5-game points per game
   - Home/away flag

2. FEATURE ENGINEERING
   - Rest days: continuous (3–14 days typically)
   - Travel distance: km (0 for home games)
   - Weather severity score: combine temp/wind/rain
   - Form: rolling average points last 5 games
   - Opponent strength: opponent ELO from ClubElo API
   - Season stage: early/mid/late season flag

3. MMM SETUP (PyMC-Marketing)
   - Target: goals scored per match (or win probability)
   - Apply saturation transformation to rest days (diminishing returns above 7 days)
   - Apply adstock to form (recent form decays over time)
   - Fit Bayesian linear model with priors

4. MODEL FITTING
   - Use PyMC-Marketing's MMM class
   - Markov Chain Monte Carlo (MCMC) sampling
   - Check convergence: R-hat, trace plots
   - Posterior predictive checks

5. CONTRIBUTION ANALYSIS
   - Decompose each match outcome into contributions per factor
   - Build waterfall chart: baseline + rest contribution + home advantage + weather + form
   - Response curves: how does each factor's contribution change at different levels?

6. KEY INSIGHTS
   - At what rest level does benefit plateau? (saturation)
   - How much does home advantage contribute vs opponent strength?
   - Is weather a significant factor?

7. VISUALIZATION
   - Contribution waterfall chart (example match)
   - Response curves per factor
   - Posterior distribution plots (showing uncertainty)

8. OUTPUT
   - Save: mmm_model.pkl
   - Save: contribution_data.csv
   - Save: response_curves.png
   - Save: waterfall_examples.html
```

**Key Output:** Contribution chart showing how much each factor drives match outcomes. Response curves showing saturation.  
**LinkedIn Hook:** *"MMM is the most in-demand skill in marketing DS right now. I learned it using football match data. Travel distance is the silent killer nobody models."*

---

### Model 15 — A/B Testing + Causal Inference

**Discipline:** Growth DS + Product DS  
**The Question:** *Did a manager change actually improve a team's results — or would they have improved anyway? Using causal inference to separate signal from noise.*

#### Data Sources (choose one)
| Option | Source | What to pull | How |
|--------|--------|-------------|-----|
| ⭐ FBref + Understat | `soccerdata` + `understat` | Match results, xG before/after manager change | `soccerdata` + `pip install understat` |
| Football-Data.org | Free API key | Match results pre/post intervention | API |
| Soccerway scrape | Manual | Manager change dates, results | Manual + BeautifulSoup |

#### Approach
Two methods: (1) Simple interrupted time series with confidence intervals, (2) Bayesian structural time series (CausalImpact equivalent). Test multiple manager changes to build generalizable findings. Explicitly discuss regression to the mean as confounder.

#### Step-by-Step Plan
```
1. DATA COLLECTION
   - Identify 10+ manager changes in top leagues (2015–2023)
   - For each: collect 15 matches before and 15 matches after the change
   - Metric: xG per match (better than goals — removes luck)
   - Also collect: league position, opponent strength per match

2. NAIVE BEFORE/AFTER ANALYSIS
   - Calculate mean xG before vs after for each manager change
   - Simple t-test: is the difference significant?
   - Finding: ~60% show improvement but this is partly regression to mean

3. CONFOUNDER CONTROL
   - Opponent strength adjustment: was schedule easier after the change?
   - League position at time of change: teams fired when performing worst (regression to mean)
   - Injury context: key players returning?

4. CAUSAL IMPACT (BAYESIAN)
   - Use the `causalimpact` Python package
   - Build counterfactual: what would have happened without the change?
   - Uses pre-intervention period to model the synthetic control
   - Run for 10 different manager changes

5. DIFFERENCE-IN-DIFFERENCES
   - Find "comparable" teams who didn't change managers in the same period
   - Compare trajectory: intervention team vs control team
   - DiD estimate = true treatment effect

6. META-ANALYSIS
   - Across 10 manager changes: what's the average causal effect?
   - Separate: "sacking effect" (short-term boost) vs sustained improvement
   - Which types of clubs benefit most from manager changes?

7. VISUALIZATION
   - CausalImpact plot: actual vs counterfactual xG trajectory
   - Forest plot: estimated effect per manager change with CI
   - Scatter: severity of decline before firing vs post-firing improvement

8. OUTPUT
   - Save: causal_impact_results.csv
   - Save: causal_plots/ (one per manager change)
   - Save: meta_analysis_summary.png
```

**Key Output:** CausalImpact chart with counterfactual. The finding: ~40% of "successful" manager changes are regression to the mean.  
**LinkedIn Hook:** *"Did the new manager actually improve results — or was it regression to the mean? I ran causal inference on 10 manager changes. The answer will upset a lot of fans."*

---

### Model 16 — Incrementality Testing

**Discipline:** Growth DS  
**The Question:** *Did a major transfer signing cause team improvement — or were they already improving before the player arrived?*

#### Data Sources (choose one)
| Option | Source | What to pull | How |
|--------|--------|-------------|-----|
| ⭐ Understat + FBref | `understat` + `soccerdata` | Match-level xG, team performance metrics by week | Both free |
| Football-Data.org | Free API key | Match results timeline | API |
| Transfermarkt | `transfermarkt-scraper` | Transfer dates, fees, selling/buying clubs | `pip install transfermarkt-scraper` |

#### Approach
CausalImpact analysis for individual transfers. Build a synthetic control using similar teams who didn't make a major signing. Compare actual post-signing trajectory vs counterfactual. Quantify the "incremental" impact of the transfer.

#### Step-by-Step Plan
```
1. EVENT SELECTION
   - Choose 5 high-profile transfers from last 5 seasons
   - Mix: clear successes (became key player), clear failures, ambiguous cases
   - Record exact signing date and debut

2. DATA COLLECTION
   - Team xG per match for the buying club (weekly/matchweek)
   - 20 matches pre-signing, 20 matches post-debut
   - Control group: similar teams who were also performing at same level pre-signing but didn't sign

3. SYNTHETIC CONTROL
   - Find 3–5 "donor" teams similar to the treatment team pre-signing
   - Optimally weight donors to best replicate pre-signing trajectory
   - The synthetic control = what would have happened without the signing

4. CAUSAL IMPACT
   - Apply `causalimpact` package
   - Compare actual post-debut performance vs synthetic control
   - Calculate: cumulative incremental xG attributed to the signing
   - Confidence intervals: is the effect statistically significant?

5. HETEROGENEITY ANALYSIS
   - Do expensive signings have higher incrementality?
   - Do signings for struggling teams have higher incrementality than signings for strong teams?
   - Is incrementality higher for signings in the team's weak position?

6. FINDINGS
   - Most honest finding: many "successful" transfers are not as incremental as they appear
   - Some "failed" transfers actually had positive incrementality despite negative perception

7. VISUALIZATION
   - CausalImpact timeline chart per transfer
   - Summary: incremental xG vs transfer fee (ROI equivalent)
   - League table: best and worst "incremental" transfers

8. OUTPUT
   - Save: incrementality_results.csv
   - Save: causal_impact_charts/ (folder of individual charts)
   - Save: transfer_roi_scatter.png
```

**Key Output:** Transfer ROI scatter: incremental xG improvement vs transfer fee. Some huge signings had near-zero incrementality.  
**LinkedIn Hook:** *"Incrementality testing is how growth teams avoid fooling themselves. I applied it to football transfers. Some £100M signings had near-zero incremental impact."*

---

### Model 17 — Uplift Modeling

**Discipline:** Growth DS  
**The Question:** *Which players actually respond to tactical/coaching interventions — and which would have improved anyway regardless of what the coach did?*

#### Data Sources (choose one)
| Option | Source | What to pull | How |
|--------|--------|-------------|-----|
| ⭐ FBref | `soccerdata` | Player stats before/after manager change (same player, different manager) | `soccerdata` |
| Understat | `understat` | Player xG/xA before/after intervention point | `pip install understat` |
| StatsBomb Open Data | `statsbombpy` | Event-level data to build player-level metrics | `statsbombpy` |

#### Approach
Meta-learner uplift model (T-learner or S-learner). Treatment = player played under new manager/system. Control = player continued under old system (or similar players who didn't change). Outcome = performance improvement. Build Qini curve to evaluate model lift.

#### Step-by-Step Plan
```
1. DATA COLLECTION
   - Identify players who experienced a manager change (treatment group)
   - Find comparable players on other teams who didn't (control group)
   - Match on: position, age, prior season stats, team strength
   - Feature: player stats in season before change
   - Outcome: performance in season after change (xG+xA per 90)

2. TREATMENT/CONTROL SETUP
   - Treatment: player whose manager changed mid-season or between seasons
   - Control: player with same profile at club without manager change
   - Propensity score matching to ensure comparable groups

3. UPLIFT MODEL (T-LEARNER)
   - Train separate models for treatment and control groups
   - T-model (treatment): predict outcome for treated players
   - C-model (control): predict outcome for control players
   - Uplift = T-model prediction - C-model prediction

4. PLAYER SEGMENTATION
   - Segment players by predicted uplift:
     "Persuadables": improve because of intervention
     "Sure Things": improve regardless  
     "Lost Causes": won't improve regardless
     "Sleeping Dogs": perform worse with intervention
   - This is the exact framework growth teams use for campaign targeting

5. QINI CURVE
   - Evaluate model: plot Qini curve (uplift equivalent of ROC)
   - Compare vs random targeting

6. INSIGHTS
   - Which positions respond most to manager changes?
   - Does age predict responsiveness to coaching?
   - Are expensive players more or less coachable?

7. VISUALIZATION
   - Uplift score distribution histogram
   - Scatter: prior performance vs uplift score (find the "coachable" players)
   - Qini curve vs random baseline

8. OUTPUT
   - Save: uplift_model.pkl
   - Save: player_uplift_scores.csv
   - Save: qini_curve.png
```

**Key Output:** Player segmentation by uplift: persuadables vs sure things vs lost causes. The Qini curve.  
**LinkedIn Hook:** *"Propensity asks 'who will improve?' Uplift asks 'who improves BECAUSE OF us?' Huge difference. I built an uplift model for football coaching. Some players are coachable. Most aren't."*

---

### Model 18 — Budget Allocation & Optimization

**Discipline:** Growth DS  
**The Question:** *Given a fixed transfer budget, how should a club optimally allocate spend across positions to maximize expected points — and where does the marginal return diminish?*

#### Data Sources (choose one)
| Option | Source | What to pull | How |
|--------|--------|-------------|-----|
| ⭐ Transfermarkt + FBref | `transfermarkt-scraper` + `soccerdata` | Player valuations, stats by position | Both free |
| FBref only | `soccerdata` | Stats + approximate market values from FBref | `soccerdata` |
| Football-Data.org + FBref | Combined | Match results + player data | Both free |

#### Approach
Two-stage optimization. Stage 1: fit a response curve for each position (investment → expected xG contribution). Stage 2: constrained optimization (scipy/PuLP) to find budget allocation that maximizes total expected xG across the squad. Show saturation — there's a point where spending more in one position stops helping.

#### Step-by-Step Plan
```
1. DATA COLLECTION
   - Pull squad valuations for top 5 leagues (Transfermarkt)
   - Group by position: GK, CB, FB, CM, AM, W, ST
   - Pull xG contribution per position (FBref): xG + xGA per 90 by position
   - Create dataset: position_investment → team_performance

2. RESPONSE CURVE FITTING
   - For each position: scatter plot of squad value vs team xG/xGA contribution
   - Fit Hill function (saturation curve): contribution = (investment^a) / (k^a + investment^a)
   - Shows diminishing returns at high investment levels
   - Validate curves against actual team data

3. MARGINAL RETURN ANALYSIS
   - Calculate marginal xG per £1M invested at different investment levels
   - Find saturation point: where does additional spending stop helping?
   - Key insight: underspending in one position hurts more than overspending in another

4. OPTIMIZATION
   - Objective: maximize total expected xG - xGA (goal difference proxy)
   - Constraint: total budget = £X (test with £50M, £100M, £200M)
   - Solver: scipy.optimize.minimize or PuLP linear programming
   - Output: optimal £ allocation per position

5. SCENARIO ANALYSIS
   - Run optimization for 3 budget levels: £50M, £100M, £200M
   - Show how allocation strategy changes with budget
   - Compare optimal allocation vs actual club spending (which clubs waste budget?)

6. VALIDATION
   - Back-test: teams who spent closer to "optimal" distribution — did they outperform?
   - Find clubs that were most efficient vs most wasteful in budget allocation

7. VISUALIZATION
   - Response curves per position (with saturation point annotated)
   - Optimal allocation bar chart for each budget level
   - Actual vs optimal spending heatmap for top 20 clubs

8. OUTPUT
   - Save: response_curves.pkl
   - Save: optimal_allocation.csv
   - Save: budget_optimization.html (interactive)
```

**Key Output:** Optimal transfer budget allocation chart + identification of which clubs are most wasteful.  
**LinkedIn Hook:** *"MMM tells you what worked. Budget optimization tells you what to do next. I built both for football transfers. Most clubs are spending in completely the wrong places."*

---

## Phase 4: Product & Platform
*Months 10–11 · Models 19–20*

---

### Model 19 — Search & Ranking Models

**Discipline:** Product DS  
**The Question:** *Can we build a learning-to-rank model that orders players by true impact — going beyond box scores — and does it agree or disagree with expert consensus rankings?*

#### Data Sources (choose one)
| Option | Source | What to pull | How |
|--------|--------|-------------|-----|
| ⭐ FBref + public rankings | `soccerdata` + Ballon d'Or voting data | Advanced stats + human expert rankings as training signal | `soccerdata` free |
| NBA Alternative | `nba_api` + ESPN rankings | Advanced stats + ESPN Power Rankings as label | `nba_api` |
| Combined | FBref stats + FiveThirtyEight Soccer Power Index | Model rankings vs algorithmic rankings vs expert | Both free |

#### Approach
Learning-to-rank using LightGBM's LambdaRank objective. Train on historical seasons where we have both stats and known rankings (awards voting, end-of-season lists). Evaluate with NDCG (Normalized Discounted Cumulative Gain). This is exactly how search engines rank results.

#### Step-by-Step Plan
```
1. DATA COLLECTION
   - Pull 5 seasons of FBref advanced stats (per-90)
   - Collect ranking labels: Ballon d'Or top-30 votes (public), PFA awards shortlists
   - For NBA: All-NBA team voting, All-Star selections as ranking proxy
   - Build dataset: player → features → ranking position

2. RANKING LABEL CREATION
   - Convert awards/votes to relevance scores:
     Ballon d'Or winner = 5, top-5 = 4, top-10 = 3, top-30 = 2, nominated = 1, not nominated = 0
   - Group by season (each season = one ranking query)

3. FEATURE ENGINEERING
   - Per-90 stats: xG, xA, progressive carries, pressures, tackles
   - Positional adjustment: normalize within position group
   - Availability: minutes played (players who play more get more credit)
   - Team context: team xG contribution share

4. LEARNING-TO-RANK MODEL
   - Use LightGBM with objective='lambdarank'
   - Query group = season (all players in same season are ranked together)
   - Eval metric: NDCG@10, NDCG@30
   - Train/test split: train on 2015–2021, test on 2022–2023

5. RANKING EVALUATION
   - NDCG score: how well does model ranking match human ranking?
   - Kendall's tau: rank correlation between model and expert consensus
   - Find: where does model most disagree with experts?

6. MODEL RANKINGS
   - Apply model to current season
   - Generate top-50 model rankings
   - Identify: who is overrated by experts? Who is underrated?

7. VISUALIZATION
   - Ranked list: model top-20 vs consensus top-20
   - Scatter: model rank vs expert rank (annotate outliers)
   - Feature importance for what drives model ranking

8. OUTPUT
   - Save: ranking_model.pkl
   - Save: player_rankings.csv (model rank, expert rank, delta)
   - Save: ranking_comparison.html (interactive)
```

**Key Output:** Top-50 model rankings vs expert consensus. The disagreements are the post content.  
**LinkedIn Hook:** *"Every feed and search engine runs on ranking models. Here's how they work — and here's what happens when I apply one to football. My model disagrees with the Ballon d'Or in interesting ways."*

---

### Model 20 — Network Effects Modeling

**Discipline:** Product DS  
**The Question:** *Do some players make everyone around them better — and can we quantify the network effect of adding a specific player to a lineup?*

#### Data Sources (choose one)
| Option | Source | What to pull | How |
|--------|--------|-------------|-----|
| ⭐ StatsBomb Open Data | `statsbombpy` | Passing networks, event data with x/y coordinates | `pip install statsbombpy` — free |
| NBA On/Off Data | `nba_api` | Lineup combinations, on/off splits per player | `nba_api.stats.endpoints.lineupadvanced` |
| FBref | `soccerdata` | Pass completion networks, team stats by lineup | `soccerdata` |

#### Approach
Two components: (1) Passing network graph analysis — who connects to whom, centrality scores, network density. (2) On/off analysis — how do teammates' stats change when a specific player is on/off the field. Combine into a "network multiplier" score per player.

#### Step-by-Step Plan
```
1. DATA COLLECTION (StatsBomb)
   - Load free match data via statsbombpy
   - Extract: all passes with passer and recipient
   - Build adjacency matrix: player_i passes to player_j (N times)
   - Load: match events with player IDs and timestamps

2. PASSING NETWORK CONSTRUCTION
   - Per match: build directed weighted graph (networkx)
   - Nodes = players, edges = passes, weight = pass count
   - Normalize by time on pitch together

3. GRAPH METRICS
   - Degree centrality: how many unique passing connections?
   - Betweenness centrality: how often does a player sit on the path between others?
   - PageRank: which players receive the most passes from highly-connected players?
   - Clustering coefficient: how tightly connected is each player's sub-network?

4. ON/OFF ANALYSIS (NBA version)
   - Pull lineup combinations from nba_api
   - For each player: calculate team offensive rating WITH vs WITHOUT them
   - Calculate: "network effect" = change in teammates' stats when player is on court
   - Find: players whose teammates shoot better when they're on court

5. NETWORK EFFECT SCORE
   - Combine: passing centrality + teammate performance lift
   - Normalize: control for overall team quality
   - Score each player: how much do they amplify their network?

6. KEY FINDING
   - Players with highest network effect vs highest individual stats — do they match?
   - Find: the "hidden connector" — low individual stats but high network effect
   - This is the sports equivalent of a platform user who creates value for others

7. VISUALIZATION
   - Passing network graph (interactive Plotly graph)
   - Bar chart: top 20 players by network effect score
   - Scatter: individual stats vs network effect score (find outliers)

8. OUTPUT
   - Save: passing_networks/ (per-match graphs)
   - Save: network_effect_scores.csv
   - Save: network_graph.html (interactive)
   - Save: individual_vs_network_scatter.png
```

**Key Output:** Passing network visualization + network effect score leaderboard.  
**LinkedIn Hook:** *"Network effects power the biggest tech platforms. I modeled them using passing networks. Some players make everyone around them dramatically better — you'd never guess who."*

---

## Appendix: Recommended Tech Stack

```
# Core
python >= 3.10
pandas >= 2.0
numpy
scikit-learn
xgboost
lightgbm
statsmodels

# Sports Data
nba_api
sportsipy
soccerdata
statsbombpy
understat
pytrends
praw

# Modelling
lifelines          # survival analysis
causalml           # uplift modeling
causalimpact       # incrementality
pymc-marketing     # MMM (Bayesian)
scipy              # optimization
PuLP               # linear programming
networkx           # graph analysis
FAISS-cpu          # similarity search

# NLP
transformers       # HuggingFace
bertopic
vaderSentiment

# Visualization
plotly
seaborn
matplotlib
streamlit          # for app deployment

# Utilities
geopy              # distance calculation
requests
tqdm
jupyter
```

---

## Appendix: Suggested File Structure

```
sports-ds-portfolio/
├── data/
│   ├── raw/          # original pulled data
│   ├── processed/    # cleaned, feature-engineered
│   └── external/     # static reference files
├── models/
│   ├── 01_segmentation/
│   ├── 02_funnel/
│   ├── 03_cohort/
│   ... (one folder per model)
├── notebooks/
│   ├── 01_segmentation_exploration.ipynb
│   ... (exploration notebooks)
├── src/
│   ├── data/         # data collection scripts
│   ├── features/     # feature engineering
│   ├── models/       # model training
│   └── viz/          # visualization helpers
├── outputs/
│   ├── figures/
│   ├── reports/
│   └── apps/         # Streamlit apps
├── README.md
└── requirements.txt
```

---

## Appendix: LinkedIn Post Calendar

| Week | Model | Post Type | Hook |
|------|-------|-----------|------|
| 1 | Segmentation | Process | "I'm building a sports analytics portfolio to learn 20 DS models. Starting with clustering. Here's why." |
| 2 | Segmentation | Results | "I clustered every NBA player. The results broke traditional positions." |
| 3 | Funnel | Hot take | "The fan acquisition funnel and a SaaS growth funnel are identical. Here's the proof." |
| 4 | Cohort | Results | "30 years of NBA draft classes as a cohort retention table. The 2003 class is an outlier." |
| 5 | Churn | Process | "Building a churn model — but the customers are NBA players." |
| 6 | Churn | Results | "SHAP values revealed the #1 signal of player decline. It's not age." |
| ... | ... | Alternate project posts with hot takes | ... |
| 22 | MMM | Hot take | "MMM is the most in-demand skill in marketing DS. Here's how I learned it in 4 weeks." |
| 40+ | Network Effects | Results | "Some players make everyone around them better. I built the model to prove it." |
```

---

*Generated: April 2026 | Scope: Marketing DS + Growth DS + Product DS | Theme: Sports Analytics*

---

## Appendix: Shared Data Ingestion Module

Rather than rewriting data collection code for each of 20 models, build this once in `src/data/` and import it everywhere. Claude Code should create this before starting Model 01.

### Structure

```
src/data/
├── __init__.py
├── nba.py          # all NBA data functions
├── soccer.py       # all soccer data functions
├── reddit.py       # Reddit/sentiment data
├── weather.py      # Open-Meteo weather
├── cache.py        # caching utilities
└── validate.py     # data quality checks
```

### cache.py — Use This Everywhere

Scrapers like `sportsipy` and `soccerdata` break when sites update and are slow to call repeatedly. Cache everything to disk.

```python
# src/data/cache.py
import os
import pickle
import hashlib
import requests_cache
from functools import wraps
from pathlib import Path

CACHE_DIR = Path("data/raw/.cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# HTTP-level caching (for requests-based sources)
requests_cache.install_cache(
    str(CACHE_DIR / "http_cache"),
    expire_after=86400 * 7  # 7-day cache
)

def disk_cache(func):
    """Decorator: cache function output to disk by args hash."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        key = hashlib.md5(str((func.__name__, args, kwargs)).encode()).hexdigest()
        cache_path = CACHE_DIR / f"{func.__name__}_{key}.pkl"
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        result = func(*args, **kwargs)
        with open(cache_path, "wb") as f:
            pickle.dump(result, f)
        return result
    return wrapper
```

### nba.py — NBA Data Functions

```python
# src/data/nba.py
import time
import pandas as pd
from nba_api.stats.endpoints import (
    leaguedashplayerstats, playbyplayv3, leaguedashlineups
)
from .cache import disk_cache

@disk_cache
def get_player_stats(season: str, per_mode: str = "PerGame") -> pd.DataFrame:
    """Pull NBA player stats for a season. season format: '2022-23'"""
    time.sleep(0.6)  # nba_api rate limit — always add delay
    stats = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        per_mode_simple=per_mode
    )
    return stats.get_data_frames()[0]

@disk_cache
def get_multiple_seasons(seasons: list, per_mode: str = "PerGame") -> pd.DataFrame:
    """Pull and stack multiple seasons of player stats."""
    dfs = []
    for season in seasons:
        df = get_player_stats(season, per_mode)
        df["SEASON"] = season
        dfs.append(df)
        time.sleep(1)
    return pd.concat(dfs, ignore_index=True)

@disk_cache
def get_play_by_play(game_id: str) -> pd.DataFrame:
    """Pull play-by-play for a single game."""
    time.sleep(0.6)
    pbp = playbyplayv3.PlayByPlayV3(game_id=game_id)
    return pbp.get_data_frames()[0]
```

### soccer.py — Soccer Data Functions

```python
# src/data/soccer.py
import pandas as pd
import soccerdata as sd
from statsbombpy import sb
from .cache import disk_cache

@disk_cache
def get_fbref_stats(league: str, season: str) -> pd.DataFrame:
    """
    Pull FBref player stats.
    league options: "ENG-Premier League", "ESP-La Liga", "GER-Bundesliga"
    season format: "2022-23"
    """
    fbref = sd.FBref(leagues=league, seasons=season)
    return fbref.read_player_season_stats(stat_type="standard")

@disk_cache
def get_statsbomb_matches(competition_id: int, season_id: int) -> pd.DataFrame:
    """Pull StatsBomb free match list for a competition/season."""
    return sb.matches(competition_id=competition_id, season_id=season_id)

@disk_cache
def get_statsbomb_events(match_id: int) -> pd.DataFrame:
    """Pull all events for a StatsBomb match."""
    return sb.events(match_id=match_id)

@disk_cache
def get_understat_team(team_name: str, season: int) -> dict:
    """Pull Understat xG data for a team-season."""
    import asyncio
    from understat import Understat
    import aiohttp

    async def fetch():
        async with aiohttp.ClientSession() as session:
            u = Understat(session)
            return await u.get_team_results(team_name, season)

    return asyncio.run(fetch())
```

### validate.py — Data Quality Checks

```python
# src/data/validate.py
import pandas as pd

def check_dataframe(df: pd.DataFrame, name: str, required_cols: list = None,
                    min_rows: int = 10, max_null_pct: float = 0.2) -> bool:
    """
    Basic data quality gate. Call this after every data pull.
    Returns True if data passes, raises AssertionError with details if not.
    """
    assert df is not None, f"{name}: DataFrame is None"
    assert len(df) >= min_rows, f"{name}: Only {len(df)} rows, expected >= {min_rows}"

    if required_cols:
        missing = [c for c in required_cols if c not in df.columns]
        assert not missing, f"{name}: Missing columns: {missing}"

    null_pct = df.isnull().mean()
    high_null = null_pct[null_pct > max_null_pct]
    if not high_null.empty:
        print(f"WARNING {name}: High null % in columns: {high_null.to_dict()}")

    print(f"✓ {name}: {len(df)} rows, {len(df.columns)} cols — passed quality check")
    return True

def check_date_range(df: pd.DataFrame, date_col: str, expected_min, expected_max):
    """Verify date range is as expected."""
    actual_min = df[date_col].min()
    actual_max = df[date_col].max()
    assert actual_min >= expected_min, f"Date range starts too early: {actual_min}"
    assert actual_max <= expected_max, f"Date range ends too late: {actual_max}"
    print(f"✓ Date range: {actual_min} to {actual_max}")
```

### Rate Limiting Reference

| Source | Rate Limit | Recommended Sleep |
|--------|-----------|------------------|
| `nba_api` | ~30 req/min | `time.sleep(0.6)` between calls |
| `sportsipy` | Scraper — polite | `time.sleep(2)` between calls |
| `soccerdata` (FBref) | Scraper — polite | `time.sleep(3)` between calls |
| `understat` | Async scraper | Built-in limits fine |
| `Football-Data.org` | 10 req/min free | `time.sleep(6)` |
| `PRAW (Reddit)` | 60 req/min | Handled by library |
| `Open-Meteo` | No limit stated | No sleep needed |
| `pytrends` | Aggressive limits | `time.sleep(5)` + retry logic |

### Handling Source Breakage

Scrapers break. When they do:
1. Check GitHub issues for the library first
2. Fall back to cached data (`CACHE_DIR`)
3. Fall back to Kaggle equivalent dataset
4. Document the fallback in the model README

```python
# Pattern: graceful fallback
try:
    df = get_player_stats("2022-23")
except Exception as e:
    print(f"Live data failed: {e}. Loading from cache/fallback.")
    df = pd.read_csv("data/raw/fallback/nba_2022_23.csv")
```

---

## Appendix: Model Evaluation Card Template

Every model should ship with a completed evaluation card saved as `outputs/evaluation_card.md`. This is what makes the portfolio look professional to hiring managers. Claude Code should fill this in at the end of each model.

```markdown
# Model Evaluation Card — [Model Name]

## Model Summary
- **Type:** [Classification / Regression / Clustering / Ranking / etc.]
- **Question:** [One sentence — the business question this answers]
- **Sports context:** [One sentence — the sports version of the problem]
- **Date completed:** [YYYY-MM]

## Data
- **Source:** [e.g., nba_api, FBref via soccerdata]
- **Time period:** [e.g., 2003–2023]
- **Records:** [e.g., 8,420 player-seasons]
- **Features:** [N features, list key ones]
- **Target:** [What are we predicting/optimizing?]
- **Known limitations:** [Missing data, scraping gaps, proxy variables used]

## Model Performance
| Metric | Value | Baseline | Notes |
|--------|-------|----------|-------|
| [Primary metric] | [X] | [Y] | [Context] |
| [Secondary metric] | [X] | [Y] | [Context] |

*Baseline = simplest possible model (e.g., predict mean, random assignment, prior season value)*

## Key Findings
1. [Most important finding]
2. [Second finding]
3. [Surprising / counterintuitive finding — this is the LinkedIn post]

## Marketing / Growth / Product Parallel
> [1–2 sentences explicitly connecting the sports finding to a real marketing/growth/product insight]
> Example: "This is directly analogous to how growth teams use LTV:CAC to compare acquisition channels — the same math, different domain."

## Known Weaknesses
- [What could make this model wrong?]
- [What data would improve it?]
- [What assumption is most fragile?]

## Reproducibility
- `src/data/[script].py` — data collection
- `notebooks/[model]_exploration.ipynb` — EDA
- `src/models/[model].py` — training pipeline
- `outputs/[model]/` — saved model + results
- Run: `python src/models/[model].py` to reproduce
```

---

## Appendix: GitHub README Template Per Project

Each model folder should have a README that makes sense to a hiring manager who clicks through from LinkedIn. Claude Code should generate this at the end of each model.

```markdown
# [Model Name] — Sports Analytics × [Discipline]

> **[One-line hook]**  
> e.g., "Can we predict NBA player churn using the same model a SaaS company would use for subscriber retention?"

## The Question
[2–3 sentences on the business question and why it matters in both sports and marketing/growth/product contexts.]

## Approach
[3–4 sentences on the method. Non-technical enough for a hiring manager, specific enough for a DS. Include: what model, why this model, what the key modelling challenge was.]

## Key Finding
**[The one sentence that would go in a LinkedIn post]**

[2–3 sentences of supporting context.]

## Results
| Metric | Value |
|--------|-------|
| [Primary metric] | [X] |
| [vs. baseline] | [Y] |

![Key visualization](outputs/figures/main_chart.png)

## Marketing / Growth Parallel
[Explicitly state how this connects to real marketing DS, growth DS, or product DS work. This is what gets you hired.]

## Data Sources
- [Source 1]: [What it provided, how to access]
- [Source 2]: [What it provided, how to access]

## How to Run
\```bash
pip install -r requirements.txt
python src/data/collect.py      # Pull data (~X minutes)
python src/models/train.py      # Train model
python src/viz/plot.py          # Generate outputs
\```

## Files
\```
├── src/
│   ├── data/collect.py         # Data ingestion
│   ├── features/engineer.py    # Feature engineering
│   ├── models/train.py         # Model training
│   └── viz/plot.py             # Visualization
├── outputs/
│   ├── figures/                # Charts
│   ├── model.pkl               # Saved model
│   └── evaluation_card.md      # Full model card
└── notebook_exploration.ipynb  # EDA
\```

## Limitations
[2–3 honest bullets about what this model doesn't do or where it could be wrong.]

---
*Part of a 20-model sports analytics portfolio. [Link to main repo]*
```

---

## Appendix: LinkedIn Post Template

Use this structure for every project post. Fill in the brackets. Aim for 150–250 words total — long enough to have substance, short enough to be read.

### Structure

```
[HOOK — 1 sentence, no context needed, slightly provocative]

[CONTEXT — 2–3 sentences. What did you build, what sport, what's the 
marketing/growth/product parallel]

[INSIGHT — 2–3 sentences. The actual finding. Be specific with numbers 
where possible. This is the part people share.]

[CROSSOVER — 1–2 sentences. Explicitly connect the sports finding to 
a real business/marketing/product insight. This is what gets you hired.]

[TAKEAWAY — 1 sentence. What should someone do with this?]

[QUESTION — 1 sentence. Invites engagement. Don't ask "what do you think?" 
— ask something specific.]

[Optional: link to GitHub / Streamlit app]
```

### Example (Model 04 — Churn)

```
The same model SaaS companies use to predict subscriber churn works on 
NBA players.

I built a churn classifier on 20 years of player data — labelling a 
player as "churned" if their minutes dropped >40% the following season. 
XGBoost + SHAP to explain why each player was flagged.

The #1 predictor wasn't age. It was usage rate trend — specifically, 
whether USG% had been declining for 2+ consecutive seasons. Players whose 
coaches were already moving away from them churned at 3x the rate of 
age-matched peers. The model flagged [Player X] two seasons before he 
actually fell out of rotation.

This maps directly to marketing churn: declining engagement before 
cancellation is a much stronger signal than demographic features. The 
model architecture is identical — only the domain changes.

If you're building churn models for subscribers, check whether your 
engagement trend features are outperforming your demographic features. 
They probably should be.

What's the most counterintuitive churn signal you've found in your work?

[GitHub link]
```

### Post Type Rotation

Across 40 posts, rotate through these types to avoid repetition:

| Type | Frequency | Format | Purpose |
|------|-----------|--------|---------|
| **Project reveal** | Every model | Hook → build → finding → crossover | Show the work |
| **Behind the build** | Every model | Process, mistakes, what surprised you | Show how you think |
| **Hot take** | 1–2x/month | 1 punchy opinion, 3 supporting points | Drive engagement |
| **Crossover insight** | 1x/month | Sports concept → marketing concept | Get hired |
| **Learning in public** | 1x/month | What you got wrong, what you'd do differently | Build trust |

### Crossover Post Template

This is the highest-value post type — the one that directly signals to hiring managers you can do the real job.

```
[SPORTS FINDING in one sentence]

Most people see this as a sports analytics result. 

But it's actually identical to [MARKETING/GROWTH/PRODUCT PROBLEM]. 
Here's why:

[PARALLEL 1]: In sports, [X]. In marketing, [Y]. Same math.
[PARALLEL 2]: In sports, [A]. In growth, [B]. Same question.
[PARALLEL 3]: In sports, [P]. In product, [Q]. Same model.

The domain is different. The framework is the same.

If you work in [role] and haven't looked at sports analytics for 
methodological inspiration — you're missing one of the best free 
sandboxes for learning measurement.

Which model from sports analytics do you think has the clearest 
marketing parallel?
```

### Streamlit App Launch Post (Model 09)

Model 09 (Recommendation) produces a live app — treat this post differently. It's the highest-engagement post in the whole portfolio because people can interact with it.

```
I built a "players like this" recommender for [sport].

Type any player name → get 5 statistically similar players from 
across leagues and eras.

[Notable finding from the app — the most surprising comparison]

Under the hood: cosine similarity on per-90 FBref stats, FAISS 
nearest-neighbour search, Streamlit frontend.

The same architecture powers "customers like this one" in marketing 
personalisation and "content similar to this" in recommendation feeds. 
The only thing that changes is the feature set.

[Link to live app]

Who's the most surprising match you find?
```

---

## Appendix: Compute & Runtime Reference

Some models are significantly more expensive than others. Use this to plan ahead and avoid surprises.

| Model | Est. Runtime | RAM needed | Notes |
|-------|-------------|-----------|-------|
| 01 Segmentation | 5–15 min | 4GB | UMAP is slow on large N — subsample if needed |
| 02 Funnel | 10–30 min | 2GB | Mostly Reddit API calls — network bound |
| 03 Cohort | 5 min | 2GB | Fast — pure pandas |
| 04 Churn | 10–20 min | 4GB | XGBoost fast; SHAP slower on large datasets |
| 05 LTV+CAC | 15–30 min | 4GB | Salary data scraping is slow |
| 06 Propensity | 10–20 min | 4GB | LightGBM fast |
| 07 Survival | 10–20 min | 2GB | lifelines is efficient |
| 08 Forecasting | 20–40 min | 4GB | Prophet is slow per series × N teams |
| 09 Recommendation | 10–20 min | 8GB | FAISS index build; Streamlit deploy adds time |
| 10 Sentiment | **1–3 hours** | **8–16GB** | DistilBERT inference on 50k posts is slow — use GPU or VADER first |
| 11 Activation | 10–20 min | 4GB | Standard classification |
| 12 Engagement | 15–30 min | 4GB | pytrends can be flaky — cache aggressively |
| 13 Attribution | 30–60 min | 8GB | Shapley computation grows exponentially with chain length |
| 14 MMM | **2–6 hours** | **8–16GB** | PyMC MCMC is slow — start with 500 samples to test, then 2000 for final |
| 15 A/B + Causal | 30–60 min | 4GB | CausalImpact fast; PyMC slower |
| 16 Incrementality | 30–60 min | 4GB | Multiple CausalImpact runs |
| 17 Uplift | 30–60 min | 8GB | causalml models moderate speed |
| 18 Budget Opt | 10–30 min | 4GB | Optimization fast once curves fitted |
| 19 Search & Rank | 20–40 min | 8GB | LambdaRank training moderate |
| 20 Network Effects | 30–60 min | 8GB | networkx graph operations on large datasets |

### When to Use Google Colab (Free)

Use Colab free tier when:
- Model 10 (Sentiment) — Colab's free GPU dramatically speeds up transformer inference
- Model 14 (MMM) — RAM-heavy; Colab Pro gives 25GB RAM
- Model 20 (Network Effects) — large graph operations benefit from more RAM

```python
# Check if running in Colab and adjust paths
import os
IN_COLAB = 'COLAB_GPU' in os.environ

if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    BASE_PATH = '/content/drive/MyDrive/sports-ds-portfolio'
else:
    BASE_PATH = '.'
```

### Streamlit Deployment (Model 09)

```
# requirements.txt for Streamlit Cloud deployment
# Keep lean — only what the app actually needs
pandas==2.0.3
scikit-learn==1.3.0
plotly==5.17.0
streamlit==1.28.0
faiss-cpu==1.7.4
soccerdata==0.4.0    # or pre-cache data and remove this

# .streamlit/config.toml
[server]
maxUploadSize = 200

[theme]
primaryColor = "#00e5ff"
backgroundColor = "#0a0a0f"
secondaryBackgroundColor = "#12121a"
textColor = "#e8e8f0"
```

**Important for Streamlit:** pre-compute and save the similarity matrix as a `.pkl` file in the repo rather than recomputing on load. The app should load pre-computed results, not run the model live.

```python
# In the app — load pre-computed, don't recompute
@st.cache_resource
def load_similarity_index():
    return pickle.load(open("data/similarity_matrix.pkl", "rb"))
```

---

*Generated: April 2026 | Scope: Marketing DS + Growth DS + Product DS | Theme: Sports Analytics*
