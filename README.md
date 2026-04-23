# Sports Analytics Data Science Portfolio

**20 end-to-end models** across Marketing DS, Growth DS, and Product DS.  

---

## Portfolio Map

| # | Model | Discipline | Method | Status |
|---|-------|-----------|--------|--------|
| **Phase 1: Customer Fundamentals** |
| 01 | [NBA Player Segmentation](model_01_segmentation/) | Marketing DS | K-Means + UMAP | ✅ Complete |
| 02 | [Fan Funnel Analysis](model_02_funnel_analysis/) | Growth DS | Funnel / Sankey | 🔧 Scaffolded |
| 03 | [NBA Draft Cohort Analysis](model_03_cohort_analysis/) | Growth DS | Cohort Retention | 🔧 Scaffolded |
| 04 | [Player Churn Model](model_04_churn_model/) | Marketing DS | XGBoost + SHAP | 🔧 Scaffolded |
| 05 | [LTV + CAC Modeling](model_05_ltv_cac/) | Growth DS | Regression + LTV | 🔧 Scaffolded |
| 06 | [Title Propensity Model](model_06_propensity_model/) | Marketing DS | LightGBM + Calibration | 🔧 Scaffolded |
| **Phase 2: Prediction & Discovery** |
| 07 | [Player Career Survival](model_07_survival_analysis/) | Marketing DS | Kaplan-Meier + Cox | 🔧 Scaffolded |
| 08 | [EPL Standings Forecasting](model_08_forecasting/) | Growth DS | ARIMA + Prophet | 🔧 Scaffolded |
| 09 | [Player Recommender System](model_09_recommender/) | Product DS | Cosine Similarity + Streamlit | 🔧 Scaffolded |
| 10 | [Transfer Sentiment Analysis](model_10_sentiment_analysis/) | Marketing DS | VADER + BERTopic | 🔧 Scaffolded |
| 11 | [Rookie Activation Model](model_11_activation_analysis/) | Product DS | XGBoost + SHAP | 🔧 Scaffolded |
| 12 | [Fan Engagement Model](model_12_engagement_model/) | Product DS | XGBoost + SHAP | 🔧 Scaffolded |
| **Phase 3: Measurement & Causality** |
| 13 | [Goal Attribution Modeling](model_13_attribution_model/) | Marketing DS | Last-touch / Linear / Shapley | 🔧 Scaffolded |
| 14 | [Marketing Mix Modeling](model_14_mmm/) | Marketing DS | PyMC-Marketing (Bayesian MMM) | 🔧 Scaffolded |
| 15 | [A/B Testing + Causal Inference](model_15_causal_inference/) | Growth DS | CausalImpact + DiD | 🔧 Scaffolded |
| 16 | [Incrementality Testing](model_16_incrementality/) | Growth DS | Synthetic Control + CausalImpact | 🔧 Scaffolded |
| 17 | [Uplift Modeling](model_17_uplift_model/) | Growth DS | T-Learner + Qini Curve | 🔧 Scaffolded |
| 18 | [Budget Allocation & Optimization](model_18_budget_optimization/) | Growth DS | Hill Curves + scipy/PuLP | 🔧 Scaffolded |
| **Phase 4: Product & Platform** |
| 19 | [Search & Ranking Model](model_19_search_ranking/) | Product DS | LightGBM LambdaRank | 🔧 Scaffolded |
| 20 | [Network Effects Modeling](model_20_network_effects/) | Product DS | networkx + On/Off Analysis | 🔧 Scaffolded |

---

## Skills Demonstrated

| Category | Tools / Methods |
|----------|----------------|
| **Languages** | Python 3.11 |
| **ML / Modeling** | scikit-learn, XGBoost, statsmodels, lifelines, PyMC |
| **NLP** | VADER, BERTopic, Hugging Face |
| **Causal Inference** | DiD, PSM, Synthetic Control, Uplift |
| **Visualization** | Plotly, Matplotlib, Seaborn, Streamlit |
| **Data** | nba_api, PRAW, pytrends, Football-Data.org, StatsBomb |

---

## Environment Setup

All models share a single conda environment:

```bash
conda env create -f model_01_segmentation/environment.yml
conda activate ds_portfolio
```

---

## Repo Structure

```
MarSci/
├── README.md                          ← You are here
└── model_01_segmentation/
    ├── README.md
    ├── environment.yml
    ├── requirements.txt
    ├── model_01_segmentation.ipynb

    └── outputs/
        ├── clustered_players.csv
        ├── umap_plot.html
        ├── cluster_profiles.png
        └── elbow_silhouette.png
```
