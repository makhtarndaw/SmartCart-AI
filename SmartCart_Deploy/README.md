# SmartCart AI — Marketing intelligence platform

SmartCart AI is a SaaS-style marketing intelligence platform for e-commerce brands. A company **uploads its own
event log** (product views, add-to-cart, removals, purchases); SmartCart cleans it, rebuilds sessions, scores every
visit with its **purchase-intent model** and turns the results into KPIs, customer segments, trends, cart-recovery
audiences and quantified campaigns.

The purchase-intent model (XGBoost, Optuna-tuned) was built in our Master 2 project on 20.7M events of a cosmetics
e-shop (REES46). It is applied as-is to the company's data and can be **fine-tuned** on it from the Data Hub.
Until a company connects its data, the platform runs on a demo workspace (5% sample of REES46 customers).

## How a company uses it — a guided 5-step journey

| Step | Screen | Question it answers |
|---|---|---|
| 1 | **Import data** | Upload a file; columns and event labels are recognised automatically |
| 2 | **Diagnosis** | What is happening in my business? (plain-English findings, forecast, visitor intent) |
| 3 | **Customers** | Who are my customers, who to keep, who to win back? (8 groups + downloadable lists) |
| 4 | **Products** | What sells, what is rising or falling, what is viewed but not bought, what to bundle? |
| 5 | **Action plan** | Which campaigns to run, for whom, when, and what they should bring (with export) |

The **Expert space** keeps the technical views: trends, cart analysis, visit behaviour, AI scoring of a visitor,
intent groups and model fine-tuning, model performance, explainability (SHAP), causal evidence (Hillstrom),
A/B test planner and methodology.

## Input format

One row per event. Column names and event labels are free — they are mapped in the Data Hub.

| Field | Required |
|---|---|
| Event timestamp, event type, customer ID, product ID, unit price | yes |
| Brand, category, product name | optional |

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Rebuild the artifacts (optional)

The app ships the trained model, the model-card tables and the demo dataset in `artifacts/` (≈ 14 MB). To regenerate them from the raw data
(REES46 Parquet files in `../SmartCart_Parquet/`, ~20M events, ~2-3 min each, 4 GB RAM):

```bash
pip install duckdb scikit-learn
python build.py            # re-trains the SmartCart model + model-card artifacts + Hillstrom
```

`build.py` reproduces `Marketing_Final.ipynb`: deduplication, 30-minute sessionization, eligibility funnel, features,
chronological split, optimized XGBoost (best Optuna parameters), threshold, deciles, SHAP, segments, A/B baseline and
the Hillstrom analysis.

## Deploy on Streamlit Community Cloud (free)

1. Create a GitHub repository and push **this folder** (`app.py`, `core/`, `views/`, `artifacts/` incl. `artifacts/demo/`, `.streamlit/`,
   `requirements.txt`). The raw data is not needed.
2. Go to <https://share.streamlit.io> → *Create app* → pick the repository, branch `main`, main file `app.py`.
3. Deploy — you get a public URL to share.

## Team

Rania El Mazouzi · Anaïs Hamitouche · Papa Makhtar Ndaw · Meryem Sabik
