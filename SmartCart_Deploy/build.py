"""
SmartCart AI - artifact builder
================================
Runs the full pipeline of `Marketing_Final.ipynb` locally on the REES46 Parquet files
and exports small artifacts (model + aggregated tables) used by the Streamlit app.

Run once:   python build.py
Input:      ../SmartCart_Parquet/*.parquet   (or set SMARTCART_PARQUET=/path/to/folder)
Output:     ./artifacts/
"""
import json
import os
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, precision_recall_curve
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent
PARQUET = Path(os.environ.get("SMARTCART_PARQUET", ROOT.parent / "SmartCart_Parquet"))
OUT = ROOT / "artifacts"
CACHE = ROOT / ".cache"
OUT.mkdir(exist_ok=True)
CACHE.mkdir(exist_ok=True)

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def save(df, name):
    df.to_csv(OUT / f"{name}.csv", index=False)


KPI = {}

# --------------------------------------------------------------------------------------
# 1. Data import & cleaning
# --------------------------------------------------------------------------------------
# In-memory database (no big file on disk); spill to a small, capped temp folder if needed
con = duckdb.connect()
con.execute(f"SET memory_limit='{os.environ.get('SMARTCART_MEM', '4GB')}'")
con.execute(f"SET temp_directory='{CACHE / 'tmp'}'")
con.execute("SET max_temp_directory_size='2GB'")
con.execute("SET preserve_insertion_order=false")
con.execute("SET threads=2")

log("Loading Parquet files and removing duplicates")
SRC = f"read_parquet('{PARQUET}/*.parquet', union_by_name=true)"
KPI["raw_events"] = int(con.execute(f"SELECT COUNT(*) FROM {SRC}").fetchone()[0])
# Only the columns used by the analysis (brand / category_code are product attributes,
# constant for a product_id, so DISTINCT on this subset = DISTINCT * in the notebook)
con.execute(f"""
CREATE OR REPLACE TABLE events AS
SELECT DISTINCT event_time, event_type, product_id, category_id, price, user_id, user_session
FROM {SRC}
WHERE user_id IS NOT NULL AND event_time IS NOT NULL AND event_type IS NOT NULL
""")
KPI["clean_events"] = int(con.execute("SELECT COUNT(*) FROM events").fetchone()[0])
save(con.execute("""SELECT event_type, COUNT(*) AS events FROM events
                    GROUP BY event_type ORDER BY events DESC""").df(), "event_types")
save(con.execute("""SELECT strftime(event_time, '%Y-%m') AS month, event_type, COUNT(*) AS events
                    FROM events GROUP BY 1, 2 ORDER BY 1, 2""").df(), "events_by_month")

# Diagnostic of the platform session IDs (values from the notebook run - this query needs
# more memory than a laptop has, it is not required to rebuild the sessions)
KPI["original_ids"] = {"events_without_session": 4085, "sessions_over_two_months": 14230,
                       "sessions_with_several_users": 272, "sessions_with_30min_break": 184471}
# user_session is not used anymore: drop it to save memory
con.execute("CREATE OR REPLACE TABLE events AS SELECT * EXCLUDE (user_session) FROM events")

# --------------------------------------------------------------------------------------
# 2. Session reconstruction (30-minute rule)
# --------------------------------------------------------------------------------------
log("Rebuilding sessions with the 30-minute inactivity rule")
order_key = "event_time, type_order, product_id, price"
con.execute(f"""
CREATE OR REPLACE TABLE session_events AS
WITH ordered AS (
    SELECT *, CASE event_type WHEN 'view' THEN 1 WHEN 'cart' THEN 2
                              WHEN 'remove_from_cart' THEN 3 WHEN 'purchase' THEN 4 END AS type_order
    FROM events),
flags AS (
    SELECT *,
           CASE WHEN LAG(event_time) OVER (PARTITION BY user_id ORDER BY {order_key}) IS NULL
                  OR event_time - LAG(event_time) OVER (PARTITION BY user_id ORDER BY {order_key})
                     > INTERVAL '30 minutes'
                THEN 1 ELSE 0 END AS new_session
    FROM ordered),
numbered AS (
    SELECT *, SUM(new_session) OVER (PARTITION BY user_id ORDER BY {order_key}
                                     ROWS UNBOUNDED PRECEDING) AS session_number
    FROM flags)
SELECT event_time, event_type, product_id, category_id, price, user_id,
       CAST(user_id AS VARCHAR) || '_' || CAST(session_number AS VARCHAR) AS session_id,
       ROW_NUMBER() OVER (PARTITION BY user_id, session_number ORDER BY {order_key}) AS event_position
FROM numbered
""")
# Keep only purchases (needed for the A/B baseline) and free the big table
con.execute("CREATE OR REPLACE TABLE purchases AS SELECT user_id, event_time FROM events WHERE event_type = 'purchase'")
con.execute("DROP TABLE events")
con.execute("CHECKPOINT")
con.execute("""
CREATE OR REPLACE TABLE sessions AS
SELECT session_id, ANY_VALUE(user_id) AS user_id, COUNT(*) AS n_events,
       MIN(event_time) AS session_start, MAX(event_time) AS session_end,
       MAX(event_time) FILTER (WHERE event_position = 3) AS prediction_time,
       MAX(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) AS has_purchase,
       MAX(CASE WHEN event_type = 'purchase' AND event_position <= 3 THEN 1 ELSE 0 END) AS early_purchase,
       MAX(CASE WHEN event_type = 'purchase' AND event_position > 3 THEN 1 ELSE 0 END) AS future_purchase
FROM session_events GROUP BY session_id
""")
s = con.execute("""
SELECT COUNT(*) AS sessions, MEDIAN(n_events) AS median_events, AVG(has_purchase) AS purchase_rate,
       COUNT(*) FILTER (WHERE strftime(session_start, '%Y-%m') <> strftime(session_end, '%Y-%m')) AS cross_month
FROM sessions""").df().iloc[0]
KPI["sessions"] = {"total": int(s.sessions), "median_events": float(s.median_events),
                   "purchase_rate": float(s.purchase_rate), "cross_month": int(s.cross_month)}
save(con.execute("""SELECT LEAST(n_events, 20) AS n_events, COUNT(*) AS sessions,
                           AVG(has_purchase) AS purchase_rate
                    FROM sessions GROUP BY 1 ORDER BY 1""").df(), "session_length")

# --------------------------------------------------------------------------------------
# 3. Prediction dataset & funnel
# --------------------------------------------------------------------------------------
log("Building the eligibility funnel")
con.execute("""
CREATE OR REPLACE TABLE session_flags AS
SELECT s.*, s.session_end <= (SELECT MAX(session_end) FROM sessions) - INTERVAL '30 minutes' AS complete
FROM sessions s""")
funnel = con.execute("""
SELECT 'All reconstructed sessions' AS step, COUNT(*) AS sessions, SUM(has_purchase) AS purchasing_sessions FROM session_flags
UNION ALL SELECT 'At least 3 events', COUNT(*), SUM(has_purchase) FROM session_flags WHERE n_events >= 3
UNION ALL SELECT 'No purchase in first 3 events', COUNT(*), SUM(has_purchase)
          FROM session_flags WHERE n_events >= 3 AND early_purchase = 0
UNION ALL SELECT 'Complete sessions (final dataset)', COUNT(*), SUM(has_purchase)
          FROM session_flags WHERE n_events >= 3 AND early_purchase = 0 AND complete
""").df()
funnel["share_of_purchasing_sessions"] = funnel.purchasing_sessions / funnel.purchasing_sessions[0]
save(funnel, "funnel")
con.execute("""CREATE OR REPLACE TABLE eligible AS SELECT * FROM session_flags
               WHERE n_events >= 3 AND early_purchase = 0 AND complete""")

# --------------------------------------------------------------------------------------
# 4. Feature engineering
# --------------------------------------------------------------------------------------
log("Computing features from the first three events")
con.execute("""
CREATE OR REPLACE TABLE model_table AS
WITH f AS (
    SELECT session_id,
           MAX(CASE WHEN event_position = 1 THEN event_type END) AS action_1,
           MAX(CASE WHEN event_position = 2 THEN event_type END) AS action_2,
           MAX(CASE WHEN event_position = 3 THEN event_type END) AS action_3,
           COUNT(*) FILTER (WHERE event_type = 'cart') AS n_cart,
           COUNT(*) FILTER (WHERE event_type = 'remove_from_cart') AS n_remove,
           AVG(price) AS avg_price,
           COUNT(DISTINCT product_id) AS distinct_products,
           COUNT(DISTINCT category_id) AS distinct_categories,
           DATE_DIFF('second', MAX(event_time) FILTER (WHERE event_position = 1),
                               MAX(event_time) FILTER (WHERE event_position = 2)) AS gap_1_2,
           DATE_DIFF('second', MAX(event_time) FILTER (WHERE event_position = 2),
                               MAX(event_time) FILTER (WHERE event_position = 3)) AS gap_2_3,
           DATE_DIFF('second', MIN(event_time), MAX(event_time)) AS session_duration_3
    FROM session_events WHERE event_position <= 3 GROUP BY session_id)
SELECT e.session_id, e.session_start, e.prediction_time, e.session_end,
       f.action_1, f.action_2, f.action_3,
       f.action_1 || ' > ' || f.action_2 || ' > ' || f.action_3 AS sequence,
       f.n_cart, f.n_remove, f.avg_price, f.distinct_products, f.distinct_categories,
       f.gap_1_2, f.gap_2_3, f.session_duration_3, e.future_purchase
FROM eligible e JOIN f ON f.session_id = e.session_id
""")
con.execute("DROP TABLE session_flags")
data = con.execute("SELECT * FROM model_table ORDER BY prediction_time, session_id").df()
con.execute("DROP TABLE model_table")
con.execute("DROP TABLE session_events")  # not needed anymore
con.execute("CHECKPOINT")
for c in ["session_start", "prediction_time", "session_end"]:
    data[c] = pd.to_datetime(data[c])
KPI["dataset"] = {"sessions": len(data), "buyers": int(data.future_purchase.sum()),
                  "purchase_rate": float(data.future_purchase.mean())}
log(f"Dataset: {data.shape}, purchase rate {data.future_purchase.mean():.4f}")

# --------------------------------------------------------------------------------------
# 5. Customer behaviour
# --------------------------------------------------------------------------------------
log("Behaviour analysis")


def conversion(col):
    t = data.groupby(col, observed=True)["future_purchase"].agg(sessions="size", purchase_rate="mean")
    return t.reset_index()


seq = conversion("sequence").sort_values("purchase_rate", ascending=False)
save(seq, "sequences")  # all 27 sequences (app filters by min size)

data["has_cart"] = (data.n_cart > 0).astype(int)
save(conversion("has_cart"), "by_cart")
save(conversion("n_remove"), "by_remove")
save(conversion("n_cart"), "by_ncart")
for i in (1, 2, 3):
    save(conversion(f"action_{i}"), f"by_action_{i}")

table = pd.crosstab(data.has_cart, data.future_purchase)
chi2, p_value, _, _ = stats.chi2_contingency(table)
KPI["cart_test"] = {"odds_ratio": float((table.loc[1, 1] * table.loc[0, 0]) / (table.loc[1, 0] * table.loc[0, 1])),
                    "chi2": float(chi2), "p_value": float(p_value)}

bins = [-1, 30, 60, 120, 300, 600, 1800, np.inf]
labels = ["<30s", "30s-1min", "1-2min", "2-5min", "5-10min", "10-30min", ">30min"]
data["duration_group"] = pd.cut(data.session_duration_3, bins=bins, labels=labels)
dur = (data.groupby(["duration_group", "has_cart"], observed=True)["future_purchase"]
       .agg(sessions="size", purchase_rate="mean").reset_index())
dur["duration_group"] = dur.duration_group.astype(str)
save(dur, "duration")

price_bins = [-0.01, 2, 4, 6, 10, 20, 50, np.inf]
price_labels = ["<$2", "$2-4", "$4-6", "$6-10", "$10-20", "$20-50", ">$50"]
data["price_group"] = pd.cut(data.avg_price, bins=price_bins, labels=price_labels)
pr = (data.groupby(["price_group", "has_cart"], observed=True)["future_purchase"]
      .agg(sessions="size", purchase_rate="mean").reset_index())
pr["price_group"] = pr.price_group.astype(str)
save(pr, "price")

# Sankey: transitions action_1 -> action_2 -> action_3 -> outcome
links = []
for a, b in [("action_1", "action_2"), ("action_2", "action_3")]:
    t = data.groupby([a, b]).size().reset_index(name="sessions")
    t.columns = ["source", "target", "sessions"]
    t["source"] = a[-1] + ":" + t.source
    t["target"] = b[-1] + ":" + t.target
    links.append(t)
t = data.groupby(["action_3", "future_purchase"]).size().reset_index(name="sessions")
t.columns = ["source", "target", "sessions"]
t["source"] = "3:" + t.source
t["target"] = np.where(t.target == 1, "out:purchase", "out:no purchase")
links.append(t)
save(pd.concat(links), "sankey")

# Daily volume & conversion
daily = (data.assign(day=data.prediction_time.dt.date)
         .groupby("day")["future_purchase"].agg(sessions="size", purchase_rate="mean").reset_index())
save(daily, "daily")

# Hour x weekday heatmap
hw = (data.assign(hour=data.prediction_time.dt.hour, weekday=data.prediction_time.dt.dayofweek)
      .groupby(["weekday", "hour"])["future_purchase"].agg(sessions="size", purchase_rate="mean").reset_index())
save(hw, "hour_weekday")

# --------------------------------------------------------------------------------------
# 6. Split
# --------------------------------------------------------------------------------------
t1, t2, t3 = pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-01"), pd.Timestamp("2020-03-01")
train = data[(data.prediction_time < t1) & (data.session_end < t1)]
val = data[(data.prediction_time >= t1) & (data.prediction_time < t2) & (data.session_end < t2)]
test = data[(data.prediction_time >= t2) & (data.prediction_time < t3)]
crossing = int(((data.prediction_time < t1) & (data.session_end >= t1)).sum()
               + ((data.prediction_time >= t1) & (data.prediction_time < t2) & (data.session_end >= t2)).sum())
KPI["split"] = {
    "crossing_removed": crossing,
    "train": {"period": "Oct - Dec 2019", "sessions": len(train), "purchase_rate": float(train.future_purchase.mean())},
    "validation": {"period": "Jan 2020", "sessions": len(val), "purchase_rate": float(val.future_purchase.mean())},
    "test": {"period": "Feb 2020", "sessions": len(test), "purchase_rate": float(test.future_purchase.mean())},
}

# --------------------------------------------------------------------------------------
# 7. Final model (optimized XGBoost, best Optuna parameters from the notebook)
# --------------------------------------------------------------------------------------
CATEGORICAL = ["action_1", "action_2", "action_3"]
NUMERICAL = ["n_cart", "n_remove", "avg_price", "distinct_products", "distinct_categories",
             "gap_1_2", "gap_2_3", "session_duration_3"]
ACTIONS = ["cart", "remove_from_cart", "view"]  # sorted, same as sklearn OneHotEncoder
FEATURES = [f"{c}_{a}" for c in CATEGORICAL for a in ACTIONS] + NUMERICAL


def encode(df):
    X = pd.DataFrame(index=df.index)
    for c in CATEGORICAL:
        for a in ACTIONS:
            X[f"{c}_{a}"] = (df[c] == a).astype(float)
    for c in NUMERICAL:
        X[c] = df[c].astype(float)
    return X[FEATURES].to_numpy()


X_train, X_val, X_test = encode(train), encode(val), encode(test)
y_train, y_val, y_test = train.future_purchase.values, val.future_purchase.values, test.future_purchase.values

BEST_PARAMS = {"n_estimators": 297, "max_depth": 7, "learning_rate": 0.026473623258453708,
               "subsample": 0.7732384546253949, "colsample_bytree": 0.6642645155626334}
log("Training the optimized XGBoost")
model = XGBClassifier(**BEST_PARAMS, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)
booster = model.get_booster()
booster.feature_names = FEATURES
booster.save_model(str(OUT / "model.json"))

proba_val = model.predict_proba(X_val)[:, 1]
proba_test = model.predict_proba(X_test)[:, 1]

precision, recall, thresholds = precision_recall_curve(y_val, proba_val)
f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-9)
threshold = float(thresholds[np.argmax(f1)])
limits = np.quantile(proba_val, [0.4, 0.7, 0.9]).tolist()

pred_test = (proba_test >= threshold).astype(int)
tp = int(((pred_test == 1) & (y_test == 1)).sum()); fp = int(((pred_test == 1) & (y_test == 0)).sum())
fn = int(((pred_test == 0) & (y_test == 1)).sum()); tn = int(((pred_test == 0) & (y_test == 0)).sum())
KPI["model"] = {
    "name": "XGBoost (Optuna-optimized)", "params": BEST_PARAMS, "features": FEATURES,
    "threshold": threshold, "segment_limits": limits,
    "val": {"pr_auc": float(average_precision_score(y_val, proba_val)), "mean_proba": float(proba_val.mean()),
            "base_rate": float(y_val.mean())},
    "test": {"pr_auc": float(average_precision_score(y_test, proba_test)), "base_rate": float(y_test.mean()),
             "precision": tp / (tp + fp), "recall": tp / (tp + fn), "f1": 2 * tp / (2 * tp + fp + fn),
             "flagged": float(pred_test.mean()), "tp": tp, "fp": fp, "fn": fn, "tn": tn},
}
log(f"Test PR-AUC {KPI['model']['test']['pr_auc']:.4f} | threshold {threshold:.3f}")

# Model comparison and Optuna results (validation, from the notebook run)
save(pd.DataFrame([
    {"model": "CatBoost", "pr_auc": 0.210087, "precision": 0.175875, "recall": 0.806756, "f1": 0.288793, "mean_proba": 0.462911},
    {"model": "LightGBM", "pr_auc": 0.208754, "precision": 0.176014, "recall": 0.798293, "f1": 0.288433, "mean_proba": 0.460348},
    {"model": "XGBoost", "pr_auc": 0.207215, "precision": 0.176986, "recall": 0.784877, "f1": 0.288840, "mean_proba": 0.457354},
    {"model": "Random Forest", "pr_auc": 0.206663, "precision": 0.180826, "recall": 0.737684, "f1": 0.290454, "mean_proba": 0.428224},
    {"model": "XGBoost optimized (final)", "pr_auc": KPI["model"]["val"]["pr_auc"], "precision": np.nan,
     "recall": np.nan, "f1": np.nan, "mean_proba": KPI["model"]["val"]["mean_proba"]},
]), "model_comparison")


# Threshold grid (validation + test) for the interactive slider
def grid(y, p, name):
    rows = []
    for t in np.round(np.arange(0.01, 0.501, 0.005), 3):
        pred = p >= t
        tp_ = int((pred & (y == 1)).sum()); fp_ = int((pred & (y == 0)).sum())
        fn_ = int((~pred & (y == 1)).sum()); tn_ = int((~pred & (y == 0)).sum())
        rows.append({"set": name, "threshold": t, "tp": tp_, "fp": fp_, "fn": fn_, "tn": tn_})
    return pd.DataFrame(rows)


save(pd.concat([grid(y_val, proba_val, "validation"), grid(y_test, proba_test, "test")]), "threshold_grid")

# PR curve on test (downsampled)
p, r, _ = precision_recall_curve(y_test, proba_test)
idx = np.unique(np.linspace(0, len(p) - 1, 400).astype(int))
save(pd.DataFrame({"recall": r[idx], "precision": p[idx]}), "pr_curve")

# Score distribution by class (test)
hist_bins = np.linspace(0, 0.6, 61)
h1, _ = np.histogram(proba_test[y_test == 1], bins=hist_bins)
h0, _ = np.histogram(proba_test[y_test == 0], bins=hist_bins)
save(pd.DataFrame({"bin_left": hist_bins[:-1], "buyers": h1, "non_buyers": h0}), "score_hist")


# Deciles & gains
def decile_table(y, p):
    d = pd.DataFrame({"y": y, "proba": p})
    d["decile"] = pd.qcut(pd.Series(p).rank(method="first", ascending=False), 10, labels=range(1, 11)).astype(int)
    d = d.groupby("decile").agg(sessions=("y", "size"), purchase_rate=("y", "mean"),
                                min_score=("proba", "min"), max_score=("proba", "max")).reset_index()
    d["lift"] = d.purchase_rate / y.mean()
    d["share_of_buyers"] = d.purchase_rate * d.sessions / y.sum()
    d["cum_share_of_buyers"] = d.share_of_buyers.cumsum()
    return d


save(decile_table(y_test, proba_test), "deciles_test")
save(decile_table(y_val, proba_val), "deciles_val")
order = np.argsort(-proba_test)
cum = np.cumsum(y_test[order]) / y_test.sum()
pts = np.linspace(0, len(order) - 1, 101).astype(int)
save(pd.DataFrame({"share_targeted": np.linspace(0, 1, 101), "share_of_buyers": np.r_[0, cum[pts[1:]]]}), "gains")

# Weekly stability
weekly = test[["prediction_time", "future_purchase"]].copy()
weekly["proba"] = proba_test
weekly["week"] = weekly.prediction_time.dt.to_period("W").astype(str)
wk = []
for w, g in weekly.groupby("week"):
    wk.append({"week": w, "sessions": len(g), "purchase_rate": g.future_purchase.mean(),
               "pr_auc": average_precision_score(g.future_purchase, g.proba),
               "recall": ((g.proba >= threshold) & (g.future_purchase == 1)).sum() / g.future_purchase.sum()})
save(pd.DataFrame(wk), "weekly")

# --------------------------------------------------------------------------------------
# 8. Error analysis
# --------------------------------------------------------------------------------------
log("Error analysis")
errors = test[["sequence", "has_cart", "n_remove", "session_duration_3"]].copy()
errors["y"] = y_test
errors["pred"] = pred_test
errors["result"] = np.select(
    [(errors.pred == 1) & (errors.y == 1), (errors.pred == 1) & (errors.y == 0), (errors.pred == 0) & (errors.y == 1)],
    ["True positive", "False positive", "False negative"], "True negative")
prof = errors.groupby("result").agg(sessions=("y", "size"), share_with_cart=("has_cart", "mean"),
                                    avg_removals=("n_remove", "mean"),
                                    avg_duration=("session_duration_3", "mean")).reset_index()
save(prof, "error_profile")
top = []
for res in ["False positive", "False negative"]:
    v = errors[errors.result == res].sequence.value_counts(normalize=True).head(8).reset_index()
    v.columns = ["sequence", "share"]
    v["result"] = res
    top.append(v)
save(pd.concat(top), "error_sequences")

# --------------------------------------------------------------------------------------
# 9. SHAP (exact TreeSHAP via XGBoost pred_contribs)
# --------------------------------------------------------------------------------------
log("SHAP values")
import xgboost as xgb

sample = np.random.RandomState(42).choice(len(X_test), 2000, replace=False)
contrib = booster.predict(xgb.DMatrix(X_test[sample], feature_names=FEATURES), pred_contribs=True)
shap_vals = contrib[:, :-1]
KPI["model"]["shap_base_value"] = float(contrib[0, -1])
pd.DataFrame(shap_vals, columns=FEATURES).to_parquet(OUT / "shap_values.parquet", index=False)
pd.DataFrame(X_test[sample], columns=FEATURES).to_parquet(OUT / "shap_features.parquet", index=False)
imp = pd.DataFrame({"feature": FEATURES, "mean_abs_shap": np.abs(shap_vals).mean(0)}).sort_values(
    "mean_abs_shap", ascending=False)
save(imp, "shap_importance")
chk = pd.DataFrame({"action_3": test.action_3.values[sample],
                    "shap": shap_vals[:, FEATURES.index("action_3_view")]})
save(chk.groupby("action_3").agg(sessions=("shap", "size"), mean_shap=("shap", "mean")).reset_index(),
     "shap_action3")

# --------------------------------------------------------------------------------------
# 10. Segmentation & actions
# --------------------------------------------------------------------------------------
log("Segments and recommended actions")


def segment(p):
    return np.select([p >= limits[2], p >= limits[1], p >= limits[0]],
                     ["1. Top 10%", "2. Next 20%", "3. Middle 30%"], "4. Bottom 40%")


def recommend_action(seg, has_cart):
    return {
        "1. Top 10%": ("Facilitate checkout", "Highlight viewed products"),
        "2. Next 20%": ("Cart reminder", "Personalized recommendations"),
        "3. Middle 30%": ("Cart reminder", "Recommend similar products"),
        "4. Bottom 40%": ("Low-cost cart reminder", "Product discovery or no action"),
    }[seg][0 if has_cart else 1]


segs = pd.DataFrame({"segment": segment(proba_test), "y": y_test, "has_cart": test.has_cart.values})
seg_tab = segs.groupby("segment").agg(sessions=("y", "size"), purchase_rate=("y", "mean"),
                                      share_with_cart=("has_cart", "mean"), buyers=("y", "sum")).reset_index()
seg_tab["share_of_sessions"] = seg_tab.sessions / len(segs)
seg_tab["share_of_buyers"] = seg_tab.buyers / segs.y.sum()
save(seg_tab, "segments")
sc = segs.groupby(["segment", "has_cart"]).agg(sessions=("y", "size"), purchase_rate=("y", "mean")).reset_index()
save(sc, "segment_cart")
segs["action"] = [recommend_action(a, b) for a, b in zip(segs.segment, segs.has_cart)]
save(segs.groupby(["segment", "has_cart", "action"]).agg(sessions=("y", "size"), purchase_rate=("y", "mean"))
     .reset_index(), "actions")

# Session explorer sample (test sessions with score, segment, action)
expl = test[["session_id", "prediction_time", "action_1", "action_2", "action_3", "n_cart", "n_remove",
             "avg_price", "distinct_products", "distinct_categories", "gap_1_2", "gap_2_3",
             "session_duration_3", "future_purchase", "has_cart"]].copy()
expl["score"] = proba_test
expl["segment"] = segs.segment.values
expl["action"] = segs.action.values
expl = expl.sample(5000, random_state=7).sort_values("prediction_time")
expl.to_parquet(OUT / "sessions_sample.parquet", index=False)

# Feature ranges for the simulator
KPI["feature_stats"] = {c: {q: float(data[c].quantile(v)) for q, v in
                            [("p01", .01), ("p25", .25), ("median", .5), ("p75", .75), ("p99", .99)]}
                        for c in ["avg_price", "gap_1_2", "gap_2_3", "session_duration_3"]}

# --------------------------------------------------------------------------------------
# 11. A/B test baseline (user-level, 7-day conversion, next 20% + middle 30%)
# --------------------------------------------------------------------------------------
log("A/B test baseline")
middle = segs.segment.isin(["2. Next 20%", "3. Middle 30%"]).to_numpy()
ab = test[["session_id", "prediction_time"]].copy()
ab["user_id"] = ab.session_id.str.split("_").str[0]
ab = ab[middle].sort_values("prediction_time").drop_duplicates("user_id")
ab = ab[ab.prediction_time < "2020-02-23"]
con.register("ab_users", ab)
ab_res = con.execute("""
SELECT a.user_id, MAX(CASE WHEN e.event_time IS NOT NULL THEN 1 ELSE 0 END) AS conversion
FROM ab_users a LEFT JOIN purchases e
  ON CAST(e.user_id AS VARCHAR) = a.user_id
 AND e.event_time > a.prediction_time AND e.event_time <= a.prediction_time + INTERVAL '7 days'
GROUP BY a.user_id""").df()
KPI["ab_test"] = {"base_rate": float(ab_res.conversion.mean()), "users_per_day": float(len(ab_res) / 22),
                  "session_purchase_rate": float(segs[middle].y.mean()),
                  "sessions_per_day": float(middle.sum() / 29)}

# --------------------------------------------------------------------------------------
# 12. Hillstrom e-mail experiment (causal effects)
# --------------------------------------------------------------------------------------
log("Hillstrom experiment")
try:
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder

    hpath = CACHE / "hillstrom.csv"
    if not hpath.exists():
        pd.read_csv("http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv").to_csv(hpath, index=False)
    hill = pd.read_csv(hpath)

    def effect_with_ci(d0, group_col, outcome, scale=1.0, cost=0.0):
        rows = []
        for g, d in d0.groupby(group_col, observed=True):
            ctrl = d.loc[d.segment == "No E-Mail", outcome]
            for email in ["Mens E-Mail", "Womens E-Mail"]:
                trt = d.loc[d.segment == email, outcome]
                diff = trt.mean() - ctrl.mean()
                se = np.sqrt(trt.var() / len(trt) + ctrl.var() / len(ctrl))
                rows.append({"group": str(g), "email": email, "effect": scale * diff - cost,
                             "ci_low": scale * (diff - 1.96 * se) - cost, "ci_high": scale * (diff + 1.96 * se) - cost,
                             "n": len(trt) + len(ctrl)})
        out = pd.DataFrame(rows)
        out["significant"] = (out.ci_low > 0) | (out.ci_high < 0)
        return out

    hill["all"] = "All customers"
    overall = []
    for col in ["visit", "conversion", "spend"]:
        e = effect_with_ci(hill, "all", col)
        e["outcome"] = col
        overall.append(e)
    save(pd.concat(overall), "hill_overall")
    save(hill.groupby("segment")[["visit", "conversion", "spend"]].mean().reset_index(), "hill_means")

    features_h = ["recency", "history", "mens", "womens", "newbie", "zip_code", "channel"]
    prop_model = make_pipeline(
        ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), ["zip_code", "channel"]),
                           ("num", "passthrough", ["recency", "history", "mens", "womens", "newbie"])]),
        LogisticRegression(max_iter=1000))
    is_control = hill.segment == "No E-Mail"
    hill["propensity"] = 0.0
    hill.loc[is_control, "propensity"] = cross_val_predict(
        prop_model, hill.loc[is_control, features_h], hill.loc[is_control, "visit"], cv=5,
        method="predict_proba")[:, 1]
    prop_model.fit(hill.loc[is_control, features_h], hill.loc[is_control, "visit"])
    hill.loc[~is_control, "propensity"] = prop_model.predict_proba(hill.loc[~is_control, features_h])[:, 1]
    hill["propensity_group"] = pd.qcut(hill.propensity, 4, labels=["1. Low", "2. Medium-low", "3. Medium-high", "4. High"])
    save(effect_with_ci(hill, "propensity_group", "visit"), "hill_propensity")

    hill["past_purchases"] = np.select(
        [(hill.mens == 1) & (hill.womens == 1), hill.mens == 1, hill.womens == 1],
        ["Men and women", "Men only", "Women only"], "None")
    save(effect_with_ci(hill, "past_purchases", "visit"), "hill_history")

    # Raw spend effect by propensity group (margin & cost applied interactively in the app)
    save(effect_with_ci(hill, "propensity_group", "spend"), "hill_spend")
    KPI["hillstrom"] = {"customers": len(hill), "available": True}
except Exception as exc:  # network issue -> app shows notebook values
    log(f"Hillstrom skipped: {exc}")
    KPI["hillstrom"] = {"available": False}

# --------------------------------------------------------------------------------------
with open(OUT / "kpis.json", "w") as f:
    json.dump(KPI, f, indent=2, default=float)
con.close()
import shutil
shutil.rmtree(CACHE / "tmp", ignore_errors=True)
log("Done. Artifacts written to ./artifacts")
