"""
SmartCart AI processing engine
==============================
Turns ANY e-commerce event log (a company's own data) into a workspace: business tables
(KPIs, customers, products, carts, basket pairs) and purchase-intent tables scored by the
SmartCart model (XGBoost pre-trained on 20.7M REES46 events, optional fine-tuning).

The method is the one validated in the project: de-duplication, 30-minute sessionization,
features from the first three events, rank-based score segments, next-best-action rules.
"""
import hashlib

import duckdb
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats

STD_EVENTS = ["view", "cart", "remove_from_cart", "purchase"]
REQUIRED = {"event_time": "Event timestamp", "event_type": "Event type (view / cart / purchase…)",
            "user_id": "Customer / visitor ID", "product_id": "Product ID (SKU)", "price": "Unit price"}
OPTIONAL = {"brand": "Brand", "category": "Category", "product_name": "Product name"}

ALIASES = {
    "event_time": ["event_time", "timestamp", "time", "datetime", "date", "event_date", "created_at", "ts", "date_heure",
                   "horodatage"],
    "event_type": ["event_type", "event", "action", "type", "event_name", "interaction"],
    "user_id": ["user_id", "customer_id", "client_id", "visitor_id", "user", "customer", "uid", "id_client", "client",
                "customer_number", "member_id"],
    "product_id": ["product_id", "sku", "item_id", "product", "item", "article_id", "id_produit"],
    "price": ["price", "unit_price", "amount", "value", "prix", "prix_unitaire", "revenue", "montant"],
    "brand": ["brand", "marque", "manufacturer"],
    "category": ["category", "category_code", "category_name", "categorie", "category_id"],
    "product_name": ["product_name", "name", "title", "item_name", "nom_produit", "nom", "libelle"],
}
EVENT_KEYWORDS = [
    ("remove_from_cart", ["remove", "delete", "retir", "retrait", "suppr"]),
    ("purchase", ["purchase", "order", "buy", "transaction", "checkout_complete", "achat", "commande", "sale"]),
    ("cart", ["cart", "basket", "panier", "add"]),
    ("view", ["view", "page", "detail", "visit", "impression", "click", "vue"]),
]

BEST_PARAMS = {"n_estimators": 297, "max_depth": 7, "learning_rate": 0.026473623258453708,
               "subsample": 0.7732384546253949, "colsample_bytree": 0.6642645155626334}
CATEGORICAL = ["action_1", "action_2", "action_3"]
ACTIONS = ["cart", "remove_from_cart", "view"]
NUMERICAL = ["n_cart", "n_remove", "avg_price", "distinct_products", "distinct_categories",
             "gap_1_2", "gap_2_3", "session_duration_3"]
FEATURES = [f"{c}_{a}" for c in CATEGORICAL for a in ACTIONS] + NUMERICAL
SEGMENTS = ["1. Top 10%", "2. Next 20%", "3. Middle 30%", "4. Bottom 40%"]


# ------------------------------------------------------------------ mapping -------
def guess_mapping(columns):
    cols = {c.lower().strip(): c for c in columns}
    out = {}
    for key, names in ALIASES.items():
        out[key] = next((cols[n] for n in names if n in cols), None)
    return out


def guess_event(value):
    v = str(value).lower()
    for std, kws in EVENT_KEYWORDS:
        if any(k in v for k in kws):
            return std
    return "ignore"


def parse_time(s):
    """Parse timestamps, detecting day-first formats (e.g. 31/01/2024)."""
    if pd.api.types.is_datetime64_any_dtype(s):
        t = pd.to_datetime(s, utc=True)
        return t.dt.tz_localize(None)
    txt = s.astype(str).str.strip()
    parts = txt.head(5000).str.extract(r"^(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{2,4})")
    dayfirst = False
    if parts.notna().all(axis=1).any():
        a, b = pd.to_numeric(parts[0], errors="coerce"), pd.to_numeric(parts[1], errors="coerce")
        dayfirst = bool(a.max() > 12) or not bool(b.max() > 12)
    sample = txt.head(2000)
    candidates = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d",
                  "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%d/%m/%Y", "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M",
                  "%m/%d/%Y", "%d-%m-%Y %H:%M:%S", "%d.%m.%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S UTC"]
    if dayfirst:
        candidates = [c for c in candidates if not c.startswith("%m")]
    for fmt in candidates:  # fast path: one exact format for the whole column
        if pd.to_datetime(sample, format=fmt, errors="coerce").notna().mean() > 0.99:
            t = pd.to_datetime(txt, format=fmt, errors="coerce")
            if t.notna().mean() > 0.95:
                return t
    t = pd.to_datetime(txt, errors="coerce", utc=True, dayfirst=dayfirst, format="mixed")
    return t.dt.tz_localize(None)


def standardize(raw, mapping, event_map):
    """Rename / cast the company's columns to the SmartCart schema."""
    df = pd.DataFrame({
        "event_time": parse_time(raw[mapping["event_time"]]),
        "event_type": raw[mapping["event_type"]].astype(str).map(event_map).fillna("ignore"),
        "user_id": raw[mapping["user_id"]].astype(str),
        "product_id": raw[mapping["product_id"]].astype(str),
        "price": pd.to_numeric(raw[mapping["price"]], errors="coerce"),
    })
    for k in OPTIONAL:
        col = mapping.get(k)
        df[k] = raw[col].astype(str).where(raw[col].notna(), None) if col else None
    return df


def fingerprint(df):
    h = hashlib.sha1(pd.util.hash_pandas_object(df.head(50000), index=False).values.tobytes())
    h.update(str(len(df)).encode())
    return h.hexdigest()[:12]


# ------------------------------------------------------------------ helpers -------
def _month_label(m):
    return pd.Timestamp(m + "-01").strftime("%b %Y")


def encode(df):
    X = pd.DataFrame(index=df.index)
    for c in CATEGORICAL:
        for a in ACTIONS:
            X[f"{c}_{a}"] = (df[c] == a).astype(float)
    for c in NUMERICAL:
        X[c] = df[c].astype(float)
    return X[FEATURES].to_numpy()


def predict(booster, X):
    return booster.predict(xgb.DMatrix(X, feature_names=FEATURES))


# ------------------------------------------------------------------ main ----------
def build_workspace(std, booster, name="My company", progress=None):
    """std: standardized DataFrame. Returns a workspace dict {meta, kpis, tables, data}."""
    def step(p, msg):
        if progress:
            progress(p, msg)

    T, meta = {}, {"name": name}
    con = duckdb.connect()
    con.execute("SET preserve_insertion_order=false")
    con.register("raw", std)
    meta["n_rows_uploaded"] = len(std)
    meta["n_ignored_events"] = int((std.event_type == "ignore").sum())
    meta["n_bad_time"] = int(std.event_time.isna().sum())
    meta["has_brand"] = bool(std.brand.notna().any())
    meta["has_category"] = bool(std.category.notna().any())
    meta["has_names"] = bool(std.product_name.notna().any())

    step(0.05, "Cleaning and de-duplicating events")
    con.execute("""
    CREATE TABLE ev0 AS SELECT DISTINCT event_time, event_type, user_id, product_id, price,
           COALESCE(brand, '(unbranded)') AS brand, category, product_name
    FROM raw WHERE event_type IN ('view','cart','remove_from_cart','purchase')
      AND event_time IS NOT NULL AND user_id IS NOT NULL""")
    con.unregister("raw")
    n_valid = len(std) - meta["n_ignored_events"] - meta["n_bad_time"]
    meta["n_events"] = int(con.execute("SELECT COUNT(*) FROM ev0").fetchone()[0])
    meta["n_duplicates"] = max(0, n_valid - meta["n_events"])
    if meta["n_events"] == 0:
        raise ValueError("No usable event after mapping: check the event-type mapping and the timestamp column.")

    step(0.15, "Rebuilding sessions (30-minute inactivity rule)")
    con.execute("""
    CREATE TABLE ev AS
    WITH o AS (SELECT *, CASE event_type WHEN 'view' THEN 1 WHEN 'cart' THEN 2 WHEN 'remove_from_cart' THEN 3
                                         ELSE 4 END AS t FROM ev0),
         f AS (SELECT *, CASE WHEN LAG(event_time) OVER w IS NULL
                                OR event_time - LAG(event_time) OVER w > INTERVAL 30 MINUTE THEN 1 ELSE 0 END AS ns
               FROM o WINDOW w AS (PARTITION BY user_id ORDER BY event_time, t, product_id, price)),
         n AS (SELECT *, SUM(ns) OVER (PARTITION BY user_id ORDER BY event_time, t, product_id, price
                                       ROWS UNBOUNDED PRECEDING) AS sn FROM f)
    SELECT event_time, event_type, user_id, product_id, price, brand, category, product_name,
           user_id || '_' || CAST(sn AS VARCHAR) AS session_id,
           ROW_NUMBER() OVER (PARTITION BY user_id, sn ORDER BY event_time, t, product_id, price) AS pos
    FROM n""")
    con.execute("DROP TABLE ev0")
    con.execute("CREATE TABLE pur AS SELECT * FROM ev WHERE event_type = 'purchase' AND price > 0")
    q = lambda s: con.execute(s).df()  # noqa: E731

    start, end = con.execute("SELECT MIN(event_time), MAX(event_time) FROM ev").fetchone()
    end = pd.Timestamp(end)
    meta.update(data_start=str(start), data_end=str(end))
    months = [r[0] for r in con.execute("SELECT DISTINCT strftime(event_time, '%Y-%m') FROM ev ORDER BY 1").fetchall()]
    meta["months"] = months
    last = months[-1]
    prev = months[-2] if len(months) > 1 else months[-1]
    meta.update(last_month=last, prev_month=prev)

    # ----------------------------------------------------------- business tables --
    step(0.25, "Computing business KPIs")
    T["event_types"] = q("SELECT event_type, COUNT(*) AS events FROM ev GROUP BY 1 ORDER BY 2 DESC")
    T["events_by_month"] = q("""SELECT strftime(event_time, '%Y-%m') AS month, event_type, COUNT(*) AS events
                                FROM ev GROUP BY 1, 2 ORDER BY 1, 2""")
    T["daily"] = q("""
    SELECT CAST(event_time AS DATE) AS day,
           COUNT(*) FILTER (WHERE event_type='view') AS views, COUNT(*) FILTER (WHERE event_type='cart') AS carts,
           COUNT(*) FILTER (WHERE event_type='remove_from_cart') AS removes,
           COUNT(*) FILTER (WHERE event_type='purchase' AND price > 0) AS units,
           COUNT(DISTINCT user_id) AS visitors, COUNT(DISTINCT user_id) FILTER (WHERE event_type='cart') AS carters,
           COUNT(DISTINCT user_id) FILTER (WHERE event_type='purchase') AS buyers,
           COUNT(DISTINCT session_id) FILTER (WHERE event_type='purchase') AS orders,
           COALESCE(SUM(price) FILTER (WHERE event_type='purchase' AND price > 0), 0) AS revenue
    FROM ev GROUP BY 1 ORDER BY 1""")
    T["monthly"] = q("""
    SELECT strftime(event_time, '%Y-%m') AS month, COUNT(DISTINCT user_id) AS visitors,
           COUNT(DISTINCT user_id) FILTER (WHERE event_type='view') AS viewers,
           COUNT(DISTINCT user_id) FILTER (WHERE event_type='cart') AS carters,
           COUNT(DISTINCT user_id) FILTER (WHERE event_type='purchase') AS buyers,
           COUNT(DISTINCT session_id) FILTER (WHERE event_type='purchase') AS orders,
           COALESCE(SUM(price) FILTER (WHERE event_type='purchase' AND price > 0), 0) AS revenue,
           COUNT(*) FILTER (WHERE event_type='purchase' AND price > 0) AS units
    FROM ev GROUP BY 1 ORDER BY 1""")
    T["hour_weekday"] = q("""
    SELECT dayofweek(event_time) AS weekday, hour(event_time) AS hour,
           COUNT(*) FILTER (WHERE event_type='view') AS views, COUNT(*) FILTER (WHERE event_type='purchase') AS units,
           COALESCE(SUM(price) FILTER (WHERE event_type='purchase' AND price > 0), 0) AS revenue
    FROM ev GROUP BY 1, 2""")

    step(0.35, "Analysing brands and products")
    T["brand_month"] = q("""
    SELECT strftime(event_time, '%Y-%m') AS month, brand,
           COUNT(*) FILTER (WHERE event_type='view') AS views, COUNT(*) FILTER (WHERE event_type='cart') AS carts,
           COUNT(*) FILTER (WHERE event_type='remove_from_cart') AS removes,
           COUNT(*) FILTER (WHERE event_type='purchase' AND price > 0) AS units,
           COALESCE(SUM(price) FILTER (WHERE event_type='purchase' AND price > 0), 0) AS revenue,
           COUNT(DISTINCT user_id) FILTER (WHERE event_type='purchase') AS buyers,
           COUNT(DISTINCT product_id) AS products
    FROM ev GROUP BY 1, 2""")
    T["products"] = q(f"""
    SELECT product_id, ANY_VALUE(brand) AS brand, ANY_VALUE(category) AS category,
           ANY_VALUE(product_name) AS product_name, MEDIAN(price) FILTER (WHERE price > 0) AS price,
           COUNT(*) FILTER (WHERE event_type='view') AS views, COUNT(*) FILTER (WHERE event_type='cart') AS carts,
           COUNT(*) FILTER (WHERE event_type='remove_from_cart') AS removes,
           COUNT(*) FILTER (WHERE event_type='purchase' AND price > 0) AS units,
           COALESCE(SUM(price) FILTER (WHERE event_type='purchase' AND price > 0), 0) AS revenue,
           COUNT(DISTINCT user_id) FILTER (WHERE event_type='purchase') AS buyers,
           COALESCE(SUM(price) FILTER (WHERE event_type='purchase' AND price > 0
                                       AND strftime(event_time, '%Y-%m') = '{last}'), 0) AS revenue_last,
           COALESCE(SUM(price) FILTER (WHERE event_type='purchase' AND price > 0
                                       AND strftime(event_time, '%Y-%m') = '{prev}'), 0) AS revenue_prev
    FROM ev GROUP BY 1""")
    meta["n_products"] = len(T["products"])
    meta["n_brands"] = int((T["products"].brand.unique() != "(unbranded)").sum())
    meta["n_categories"] = int(T["products"].category.nunique()) if meta["has_category"] else 0
    T["price_bands"] = q("""
    SELECT CASE WHEN price < 2 THEN '1. < $2' WHEN price < 5 THEN '2. $2-5' WHEN price < 10 THEN '3. $5-10'
                WHEN price < 20 THEN '4. $10-20' WHEN price < 50 THEN '5. $20-50' ELSE '6. $50+' END AS price_band,
           COUNT(*) FILTER (WHERE event_type='view') AS views, COUNT(*) FILTER (WHERE event_type='cart') AS carts,
           COUNT(*) FILTER (WHERE event_type='purchase') AS units,
           COALESCE(SUM(price) FILTER (WHERE event_type='purchase'), 0) AS revenue
    FROM ev WHERE price > 0 GROUP BY 1 ORDER BY 1""")

    step(0.42, "Finding products bought together")
    con.execute("""CREATE TABLE basket AS
    WITH p AS (SELECT DISTINCT session_id, product_id FROM pur),
         s AS (SELECT session_id FROM p GROUP BY 1 HAVING COUNT(*) BETWEEN 2 AND 30)
    SELECT p.* FROM p JOIN s USING (session_id)""")
    n_orders = int(con.execute("SELECT COUNT(DISTINCT session_id) FROM pur").fetchone()[0])
    min_pair = max(3, int(n_orders * 0.0002))
    pairs = q(f"""
    SELECT a.product_id AS product_a, b.product_id AS product_b, COUNT(*) AS together
    FROM basket a JOIN basket b ON a.session_id = b.session_id AND a.product_id < b.product_id
    GROUP BY 1, 2 HAVING COUNT(*) >= {min_pair} ORDER BY together DESC LIMIT 400""")
    po = q("SELECT product_id, COUNT(DISTINCT session_id) AS orders FROM pur GROUP BY 1").set_index("product_id").orders
    pairs["orders_a"] = pairs.product_a.map(po)
    pairs["orders_b"] = pairs.product_b.map(po)
    pairs["confidence_a_b"] = pairs.together / pairs.orders_a
    pairs["confidence_b_a"] = pairs.together / pairs.orders_b
    pairs["lift"] = pairs.together * max(n_orders, 1) / (pairs.orders_a * pairs.orders_b)
    T["pairs"] = pairs
    T["brand_pairs"] = q("""
    WITH b AS (SELECT DISTINCT session_id, brand FROM pur WHERE brand <> '(unbranded)'),
         s AS (SELECT session_id FROM b GROUP BY 1 HAVING COUNT(*) BETWEEN 2 AND 15),
         bb AS (SELECT b.* FROM b JOIN s USING (session_id))
    SELECT x.brand AS brand_a, y.brand AS brand_b, COUNT(*) AS together
    FROM bb x JOIN bb y ON x.session_id = y.session_id AND x.brand < y.brand GROUP BY 1, 2 ORDER BY 3 DESC LIMIT 60""")
    con.execute("DROP TABLE basket")
    meta["orders_total"] = n_orders

    step(0.5, "Building the customer 360 view")
    T["customers"] = q("""
    WITH fav AS (SELECT user_id, arg_max(brand, rev) AS favorite_brand
                 FROM (SELECT user_id, brand, SUM(price) AS rev FROM pur GROUP BY 1, 2) GROUP BY 1),
         act AS (SELECT user_id, MIN(event_time) AS first_seen, MAX(event_time) AS last_seen,
                        COUNT(*) FILTER (WHERE event_type='view') AS n_views,
                        COUNT(*) FILTER (WHERE event_type='cart') AS n_carts, COUNT(DISTINCT session_id) AS n_sessions
                 FROM ev WHERE user_id IN (SELECT user_id FROM pur) GROUP BY 1)
    SELECT p.user_id, MIN(p.event_time) AS first_purchase, MAX(p.event_time) AS last_purchase,
           COUNT(DISTINCT p.session_id) AS orders, COUNT(DISTINCT CAST(p.event_time AS DATE)) AS purchase_days,
           COUNT(*) AS items, SUM(p.price) AS revenue, ANY_VALUE(f.favorite_brand) AS favorite_brand,
           ANY_VALUE(a.first_seen) AS first_seen, ANY_VALUE(a.last_seen) AS last_seen,
           ANY_VALUE(a.n_views) AS n_views, ANY_VALUE(a.n_carts) AS n_carts, ANY_VALUE(a.n_sessions) AS n_sessions
    FROM pur p LEFT JOIN fav f USING (user_id) LEFT JOIN act a USING (user_id) GROUP BY 1""")
    meta["n_customers"] = len(T["customers"])
    meta["n_visitors"] = int(con.execute("SELECT COUNT(DISTINCT user_id) FROM ev").fetchone()[0])
    T["cohorts"] = q("""
    WITH m AS (SELECT DISTINCT user_id, date_trunc('month', event_time) AS month FROM pur),
         f AS (SELECT user_id, MIN(month) AS cohort FROM m GROUP BY 1)
    SELECT strftime(f.cohort, '%Y-%m') AS cohort, date_diff('month', f.cohort, m.month) AS month_index,
           COUNT(*) AS customers FROM m JOIN f USING (user_id) GROUP BY 1, 2 ORDER BY 1, 2""")
    T["new_vs_returning"] = q("""
    WITH m AS (SELECT user_id, date_trunc('month', event_time) AS month, SUM(price) AS rev FROM pur GROUP BY 1, 2),
         f AS (SELECT user_id, MIN(month) AS cohort FROM m GROUP BY 1)
    SELECT strftime(m.month, '%Y-%m') AS month,
           CASE WHEN m.month = f.cohort THEN 'New customers' ELSE 'Returning customers' END AS type,
           COUNT(*) AS customers, SUM(rev) AS revenue FROM m JOIN f USING (user_id) GROUP BY 1, 2 ORDER BY 1, 2""")

    step(0.58, "Measuring cart abandonment")
    T["abandon_monthly"] = q("""
    WITH c AS (SELECT strftime(event_time, '%Y-%m') AS month, user_id, product_id, MAX(price) AS price
               FROM ev WHERE event_type='cart' AND price > 0 GROUP BY 1, 2, 3),
         p AS (SELECT DISTINCT strftime(event_time, '%Y-%m') AS month, user_id, product_id FROM pur)
    SELECT c.month, COUNT(*) AS carted_items, SUM(c.price) AS carted_value,
           COUNT(*) FILTER (WHERE p.user_id IS NULL) AS abandoned_items,
           COALESCE(SUM(c.price) FILTER (WHERE p.user_id IS NULL), 0) AS abandoned_value,
           COUNT(DISTINCT c.user_id) AS carters,
           COUNT(DISTINCT c.user_id) FILTER (WHERE p.user_id IS NULL) AS users_with_abandoned
    FROM c LEFT JOIN p USING (month, user_id, product_id) GROUP BY 1 ORDER BY 1""")
    since = end - pd.Timedelta(days=30)
    con.execute(f"""
    CREATE TABLE ab AS
    WITH c AS (SELECT user_id, product_id, brand, MAX(price) AS price, MAX(event_time) AS carted_at
               FROM ev WHERE event_type='cart' AND price > 0 AND event_time >= TIMESTAMP '{since}' GROUP BY 1, 2, 3),
         p AS (SELECT user_id, product_id, MAX(event_time) AS bought_at FROM pur GROUP BY 1, 2)
    SELECT c.* FROM c LEFT JOIN p ON p.user_id = c.user_id AND p.product_id = c.product_id AND p.bought_at >= c.carted_at
    WHERE p.user_id IS NULL""")
    T["abandoners"] = q("""SELECT user_id, COUNT(*) AS items, SUM(price) AS cart_value, MAX(carted_at) AS last_cart,
                                  arg_max(brand, price) AS top_brand FROM ab GROUP BY 1""")
    T["abandoned_products"] = q("""SELECT product_id, ANY_VALUE(brand) AS brand, COUNT(*) AS users, SUM(price) AS value
                                   FROM ab GROUP BY 1 ORDER BY value DESC LIMIT 200""")
    T["purchases_light"] = q("SELECT user_id, event_time FROM pur")

    # ----------------------------------------------------------- intent tables ----
    step(0.66, "Reconstructing sessions and the eligibility funnel")
    con.execute("""
    CREATE TABLE sess AS
    SELECT session_id, ANY_VALUE(user_id) AS user_id, COUNT(*) AS n_events, MIN(event_time) AS session_start,
           MAX(event_time) AS session_end, MAX(event_time) FILTER (WHERE pos = 3) AS prediction_time,
           MAX(CASE WHEN event_type='purchase' THEN 1 ELSE 0 END) AS has_purchase,
           MAX(CASE WHEN event_type='purchase' AND pos <= 3 THEN 1 ELSE 0 END) AS early_purchase,
           MAX(CASE WHEN event_type='purchase' AND pos > 3 THEN 1 ELSE 0 END) AS future_purchase
    FROM ev GROUP BY 1""")
    con.execute("""CREATE TABLE sf AS SELECT s.*, s.session_end <= (SELECT MAX(session_end) FROM sess)
                   - INTERVAL 30 MINUTE AS complete FROM sess s""")
    s = q("""SELECT COUNT(*) AS sessions, MEDIAN(n_events) AS median_events, AVG(has_purchase) AS purchase_rate FROM sess""").iloc[0]
    sessions_info = {"total": int(s.sessions), "median_events": float(s.median_events),
                     "purchase_rate": float(s.purchase_rate)}
    funnel = q("""
    SELECT 'All sessions' AS step, COUNT(*) AS sessions, SUM(has_purchase) AS purchasing_sessions FROM sf
    UNION ALL SELECT 'At least 3 events', COUNT(*), SUM(has_purchase) FROM sf WHERE n_events >= 3
    UNION ALL SELECT 'No purchase in first 3 events', COUNT(*), SUM(has_purchase) FROM sf
              WHERE n_events >= 3 AND early_purchase = 0
    UNION ALL SELECT 'Complete sessions (scored)', COUNT(*), SUM(has_purchase) FROM sf
              WHERE n_events >= 3 AND early_purchase = 0 AND complete""")
    funnel["share_of_purchasing_sessions"] = funnel.purchasing_sessions / max(funnel.purchasing_sessions[0], 1)
    T["funnel"] = funnel
    session_length = q("""SELECT LEAST(n_events, 20) AS n_events, COUNT(*) AS sessions, AVG(has_purchase) AS purchase_rate
                          FROM sess GROUP BY 1 ORDER BY 1""")
    T["session_length"] = session_length

    step(0.74, "Computing features from the first three events")
    cat_expr = ("COUNT(DISTINCT category)" if meta["has_category"] else "1")
    data = q(f"""
    WITH f AS (
        SELECT session_id,
               MAX(CASE WHEN pos = 1 THEN event_type END) AS action_1,
               MAX(CASE WHEN pos = 2 THEN event_type END) AS action_2,
               MAX(CASE WHEN pos = 3 THEN event_type END) AS action_3,
               COUNT(*) FILTER (WHERE event_type = 'cart') AS n_cart,
               COUNT(*) FILTER (WHERE event_type = 'remove_from_cart') AS n_remove,
               AVG(price) AS avg_price, COUNT(DISTINCT product_id) AS distinct_products,
               {cat_expr} AS distinct_categories,
               DATE_DIFF('second', MAX(event_time) FILTER (WHERE pos = 1), MAX(event_time) FILTER (WHERE pos = 2)) AS gap_1_2,
               DATE_DIFF('second', MAX(event_time) FILTER (WHERE pos = 2), MAX(event_time) FILTER (WHERE pos = 3)) AS gap_2_3,
               DATE_DIFF('second', MIN(event_time), MAX(event_time)) AS session_duration_3
        FROM ev WHERE pos <= 3 GROUP BY 1)
    SELECT e.session_id, e.user_id, e.session_start, e.prediction_time, e.session_end,
           f.action_1, f.action_2, f.action_3, f.action_1 || ' > ' || f.action_2 || ' > ' || f.action_3 AS sequence,
           f.n_cart, f.n_remove, f.avg_price, f.distinct_products, f.distinct_categories,
           f.gap_1_2, f.gap_2_3, f.session_duration_3, e.future_purchase
    FROM sf e JOIN f ON f.session_id = e.session_id
    WHERE e.n_events >= 3 AND e.early_purchase = 0 AND e.complete
    ORDER BY e.prediction_time, e.session_id""")
    con.close()
    for c in ["session_start", "prediction_time", "session_end"]:
        data[c] = pd.to_datetime(data[c])
    meta["n_scored_sessions"] = len(data)

    ws = {"meta": meta, "tables": T, "data": data, "sessions_info": sessions_info, "memo": {}}
    step(0.82, "Scoring sessions with the SmartCart purchase-intent model")
    apply_intent(ws, booster, model_name="SmartCart pre-trained (20.7M events)")
    step(1.0, "Workspace ready")
    return ws


# ------------------------------------------------------------------ intent --------
def apply_intent(ws, booster, model_name):
    """(Re)score the eligible sessions and rebuild all intent tables."""
    T, data = ws["tables"], ws["data"].copy()
    ws["memo"] = {}
    K = {"sessions": ws["sessions_info"], "raw_events": ws["meta"]["n_rows_uploaded"],
         "clean_events": ws["meta"]["n_events"]}
    if len(data) < 30:
        ws["intent_available"] = False
        K["dataset"] = {"sessions": len(data), "buyers": int(data.future_purchase.sum()) if len(data) else 0,
                        "purchase_rate": float(data.future_purchase.mean()) if len(data) else 0.0}
        ws["kpis"] = K
        return ws
    ws["intent_available"] = True
    ws["model_name"] = model_name
    ws["booster"] = booster
    y = data.future_purchase.values
    proba = predict(booster, encode(data))
    data["score"] = proba
    K["dataset"] = {"sessions": len(data), "buyers": int(y.sum()), "purchase_rate": float(y.mean())}

    def conversion(col):
        return (data.groupby(col, observed=True)["future_purchase"].agg(sessions="size", purchase_rate="mean")
                .reset_index())

    T["sequences"] = conversion("sequence").sort_values("purchase_rate", ascending=False)
    data["has_cart"] = (data.n_cart > 0).astype(int)
    T["by_cart"] = conversion("has_cart")
    T["by_remove"] = conversion("n_remove")
    T["by_ncart"] = conversion("n_cart")
    for i in (1, 2, 3):
        T[f"by_action_{i}"] = conversion(f"action_{i}")
    tab = pd.crosstab(data.has_cart, data.future_purchase).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    try:
        chi2, p_value, _, _ = stats.chi2_contingency(tab)
    except ValueError:
        chi2, p_value = np.nan, np.nan
    orr = (tab.loc[1, 1] * tab.loc[0, 0]) / max(tab.loc[1, 0] * tab.loc[0, 1], 1)
    K["cart_test"] = {"odds_ratio": float(orr), "chi2": float(chi2), "p_value": float(p_value)}

    bins = [-1, 30, 60, 120, 300, 600, 1800, np.inf]
    labels = ["<30s", "30s-1min", "1-2min", "2-5min", "5-10min", "10-30min", ">30min"]
    data["duration_group"] = pd.cut(data.session_duration_3, bins=bins, labels=labels).astype(str)
    T["duration"] = (data.groupby(["duration_group", "has_cart"])["future_purchase"]
                     .agg(sessions="size", purchase_rate="mean").reset_index())
    pb = [-0.01, 2, 4, 6, 10, 20, 50, np.inf]
    pl = ["<$2", "$2-4", "$4-6", "$6-10", "$10-20", "$20-50", ">$50"]
    data["price_group"] = pd.cut(data.avg_price, bins=pb, labels=pl).astype(str)
    T["price"] = (data.groupby(["price_group", "has_cart"])["future_purchase"]
                  .agg(sessions="size", purchase_rate="mean").reset_index())
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
    T["sankey"] = pd.concat(links)
    T["sess_daily"] = (data.assign(day=data.prediction_time.dt.date).groupby("day")["future_purchase"]
                       .agg(sessions="size", purchase_rate="mean").reset_index())
    T["sess_hour_weekday"] = (data.assign(hour=data.prediction_time.dt.hour, weekday=data.prediction_time.dt.dayofweek)
                              .groupby(["weekday", "hour"])["future_purchase"].agg(sessions="size", purchase_rate="mean")
                              .reset_index())

    # Rank-based score segments on the company's own score distribution
    limits = np.quantile(proba, [0.4, 0.7, 0.9]).tolist()
    seg = np.select([proba >= limits[2], proba >= limits[1], proba >= limits[0]], SEGMENTS[:3], SEGMENTS[3])
    data["segment"] = seg
    rec = {"1. Top 10%": ("Facilitate checkout", "Highlight viewed products"),
           "2. Next 20%": ("Cart reminder", "Personalized recommendations"),
           "3. Middle 30%": ("Cart reminder", "Recommend similar products"),
           "4. Bottom 40%": ("Low-cost cart reminder", "Product discovery or no action")}
    data["action"] = [rec[s][0 if h else 1] for s, h in zip(data.segment, data.has_cart)]
    st_ = data.groupby("segment").agg(sessions=("future_purchase", "size"), purchase_rate=("future_purchase", "mean"),
                                      share_with_cart=("has_cart", "mean"), buyers=("future_purchase", "sum")).reset_index()
    st_["share_of_sessions"] = st_.sessions / len(data)
    st_["share_of_buyers"] = st_.buyers / max(y.sum(), 1)
    T["segments"] = st_
    T["segment_cart"] = data.groupby(["segment", "has_cart"]).agg(sessions=("future_purchase", "size"),
                                                                  purchase_rate=("future_purchase", "mean")).reset_index()
    T["actions"] = data.groupby(["segment", "has_cart", "action"]).agg(sessions=("future_purchase", "size"),
                                                                       purchase_rate=("future_purchase", "mean")).reset_index()
    T["sessions_sample"] = data.sample(min(5000, len(data)), random_state=7).sort_values("prediction_time")[
        ["session_id", "prediction_time", "action_1", "action_2", "action_3", "n_cart", "n_remove", "avg_price",
         "distinct_products", "distinct_categories", "gap_1_2", "gap_2_3", "session_duration_3", "future_purchase",
         "has_cart", "score", "segment", "action"]]

    # Model check on the company's data (outcomes are known historically)
    from sklearn.metrics import average_precision_score
    chk = {"model_name": model_name, "base_rate": float(y.mean())}
    if 0 < y.sum() < len(y):
        chk["pr_auc"] = float(average_precision_score(y, proba))
        d = pd.DataFrame({"y": y, "p": proba})
        d["decile"] = pd.qcut(pd.Series(proba).rank(method="first", ascending=False), 10, labels=range(1, 11)).astype(int)
        dt = d.groupby("decile").agg(sessions=("y", "size"), purchase_rate=("y", "mean")).reset_index()
        dt["lift"] = dt.purchase_rate / y.mean()
        dt["share_of_buyers"] = dt.purchase_rate * dt.sessions / y.sum()
        dt["cum_share_of_buyers"] = dt.share_of_buyers.cumsum()
        T["deciles_ws"] = dt
        order = np.argsort(-proba)
        cum = np.cumsum(y[order]) / y.sum()
        pts = np.linspace(0, len(order) - 1, 101).astype(int)
        T["gains_ws"] = pd.DataFrame({"share_targeted": np.linspace(0, 1, 101), "share_of_buyers": np.r_[0, cum[pts[1:]]]})
    K["model"] = {"segment_limits": limits, "check": chk}
    K["feature_stats"] = {c: {q: float(data[c].quantile(v)) for q, v in
                              [("p01", .01), ("p25", .25), ("median", .5), ("p75", .75), ("p99", .99)]}
                          for c in ["avg_price", "gap_1_2", "gap_2_3", "session_duration_3"]}

    # A/B test baselines (mid-intent segments)
    mid = data.segment.isin(["2. Next 20%", "3. Middle 30%"])
    n_days = max((data.prediction_time.max() - data.prediction_time.min()).days, 1)
    ab = data.loc[mid, ["user_id", "prediction_time"]].sort_values("prediction_time").drop_duplicates("user_id")
    ab = ab[ab.prediction_time <= data.prediction_time.max() - pd.Timedelta(days=7)]
    pl_ = T["purchases_light"]
    if len(ab):
        m = ab.merge(pl_, on="user_id", how="left")
        m["hit"] = (m.event_time > m.prediction_time) & (m.event_time <= m.prediction_time + pd.Timedelta(days=7))
        conv = m.groupby("user_id").hit.max().mean()
    else:
        conv = np.nan
    K["ab_test"] = {"base_rate": float(conv) if conv == conv else float(data.loc[mid, "future_purchase"].mean()),
                    "users_per_day": float(len(ab) / max(n_days - 7, 1)),
                    "session_purchase_rate": float(data.loc[mid, "future_purchase"].mean()),
                    "sessions_per_day": float(mid.sum() / n_days)}
    ws["kpis"] = K
    ws["data"] = data
    return ws


def fine_tune(ws, base_booster, n_estimators=200):
    """Re-train the SmartCart model on the company's own sessions (chronological 80/20 split)
    and compare it with the pre-trained model on the same hold-out period."""
    from sklearn.metrics import average_precision_score
    data = ws["data"].sort_values("prediction_time")
    cut = int(len(data) * 0.8)
    tr, te = data.iloc[:cut], data.iloc[cut:]
    if tr.future_purchase.sum() < 50 or te.future_purchase.sum() < 20:
        raise ValueError("Not enough purchasing sessions to fine-tune (need ≥ 50 in training and ≥ 20 in hold-out).")
    params = {**BEST_PARAMS, "n_estimators": n_estimators}
    model = xgb.XGBClassifier(**params, n_jobs=-1, random_state=42)
    model.fit(encode(tr), tr.future_purchase.values)
    b = model.get_booster()
    b.feature_names = FEATURES
    p_new = predict(b, encode(te))
    p_old = predict(base_booster, encode(te))
    yt = te.future_purchase.values
    return b, {"train_sessions": len(tr), "test_sessions": len(te), "test_from": str(te.prediction_time.min().date()),
               "base_rate": float(yt.mean()), "pr_auc_pretrained": float(average_precision_score(yt, p_old)),
               "pr_auc_finetuned": float(average_precision_score(yt, p_new))}
