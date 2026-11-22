"""Workspace management + loaders.

Every analytics module reads the *current workspace*: the company's own data processed by the
SmartCart engine (or the demo workspace until a company connects its data). The model card
pages read the static artifacts of the SmartCart model (trained on 20.7M REES46 events)."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

from core import engine as E

ART = Path(__file__).resolve().parent.parent / "artifacts"
DEMO = ART / "demo" / "demo_events.parquet"

CATEGORICAL, ACTIONS, NUMERICAL, FEATURES, SEGMENTS = E.CATEGORICAL, E.ACTIONS, E.NUMERICAL, E.FEATURES, E.SEGMENTS

FEATURE_LABELS = {
    "n_cart": "Cart additions", "n_remove": "Cart removals", "avg_price": "Average price",
    "distinct_products": "Distinct products", "distinct_categories": "Distinct categories",
    "gap_1_2": "Time 1→2 (s)", "gap_2_3": "Time 2→3 (s)", "session_duration_3": "Duration of 3 events (s)",
}
for _c in CATEGORICAL:
    for _a, _l in [("cart", "cart"), ("remove_from_cart", "remove"), ("view", "view")]:
        FEATURE_LABELS[f"{_c}_{_a}"] = f"Action {_c[-1]} = {_l}"

ACTION_PLAYBOOK = {
    "Facilitate checkout": ("Remove friction: express checkout, saved payment details, visible free-shipping "
                            "threshold. No discount: most of these visitors would buy anyway.", "Very low"),
    "Highlight viewed products": ("Bring back the products already seen (recently-viewed carousel).", "Very low"),
    "Cart reminder": ("On-site or e-mail reminder of the products left in the cart. Matches the observed "
                      "behaviour, the most promising action.", "Low"),
    "Personalized recommendations": ("Recommendations based on the products viewed, a behaviour-matched "
                                     "message beats a generic one (Hillstrom lesson).", "Low"),
    "Recommend similar products": ("Show alternatives to the products viewed to help an undecided visitor "
                                   "choose.", "Low"),
    "Low-cost cart reminder": ("Rare case of a cart in a low-score session: automated, near-zero-cost "
                               "reminder only.", "Very low"),
    "Product discovery or no action": ("Very low purchase probability: inspiration content at most. Do not "
                                       "spend marketing budget here.", "None"),
}


# ------------------------------------------------------------- SmartCart model ----
@st.cache_resource
def pretrained_booster():
    b = xgb.Booster()
    b.load_model(str(ART / "model.json"))
    return b


@st.cache_data
def model_card_kpis():
    return json.loads((ART / "kpis.json").read_text())


@st.cache_data
def static_table(name):
    p = ART / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else None


@st.cache_data
def static_parquet(name):
    return pd.read_parquet(ART / f"{name}.parquet")


# ------------------------------------------------------------------ workspace -----
@st.cache_resource(show_spinner="Preparing the demo workspace…")
def demo_workspace():
    raw = pd.read_parquet(DEMO)
    m = E.guess_mapping(raw.columns)
    emap = {v: E.guess_event(v) for v in raw.event_type.unique()}
    ws = E.build_workspace(E.standardize(raw, m, emap), pretrained_booster(), name="Demo · Cosmetics e-shop")
    ws["is_demo"] = True
    ws["id"] = "demo"
    return ws


def ws():
    w = st.session_state.get("workspace")
    return w if w is not None else demo_workspace()


def set_workspace(w):
    st.session_state["workspace"] = w


def is_demo():
    return ws().get("is_demo", False)


def memo(key, fn):
    m = ws().setdefault("memo", {})
    if key not in m:
        m[key] = fn()
    return m[key]


# ------------------------------------------------------------------ loaders -------
def table(name):
    """Workspace table first, SmartCart model-card artifact otherwise."""
    t = ws()["tables"].get(name)
    return t if t is not None else static_table(name)


def parquet(name):
    t = ws()["tables"].get(name)
    return t if t is not None else static_parquet(name)


def kpis():
    """Model-card KPIs overlaid with the workspace KPIs."""
    k = copy.deepcopy(model_card_kpis())
    w = ws().get("kpis", {})
    for key, val in w.items():
        if isinstance(val, dict) and isinstance(k.get(key), dict):
            k[key].update(val)
        else:
            k[key] = val
    return k


def intent_available():
    return ws().get("intent_available", False)


def booster():
    return ws().get("booster") or pretrained_booster()


def encode_row(a1, a2, a3, avg_price, distinct_products, distinct_categories, gap_1_2, gap_2_3):
    acts = [a1, a2, a3]
    row = {}
    for c, a in zip(CATEGORICAL, acts):
        for x in ACTIONS:
            row[f"{c}_{x}"] = float(a == x)
    row.update({
        "n_cart": float(acts.count("cart")), "n_remove": float(acts.count("remove_from_cart")),
        "avg_price": float(avg_price), "distinct_products": float(distinct_products),
        "distinct_categories": float(distinct_categories), "gap_1_2": float(gap_1_2),
        "gap_2_3": float(gap_2_3), "session_duration_3": float(gap_1_2 + gap_2_3),
    })
    return np.array([[row[f] for f in FEATURES]])


def score(X):
    """Return (probabilities, SHAP contributions in log-odds, base value) with the active model."""
    b = booster()
    d = xgb.DMatrix(X, feature_names=FEATURES)
    contrib = b.predict(d, pred_contribs=True)
    return b.predict(d), contrib[:, :-1], contrib[:, -1]


def segment_of(p, limits):
    if p >= limits[2]:
        return SEGMENTS[0]
    if p >= limits[1]:
        return SEGMENTS[1]
    if p >= limits[0]:
        return SEGMENTS[2]
    return SEGMENTS[3]


def recommend_action(seg, has_cart):
    return {
        "1. Top 10%": ("Facilitate checkout", "Highlight viewed products"),
        "2. Next 20%": ("Cart reminder", "Personalized recommendations"),
        "3. Middle 30%": ("Cart reminder", "Recommend similar products"),
        "4. Bottom 40%": ("Low-cost cart reminder", "Product discovery or no action"),
    }[seg][0 if has_cart else 1]
