import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core import biz as B

from core import data as D
from core.theme import (ACTION_LABELS, BLUE, INK, INK_3, ORANGE, SEGMENT_COLORS, chart, footer, insight,
                        journey_pills, kpi, page_header, pct)

K = D.kpis()
M = K["model"]
limits = M["segment_limits"]
FS = K["feature_stats"]
ACTS = ["view", "cart", "remove_from_cart"]

page_header("Purchase Intent AI", "Live intent scoring",
            "Describe the first three actions of a visitor — or load a real session from your data — and get the "
            "purchase probability, the score segment, the next best action and the reasons behind the score.")

if not D.intent_available():
    st.warning("Not enough eligible sessions in this workspace to run the purchase-intent model (sessions with at "
               "least 3 events and no purchase in the first three). Upload more history or include view/cart events.")
    st.stop()

sample = D.parquet("sessions_sample")

# ------------------------------------------------------------------- presets -----
PRESETS = {
    "Pure browser": ("view", "view", "view", 8.0, 3, 2, 40, 45),
    "Quick cart": ("view", "cart", "cart", 6.0, 2, 2, 25, 20),
    "Cart manager (returning)": ("cart", "remove_from_cart", "cart", 5.0, 2, 1, 15, 30),
    "Hesitant, expensive": ("view", "view", "cart", 45.0, 3, 3, 400, 600),
}

if "sim" not in st.session_state:
    st.session_state.sim = PRESETS["Quick cart"]
    st.session_state.truth = None


def load(values, truth=None):
    st.session_state.sim = values
    st.session_state.truth = truth
    for k in ["a1", "a2", "a3", "price", "dp", "dc", "g12", "g23"]:
        st.session_state.pop(k, None)


st.markdown("**Start from a profile**")
pc = st.columns(len(PRESETS) + 1)
for col, (name, vals) in zip(pc, PRESETS.items()):
    if col.button(name, use_container_width=True):
        load(vals)
if pc[-1].button("Real session from your data", icon=":material/shuffle:", use_container_width=True, type="primary"):
    r = sample.sample(1).iloc[0]
    load((r.action_1, r.action_2, r.action_3, round(float(r.avg_price), 2), int(r.distinct_products),
          int(r.distinct_categories), int(r.gap_1_2), int(r.gap_2_3)), truth=dict(r))

a1, a2, a3, price, dp, dc, g12, g23 = st.session_state.sim

left, right = st.columns([1, 1.35], gap="large")
with left:
    with st.container(border=True):
        st.markdown("##### Visitor's first three actions")
        cols = st.columns(3)
        fmt = lambda v: ACTION_LABELS[v]
        a1 = cols[0].selectbox("Action 1", ACTS, ACTS.index(a1), format_func=fmt, key="a1")
        a2 = cols[1].selectbox("Action 2", ACTS, ACTS.index(a2), format_func=fmt, key="a2")
        a3 = cols[2].selectbox("Action 3", ACTS, ACTS.index(a3), format_func=fmt, key="a3")
        st.markdown(journey_pills([a1, a2, a3]), unsafe_allow_html=True)
        st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
        price = st.slider(B.cfmt("Average product price ($)"), 0.5, 100.0, float(min(max(price, 0.5), 100.0)), 0.5, key="price")
        cols = st.columns(2)
        dp = cols[0].number_input("Distinct products", 1, 3, int(min(max(dp, 1), 3)), key="dp")
        dc = cols[1].number_input("Distinct categories", 1, 3, int(min(max(dc, 1), 3)), key="dc")
        cols = st.columns(2)
        g12 = cols[0].number_input("Seconds between action 1 → 2", 0, 1800, int(min(g12, 1800)), 5, key="g12")
        g23 = cols[1].number_input("Seconds between action 2 → 3", 0, 1800, int(min(g23, 1800)), 5, key="g23")
        st.caption(f"Typical values: median price {B.cur()}{FS['avg_price']['median']:.1f}, median gap "
                   f"{FS['gap_1_2']['median']:.0f}s / {FS['gap_2_3']['median']:.0f}s.")

X = D.encode_row(a1, a2, a3, price, dp, dc, g12, g23)
proba, contrib, base = D.score(X)
p = float(proba[0])
seg = D.segment_of(p, limits)
has_cart = "cart" in (a1, a2, a3)
action = D.recommend_action(seg, has_cart)
avg = K["dataset"]["purchase_rate"]

with right:
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta", value=p * 100,
        number=dict(suffix="%", font=dict(size=46, color=INK), valueformat=".1f"),
        delta=dict(reference=avg * 100, valueformat=".1f", suffix=" pts vs avg",
                   increasing=dict(color="#0ca30c"), decreasing=dict(color="#d03b3b")),
        gauge=dict(axis=dict(range=[0, 40], ticksuffix="%", tickcolor=INK_3),
                   bar=dict(color=SEGMENT_COLORS[seg], thickness=0.32),
                   bgcolor="white", borderwidth=0,
                   steps=[dict(range=[0, limits[0] * 100], color="#F1F5F9"),
                          dict(range=[limits[0] * 100, limits[1] * 100], color="#E2ECFA"),
                          dict(range=[limits[1] * 100, limits[2] * 100], color="#CFE0F7"),
                          dict(range=[limits[2] * 100, 40], color="#B7D3F6")],
                   threshold=dict(line=dict(color=ORANGE, width=3), thickness=0.8, value=avg * 100)),
        title=dict(text="Probability of a purchase later in this session", font=dict(size=14, color="#475569")),
    ))
    fig.update_layout(margin=dict(l=30, r=30, t=60, b=10))
    chart(fig, 290)

    c = st.columns(2)
    kpi(c[0], "Score segment", seg.split(". ")[1], f"segment limits {', '.join(f'{v:.3f}' for v in limits)}",
        accent=True)
    kpi(c[1], "Next best action", action, f"Cost: {D.ACTION_PLAYBOOK[action][1]}")
    insight(f"<b>{action}.</b> {D.ACTION_PLAYBOOK[action][0]}")

    truth = st.session_state.get("truth")
    if truth is not None and (a1, a2, a3) == (truth["action_1"], truth["action_2"], truth["action_3"]):
        outcome = "**bought** later in the session" if truth["future_purchase"] == 1 else "**did not buy**"
        st.info(f"Real session **{truth['session_id']}** ({truth['prediction_time']:%d %b %Y %H:%M}) — "
                f"this visitor {outcome}.")

st.markdown("##### Why this score? — SHAP explanation of this session")
c = contrib[0]
order = np.argsort(-np.abs(c))
keep = order[:8]
rest = c[order[8:]].sum()
labels = [D.FEATURE_LABELS[D.FEATURES[i]] + f" = {X[0, i]:g}" for i in keep] + ["Other features"]
vals = list(c[keep]) + [rest]
fig = go.Figure(go.Waterfall(
    orientation="h", y=["Average visitor"] + labels + ["This session"],
    x=[base[0]] + vals + [0], measure=["absolute"] + ["relative"] * len(vals) + ["total"],
    increasing=dict(marker=dict(color=BLUE)), decreasing=dict(marker=dict(color=ORANGE)),
    totals=dict(marker=dict(color="#0d366b")), connector=dict(line=dict(color="#E3E7EF")),
    hovertemplate="%{y}<br>%{x:+.3f} log-odds<extra></extra>"))
fig.update_layout(yaxis=dict(autorange="reversed"), xaxis=dict(title="Log-odds of purchase  (blue raises · orange lowers)"),
                  margin=dict(l=10, r=10, t=20, b=10), showlegend=False)
chart(fig, 420)
st.caption(f"Base value = average log-odds of the model ({1 / (1 + np.exp(-base[0])):.1%}). Each bar adds the "
           f"contribution of one feature; the final bar converts to the probability {pct(p)}.")

footer()
