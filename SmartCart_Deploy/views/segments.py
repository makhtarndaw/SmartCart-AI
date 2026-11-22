import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core import biz as B

from core import data as D
from core import engine as E
from core.theme import (ACTION_LABELS, AQUA, BLUE, INK_3, SEGMENT_COLORS, chart, footer, insight, kpi,
                        page_header, pct)

K = D.kpis()
limits = K["model"]["segment_limits"]

page_header("Purchase Intent AI", "Intent segments & next best actions",
            "Fixed probability thresholds do not fit a calibrated model whose scores rarely exceed 0.4. Visitors are "
            "ranked into four score groups computed on your own score distribution (top 10%, next 20%, middle 30%, bottom 40%).")

if not D.intent_available():
    st.warning("Not enough eligible sessions in this workspace to run the purchase-intent model (sessions with at "
               "least 3 events and no purchase in the first three). Upload more history or include view/cart events.")
    st.stop()

ck = K["model"].get("check", {})
if "pr_auc" in ck:
    dw = D.table("deciles_ws")
    q = st.columns(4)
    kpi(q[0], "Model in use", ck["model_name"].split(" (")[0], "change it in the Data Hub")
    kpi(q[1], "PR-AUC on your data", f"{ck['pr_auc']:.3f}", f"random targeting = {ck['base_rate']:.3f}")
    kpi(q[2], "Top-10% lift", f"×{dw.lift.iloc[0]:.2f}", f"{pct(dw.purchase_rate.iloc[0])} of these sessions buy")
    kpi(q[3], "Buyers in top 30%", pct(dw.cum_share_of_buyers.iloc[2], 0), "of all your buyers", accent=True)
    st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)

seg = D.table("segments").set_index("segment")
c = st.columns(4)
for col, s in zip(c, D.SEGMENTS):
    r = seg.loc[s]
    col.markdown(f"""
    <div class="card" style="border-top:5px solid {SEGMENT_COLORS[s]}">
      <div style="font-size:.74rem;font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:#475569">{s.split('. ')[1]} of scores</div>
      <div class="big" style="margin:.3rem 0 .1rem">{pct(r.purchase_rate)}</div>
      <p>purchase rate · {r.sessions:,.0f} sessions</p>
      <p style="margin-top:.5rem"><b>{pct(r.share_of_buyers, 0)}</b> of buyers · <b>{pct(r.share_with_cart, 0)}</b> with a cart</p>
    </div>""", unsafe_allow_html=True)

l, r = st.columns(2)
with l:
    sc = D.table("segment_cart")
    fig = go.Figure()
    for hc, name, color in [(1, "With cart addition", BLUE), (0, "No cart addition", AQUA)]:
        d = sc[sc.has_cart == hc].set_index("segment").reindex(D.SEGMENTS)
        fig.add_trace(go.Bar(x=[s.split(". ")[1] for s in D.SEGMENTS], y=d.purchase_rate, name=name,
                             marker=dict(color=color, cornerradius=4), customdata=d.sessions,
                             hovertemplate=name + "<br>%{x}: %{y:.1%} buy<br>%{customdata:,.0f} sessions<extra></extra>"))
    fig.add_hline(y=K["dataset"]["purchase_rate"], line=dict(color=INK_3, dash="dash"))
    fig.update_layout(title="Purchase rate by segment and cart behaviour", barmode="group",
                      yaxis=dict(tickformat=".0%"), bargroupgap=0.08)
    chart(fig, 380)
with r:
    fig = go.Figure()
    for s in D.SEGMENTS:
        fig.add_trace(go.Bar(y=["Sessions", "Buyers"], x=[seg.loc[s, "share_of_sessions"], seg.loc[s, "share_of_buyers"]],
                             name=s.split(". ")[1], orientation="h",
                             marker=dict(color=SEGMENT_COLORS[s], line=dict(color="white", width=2)),
                             hovertemplate=s.split(". ")[1] + "<br>%{x:.1%} of %{y}<extra></extra>"))
    fig.update_layout(title="Where the buyers are: share of sessions vs share of buyers", barmode="stack",
                      xaxis=dict(tickformat=".0%"), bargap=0.45)
    chart(fig, 380)

insight("The middle 30% is the most interesting group: ~44,000 cart sessions that may be abandoned (13% buy), a "
        "natural target for a <b>cart reminder</b>. The bottom 40% has almost no cart activity and buys at 4%: "
        "spend nothing there.")

st.markdown('<div class="section">The decision matrix</div><div class="section-sub">Action recommended for every '
            'combination of score segment × cart behaviour, and the reasoning behind it.</div>', unsafe_allow_html=True)
act = D.table("actions")
rows = []
for s in D.SEGMENTS:
    for hc in (1, 0):
        a = D.recommend_action(s, hc)
        m = act[(act.segment == s) & (act.has_cart == hc)]
        rows.append({"Segment": s.split(". ")[1], "Cart": "With cart" if hc else "No cart", "Action": a,
                     "Sessions": int(m.sessions.sum()) if len(m) else 0,
                     "Purchase rate": float(m.purchase_rate.iloc[0]) * 100 if len(m) else np.nan,
                     "Cost": D.ACTION_PLAYBOOK[a][1], "Why": D.ACTION_PLAYBOOK[a][0]})
import pandas as pd  # noqa: E402

mat = pd.DataFrame(rows)
st.dataframe(mat, hide_index=True, use_container_width=True,
             column_config={"Purchase rate": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=30),
                            "Sessions": st.column_config.NumberColumn(format="%d"),
                            "Why": st.column_config.TextColumn(width="large")})
mat_disp = mat.copy()

l, r = st.columns([1, 1])
with l:
    agg = act.groupby("action").sessions.sum().sort_values()
    fig = go.Figure(go.Bar(y=agg.index, x=agg.values, orientation="h", marker=dict(color=BLUE, cornerradius=4),
                           text=[f"{v / agg.sum():.0%}" for v in agg.values], textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}<br>%{x:,.0f} sessions<extra></extra>"))
    fig.update_layout(title="Distribution of recommended actions", xaxis=dict(title="Sessions"),
                      margin=dict(l=10, r=40, t=56, b=10))
    chart(fig, 360)
with r:
    insight("<b>No discounts for the top 10%.</b> They buy in almost a quarter of cases and nearly all have a cart: a "
            "discount would mostly pay customers who would have bought anyway. Only frictionless checkout.")
    insight("<b>Match the action to the behaviour.</b> Hillstrom shows a behaviour-matched message beats a generic one: "
            "cart → cart reminder; browsing → recommendations based on viewed products.")
    insight("<b>Predict ≠ persuade.</b> A high score means likely to buy, not likely to react. Only the A/B test "
            "measures incremental conversion.", kind="warn")

st.markdown('<div class="section">Session explorer</div><div class="section-sub">Up to 5,000 real sessions of your data '
            'scored by the model. Filter and sort to see who lands where.</div>', unsafe_allow_html=True)
sample = D.parquet("sessions_sample").copy()
f1, f2, f3 = st.columns([1.2, 1.2, 1])
segs = f1.multiselect("Segment", D.SEGMENTS, default=D.SEGMENTS, format_func=lambda s: s.split(". ")[1])
acts = f2.multiselect("Recommended action", sorted(sample.action.unique()), default=[])
only = f3.selectbox("Outcome", ["All", "Bought", "Did not buy"])
v = sample[sample.segment.isin(segs)]
if acts:
    v = v[v.action.isin(acts)]
if only != "All":
    v = v[v.future_purchase == (1 if only == "Bought" else 0)]
short = {"view": "view", "cart": "cart", "remove_from_cart": "remove"}
v = v.assign(journey=v.action_1.map(short) + " → " + v.action_2.map(short) + " → " + v.action_3.map(short),
             segment=v.segment.str.split(". ").str[1], bought=np.where(v.future_purchase == 1, "Yes", "No"))
c = st.columns(3)
kpi(c[0], "Sessions shown", f"{len(v):,}")
kpi(c[1], "Real purchase rate", pct(v.future_purchase.mean()) if len(v) else "n/a")
kpi(c[2], "Mean predicted score", pct(v.score.mean()) if len(v) else "n/a")
st.dataframe(
    v.sort_values("score", ascending=False)[["prediction_time", "journey", "avg_price", "session_duration_3", "score",
                                             "segment", "action", "bought"]],
    hide_index=True, use_container_width=True, height=380,
    column_config={"prediction_time": st.column_config.DatetimeColumn("Time", format="D MMM, HH:mm"),
                   "journey": "First 3 actions", "avg_price": st.column_config.NumberColumn("Avg price", format=B.cfmt("$%.2f")),
                   "session_duration_3": st.column_config.NumberColumn("Duration (s)", format="%d"),
                   "score": st.column_config.ProgressColumn("Score", format="%.3f", min_value=0, max_value=0.5),
                   "segment": "Segment", "action": "Next best action", "bought": "Bought"})

# ----------------------------------------------------- fine-tuning on the company's data ---
if D.intent_available():
    with st.expander("Purchase-intent model: pre-trained vs fine-tuned on your data", icon=":material/model_training:", expanded=False):
        st.markdown("The SmartCart model was trained on **20.7M events** of a cosmetics e-shop. You can re-train it on "
                    "your own sessions (80% oldest for training, 20% most recent for testing) and keep whichever "
                    "ranks your visitors better on the test period.")
        W = D.ws()
        ck = W["kpis"]["model"].get("check", {})
        if "pr_auc" in ck:
            st.markdown(f"Current model on your data: **PR-AUC {ck['pr_auc']:.3f}** vs {ck['base_rate']:.3f} for "
                        f"random targeting (×{ck['pr_auc'] / ck['base_rate']:.1f}).")
        if st.button("Train a model on my data and compare"):
            with st.spinner("Training on your sessions…"):
                try:
                    st.session_state.finetune = E.fine_tune(W, D.pretrained_booster())
                except ValueError as exc:
                    st.warning(str(exc))
        ft = st.session_state.get("finetune")
        if ft:
            b_new, res = ft
            c = st.columns(3)
            kpi(c[0], "Pre-trained SmartCart", f"{res['pr_auc_pretrained']:.3f}", "PR-AUC on your test period")
            kpi(c[1], "Fine-tuned on your data", f"{res['pr_auc_finetuned']:.3f}", "PR-AUC on your test period")
            kpi(c[2], "Random targeting", f"{res['base_rate']:.3f}", f"test from {res['test_from']} · "
                f"{res['test_sessions']:,} sessions")
            better = res["pr_auc_finetuned"] > res["pr_auc_pretrained"]
            insight("The <b>fine-tuned</b> model ranks your visitors better. → You can switch to it." if better else
                    "The <b>pre-trained SmartCart model</b> ranks your visitors better than a model trained on your data "
                    "alone: the 20.7M-event training brings knowledge your history does not contain yet. Keep it.")
            if better and st.button("Use the fine-tuned model", type="primary"):
                E.apply_intent(W, b_new, model_name="Fine-tuned on your data")
                st.session_state.pop("finetune", None)
                st.rerun()


footer()
