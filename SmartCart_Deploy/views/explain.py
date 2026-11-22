import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core import data as D
from core.theme import (AQUA, BLUE, INK_3, NEUTRAL, ORANGE, chart, footer, insight, kpi, page_header, pct,
                        short_seq)

K = D.kpis()

page_header("Purchase Intent AI · Model card", "Explainability",
            "Why does the model score a session high or low? Exact TreeSHAP values on 2,000 February sessions, "
            "and an analysis of where and why the model is wrong.")

tab1, tab2 = st.tabs(["SHAP: what drives the score", "Error analysis: who is misclassified"])

with tab1:
    imp = D.table("shap_importance")
    sv = D.parquet("shap_values")
    fv = D.parquet("shap_features")
    top_n = st.slider("Features shown", 6, 17, 12)
    top = imp.head(top_n).feature.tolist()[::-1]

    l, r = st.columns([1, 1.5])
    with l:
        s = imp.head(top_n).iloc[::-1]
        fig = go.Figure(go.Bar(y=[D.FEATURE_LABELS[f] for f in s.feature], x=s.mean_abs_shap, orientation="h",
                               marker=dict(color=BLUE, cornerradius=4),
                               hovertemplate="%{y}<br>mean |SHAP| %{x:.3f}<extra></extra>"))
        fig.update_layout(title="Global importance (mean |SHAP|, log-odds)", xaxis=dict(title=None),
                          margin=dict(l=10, r=10, t=56, b=10))
        chart(fig, 30 * top_n + 120)
    with r:
        fig = go.Figure()
        rng = np.random.RandomState(0)
        for i, f in enumerate(top):
            x = sv[f].values
            v = fv[f].values
            lo, hi = np.nanpercentile(v, 5), np.nanpercentile(v, 95)
            norm = np.clip((v - lo) / (hi - lo + 1e-9), 0, 1)
            fig.add_trace(go.Scatter(
                x=x, y=i + rng.uniform(-0.28, 0.28, len(x)), mode="markers",
                marker=dict(size=5, color=norm, colorscale=[[0, AQUA], [0.5, "#E2E8F0"], [1, ORANGE]], opacity=0.75,
                            showscale=(i == 0), colorbar=dict(title="Feature<br>value", tickvals=[0, 1],
                                                              ticktext=["low", "high"], thickness=10, outlinewidth=0)),
                customdata=v, name=D.FEATURE_LABELS[f], showlegend=False,
                hovertemplate=D.FEATURE_LABELS[f] + "<br>value %{customdata:.2f}<br>SHAP %{x:.3f}<extra></extra>"))
        fig.add_vline(x=0, line=dict(color=INK_3, width=1))
        fig.update_layout(title="SHAP summary: each dot is one session",
                          yaxis=dict(tickvals=list(range(len(top))), ticktext=[D.FEATURE_LABELS[f] for f in top],
                                     showgrid=False),
                          xaxis=dict(title="Impact on the score (log-odds) ← lowers · raises →"))
        chart(fig, 30 * top_n + 120)

    insight("<b>Cart additions dominate:</b> zero cart additions pushes the score down; one or more pushes it up. "
            "Removals act the same way. High prices and long gaps between events lower the score.")

    a3 = D.table("shap_action3")
    cols = st.columns(3)
    rates = D.static_table("by_action_3").set_index("action_3").purchase_rate
    for col, (_, row) in zip(cols, a3.iterrows()):
        kpi(col, f"3rd action = {row.action_3.replace('remove_from_cart', 'remove')}",
            f"{row.mean_shap:+.3f}", f"mean SHAP of “action 3 = view” · purchase rate {pct(rates[row.action_3])}")
    insight("Why do the <i>view</i> variables look negative? With one-hot encoding, “action 3 = view” really means "
            "<b>“no cart action at step 3”</b>. Everyone views products, so a view on its own carries little "
            "information: the model reads it as the absence of a cart signal.")

with tab2:
    ep = D.table("error_profile").set_index("result")
    order = ["True positive", "False positive", "False negative", "True negative"]
    ep = ep.reindex(order)
    c = st.columns(4)
    for col, r in zip(c, order):
        kpi(col, r, f"{ep.loc[r, 'sessions']:,.0f}",
            f"{pct(ep.loc[r, 'share_with_cart'], 0)} with a cart · {ep.loc[r, 'avg_duration']:.0f}s",
            accent=(r == "False positive"))

    fig = go.Figure()
    metrics = [("share_with_cart", "Share with a cart addition", ".0%"), ("avg_removals", "Avg. removals", ".2f")]
    colors = {"True positive": "#0d366b", "False positive": BLUE, "False negative": ORANGE, "True negative": NEUTRAL}
    l, r = st.columns(2)
    for col, (m, title, fmt) in zip([l, r], metrics):
        fig = go.Figure(go.Bar(x=order, y=ep[m], marker=dict(color=[colors[o] for o in order], cornerradius=4),
                               text=[format(v, fmt) for v in ep[m]], textposition="outside", cliponaxis=False,
                               hovertemplate="%{x}<br>" + title + " %{y:" + fmt + "}<extra></extra>"))
        fig.update_layout(title=title + " by prediction outcome", yaxis=dict(tickformat=fmt))
        with col:
            chart(fig, 330)

    es = D.table("error_sequences")
    l, r = st.columns(2)
    for col, res, color in [(l, "False positive", BLUE), (r, "False negative", ORANGE)]:
        d = es[es.result == res].iloc[::-1]
        fig = go.Figure(go.Bar(y=d.sequence.map(short_seq), x=d.share, orientation="h",
                               marker=dict(color=color, cornerradius=4), text=[f"{v:.0%}" for v in d.share],
                               textposition="outside", cliponaxis=False,
                               hovertemplate="%{y}<br>%{x:.1%} of " + res.lower() + "s<extra></extra>"))
        fig.update_layout(title=f"Most frequent sequences · {res.lower()}s", xaxis=dict(tickformat=".0%"),
                          margin=dict(l=10, r=40, t=56, b=10))
        with col:
            chart(fig, 360)

    l, r = st.columns(2)
    with l:
        insight("<b>False positives are abandoned carts.</b> 85% contain a cart addition and their profile is "
                "almost identical to true positives. From three events, nothing separates a cart that will be paid "
                "from one that will be abandoned, and these are exactly the visitors a <b>cart reminder</b> should target.")
    with r:
        insight("<b>False negatives are late deciders.</b> 41% are <i>view → view → view</i>: they browsed first and "
                "decided later. Catching them would require waiting for more events, which leaves less time to act.",
                kind="warn")

footer()
