import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import data as D
from core.theme import (AQUA, BLUE, INK_3, NEUTRAL, ORANGE, SEQ_SCALE, chart, footer, insight, kpi,
                        page_header, pct, section)

K = D.kpis()
M = K["model"]
T = M["test"]

page_header("Purchase Intent AI · Model card", "Model performance",
            "Four gradient-boosting / tree models compared, the best tuned with Optuna, evaluated out-of-time on "
            "February 2020, a month the model never saw.")

c = st.columns(5)
kpi(c[0], "PR-AUC (test)", f"{T['pr_auc']:.3f}", f"random model = {T['base_rate']:.3f}", accent=True)
kpi(c[1], "Precision", pct(T["precision"]), f"at threshold {M['threshold']:.3f}")
kpi(c[2], "Recall", pct(T["recall"]), "share of buyers caught")
kpi(c[3], "F1-score", pct(T["f1"]), "chosen on validation")
kpi(c[4], "Calibration", f"{M['val']['mean_proba']:.3f}", f"mean score vs {M['val']['base_rate']:.3f} real rate")

tab1, tab2, tab3, tab4 = st.tabs(["Model comparison", "Threshold explorer", "Ranking & lift", "Stability"])

# -------------------------------------------------------------- comparison -------
with tab1:
    mc = D.table("model_comparison")
    l, r = st.columns([1.1, 1])
    with l:
        s = mc.sort_values("pr_auc")
        colors = [BLUE if "final" in m else "#9ec5f4" for m in s.model]
        fig = go.Figure(go.Bar(y=s.model, x=s.pr_auc, orientation="h", marker=dict(color=colors, cornerradius=4),
                               text=[f"{v:.3f}" for v in s.pr_auc], textposition="outside", cliponaxis=False,
                               hovertemplate="%{y}<br>PR-AUC %{x:.4f}<extra></extra>"))
        fig.add_vline(x=M["val"]["base_rate"], line=dict(color=INK_3, dash="dash"),
                      annotation_text=f"Random {M['val']['base_rate']:.3f}", annotation_position="top")
        fig.update_layout(title="PR-AUC on the validation month (January 2020)",
                          xaxis=dict(range=[0, 0.24], title=None), margin=dict(l=10, r=50, t=60, b=10))
        chart(fig, 340)
    with r:
        s = mc.sort_values("mean_proba")
        fig = go.Figure(go.Bar(y=s.model, x=s.mean_proba, orientation="h",
                               marker=dict(color=[BLUE if "final" in m else ORANGE for m in s.model], cornerradius=4),
                               text=[f"{v:.3f}" for v in s.mean_proba], textposition="outside", cliponaxis=False,
                               hovertemplate="%{y}<br>Mean predicted probability %{x:.3f}<extra></extra>"))
        fig.add_vline(x=M["val"]["base_rate"], line=dict(color=INK_3, dash="dash"),
                      annotation_text="Real purchase rate", annotation_position="top")
        fig.update_layout(title="Mean predicted probability (calibration)", xaxis=dict(range=[0, 0.55], title=None),
                          margin=dict(l=10, r=50, t=60, b=10))
        chart(fig, 340)
    disp = mc.rename(columns={"model": "Model", "pr_auc": "PR-AUC", "precision": "Precision @0.5",
                              "recall": "Recall @0.5", "f1": "F1 @0.5", "mean_proba": "Mean predicted prob."})
    st.dataframe(disp.style.format({c: "{:.3f}" for c in disp.columns if c != "Model"}, na_rep="n/a"),
                 hide_index=True, use_container_width=True)
    insight("The four models are within 0.004 PR-AUC of each other: the ceiling comes from the information in "
            "three events, not the algorithm. Class weighting inflated the scores (mean ≈ 0.46 vs 0.12 real): the "
            f"<b>final XGBoost</b> drops weighting, is tuned with 30 Optuna trials ({M['params']['n_estimators']} "
            f"trees, depth {M['params']['max_depth']}) and outputs <b>calibrated probabilities</b>.")

# ------------------------------------------------------------- threshold ---------
with tab2:
    g = D.table("threshold_grid")
    l, r = st.columns([1, 2.2])
    with l:
        which = st.radio("Evaluate on", ["test", "validation"], horizontal=True,
                         format_func=lambda v: "February (test)" if v == "test" else "January (validation)")
        t = st.slider("Decision threshold", 0.02, 0.40, float(round(M["threshold"] / 0.005) * 0.005), 0.005,
                      format="%.3f")
        row = g[(g.set == which) & (np.isclose(g.threshold, t))].iloc[0]
        tp, fp, fn, tn = row.tp, row.fp, row.fn, row.tn
        prec = tp / (tp + fp) if tp + fp else 0
        rec = tp / (tp + fn)
        flagged = (tp + fp) / (tp + fp + fn + tn)
        st.metric("Precision", pct(prec))
        st.metric("Recall", pct(rec))
        st.metric("Sessions flagged", pct(flagged))
        st.metric("Non-buyers flagged per buyer found", f"{fp / tp:.1f}" if tp else "n/a")
    with r:
        gg = g[g.set == which].copy()
        gg["precision"] = gg.tp / (gg.tp + gg.fp).replace(0, np.nan)
        gg["recall"] = gg.tp / (gg.tp + gg.fn)
        gg["f1"] = 2 * gg.tp / (2 * gg.tp + gg.fp + gg.fn)
        gg["flagged"] = (gg.tp + gg.fp) / (gg.tp + gg.fp + gg.fn + gg.tn)
        gg = gg[gg.threshold <= 0.4]
        fig = go.Figure()
        for col, name, color, dash in [("precision", "Precision", BLUE, None), ("recall", "Recall", ORANGE, None),
                                       ("f1", "F1-score", AQUA, None), ("flagged", "Share flagged", "#64748B", "dot")]:
            fig.add_trace(go.Scatter(x=gg.threshold, y=gg[col], name=name, mode="lines",
                                     line=dict(color=color, width=2.5, dash=dash),
                                     hovertemplate=name + " %{y:.1%}<extra></extra>"))
        fig.add_vline(x=t, line=dict(color="#0F172A", width=1.5))
        fig.add_vline(x=M["threshold"], line=dict(color=INK_3, dash="dash"),
                      annotation_text=f"Best F1 on validation ({M['threshold']:.3f})", annotation_position="top right")
        fig.update_layout(title="Precision / recall trade-off", hovermode="x unified",
                          xaxis=dict(title="Threshold", tickformat=".2f"), yaxis=dict(tickformat=".0%", range=[0, 1]))
        chart(fig, 400)

    l, r = st.columns(2)
    with l:
        z = np.array([[tn, fp], [fn, tp]])
        share = z / z.sum()
        fig = go.Figure(go.Heatmap(
            z=share, x=["Predicted: no purchase", "Predicted: purchase"], y=["Actual: no purchase", "Actual: purchase"],
            colorscale=SEQ_SCALE, showscale=False, xgap=3, ygap=3,
            text=[[f"{v:,.0f}<br>{s:.1%}" for v, s in zip(rz, rs)] for rz, rs in zip(z, share)],
            texttemplate="%{text}", textfont=dict(size=14),
            hovertemplate="%{y}<br>%{x}<br>%{text}<extra></extra>"))
        fig.update_layout(title=f"Confusion matrix at threshold {t:.3f}", yaxis=dict(autorange="reversed"))
        chart(fig, 340)
    with r:
        pr = D.table("pr_curve")
        fig = go.Figure(go.Scatter(x=pr.recall, y=pr.precision, mode="lines", line=dict(color=BLUE, width=2.5),
                                   fill="tozeroy", fillcolor="rgba(42,120,214,.08)", name="XGBoost",
                                   hovertemplate="Recall %{x:.1%}<br>Precision %{y:.1%}<extra></extra>"))
        fig.add_hline(y=T["base_rate"], line=dict(color=INK_3, dash="dash"),
                      annotation_text=f"Random model ({T['base_rate']:.3f})", annotation_position="bottom right")
        fig.update_layout(title=f"Precision–recall curve (test) · PR-AUC {T['pr_auc']:.3f}",
                          xaxis=dict(title="Recall", tickformat=".0%"), yaxis=dict(title="Precision", tickformat=".0%",
                                                                                   range=[0, 0.6]), showlegend=False)
        chart(fig, 340)
    insight("At 0.5 the model flags <b>nobody</b>: calibrated probabilities rarely exceed 0.4. Even at the best-F1 "
            f"threshold the model flags {pct(T['flagged'], 0)} of visitors, too broad for a campaign. That is why "
            "SmartCart AI <b>ranks</b> visitors instead of using a single cut-off.")

# --------------------------------------------------------------- ranking ---------
with tab3:
    dec = D.table("deciles_test")
    gains = D.table("gains")
    l, r = st.columns(2)
    with l:
        colors = [BLUE if d <= 5 else "#9ec5f4" for d in dec.decile]
        fig = go.Figure(go.Bar(x=dec.decile.astype(str), y=dec.purchase_rate, marker=dict(color=colors, cornerradius=4),
                               customdata=np.c_[dec.lift, dec.share_of_buyers],
                               text=[f"×{v:.1f}" for v in dec.lift], textposition="outside", cliponaxis=False,
                               hovertemplate="Decile %{x}<br>Purchase rate %{y:.1%}<br>Lift ×%{customdata[0]:.2f}"
                                             "<br>%{customdata[1]:.1%} of all buyers<extra></extra>"))
        fig.add_hline(y=T["base_rate"], line=dict(color=INK_3, dash="dash"),
                      annotation_text="Average", annotation_position="top right")
        fig.update_layout(title="Purchase rate by score decile (1 = highest scores)", yaxis=dict(tickformat=".0%"),
                          xaxis=dict(title="Score decile"))
        chart(fig, 380)
    with r:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=gains.share_targeted, y=gains.share_of_buyers, name="SmartCart AI",
                                 line=dict(color=BLUE, width=2.5), fill="tonexty",
                                 hovertemplate="Target top %{x:.0%}<br>→ %{y:.1%} of buyers<extra></extra>"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random targeting", line=dict(color=INK_3, dash="dash"),
                                 hoverinfo="skip"))
        fig.update_layout(title="Cumulative gains: share of buyers reached", hovermode="x unified",
                          xaxis=dict(title="Share of sessions targeted (by score)", tickformat=".0%"),
                          yaxis=dict(title="Share of buyers reached", tickformat=".0%"))
        chart(fig, 380)
    budget = st.slider("Campaign reach: share of visitors you can target", 0.05, 1.0, 0.20, 0.05, format="%.2f")
    reached = float(np.interp(budget, gains.share_targeted, gains.share_of_buyers))
    c = st.columns(3)
    kpi(c[0], "Visitors targeted", pct(budget, 0), "highest scores first")
    kpi(c[1], "Buyers reached", pct(reached, 0), f"vs {pct(budget, 0)} with random targeting")
    kpi(c[2], "Efficiency gain", f"×{reached / budget:.2f}", "buyers per contact vs random", accent=True)

# --------------------------------------------------------------- stability -------
with tab4:
    wk = D.table("weekly")
    wk["label"] = ["1–2 Feb", "3–9 Feb", "10–16 Feb", "17–23 Feb", "24 Feb–1 Mar"][:len(wk)]
    l, r = st.columns(2)
    with l:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=wk.label, y=wk.pr_auc, name="PR-AUC", mode="lines+markers",
                                 line=dict(color=BLUE, width=2.5), marker=dict(size=9, line=dict(color="white", width=2)),
                                 hovertemplate="%{x}<br>PR-AUC %{y:.3f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=wk.label, y=wk.purchase_rate, name="Random baseline (purchase rate)",
                                 mode="lines+markers", line=dict(color=INK_3, width=2, dash="dash"),
                                 marker=dict(size=8), hovertemplate="%{x}<br>Base rate %{y:.3f}<extra></extra>"))
        fig.update_layout(title="Weekly PR-AUC on February", yaxis=dict(range=[0, 0.25]), hovermode="x unified")
        chart(fig, 340)
    with r:
        fig = go.Figure(go.Bar(x=wk.label, y=wk.recall, marker=dict(color=BLUE, cornerradius=4),
                               customdata=wk.sessions, text=[f"{v:.0%}" for v in wk.recall], textposition="outside",
                               cliponaxis=False,
                               hovertemplate="%{x}<br>Recall %{y:.1%}<br>%{customdata:,.0f} sessions<extra></extra>"))
        fig.update_layout(title="Weekly recall at the chosen threshold", yaxis=dict(tickformat=".0%", range=[0, 0.8]))
        chart(fig, 340)
    sh = D.table("score_hist")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=sh.bin_left + 0.005, y=sh.non_buyers / sh.non_buyers.sum(), name="Non-buyers",
                         marker=dict(color=NEUTRAL), hovertemplate="Score %{x:.2f}<br>%{y:.1%} of non-buyers<extra></extra>"))
    fig.add_trace(go.Bar(x=sh.bin_left + 0.005, y=sh.buyers / sh.buyers.sum(), name="Buyers",
                         marker=dict(color=BLUE, opacity=0.75),
                         hovertemplate="Score %{x:.2f}<br>%{y:.1%} of buyers<extra></extra>"))
    fig.update_layout(title="Score distribution by actual outcome (test)", barmode="overlay", bargap=0.05,
                      xaxis=dict(title="Predicted purchase probability", range=[0, 0.5]),
                      yaxis=dict(tickformat=".0%", title="Share of class"))
    chart(fig, 340)
    insight("PR-AUC stays between 0.19 and 0.21 every week and moves with the weekly purchase rate: the model does "
            "not degrade over time. Buyers' scores are shifted right, but the two distributions overlap heavily: "
            "abandoned carts look exactly like carts that will be paid.")

footer()
