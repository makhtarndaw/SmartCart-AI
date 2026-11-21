import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core import biz as B

from core import data as D
from core.theme import (ACTION_COLORS, AQUA, BLUE, INK_3, NEUTRAL, ORANGE, SEQ_SCALE, chart, footer,
                        insight, kpi, page_header, pct, section, short_seq)

K = D.kpis()
AVG = K["dataset"]["purchase_rate"]

page_header("Analytics", "Session behaviour",
            "Which behaviours in the first three actions of a session are linked to a later purchase? "
            f"All results are correlations on {K['dataset']['sessions']:,} scored sessions of your data — not causal effects.")

if not D.intent_available():
    st.warning("Not enough eligible sessions in this workspace to run the purchase-intent model (sessions with at "
               "least 3 events and no purchase in the first three). Upload more history or include view/cart events.")
    st.stop()

cart = D.table("by_cart").set_index("has_cart")
rem = D.table("by_remove")
c = st.columns(4)
kpi(c[0], "Purchase rate · no cart", pct(cart.purchase_rate[0]), f"{cart.sessions[0]:,.0f} sessions")
kpi(c[1], "Purchase rate · with cart", pct(cart.purchase_rate[1]), f"{cart.sessions[1]:,.0f} sessions")
pv = K["cart_test"].get("p_value", float("nan"))
kpi(c[2], "Odds ratio (cart)", f"{K['cart_test']['odds_ratio']:.2f}",
    "χ² test p-value " + ("< 0.001" if pv < 0.001 else f"= {pv:.3f}"))
r1 = rem.set_index("n_remove").purchase_rate
kpi(c[3], "One removal", pct(r1.get(1, float("nan"))), f"vs {pct(r1.get(0, float('nan')))} without removal",
    accent=True)

tab1, tab2, tab3, tab4 = st.tabs(["Sequences", "Cart & removals", "Timing & price", "Journey flow"])

# ------------------------------------------------------------------ sequences ----
with tab1:
    seq = D.table("sequences")
    min_n = st.select_slider("Minimum sessions per sequence", [100, 500, 1000, 5000], value=500)
    s = seq[seq.sessions >= min_n].sort_values("purchase_rate")
    s["label"] = s.sequence.map(short_seq)
    s["has_cart_action"] = s.sequence != "view > view > view"
    colors = [BLUE if h else AQUA for h in s.has_cart_action]
    fig = go.Figure(go.Bar(
        y=s.label, x=s.purchase_rate, orientation="h", marker=dict(color=colors, cornerradius=4),
        customdata=np.c_[s.sessions], text=[f"{v:.1%}" for v in s.purchase_rate], textposition="outside",
        textfont=dict(size=11, color="#475569"), cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>Purchase rate %{x:.1%}<br>%{customdata[0]:,.0f} sessions<extra></extra>"))
    fig.add_vline(x=AVG, line=dict(color=INK_3, dash="dash", width=1.5),
                  annotation_text=f"Average {AVG:.1%}", annotation_position="top")
    fig.update_layout(title="Purchase rate by sequence of the first three events",
                      xaxis=dict(tickformat=".0%", title=None), yaxis=dict(title=None),
                      margin=dict(l=10, r=60, t=60, b=10))
    chart(fig, max(420, 24 * len(s) + 100))
    st.caption("■ Blue = at least one cart action · ■ Green = view only. Hover a bar for the number of sessions.")
    insight("Every sequence with a cart action beats the average; <b>view → view → view</b> — the most common "
            "journey — is the only one below it. The best sequences contain a <b>removal</b>: these visitors "
            "manage their cart just before checking out (often returning visitors).")

# ------------------------------------------------------------- cart & removals ---
with tab2:
    l, r = st.columns(2)
    with l:
        nc = D.table("by_ncart")
        fig = go.Figure(go.Bar(x=nc.n_cart.astype(str), y=nc.purchase_rate, marker=dict(color=BLUE, cornerradius=4),
                               customdata=nc.sessions, text=[f"{v:.1%}" for v in nc.purchase_rate],
                               textposition="outside", cliponaxis=False,
                               hovertemplate="%{x} cart additions<br>%{y:.1%} buy<br>%{customdata:,.0f} sessions<extra></extra>"))
        fig.add_hline(y=AVG, line=dict(color=INK_3, dash="dash", width=1.5))
        fig.update_layout(title="By number of cart additions (first 3 events)",
                          yaxis=dict(tickformat=".0%"), xaxis=dict(title="Cart additions"))
        chart(fig, 360)
    with r:
        fig = go.Figure(go.Bar(x=rem.n_remove.astype(str), y=rem.purchase_rate, marker=dict(color=ORANGE, cornerradius=4),
                               customdata=rem.sessions, text=[f"{v:.1%}" for v in rem.purchase_rate],
                               textposition="outside", cliponaxis=False,
                               hovertemplate="%{x} removals<br>%{y:.1%} buy<br>%{customdata:,.0f} sessions<extra></extra>"))
        fig.add_hline(y=AVG, line=dict(color=INK_3, dash="dash", width=1.5))
        fig.update_layout(title="By number of cart removals (first 3 events)",
                          yaxis=dict(tickformat=".0%"), xaxis=dict(title="Cart removals"))
        chart(fig, 360)
    insight("One removal <b>doubles</b> the purchase rate; two or three removals bring it down again — those "
            "visitors are emptying the cart rather than finalising it. Motivated visitors add to cart: forcing a "
            "cart addition would not necessarily create purchases (self-selection).")

    st.markdown("**Position of each action in the sequence**")
    cols = st.columns(3)
    for i, col in enumerate(cols, 1):
        a = D.table(f"by_action_{i}")
        fig = go.Figure(go.Bar(x=a[f"action_{i}"].map(lambda v: v.replace("remove_from_cart", "remove")),
                               y=a.purchase_rate, marker=dict(color=[ACTION_COLORS[v] for v in a[f"action_{i}"]],
                                                              cornerradius=4),
                               customdata=a.sessions, text=[f"{v:.1%}" for v in a.purchase_rate],
                               textposition="outside", cliponaxis=False,
                               hovertemplate="%{x}<br>%{y:.1%} buy<br>%{customdata:,.0f} sessions<extra></extra>"))
        fig.update_layout(title=f"Action {i}", yaxis=dict(tickformat=".0%", range=[0, 0.22]), showlegend=False)
        with col:
            chart(fig, 280)

# --------------------------------------------------------------- timing & price --
with tab3:
    dur = D.table("duration")
    order = ["<30s", "30s-1min", "1-2min", "2-5min", "5-10min", "10-30min", ">30min"]
    fig = go.Figure()
    for hc, name, color in [(1, "With cart addition", BLUE), (0, "No cart addition", AQUA)]:
        d = dur[dur.has_cart == hc].set_index("duration_group").reindex(order)
        fig.add_trace(go.Scatter(x=order, y=d.purchase_rate, name=name, mode="lines+markers",
                                 line=dict(color=color, width=2.5), marker=dict(size=9, line=dict(color="white", width=2)),
                                 customdata=d.sessions,
                                 hovertemplate=name + "<br>%{x}: %{y:.1%} buy<br>%{customdata:,.0f} sessions<extra></extra>"))
    fig.add_hline(y=AVG, line=dict(color=INK_3, dash="dash", width=1.5))
    fig.update_layout(title="Purchase rate by duration of the first three events", yaxis=dict(tickformat=".0%"),
                      hovermode="x unified")
    chart(fig, 380)
    insight("Without a cart, the longer the visitor browses, the less they buy (10% → 4%). With a cart, the rate is "
            "<b>stable around 17–18%</b> between 30 s and 10 min: a visitor who takes time to choose and adds a "
            "product is the most likely buyer.")

    pr = D.table("price")
    porder = ["<$2", "$2-4", "$4-6", "$6-10", "$10-20", "$20-50", ">$50"]
    fig = go.Figure()
    for hc, name, color in [(1, "With cart addition", BLUE), (0, "No cart addition", AQUA)]:
        d = pr[pr.has_cart == hc].set_index("price_group").reindex(porder)
        fig.add_trace(go.Bar(x=[B.cfmt(x) for x in porder], y=d.purchase_rate, name=name, marker=dict(color=color, cornerradius=4),
                             customdata=d.sessions,
                             hovertemplate=name + "<br>%{x}: %{y:.1%} buy<br>%{customdata:,.0f} sessions<extra></extra>"))
    fig.update_layout(title="Purchase rate by average price of the products interacted with", barmode="group",
                      yaxis=dict(tickformat=".0%"), bargroupgap=0.08)
    chart(fig, 360)

    hw = D.table("sess_hour_weekday")
    z = hw.pivot(index="weekday", columns="hour", values="purchase_rate").reindex(range(7))
    fig = go.Figure(go.Heatmap(z=z.values, x=[f"{h}h" for h in z.columns],
                               y=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], colorscale=SEQ_SCALE,
                               xgap=2, ygap=2, colorbar=dict(tickformat=".0%", thickness=10, outlinewidth=0),
                               hovertemplate="%{y} %{x}<br>Purchase rate %{z:.1%}<extra></extra>"))
    fig.update_layout(title="When do engaged sessions convert? (hour of the 3rd event × weekday)",
                      yaxis=dict(autorange="reversed"))
    chart(fig, 340)

# --------------------------------------------------------------- journey flow ----
with tab4:
    sk = D.table("sankey")
    nodes = list(dict.fromkeys(list(sk.source) + list(sk.target)))
    idx = {n: i for i, n in enumerate(nodes)}

    def node_label(n):
        step, a = n.split(":")
        nice = {"view": "View", "cart": "Cart", "remove_from_cart": "Remove", "purchase": "Purchase",
                "no purchase": "No purchase"}[a]
        return nice if step == "out" else f"{nice} ({step})"

    def node_color(n):
        a = n.split(":")[1]
        return {"purchase": "#0d366b", "no purchase": NEUTRAL}.get(a, ACTION_COLORS.get(a, NEUTRAL))

    def link_color(row):
        a = row.target.split(":")[1]
        base = {"purchase": (13, 54, 107), "no purchase": (203, 213, 225)}.get(a)
        if base is None:
            h = ACTION_COLORS[a].lstrip("#")
            base = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
        return f"rgba({base[0]},{base[1]},{base[2]},0.28)"

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(label=[node_label(n) for n in nodes], color=[node_color(n) for n in nodes], pad=18, thickness=16,
                  line=dict(color="white", width=1),
                  hovertemplate="%{label}<br>%{value:,.0f} sessions<extra></extra>"),
        link=dict(source=sk.source.map(idx), target=sk.target.map(idx), value=sk.sessions,
                  color=[link_color(r) for r in sk.itertuples()],
                  hovertemplate="%{source.label} → %{target.label}<br>%{value:,.0f} sessions<extra></extra>"),
    ))
    fig.update_layout(title=f"How {K['dataset']['sessions']:,} sessions flow through their first three actions to the outcome",
                      font=dict(size=12))
    chart(fig, 520)
    insight("Read left to right: action 1 → action 2 → action 3 → purchase later in the session. Most journeys "
            "stay in the same state (view → view, cart → cart); the thin dark band on the right is the 11% who buy.")

    daily = D.table("sess_daily")
    fig = go.Figure(go.Scatter(x=daily.day, y=daily.purchase_rate, mode="lines", line=dict(color=BLUE, width=2),
                               customdata=daily.sessions,
                               hovertemplate="%{x}<br>%{y:.1%} buy<br>%{customdata:,.0f} sessions<extra></extra>"))
    fig.update_layout(title="Daily future-purchase rate of scored sessions", yaxis=dict(tickformat=".0%"))
    chart(fig, 320)

footer()
