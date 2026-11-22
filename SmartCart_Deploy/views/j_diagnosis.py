import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import biz as B
from core import data as D
from core import ui as U
from core.theme import BLUE, NEUTRAL, ORANGE, chart, footer, kpi

M = B.meta()
U.header(1, f"Your diagnosis · {M['name']}",
         "The essentials about your business, written from your data. Read the cards, then move on to your "
         "customers, products and action plan.")

mk = B.monthly_kpis()
cur = mk.iloc[-1]
prev = mk.iloc[-2] if len(mk) > 1 else None
P = (lambda k: prev[k]) if prev is not None else (lambda k: None)
LM = B.month_label(cur.month)

c = st.columns(4)
kpi(c[0], f"Revenue · {LM}", B.money(cur.revenue), B.delta_html(cur.revenue, P("revenue")))
kpi(c[1], f"Customers · {LM}", f"{cur.buyers:,.0f}", B.delta_html(cur.buyers, P("buyers")))
kpi(c[2], "Visitors who buy", f"{cur.conversion:.1%}", B.delta_html(cur.conversion, P("conversion"), pct_points=True))
kpi(c[3], "Average basket", B.cfmt(f"${cur.aov:.2f}"), B.delta_html(cur.aov, P("aov")))

st.markdown("### What we found")
U.render_findings(U.diagnosis())

st.markdown("### Your sales, and what's coming")
hist, fc, _, _ = B.forecast_revenue()
h = hist.copy()
h["week"] = h.day.dt.to_period("W").dt.start_time
wk = h.groupby("week").revenue.sum().reset_index().iloc[1:-1] if len(h) > 21 else h.rename(columns={"day": "week"})
fig = go.Figure()
fig.add_trace(go.Bar(x=wk.week, y=wk.revenue, name="Weekly revenue", marker=dict(color=BLUE, cornerradius=3),
                     hovertemplate=B.cfmt("Week of %{x|%d %b %Y}<br>$%{y:,.0f}<extra></extra>")))
if fc is not None:
    f = fc.copy()
    f["week"] = f.day.dt.to_period("W").dt.start_time
    fw = f.groupby("week")[["forecast", "low", "high"]].sum().reset_index()
    fw = fw[fw.week > wk.week.max()]
    fig.add_trace(go.Bar(x=fw.week, y=fw.forecast, name="Forecast", marker=dict(color="#b7d3f6", cornerradius=3,
                         line=dict(color=BLUE, width=1.5)),
                         error_y=dict(type="data", symmetric=False, array=fw.high - fw.forecast,
                                      arrayminus=fw.forecast - fw.low, color="#64748B", thickness=1.2, width=4),
                         hovertemplate=B.cfmt("Week of %{x|%d %b %Y}<br>forecast $%{y:,.0f}<extra></extra>")))
fig.update_layout(title="Revenue per week and our forecast for the next 4 weeks", bargap=0.2,
                  yaxis=dict(tickprefix=B.cur(), tickformat=",.0f"))
chart(fig, 340)

if D.intent_available():
    st.markdown("### Who is visiting your shop right now")
    l, r = st.columns([1, 1.3])
    seg = D.table("segments").set_index("segment").reindex(D.SEGMENTS)
    with l:
        fig = go.Figure(go.Pie(labels=[U.INTENT_GROUPS[s][0] for s in seg.index], values=seg.sessions, hole=0.6,
                               sort=False, marker=dict(colors=[U.INTENT_GROUPS[s][1] for s in seg.index],
                                                       line=dict(color="white", width=2)),
                               textinfo="percent", textfont=dict(color="white"),
                               hovertemplate="<b>%{label}</b><br>%{value:,} visits · %{percent}<extra></extra>"))
        fig.update_layout(title="Visits, read by our AI after 3 clicks", legend=dict(orientation="v", x=1, y=0.5))
        chart(fig, 300)
    with r:
        rows = "".join(
            f"<tr><td><span style='display:inline-block;width:10px;height:10px;border-radius:3px;background:{U.INTENT_GROUPS[s][1]}'></span>"
            f" <b>{U.INTENT_GROUPS[s][0]}</b></td><td style='text-align:right'>{seg.loc[s, 'purchase_rate']:.0%}</td>"
            f"<td style='text-align:right'>{seg.loc[s, 'share_of_buyers']:.0%}</td></tr>" for s in seg.index)
        st.markdown(f"""
        <div class="card"><p style="margin-bottom:.6rem">After a visitor's 3rd click, the SmartCart AI estimates
        how likely they are to buy before leaving. Each group gets a different action in your plan.</p>
        <table style="width:100%;font-size:.92rem"><tr style="color:#64748B;font-size:.75rem;text-transform:uppercase">
        <td>Group</td><td style="text-align:right">Buy</td><td style="text-align:right">Share of your buyers</td></tr>{rows}</table></div>
        """, unsafe_allow_html=True)

U.nav_buttons(1)
footer()
