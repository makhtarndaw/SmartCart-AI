import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import biz as B
from core.theme import AQUA, BLUE, CRITICAL, GOOD, INK_3, NEUTRAL, ORANGE, SEQ_SCALE, chart, footer, insight, kpi, \
    page_header

page_header("Analytics", "Trends & forecasting",
            "Spot what is moving: revenue trajectory, seasonality, unusual days, rising and declining brands, "
            "and a 4-week revenue forecast.")

hist, fc, fitted, wtrend = B.forecast_revenue()
an = B.anomalies()

c = st.columns(4)
last4 = hist.tail(28).revenue.sum()
prev4 = hist.iloc[-56:-28].revenue.sum()
kpi(c[0], "Revenue · last 4 weeks", B.money(last4), B.delta_html(last4, prev4).replace(" vs prev. month", " vs prior 4 weeks"))
if fc is not None:
    kpi(c[1], "Forecast · next 4 weeks", B.money(fc.forecast.sum()),
        f"80% range: {B.money(fc.low.sum())} to {B.money(fc.high.sum())}", accent=True)
    kpi(c[2], "Underlying trend", f"{wtrend:+.1%} / week", "last 10 weeks, seasonality removed")
else:
    kpi(c[1], "Forecast · next 4 weeks", "n/a", "needs ≥ 2 weeks of sales data", accent=True)
    kpi(c[2], "Underlying trend", "n/a", "needs ≥ 2 weeks of sales data")
kpi(c[3], "Unusual days detected", f"{int(an.anomaly.sum())}", "robust z-score > 3 vs 4-week baseline")

tab1, tab2, tab3, tab4 = st.tabs(["Revenue & forecast", "Seasonality", "Brands on the move", "Products on the move"])

with tab1:
    metric = st.segmented_control("Metric", ["revenue", "orders", "visitors", "buyers"], default="revenue",
                                  format_func=str.capitalize)
    metric = metric or "revenue"
    a = B.anomalies(metric)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=a.day, y=a[metric], name="Actual", mode="lines", line=dict(color=BLUE, width=2),
                             hovertemplate="%{x|%a %d %b %Y}<br>%{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=a.day, y=a.baseline, name="Baseline (rolling median)", mode="lines",
                             line=dict(color=INK_3, width=1.5, dash="dot"), hoverinfo="skip"))
    pts = a[a.anomaly]
    fig.add_trace(go.Scatter(x=pts.day, y=pts[metric], name="Unusual day", mode="markers",
                             marker=dict(size=10, color=[GOOD if z > 0 else CRITICAL for z in pts.robust_z],
                                         line=dict(color="white", width=2)),
                             customdata=np.c_[pts.robust_z],
                             hovertemplate="%{x|%a %d %b}<br>%{y:,.0f}<br>z = %{customdata[0]:.1f}<extra></extra>"))
    if metric == "revenue" and fc is not None:
        fig.add_trace(go.Scatter(x=pd.concat([fc.day, fc.day[::-1]]), y=pd.concat([fc.high, fc.low[::-1]]),
                                 fill="toself", fillcolor="rgba(235,104,52,.12)", line=dict(width=0),
                                 name="80% forecast range", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=fc.day, y=fc.forecast, name="Forecast", mode="lines",
                                 line=dict(color=ORANGE, width=2.5, dash="dash"),
                                 hovertemplate=B.cfmt("%{x|%a %d %b}<br>forecast $%{y:,.0f}<extra></extra>")))
    fig.update_layout(title=f"Daily {metric} with unusual days" + (" and 4-week forecast" if metric == "revenue" else ""),
                      hovermode="x unified", yaxis=dict(tickprefix=B.cur() if metric == "revenue" else "", tickformat=",.0f"))
    chart(fig, 430)
    top = a[a.anomaly].assign(change=lambda x: x[metric] / x.baseline - 1).sort_values("robust_z", key=abs,
                                                                                         ascending=False).head(8)
    if len(top):
        rows = [f"<b>{r.day:%a %d %b %Y}</b>: {r[metric]:,.0f} vs usual {r.baseline:,.0f} ({r.change:+.0%})"
                for _, r in top.iterrows()]
        insight("Most unusual days: " + " · ".join(rows[:5]))
    st.caption("Forecast model: log-linear trend + day-of-week effects fitted on the last 10 weeks (holiday dip "
               "excluded). Use it for staffing, stock and budget planning, not as a precise target.")

with tab2:
    hw = B.load("hour_weekday")
    days = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    z = hw.pivot(index="weekday", columns="hour", values="revenue").reindex(range(7))
    order = [1, 2, 3, 4, 5, 6, 0]
    z = z.loc[order]
    fig = go.Figure(go.Heatmap(z=z.values, x=[f"{h}h" for h in z.columns], y=[days[i] for i in order],
                               colorscale=SEQ_SCALE, xgap=2, ygap=2,
                               colorbar=dict(tickprefix=B.cur(), thickness=10, outlinewidth=0),
                               hovertemplate=B.cfmt("%{y} %{x}<br>$%{z:,.0f} revenue (5 months)<extra></extra>")))
    fig.update_layout(title="When your customers buy: revenue by weekday × hour (UTC)", yaxis=dict(autorange="reversed"))
    chart(fig, 360)
    d = B.load("daily").copy()
    d["day"] = pd.to_datetime(d.day)
    d["dow"] = d.day.dt.dayofweek
    dw = d.groupby("dow")[["revenue", "orders", "visitors", "buyers"]].mean()
    dw["conversion"] = dw.buyers / dw.visitors
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    l, r = st.columns(2)
    with l:
        fig = go.Figure(go.Bar(x=names, y=dw.revenue, marker=dict(color=BLUE, cornerradius=4),
                               hovertemplate=B.cfmt("%{x}<br>avg $%{y:,.0f}/day<extra></extra>")))
        fig.update_layout(title="Average daily revenue by weekday", yaxis=dict(tickprefix=B.cur(), tickformat=",.0f"))
        chart(fig, 300)
    with r:
        fig = go.Figure(go.Bar(x=names, y=dw.conversion, marker=dict(color=AQUA, cornerradius=4),
                               hovertemplate="%{x}<br>%{y:.2%} of visitors buy<extra></extra>"))
        fig.update_layout(title="Visitor conversion by weekday", yaxis=dict(tickformat=".1%"))
        chart(fig, 300)
    hr = hw.groupby("hour").revenue.sum()
    best = hr.sort_values(ascending=False).head(3).index.tolist()
    insight(f"Peak buying hours: <b>{', '.join(f'{h}h' for h in sorted(best))}</b> (UTC) and best day "
            f"<b>{names[int(dw.revenue.idxmax())]}</b>. Schedule e-mails and promotions just before these peaks; "
            f"weakest day: <b>{names[int(dw.revenue.idxmin())]}</b>. → A good slot for a flash offer.")

with tab3:
    g = B.brand_growth()
    LM, PM = B.month_label(B.last_prev()[0]), B.month_label(B.last_prev()[1])
    dflt = B.default_min_brand_rev()
    min_rev = st.number_input(B.cfmt(f"Minimum brand revenue over {PM} + {LM} ($)"), 0.0, value=dflt, step=max(dflt / 2, 10.0))
    g = B.brand_growth(min_rev)
    l, r = st.columns(2)
    for col, asc, title, color in [(l, False, B.cfmt(f"Rising brands ({LM} vs {PM}, $ change)"), GOOD),
                                   (r, True, B.cfmt(f"Declining brands ({LM} vs {PM}, $ change)"), CRITICAL)]:
        s = g.sort_values("abs_change", ascending=asc).head(10).iloc[::-1]
        fig = go.Figure(go.Bar(y=s.brand.str.title(), x=s.abs_change, orientation="h",
                               marker=dict(color=color, cornerradius=4),
                               customdata=np.c_[s.growth * 100, s.rev_last],
                               text=[f"{v:+.0%}" for v in s.growth], textposition="outside", cliponaxis=False,
                               hovertemplate=B.cfmt("%{y}<br>$%{x:,.0f} change (%{customdata[0]:+.0f}%)<br>Last-month revenue "
                                             "$%{customdata[1]:,.0f}<extra></extra>")))
        fig.update_layout(title=title, xaxis=dict(tickprefix=B.cur(), tickformat=",.0f"), margin=dict(l=10, r=50, t=56, b=10))
        with col:
            chart(fig, 400)
    fig = go.Figure(go.Scatter(
        x=g.views_growth, y=g.growth, mode="markers", text=g.brand.str.title(),
        marker=dict(size=np.sqrt(g.rev_last.clip(lower=1)) / 4 + 4, color=np.where(g.growth > 0, BLUE, ORANGE),
                    opacity=0.7, line=dict(color="white", width=1)),
        hovertemplate="<b>%{text}</b><br>Traffic %{x:+.0%}<br>Revenue %{y:+.0%}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=INK_3, width=1))
    fig.add_vline(x=0, line=dict(color=INK_3, width=1))
    for x, y, t in [(0.5, 0.9, "Momentum"), (-0.4, 0.9, "Converting better"), (0.5, -0.6, "Traffic not converting"),
                    (-0.4, -0.6, "Losing interest")]:
        fig.add_annotation(x=x, y=y, text=t, showarrow=False, font=dict(size=12, color="#94A3B8"))
    fig.update_layout(title="Brand momentum: traffic growth vs revenue growth (bubble = last-month revenue)",
                      xaxis=dict(tickformat="+.0%", title="Traffic (views) growth", range=[-0.8, 1.0]),
                      yaxis=dict(tickformat="+.0%", title="Revenue growth", range=[-1, 1.5]))
    chart(fig, 440)

with tab4:
    p = B.product_table()
    p = p[(p.revenue_prev + p.revenue_last) >= p.revenue_last.sum() * 0.0005].copy()
    p["abs_change"] = p.revenue_last - p.revenue_prev
    l, r = st.columns(2)
    cfg = {"label": "Product", "revenue_prev": st.column_config.NumberColumn(f"{PM} revenue", format=B.cfmt("$%.0f")),
           "revenue_last": st.column_config.NumberColumn(f"{LM} revenue", format=B.cfmt("$%.0f")),
           "abs_change": st.column_config.NumberColumn("Change", format=B.cfmt("$%.0f")),
           "price": st.column_config.NumberColumn("Price", format=B.cfmt("$%.2f"))}
    with l:
        st.markdown("**Trending up**")
        st.dataframe(p.sort_values("abs_change", ascending=False).head(15)[["label", "price", "revenue_prev", "revenue_last", "abs_change"]],
                     hide_index=True, use_container_width=True, column_config=cfg)
    with r:
        st.markdown("**Slowing down**")
        st.dataframe(p.sort_values("abs_change").head(15)[["label", "price", "revenue_prev", "revenue_last", "abs_change"]],
                     hide_index=True, use_container_width=True, column_config=cfg)

footer()
