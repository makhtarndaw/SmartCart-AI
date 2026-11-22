import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core import biz as B
from core import ui as U
from core.icons import icon
from core.theme import BLUE, CRITICAL, GOOD, chart, footer

M = B.meta()
U.header(3, "Your products",
         "What sells, what is gaining or losing momentum, what customers look at without buying, and which products "
         "are bought together.")

p = B.product_table()
bm = B.load("brand_month")
br = bm.groupby("brand").revenue.sum().drop(index="(unbranded)", errors="ignore").sort_values(ascending=False)
tot = bm.revenue.sum()
g = B.brand_growth() if len(M["months"]) > 1 else None
pairs = B.load("pairs")
lbl = B.product_label_map()
looked = p[p.views >= max(200, p.views.quantile(0.99))].assign(conv=lambda x: x.units / x.views).sort_values("conv").head(10)

tk = []
if len(br):
    tk.append(f"Your top 3 brands, <b>{', '.join(b.title() for b in br.index[:3])}</b>, make "
              f"<b>{br.head(3).sum() / tot:.0%}</b> of revenue.")
if g is not None and len(g):
    up = g.sort_values("abs_change", ascending=False).iloc[0]
    tk.append(f"<b>{up.brand.title()}</b> is your fastest-growing brand ({B.money(up.abs_change)} more than the month "
              "before): give it visibility and secure stock.")
if len(looked):
    tk.append(f"<b>{len(looked)} popular products</b> are viewed a lot but rarely bought. → Fix their page, price or "
              "availability to convert demand you already have.")
if len(pairs):
    tk.append(f"We found <b>{len(pairs)} product pairs</b> often bought together: ready-made bundle ideas.")
U.takeaways(tk)

st.markdown("### Best sellers")
l, r = st.columns([1, 1.2])
with l:
    t = br.head(10).iloc[::-1]
    fig = go.Figure(go.Bar(y=t.index.str.title(), x=t.values, orientation="h", marker=dict(color=BLUE, cornerradius=4),
                           text=[f"{v / tot:.0%}" for v in t.values], textposition="outside", cliponaxis=False,
                           hovertemplate=B.cfmt("%{y}<br>$%{x:,.0f}<extra></extra>")))
    fig.update_layout(title="Top 10 brands by revenue", xaxis=dict(tickprefix=B.cur(), tickformat=",.0f"),
                      margin=dict(l=10, r=40, t=56, b=10))
    chart(fig, 400)
with r:
    top = p.sort_values("revenue", ascending=False).head(10)
    st.markdown("**Top 10 products**")
    st.dataframe(top[["label", "price", "units", "revenue"]], hide_index=True, use_container_width=True, height=378,
                 column_config={"label": "Product", "price": st.column_config.NumberColumn("Price", format=B.cfmt("$%.2f")),
                                "units": st.column_config.NumberColumn("Sold", format="%d"),
                                "revenue": st.column_config.NumberColumn("Revenue", format=B.cfmt("$%.0f"))})

if g is not None and len(g):
    LM, PM = (B.month_label(m) for m in B.last_prev())
    st.markdown(f"### Gaining and losing momentum · {LM} vs {PM}")
    l, r = st.columns(2)
    for col, asc, title, color in [(l, False, "Growing", GOOD), (r, True, "Declining", CRITICAL)]:
        s = g.sort_values("abs_change", ascending=asc).head(6)
        s = s[(s.abs_change > 0) if not asc else (s.abs_change < 0)].iloc[::-1]
        fig = go.Figure(go.Bar(y=s.brand.str.title(), x=s.abs_change, orientation="h", marker=dict(color=color, cornerradius=4),
                               text=[f"{v:+.0%}" for v in s.growth], textposition="outside", cliponaxis=False,
                               hovertemplate=B.cfmt("%{y}<br>$%{x:,.0f} vs previous month<extra></extra>")))
        fig.update_layout(title=title, xaxis=dict(tickprefix=B.cur(), tickformat=",.0f"), margin=dict(l=10, r=50, t=56, b=10))
        with col:
            chart(fig, 300)

st.markdown("### Looked at, but not bought")
st.caption("Products with lots of views and few sales. Check photos, reviews, price, delivery cost and stock.")
if len(looked):
    st.dataframe(looked[["label", "price", "views", "carts", "units"]], hide_index=True, use_container_width=True,
                 column_config={"label": "Product", "price": st.column_config.NumberColumn("Price", format=B.cfmt("$%.2f")),
                                "views": st.column_config.NumberColumn("Views", format="%d"),
                                "carts": st.column_config.NumberColumn("Added to cart", format="%d"),
                                "units": st.column_config.NumberColumn("Sold", format="%d")})

if len(pairs):
    st.markdown("### Bundle ideas: bought together")
    st.caption("Show these together on product pages, in post-purchase e-mails or as a bundle.")
    cols = st.columns(2)
    for i, (_, x) in enumerate(pairs.sort_values("together", ascending=False).head(8).iterrows()):
        cols[i % 2].markdown(
            f'<div class="finding"><div class="f-icon" style="background:#E0EAFF">{icon("layers", 21, "#1d4ed8")}</div><div>'
            f'<div class="f-title">{lbl.get(x.product_a, x.product_a)} + {lbl.get(x.product_b, x.product_b)}</div>'
            f'<div class="f-text">Bought together in {int(x.together):,} orders · {x.confidence_a_b:.0%} of buyers of the '
            f'first also take the second.</div></div></div>', unsafe_allow_html=True)

U.nav_buttons(3)
footer()
