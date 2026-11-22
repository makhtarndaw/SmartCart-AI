import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core import biz as B
from core.theme import BLUE, ORANGE, chart, footer, insight, kpi, page_header

page_header("Analytics", "Cart recovery",
            "Abandoned carts are revenue that was almost yours. Measure the leak, find the products and shoppers "
            "concerned, and size a recovery campaign.")

am = B.load("abandon_monthly")
ab = B.load("abandoners")
ap = B.load("abandoned_products")
mk = B.monthly_kpis()
last = am.iloc[-1]
LM = B.month_label(last.month)

k = st.columns(4)
kpi(k[0], "Abandonment rate", f"{last.abandoned_items / last.carted_items:.0%}", f"of carted items not bought · {LM}")
kpi(k[1], "Abandoned value", B.money(last.abandoned_value), f"{LM} (item × price, est.)")
kpi(k[2], "Shoppers to recover", f"{len(ab):,}", "cart in the last 30 days, not bought since", accent=True)
kpi(k[3], "Median cart left", B.cfmt(f"${ab.cart_value.median():.2f}"), B.cfmt(f"mean ${ab.cart_value.mean():.2f}"))

l, r = st.columns(2)
with l:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[B.MONTH_LABEL[m] for m in am.month], y=am.carted_value - am.abandoned_value, name="Bought",
                         marker=dict(color=BLUE, line=dict(color="white", width=2)),
                         hovertemplate=B.cfmt("%{x}<br>$%{y:,.0f} carted & bought<extra></extra>")))
    fig.add_trace(go.Bar(x=[B.MONTH_LABEL[m] for m in am.month], y=am.abandoned_value, name="Abandoned",
                         marker=dict(color=ORANGE, line=dict(color="white", width=2)),
                         hovertemplate=B.cfmt("%{x}<br>$%{y:,.0f} abandoned<extra></extra>")))
    fig.update_layout(title="Value placed in carts: bought vs abandoned", barmode="stack",
                      yaxis=dict(tickprefix=B.cur(), tickformat=",.0f"))
    chart(fig, 360)
with r:
    bins = [0, 5, 10, 20, 50, 100, 1e9]
    labels = [B.cfmt("<$5"), B.cfmt("$5-10"), B.cfmt("$10-20"), B.cfmt("$20-50"), B.cfmt("$50-100"), B.cfmt("$100+")]
    ab2 = ab.assign(band=np.digitize(ab.cart_value, bins[1:-1]))
    g = ab2.groupby("band").agg(users=("user_id", "size"), value=("cart_value", "sum")).reindex(range(6), fill_value=0)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=g.users, name="Shoppers", marker=dict(color="#b7d3f6", cornerradius=4),
                         customdata=g.value, hovertemplate=B.cfmt("%{x}<br>%{y:,} shoppers<br>$%{customdata:,.0f} in carts<extra></extra>")))
    fig.update_layout(title="Shoppers to recover by cart value left", yaxis=dict(title="Shoppers"))
    chart(fig, 360)

st.markdown('<div class="section">Recovery simulator</div><div class="section-sub">Size a 3-step reminder sequence '
            '(1 h · 24 h · 72 h). Adjust the assumptions to your context.</div>', unsafe_allow_html=True)
s1, s2, s3, s4 = st.columns(4)
min_cart = s1.number_input(B.cfmt("Only carts above ($)"), 0.0, 500.0, 10.0, 5.0)
rate = s2.slider("Share of abandoned value recovered", 0.01, 0.15, 0.03, 0.01, format="%.2f")
margin = s3.slider("Gross margin", 0.2, 0.7, 0.4, 0.05, format="%.2f")
cost = s4.number_input(B.cfmt("Cost per e-mail ($)"), 0.0, 1.0, 0.05, 0.01)
tgt = ab[ab.cart_value >= min_cart]
rev = tgt.cart_value.sum() * rate
profit = rev * margin - len(tgt) * 3 * cost
k = st.columns(4)
kpi(k[0], "Audience", f"{len(tgt):,}", f"{len(tgt) / len(ab):.0%} of abandoners")
kpi(k[1], "Value in their carts", B.money(tgt.cart_value.sum()), f"{tgt.cart_value.sum() / ab.cart_value.sum():.0%} of total")
kpi(k[2], "Recovered revenue / month", B.money(rev), f"at {rate:.0%} recovery")
kpi(k[3], "Net profit / month", B.money(profit), f"ROI ×{profit / max(len(tgt) * 3 * cost, 1e-9):.0f}" if cost else "no send cost",
    accent=True)
st.download_button(f"Export the {len(tgt):,} shoppers (CSV)", icon=":material/download:", data=tgt.to_csv(index=False).encode(),
                   file_name="smartcart_cart_recovery.csv", mime="text/csv")
st.caption("Carted value overstates real buying intent (many shoppers use the cart as a wishlist), so keep the recovery "
           "rate conservative and validate it with a holdout group.")

st.markdown('<div class="section">Most abandoned products · last 30 days</div>', unsafe_allow_html=True)
lbl = B.product_label_map()
ap = ap.assign(product=[lbl.get(pid, f"{str(b).title()} · #{pid}") for pid, b in zip(ap.product_id, ap.brand)])
l, r = st.columns([1.2, 1])
with l:
    t = ap.head(15).iloc[::-1]
    fig = go.Figure(go.Bar(y=t["product"], x=t.value, orientation="h", marker=dict(color=ORANGE, cornerradius=4),
                           customdata=t.users, hovertemplate=B.cfmt("%{y}<br>$%{x:,.0f} left in carts<br>%{customdata:,} shoppers<extra></extra>")))
    fig.update_layout(title="Abandoned value by product", xaxis=dict(tickprefix=B.cur(), tickformat=",.0f"))
    chart(fig, 460)
with r:
    byb = ab.groupby("top_brand").agg(users=("user_id", "size"), value=("cart_value", "sum")).sort_values("value",
                                                                                                         ascending=False)
    byb = byb.head(12).iloc[::-1]
    fig = go.Figure(go.Bar(y=byb.index.str.title(), x=byb.value, orientation="h", marker=dict(color="#e87ba4", cornerradius=4),
                           customdata=byb.users, hovertemplate=B.cfmt("%{y}<br>$%{x:,.0f}<br>%{customdata:,} shoppers<extra></extra>")))
    fig.update_layout(title="Abandoned value by main brand in cart", xaxis=dict(tickprefix=B.cur(), tickformat=",.0f"))
    chart(fig, 460)
insight("Behaviour insight from the Purchase Intent AI: shoppers who <b>remove</b> an item are often finalising their "
        "order. The real risk is the cart left untouched. → Prioritise reminders on high-value carts and on the products "
        "above (check their price, delivery cost and stock first).")

footer()
