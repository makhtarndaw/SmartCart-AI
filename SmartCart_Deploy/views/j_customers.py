import plotly.graph_objects as go
import streamlit as st

from core import biz as B
from core import ui as U
from core.icons import icon
from core.theme import BLUE, chart, footer

U.header(2, "Your customers",
         "Every customer who bought from you, sorted into 8 simple groups based on how recently, how often and how "
         "much they buy, with what to do for each group.")

c = B.customers()
seg = B.segment_summary().set_index("segment")
segf = seg.fillna(0)
tot = len(c)
best = segf.loc[["Champions", "Loyal"]]
risk = segf.loc["At risk"]
sleep = segf.loc[["Hibernating", "Lost"]]
tk = [f"You have <b>{tot:,} customers</b>. Your <b>best and loyal customers</b> are "
      f"{best.customers.sum() / max(tot, 1):.0%} of them and bring "
      f"<b>{best.revenue.sum() / max(segf.revenue.sum(), 1e-9):.0%} of revenue</b>. → Protect them."]
if risk.customers > 0:
    tk.append(f"<b>{int(risk.customers):,} good customers are slipping away</b> ({B.money(risk.revenue)} spent so far): "
              "a win-back e-mail is worth it.")
new = segf.loc["New customers"]
if new.customers > 0:
    tk.append(f"<b>{int(new.customers):,} new customers</b> bought for the first time in the last 30 days: welcome "
              "them and give them a reason to come back.")
if sleep.customers.sum() > 0:
    tk.append(f"<b>{sleep.customers.sum() / max(tot, 1):.0%}</b> bought only once and did not come back. The second "
              "order is where you grow fastest.")
U.takeaways(tk)

st.markdown("### Your 8 customer groups")
cols = st.columns(4)
for i, s in enumerate(B.SEGMENT_ORDER):
    if s not in seg.index or not seg.loc[s].customers > 0:
        continue
    r = seg.loc[s]
    name, emo, mean, do = U.CUSTOMER_GROUPS[s]
    with cols[i % 4]:
        st.markdown(f"""
        <div class="group" style="border-top:4px solid {B.SEGMENT_INFO[s][0]}">
          <div class="g-head"><span style="color:{B.SEGMENT_INFO[s][0]}">{icon(emo, 19)}</span> {name}</div>
          <div class="g-num">{int(r.customers):,}</div>
          <div class="g-sub">{r.share_customers:.0%} of customers · {r.share_revenue:.0%} of revenue</div>
          <div class="g-mean">{mean}</div>
          <div class="g-do">→ {do}</div>
        </div>""", unsafe_allow_html=True)
        aud = c[c.segment == s][["user_id", "orders", "revenue", "last_purchase", "favorite_brand"]]
        st.download_button(f"Download list ({len(aud):,})", icon=":material/download:", data= aud.to_csv(index=False).encode(),
                           file_name=f"customers_{name.lower().replace(' ', '_')}.csv", mime="text/csv",
                           key=f"dl_{s}", use_container_width=True)
        st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)

st.markdown("### Where your revenue comes from")
s = seg.dropna().reset_index()
fig = go.Figure()
fig.add_trace(go.Bar(y=[U.CUSTOMER_GROUPS[x][0] for x in s.segment], x=s.share_customers, name="Share of customers",
                     orientation="h", marker=dict(color="#b7d3f6", cornerradius=3),
                     hovertemplate="%{y}<br>%{x:.0%} of customers<extra></extra>"))
fig.add_trace(go.Bar(y=[U.CUSTOMER_GROUPS[x][0] for x in s.segment], x=s.share_revenue, name="Share of revenue",
                     orientation="h", marker=dict(color=BLUE, cornerradius=3),
                     hovertemplate="%{y}<br>%{x:.0%} of revenue<extra></extra>"))
fig.update_layout(barmode="group", xaxis=dict(tickformat=".0%"), yaxis=dict(autorange="reversed"), bargroupgap=0.05,
                  title="Each group: share of customers vs share of revenue")
chart(fig, 380)

with st.expander("Look up one customer", icon=":material/search:"):
    uid = st.text_input("Customer ID", value=str(c.sort_values("revenue", ascending=False).user_id.iloc[0]))
    row = c[c.user_id.astype(str) == uid.strip()]
    if row.empty:
        st.info("No purchase found for this ID.")
    else:
        x = row.iloc[0]
        name, emo, mean, do = U.CUSTOMER_GROUPS[x.segment]
        st.markdown(f"**{name}** · {int(x.orders)} orders · **{B.money(x.revenue, 2)}** spent · last order "
                    f"{int(x.recency)} days ago · favourite brand **{str(x.favorite_brand).title()}**  \n→ {do}")

U.nav_buttons(2)
footer()
