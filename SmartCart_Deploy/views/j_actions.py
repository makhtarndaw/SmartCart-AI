import streamlit as st

from core import biz as B
from core import ui as U
from core.theme import footer, kpi

U.header(4, "Your action plan",
         "Concrete marketing actions built from your data, ranked by expected profit. Each one tells you who to "
         "target, what to do and when — and gives you the list of customers to export to your e-mailing tool.")

with st.expander("Adjust the assumptions behind the estimates", icon=":material/tune:"):
    a1, a2, a3 = st.columns(3)
    A = {"margin": a1.slider("Your gross margin", 0.2, 0.8, 0.40, 0.05, format="%.2f"),
         "email_cost": a2.number_input(B.cfmt("Cost of one e-mail ($)"), 0.0, 1.0, 0.05, 0.01),
         "recovery_rate": a3.slider("Share of abandoned carts you recover", 0.01, 0.15, 0.03, 0.01, format="%.2f")}
    st.caption("Estimates are monthly and deliberately cautious. Validate each action on a small group first "
               "(see the A/B test planner in the Expert space).")

camps = B.campaigns(A)
day, hour = U.best_send_time()
when_default = f"{day}s, just before {hour}:00 (your sales peak)" if day else "When your customers are most active"
WHEN = {"cart_recovery": "Automatically 1 h, 24 h and 72 h after the cart is left",
        "intent": "Live, on your site, right after the visitor's 3rd click",
        "bundles": "Permanently on product pages + in the order-confirmation e-mail",
        "rising": "This month, while the brand is growing",
        "second_order": "14–21 days after the first order"}
PLAIN = {"cart_recovery": "Bring back abandoned carts", "winback": "Win back customers who are slipping away",
         "second_order": "Get new customers to order a 2nd time", "vip": "Look after your best customers",
         "bundles": "Sell more per order with bundles", "rising": "Push your rising brands",
         "intent": "Convert hesitating visitors in real time (AI)"}

tot_rev = sum(c["revenue"] for c in camps)
tot_profit = sum(c["profit"] for c in camps)
c = st.columns(3)
kpi(c[0], "Actions ready", f"{len(camps)}", "ranked by expected profit")
kpi(c[1], "Potential extra revenue", f"{B.money(tot_rev)} / month", "if all actions are run")
kpi(c[2], "Potential extra profit", f"{B.money(tot_profit)} / month", f"at {A['margin']:.0%} margin", accent=True)
st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

for i, cp in enumerate(camps, 1):
    with st.container(border=True):
        l, r = st.columns([3.2, 1])
        with l:
            st.markdown(f"""
            <div style="display:flex;gap:.8rem;align-items:center">
              <div class="action"><div class="a-rank">{i}</div></div>
              <div class="a-title" style="font-size:1.18rem;font-weight:800;color:#0F172A">{PLAIN.get(cp['id'], cp['name'])}</div>
            </div>
            <div class="action" style="border:none;padding:0">
              <div class="a-row">
                <div class="a-k">Why</div><div class="a-v">{cp['why']}</div>
                <div class="a-k">Who</div><div class="a-v"><b>{cp['audience']:,}</b> {'visits / month' if cp['id'] == 'intent' else 'people' if cp['id'] not in ('bundles', 'rising') else 'shoppers concerned'}</div>
                <div class="a-k">What to do</div><div class="a-v">{cp['action']}</div>
                <div class="a-k">When</div><div class="a-v">{WHEN.get(cp['id'], when_default)}</div>
                <div class="a-k">Channel</div><div class="a-v">{cp['channel']}</div>
              </div>
            </div>""", unsafe_allow_html=True)
        with r:
            st.markdown(f"""
            <div style="text-align:right">
              <div style="font-size:.72rem;color:#64748B;font-weight:700;text-transform:uppercase">Expected profit</div>
              <div style="font-size:1.6rem;font-weight:800;color:#0a8a0a">{B.money(cp['profit'])}</div>
              <div style="font-size:.8rem;color:#64748B">per month · revenue {B.money(cp['revenue'])}</div>
            </div>""", unsafe_allow_html=True)
            st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
            if cp["audience_df"] is not None:
                st.download_button("Download the list", icon=":material/download:", data=cp["audience_df"].to_csv(index=False).encode(),
                                   file_name=f"smartcart_{cp['id']}.csv", mime="text/csv", key=f"dl_{cp['id']}",
                                   use_container_width=True)
            else:
                st.page_link("views/simulator.py", label="Try the AI on a visitor", icon=":material/bolt:",
                             use_container_width=True)
        st.caption(f"How we estimated it: {cp['assumption']}")

U.nav_buttons(4)
footer()
