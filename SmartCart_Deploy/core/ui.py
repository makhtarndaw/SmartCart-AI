"""Guided-journey UI: stepper, takeaways, navigation buttons, plain-language names and the
automatic diagnosis (findings written in plain English from the company's data)."""
import numpy as np
import pandas as pd
import streamlit as st

from core import biz as B
from core import data as D
from core.icons import icon

STEPS = [
    ("views/j_import.py", "Import data"),
    ("views/j_diagnosis.py", "Diagnosis"),
    ("views/j_customers.py", "Customers"),
    ("views/j_products.py", "Products"),
    ("views/j_actions.py", "Action plan"),
]

CUSTOMER_GROUPS = {  # RFM segment -> plain name, emoji, one-line meaning, what to do
    "Champions": ("Best customers", "award", "Buy often, bought recently.", "Reward them with early access and gifts. No discounts."),
    "Loyal": ("Loyal customers", "heart", "Buy often, a bit less recently.", "Loyalty points and replenishment reminders."),
    "Promising": ("Promising", "trend_up", "2 orders, recent.", "Push the 3rd order with product bundles."),
    "New customers": ("New customers", "sparkles", "First order this month.", "Welcome series and a reason to come back."),
    "Need attention": ("Need a nudge", "bell", "2 orders, quieter lately.", "Reminder on their favourite brand."),
    "At risk": ("Slipping away", "alert", "Good customers gone silent for 75+ days.", "Win-back e-mail with a personal offer."),
    "Hibernating": ("Sleeping", "moon", "One order, 1–3 months ago.", "Low-cost reactivation: new arrivals."),
    "Lost": ("Lost", "user_x", "One order, more than 3 months ago.", "Newsletter only, don't spend budget."),
}
INTENT_GROUPS = {
    "1. Top 10%": ("Ready to buy", "#0d366b"),
    "2. Next 20%": ("Hesitating · high potential", "#1c5cab"),
    "3. Middle 30%": ("Hesitating", "#3987e5"),
    "4. Bottom 40%": ("Just browsing", "#86b6ef"),
}
DAYS = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]  # DuckDB dayofweek: 0 = Sunday


def stepper(current):
    html = '<div class="stepper">'
    for i, (_, label) in enumerate(STEPS):
        state = "done" if i < current else ("now" if i == current else "todo")
        html += (f'<div class="st {state}"><span class="dot">{"✓" if i < current else i + 1}</span>'
                 f'<span class="lbl">{label}</span></div>')
        if i < len(STEPS) - 1:
            html += '<div class="bar"></div>'
    st.markdown(html + "</div>", unsafe_allow_html=True)


def header(step, title, subtitle):
    stepper(step)
    st.markdown(f'<div class="page-head"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)
    if D.is_demo() and step > 0:
        st.markdown('<div class="demo-banner">You are looking at <b>demo data</b> (a cosmetics e-shop). '
                    'Go to <b>step 1 · Import data</b> to see your own results.</div>', unsafe_allow_html=True)


def takeaways(items):
    lis = "".join(f"<li>{t}</li>" for t in items)
    st.markdown(f'<div class="takeaways"><div class="tk-title">What you need to know</div><ul>{lis}</ul></div>',
                unsafe_allow_html=True)


def nav_buttons(step):
    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    l, _, r = st.columns([1, 2, 1])
    if step > 0:
        l.page_link(STEPS[step - 1][0], label=f"← {STEPS[step - 1][1]}", use_container_width=True)
    if step < len(STEPS) - 1:
        r.page_link(STEPS[step + 1][0], label=f"Next: {STEPS[step + 1][1]} →", use_container_width=True)


def best_send_time():
    hw = B.load("hour_weekday")
    if hw.revenue.sum() == 0:
        return None, None
    day = int(hw.groupby("weekday").revenue.sum().idxmax())
    hour = int(hw.groupby("hour").revenue.sum().idxmax())
    return DAYS[day], hour


def finding(icon, title, text, tone="neutral"):
    return {"icon": icon, "title": title, "text": text, "tone": tone}


def diagnosis():
    """Plain-English findings from the company's data, most important first."""
    out = []
    mk = B.monthly_kpis()
    m = B.meta()
    cur = mk.iloc[-1]
    prev = mk.iloc[-2] if len(mk) > 1 else None

    # 1. Revenue
    if prev is not None and prev.revenue > 0:
        ch = cur.revenue / prev.revenue - 1
        out.append(finding("trend_up" if ch >= 0 else "trend_down", f"Revenue {'up' if ch >= 0 else 'down'} {abs(ch):.0%} last month",
                           f"{B.money(cur.revenue)} in {B.month_label(cur.month)} vs {B.money(prev.revenue)} the month "
                           f"before, from {cur.orders:,.0f} orders (average basket {B.money(cur.aov, 2)}).",
                           "good" if ch >= 0 else "bad"))
    hist, fc, _, wtrend = B.forecast_revenue()
    if fc is not None:
        out.append(finding("forecast", f"Next 4 weeks: about {B.money(fc.forecast.sum())}",
                           f"Our forecast, based on your recent trend ({wtrend:+.1%} per week) and your weekly pattern. "
                           f"Likely range: {B.money(fc.low.sum())} to {B.money(fc.high.sum())}.", "neutral"))

    # 2. Abandoned carts
    ab = B.load("abandoners")
    if len(ab):
        out.append(finding("cart", f"{len(ab):,} shoppers left products in their cart",
                           f"Over the last 30 days, {B.money(ab.cart_value.sum())} of products were added to a cart and "
                           "never bought. A reminder sequence is your quickest win.", "bad"))

    # 3. Customers slipping away
    c = B.customers()
    risk = c[c.segment == "At risk"]
    if len(risk):
        out.append(finding("alert", f"{len(risk):,} good customers are slipping away",
                           f"They ordered at least twice but nothing for 75+ days. Together they spent "
                           f"{B.money(risk.revenue.sum())} with you.", "bad"))

    # 4. Concentration
    if len(c) >= 10:
        s = c.revenue.sort_values(ascending=False)
        t20 = s.head(max(1, int(len(s) * 0.2))).sum() / s.sum()
        rep = (c.orders >= 2).mean()
        out.append(finding("gem", f"20% of your customers bring {t20:.0%} of revenue",
                           f"Only {rep:.0%} of customers have ordered more than once: turning first-time buyers into "
                           "repeat buyers is your biggest growth lever.", "neutral"))

    # 5. Visitor intent (SmartCart AI)
    if D.intent_available():
        seg = D.table("segments").set_index("segment")
        k = D.ws()["kpis"]
        hes = seg.loc[["2. Next 20%", "3. Middle 30%"]]
        out.append(finding("cpu", f"{k['ab_test']['sessions_per_day']:,.0f} hesitating visitors a day",
                           f"Our AI reads each visit after 3 clicks. Half of your engaged visits are hesitating: they buy "
                           f"{(hes.buyers.sum() / hes.sessions.sum()):.0%} of the time. With the right nudge, more of them will.",
                           "neutral"))
        cart = D.table("by_cart").set_index("has_cart").purchase_rate
        if 0 in cart and 1 in cart and cart[0] > 0:
            out.append(finding("cart", f"Adding to cart early multiplies buying by {cart[1] / cart[0]:.1f}",
                               f"Visitors who use the cart in their first 3 clicks buy {cart[1]:.0%} of the time, vs "
                               f"{cart[0]:.0%} for those who only browse. Make 'Add to cart' easy and visible.", "good"))

    # 6. Brands
    if len(m["months"]) > 1:
        g = B.brand_growth()
        if len(g):
            up = g.sort_values("abs_change", ascending=False).iloc[0]
            dn = g.sort_values("abs_change").iloc[0]
            if up.abs_change > 0:
                out.append(finding("trend_up", f"{up.brand.title()} is your rising brand",
                                   f"{B.money(up.abs_change)} more revenue than the previous month ({up.growth:+.0%}).",
                                   "good"))
            if dn.abs_change < 0:
                out.append(finding("trend_down", f"{dn.brand.title()} is losing ground",
                                   f"{B.money(-dn.abs_change)} less than the previous month ({dn.growth:+.0%}). Check stock, "
                                   "price and visibility.", "bad"))

    # 7. Best time
    day, hour = best_send_time()
    if day:
        out.append(finding("clock", f"Your customers buy most on {day}s around {hour}:00",
                           "Schedule e-mails and promotions just before this peak.", "neutral"))
    return out


def render_findings(items, n=None):
    tone = {"good": ("#DCFCE7", "#15803d"), "bad": ("#FEE2E2", "#b91c1c"), "neutral": ("#E0EAFF", "#1d4ed8")}
    items = items[:n] if n else items
    cols = st.columns(2)
    for i, f in enumerate(items):
        cols[i % 2].markdown(
            f'<div class="finding"><div class="f-icon" style="background:{tone[f["tone"]][0]}">'
            f'{icon(f["icon"], 21, tone[f["tone"]][1])}</div>'
            f'<div><div class="f-title">{f["title"]}</div><div class="f-text">{f["text"]}</div></div></div>',
            unsafe_allow_html=True)


def money_short(x):
    return B.money(x)


def safe_pct(a, b):
    return a / b if b else np.nan


__all__ = ["pd"]
