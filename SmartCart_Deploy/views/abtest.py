import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from core import data as D
from core.theme import BLUE, INK_3, ORANGE, chart, footer, insight, kpi, page_header, pct

K = D.kpis()
AB = K["ab_test"]

page_header("Activation", "A/B test planner",
            "Validate the recommendations on the site itself: target the <b>next 20%</b> and <b>middle 30%</b>, "
            "randomise <b>by user</b> (not session), treatment = recommended action, control = nothing.")


def sample_size(p1, uplift, alpha=0.05, power=0.8):
    p2 = p1 * (1 + uplift)
    z_a, z_b = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    p_mean = (p1 + p2) / 2
    n = (z_a * np.sqrt(2 * p_mean * (1 - p_mean)) + z_b * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2 / (p2 - p1) ** 2
    return int(np.ceil(n))


l, r = st.columns([1, 1.6], gap="large")
with l:
    with st.container(border=True):
        st.markdown("##### Test design")
        unit = st.radio("Primary metric", ["user", "session"], horizontal=True,
                        format_func=lambda v: "User · 7-day conversion" if v == "user" else "Session purchase rate")
        default_p = AB["base_rate"] if unit == "user" else AB["session_purchase_rate"]
        default_t = AB["users_per_day"] if unit == "user" else AB["sessions_per_day"]
        dp = float(min(max(round(default_p, 4), 0.001), 0.9)) if default_p == default_p else 0.1
        base = st.number_input("Baseline conversion (control)", 0.001, 0.9, dp, 0.005, format="%.4f", key=f"base_{unit}")
        traffic = st.number_input(f"Eligible {unit}s per day", 1, 1_000_000, int(max(default_t, 1)), 10, key=f"tr_{unit}")
        uplift = st.slider("Minimum detectable relative uplift", 0.02, 0.30, 0.10, 0.01, format="%.2f")
        c = st.columns(2)
        alpha = c[0].selectbox("Significance α", [0.01, 0.05, 0.10], index=1)
        power = c[1].selectbox("Power 1−β", [0.8, 0.9], index=0)
        st.caption("Defaults computed on your data for the targeted segments. Minimum duration = 14 days to "
                   "cover two full weekly cycles" + (" + 7-day observation window." if unit == "user" else "."))

n = sample_size(base, uplift, alpha, power)
days = max(14, int(np.ceil(2 * n / traffic)))
total = days + (7 if unit == "user" else 0)
with r:
    c = st.columns(3)
    kpi(c[0], f"{unit.capitalize()}s per group", f"{n:,}", f"{2 * n:,} in total")
    kpi(c[1], "Recruitment", f"{days} days", f"{int(np.ceil(2 * n / traffic))} days of traffic needed")
    kpi(c[2], "Total duration", f"{total} days", f"{pct(base)} → {pct(base * (1 + uplift))}", accent=True)

    ups = np.linspace(0.02, 0.30, 57)
    ns = [sample_size(base, u, alpha, power) for u in ups]
    fig = go.Figure(go.Scatter(x=ups, y=ns, mode="lines", line=dict(color=BLUE, width=2.5),
                               hovertemplate="Uplift %{x:.0%}<br>%{y:,.0f} per group<extra></extra>"))
    fig.add_trace(go.Scatter(x=[uplift], y=[n], mode="markers", marker=dict(size=12, color=ORANGE,
                                                                             line=dict(color="white", width=2)),
                             hoverinfo="skip"))
    fig.update_layout(title="Sample size per group vs detectable uplift", showlegend=False,
                      xaxis=dict(tickformat=".0%", title="Relative uplift to detect"),
                      yaxis=dict(type="log", title="Per group (log scale)"))
    chart(fig, 330)

st.markdown('<div class="section">Protocol</div>', unsafe_allow_html=True)
c = st.columns(2)
with c[0]:
    for i, (t, d) in enumerate([
        ("Population", "Visitors scored in the next 20% and middle 30% after their third action."),
        ("Randomisation", "By user ID (hash), so a person always sees the same version across sessions."),
        ("Treatment", "Cart reminder if the visitor has a cart, viewed-product recommendations otherwise."),
        ("Control", "No action — measures the incremental effect, not the propensity."),
    ], 1):
        st.markdown(f'<div class="step"><div class="num">{i}</div><div class="txt"><b>{t}.</b> {d}</div></div>',
                    unsafe_allow_html=True)
with c[1]:
    for i, (t, d) in enumerate([
        ("Primary KPI", "Incremental conversion = treatment − control purchase rate."),
        ("Guardrail", "Margin after action cost — Hillstrom shows profit is much noisier than conversion."),
        ("Heterogeneity", "Read effects by segment and with / without cart (pre-registered)."),
        ("Duration", "At least two full weeks to neutralise day-of-week effects."),
    ], 5):
        st.markdown(f'<div class="step"><div class="num">{i}</div><div class="txt"><b>{t}.</b> {d}</div></div>',
                    unsafe_allow_html=True)

insight("Sub-group analyses (segment × cart) divide the sample: plan the size on the smallest sub-group you want to "
        "read, or accept wider intervals there. Measuring profit precisely needs a larger sample than conversion.",
        kind="warn")

footer()
