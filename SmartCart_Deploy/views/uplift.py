import plotly.graph_objects as go
import streamlit as st

from core import data as D
from core.theme import BLUE, INK_3, ORANGE, chart, footer, insight, kpi, page_header, pct

K = D.kpis()

page_header("Activation", "Campaign uplift — causal evidence",
            "Our click data contains no marketing action, so it cannot tell who <i>reacts</i> to one. We use the "
            "<b>Hillstrom</b> experiment: 64,000 customers randomly assigned to a men's e-mail, a women's e-mail or no "
            "e-mail. Randomisation makes the difference between groups a causal effect.")

if not K.get("hillstrom", {}).get("available"):
    st.warning("Hillstrom data was not available when the artifacts were built. Re-run `python build.py` with "
               "an internet connection.")
    st.stop()

EMAIL_COLORS = {"Mens E-Mail": BLUE, "Womens E-Mail": ORANGE}
EMAIL_LABELS = {"Mens E-Mail": "Men's e-mail", "Womens E-Mail": "Women's e-mail"}

means = D.table("hill_means").set_index("segment")
ov = D.table("hill_overall")
c = st.columns(4)
kpi(c[0], "Visit rate · no e-mail", pct(means.loc["No E-Mail", "visit"]), "control group")
kpi(c[1], "Visit rate · men's e-mail", pct(means.loc["Mens E-Mail", "visit"]),
    f"+{ov[(ov.outcome == 'visit') & (ov.email == 'Mens E-Mail')].effect.iloc[0] * 100:.1f} pts", accent=True)
kpi(c[2], "Conversion · men's e-mail", pct(means.loc["Mens E-Mail", "conversion"], 2),
    f"vs {pct(means.loc['No E-Mail', 'conversion'], 2)} without — more than ×2")
kpi(c[3], "Extra spend / customer", f"${ov[(ov.outcome == 'spend') & (ov.email == 'Mens E-Mail')].effect.iloc[0]:.2f}",
    "men's e-mail vs nothing")


def effect_plot(df, title, ytitle, fmt=".1%", scale=1.0, shift=0.0):
    fig = go.Figure()
    for email in ["Mens E-Mail", "Womens E-Mail"]:
        d = df[df.email == email]
        y = d.effect * scale - shift
        lo = d.ci_low * scale - shift
        hi = d.ci_high * scale - shift
        fig.add_trace(go.Bar(
            x=d.group, y=y, name=EMAIL_LABELS[email], marker=dict(color=EMAIL_COLORS[email], cornerradius=4),
            error_y=dict(type="data", symmetric=False, array=hi - y, arrayminus=y - lo, color="#334155",
                         thickness=1.5, width=5),
            customdata=list(zip(lo, hi)),
            hovertemplate=EMAIL_LABELS[email] + "<br>%{x}<br>Effect %{y:" + fmt + "}<br>95% CI [%{customdata[0]:" + fmt +
                          "}, %{customdata[1]:" + fmt + "}]<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=INK_3, width=1))
    fig.update_layout(title=title, barmode="group", yaxis=dict(tickformat=fmt, title=ytitle), bargroupgap=0.08)
    return fig


tab1, tab2, tab3 = st.tabs(["Who reacts? (propensity)", "Matching the message (history)", "Is it profitable?"])

with tab1:
    chart(effect_plot(D.table("hill_propensity"), "Effect of the e-mail on the visit rate, by propensity to visit",
                      "Increase in visit rate (pts)"), 400)
    insight("Both e-mails raise visits in <b>every</b> group (all 95% intervals above zero). The men's e-mail effect "
            "grows with propensity (≈ 6 → 9 pts); the women's e-mail shows no pattern. <b>Propensity alone does not "
            "tell where an action works — it depends on the action.</b>")

with tab2:
    chart(effect_plot(D.table("hill_history"), "Effect of the e-mail on the visit rate, by past purchases",
                      "Increase in visit rate (pts)"), 400)
    insight("Customers who only bought men's products: men's e-mail +7 pts vs women's e-mail +1 pt — intervals far "
            "apart. <b>A message that matches past behaviour works much better than a generic one</b> → SmartCart AI "
            "sends cart reminders to cart users and viewed-product recommendations to browsers.")

with tab3:
    l, r = st.columns([1, 2.4])
    with l:
        margin = st.slider("Gross margin on sales", 0.10, 0.60, 0.30, 0.05, format="%.2f")
        cost = st.slider("Cost per e-mail ($)", 0.00, 0.30, 0.05, 0.01)
        st.caption("Profit per customer = margin × extra spend − cost. Notebook assumptions: 30% margin, $0.05.")
    with r:
        chart(effect_plot(D.table("hill_spend"), "Expected profit per customer vs sending nothing ($)",
                          "Profit per customer ($)", fmt="$.3f", scale=margin, shift=cost), 400)
    sp = D.table("hill_spend")
    sig = ((sp.ci_low * margin - cost) > 0).sum()
    insight(f"With these assumptions, <b>{sig} of 8</b> e-mail × group combinations are significantly profitable. Spend "
            "is very noisy (few buyers), so profitability cannot be assumed — <b>the A/B test must track margin, not "
            "only conversion.</b>", kind="warn" if sig < 4 else "insight")

st.markdown('<div class="section">Three lessons applied to SmartCart AI</div>', unsafe_allow_html=True)
c = st.columns(3)
for col, (t, b) in zip(c, [
    ("1 · Scores say who to look at", "not which action works. The effect depends on the action itself."),
    ("2 · Match the behaviour", "Cart reminder for cart users, viewed-product recommendations for browsers."),
    ("3 · Cheap actions for likely buyers", "Discounts are also paid by customers who would have bought anyway."),
]):
    col.markdown(f'<div class="card"><h4>{t}</h4><p>{b}</p></div>', unsafe_allow_html=True)

footer()
