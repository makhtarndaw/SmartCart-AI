import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import biz as B
from core import data as D
from core.theme import AQUA, BLUE, ORANGE, chart, footer, insight, kpi, page_header, pct

K = D.kpis()
O = K["original_ids"]

page_header("Settings", "Data & methodology",
            "How the data was prepared, how the prediction problem was framed and what the limits of the approach are.")

tab1, tab2, tab3, tab4 = st.tabs(["Your data & sessions", "Purchase-intent problem", "How the model was trained", "Limits"])

with tab1:
    c = st.columns(4)
    kpi(c[0], "Raw events", f"{K['raw_events']:,}", f"{B.month_label(B.meta()['months'][0])} – {B.month_label(B.meta()['months'][-1])}")
    kpi(c[1], "Duplicates removed", f"{K['raw_events'] - K['clean_events']:,}", "exact duplicate rows")
    kpi(c[2], "Reconstructed sessions", f"{K['sessions']['total']:,}", "30-minute inactivity rule")
    kpi(c[3], "Median session length", f"{K['sessions']['median_events']:.0f} event",
        f"{pct(K['sessions']['purchase_rate'])} include a purchase")

    ev = D.table("event_types")
    em = D.table("events_by_month")
    l, r = st.columns(2)
    with l:
        colors = {"view": AQUA, "cart": BLUE, "remove_from_cart": ORANGE, "purchase": "#0d366b"}
        fig = go.Figure()
        for et in ["view", "cart", "remove_from_cart", "purchase"]:
            d = em[em.event_type == et]
            fig.add_trace(go.Bar(x=d.month, y=d.events, name=et.replace("remove_from_cart", "remove"),
                                 marker=dict(color=colors[et], line=dict(color="white", width=1.5)),
                                 hovertemplate=et + "<br>%{x}: %{y:,.0f}<extra></extra>"))
        fig.update_layout(title="Events per month and type", barmode="stack", yaxis=dict(title=None))
        chart(fig, 360)
    with r:
        st.markdown("##### Why rebuild the sessions?")
        st.markdown(f"""
In the SmartCart training data (REES46, 20.7M events), the platform's own session IDs were unreliable:

- **{O['events_without_session']:,}** events have no session ID
- **{O['sessions_over_two_months']:,}** sessions span two months: they would leak across the monthly split
- **{O['sessions_with_30min_break']:,}** sessions contain a > 30-minute break (several visits merged)
- **{O['sessions_with_several_users']:,}** sessions are shared by several users

That is why SmartCart **never relies on your session IDs**: visits are rebuilt per customer with the **30-minute
inactivity rule** (Google Analytics default). Ties at the same second are ordered deterministically:
view → cart → remove → purchase, then product ID.
""")

with tab2:
    st.dataframe(pd.DataFrame({
        "Element": ["Unit of analysis", "Prediction moment", "Eligible population", "Horizon", "Target",
                    "Incomplete sessions"],
        "Definition": ["A reconstructed session (one user, events ≤ 30 min apart)",
                       "Right after the 3rd event, using only the information available at that moment",
                       "Sessions with ≥ 3 events and no purchase among the first three",
                       "Until the end of the same session",
                       "future_purchase = 1 if at least one purchase happens after the 3rd event",
                       "Sessions ending < 30 min before the end of the data are removed"]}),
        hide_index=True, use_container_width=True)
    f = D.table("funnel")
    f["Sessions"] = f.sessions.map("{:,.0f}".format)
    f["Purchasing sessions"] = f.purchasing_sessions.map("{:,.0f}".format)
    f["Share of all purchasing sessions"] = f.share_of_purchasing_sessions.map("{:.1%}".format)
    st.dataframe(f[["step", "Sessions", "Purchasing sessions", "Share of all purchasing sessions"]]
                 .rename(columns={"step": "Step"}), hide_index=True, use_container_width=True)
    st.markdown("##### Features (first three events only)")
    st.dataframe(pd.DataFrame({
        "Feature": ["action_1, action_2, action_3", "n_cart, n_remove", "avg_price",
                    "distinct_products, distinct_categories", "gap_1_2, gap_2_3", "session_duration_3"],
        "Description": ["Type of each event (one-hot encoded)", "Number of cart additions / removals",
                        "Average price of the products interacted with", "Focused vs exploring visitor",
                        "Seconds between consecutive events", "Seconds between the 1st and 3rd events"]}),
        hide_index=True, use_container_width=True)

with tab3:
    sp = K["split"]
    c = st.columns(3)
    for col, key, label in [(c[0], "train", "Training"), (c[1], "validation", "Validation"), (c[2], "test", "Test")]:
        kpi(col, f"{label} · {sp[key]['period']}", f"{sp[key]['sessions']:,}",
            f"purchase rate {pct(sp[key]['purchase_rate'])}", accent=(key == "test"))
    st.markdown(f"""
- **Chronological split** reproduces a real forecasting situation. Each session belongs to the month of its prediction moment.
- **Temporal leakage**: {sp['crossing_removed']} train/validation sessions whose outcome falls in the next month were removed.
- **Metric**: PR-AUC. Accuracy is meaningless with 89% negatives; a random model scores the purchase rate.
- **Model selection** on January only; the **threshold** (best F1 = {K['model']['threshold']:.3f}) and **segment limits**
  are learned on January; February is touched once, for the final evaluation.
- **Final model**: XGBoost, 30 Optuna trials, no class weighting → calibrated probabilities
  (mean {K['model']['val']['mean_proba']:.3f} vs real {K['model']['val']['base_rate']:.3f}).
""")

with tab4:
    for t, d in [
        ("Selection bias", "Sessions that buy within their first three events (14.2% of purchasing sessions) are "
                           "excluded: the model only applies to visitors still browsing after three actions."),
        ("Limited signal", "Three events cap performance (PR-AUC ≈ 0.20): abandoned carts look exactly like carts "
                           "that will be paid."),
        ("Correlation ≠ causation", "Behavioural results describe associations. Who reacts to an action is learned from "
                                    "Hillstrom (different retailer, e-mail channel) and must be confirmed by an A/B test."),
        ("Session definition", "The 30-minute rule is a convention; a different cut-off would change sessions and target."),
        ("Reproduction", "This app re-runs the notebook pipeline locally (XGBoost 2.1); metrics can differ from the "
                         "Colab run in the third decimal."),
    ]:
        insight(f"<b>{t}.</b> {d}", kind="warn")

footer()
