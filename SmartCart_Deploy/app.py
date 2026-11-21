"""SmartCart AI — Marketing intelligence platform for e-commerce."""
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="SmartCart AI", page_icon="🛍️", layout="wide", initial_sidebar_state="expanded")

from core import biz as B  # noqa: E402
from core import data as D  # noqa: E402
from core.theme import inject_css  # noqa: E402

inject_css()
st.logo(str(Path(__file__).resolve().parent / "assets" / "logo.svg"), size="large")

M = D.ws()["meta"]
with st.sidebar:
    st.markdown(f'<div class="workspace"><div class="ws-label">{"Demo data" if D.is_demo() else "Your data"}</div>'
                f'<div class="ws-name">● {M["name"]}</div>'
                f'<div class="ws-sub">{B.month_label(M["months"][0])} – {B.month_label(M["months"][-1])}</div></div>',
                unsafe_allow_html=True)

pages = {
    "": [
        st.Page("views/home.py", title="Home", icon=":material/home:", default=True),
    ],
    "Your journey": [
        st.Page("views/j_import.py", title="1 · Import data", icon=":material/cloud_upload:"),
        st.Page("views/j_diagnosis.py", title="2 · Diagnosis", icon=":material/insights:"),
        st.Page("views/j_customers.py", title="3 · Customers", icon=":material/groups:"),
        st.Page("views/j_products.py", title="4 · Products", icon=":material/inventory_2:"),
        st.Page("views/j_actions.py", title="5 · Action plan", icon=":material/rocket_launch:"),
    ],
    "Expert space": [
        st.Page("views/trends.py", title="Trends & forecast", icon=":material/trending_up:"),
        st.Page("views/cart.py", title="Cart analysis", icon=":material/remove_shopping_cart:"),
        st.Page("views/journey.py", title="Visit behaviour", icon=":material/route:"),
        st.Page("views/simulator.py", title="AI · score a visitor", icon=":material/bolt:"),
        st.Page("views/segments.py", title="AI · intent groups", icon=":material/target:"),
        st.Page("views/model.py", title="AI · model performance", icon=":material/monitoring:"),
        st.Page("views/explain.py", title="AI · explainability", icon=":material/psychology:"),
        st.Page("views/uplift.py", title="Campaign evidence", icon=":material/campaign:"),
        st.Page("views/abtest.py", title="A/B test planner", icon=":material/experiment:"),
        st.Page("views/method.py", title="Methodology", icon=":material/menu_book:"),
    ],
}

st.navigation(pages).run()
