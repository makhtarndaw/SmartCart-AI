import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="SmartCart AI", page_icon="🛍️", layout="wide")
st.markdown("""
<style>
.stApp{background:#F8F7FC}
section[data-testid="stSidebar"]{background:#211A35}
section[data-testid="stSidebar"] *{color:#F7F4FF!important}
.hero{background:linear-gradient(115deg,#211A35,#6546C7);padding:28px 32px;border-radius:18px;color:white;margin-bottom:22px}
.hero h1{color:white;margin:0}.hero p{color:#E8E0FF;margin:8px 0 0}
div.stButton>button{background:#6D4AFF;color:white;border:0;border-radius:9px}
</style>""", unsafe_allow_html=True)

ROOT=Path(__file__).parent
FEATURES=["action_1","action_2","action_3","avg_price","distinct_products","session_duration_3"]

@st.cache_data
def demo_data():
    rng=np.random.default_rng(42); n=500
    actions=np.array(["view","cart","remove_from_cart"])
    return pd.DataFrame({
      "action_1":rng.choice(actions,n,p=[.82,.15,.03]),
      "action_2":rng.choice(actions,n,p=[.76,.20,.04]),
      "action_3":rng.choice(actions,n,p=[.72,.24,.04]),
      "avg_price":np.round(rng.lognormal(3.1,.7,n),2),
      "distinct_products":rng.integers(1,6,n),
      "session_duration_3":rng.integers(5,900,n)})

def read_upload(f):
    return pd.read_parquet(f) if f.name.lower().endswith((".parquet",".pq")) else pd.read_csv(f)

with st.sidebar:
    st.markdown("## 🛍️ SmartCart AI")
    st.caption("MARKETING INTELLIGENCE")
    page=st.radio("NAVIGATION",["Executive Dashboard","Data Center","Customer Intelligence","Purchase Prediction","Marketing Studio","Campaign Performance"])
    source=st.radio("WORKSPACE",["Demo data","My company data"])
    st.caption("Prototype • English")

if "company_df" not in st.session_state: st.session_state.company_df=None
df=st.session_state.company_df if source=="My company data" and st.session_state.company_df is not None else demo_data()
st.markdown('<div class="hero"><h1>SmartCart AI</h1><p>Turn customer data into smarter marketing decisions.</p></div>',unsafe_allow_html=True)

if page=="Executive Dashboard":
    st.subheader("Executive overview")
    st.write("Understand customer activity and purchase intent without needing to interpret raw data.")
    a,b,c=st.columns(3); a.metric("Sessions available",f"{len(df):,}"); b.metric("Data source","Demo" if source=="Demo data" else "Company upload"); c.metric("Model status","Ready to connect")
    counts=pd.concat([df[x] for x in FEATURES[:3]]).value_counts().rename_axis("Action").reset_index(name="Occurrences")
    st.plotly_chart(px.bar(counts,x="Action",y="Occurrences",title="Early-session action mix",color_discrete_sequence=["#6D4AFF"]),use_container_width=True)
    st.info("Demo data are synthetic and illustrate the interface only. They are not actual REES46 results.")

elif page=="Data Center":
    st.subheader("Upload your company data")
    st.write("Upload session-level CSV or Parquet data. The current prototype expects the six model input fields.")
    f=st.file_uploader("Choose a CSV or Parquet file",type=["csv","parquet","pq"])
    if f:
        try:
            candidate=read_upload(f); missing=[x for x in FEATURES if x not in candidate.columns]
            st.write(f"**{len(candidate):,} rows** · **{len(candidate.columns)} columns**")
            st.dataframe(candidate.head(10),use_container_width=True)
            if missing: st.error("Missing required fields: "+", ".join(missing))
            else:
                st.success("Required columns detected.")
                if st.button("Use this dataset"): st.session_state.company_df=candidate; st.success("Dataset loaded for this session.")
        except Exception as e: st.error(f"Unable to read file: {e}")
    with st.expander("Expected fields"): st.code("\\n".join(FEATURES))
    st.caption("Matching column names do not guarantee compatibility. Feature definitions and data distributions must also match the REES46 training data.")

elif page=="Customer Intelligence":
    st.subheader("Customer intelligence")
    long=pd.concat([df[x].rename("Action") for x in FEATURES[:3]],ignore_index=True)
    counts=long.value_counts().rename_axis("Action").reset_index(name="Occurrences")
    st.plotly_chart(px.pie(counts,names="Action",values="Occurrences",hole=.5,title="Early-session action mix",color_discrete_sequence=px.colors.sequential.Purples),use_container_width=True)
    top=counts.iloc[0]["Action"]
    st.info(f"The most frequent early-session action is **{top}**. This is a behavioural observation, not proof of purchase intent.")

elif page=="Purchase Prediction":
    st.subheader("Purchase prediction")
    st.write("Estimate the purchase likelihood for one session using its first three events.")
    model_path=ROOT/"models"/"best_xgb_model.pkl"; prep_path=ROOT/"models"/"preprocessor.pkl"
    if not model_path.exists() or not prep_path.exists():
        st.warning("The trained model artifacts are not installed yet. Export best_xgb_model and preprocessor from Colab and place them in the models folder.")
        st.code("models/best_xgb_model.pkl\\nmodels/preprocessor.pkl")
        st.caption("No illustrative or synthetic probability is presented as a real model prediction.")
    else:
        import joblib
        model=joblib.load(model_path); prep=joblib.load(prep_path)
        with st.form("predict"):
            vals=[st.selectbox(f"Action {i}",["view","cart","remove_from_cart"],key=f"a{i}") for i in range(1,4)]
            x,y,z=st.columns(3)
            price=x.number_input("Average price",min_value=0.0,value=25.0)
            distinct=y.number_input("Distinct products",min_value=1,value=2)
            duration=z.number_input("Session duration (seconds)",min_value=0,value=120)
            go=st.form_submit_button("Predict purchase likelihood")
        if go:
            row=pd.DataFrame([dict(zip(FEATURES,vals+[price,distinct,duration]))])
            proba=float(model.predict_proba(prep.transform(row))[0][1])
            st.metric("Estimated purchase probability",f"{proba:.1%}"); st.progress(proba)
            st.caption("This is a model estimate, not a guarantee of purchase.")

elif page=="Marketing Studio":
    st.subheader("Marketing studio")
    st.write("Practical marketing ideas based on customer behaviour. These actions should be tested before being scaled.")
    items=[
      ("Personalized recommendations","Suggest products related to the visitor's browsing activity."),
      ("Targeted incentive experiment","Test a relevant offer with a defined visitor segment and compare it with a control group."),
      ("Cross-selling","Recommend complementary products when browsing behaviour suggests relevant interest."),
      ("Customer journey improvement","Review where visitors stop engaging and test clearer product information or calls to action.")]
    for title,body in items:
        with st.container(border=True): st.markdown("**"+title+"**"); st.write(body)
    st.warning("The model predicts purchase likelihood; it does not estimate whether a discount or recommendation causes a purchase.")

else:
    st.subheader("Campaign performance")
    st.write("Measure whether marketing actions improve business outcomes.")
    st.info("Campaign data are not connected yet. To calculate campaign impact, upload exposure and outcome data, ideally including treatment and control groups.")
    x,y,z=st.columns(3); x.metric("Conversion rate","Awaiting data"); y.metric("Incremental lift","Awaiting experiment"); z.metric("Revenue per visitor","Awaiting data")
