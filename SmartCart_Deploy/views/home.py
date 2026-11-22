import numpy as np
import pandas as pd
import streamlit as st

from core import biz as B
from core import data as D
from core.icons import icon
from core.theme import footer

go = st.query_params.get("go")
if go:
    st.query_params.clear()
    if go == "demo":
        D.set_workspace(None)
        st.switch_page("views/j_diagnosis.py")
    st.switch_page("views/j_import.py")

W = D.ws()
MC = D.model_card_kpis()
mk = B.monthly_kpis()
cur = mk.iloc[-1]
prev = mk.iloc[-2] if len(mk) > 1 else cur

# --------------------------------------------------------------- live preview data --
d = B.load("daily").copy()
d["day"] = pd.to_datetime(d.day)
wk = d.set_index("day").revenue.resample("W").sum().iloc[1:-1].tail(14)
y = wk.values
pts = " ".join(f"{i * 300 / max(len(y) - 1, 1):.1f},{70 - 60 * (v - y.min()) / max(y.max() - y.min(), 1):.1f}"
               for i, v in enumerate(y))
area = f"0,80 {pts} 300,80"
seg = D.table("segments").set_index("segment").reindex(D.SEGMENTS) if D.intent_available() else None
if seg is not None:
    sh = (seg.sessions / seg.sessions.sum()).values
    cuts = np.cumsum(sh) * 360
    donut = (f"conic-gradient(#60a5fa 0 {cuts[0]:.0f}deg, #3b82f6 {cuts[0]:.0f}deg {cuts[1]:.0f}deg, "
             f"#6366f1 {cuts[1]:.0f}deg {cuts[2]:.0f}deg, rgba(255,255,255,.12) {cuts[2]:.0f}deg 360deg)")
    hes = int(W["kpis"]["ab_test"]["sessions_per_day"])
else:
    donut, hes = "conic-gradient(#3b82f6 0 360deg)", 0
camps = B.campaigns()[:3]
rev_delta = cur.revenue / prev.revenue - 1 if prev.revenue else 0
dec = D.static_table("deciles_test")
lift = float(dec.lift.iloc[0])
events_m = MC["raw_events"] / 1e6

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700;800&display=swap');
.block-container {{ padding-top: 4rem !important; max-width: 1320px !important; }}
.lp-hero {{ position:relative; overflow:hidden; border-radius:28px; background:#060A18; padding:3.6rem 3.4rem 3rem;
  color:white; isolation:isolate; }}
.lp-hero:before {{ content:""; position:absolute; inset:0; z-index:-1;
  background-image:linear-gradient(rgba(255,255,255,.045) 1px, transparent 1px),
                   linear-gradient(90deg, rgba(255,255,255,.045) 1px, transparent 1px);
  background-size:44px 44px; mask-image:radial-gradient(ellipse 80% 70% at 50% 30%, black 30%, transparent 75%); }}
.lp-orb {{ position:absolute; border-radius:50%; filter:blur(70px); z-index:-1; opacity:.75; }}
.lp-orb.o1 {{ width:520px; height:520px; background:#2563eb; top:-220px; right:-60px; animation:lpDrift 14s ease-in-out infinite; }}
.lp-orb.o2 {{ width:420px; height:420px; background:#7c3aed; bottom:-240px; left:-120px; animation:lpDrift 18s ease-in-out infinite reverse; }}
.lp-orb.o3 {{ width:300px; height:300px; background:#06b6d4; top:30%; left:40%; opacity:.35; animation:lpDrift 22s ease-in-out infinite; }}
@keyframes lpDrift {{ 0%,100% {{ transform:translate(0,0) scale(1); }} 50% {{ transform:translate(40px,30px) scale(1.08); }} }}
@keyframes lpFloat {{ 0%,100% {{ transform:translateY(0) rotate(-1deg); }} 50% {{ transform:translateY(-12px) rotate(-1deg); }} }}
@keyframes lpUp {{ from {{ opacity:0; transform:translateY(18px); }} to {{ opacity:1; transform:none; }} }}
@keyframes lpShine {{ to {{ background-position:200% center; }} }}
@keyframes lpPulse {{ 0%,100% {{ box-shadow:0 0 0 0 rgba(52,211,153,.6); }} 50% {{ box-shadow:0 0 0 6px rgba(52,211,153,0); }} }}
@keyframes lpDraw {{ from {{ stroke-dashoffset:900; }} to {{ stroke-dashoffset:0; }} }}
.lp-grid {{ display:grid; grid-template-columns:minmax(0,1.15fr) minmax(0,.9fr); gap:3rem; align-items:center; }}
@media (max-width: 1380px) {{ .lp-grid {{ grid-template-columns:1fr; gap:2.6rem; }} .lp-mock {{ max-width:560px; }} .lp-float.f1 {{ right:-14px; }} }}
@media (max-width: 700px) {{ .lp-hero {{ padding:2.4rem 1.4rem; }} .lp-float {{ display:none; }} }}
.lp-pill {{ display:inline-flex; align-items:center; gap:.55rem; padding:.42rem .9rem .42rem .5rem; border-radius:999px;
  background:rgba(255,255,255,.06); border:1px solid rgba(255,255,255,.12); font-size:.82rem; color:#cbd5e1;
  animation:lpUp .6s ease both; }}
.lp-pill b {{ background:linear-gradient(90deg,#60a5fa,#a78bfa); color:#0b1020; padding:.12rem .55rem; border-radius:999px;
  font-size:.7rem; letter-spacing:.06em; }}
.lp-h1 {{ font-family:Sora, Inter, sans-serif; font-size:clamp(2.3rem, 4vw, 3.55rem); line-height:1.04; font-weight:800;
  letter-spacing:-.035em; margin:1.3rem 0 1.2rem; color:white; animation:lpUp .7s .1s ease both; }}
.lp-h1 span {{ background:linear-gradient(90deg,#93c5fd,#a78bfa,#67e8f9,#93c5fd); background-size:200% auto;
  -webkit-background-clip:text; background-clip:text; color:transparent; animation:lpShine 6s linear infinite; }}
.lp-sub {{ font-size:1.15rem; line-height:1.6; color:#94a3b8; max-width:560px; animation:lpUp .7s .2s ease both; }}
.lp-ctas {{ display:flex; gap:.8rem; margin-top:2rem; flex-wrap:wrap; }}
.lp-btn {{ display:inline-flex; align-items:center; justify-content:center; gap:.5rem; height:3.3rem; padding:0 1.7rem; border-radius:14px; font-weight:700; font-size:1rem; text-decoration:none !important; transition:all .2s ease; }}
.lp-btn.pri {{ background:linear-gradient(135deg,#3b82f6,#7c3aed); color:white !important; box-shadow:0 12px 30px -8px rgba(99,102,241,.7); }}
.lp-btn.pri:hover {{ filter:brightness(1.12); transform:translateY(-1px); }}
.lp-btn.sec {{ background:rgba(255,255,255,.06); color:white !important; border:1px solid rgba(255,255,255,.2); }}
.lp-btn.sec:hover {{ background:rgba(255,255,255,.12); }}
.lp-trust {{ display:flex; gap:1.2rem; margin-top:1.4rem; color:#7c8aa5; font-size:.8rem; flex-wrap:wrap; animation:lpUp .7s .4s ease both; }}
.lp-trust div {{ display:flex; align-items:center; gap:.45rem; }}
.lp-mock {{ position:relative; animation:lpUp .9s .25s ease both; }}
.lp-card {{ background:linear-gradient(160deg, rgba(255,255,255,.10), rgba(255,255,255,.03)); border:1px solid rgba(255,255,255,.14);
  border-radius:20px; backdrop-filter:blur(14px); -webkit-backdrop-filter:blur(14px);
  box-shadow:0 30px 80px -20px rgba(0,0,0,.6), inset 0 1px 0 rgba(255,255,255,.12); }}
.lp-main {{ padding:1.3rem 1.4rem; animation:lpFloat 7s ease-in-out infinite; }}
.lp-top {{ display:flex; justify-content:space-between; align-items:center; font-size:.78rem; color:#94a3b8; }}
.lp-live {{ display:flex; align-items:center; gap:.4rem; color:#6ee7b7; font-weight:600; }}
.lp-live i {{ width:7px; height:7px; border-radius:50%; background:#34d399; animation:lpPulse 2s infinite; display:block; }}
.lp-big {{ font-family:Sora, Inter; font-size:2.1rem; font-weight:700; color:white; margin:.5rem 0 0; letter-spacing:-.02em; }}
.lp-delta {{ font-size:.82rem; color:{'#6ee7b7' if rev_delta >= 0 else '#fca5a5'}; font-weight:600; }}
.lp-spark polyline {{ stroke-dasharray:900; animation:lpDraw 2.4s .4s ease both; }}
.lp-row {{ display:grid; grid-template-columns:1fr 1fr; gap:.8rem; margin-top:1rem; }}
.lp-mini {{ background:rgba(255,255,255,.05); border:1px solid rgba(255,255,255,.08); border-radius:14px; padding:.8rem .9rem; }}
.lp-mini .k {{ font-size:.7rem; color:#94a3b8; text-transform:uppercase; letter-spacing:.08em; font-weight:600; }}
.lp-mini .v {{ font-size:1.25rem; font-weight:700; color:white; margin-top:.2rem; }}
.lp-donut {{ width:64px; height:64px; border-radius:50%; background:{donut}; position:relative; flex:none; }}
.lp-donut:after {{ content:""; position:absolute; inset:11px; border-radius:50%; background:#111a33; }}
.lp-float {{ position:absolute; padding:.85rem 1rem; display:flex; gap:.7rem; align-items:center; width:270px; }}
.lp-float.f1 {{ right:-28px; bottom:-56px; animation:lpFloat 6s 1s ease-in-out infinite; }}
.lp-float .ic {{ width:36px; height:36px; border-radius:10px; display:flex; align-items:center; justify-content:center;
  background:linear-gradient(135deg,#2563eb,#7c3aed); color:white; flex:none; }}
.lp-float .t {{ font-size:.8rem; color:#e2e8f0; font-weight:600; line-height:1.3; }}
.lp-float .s {{ font-size:.72rem; color:#94a3b8; }}
  box-shadow:0 12px 30px -8px rgba(99,102,241,.7); }}
  border:1px solid rgba(255,255,255,.2) !important; }}
.lp-stats {{ display:grid; grid-template-columns:repeat(4,1fr); gap:0; margin:1.6rem 0 4.5rem; border:1px solid #e3e7ef;
  border-radius:20px; background:white; overflow:hidden; }}
@media (max-width: 900px) {{ .lp-stats {{ grid-template-columns:repeat(2,1fr); }} }}
.lp-stat {{ padding:1.6rem 1.8rem; border-right:1px solid #eef0f5; }}
.lp-stat:last-child {{ border-right:none; }}
.lp-stat .n {{ font-family:Sora, Inter; font-size:2.2rem; font-weight:800; letter-spacing:-.03em;
  background:linear-gradient(135deg,#0f172a,#3b82f6); -webkit-background-clip:text; background-clip:text; color:transparent; }}
.lp-stat .l {{ color:#64748b; font-size:.9rem; margin-top:.2rem; }}
.lp-eyebrow {{ font-size:.78rem; font-weight:700; letter-spacing:.16em; text-transform:uppercase; color:#3b82f6; text-align:center; }}
.lp-h2 {{ font-family:Sora, Inter; font-size:clamp(1.8rem,3vw,2.6rem); font-weight:800; letter-spacing:-.03em; color:#0f172a;
  text-align:center; margin:.6rem auto .8rem; max-width:760px; line-height:1.12; }}
.lp-lead {{ text-align:center; color:#64748b; font-size:1.08rem; max-width:640px; margin:0 auto 2.6rem; }}
.lp-feat {{ display:grid; grid-template-columns:repeat(3,1fr); gap:1.1rem; margin-bottom:5rem; }}
@media (max-width: 900px) {{ .lp-feat {{ grid-template-columns:1fr; }} }}
.lp-f {{ background:white; border:1px solid #e3e7ef; border-radius:20px; padding:1.6rem; transition:all .25s ease; position:relative; overflow:hidden; }}
.lp-f:hover {{ transform:translateY(-4px); box-shadow:0 24px 50px -24px rgba(37,99,235,.35); border-color:#c7d7fe; }}
.lp-f .ic {{ width:46px; height:46px; border-radius:13px; display:flex; align-items:center; justify-content:center; color:#2563eb;
  background:linear-gradient(135deg,#eff6ff,#eef2ff); border:1px solid #dbeafe; margin-bottom:1rem; }}
.lp-f h4 {{ font-family:Sora, Inter; font-size:1.08rem; color:#0f172a; margin:0 0 .4rem; padding:0; }}
.lp-f p {{ color:#64748b; font-size:.93rem; margin:0; line-height:1.55; }}
.lp-f .tag {{ position:absolute; top:1.2rem; right:1.2rem; font-size:.66rem; font-weight:700; letter-spacing:.08em;
  color:#7c3aed; background:#f5f3ff; border:1px solid #ede9fe; padding:.2rem .5rem; border-radius:6px; }}
.lp-steps {{ display:grid; grid-template-columns:repeat(3,1fr); gap:1.4rem; position:relative; margin-bottom:5rem; }}
@media (max-width: 900px) {{ .lp-steps {{ grid-template-columns:1fr; }} }}
.lp-steps:before {{ content:""; position:absolute; top:27px; left:12%; right:12%; height:2px;
  background:linear-gradient(90deg,#bfdbfe,#c4b5fd); z-index:0; }}
.lp-s {{ text-align:center; position:relative; z-index:1; }}
.lp-s .num {{ width:56px; height:56px; margin:0 auto 1rem; border-radius:16px; background:#0f172a; color:white;
  display:flex; align-items:center; justify-content:center; box-shadow:0 0 0 6px #f6f7fb; }}
.lp-s h4 {{ font-family:Sora, Inter; font-size:1.1rem; color:#0f172a; margin:0 0 .35rem; padding:0; }}
.lp-s p {{ color:#64748b; font-size:.93rem; max-width:300px; margin:0 auto; }}
.lp-ai {{ border-radius:28px; background:#060A18; color:white; padding:3.4rem; position:relative; overflow:hidden;
  isolation:isolate; margin-bottom:5rem; }}
.lp-ai .grid2 {{ display:grid; grid-template-columns:1.1fr 1fr; gap:3rem; align-items:center; }}
@media (max-width: 900px) {{ .lp-ai .grid2 {{ grid-template-columns:1fr; }} }}
.lp-ai h2 {{ font-family:Sora, Inter; font-size:clamp(1.7rem,2.6vw,2.4rem); font-weight:800; letter-spacing:-.03em; color:white;
  line-height:1.12; margin:.6rem 0 1rem; padding:0; }}
.lp-ai p {{ color:#94a3b8; font-size:1.02rem; line-height:1.65; }}
.lp-ai .eb {{ font-size:.75rem; font-weight:700; letter-spacing:.16em; text-transform:uppercase; color:#93c5fd; }}
.lp-metrics {{ display:grid; grid-template-columns:1fr 1fr; gap:.9rem; }}
.lp-m {{ padding:1.3rem; border-radius:18px; background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.1); }}
.lp-m .n {{ font-family:Sora, Inter; font-size:2rem; font-weight:800; color:white; letter-spacing:-.02em; }}
.lp-m .n small {{ font-size:1rem; color:#64748b; font-weight:600; }}
.lp-m .l {{ color:#94a3b8; font-size:.86rem; margin-top:.25rem; }}
.lp-bar {{ height:6px; border-radius:6px; background:rgba(255,255,255,.08); margin-top:.8rem; overflow:hidden; }}
.lp-bar i {{ display:block; height:100%; border-radius:6px; background:linear-gradient(90deg,#3b82f6,#a78bfa); }}
.lp-final {{ text-align:center; padding:3.6rem 2rem 3.2rem; border-radius:28px; margin-bottom:1rem;
  background:radial-gradient(600px 220px at 50% 0%, rgba(59,130,246,.18), transparent), white; border:1px solid #e3e7ef; }}
.demo-banner {{ display:none; }}
</style>

<div class="lp-hero">
  <div class="lp-orb o1"></div><div class="lp-orb o2"></div><div class="lp-orb o3"></div>
  <div class="lp-grid">
    <div>
      <div class="lp-pill"><b>NEW</b> Purchase-intent AI trained on {events_m:.1f}M e-commerce events</div>
      <div class="lp-h1">Know&nbsp;who will&nbsp;buy.<br><span>Before&nbsp;they&nbsp;do.</span></div>
      <div class="lp-sub">SmartCart turns your store's clicks into a clear diagnosis and a ready-to-run marketing plan.
        Upload your data, and in under a minute see your customers, your products and the actions that will pay off.</div>
      <div class="lp-ctas">
        <a class="lp-btn pri" href="?go=import" target="_self">Analyse my store {icon("arrow", 18, "white", 2.2)}</a>
        <a class="lp-btn sec" href="?go=demo" target="_self">Explore the demo</a>
      </div>
      <div class="lp-trust">
        <div>{icon("shield", 16, "#64748b")} Private by design</div>
        <div>{icon("zap", 16, "#64748b")} Results in under a minute</div>
        <div>{icon("check", 16, "#64748b")} No data team needed</div>
      </div>
    </div>
    <div class="lp-mock">
      <div class="lp-card lp-main">
        <div class="lp-top"><span>Revenue · {B.month_label(cur.month)}</span><span class="lp-live"><i></i>Live workspace</span></div>
        <div class="lp-big">{B.money(cur.revenue)} <span class="lp-delta">{'+' if rev_delta >= 0 else ''}{rev_delta:.1%}</span></div>
        <svg class="lp-spark" viewBox="0 0 300 80" width="100%" height="90" preserveAspectRatio="none">
          <defs><linearGradient id="lpg" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="#3b82f6" stop-opacity=".45"/>
          <stop offset="1" stop-color="#3b82f6" stop-opacity="0"/></linearGradient></defs>
          <polygon points="{area}" fill="url(#lpg)"/>
          <polyline points="{pts}" fill="none" stroke="#93c5fd" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>
        </svg>
        <div class="lp-row">
          <div class="lp-mini" style="display:flex;gap:.8rem;align-items:center">
            <div class="lp-donut"></div>
            <div><div class="k">Hesitating visitors</div><div class="v">{hes:,}<span style="font-size:.75rem;color:#94a3b8"> / day</span></div></div>
          </div>
          <div class="lp-mini"><div class="k">Customers</div><div class="v">{cur.buyers:,.0f}</div>
            <div style="font-size:.72rem;color:#94a3b8">conversion {cur.conversion:.1%}</div></div>
        </div>
        <div class="lp-mini" style="margin-top:.8rem">
          <div class="k" style="margin-bottom:.45rem">Next best actions</div>
          {''.join(f'<div style="display:flex;justify-content:space-between;font-size:.84rem;color:#e2e8f0;padding:.28rem 0;border-top:1px solid rgba(255,255,255,.06)"><span>{c["name"]}</span><span style="color:#6ee7b7;font-weight:700">+{B.money(c["profit"])}</span></div>' for c in camps)}
        </div>
      </div>
      <div class="lp-card lp-float f1">
        <div class="ic">{icon("cpu", 18, "white")}</div>
        <div><div class="t">New visitor scored: hesitating, high potential</div><div class="s">Cart reminder suggested after 3 clicks</div></div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="lp-stats">
  <div class="lp-stat"><div class="n">{events_m:.1f}M</div><div class="l">e-commerce events behind the AI</div></div>
  <div class="lp-stat"><div class="n">×{lift:.1f}</div><div class="l">more buyers in the top 10% of AI scores</div></div>
  <div class="lp-stat"><div class="n">3 clicks</div><div class="l">is all the AI needs to read a visitor</div></div>
  <div class="lp-stat"><div class="n">&lt; 1 min</div><div class="l">from your file to your action plan</div></div>
</div>

<div class="lp-eyebrow">Everything in one place</div>
<div class="lp-h2">From raw clicks to decisions your team can act on today</div>
<div class="lp-lead">No dashboards to decode. SmartCart reads your data and tells you, in plain words, what is happening and what to do next.</div>
<div class="lp-feat">
  <div class="lp-f"><div class="ic">{icon("chart", 22)}</div><h4>Instant diagnosis</h4>
    <p>Revenue, conversion and basket trends, unusual days and a 4-week sales forecast, explained in plain sentences.</p></div>
  <div class="lp-f"><span class="tag">AI</span><div class="ic">{icon("cpu", 22)}</div><h4>Purchase-intent scoring</h4>
    <p>After only three clicks, every visit is scored by our model: ready to buy, hesitating or just browsing.</p></div>
  <div class="lp-f"><div class="ic">{icon("users", 22)}</div><h4>Customer groups</h4>
    <p>Best customers, loyal, new, slipping away, sleeping: eight clear groups, each with the right move.</p></div>
  <div class="lp-f"><div class="ic">{icon("trend_up", 22)}</div><h4>Product momentum</h4>
    <p>Rising and declining brands, products viewed but not bought, and bundles your customers already create.</p></div>
  <div class="lp-f"><div class="ic">{icon("cart", 22)}</div><h4>Cart recovery</h4>
    <p>See the revenue left in abandoned carts and who to remind, with the value at stake for each shopper.</p></div>
  <div class="lp-f"><div class="ic">{icon("target", 22)}</div><h4>Action plan with export</h4>
    <p>Campaigns ranked by expected profit, with who, what and when, plus the customer list ready for your e-mail tool.</p></div>
</div>

<div class="lp-eyebrow">How it works</div>
<div class="lp-h2">Three steps. No setup.</div>
<div class="lp-lead">Export your activity from your e-commerce platform and let SmartCart do the rest.</div>
<div class="lp-steps">
  <div class="lp-s"><div class="num">{icon("upload", 22, "white")}</div><h4>1. Upload</h4>
    <p>CSV, Excel or Parquet. Columns and event names are recognised automatically, in any language.</p></div>
  <div class="lp-s"><div class="num">{icon("sparkles", 22, "white")}</div><h4>2. Analyse</h4>
    <p>We clean your data, rebuild every visit and score it with the SmartCart AI.</p></div>
  <div class="lp-s"><div class="num">{icon("target", 22, "white")}</div><h4>3. Act</h4>
    <p>Read your diagnosis, pick the actions, download the audiences and launch.</p></div>
</div>

<div class="lp-ai">
  <div class="lp-orb o1" style="opacity:.45"></div><div class="lp-orb o2" style="opacity:.4"></div>
  <div class="grid2">
    <div>
      <div class="eb">The model behind SmartCart</div>
      <h2>An AI that reads intent from the first three clicks</h2>
      <p>Our gradient-boosting model was trained and tuned on {events_m:.1f} million events and {MC['dataset']['sessions'] / 1e6:.2f}
      million visits of a cosmetics e-shop, then validated on a month it had never seen. Each visit is explained, not
      just scored, and the recommended actions are grounded in a randomised marketing experiment.</p>
      <p>When a new store connects its data, the pre-trained model is applied directly. In our tests it ranked a new
      store's visitors better than a model trained on that store's history alone.</p>
    </div>
    <div class="lp-metrics">
      <div class="lp-m"><div class="n">{MC['model']['test']['pr_auc']:.2f} <small>vs {MC['model']['test']['base_rate']:.2f}</small></div>
        <div class="l">Precision-recall score vs random targeting</div>
        <div class="lp-bar"><i style="width:{MC['model']['test']['pr_auc'] / 0.3 * 100:.0f}%"></i></div></div>
      <div class="lp-m"><div class="n">{dec.cum_share_of_buyers.iloc[2]:.0%}</div>
        <div class="l">of buyers found in the top 30% of scores</div>
        <div class="lp-bar"><i style="width:{dec.cum_share_of_buyers.iloc[2] * 100:.0f}%"></i></div></div>
      <div class="lp-m"><div class="n">5 weeks</div><div class="l">of stable out-of-time performance</div>
        <div class="lp-bar"><i style="width:100%"></i></div></div>
      <div class="lp-m"><div class="n">17</div><div class="l">behavioural signals, explained with SHAP</div>
        <div class="lp-bar"><i style="width:68%"></i></div></div>
    </div>
  </div>
</div>

<div class="lp-final">
  <div class="lp-eyebrow">Ready?</div>
  <div class="lp-h2">See what your data has been trying to tell you</div>
  <div class="lp-lead" style="margin-bottom:1.8rem">Upload a file and get your diagnosis and action plan in under a minute.</div>
  <a class="lp-btn pri" href="?go=import" target="_self" style="padding:0 2.4rem">Get started {icon("arrow", 18, "white", 2.2)}</a>
</div>
""", unsafe_allow_html=True)

footer()
