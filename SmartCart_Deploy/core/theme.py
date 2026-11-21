"""Visual identity: CSS, Plotly template and reusable UI components."""
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# ---------------------------------------------------------------- palette ----------
INK = "#0F172A"
INK_2 = "#475569"
INK_3 = "#94A3B8"
SURFACE = "#FFFFFF"
CANVAS = "#F6F7FB"
GRID = "#EEF0F5"
BORDER = "#E3E7EF"

# Categorical (validated default order): blue, orange, aqua
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
ACTION_COLORS = {"cart": BLUE, "remove_from_cart": ORANGE, "view": AQUA}
ACTION_LABELS = {"cart": "Add to cart", "remove_from_cart": "Remove from cart", "view": "View"}

# Ordinal blue ramp for the 4 score segments (dark = high score)
SEGMENT_COLORS = {"1. Top 10%": "#0d366b", "2. Next 20%": "#1c5cab",
                  "3. Middle 30%": "#3987e5", "4. Bottom 40%": "#86b6ef"}
SEQ_SCALE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
NEUTRAL = "#CBD5E1"
GOOD, CRITICAL = "#0ca30c", "#d03b3b"

FONT = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"

# ----------------------------------------------------------- plotly template -------
pio.templates["smartcart"] = go.layout.Template(
    layout=dict(
        font=dict(family=FONT, color=INK_2, size=13),
        title=dict(font=dict(size=15, color=INK), x=0, xanchor="left", y=0.97),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=[BLUE, ORANGE, AQUA],
        margin=dict(l=10, r=10, t=56, b=10),
        xaxis=dict(gridcolor=GRID, zeroline=False, linecolor=BORDER, ticks="", title_font=dict(size=12)),
        yaxis=dict(gridcolor=GRID, zeroline=False, linecolor=BORDER, ticks="", title_font=dict(size=12)),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
        hoverlabel=dict(bgcolor="white", bordercolor=BORDER, font=dict(family=FONT, color=INK, size=12)),
        bargap=0.3,
    )
)
pio.templates.default = "smartcart"

CHART_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}


def chart(fig, height=380):
    fig.update_layout(height=height)
    st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)


# --------------------------------------------------------------------- CSS ---------
CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, .stApp, .stMarkdown, button, input, select, textarea {{ font-family: {FONT}; }}
.stApp {{ background: {CANVAS}; }}
.block-container {{ padding-top: 4.6rem; padding-bottom: 4rem; max-width: 1280px; }}
header[data-testid="stHeader"] {{ background: rgba(246,247,251,.85); backdrop-filter: blur(8px); height: 3rem; }}
h1, h2, h3, h4 {{ color: {INK}; letter-spacing: -0.02em; }}
.stMarkdown p, .stMarkdown li {{ color: {INK_2}; }}

/* Sidebar */
[data-testid="stSidebar"] {{ background: #0B1530; }}
[data-testid="stSidebar"] .stMarkdown p, [data-testid="stSidebar"] .stMarkdown span {{ color: #CBD5E1; }}
[data-testid="stSidebarNav"] a span {{ color: #CBD5E1 !important; font-weight: 500; }}
[data-testid="stSidebarNav"] a[aria-current="page"] {{ background: rgba(57,135,229,.22); border-radius: 8px; }}
[data-testid="stSidebarNav"] a[aria-current="page"] span {{ color: #FFFFFF !important; font-weight: 600; }}
[data-testid="stSidebarNav"] a:hover {{ background: rgba(255,255,255,.06); border-radius: 8px; }}
[data-testid="stNavSectionHeader"] {{ color: #64748B !important; text-transform: uppercase;
    letter-spacing: .08em; font-size: .7rem !important; font-weight: 700; }}
[data-testid="stSidebar"] [data-testid="stIconMaterial"] {{ color: #93C5FD; }}
.brand {{ display:flex; align-items:center; gap:.6rem; padding:.2rem 0 1rem; }}
.brand .logo {{ width:36px; height:36px; border-radius:10px; background:linear-gradient(135deg,#3987e5,#1c5cab);
    display:flex; align-items:center; justify-content:center; color:white; font-weight:800; font-size:1rem; }}
.brand .name {{ color:white; font-weight:800; font-size:1.1rem; line-height:1; }}
.brand .tag {{ color:#64748B; font-size:.72rem; margin-top:.2rem; }}
.workspace {{ background:rgba(255,255,255,.05); border:1px solid rgba(255,255,255,.08); border-radius:12px;
    padding:.6rem .8rem; margin-bottom:.8rem; }}
.workspace .ws-label {{ color:#64748B; font-size:.65rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase; }}
.workspace .ws-name {{ color:#FFFFFF; font-weight:700; font-size:.92rem; margin-top:.15rem; }}
.workspace .ws-name::first-letter {{ color:#1baf7a; }}
.workspace .ws-sub {{ color:#94A3B8; font-size:.72rem; margin-top:.1rem; }}

/* Hero */
.hero {{
    background: radial-gradient(900px 380px at 88% -30%, rgba(57,135,229,.55), transparent 60%),
                linear-gradient(135deg, #0B1530 0%, #12245A 60%, #1C3A8A 100%);
    border-radius: 20px; padding: 2.4rem 2.6rem; margin-bottom: 1.6rem;
    box-shadow: 0 20px 40px -24px rgba(11,21,48,.6);
}}
.hero .eyebrow {{ font-size: .75rem; letter-spacing: .14em; text-transform: uppercase; color: #93C5FD; font-weight: 700; }}
.hero h1 {{ color: white !important; font-size: 2.5rem; font-weight: 800; margin: .4rem 0 .6rem; line-height: 1.1; padding:0; }}
.hero p {{ color: #CBD5E1 !important; font-size: 1.06rem; max-width: 780px; margin: 0; }}
.hero .chips {{ margin-top: 1.3rem; display: flex; gap: .5rem; flex-wrap: wrap; }}
.hero .chip {{ background: rgba(255,255,255,.1); border: 1px solid rgba(255,255,255,.16);
    padding: .3rem .8rem; border-radius: 999px; font-size: .8rem; color: #E2E8F0; }}

/* Page header */
.page-head {{ margin-bottom: 1.3rem; }}
.page-head .eyebrow {{ font-size: .72rem; letter-spacing: .14em; text-transform: uppercase; color: {BLUE}; font-weight: 700; }}
.page-head h1 {{ font-size: 2.05rem; font-weight: 800; margin: .25rem 0 .35rem; padding: 0; }}
.page-head p {{ font-size: 1.02rem; margin: 0; max-width: 860px; color: {INK_2}; }}

/* KPI cards */
.kpi {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 16px; padding: 1.05rem 1.2rem;
    height: 100%; box-shadow: 0 1px 2px rgba(15,23,42,.04); }}
.kpi .label {{ font-size: .74rem; color: {INK_2}; font-weight: 700; text-transform: uppercase; letter-spacing: .06em; }}
.kpi .value {{ font-size: 1.8rem; font-weight: 800; color: {INK}; margin-top: .35rem; line-height: 1.1;
    font-variant-numeric: tabular-nums; }}
.kpi .sub {{ font-size: .82rem; color: #64748B; margin-top: .35rem; }}
.delta-up {{ color:#0a8a0a; font-weight:700; }} .delta-down {{ color:#c42f2f; font-weight:700; }}
.kpi.accent {{ background: linear-gradient(135deg,#0B1530,#1C3A8A); border: none; }}
.kpi.accent .label {{ color: #93C5FD; }} .kpi.accent .value {{ color: white; }} .kpi.accent .sub {{ color: #CBD5E1; }}

/* Cards / callouts */
.card {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 16px; padding: 1.15rem 1.3rem;
    box-shadow: 0 1px 2px rgba(15,23,42,.04); height: 100%; }}
.card h4 {{ margin: 0 0 .45rem; font-size: 1rem; padding: 0; }}
.card p {{ margin: 0; font-size: .92rem; color: {INK_2}; }}
.card .big {{ font-size: 1.6rem; font-weight: 800; color: {INK}; }}
.insight {{ background: #EFF6FF; border: 1px solid #DBEAFE; border-left: 4px solid {BLUE};
    border-radius: 12px; padding: .9rem 1.1rem; margin: .5rem 0 1rem; color: #1E3A8A; font-size: .94rem; }}
.insight b {{ color: #0F2A6B; }}
.warn {{ background: #FFF7ED; border: 1px solid #FED7AA; border-left: 4px solid {ORANGE};
    border-radius: 12px; padding: .9rem 1.1rem; margin: .5rem 0 1rem; color: #7C2D12; font-size: .94rem; }}
.section {{ font-size: 1.25rem; font-weight: 800; color: {INK}; margin: 2rem 0 .15rem; letter-spacing: -.01em; }}
.section-sub {{ color: {INK_2}; font-size: .93rem; margin-bottom: .7rem; }}

/* Journey pills */
.pill {{ display: inline-block; padding: .3rem .75rem; border-radius: 999px; font-size: .82rem;
    font-weight: 600; color: white; }}
.arrow {{ color: {INK_3}; margin: 0 .35rem; font-weight: 700; }}

/* Step list */
.step {{ display: flex; gap: .9rem; align-items: flex-start; margin-bottom: 1rem; }}
.step .num {{ flex: 0 0 30px; height: 30px; border-radius: 50%; background: {BLUE}; color: white;
    font-weight: 700; display: flex; align-items: center; justify-content: center; font-size: .85rem; }}
.step .txt {{ font-size: .93rem; color: {INK_2}; }}
.step .txt b {{ color: {INK}; }}

/* Badge */
.badge {{ display:inline-block; padding:.2rem .6rem; border-radius:999px; font-size:.75rem; font-weight:700; }}

/* Widgets */
.stTabs [data-baseweb="tab-list"] {{ gap: .25rem; border-bottom: 1px solid {BORDER}; }}
.stTabs [data-baseweb="tab"] {{ padding: .55rem 1rem; font-weight: 600; }}
[data-testid="stPlotlyChart"] {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 16px;
    padding: .7rem .9rem .3rem; box-shadow: 0 1px 2px rgba(15,23,42,.04); }}
[data-testid="stExpander"] {{ background: {SURFACE}; border-radius: 12px; }}
/* Guided journey */
.stepper {{ display:flex; align-items:center; gap:.4rem; margin:0 0 1.4rem; flex-wrap:wrap; }}
.stepper .st {{ display:flex; align-items:center; gap:.45rem; }}
.stepper .dot {{ width:28px; height:28px; border-radius:50%; display:flex; align-items:center; justify-content:center;
    font-weight:700; font-size:.82rem; border:2px solid #CBD5E1; color:#64748B; background:white; }}
.stepper .lbl {{ font-size:.85rem; font-weight:600; color:#64748B; }}
.stepper .done .dot {{ background:#1baf7a; border-color:#1baf7a; color:white; }}
.stepper .now .dot {{ background:{BLUE}; border-color:{BLUE}; color:white; box-shadow:0 0 0 4px rgba(42,120,214,.15); }}
.stepper .now .lbl {{ color:{INK}; }}
.stepper .bar {{ flex:1 1 24px; min-width:16px; height:2px; background:#E3E7EF; }}
.takeaways {{ background:{SURFACE}; border:1px solid {BORDER}; border-left:5px solid {BLUE}; border-radius:14px;
    padding:1rem 1.3rem .6rem; margin:.2rem 0 1.4rem; box-shadow:0 1px 2px rgba(15,23,42,.04); }}
.takeaways .tk-title {{ font-size:.75rem; font-weight:800; letter-spacing:.1em; text-transform:uppercase; color:{BLUE}; }}
.takeaways ul {{ margin:.5rem 0 .4rem 1.1rem; padding:0; }}
.takeaways li {{ color:{INK}; font-size:1rem; margin-bottom:.45rem; }}
.finding {{ display:flex; gap:.9rem; background:{SURFACE}; border:1px solid {BORDER}; border-radius:14px;
    padding:1rem 1.1rem; margin-bottom:.8rem; box-shadow:0 1px 2px rgba(15,23,42,.04); }}
.finding .f-icon {{ flex:0 0 42px; height:42px; border-radius:12px; display:flex; align-items:center;
    justify-content:center; font-size:1.3rem; }}
.finding .f-title {{ font-weight:750; color:{INK}; font-size:1rem; }}
.finding .f-text {{ color:{INK_2}; font-size:.9rem; margin-top:.2rem; }}
.group {{ background:{SURFACE}; border:1px solid {BORDER}; border-radius:14px; padding:1rem 1.1rem; height:100%; }}
.group .g-head {{ display:flex; align-items:center; gap:.5rem; font-weight:800; color:{INK}; font-size:1.02rem; }}
.group .g-num {{ font-size:1.6rem; font-weight:800; color:{INK}; margin:.35rem 0 0; }}
.group .g-sub {{ font-size:.8rem; color:#64748B; }}
.group .g-mean {{ font-size:.86rem; color:{INK_2}; margin-top:.5rem; }}
.group .g-do {{ font-size:.86rem; color:#1E3A8A; font-weight:600; margin-top:.4rem; }}
.action {{ background:{SURFACE}; border:1px solid {BORDER}; border-radius:16px; padding:1.1rem 1.3rem; }}
.action .a-rank {{ width:34px; height:34px; border-radius:10px; background:{BLUE}; color:white; font-weight:800;
    display:flex; align-items:center; justify-content:center; }}
.action .a-title {{ font-size:1.15rem; font-weight:800; color:{INK}; }}
.action .a-row {{ display:grid; grid-template-columns:110px 1fr; gap:.4rem .8rem; margin-top:.7rem; font-size:.92rem; }}
.action .a-k {{ color:#64748B; font-weight:700; font-size:.78rem; text-transform:uppercase; letter-spacing:.05em; padding-top:.15rem; }}
.action .a-v {{ color:{INK}; }}
.big-option {{ background:{SURFACE}; border:2px solid {BORDER}; border-radius:18px; padding:1.4rem 1.5rem; height:100%; }}
.big-option .bo-ic {{ width:46px; height:46px; border-radius:13px; display:flex; align-items:center; justify-content:center;
    color:#2563eb; background:linear-gradient(135deg,#eff6ff,#eef2ff); border:1px solid #dbeafe; margin-bottom:.9rem; }}
.big-option h3 {{ margin:0 0 .3rem; font-size:1.2rem; }}
.big-option p {{ margin:0; color:{INK_2}; }}
.recognised {{ display:flex; flex-wrap:wrap; gap:.4rem; margin:.3rem 0 .6rem; }}
.recognised span {{ background:#ECFDF5; border:1px solid #A7F3D0; color:#065F46; border-radius:999px;
    padding:.25rem .7rem; font-size:.82rem; }}
.recognised span.miss {{ background:#F1F5F9; border-color:#E2E8F0; color:#64748B; }}
.demo-banner {{ background:#FFFBEB; border:1px solid #FDE68A; color:#78350F; border-radius:12px;
    padding:.6rem 1rem; font-size:.9rem; margin-bottom:.2rem; }}
.foot {{ color: {INK_3}; font-size: .8rem; margin-top: 3rem; border-top: 1px solid {BORDER}; padding-top: 1rem; }}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------- components --------
def page_header(eyebrow, title, subtitle):
    st.markdown(f'<div class="page-head"><div class="eyebrow">{eyebrow}</div><h1>{title}</h1>'
                f'<p>{subtitle}</p></div>', unsafe_allow_html=True)


def kpi(col, label, value, sub="", accent=False):
    col.markdown(f'<div class="kpi{" accent" if accent else ""}"><div class="label">{label}</div>'
                 f'<div class="value">{value}</div><div class="sub">{sub}</div></div>', unsafe_allow_html=True)


def section(title, sub=""):
    st.markdown(f'<div class="section">{title}</div>' + (f'<div class="section-sub">{sub}</div>' if sub else ""),
                unsafe_allow_html=True)


def insight(html, kind="insight"):
    st.markdown(f'<div class="{kind}">{html}</div>', unsafe_allow_html=True)


def card(col, title, body):
    col.markdown(f'<div class="card"><h4>{title}</h4><p>{body}</p></div>', unsafe_allow_html=True)


def journey_pills(actions):
    parts = [f'<span class="pill" style="background:{ACTION_COLORS[a]}">{ACTION_LABELS[a]}</span>' for a in actions]
    return '<span class="arrow">→</span>'.join(parts)


def short_seq(seq):
    return seq.replace("remove_from_cart", "remove")


def footer():
    st.markdown('<div class="foot">SmartCart AI · Marketing intelligence for e-commerce · Demo workspace: cosmetics '
                'e-shop (REES46 open data, Oct 2019 – Feb 2020) · Causal benchmarks: Hillstrom e-mail experiment<br>'
                'Built by Rania El Mazouzi, Anaïs Hamitouche, Papa Makhtar Ndaw, Meryem Sabik · Master 2 EDS – AMSE</div>',
                unsafe_allow_html=True)


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


def num(x):
    return f"{x:,.0f}"
