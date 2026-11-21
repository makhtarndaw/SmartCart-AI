import io

import pandas as pd
import streamlit as st

from core import biz as B
from core import data as D
from core import engine as E
from core import ui as U
from core.icons import icon
from core.theme import footer, kpi

U.header(0, "Welcome to SmartCart AI",
         "Upload your store's activity and get, in a minute, a clear diagnosis of your business and a ready-to-use "
         "marketing action plan — powered by our purchase-prediction AI.")

W = D.ws()
M = W["meta"]


@st.cache_data(show_spinner="Reading your file…", max_entries=2)
def read_file(name, content):
    buf = io.BytesIO(content)
    low = name.lower()
    if low.endswith(".zip"):
        import zipfile
        with zipfile.ZipFile(buf) as z:
            inner = [n for n in z.namelist() if not n.startswith("__MACOSX")
                     and n.lower().endswith((".csv", ".txt", ".xlsx", ".xls", ".parquet"))]
            if not inner:
                raise ValueError("The zip file contains no CSV, Excel or Parquet file.")
            return read_file(inner[0].rsplit("/", 1)[-1], z.read(inner[0]))
    if low.endswith(".parquet"):
        return pd.read_parquet(buf)
    if low.endswith((".xlsx", ".xls")):
        return pd.read_excel(buf)
    head = content[:4096].decode("utf-8", errors="ignore").splitlines()[0] if content else ""
    sep = max([",", ";", "\t", "|"], key=head.count)
    try:
        return pd.read_csv(buf, sep=sep, low_memory=False)
    except UnicodeDecodeError:
        buf.seek(0)
        return pd.read_csv(buf, sep=sep, low_memory=False, encoding="latin-1")


# ------------------------------------------------- already analysed: show summary --
if not D.is_demo() and not st.session_state.get("replace_data"):
    st.success(icon=":material/check_circle:", body=f"**{M['name']}** is connected — {M['n_events']:,} events from "
               f"{B.month_label(M['months'][0])} to {B.month_label(M['months'][-1])}.")
    c = st.columns(4)
    kpi(c[0], "Customers who bought", f"{M['n_customers']:,}")
    kpi(c[1], "Visitors", f"{M['n_visitors']:,}")
    kpi(c[2], "Products", f"{M['n_products']:,}")
    kpi(c[3], "Visits analysed by the AI", f"{M['n_scored_sessions']:,}", accent=True)
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    l, r = st.columns([1, 1])
    l.page_link("views/j_diagnosis.py", label="See my diagnosis →", icon=":material/arrow_forward:",
                use_container_width=True)
    if r.button("Import another file", use_container_width=True):
        st.session_state.replace_data = True
        st.rerun()
    with st.expander("Data quality details"):
        st.markdown(f"""
- **{M['n_rows_uploaded']:,}** rows received
- **{M['n_ignored_events']:,}** rows ignored (event type not used) · **{M['n_bad_time']:,}** rows with an unreadable date
- **{M['n_duplicates']:,}** exact duplicates removed
- **{M['n_events']:,}** events kept · brand {'yes' if M['has_brand'] else 'no'} · category {'yes' if M['has_category'] else 'no'} · product names {'yes' if M['has_names'] else 'no'}
""")
    footer()
    st.stop()

# ------------------------------------------------------------------ choose --------
st.markdown("### How would you like to start?")
l, r = st.columns(2)
with l:
    st.markdown(f'<div class="big-option"><div class="bo-ic">{icon("upload", 24)}</div><h3>Use my own data</h3><p>Upload an export of your e-shop activity: '
                f'one row per product view, add-to-cart or purchase. CSV, Excel or Parquet.</p></div>',
                unsafe_allow_html=True)
with r:
    st.markdown(f'<div class="big-option"><div class="bo-ic">{icon("chart", 24)}</div><h3>Explore with demo data</h3><p>See what SmartCart does on a real '
                f'cosmetics e-shop (80,000 visitors, 5 months) before using your own data.</p></div>',
                unsafe_allow_html=True)
    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
    if st.button("Continue with demo data →", use_container_width=True):
        D.set_workspace(None)
        st.session_state.pop("replace_data", None)
        st.switch_page("views/j_diagnosis.py")

with l:
    up = st.file_uploader("Your file", type=["csv", "txt", "xlsx", "xls", "parquet", "zip"], label_visibility="collapsed")
    st.download_button("No file at hand? Download a sample company export (Maison Lumière)", icon=":material/download:",
                       data=open("assets/maison_lumiere_export.zip", "rb").read(), file_name="maison_lumiere_export.zip",
                       mime="application/zip", use_container_width=True)
    st.caption("A fictional French beauty e-shop: unzip it, then drop the CSV above.")

with st.expander("What should my file contain?"):
    st.markdown("""
One row per customer action on your site, with at least these 5 pieces of information (column names can be anything):

| Information | Example |
|---|---|
| **When** it happened | 2024-03-01 14:02 |
| **What** happened | product view · add to cart · remove from cart · purchase |
| **Who** (customer or visitor ID) | 51234 |
| **Which product** | SKU-0042 |
| **Price** | 12.90 |

Optional but useful: brand, category, product name. Most e-commerce platforms (Shopify, WooCommerce, Magento,
Google Analytics exports…) can produce this file.
""")
    tpl = pd.read_parquet(D.DEMO).head(3000).rename(columns={"category_id": "category"})
    st.download_button("Download an example file", icon=":material/download:", data=tpl.to_csv(index=False).encode(), file_name="smartcart_example.csv", mime="text/csv")
st.caption("Your file is analysed in memory for your session only. It is not stored.")

if up is None:
    footer()
    st.stop()

raw = read_file(up.name, up.getvalue())

# ------------------------------------------------------------------ recognise -----
st.markdown("### We read your file")
guess = E.guess_mapping(raw.columns)
cols_opt = ["— not in my file —"] + list(raw.columns)
labels = {**E.REQUIRED, **E.OPTIONAL}
nice = {"event_time": "When", "event_type": "What happened", "user_id": "Customer", "product_id": "Product",
        "price": "Price", "brand": "Brand", "category": "Category", "product_name": "Product name"}

with st.expander("Change the column matching", expanded=any(guess[k] is None for k in E.REQUIRED)):
    c = st.columns(4)
    mapping = {}
    for i, k in enumerate(labels):
        idx = cols_opt.index(guess[k]) if guess.get(k) in cols_opt else 0
        v = c[i % 4].selectbox(nice[k] + (" *" if k in E.REQUIRED else ""), cols_opt, index=idx, key=f"map_{k}")
        mapping[k] = None if v == cols_opt[0] else v

chips = "".join(f'<span>✓ {nice[k]} → <b>{mapping[k]}</b></span>' if mapping[k] else
                f'<span class="miss">{nice[k]}: not found</span>' for k in labels)
st.markdown(f"**{len(raw):,} rows** found. Columns recognised:<div class='recognised'>{chips}</div>",
            unsafe_allow_html=True)
missing = [nice[k] for k in E.REQUIRED if not mapping[k]]
if missing:
    st.error("We need these columns to continue: **" + ", ".join(missing) + "**. Open “Change the column matching”.")
    st.stop()

vc = raw[mapping["event_type"]].astype(str).value_counts().head(40)
OPTIONS = ["view", "cart", "remove_from_cart", "purchase", "ignore"]
NICE = {"view": "Product view", "cart": "Add to cart", "remove_from_cart": "Remove from cart", "purchase": "Purchase",
        "ignore": "Not used"}
event_map = {v: E.guess_event(v) for v in vc.index}
with st.expander("Change how actions are understood", expanded="purchase" not in event_map.values()):
    c = st.columns(4)
    for i, (val, n) in enumerate(vc.items()):
        event_map[val] = c[i % 4].selectbox(f"“{val}” ({n:,})", OPTIONS, index=OPTIONS.index(event_map[val]),
                                            format_func=NICE.get, key=f"ev_{val}")
chips = "".join(f'<span>“{v}” → <b>{NICE[m]}</b></span>' if m != "ignore" else f'<span class="miss">“{v}” → not used</span>'
                for v, m in event_map.items())
st.markdown(f"Actions recognised:<div class='recognised'>{chips}</div>", unsafe_allow_html=True)
if "purchase" not in event_map.values():
    st.error("We could not find **purchases** in your file. Open “Change how actions are understood” and tell us "
             "which label means a purchase.")
    st.stop()

# ------------------------------------------------------------------ run -----------
st.markdown("### Last step")
l, m, r = st.columns([2, 0.8, 1.2])
name = l.text_input("Your company name", value=up.name.rsplit(".", 1)[0].replace("_", " ").title())
CURRENCIES = {"€": "€ Euro", "$": "$ US dollar", "£": "£ Pound", "CHF ": "CHF Swiss franc"}
currency = m.selectbox("Currency", list(CURRENCIES), format_func=CURRENCIES.get)
r.markdown("<div style='height:1.8rem'></div>", unsafe_allow_html=True)
if r.button("Analyse my data", icon=":material/arrow_forward:", type="primary", use_container_width=True):
    bar = st.progress(0.0, text="Starting…")
    try:
        std = E.standardize(raw, mapping, event_map)
        w = E.build_workspace(std, D.pretrained_booster(), name=name,
                              progress=lambda p, msg: bar.progress(min(p, 1.0), text=msg))
        w["is_demo"] = False
        w["meta"]["currency"] = currency
        w["id"] = E.fingerprint(std)
        D.set_workspace(w)
        st.session_state.pop("replace_data", None)
        st.switch_page("views/j_diagnosis.py")
    except Exception as exc:
        bar.empty()
        st.error(f"We could not analyse this file: {exc}")

footer()
