"""Business-intelligence layer: loaders, RFM segmentation, forecasting, anomalies, trend
detection and the campaign generator used by the Action Center and the Command Center."""
import numpy as np
import pandas as pd
import streamlit as st

from core import data as D


def month_label(m):
    return pd.Timestamp(str(m) + "-01").strftime("%b %Y")


class _MonthLabels(dict):
    def __missing__(self, k):
        return month_label(k)

    def get(self, k, default=None):
        return month_label(k)


MONTH_LABEL = _MonthLabels()

# Hillstrom (randomised experiment): conversion uplift of a targeted e-mail, 2-week window
HILLSTROM_CONV_UPLIFT = 0.0068


def meta():
    return D.ws()["meta"]


def load(name):
    return D.ws()["tables"][name]


def last_prev():
    m = meta()
    return m["last_month"], m["prev_month"]


def cur():
    """Currency symbol of the current workspace (chosen at import; demo data is in $)."""
    return meta().get("currency", "$")


def cfmt(text):
    """Swap the $ placeholder in a label / format string for the workspace currency."""
    return text.replace("$", cur())


def money(x, d=0):
    s = cur()
    if abs(x) >= 1e6:
        return f"{s}{x / 1e6:.2f}M"
    if abs(x) >= 1e4:
        return f"{s}{x / 1e3:.1f}k"
    return f"{s}{x:,.{d}f}"


def delta_html(cur, prev, pct_points=False, inverse=False):
    if prev is None or prev == 0 or pd.isna(prev):
        return ""
    if pct_points:
        d = (cur - prev) * 100
        txt = f"{d:+.1f} pts"
    else:
        d = (cur / prev - 1) * 100
        txt = f"{d:+.1f}%"
    good = (d >= 0) != inverse
    cls = "delta-up" if good else "delta-down"
    arrow = "▲" if d >= 0 else "▼"
    return f'<span class="{cls}">{arrow} {txt}</span> vs prev. month'


# ------------------------------------------------------------------ KPIs ----------
def monthly_kpis():
    return D.memo(('monthly_kpis',) + tuple(locals().values()), lambda: _monthly_kpis())


def _monthly_kpis():
    m = load("monthly").copy()
    ab = load("abandon_monthly").set_index("month")
    nvr = load("new_vs_returning")
    m["aov"] = m.revenue / m.orders
    m["conversion"] = m.buyers / m.visitors
    m["cart_rate"] = m.carters / m.visitors
    m["abandon_rate"] = m.month.map(ab.abandoned_items / ab.carted_items)
    m["abandoned_value"] = m.month.map(ab.abandoned_value)
    ret = nvr[nvr.type == "Returning customers"].set_index("month")
    m["returning_rev_share"] = m.month.map(ret.revenue) / m.revenue
    m["returning_rev_share"] = m.returning_rev_share.fillna(0)
    m["items_per_order"] = m.units / m.orders
    return m


# ------------------------------------------------------------------ RFM -----------
SEGMENT_ORDER = ["Champions", "Loyal", "Promising", "New customers", "Need attention", "At risk",
                 "Hibernating", "Lost"]
SEGMENT_INFO = {
    "Champions": ("#0d366b", "Bought 3+ times, last order < 30 days. Your best customers.",
                  "Reward, early access to launches, referral program. No discounts."),
    "Loyal": ("#1c5cab", "Bought 3+ times, last order 30–75 days ago.",
              "Loyalty points, personalised replenishment reminders, upsell premium lines."),
    "Promising": ("#3987e5", "2 orders, last one < 30 days. On the way to loyalty.",
                  "Push the 3rd order: cross-sell bundles of frequently-bought-together products."),
    "New customers": ("#1baf7a", "First order < 30 days ago.",
                      "Onboarding flow, how-to content, second-purchase incentive at day 14–21."),
    "Need attention": ("#eda100", "2 orders, last one 30–75 days ago.",
                       "Replenishment reminder on their favourite brand, limited-time offer."),
    "At risk": ("#eb6834", "2+ orders but nothing for 75+ days. Valuable and slipping away.",
                "Win-back campaign: personalised e-mail on favourite brand + incentive."),
    "Hibernating": ("#e87ba4", "Single order 30–90 days ago.",
                    "Low-cost reactivation: new arrivals of their favourite brand."),
    "Lost": ("#94A3B8", "Single order more than 90 days ago.",
             "Exclude from paid campaigns; include only in cheap seasonal newsletters."),
}


def customers():
    return D.memo(('customers',) + tuple(locals().values()), lambda: _customers())


def _customers():
    c = load("customers").copy()
    ref = pd.Timestamp(meta()["data_end"]) + pd.Timedelta(seconds=1)
    c["recency"] = (ref - c.last_purchase).dt.days
    c["tenure"] = (ref - c.first_purchase).dt.days
    c["aov"] = c.revenue / c.orders
    r, f = c.recency, c.orders
    c["segment"] = np.select(
        [(f >= 3) & (r <= 30), (f >= 3) & (r <= 75), (f >= 2) & (r > 75),
         (f == 2) & (r <= 30), (f == 2) & (r <= 75), (f == 1) & (r <= 30), (f == 1) & (r <= 90)],
        ["Champions", "Loyal", "At risk", "Promising", "Need attention", "New customers", "Hibernating"],
        "Lost")
    # R, F, M scores 1-5 (for display)
    q5 = lambda x: pd.qcut(x.rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int) if len(x) >= 5 else 3  # noqa: E731
    c["R"] = q5(-c.recency)
    c["F"] = np.select([f >= 5, f >= 3, f == 2], [5, 4, 3], 1)
    c["M"] = q5(c.revenue)
    c["vip"] = c.revenue >= c.revenue.quantile(0.95)
    return c


def segment_summary():
    return D.memo(('segment_summary',) + tuple(locals().values()), lambda: _segment_summary())


def _segment_summary():
    c = customers()
    s = c.groupby("segment").agg(customers=("user_id", "size"), revenue=("revenue", "sum"),
                                 avg_revenue=("revenue", "mean"), avg_orders=("orders", "mean"),
                                 avg_recency=("recency", "mean"), aov=("aov", "mean")).reindex(SEGMENT_ORDER)
    s["share_customers"] = s.customers / s.customers.sum()
    s["share_revenue"] = s.revenue / s.revenue.sum()
    return s.reset_index()


# -------------------------------------------------------------- forecasting -------
def forecast_revenue(horizon=28, train_weeks=10):
    return D.memo(('forecast_revenue',) + tuple(locals().values()), lambda: _forecast_revenue(horizon, train_weeks))


def _forecast_revenue(horizon=28, train_weeks=10):
    """Log-linear trend + day-of-week seasonality, fitted on the last `train_weeks` weeks
    (holiday dip 24 Dec – 2 Jan excluded). Returns history and forecast with an 80% band."""
    d = load("daily").copy()
    d["day"] = pd.to_datetime(d.day)
    d = d.sort_values("day")
    last = d.day.max()
    train = d[d.day > last - pd.Timedelta(weeks=train_weeks)].copy()
    hol = ((train.day.dt.month == 12) & (train.day.dt.day >= 24)) | ((train.day.dt.month == 1) & (train.day.dt.day <= 2))
    train = train[~hol & (train.revenue > 0)]
    if len(train) < 14:
        return d, None, None, None
    t0 = train.day.min()

    def design(days):
        t = (days - t0).dt.days.values.astype(float)
        dow = pd.get_dummies(days.dt.dayofweek).reindex(columns=range(7), fill_value=0).values[:, 1:]
        return np.c_[np.ones(len(days)), t, dow]

    X = design(train.day)
    y = np.log(train.revenue.values)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    sigma = resid.std(ddof=X.shape[1])
    fut = pd.Series(pd.date_range(last + pd.Timedelta(days=1), periods=horizon, freq="D"))
    yhat = design(fut) @ beta
    fc = pd.DataFrame({"day": fut, "forecast": np.exp(yhat), "low": np.exp(yhat - 1.2816 * sigma),
                       "high": np.exp(yhat + 1.2816 * sigma)})
    fitted = pd.DataFrame({"day": train.day, "fitted": np.exp(X @ beta)})
    weekly_trend = np.exp(beta[1] * 7) - 1
    return d, fc, fitted, weekly_trend


def anomalies(metric="revenue", window=28, z=3.0):
    return D.memo(('anomalies',) + tuple(locals().values()), lambda: _anomalies(metric, window, z))


def _anomalies(metric="revenue", window=28, z=3.0):
    d = load("daily").copy()
    d["day"] = pd.to_datetime(d.day)
    d = d.sort_values("day").reset_index(drop=True)
    d["dow"] = d.day.dt.dayofweek
    x = d[metric].astype(float)
    med = x.rolling(window, center=True, min_periods=7).median()
    mad = (x - med).abs().rolling(window, center=True, min_periods=7).median()
    d["baseline"] = med
    d["robust_z"] = 0.6745 * (x - med) / mad.replace(0, np.nan)
    d["anomaly"] = d.robust_z.abs() > z
    return d


# ---------------------------------------------------------- trend detection -------
def default_min_brand_rev():
    lm, pm = last_prev()
    b = load("brand_month")
    return float(max(50.0, round(b[b.month.isin([lm, pm])].revenue.sum() * 0.002, -1)))


def brand_growth(min_rev=None):
    return D.memo(('brand_growth',) + tuple(locals().values()), lambda: _brand_growth(min_rev))


def _brand_growth(min_rev=None):
    """Brand revenue in the last month vs the previous month."""
    lm, pm = last_prev()
    if min_rev is None:
        min_rev = default_min_brand_rev()
    b = load("brand_month")
    b = b[b.brand != "(unbranded)"]
    p = b.pivot_table(index="brand", columns="month", values="revenue", aggfunc="sum").fillna(0)
    v = b.pivot_table(index="brand", columns="month", values="views", aggfunc="sum").fillna(0)
    get = lambda t, m: t[m].values if m in t.columns else np.zeros(len(t))  # noqa: E731
    out = pd.DataFrame({"brand": p.index, "rev_prev": get(p, pm), "rev_last": get(p, lm),
                        "rev_total": p.sum(1).values, "views_prev": get(v, pm), "views_last": get(v, lm)})
    out = out[(out.rev_prev + out.rev_last) >= min_rev]
    out["growth"] = out.rev_last / out.rev_prev.replace(0, np.nan) - 1
    out["views_growth"] = out.views_last / out.views_prev.replace(0, np.nan) - 1
    out["abs_change"] = out.rev_last - out.rev_prev
    return out.dropna(subset=["growth"])


def product_table():
    return D.memo(('product_table',) + tuple(locals().values()), lambda: _product_table())


def _product_table():
    p = load("products").copy()
    p["cart_to_purchase"] = p.units / p.carts.replace(0, np.nan)
    p["view_to_cart"] = p.carts / p.views.replace(0, np.nan)
    p["growth"] = p.revenue_last / p.revenue_prev.replace(0, np.nan) - 1
    p["label"] = _labels(p)
    return p


def _labels(p):
    base = p.brand.str.title() + " · #" + p.product_id.astype(str)
    if "product_name" in p and p.product_name.notna().any():
        return p.product_name.where(p.product_name.notna(), base).astype(str)
    return base


def product_label_map():
    return D.memo(("plm",), lambda: dict(zip(load("products").product_id, _labels(load("products")))))


# ---------------------------------------------------------- campaign engine -------
DEFAULT_ASSUMPTIONS = {"recovery_rate": 0.03, "margin": 0.40, "email_cost": 0.05, "uplift": HILLSTROM_CONV_UPLIFT,
                       "cross_sell_rate": 0.03}


def campaigns(a=None):
    """Data-driven campaign recommendations for the current workspace, each with audience,
    rationale and estimated monthly impact."""
    a = {**DEFAULT_ASSUMPTIONS, **(a or {})}
    c = customers()
    n_months = max(len(meta()["months"]), 1)
    aov_all = float(c.aov.median()) if len(c) else 0.0
    med_aov = lambda df: float(df.aov.median()) if len(df) else aov_all  # noqa: E731
    out = []

    # 1. Cart recovery
    ab = load("abandoners")
    if len(ab):
        val = float(ab.cart_value.sum())
        out.append(dict(
            id="cart_recovery", name="Abandoned-cart recovery", module="Cart Recovery",
            channel="E-mail + on-site reminder", audience=len(ab), audience_df=ab.assign(segment="Abandoned cart (30 days)"),
            why=f"{len(ab):,} shoppers left {money(val)} of products in their cart over the last 30 days without buying.",
            action="Trigger a reminder 1 h, 24 h and 72 h after abandonment showing the exact products; no discount on "
                   "the first reminder.", kpi="Recovered revenue, cart-to-order rate",
            revenue=val * a["recovery_rate"], cost=len(ab) * 3 * a["email_cost"],
            assumption=f"{a['recovery_rate']:.0%} of abandoned value recovered. Cart value is inflated by wishlist-style "
                       "carts, so keep this conservative."))

    # 2. Win-back at risk
    risk = c[c.segment == "At risk"]
    if len(risk):
        out.append(dict(
            id="winback", name="Win back at-risk customers", module="Customers 360", channel="Personalised e-mail",
            audience=len(risk), audience_df=risk,
            why=f"{len(risk):,} repeat customers ({money(risk.revenue.sum())} lifetime revenue) have not ordered for "
                "75+ days.",
            action="E-mail featuring their favourite brand's new arrivals + a time-limited incentive.",
            kpi="Reactivation rate within 14 days", revenue=len(risk) * a["uplift"] * 4 * med_aov(risk),
            cost=len(risk) * a["email_cost"],
            assumption=f"Conversion uplift ×4 the Hillstrom average ({a['uplift'] * 4:.1%}) — repeat buyers react more."))

    # 3. Second purchase for new customers
    new = c[c.segment == "New customers"]
    if len(new):
        coh = load("cohorts")
        lm, _ = last_prev()
        base = coh[(coh.month_index == 0) & (coh.cohort < lm)].customers.sum()
        m1 = coh[coh.month_index == 1].customers.sum() / base if base else np.nan
        txt = f"Only {m1:.0%} of new customers order again the following month. " if m1 == m1 else ""
        out.append(dict(
            id="second_order", name="Turn new customers into repeat buyers", module="Customers 360",
            channel="Onboarding e-mail flow", audience=len(new), audience_df=new,
            why=txt + f"{len(new):,} customers bought for the first time in the last 30 days.",
            action="Day-14 e-mail with products frequently bought with their first order (cross-sell pairs).",
            kpi="2nd-order rate at 30 days", revenue=len(new) * a["cross_sell_rate"] * med_aov(new),
            cost=len(new) * a["email_cost"], assumption=f"{a['cross_sell_rate']:.0%} extra second orders."))

    # 4. Protect champions (no discount)
    ch = c[c.segment.isin(["Champions", "Loyal"])]
    if len(ch):
        share = ch.revenue.sum() / max(c.revenue.sum(), 1e-9)
        out.append(dict(
            id="vip", name="VIP loyalty program", module="Customers 360", channel="Loyalty / early access",
            audience=len(ch), audience_df=ch,
            why=f"Champions + Loyal are {len(ch) / len(c):.0%} of customers but {share:.0%} of revenue.",
            action="Early access to launches, free samples, referral rewards. Avoid discounts: they buy anyway.",
            kpi="Retention of top customers, referral sign-ups", revenue=0.02 * ch.revenue.sum() / n_months,
            cost=len(ch) * 0.2,
            assumption=f"+2% monthly revenue from this group; {money(0.20, 2)} per member per month (samples, perks)."))

    # 5. Cross-sell bundles
    pairs = load("pairs")
    if len(pairs):
        top = pairs.sort_values("together", ascending=False).head(10)
        lbl = product_label_map()
        p0 = top.iloc[0]
        out.append(dict(
            id="bundles", name="Cross-sell bundles", module="Products & Brands", channel="On-site recommendations",
            audience=int(top.orders_a.sum() / n_months),
            audience_df=top.assign(product_a_label=top.product_a.map(lbl), product_b_label=top.product_b.map(lbl)),
            why=f"Strong product affinities: e.g. {lbl.get(p0.product_a, p0.product_a)} and "
                f"{lbl.get(p0.product_b, p0.product_b)} are bought together {int(p0.together):,} times "
                f"(lift ×{p0.lift:.0f}).",
            action="'Frequently bought together' block on product pages + bundle pricing for the top 10 pairs.",
            kpi="Items per order, attach rate", revenue=float(top.orders_a.sum() / n_months * 0.05 * aov_all),
            cost=0, assumption="5% of buyers of product A add product B."))

    # 6. Rising brands
    g = brand_growth()
    rising = g[(g.growth > 0.15)].sort_values("abs_change", ascending=False).head(5)
    if len(rising):
        out.append(dict(
            id="rising", name="Ride the rising brands", module="Trends", channel="Homepage merchandising + ads",
            audience=int(rising.views_last.sum()), audience_df=rising,
            why="Brands growing fastest month-on-month: " + ", ".join(
                f"{b.title()} ({gr:+.0%})" for b, gr in zip(rising.brand, rising.growth)) + ".",
            action="Give them homepage visibility and secure stock before demand peaks.",
            kpi="Brand revenue growth", revenue=float(rising.abs_change.sum() * 0.25), cost=0,
            assumption="Merchandising adds 25% to the observed monthly gain."))

    # 7. On-site intent (AI)
    if D.intent_available():
        ab_k = D.ws()["kpis"]["ab_test"]
        per_day = ab_k["sessions_per_day"]
        out.append(dict(
            id="intent", name="Real-time intent nudges", module="Purchase Intent AI", channel="On-site (live session)",
            audience=int(per_day * 30), audience_df=None,
            why=f"≈ {per_day:,.0f} sessions/day land in the mid-intent segments after 3 clicks: undecided, often "
                "with a cart.",
            action="Cart reminder banner for mid-intent sessions with a cart; viewed-product recommendations otherwise.",
            kpi="Incremental conversion (A/B test)",
            revenue=per_day * 30 * ab_k["session_purchase_rate"] * 0.05 * aov_all, cost=0,
            assumption="+5% relative conversion on these sessions (to validate by A/B test)."))

    for o in out:
        o["profit"] = o["revenue"] * a["margin"] - o["cost"]
        o["roi"] = o["profit"] / o["cost"] if o["cost"] else np.inf
    out.sort(key=lambda o: -o["profit"])
    for i, o in enumerate(out):
        o["priority"] = "High" if i < 3 else ("Medium" if i < 5 else "Low")
    return out
