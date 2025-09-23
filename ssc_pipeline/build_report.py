# ssc_pipeline/build_report.py
import os, sys, io, base64, pathlib, datetime as dt, textwrap
import numpy as np
import pandas as pd

# Headless plotting for CI
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.environ.get("SSC_ROOT", os.getcwd())
OUT_DIR = os.path.join(ROOT, "ssc_pipeline")
CSV = os.path.join(ROOT, "main.csv")

# =========================
# Optional: GPT executive summary
# =========================
def gpt_exec_summary(context: dict) -> str:
    """
    Returns an executive English summary using OpenAI if OPENAI_API_KEY is set.
    Otherwise returns a concise fallback.
    """
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        return ("(AI summary disabled — add an OPENAI_API_KEY repository secret to enable "
                "the executive narrative.)")
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        sys_msg = "You are a senior supply-chain economist. Be concise, executive, factual."
        prompt = f"""
Write an executive summary (bullets + short paragraphs) using ONLY the dataset profile below.
Avoid jargon and do not fabricate numbers beyond the aggregates given.

DATA PROFILE:
- Sources & row counts: {context.get('by_source')}
- Years (from 'time'): {context.get('years')}
- WITS totals (USA↔World, annual): {context.get('wits_last_points')}
- Tariffs (MFN avg, USA↔World, annual): {context.get('tariff_last_points')}
- Census monthly totals (imports, exports): {context.get('census_last_points')}
- BLS PPI last points: {context.get('ppi_last_points')}
- Forecast horizon: 12 months for Census and PPI (ETS)

TASKS:
1) Current situation: where volumes and tariffs stand; what PPI implies for costs.
2) What to expect (next 12 months): directionality + risks (FX, tariffs, lead times).
3) 3–5 risks/opportunities for service supply chains.
4) Close with a brief action list (3 items).

Tone: crisp, non-technical, public-facing.
"""
        messages = [{"role":"system","content":sys_msg},{"role":"user","content":prompt}]
        out = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.35,
            max_tokens=900
        )
        return out.choices[0].message.content.strip()
    except Exception as e:
        return f"(AI summary error: {e})"

# =========================
# Helpers
# =========================
def now_utc():
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def img64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")

def choose_numcol(df, candidates=("Value","value")):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def table_html(df: pd.DataFrame, max_rows=12, cols=None):
    if df is None or df.empty:
        return "<p><em>(no data)</em></p>"
    out = df.copy()
    if cols:
        keep = [c for c in cols if c in out.columns]
        if keep:
            out = out[keep]
    return out.fillna("").head(max_rows).to_html(index=False, border=0, escape=False)

def to_year(s):
    return pd.to_numeric(s.astype(str).str[:4], errors="coerce")

def to_ym(s):
    # returns pandas Period('YYYY-MM') for grouping, robust to 'YYYY' or 'YYYY-MM'
    ss = s.astype(str)
    y = ss.str[:4]
    m = ss.str[5:7].where(ss.str.len()>=7, "01")
    return pd.PeriodIndex(y + "-" + m, freq="M")

# =========================
# Forecasting (ETS: robust defaults)
# =========================
# --- replace your current ets_forecast() with this safe version ---

def ets_forecast(series: pd.Series, periods=12):
    """
    Safe Exponential Smoothing forecast:
    - If statsmodels is available: Holt-Winters (trend + seasonality if enough history)
    - If not, fall back to a simple moving-average extrapolation
    - Always returns a DataFrame with columns ['date','yhat'] (may be empty)
    """
    s = series.dropna().astype(float)
    if len(s) < 6:
        return pd.DataFrame(columns=["date","yhat"])

    # Try real ETS first
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        seasonal = "add" if len(s) >= 24 else None
        m = 12 if seasonal else None
        try:
            model = ExponentialSmoothing(
                s, trend="add", seasonal=seasonal, seasonal_periods=m, initialization_method="estimated"
            ).fit(optimized=True)
        except Exception:
            # fallback to Holt (trend only) if seasonal fit fails
            model = ExponentialSmoothing(
                s, trend="add", initialization_method="estimated"
            ).fit(optimized=True)
        f = model.forecast(periods)
        return pd.DataFrame({"date": f.index.to_timestamp(), "yhat": f.values})

    except Exception:
        # No statsmodels (or import error) -> moving-average drift fallback
        try:
            import pandas as _pd
            mean_val = float(_pd.Series(s).tail(12).mean()) if len(s) >= 12 else float(_pd.Series(s).mean())
            last_date = s.index[-1]
            # build a monthly date index forward
            if hasattr(last_date, "to_timestamp"):
                start = last_date.to_timestamp()
            else:
                start = _pd.Timestamp(last_date)
            dates = _pd.date_range(start=start + _pd.offsets.MonthBegin(1), periods=periods, freq="MS")
            return _pd.DataFrame({"date": dates, "yhat": [mean_val]*periods})
        except Exception:
            return pd.DataFrame(columns=["date","yhat"])


# =========================
# Charts
# =========================
def chart_wits_trade(df):
    w = df[df["source"].isin(["wits_trade"])].copy()
    if w.empty: return ""
    vcol = choose_numcol(w);  w["year"] = to_year(w["time"])
    q = (w["reporter"].str.upper().eq("USA") &
         w["partner"].str.upper().eq("WLD") &
         w["indicator"].isin(["MPRT-TRD-VL","XPRT-TRD-VL"]))
    g = w.loc[q].groupby(["indicator","year"], dropna=True)[vcol].sum().reset_index()
    if g.empty: return ""
    fig, ax = plt.subplots(figsize=(7.5,4.2))
    for ind, sub in g.groupby("indicator"):
        sub = sub.dropna(subset=["year"]).sort_values("year")
        ax.plot(sub["year"], sub[vcol], marker="o", label=ind)
    ax.set_title("WITS — USA ↔ World (Annual Trade Totals)")
    ax.set_xlabel("Year"); ax.set_ylabel("Value")
    ax.grid(True, alpha=.3); ax.legend()
    return img64(fig)

def chart_wits_tariff(df):
    t = df[df["source"].isin(["wits_tariff"])].copy()
    if t.empty: return ""
    vcol = choose_numcol(t);  t["year"] = to_year(t["time"])
    q = (t["reporter"].str.upper().eq("USA") &
         t["partner"].str.upper().eq("WLD") &
         t["indicator"].isin(["MFN-WGHTD-AVRG","MFN-SMPL-AVRG"]))
    g = t.loc[q].groupby(["indicator","year"], dropna=True)[vcol].mean().reset_index()
    if g.empty: return ""
    fig, ax = plt.subplots(figsize=(7.5,4.2))
    for ind, sub in g.groupby("indicator"):
        sub = sub.dropna(subset=["year"]).sort_values("year")
        ax.plot(sub["year"], sub[vcol], marker="o", label=ind)
    ax.set_title("WITS — MFN Average Tariff (USA ↔ World)")
    ax.set_xlabel("Year"); ax.set_ylabel("Percent (average)")
    ax.grid(True, alpha=.3); ax.legend()
    return img64(fig)

def chart_monthly(df, source_name, title, value_cols=("ALL_VAL_MO","Value","value")):
    m = df[df["source"].eq(source_name)].copy()
    if m.empty: return "", pd.DataFrame(), pd.DataFrame()
    # pick value column
    vcol = None
    for c in value_cols:
        if c in m.columns:
            vcol = c; break
    if not vcol: return "", pd.DataFrame(), pd.DataFrame()
    # time index (monthly)
    m["ym"] = to_ym(m["time"])
    # total across partners for each month
    g = m.groupby("ym")[vcol].sum().sort_index()
    if g.empty: return "", pd.DataFrame(), pd.DataFrame()
    # build chart
    fig, ax = plt.subplots(figsize=(7.5,4.2))
    ax.plot(g.index.to_timestamp(), g.values, marker="o")
    ax.set_title(title)
    ax.set_xlabel("Month"); ax.set_ylabel("Value")
    ax.grid(True, alpha=.3)
    img = img64(fig)
    # forecast 12 months
    fc = ets_forecast(g)
    fig2, ax2 = plt.subplots(figsize=(7.5,4.2))
    ax2.plot(g.index.to_timestamp(), g.values, label="History")
    if not fc.empty:
        ax2.plot(fc["date"], fc["yhat"], linestyle="--", marker="o", label="Forecast (12m)")
    ax2.set_title(title + " — 12-month Forecast")
    ax2.set_xlabel("Month"); ax2.set_ylabel("Value")
    ax2.grid(True, alpha=.3); ax2.legend()
    img_fc = img64(fig2)
    return img, g.to_frame(name="value").reset_index(), fc

def chart_ppi(df):
    b = df[df["source"].eq("bls_ppi")].copy()
    if b.empty: return "", pd.DataFrame(), pd.DataFrame()
    # pick numeric
    vcol = choose_numcol(b, ("value","Value"))
    if not vcol: return "", pd.DataFrame(), pd.DataFrame()
    # build monthly datetime from year+period (M01, M02, …)
    try:
        b = b.dropna(subset=["year","period"])[["year","period",vcol]].copy()
        b["month"] = b["period"].astype(str).str.replace("M","", regex=False).astype(int)
        b["date"] = pd.to_datetime(dict(year=b["year"].astype(int), month=b["month"], day=1))
        s = b.set_index("date")[vcol].astype(float).sort_index()
    except Exception:
        return "", pd.DataFrame(), pd.DataFrame()
    if s.empty: return "", pd.DataFrame(), pd.DataFrame()
    # chart
    fig, ax = plt.subplots(figsize=(7.5,4.2))
    ax.plot(s.index, s.values, marker="o")
    ax.set_title("BLS PPI — Producer Price Index (selected series)")
    ax.set_xlabel("Month"); ax.set_ylabel("Index")
    ax.grid(True, alpha=.3)
    img = img64(fig)
    # forecast 12 months
    fc = ets_forecast(s)
    fig2, ax2 = plt.subplots(figsize=(7.5,4.2))
    ax2.plot(s.index, s.values, label="History")
    if not fc.empty:
        ax2.plot(fc["date"], fc["yhat"], linestyle="--", marker="o", label="Forecast (12m)")
    ax2.set_title("BLS PPI — 12-month Forecast")
    ax2.set_xlabel("Month"); ax2.set_ylabel("Index")
    ax2.grid(True, alpha=.3); ax2.legend()
    img_fc = img64(fig2)
    return img, s.reset_index().rename(columns={vcol:"value"}), fc

# =========================
# Build HTML
# =========================
STYLE = """
<style>
body { font-family: system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif; margin: 24px; }
h1,h2,h3 { margin: 0.4em 0; }
.section { margin-top: 28px; }
table { border-collapse: collapse; width: 100%; font-size: 14px; }
th,td { border-bottom: 1px solid #eee; text-align: left; padding: 6px 8px; }
.badge { display:inline-block; background:#f3f4f6; padding:3px 8px; border-radius:8px; margin-right:8px; }
small { color:#666; }
code { background:#f6f8fa; padding:2px 5px; border-radius:4px; }
pre { white-space: pre-wrap; }
img { max-width: 100%; height: auto; }
.kpi { display:flex; gap:16px; flex-wrap:wrap; margin-top:8px; }
.card { border:1px solid #eee; border-radius:10px; padding:12px 14px; min-width:240px; }
.card b { display:block; font-size:18px; }
</style>
"""

def section(title, body):
    return f"<div class='section'><h2>{title}</h2>{body}</div>"

def build_report():
    pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    if not os.path.exists(CSV):
        html = f"""<html><body>
        <h1>Service Supply Chain — Auto Report</h1>
        <p><small>Generated: <b>{now_utc()}</b></small></p>
        <p>No <code>main.csv</code> found. Run <b>Update SSC Data</b> first.</p>
        </body></html>"""
        open(os.path.join(OUT_DIR, "Auto_Report.html"), "w", encoding="utf-8").write(html)
        print("[report] main.csv missing; wrote minimal HTML.")
        return

    df = pd.read_csv(CSV)

    # Overview by source (never blank)
    if "source" in df.columns:
        by_src = (df["source"].value_counts().rename_axis("source")
                  .reset_index(name="rows"))
    else:
        by_src = pd.DataFrame(columns=["source","rows"])

    # Build data profile for GPT & KPIs
    years = sorted(pd.to_numeric(df.get("time", pd.Series(dtype=str)).astype(str).str[:4], errors="coerce").dropna().unique().tolist())
    # Last values for quick KPIs
    def last_or_na(s): 
        return None if s is None or len(s)==0 else s[-1]

    # WITS KPIs
    wits = df[df["source"].eq("wits_trade")]
    wv = choose_numcol(wits)
    wits_kpi = {}
    if not wits.empty and wv:
        sel = (wits["reporter"].str.upper().eq("USA") &
               wits["partner"].str.upper().eq("WLD") &
               wits["indicator"].isin(["MPRT-TRD-VL","XPRT-TRD-VL"]))
        g = (wits.loc[sel].assign(year=to_year(wits.loc[sel,"time"]))
             .groupby(["indicator","year"], dropna=True)[wv].sum().reset_index()
             .sort_values(["indicator","year"]))
        for k, sub in g.groupby("indicator"):
            wits_kpi[k] = last_or_na(sub[wv].values.tolist())
    # Tariffs KPIs
    tariffs = df[df["source"].eq("wits_tariff")]
    tv = choose_numcol(tariffs)
    tariff_kpi = {}
    if not tariffs.empty and tv:
        sel = (tariffs["reporter"].str.upper().eq("USA") &
               tariffs["partner"].str.upper().eq("WLD") &
               tariffs["indicator"].isin(["MFN-WGHTD-AVRG","MFN-SMPL-AVRG"]))
        g = (tariffs.loc[sel].assign(year=to_year(tariffs.loc[sel,"time"]))
             .groupby(["indicator","year"], dropna=True)[tv].mean().reset_index()
             .sort_values(["indicator","year"]))
        for k, sub in g.groupby("indicator"):
            tariff_kpi[k] = last_or_na(sub[tv].values.tolist())

    # Charts + Forecasts
    img_trade = chart_wits_trade(df)
    img_tariff = chart_wits_tariff(df)

    img_cimp, cimp_hist, cimp_fc = chart_monthly(df, "census_imports", "Census — Total Imports (Monthly)")
    img_cexp, cexp_hist, cexp_fc = chart_monthly(df, "census_exports", "Census — Total Exports (Monthly)")
    img_ppi, ppi_hist, ppi_fc       = chart_ppi(df)

    # KPI cards
    pulled = ""
    if "pulled_at_utc" in df.columns and df["pulled_at_utc"].notna().any():
        pulled = str(df["pulled_at_utc"].dropna().astype(str).max())

    kpi_html = f"""
    <div class="kpi">
      <div class="card"><small>Last pull</small><b>{pulled or 'N/A'}</b></div>
      <div class="card"><small>Rows</small><b>{len(df):,}</b></div>
      <div class="card"><small>Sources</small><b>{len(by_src):,}</b></div>
    </div>
    """

    # Prepare context for GPT
    context = {
        "by_source": dict(by_src.values) if not by_src.empty else {},
        "years": years,
        "wits_last_points": wits_kpi,
        "tariff_last_points": tariff_kpi,
        "census_last_points": {
            "imports_last": None if cimp_hist.empty else float(cimp_hist["value"].iloc[-1]),
            "exports_last": None if cexp_hist.empty else float(cexp_hist["value"].iloc[-1]),
        },
        "ppi_last_points": None if ppi_hist.empty else float(ppi_hist["value"].iloc[-1]),
    }
    ai_summary = gpt_exec_summary(context)

    # Sample tables (clean)
    wits_cols   = ["freq","reporter","partner","product","indicator","time","Value"]
    census_cols = ["time","partner","partner_name","ALL_VAL_MO","ALL_VAL_YR"]
    bls_cols    = ["series_id","year","period","periodName","value"]

    samples_html = f"""
      <h3 class="badge">WITS Trade</h3>
      {table_html(df[df["source"].eq("wits_trade")], cols=wits_cols)}
      <h3 class="badge">WITS Tariff</h3>
      {table_html(df[df["source"].eq("wits_tariff")], cols=wits_cols)}
      <h3 class="badge">Census Imports</h3>
      {table_html(df[df["source"].eq("census_imports")], cols=census_cols)}
      <h3 class="badge">Census Exports</h3>
      {table_html(df[df["source"].eq("census_exports")], cols=census_cols)}
      <h3 class="badge">BLS PPI</h3>
      {table_html(df[df["source"].eq("bls_ppi")], cols=bls_cols)}
    """

    # Forecast commentary (short, deterministic)
    fc_notes = []
    if not cimp_fc.empty:
        fc_notes.append("Census imports: 12-month ETS forecast rendered.")
    if not cexp_fc.empty:
        fc_notes.append("Census exports: 12-month ETS forecast rendered.")
    if not ppi_fc.empty:
        fc_notes.append("BLS PPI: 12-month ETS forecast rendered.")

    fc_explainer = (
        "We forecast the next 12 months with an Exponential Smoothing (Holt-Winters) model. "
        "It is robust for monthly series, handles trend and—when enough history exists—seasonality (12). "
        "Use forecasts as directional guidance (baseline), not as hard targets."
    )

    # ---- NEW: build the forecasts block outside the big f-string (no nested f-strings)
    forecast_items = ''.join(f'<li>{x}</li>' for x in fc_notes) if fc_notes else '<li>No forecast series available.</li>'
    forecast_block = (
        "<ul>" + forecast_items + "</ul>"
        f"<p>{fc_explainer}</p>"
    )

    # Sample tables (clean)
    wits_cols   = ["freq","reporter","partner","product","indicator","time","Value"]
    census_cols = ["time","partner","partner_name","ALL_VAL_MO","ALL_VAL_YR"]
    bls_cols    = ["series_id","year","period","periodName","value"]

    samples_html = (
        "<h3 class='badge'>WITS Trade</h3>" +
        table_html(df[df["source"].eq("wits_trade")], cols=wits_cols) +
        "<h3 class='badge'>WITS Tariff</h3>" +
        table_html(df[df["source"].eq("wits_tariff")], cols=wits_cols) +
        "<h3 class='badge'>Census Imports</h3>" +
        table_html(df[df["source"].eq("census_imports")], cols=census_cols) +
        "<h3 class='badge'>Census Exports</h3>" +
        table_html(df[df["source"].eq("census_exports")], cols=census_cols) +
        "<h3 class='badge'>BLS PPI</h3>" +
        table_html(df[df["source"].eq("bls_ppi")], cols=bls_cols)
    )

    # Compose HTML (outer f-string only; we inject pre-built blocks/strings)
    html = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8"/>
<title>Service Supply Chain — Auto Report</title>
{STYLE}
<body>
<h1>Service Supply Chain — Auto Report</h1>
<p><small>Generated: <b>{now_utc()}</b>{' · Last pull: <b>'+pulled+'</b>' if pulled else ''}</small></p>
{section("Executive Summary (AI)", "<pre>"+ai_summary+"</pre>")}
{section("Overview", kpi_html + table_html(by_src, max_rows=20))}
{section("Trade — WITS (USA ↔ World)", ('<img src="'+img_trade+'"/>') if img_trade else '<p><em>No WITS trade data available.</em></p>')}
{section("Tariffs — WITS MFN Averages", ('<img src="'+img_tariff+'"/>') if img_tariff else '<p><em>No tariff data available.</em></p>')}
{section("Census Monthly Imports", ('<img src="'+img_cimp+'"/>') if img_cimp else '<p><em>No monthly imports data.</em></p>')}
{section("Census Monthly Exports", ('<img src="'+img_cexp+'"/>') if img_cexp else '<p><em>No monthly exports data.</em></p>')}
{section("BLS Producer Price Index", ('<img src="'+img_ppi+'"/>') if img_ppi else '<p><em>No PPI data.</em></p>')}
{section("12-Month Forecasts", forecast_block)}
{section("Data Samples", samples_html)}
</body>
</html>"""

    out = os.path.join(OUT_DIR, "Auto_Report.html")
    open(out, "w", encoding="utf-8").write(html)
    print("[report] Wrote:", out)

if __name__ == "__main__":
    build_report()



