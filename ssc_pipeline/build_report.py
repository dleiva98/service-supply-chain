# ssc_pipeline/build_report.py
import os, sys, io, base64, pathlib, datetime as dt
import numpy as np
import pandas as pd

# Headless plotting for CI
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.environ.get("SSC_ROOT", os.getcwd())
OUT_DIR = os.path.join(ROOT, "ssc_pipeline")
CSV = os.path.join(ROOT, "main.csv")

# ========= Partner name mapping (ISO-3 and common WITS tags) =========
PARTNER_NAME = {
    "USA":"United States", "WLD":"World", "CHN":"China", "MEX":"Mexico", "CAN":"Canada",
    "JPN":"Japan", "DEU":"Germany", "GBR":"United Kingdom", "KOR":"South Korea",
    "IND":"India", "FRA":"France", "ITA":"Italy", "BRA":"Brazil", "TWN":"Taiwan",
    "HKG":"Hong Kong", "VNM":"Vietnam", "NLD":"Netherlands", "SGP":"Singapore",
    "IRL":"Ireland", "CHE":"Switzerland", "ESP":"Spain", "AUS":"Australia",
    "TUR":"Türkiye", "POL":"Poland", "THA":"Thailand", "MYS":"Malaysia",
    "IDN":"Indonesia", "PHL":"Philippines", "BEL":"Belgium", "SWE":"Sweden",
    "NOR":"Norway", "ISR":"Israel", "SAU":"Saudi Arabia", "ARE":"United Arab Emirates",
    "ARG":"Argentina", "CHL":"Chile", "PER":"Peru", "COL":"Colombia",
}

# ====================== Utilities ======================
def human(x):
    try:
        x = float(x)
    except Exception:
        return str(x)
    sgn = "-" if x < 0 else ""
    y = abs(x)
    if y >= 1e12: return f"{sgn}{y/1e12:.2f}T"
    if y >= 1e9:  return f"{sgn}{y/1e9:.2f}B"
    if y >= 1e6:  return f"{sgn}{y/1e6:.2f}M"
    if y >= 1e3:  return f"{sgn}{y/1e3:.2f}K"
    return f"{sgn}{y:.0f}"

def now_utc(): return dt.datetime.utcnow().replace(microsecond=0).isoformat()+"Z"

def img64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig); buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")

def choose_numcol(df, candidates=("Value","value","ALL_VAL_MO")):
    for c in candidates:
        if c in df.columns: return c
    return None

def safe_year(s):   return pd.to_numeric(s.astype(str).str[:4], errors="coerce")

def to_ym(s):
    ss = s.astype(str)
    y = ss.str[:4]; m = ss.str[5:7].where(ss.str.len()>=7, "01")
    return pd.PeriodIndex(y+"-"+m, freq="M")

def table_html(df, max_rows=12, cols=None):
    if df is None or df.empty: return "<p><em>(no data)</em></p>"
    out = df.copy()
    if cols:
        keep = [c for c in cols if c in out.columns]
        if keep: out = out[keep]
    return out.fillna("").head(max_rows).to_html(index=False, border=0, escape=False)

# ====================== Optional AI ======================
def _gpt():
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        return None
    try:
        from openai import OpenAI
        return OpenAI(api_key=key)
    except Exception:
        return None

def ai_narrative(title, context_dict, max_tokens=450):
    client = _gpt()
    if client is None:
        return f"({title}: AI commentary disabled — add OPENAI_API_KEY as a repository secret.)"
    try:
        sysmsg = "You are a senior supply-chain economist and data storyteller. Be concise, factual, executive."
        prompt = (
            f'Write a short English narrative (2 short paragraphs + 3 bullets) for the chart titled "{title}".\n'
            "Use ONLY the context provided; do not fabricate numbers.\n\n"
            f"Context:\n{context_dict}\n\n"
            "Explain acronyms inline (e.g., MFN, PPI). State what changed (YoY), the latest value in words, "
            "and what the next 3–6 months likely mean operationally. Keep it non-technical and public-facing."
        )
        out = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":sysmsg},{"role":"user","content":prompt}],
            temperature=0.35, max_tokens=max_tokens
        )
        return out.choices[0].message.content.strip()
    except Exception as e:
        return f"({title}: AI error: {e})"

# ====================== Forecasting & metrics ======================
def ets_forecast_with_bands(series: pd.Series, periods=12):
    """
    ETS/Holt-Winters with 95% band; safe fallback if statsmodels missing.
    Returns (fc_df, band_df)
    """
    s = series.dropna().astype(float)
    if len(s) < 6:
        return pd.DataFrame(columns=["date","yhat"]), pd.DataFrame(columns=["date","lower","upper"])
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        seasonal = "add" if len(s) >= 24 else None
        m = 12 if seasonal else None
        model = ExponentialSmoothing(
            s, trend="add", seasonal=seasonal, seasonal_periods=m, initialization_method="estimated"
        ).fit(optimized=True)
        f = model.forecast(periods)
        resid = getattr(model, "resid", s - model.fittedvalues)
        resid_sd = float(np.nanstd(resid, ddof=1)) if len(resid) > 1 else float(np.nanstd(s))
        dates = f.index.to_timestamp() if hasattr(f.index, "to_timestamp") else pd.to_datetime(f.index)
        fc   = pd.DataFrame({"date": dates, "yhat": f.values})
        band = pd.DataFrame({"date": dates,
                             "lower": f.values - 1.96*resid_sd,
                             "upper": f.values + 1.96*resid_sd})
        return fc, band
    except Exception:
        # Fallback: flat mean ± 1σ
        mean_val = float(pd.Series(s).tail(12).mean()) if len(s)>=12 else float(pd.Series(s).mean())
        std_val  = float(pd.Series(s).tail(12).std(ddof=1)) if len(s)>=12 else float(pd.Series(s).std(ddof=1))
        last = s.index[-1]
        start = last.to_timestamp() if hasattr(last, "to_timestamp") else pd.Timestamp(last)
        dates = pd.date_range(start=start + pd.offsets.MonthBegin(1), periods=periods, freq="MS")
        fc   = pd.DataFrame({"date": dates, "yhat": [mean_val]*periods})
        band = pd.DataFrame({"date": dates, "lower": [mean_val-1.96*std_val]*periods,
                             "upper": [mean_val+1.96*std_val]*periods})
        return fc, band

def backtest_last_12_with_coverage(series: pd.Series):
    """
    One-shot backtest:
      - train up to t-12, forecast 12 months
      - build 95% band from residual std of training fit
      - compute MAPE, sMAPE, RMSE, Coverage@95%
    """
    s = series.dropna().astype(float)
    if len(s) < 24:
        return {"MAPE": None, "sMAPE": None, "RMSE": None, "Coverage95": None, "n": 0}

    hist = s.iloc[:-12]
    true = s.iloc[-12:]
    fc, band = ets_forecast_with_bands(hist, periods=12)
    if fc.empty:
        return {"MAPE": None, "sMAPE": None, "RMSE": None, "Coverage95": None, "n": 0}

    pred = pd.Series(fc["yhat"].values, index=true.index)
    if not band.empty:
        lower = pd.Series(band["lower"].values, index=true.index)
        upper = pd.Series(band["upper"].values, index=true.index)
        coverage = float(((true >= lower) & (true <= upper)).mean()) * 100.0
    else:
        coverage = None

    eps = 1e-9
    mape  = float((np.abs((true - pred) / (true.replace(0, eps)))).mean()) * 100.0
    smape = float((np.abs(true - pred) / ((np.abs(true) + np.abs(pred)) / 2.0 + eps)).mean()) * 100.0
    rmse  = float(np.sqrt(((true - pred) ** 2).mean()))
    return {"MAPE": mape, "sMAPE": smape, "RMSE": rmse, "Coverage95": coverage, "n": int(true.shape[0])}

# ====================== Styles & sections ======================
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
.note { color:#374151; font-size:13px; }
</style>
"""
def section(title, body): return f"<div class='section'><h2>{title}</h2>{body}</div>"

# ====================== Charts & narratives ======================
def line_wits_trade(df):
    w = df[df["source"].eq("wits_trade")].copy()
    if w.empty: return "", "(no data)"
    v = choose_numcol(w);  w["year"] = safe_year(w["time"])
    q = (w["reporter"].str.upper().eq("USA") &
         w["partner"].str.upper().eq("WLD") &
         w["indicator"].isin(["MPRT-TRD-VL","XPRT-TRD-VL"]))
    g = w.loc[q].groupby(["indicator","year"], dropna=True)[v].sum().reset_index()
    if g.empty: return "", "(no data)"
    fig, ax = plt.subplots(figsize=(7.8,4.4))
    ctx = {}
    for ind, sub in g.groupby("indicator"):
        sub = sub.dropna(subset=["year"]).sort_values("year")
        label = "Imports (MPRT)" if ind=="MPRT-TRD-VL" else "Exports (XPRT)"
        ax.plot(sub["year"], sub[v], marker="o", label=label)
        last = sub[v].iloc[-1]
        prev = sub[v].iloc[-2] if len(sub)>1 else None
        ctx[label] = {
            "last": human(last),
            "yoy%": round(100.0*(last/prev-1.0),1) if prev not in (None,0) else None
        }
    ax.set_title("WITS — US Goods Trade with World (Annual)"); ax.set_xlabel("Year"); ax.set_ylabel("Value")
    ax.grid(True, alpha=.3); ax.legend()
    img = img64(fig)
    narrative = ai_narrative("WITS Trade (US↔World)", ctx)
    return img, "<pre>"+narrative+"</pre>"

def line_wits_tariffs(df):
    t = df[df["source"].eq("wits_tariff")].copy()
    if t.empty: return "", "(no data)"
    v = choose_numcol(t);  t["year"] = safe_year(t["time"])
    q = (t["reporter"].str.upper().eq("USA") &
         t["partner"].str.upper().eq("WLD") &
         t["indicator"].isin(["MFN-WGHTD-AVRG","MFN-SMPL-AVRG"]))
    g = t.loc[q].groupby(["indicator","year"], dropna=True)[v].mean().reset_index()
    if g.empty: return "", "(no data)"
    fig, ax = plt.subplots(figsize=(7.8,4.4))
    ctx = {}
    for ind, sub in g.groupby("indicator"):
        sub = sub.dropna(subset=["year"]).sort_values("year")
        label = "MFN Weighted Avg" if ind=="MFN-WGHTD-AVRG" else "MFN Simple Avg"
        ax.plot(sub["year"], sub[v], marker="o", label=label)
        last = float(sub[v].iloc[-1]); prev = float(sub[v].iloc[-2]) if len(sub)>1 else None
        ctx[label] = {"last_%": round(last,2), "yoy_pp": round(last - prev,2) if prev is not None else None}
    ax.set_title("WITS — MFN (Most-Favoured-Nation) Average Tariffs"); ax.set_xlabel("Year"); ax.set_ylabel("Percent")
    ax.grid(True, alpha=.3); ax.legend()
    img = img64(fig)
    narrative = ai_narrative("WITS MFN Tariffs", ctx)
    return img, "<pre>"+narrative+"</pre>"

def monthly_series(df, source_name, title):
    m = df[df["source"].eq(source_name)].copy()
    if m.empty: return "", "", "", "", {}, pd.DataFrame(), pd.DataFrame()
    v = choose_numcol(m, ("ALL_VAL_MO","Value","value"))
    if not v: return "", "", "", "", {}, pd.DataFrame(), pd.DataFrame()
    m["ym"] = to_ym(m["time"])
    g = m.groupby("ym")[v].sum().sort_index()
    if g.empty: return "", "", "", "", {}, pd.DataFrame(), pd.DataFrame()

    # history chart
    fig, ax = plt.subplots(figsize=(7.8,4.4))
    ax.plot(g.index.to_timestamp(), g.values, marker="o")
    ax.set_title(title); ax.set_xlabel("Month"); ax.set_ylabel("Value"); ax.grid(True, alpha=.3)
    img_hist = img64(fig)

    # forecast + bands + metrics (incl. coverage)
    fc, band = ets_forecast_with_bands(g)
    metrics = backtest_last_12_with_coverage(g)

    fig2, ax2 = plt.subplots(figsize=(7.8,4.4))
    ax2.plot(g.index.to_timestamp(), g.values, label="History")
    if not fc.empty:
        ax2.plot(fc["date"], fc["yhat"], linestyle="--", marker="o", label="Forecast (12m)")
    if not band.empty:
        ax2.fill_between(band["date"], band["lower"], band["upper"], alpha=0.15, label="95% conf. band")
    ax2.set_title(title + " — 12-month Forecast"); ax2.set_xlabel("Month"); ax2.set_ylabel("Value")
    ax2.grid(True, alpha=.3); ax2.legend()
    img_fc = img64(fig2)

    # AI narrative for HISTORY
    ctx_hist = {
        "last": (human(g.values[-1]) if len(g)>0 else None),
        "yoy%": round(100.0*(g.values[-1]/g.values[-13]-1.0),1) if len(g)>=13 and g.values[-13]!=0 else None
    }
    hist_narrative = ai_narrative(title, ctx_hist)

    # AI narrative for FORECAST (includes scores)
    ctx_fc = {
        "last_forecast_point": (human(fc["yhat"].iloc[-1]) if not fc.empty else None),
        "band": "95% confidence band",
        "metrics": metrics
    }
    fc_title = title + " — Forecast"
    fc_narrative = ai_narrative(fc_title, ctx_fc)

    return img_hist, img_fc, "<pre>"+hist_narrative+"</pre>", "<pre>"+fc_narrative+"</pre>", metrics, g.to_frame("value"), fc

def ppi_series(df):
    b = df[df["source"].eq("bls_ppi")].copy()
    if b.empty: return "", "", "", "", {}, pd.DataFrame(), pd.DataFrame()
    v = choose_numcol(b, ("value","Value"))
    if not v: return "", "", "", "", {}, pd.DataFrame(), pd.DataFrame()
    try:
        b = b.dropna(subset=["year","period"])[["year","period",v]].copy()
        b["month"] = b["period"].astype(str).str.replace("M","", regex=False).astype(int)
        b["date"] = pd.to_datetime(dict(year=b["year"].astype(int), month=b["month"], day=1))
        s = b.set_index("date")[v].astype(float).sort_index()
    except Exception:
        return "", "", "", "", {}, pd.DataFrame(), pd.DataFrame()
    if s.empty: return "", "", "", "", {}, pd.DataFrame(), pd.DataFrame()

    # history
    fig, ax = plt.subplots(figsize=(7.8,4.4))
    ax.plot(s.index, s.values, marker="o")
    ax.set_title("BLS PPI — Producer Price Index (Upstream cost)"); ax.set_xlabel("Month"); ax.set_ylabel("Index")
    ax.grid(True, alpha=.3)
    img_hist = img64(fig)

    # forecast + bands + metrics (incl. coverage)
    fc, band = ets_forecast_with_bands(s)
    metrics = backtest_last_12_with_coverage(s)

    fig2, ax2 = plt.subplots(figsize=(7.8,4.4))
    ax2.plot(s.index, s.values, label="History")
    if not fc.empty:
        ax2.plot(fc["date"], fc["yhat"], linestyle="--", marker="o", label="Forecast (12m)")
    if not band.empty:
        ax2.fill_between(band["date"], band["lower"], band["upper"], alpha=0.15, label="95% conf. band")
    ax2.set_title("BLS PPI — 12-month Forecast"); ax2.set_xlabel("Month"); ax2.set_ylabel("Index")
    ax2.grid(True, alpha=.3); ax2.legend()
    img_fc = img64(fig2)

    # AI narratives
    ctx_hist = {
        "last_index": round(float(s.values[-1]), 1) if len(s)>0 else None,
        "yoy%": round(100.0*(s.values[-1]/s.values[-13]-1.0),1) if len(s)>=13 and s.values[-13]!=0 else None
    }
    hist_narrative = ai_narrative("BLS PPI (History)", ctx_hist)

    ctx_fc = {
        "last_forecast_point": (round(float(fc['yhat'].iloc[-1]),1) if not fc.empty else None),
        "band": "95% confidence band",
        "metrics": metrics
    }
    fc_narrative = ai_narrative("BLS PPI — Forecast", ctx_fc)

    return img_hist, img_fc, "<pre>"+hist_narrative+"</pre>", "<pre>"+fc_narrative+"</pre>", metrics, s.reset_index().rename(columns={v:"value"}), fc

# ====================== Partner Deep Dive (names + HHI) ======================
def partner_name_series(x, fallback):
    if pd.isna(x): return fallback
    x = str(x).upper()
    return PARTNER_NAME.get(x, fallback)

def partner_deep_dive(df, source_name, title, top=5, with_ai=False):
    m = df[df["source"].eq(source_name)].copy()
    if m.empty: return "", "<p><em>(no data)</em></p>"
    v = choose_numcol(m, ("ALL_VAL_MO","Value","value"))
    if not v: return "", "<p><em>(no data)</em></p>"
    m["ym"] = to_ym(m["time"])
    # human partner labels
    if "partner_name" in m.columns and m["partner_name"].notna().any():
        m["partner_label"] = m["partner_name"].astype(str)
    else:
        m["partner_label"] = m["partner"].apply(lambda c: partner_name_series(c, str(c)))
    last12 = m[m["ym"] >= (m["ym"].max() - 11)]
    top_partners = (last12.groupby("partner_label")[v].sum()
                    .sort_values(ascending=False).head(top).index.tolist())
    mm = m[m["partner_label"].isin(top_partners)]
    g = mm.groupby(["partner_label","ym"])[v].sum().reset_index()

    fig, ax = plt.subplots(figsize=(8.2,4.6))
    for p, sub in g.groupby("partner_label"):
        sub = sub.sort_values("ym")
        ax.plot(sub["ym"].dt.to_timestamp(), sub[v], marker="o", label=str(p))
    ax.set_title(title + " — Top Partners (last 12m leaders)")
    ax.set_xlabel("Month"); ax.set_ylabel("Value"); ax.grid(True, alpha=.3); ax.legend(ncol=2, fontsize=9)
    img = img64(fig)

    latest = m["ym"].max()
    this = m[m["ym"].eq(latest)].groupby("partner_label")[v].sum()
    prev = m[m["ym"].eq(latest - 12)].groupby("partner_label")[v].sum() if (latest - 12) in m["ym"].unique() else pd.Series(dtype=float)
    yoy = pd.DataFrame({"Latest": this, "PrevYear": prev}).reset_index()
    yoy["YoY%"] = 100.0*((yoy["Latest"]/yoy["PrevYear"]) - 1.0)
    yoy = yoy.sort_values("Latest", ascending=False)

    shares = last12.groupby("partner_label")[v].sum()
    hhi = None
    if shares.sum()>0:
        s = shares / shares.sum()
        hhi = float((s.pow(2).sum())*10000)
    expl = "<p class='note'>Concentration index (HHI, last 12m): " + (f"{hhi:,.0f}" if hhi is not None else "n/a") + \
           " &nbsp; · &nbsp; Rule of thumb: ≥2500 = high concentration.</p>"

    html_table = table_html(yoy, cols=["partner_label","Latest","PrevYear","YoY%"])
    if with_ai:
        ctx = {
            "top_partners": yoy["partner_label"].head(5).tolist(),
            "latest_top": human(yoy["Latest"].head(1).values[0]) if len(yoy)>0 else None,
            "hhi": hhi
        }
        narr = ai_narrative(title, ctx)
        return img, "<pre>"+narr+"</pre>" + expl + html_table
    else:
        return img, expl + html_table

# ====================== Build report ======================
def build_report():
    pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(CSV):
        html = f"<html><body><h1>Service Supply Chain — Auto Report</h1><p>No main.csv found.</p></body></html>"
        open(os.path.join(OUT_DIR,"Auto_Report.html"),"w",encoding="utf-8").write(html); return

    df = pd.read_csv(CSV)
    pulled = df.get("pulled_at_utc", pd.Series(dtype=str)).dropna().astype(str).max() if "pulled_at_utc" in df.columns else ""

    # Overview by source
    by_src = (df["source"].value_counts().rename_axis("source").reset_index(name="rows")
              if "source" in df.columns else pd.DataFrame(columns=["source","rows"]))

    # KPI cards
    kpi_html = f"""
    <div class="kpi">
      <div class="card"><small>Last pull</small><b>{pulled or 'N/A'}</b></div>
      <div class="card"><small>Total rows</small><b>{len(df):,}</b></div>
      <div class="card"><small>Sources</small><b>{len(by_src):,}</b></div>
    </div>
    """

    # WITS Trade & Tariffs (history with AI narratives)
    trade_img,  trade_ai  = line_wits_trade(df)
    tariff_img, tariff_ai = line_wits_tariffs(df)

    # Monthly series & PPI: history + forecast + AI + metrics
    imp_img, imp_fc_img, imp_hist_ai, imp_fc_ai, imp_metrics, imp_hist, imp_fc = monthly_series(
        df, "census_imports", "Census — Total Imports (Monthly)"
    )
    exp_img, exp_fc_img, exp_hist_ai, exp_fc_ai, exp_metrics, exp_hist, exp_fc = monthly_series(
        df, "census_exports", "Census — Total Exports (Monthly)"
    )
    ppi_img, ppi_fc_img, ppi_hist_ai, ppi_fc_ai, ppi_metrics, ppi_hist, ppi_fc = ppi_series(df)

    # Partner deep-dives (with names) — add with_ai=True to include narrative if desired
    imp_pd_img, imp_pd_table = partner_deep_dive(df, "census_imports", "Partner Deep-Dive — Imports", with_ai=True)
    exp_pd_img, exp_pd_table = partner_deep_dive(df, "census_exports", "Partner Deep-Dive — Exports", with_ai=True)

    # Scores table for Methodology
    def fmt_pct(x): return "" if x is None else f"{x:.1f}%"
    def fmt_num(x): return "" if x is None else f"{x:,.2f}"

    scores_df = pd.DataFrame([
        {"Series": "Census Imports (Monthly)",
         "MAPE": fmt_pct(imp_metrics.get("MAPE")),
         "sMAPE": fmt_pct(imp_metrics.get("sMAPE")),
         "RMSE": fmt_num(imp_metrics.get("RMSE")),
         "Coverage@95": fmt_pct(imp_metrics.get("Coverage95"))},
        {"Series": "Census Exports (Monthly)",
         "MAPE": fmt_pct(exp_metrics.get("MAPE")),
         "sMAPE": fmt_pct(exp_metrics.get("sMAPE")),
         "RMSE": fmt_num(exp_metrics.get("RMSE")),
         "Coverage@95": fmt_pct(exp_metrics.get("Coverage95"))},
        {"Series": "BLS PPI",
         "MAPE": fmt_pct(ppi_metrics.get("MAPE")),
         "sMAPE": fmt_pct(ppi_metrics.get("sMAPE")),
         "RMSE": fmt_num(ppi_metrics.get("RMSE")),
         "Coverage@95": fmt_pct(ppi_metrics.get("Coverage95"))},
    ]).fillna("")

    methodology_html = """
<ul class='note'>
  <li><b>Model:</b> ETS / Holt-Winters (trend always on; seasonality=12 when history allows).</li>
  <li><b>Uncertainty:</b> We plot a 95% confidence band using residual variance from the training fit.</li>
  <li><b>Backtest:</b> Train up to t−12, forecast 12; we report MAPE, sMAPE, RMSE, and Coverage@95 (share of backtest points inside the 95% band).</li>
</ul>
"""
    glossary = """
<ul class='note'>
  <li><b>WITS</b>: World Integrated Trade Solution (World Bank).</li>
  <li><b>MFN</b>: Most-Favoured-Nation (baseline tariff for WTO members).</li>
  <li><b>PPI</b>: Producer Price Index (upstream cost indicator).</li>
  <li><b>ETS</b>: Exponential Smoothing (Holt-Winters) time-series model.</li>
  <li><b>HHI</b>: Herfindahl–Hirschman Index (0–10,000; ≥2,500 = high concentration).</li>
  <li><b>YoY</b>: Year-over-Year change.</li>
</ul>
"""
    methodology_block = methodology_html + "<h3>Model Scores</h3>" + table_html(scores_df) + "<h3>Glossary</h3>" + glossary

    # Compose full HTML
    html = f"""<!doctype html>
<html lang="en"><meta charset="utf-8"/>
<title>Service Supply Chain — Auto Report</title>
{STYLE}
<body>
<h1>Service Supply Chain — Auto Report</h1>
<p><small>Generated: <b>{now_utc()}</b>{' · Last pull: <b>'+pulled+'</b>' if pulled else ''}</small></p>

{section("Overview", kpi_html + table_html(by_src, max_rows=20))}

{section("WITS — US Trade with World (Annual)", (('<img src="'+trade_img+'"/>') if trade_img else '') + trade_ai)}
{section("WITS — MFN Average Tariffs", (('<img src="'+tariff_img+'"/>') if tariff_img else '') + tariff_ai)}

{section("Census Imports (Monthly)", (('<img src="'+imp_img+'"/>') if imp_img else '') + imp_hist_ai)}
{section("Census Imports — 12-month Forecast", (('<img src="'+imp_fc_img+'"/>') if imp_fc_img else '') + imp_fc_ai)}

{section("Census Exports (Monthly)", (('<img src="'+exp_img+'"/>') if exp_img else '') + exp_hist_ai)}
{section("Census Exports — 12-month Forecast", (('<img src="'+exp_fc_img+'"/>') if exp_fc_img else '') + exp_fc_ai)}

{section("BLS PPI (Upstream Cost)", (('<img src="'+ppi_img+'"/>') if ppi_img else '') + ppi_hist_ai)}
{section("BLS PPI — 12-month Forecast", (('<img src="'+ppi_fc_img+'"/>') if ppi_fc_img else '') + ppi_fc_ai)}

{section("Partner Deep-Dive — Imports (Top partners, YoY & HHI)", (('<img src="'+imp_pd_img+'"/>') if imp_pd_img else '') + imp_pd_table)}
{section("Partner Deep-Dive — Exports (Top partners, YoY & HHI)", (('<img src="'+exp_pd_img+'"/>') if exp_pd_img else '') + exp_pd_table)}

{section("Methodology", methodology_block)}

</body></html>"""

    out = os.path.join(OUT_DIR, "Auto_Report.html")
    open(out, "w", encoding="utf-8").write(html)
    print("[report] Wrote:", out)

if __name__ == "__main__":
    build_report()
