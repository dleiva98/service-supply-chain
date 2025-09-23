# ssc_pipeline/build_report.py
import os, sys, io, base64, pathlib, datetime as dt
import numpy as np
import pandas as pd

# Headless plotting (CI)
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
    if not key: return None
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
        prompt = f"""
Write a short English narrative (2 short paragraphs + 3 bullets) for the chart titled "{title}".
Use ONLY the context provided; do not fabricate numbers.

Context:
{context_dict}

Explain acronyms inline (e.g., MFN, PPI). State what changed (YoY), the latest value in words,
and what the next 3–6 months likely mean operationally. Keep it non-technical and public-facing.
"""
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
    s = series.dropna().astype(float)
    if len(s) < 6:
        return pd.DataFrame(columns=["date","yhat"]), pd.DataFrame(columns=["date","lower","upper"])
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        seasonal = "add" if len(s) >= 24 else None
        m = 12 if seasonal else None
        model = ExponentialSmoothing(s, trend="add", seasonal=seasonal,
                                     seasonal_periods=m, initialization_method="estimated").fit(optimized=True)
        f = model.forecast(periods)
        resid_sd = float(np.nanstd(getattr(model, "resid", s - model.fittedvalues), ddof=1))
        dates = f.index.to_timestamp()
        fc   = pd.DataFrame({"date": dates, "yhat": f.values})
        band = pd.DataFrame({"date": dates,
                             "lower": f.values-1.96*resid_sd, "upper": f.values+1.96*resid_sd})
        return fc, band
    except Exception:
        # flat fallback
        mean_val = float(pd.Series(s).tail(12).mean()) if len(s)>=12 else float(pd.Series(s).mean())
        std_val  = float(pd.Series(s).tail(12).std(ddof=1)) if len(s)>=12 else float(pd.Series(s).std(ddof=1))
        start = s.index[-1].to_timestamp() if hasattr(s.index[-1],"to_timestamp") else pd.Timestamp(s.index[-1])
        dates = pd.date_range(start=start + pd.offsets.MonthBegin(1), periods=periods, freq="MS")
        fc   = pd.DataFrame({"date": dates, "yhat": [mean_val]*periods})
        band = pd.DataFrame({"date": dates, "lower": [mean_val-1.96*std_val]*periods,
                             "upper": [mean_val+1.96*std_val]*periods})
        return fc, band

def backtest_last_12(series: pd.Series):
    """One-shot backtest: train up to t-12, forecast 12, compare to actual 12."""
    s = series.dropna().astype(float)
    if len(s) < 24:
        return {"MAPE": None, "sMAPE": None, "RMSE": None, "n": 0}
    hist = s.iloc[:-12]
    true = s.iloc[-12:]
    fc, _ = ets_forecast_with_bands(hist, periods=12)
    if fc.empty: return {"MAPE": None, "sMAPE": None, "RMSE": None, "n": 0}
    pred = pd.Series(fc["yhat"].values, index=true.index)
    # metrics
    eps = 1e-9
    mape = float((np.abs((true - pred) / (true.replace(0, eps)))) .mean()) * 100.0
    smape = float((np.abs(true - pred) / ((np.abs(true) + np.abs(pred)) / 2.0 + eps)).mean()) * 100.0
    rmse = float(np.sqrt(((true - pred)**2).mean()))
    return {"MAPE": mape, "sMAPE": smape, "RMSE": rmse, "n": len(true)}

# ====================== Styles ======================
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

# ====================== Chart helpers (with AI narratives) ======================
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
        ctx[label] = {
            "last": human(sub[v].iloc[-1]),
            "yoy%": round(100.0*(sub[v].iloc[-1]/sub[v].iloc[-2]-1.0),1) if len(sub)>1 and sub[v].iloc[-2]!=0 else None
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
        ctx[label] = {
            "last_%": round(float(sub[v].iloc[-1]),2),
            "yoy_pp": round(float(sub[v].iloc[-1] - sub[v].iloc[-2]),2) if len(sub)>1 else None
        }
    ax.set_title("WITS — MFN (Most-Favoured-Nation) Average Tariffs"); ax.set_xlabel("Year"); ax.set_ylabel("Percent")
    ax.grid(True, alpha=.3); ax.legend()
    img = img64(fig)
    narrative = ai_narrative("WITS MFN Tariffs", ctx)
    return img, "<pre>"+narrative+"</pre>"

def monthly_series(df, source_name, title):
    m = df[df["source"].eq(source_name)].copy()
    if m.empty: return "", "", {}, pd.DataFrame(), pd.DataFrame()
    v = choose_numcol(m, ("ALL_VAL_MO","Value","value"))
    if not v: return "", "", {}, pd.DataFrame(), pd.DataFrame()
    m["ym"] = to_ym(m["time"])
    g = m.groupby("ym")[v].sum().sort_index()
    if g.empty: return "", "", {}, pd.DataFrame(), pd.DataFrame()

    # history
    fig, ax = plt.subplots(figsize=(7.8,4.4))
    ax.plot(g.index.to_timestamp(), g.values, marker="o")
    ax.set_title(title); ax.set_xlabel("Month"); ax.set_ylabel("Value"); ax.grid(True, alpha=.3)
    img = img64(fig)

    # forecast + bands + metrics
    fc, band = ets_forecast_with_bands(g)
    metrics = backtest_last_12(g)
    fig2, ax2 = plt.subplots(figsize=(7.8,4.4))
    ax2.plot(g.index.to_timestamp(), g.values, label="History")
    if not fc.empty:
        ax2.plot(fc["date"], fc["yhat"], linestyle="--", marker="o", label="Forecast (12m)")
    if not band.empty:
        ax2.fill_between(band["date"], band["lower"], band["upper"], alpha=0.15, label="95% conf. band")
    ax2.set_title(title + " — 12-month Forecast"); ax2.set_xlabel("Month"); ax2.set_ylabel("Value")
    ax2.grid(True, alpha=.3); ax2.legend()
    img_fc = img64(fig2)

    ctx = {
        "last": human(g.values[-1]),
        "yoy%": round(100.0*(g.values[-1]/g.values[-13]-1.0),1) if len(g)>=13 and g.values[-13]!=0 else None,
        "fc_last": (human(fc["yhat"].iloc[-1]) if not fc.empty else None),
        "metrics": metrics
    }
    narrative = ai_narrative(title, ctx)
    return img, img_fc, "<pre>"+narrative+"</pre>", g.to_frame("value"), fc

def ppi_series(df):
    b = df[df["source"].eq("bls_ppi")].copy()
    if b.empty: return "", "", {}, pd.DataFrame(), pd.DataFrame()
    v = choose_numcol(b, ("value","Value"))
    if not v: return "", "", {}, pd.DataFrame(), pd.DataFrame()
    try:
        b = b.dropna(subset=["year","period"])[["year","period",v]].copy()
        b["month"] = b["period"].astype(str).str.replace("M","", regex=False).astype(int)
        b["date"] = pd.to_datetime(dict(year=b["year"].astype(int), month=b["month"], day=1))
        s = b.set_index("date")[v].astype(float).sort_index()
    except Exception:
        return "", "", {}, pd.DataFrame(), pd.DataFrame()
    if s.empty: return "", "", {}, pd.DataFrame(), pd.DataFrame()

    # history
    fig, ax = plt.subplots(figsize=(7.8,4.4))
    ax.plot(s.index, s.values, marker="o")
    ax.set_title("BLS PPI — Producer Price Index (Upstream cost)"); ax.set_xlabel("Month"); ax.set_ylabel("Index")
    ax.grid(True, alpha=.3)
    img = img64(fig)

    # forecast + bands + metrics
    fc, band = ets_forecast_with_bands(s)
    metrics = backtest_last_12(s)
    fig2, ax2 = plt.subplots(figsize=(7.8,4.4))
    ax2.plot(s.index, s.values, label="History")
    if not fc.empty:
        ax2.plot(fc["date"], fc["yhat"], linestyle="--", marker="o", label="Forecast (12m)")
    if not band.empty:
        ax2.fill_between(band["date"], band["lower"], band["upper"], alpha=0.15, label="95% conf. band")
    ax2.set_title("BLS PPI — 12-month Forecast"); ax2.set_xlabel("Month"); ax2.set_ylabel("Index")
    ax2.grid(True, alpha=.3); ax2.legend()
    img_fc = img64(fig2)

    ctx = {
        "last_index": round(float(s.values[-1]), 1),
        "yoy%": round(100.0*(s.values[-1]/s.values[-13]-1.0),1) if len(s)>=13 and s.values[-13]!=0 else None,
        "fc_last": (round(float(fc['yhat'].iloc[-1]),1) if not fc.empty else None),
        "metrics": metrics
    }
    narrative = ai_narrative("BLS PPI", ctx)
    return img, img_fc, "<pre>"+narrative+"</pre>", s.reset_index().rename(columns={v:"value"}), fc

# ====================== Partner Deep Dive (names + HHI) ======================
def partner_name_series(x, fallback):
    if pd.isna(x): return fallback
    x = str(x).upper()
    return PARTNER_NAME.get(x, fallback)

def partner_deep_dive(df, source_name, title, top=5):
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
    return img, expl + table_html(yoy, cols=["partner_label","Latest","PrevYear","YoY%"])

# ====================== Build ======================
def build_report():
    pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(CSV):
        html = f"<html><body><h1>Service Supply Chain — Auto Report</h1><p>No main.csv found.</p></body></html>"
        open(os.path.join(OUT_DIR,"Auto_Report.html"),"w",encoding="utf-8").write(html); return

    df = pd.read_csv(CSV)
    pulled = df.get("pulled_at_utc", pd.Series(dtype=str)).dropna().astype(str).max() if "pulled_at_utc" in df.columns else ""

    by_src = (df["source"].value_counts().rename_axis("source").reset_index(name="rows")
              if "source" in df.columns else pd.DataFrame(columns=["source","rows"]))

    # KPI
    kpi_html = f"""
    <div class="kpi">
      <div class="card"><small>Last pull</small><b>{pulled or 'N/A'}</b></div>
      <div class="card"><small>Total rows</small><b>{len(df):,}</b></div>
      <div class="card"><small>Sources</small><b>{len(by_src):,}</b></div>
    </div>
    """

    # Charts + AI narratives
    trade_img,  trade_ai  = line_wits_trade(df)
    tariff_img, tariff_ai = line_wits_tariffs(df)

    imp_img, imp_fc_img, imp_ai, imp_hist, imp_fc = monthly_series(df, "census_imports", "Census — Total Imports (Monthly)")
    exp_img, exp_fc_img, exp_ai, exp_hist, exp_fc = monthly_series(df, "census_exports", "Census — Total Exports (Monthly)")
    ppi_img, ppi_fc_img, ppi_ai, ppi_hist, ppi_fc = ppi_series(df)

    # Partner deep dives with names
    imp_pd_img, imp_pd_table = partner_deep_dive(df, "census_imports", "Partner Deep-Dive — Imports")
    exp_pd_img, exp_pd_table = partner_deep_dive(df, "census_exports", "Partner Deep-Dive — Exports")

    # Methodology & glossary
    methodology = """
<ul class='note'>
  <li><b>Models</b>: Exponential Smoothing (Holt-Winters, ETS). Trend always on; seasonality auto (12) if enough history.</li>
  <li><b>Uncertainty</b>: Shaded area shows ~95% confidence band from residual variance (fallback: ±1σ).</li>
  <li><b>Backtest</b>: One rolling split: train up to t-12, forecast 12; report MAPE, sMAPE, RMSE for each forecasted series.</li>
  <li><b>Data</b>: WITS (trade, tariffs), US Census (monthly flows), BLS (PPI).</li>
  <li><b>Units</b>: K=thousand, M=million, B=billion, T=trillion.</li>
</ul>
"""
    glossary = """
<ul class='note'>
  <li><b>WITS</b>: World Integrated Trade Solution (World Bank).</li>
  <li><b>MFN</b>: Most-Favoured-Nation tariff (baseline rate applied to WTO members).</li>
  <li><b>PPI</b>: Producer Price Index (upstream cost indicator).</li>
  <li><b>ETS</b>: Exponential Smoothing (Holt-Winters) time-series model.</li>
  <li><b>HHI</b>: Herfindahl–Hirschman Index (market/partner concentration; 0–10,000; ≥2,500 high).</li>
  <li><b>YoY</b>: Year-over-Year change.</li>
</ul>
"""

    # Compose HTML
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

{section("Census Imports (Monthly)", (('<img src="'+imp_img+'"/>') if imp_img else '') + imp_ai)}
{section("Census Imports — 12-month Forecast", (('<img src="'+imp_fc_img+'"/>') if imp_fc_img else '') )}

{section("Census Exports (Monthly)", (('<img src="'+exp_img+'"/>') if exp_img else '') + exp_ai)}
{section("Census Exports — 12-month Forecast", (('<img src="'+exp_fc_img+'"/>') if exp_fc_img else '') )}

{section("BLS PPI (Upstream Cost)", (('<img src="'+ppi_img+'"/>') if ppi_img else '') + ppi_ai)}
{section("BLS PPI — 12-month Forecast", (('<img src="'+ppi_fc_img+'"/>') if ppi_fc_img else '') )}

{section("Partner Deep-Dive — Imports (Top partners, YoY & HHI)", (('<img src="'+imp_pd_img+'"/>') if imp_pd_img else '') + imp_pd_table)}
{section("Partner Deep-Dive — Exports (Top partners, YoY & HHI)", (('<img src="'+exp_pd_img+'"/>') if exp_pd_img else '') + exp_pd_table)}

{section("Methodology & Credibility", methodology)}
{section("Glossary", glossary)}

</body></html>"""

    out = os.path.join(OUT_DIR, "Auto_Report.html")
    open(out, "w", encoding="utf-8").write(html)
    print("[report] Wrote:", out)

if __name__ == "__main__":
    build_report()
