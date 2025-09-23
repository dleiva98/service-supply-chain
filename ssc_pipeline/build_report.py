# ssc_pipeline/build_report.py
import os, sys, io, base64, pathlib, datetime as dt, textwrap
import numpy as np
import pandas as pd

# Headless plotting (CI-safe)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.environ.get("SSC_ROOT", os.getcwd())
OUT_DIR = os.path.join(ROOT, "ssc_pipeline")
CSV = os.path.join(ROOT, "main.csv")

# ------------------------------------------------------------
# Optional GPT sections (executive summary & geopolitics)
# ------------------------------------------------------------
def _gpt_enabled():
    return bool(os.environ.get("OPENAI_API_KEY", "").strip())

def gpt_brief(title: str, prompt: str, max_tokens=900):
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        return f"({title}: AI section disabled — add OPENAI_API_KEY to repo secrets.)"
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        messages = [
            {"role":"system","content":"You are a senior supply-chain economist. Be factual, concise, executive."},
            {"role":"user","content":prompt}
        ]
        out = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.35,
            max_tokens=max_tokens
        )
        return out.choices[0].message.content.strip()
    except Exception as e:
        return f"({title}: AI error: {e})"

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def now_utc(): return dt.datetime.utcnow().replace(microsecond=0).isoformat()+"Z"

def img64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig); buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")

def choose_numcol(df, candidates=("Value","value")):
    for c in candidates:
        if c in df.columns: return c
    return None

def safe_to_year(s):  return pd.to_numeric(s.astype(str).str[:4], errors="coerce")
def to_ym(s):
    ss = s.astype(str)
    y = ss.str[:4]
    m = ss.str[5:7].where(ss.str.len()>=7, "01")
    return pd.PeriodIndex(y+"-"+m, freq="M")

def table_html(df: pd.DataFrame, max_rows=12, cols=None):
    if df is None or df.empty: return "<p><em>(no data)</em></p>"
    out = df.copy()
    if cols:
        keep = [c for c in cols if c in out.columns]
        if keep: out = out[keep]
    return out.fillna("").head(max_rows).to_html(index=False, border=0, escape=False)

def last_val(series):
    try:
        return float(pd.Series(series).dropna().iloc[-1])
    except Exception:
        return None

def yoy_pct(series):
    s = pd.Series(series).dropna()
    if len(s) < 13: return None
    try:
        return 100.0*(s.iloc[-1] / s.iloc[-13] - 1.0)
    except Exception:
        return None

# ------------------------------------------------------------
# Forecasting (ETS with bands; safe fallback)
# ------------------------------------------------------------
def ets_forecast_with_bands(series: pd.Series, periods=12):
    """
    Returns (fc_df, band_df) where:
      fc_df:  date,yhat
      band_df: date,lower,upper  (95% PI if available; else flat ±1σ)
    """
    s = series.dropna().astype(float)
    if len(s) < 6:
        return pd.DataFrame(columns=["date","yhat"]), pd.DataFrame(columns=["date","lower","upper"])

    # try statsmodels ETS
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        seasonal = "add" if len(s) >= 24 else None
        m = 12 if seasonal else None
        model = ExponentialSmoothing(s, trend="add", seasonal=seasonal,
                                     seasonal_periods=m, initialization_method="estimated").fit(optimized=True)
        f = model.forecast(periods)
        # prediction interval (approx via residual std if conf_int not available)
        resid_sd = float(np.nanstd(model.resid, ddof=1)) if hasattr(model, "resid") else float(np.nanstd(s))
        dates = f.index.to_timestamp()
        fc_df = pd.DataFrame({"date": dates, "yhat": f.values})
        band_df = pd.DataFrame({"date": dates, "lower": f.values-1.96*resid_sd, "upper": f.values+1.96*resid_sd})
        return fc_df, band_df
    except Exception:
        # fallback: flat forecast around recent mean ± 1σ
        mean_val = float(pd.Series(s).tail(12).mean()) if len(s)>=12 else float(pd.Series(s).mean())
        std_val  = float(pd.Series(s).tail(12).std(ddof=1)) if len(s)>=12 else float(pd.Series(s).std(ddof=1))
        start = s.index[-1].to_timestamp() if hasattr(s.index[-1], "to_timestamp") else pd.Timestamp(s.index[-1])
        dates = pd.date_range(start=start + pd.offsets.MonthBegin(1), periods=periods, freq="MS")
        fc_df   = pd.DataFrame({"date": dates, "yhat": [mean_val]*periods})
        band_df = pd.DataFrame({"date": dates, "lower": [mean_val-1.96*std_val]*periods,
                                "upper": [mean_val+1.96*std_val]*periods})
        return fc_df, band_df

# ------------------------------------------------------------
# Charts with explanations
# ------------------------------------------------------------
def explain_block(title, lines):
    bullets = "".join(f"<li>{l}</li>" for l in lines if l)
    return f"<div class='explain'><h3>How to read</h3><ul>{bullets}</ul></div>"

def chart_wits_trade(df):
    w = df[df["source"].eq("wits_trade")].copy()
    if w.empty: return "", ""
    v = choose_numcol(w);  w["year"] = safe_to_year(w["time"])
    q = (w["reporter"].str.upper().eq("USA") &
         w["partner"].str.upper().eq("WLD") &
         w["indicator"].isin(["MPRT-TRD-VL","XPRT-TRD-VL"]))
    g = w.loc[q].groupby(["indicator","year"], dropna=True)[v].sum().reset_index()
    if g.empty: return "", ""
    fig, ax = plt.subplots(figsize=(7.5,4.2))
    lines = []
    for ind, sub in g.groupby("indicator"):
        sub = sub.dropna(subset=["year"]).sort_values("year")
        ax.plot(sub["year"], sub[v], marker="o", label=("Imports" if ind=="MPRT-TRD-VL" else "Exports"))
        lv  = last_val(sub[v]); yp = yoy_pct(sub[v])
        lines.append(f"{'Imports' if ind=='MPRT-TRD-VL' else 'Exports'}: last={lv:,.0f}"
                     + (f", YoY={yp:+.1f}%." if yp is not None else "."))
    ax.set_title("WITS — USA ↔ World (Annual Trade Totals)")
    ax.set_xlabel("Year"); ax.set_ylabel("Value"); ax.grid(True, alpha=.3); ax.legend()
    return img64(fig), explain_block("WITS Trade", [
        "Annual totals of US goods trade with the world (imports vs. exports).",
        *lines,
        "Gap between lines indicates trade balance exposure."
    ])

def chart_wits_tariff(df):
    t = df[df["source"].eq("wits_tariff")].copy()
    if t.empty: return "", ""
    v = choose_numcol(t);  t["year"] = safe_to_year(t["time"])
    q = (t["reporter"].str.upper().eq("USA") &
         t["partner"].str.upper().eq("WLD") &
         t["indicator"].isin(["MFN-WGHTD-AVRG","MFN-SMPL-AVRG"]))
    g = t.loc[q].groupby(["indicator","year"], dropna=True)[v].mean().reset_index()
    if g.empty: return "", ""
    fig, ax = plt.subplots(figsize=(7.5,4.2))
    lines = []
    for ind, sub in g.groupby("indicator"):
        sub = sub.dropna(subset=["year"]).sort_values("year")
        ax.plot(sub["year"], sub[v], marker="o", label=("MFN Weighted Avg" if ind=="MFN-WGHTD-AVRG" else "MFN Simple Avg"))
        lv = last_val(sub[v]); yp = yoy_pct(sub[v])
        lines.append(f"{'Weighted' if ind=='MFN-WGHTD-AVRG' else 'Simple'} average tariff: last={lv:.2f}%"
                     + (f", YoY={yp:+.1f}%." if yp is not None else "."))
    ax.set_title("WITS — MFN Average Tariff (USA ↔ World)")
    ax.set_xlabel("Year"); ax.set_ylabel("Percent"); ax.grid(True, alpha=.3); ax.legend()
    return img64(fig), explain_block("WITS Tariffs", [
        "Average MFN tariffs—simple vs. trade-weighted.",
        *lines,
        "Trade-weighted average is more representative of actual exposure."
    ])

def chart_monthly(df, source_name, title):
    m = df[df["source"].eq(source_name)].copy()
    if m.empty: return "", "", pd.DataFrame(), pd.DataFrame()
    vcol = choose_numcol(m, ("ALL_VAL_MO","Value","value"))
    if not vcol: return "", "", pd.DataFrame(), pd.DataFrame()
    m["ym"] = to_ym(m["time"])
    g = m.groupby("ym")[vcol].sum().sort_index()
    if g.empty: return "", "", pd.DataFrame(), pd.DataFrame()

    # history chart
    fig, ax = plt.subplots(figsize=(7.5,4.2))
    ax.plot(g.index.to_timestamp(), g.values, marker="o")
    ax.set_title(title); ax.set_xlabel("Month"); ax.set_ylabel("Value"); ax.grid(True, alpha=.3)
    img = img64(fig)

    # forecast
    fc, band = ets_forecast_with_bands(g)
    fig2, ax2 = plt.subplots(figsize=(7.5,4.2))
    ax2.plot(g.index.to_timestamp(), g.values, label="History")
    if not fc.empty:
        ax2.plot(fc["date"], fc["yhat"], linestyle="--", marker="o", label="Forecast (12m)")
    if not band.empty:
        ax2.fill_between(band["date"], band["lower"], band["upper"], alpha=0.15, label="95% band")
    ax2.set_title(title + " — 12-month Forecast")
    ax2.set_xlabel("Month"); ax2.set_ylabel("Value"); ax2.grid(True, alpha=.3); ax2.legend()
    img_fc = img64(fig2)

    # explanation
    lv = last_val(g.values); yp = yoy_pct(g.values)
    expl = explain_block(title, [
        "Monthly totals (sum across partners).",
        f"Last value={lv:,.0f}" if lv is not None else None,
        (f"Year-over-year change={yp:+.1f}%." if yp is not None else None),
        "Dashed line shows baseline ETS forecast; shaded area is the 95% band (uncertainty)."
    ])
    return img, img_fc, g.to_frame(name="value").reset_index(), fc

def chart_ppi(df):
    b = df[df["source"].eq("bls_ppi")].copy()
    if b.empty: return "", "", pd.DataFrame(), pd.DataFrame()
    v = choose_numcol(b, ("value","Value"))
    if not v: return "", "", pd.DataFrame(), pd.DataFrame()
    try:
        b = b.dropna(subset=["year","period"])[["year","period",v]].copy()
        b["month"] = b["period"].astype(str).str.replace("M","", regex=False).astype(int)
        b["date"] = pd.to_datetime(dict(year=b["year"].astype(int), month=b["month"], day=1))
        s = b.set_index("date")[v].astype(float).sort_index()
    except Exception:
        return "", "", pd.DataFrame(), pd.DataFrame()
    if s.empty: return "", "", pd.DataFrame(), pd.DataFrame()

    # history
    fig, ax = plt.subplots(figsize=(7.5,4.2))
    ax.plot(s.index, s.values, marker="o"); ax.set_title("BLS PPI — Producer Price Index")
    ax.set_xlabel("Month"); ax.set_ylabel("Index"); ax.grid(True, alpha=.3)
    img = img64(fig)

    # forecast
    fc, band = ets_forecast_with_bands(s)
    fig2, ax2 = plt.subplots(figsize=(7.5,4.2))
    ax2.plot(s.index, s.values, label="History")
    if not fc.empty: ax2.plot(fc["date"], fc["yhat"], linestyle="--", marker="o", label="Forecast (12m)")
    if not band.empty: ax2.fill_between(band["date"], band["lower"], band["upper"], alpha=0.15, label="95% band")
    ax2.set_title("BLS PPI — 12-month Forecast")
    ax2.set_xlabel("Month"); ax2.set_ylabel("Index"); ax2.grid(True, alpha=.3); ax2.legend()
    img_fc = img64(fig2)

    lv = last_val(s.values); yp = yoy_pct(s.values)
    expl = explain_block("BLS PPI", [
        "Upstream producer prices (useful cost-pressure leading indicator).",
        f"Last index={lv:.1f}" if lv is not None else None,
        (f"Year-over-year change={yp:+.1f}%." if yp is not None else None),
        "Dashed line shows ETS forecast; band is 95% interval."
    ])
    return img, img_fc, s.reset_index().rename(columns={v:"value"}), fc

# ------------------------------------------------------------
# Partner deep dive (Top-5, YoY, HHI concentration)
# ------------------------------------------------------------
def partner_deep_dive(df, source_name, title, top=5):
    m = df[df["source"].eq(source_name)].copy()
    if m.empty: return "", "<p><em>(no data)</em></p>"
    vcol = choose_numcol(m, ("ALL_VAL_MO","Value","value"))
    if not vcol: return "", "<p><em>(no data)</em></p>"
    m["ym"] = to_ym(m["time"])
    # pick partners by last 12m total
    last12 = m[m["ym"] >= (m["ym"].max() - 11)]
    top_partners = (last12.groupby("partner")[vcol].sum()
                    .sort_values(ascending=False).head(top).index.tolist())
    mm = m[m["partner"].isin(top_partners)]
    g = mm.groupby(["partner","ym"])[vcol].sum().reset_index()
    # chart (lines)
    fig, ax = plt.subplots(figsize=(7.8,4.2))
    for p, sub in g.groupby("partner"):
        sub = sub.sort_values("ym")
        ax.plot(sub["ym"].dt.to_timestamp(), sub[vcol], marker="o", label=str(p))
    ax.set_title(title + " — Top Partners (last 12m leaders)")
    ax.set_xlabel("Month"); ax.set_ylabel("Value"); ax.grid(True, alpha=.3); ax.legend(ncol=2, fontsize=9)
    img = img64(fig)
    # YoY table for last month
    latest = m["ym"].max()
    this = m[m["ym"].eq(latest)].groupby("partner")[vcol].sum()
    prev = m[m["ym"].eq(latest - 12)].groupby("partner")[vcol].sum() if (latest - 12) in m["ym"].unique() else pd.Series(dtype=float)
    yoy = pd.DataFrame({"value": this}).join(prev.rename("prev"), how="left")
    yoy["YoY%"] = 100.0*(yoy["value"]/yoy["prev"] - 1.0)
    yoy = yoy.sort_values("value", ascending=False).reset_index().rename(columns={"partner":"Partner","value":"Latest"})
    # HHI concentration over last 12m
    shares = last12.groupby("partner")[vcol].sum()
    if shares.sum() > 0:
        share = (shares / shares.sum())
        hhi = float((share.pow(2).sum())*10000)  # 0–10000
        hhi_note = f"Concentration (HHI, last 12m) = {hhi:,.0f} (≥2500 = high concentration)."
    else:
        hhi_note = "Concentration (HHI) unavailable."
    html = (
        "<p><small>"+hhi_note+"</small></p>" +
        table_html(yoy, cols=["Partner","Latest","prev","YoY%"])
    )
    explain = explain_block(title, [
        "Lines show monthly totals for the top partners (ranked by last 12 months).",
        "Table shows latest month, prior-year month, and YoY%.",
        hhi_note
    ])
    return img, explain + html

# ------------------------------------------------------------
# HTML skeleton
# ------------------------------------------------------------
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
.explain h3 { margin: 0.2em 0 0.4em 0; font-size: 15px; color:#374151; }
.explain ul { margin: 0 0 0 1.2em; }
</style>
"""
def section(title, body): return f"<div class='section'><h2>{title}</h2>{body}</div>"

# ------------------------------------------------------------
# Build report
# ------------------------------------------------------------
def build_report():
    pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    if not os.path.exists(CSV):
        html = f"""<html><body>
        <h1>Service Supply Chain — Auto Report</h1>
        <p><small>Generated: <b>{now_utc()}</b></small></p>
        <p>No <code>main.csv</code> found. Run <b>Update SSC Data</b> first.</p>
        </body></html>"""
        open(os.path.join(OUT_DIR, "Auto_Report.html"), "w", encoding="utf-8").write(html)
        print("[report] main.csv missing; wrote minimal HTML."); return

    df = pd.read_csv(CSV)
    pulled = ""
    if "pulled_at_utc" in df.columns and df["pulled_at_utc"].notna().any():
        pulled = str(df["pulled_at_utc"].dropna().astype(str).max())

    # Overview by source
    if "source" in df.columns:
        by_src = df["source"].value_counts().rename_axis("source").reset_index(name="rows")
    else:
        by_src = pd.DataFrame(columns=["source","rows"])

    # KPI cards
    kpi_html = f"""
    <div class="kpi">
      <div class="card"><small>Last pull</small><b>{pulled or 'N/A'}</b></div>
      <div class="card"><small>Total rows</small><b>{len(df):,}</b></div>
      <div class="card"><small>Sources</small><b>{len(by_src):,}</b></div>
    </div>
    """

    # Charts + explanations
    img_trade, expl_trade   = chart_wits_trade(df)
    img_tariff, expl_tariff = chart_wits_tariff(df)

    img_cimp, img_cimp_fc, cimp_hist, cimp_fc = chart_monthly(df, "census_imports", "Census — Total Imports (Monthly)")
    img_cexp, img_cexp_fc, cexp_hist, cexp_fc = chart_monthly(df, "census_exports", "Census — Total Exports (Monthly)")
    img_ppi,  img_ppi_fc,  ppi_hist,  ppi_fc  = chart_ppi(df)

    # Partner deep dives
    pd_imp_img, pd_imp_block = partner_deep_dive(df, "census_imports", "Partner Deep-Dive — Imports")
    pd_exp_img, pd_exp_block = partner_deep_dive(df, "census_exports", "Partner Deep-Dive — Exports")

    # AI sections
    years = sorted(pd.to_numeric(df.get("time", pd.Series(dtype=str)).astype(str).str[:4], errors="coerce").dropna().unique().tolist())
    ai_exec = gpt_brief("Executive Summary", f"""
Write an **executive summary in English** (bullets + short paragraphs) for a public audience.
Use only the dataset profile below (do not invent specific numbers).
DATA PROFILE:
- Sources & rows: {dict(by_src.values) if not by_src.empty else {}}
- Years present (from 'time'): {years}
- We chart: WITS annual imports/exports and MFN tariffs; Census monthly imports/exports; BLS PPI; 12-month ETS forecasts.
TASKS:
1) Current situation across trade, tariffs, and producer prices.
2) What to expect next 12 months (directionality; mention uncertainty).
3) 3–5 risks/opportunities for US **service** supply chains (lead times, tariff exposure, concentration).
4) Close with 3 actionable steps.
Tone: crisp, non-technical, executive-ready.
""")

    # Geopolitical/Policy context (neutral, high-level)
    ai_geo = gpt_brief("Geopolitical Context", """
Provide a neutral, high-level geopolitical & policy context that impacts US trade and service supply chains.
Cover: US–China tech/trade frictions, USMCA dynamics (Canada/Mexico), EU regulatory alignment/divergence,
semiconductor/EV/clean-tech industrial policy, sanctions risk, Red Sea/Suez & Panama Canal logistics,
FX & rate environment, election/leadership cycles that can change tariff or export-control regimes.
Keep it short (5–8 bullets) and public-facing (no jargon).
""", max_tokens=600)

    # Build page
    STYLE_BLOCK = STYLE
    html = f"""<!doctype html>
<html lang="en"><meta charset="utf-8"/>
<title>Service Supply Chain — Auto Report</title>
{STYLE_BLOCK}
<body>
<h1>Service Supply Chain — Auto Report</h1>
<p><small>Generated: <b>{now_utc()}</b>{' · Last pull: <b>'+pulled+'</b>' if pulled else ''}</small></p>

<div class="section"><h2>Executive Summary (AI)</h2><pre>{ai_exec}</pre></div>

{section("Overview", kpi_html + table_html(by_src, max_rows=20))}

{section("Trade — WITS (USA ↔ World)", (('<img src="'+img_trade+'"/>') if img_trade else '<p><em>No WITS trade data.</em></p>') + expl_trade)}

{section("Tariffs — WITS MFN Averages", (('<img src="'+img_tariff+'"/>') if img_tariff else '<p><em>No tariff data.</em></p>') + expl_tariff)}

{section("Census Monthly Imports", (('<img src="'+img_cimp+'"/>') if img_cimp else '<p><em>No monthly imports.</em></p>'))}
{section("Census Monthly Imports — 12-month Forecast", (('<img src="'+img_cimp_fc+'"/>') if img_cimp_fc else '<p><em>No forecast available.</em></p>'))}

{section("Census Monthly Exports", (('<img src="'+img_cexp+'"/>') if img_cexp else '<p><em>No monthly exports.</em></p>'))}
{section("Census Monthly Exports — 12-month Forecast", (('<img src="'+img_cexp_fc+'"/>') if img_cexp_fc else '<p><em>No forecast available.</em></p>'))}

{section("BLS Producer Price Index", (('<img src="'+img_ppi+'"/>') if img_ppi else '<p><em>No PPI series.</em></p>'))}
{section("BLS Producer Price Index — 12-month Forecast", (('<img src="'+img_ppi_fc+'"/>') if img_ppi_fc else '<p><em>No forecast available.</em></p>'))}

{section("Geopolitical Context & Policy Watch (AI)", "<pre>"+ai_geo+"</pre>")}

{section("Partner Deep-Dive — Imports", (('<img src="'+pd_imp_img+'"/>') if pd_imp_img else '') + pd_imp_block)}
{section("Partner Deep-Dive — Exports", (('<img src="'+pd_exp_img+'"/>') if pd_exp_img else '') + pd_exp_block)}

</body></html>"""

    out = os.path.join(OUT_DIR, "Auto_Report.html")
    open(out, "w", encoding="utf-8").write(html)
    print("[report] Wrote:", out)

if __name__ == "__main__":
    build_report()

