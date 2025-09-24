# ssc_pipeline/build_report.py
# Executive trade dashboard with AI narratives, multi-horizon forecasts,
# base64-embedded images (works on GitHub Pages), and named-country deep dives.

import os, io, json, base64
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ----------------------------
# Paths / environment
# ----------------------------
ROOT = Path(os.environ.get("SSC_ROOT", ".")).resolve()
OUT_DIR = ROOT / "ssc_pipeline"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = ROOT / "main.csv"
HTML_PATH = OUT_DIR / "Auto_Report.html"

# AI layer toggles
USE_AI = os.getenv("SSC_USE_AI", "0") == "1"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# WITS conversion (US$ Mil → USD)
WITS_TO_USD = 1_000_000

# Named partners for deep dive
FOCUS_PARTNERS = ["Mexico","Canada","China","Japan","Germany","United Kingdom"]

# ----------------------------
# Formatting helpers
# ----------------------------
def human_usd(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    n = float(x); sgn = "-" if n < 0 else ""; n = abs(n)
    if n >= 1e12: return f"{sgn}{n/1e12:.2f}T"
    if n >= 1e9:  return f"{sgn}{n/1e9:.2f}B"
    if n >= 1e6:  return f"{sgn}{n/1e6:.2f}M"
    return f"{sgn}{n:,.0f}"

def pct(x: Optional[float]) -> str:
    return "—" if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))) else f"{x:.1f}%"

def H(n: int, txt: str) -> str:
    return f"<h{n}>{txt}</h{n}>"

def small(txt: str) -> str:
    return f"<div style='color:#666;font-size:0.9em'>{txt}</div>"

def bullet_list(items: List[str]) -> str:
    return "<ul>" + "".join([f"<li>{it}</li>" for it in items]) + "</ul>"

def embed_png(fig_path: Path) -> str:
    """Embed a saved PNG as <img src='data:image/png;base64,...'> to avoid broken paths on Pages."""
    with open(fig_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"<img src='data:image/png;base64,{b64}' />"

# ----------------------------
# Load & normalize main.csv
# ----------------------------
def load_main() -> pd.DataFrame:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"main.csv not found at {CSV_PATH} — run update-data first.")
    df = pd.read_csv(CSV_PATH, low_memory=False)
    if "Value" in df.columns and "value" not in df.columns:
        df = df.rename(columns={"Value": "value"})
    return df

def prepare_wits(df: pd.DataFrame) -> pd.DataFrame:
    w = df[df["source"].isin(["wits_trade","wits_tariff"])].copy()
    if w.empty: return w
    # annual year
    w["year"] = pd.to_numeric(w.get("time", np.nan), errors="coerce").astype("Int64")
    # convert trade values US$ Mil → USD
    is_trd = w["indicator"].astype(str).str.contains("TRD-VL", case=False, na=False)
    w.loc[is_trd, "value_usd"] = pd.to_numeric(w["value"], errors="coerce") * WITS_TO_USD
    return w

def prepare_census(df: pd.DataFrame) -> pd.DataFrame:
    c = df[df["source"].isin(["census_imports","census_exports"])].copy()
    if c.empty: return c

    name_col = "CTY_NAME" if "CTY_NAME" in c.columns else ("partner_name" if "partner_name" in c.columns else ("partner" if "partner" in c.columns else None))
    code_col = "CTY_CODE" if "CTY_CODE" in c.columns else ("partner_code" if "partner_code" in c.columns else ("partner" if "partner" in c.columns else None))
    if name_col is None: name_col = "partner"
    if code_col is None: code_col = "partner"

    upnames = c[name_col].astype(str).str.upper()
    bad = upnames.str.contains("TOTAL") | upnames.str.contains("ALL COUNTRIES") | upnames.eq("WLD")
    bad |= c[code_col].astype(str).str.upper().isin(["ALL","WLD","000","TOT","0"])
    c = c.loc[~bad].copy()

    if "MONTH" in c.columns and "YEAR" in c.columns:
        c["month"] = pd.to_numeric(c["MONTH"], errors="coerce")
        c["year"]  = pd.to_numeric(c["YEAR"], errors="coerce")
    elif "period" in c.columns and "year" in c.columns:
        c["month"] = c["period"].astype(str).str.extract(r"M(\d{2})").astype(float)
        c["year"]  = pd.to_numeric(c["year"], errors="coerce")
    else:
        raise ValueError("Census rows missing YEAR/MONTH or year/period columns.")

    c["date"] = pd.PeriodIndex(year=c["year"].astype(int), month=c["month"].astype(int), freq="M").to_timestamp("M")

    val_col = "ALL_VAL_MO" if "ALL_VAL_MO" in c.columns else "value"
    c["value_usd"] = pd.to_numeric(c[val_col], errors="coerce")
    c["partner_label"] = c[name_col].astype(str)
    return c

def prepare_bls(df: pd.DataFrame) -> pd.DataFrame:
    b = df[df["source"].eq("bls_ppi")].copy()
    if b.empty: return b
    b["month"] = b["period"].astype(str).str.extract(r"M(\d{2})").astype(float)
    b["year"]  = pd.to_numeric(b["year"], errors="coerce")
    b["date"]  = pd.PeriodIndex(year=b["year"].astype(int), month=b["month"].astype(int), freq="M").to_timestamp("M")
    b["value"] = pd.to_numeric(b["value"], errors="coerce")
    return b.dropna(subset=["date","value"]).sort_values("date")

# ----------------------------
# Time-series helpers
# ----------------------------
def to_monthly_unique(series: pd.Series, how: str="sum") -> pd.Series:
    s = series.copy()
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index, errors="coerce")
    s = s.dropna().sort_index()
    if s.index.has_duplicates:
        s = s.groupby(s.index).sum() if how=="sum" else s.groupby(s.index).mean()
    s = s.resample("M").sum() if how=="sum" else s.resample("M").mean()
    return s.dropna()

# ----------------------------
# Metrics & Forecasts
# ----------------------------
def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred)
    ok = denom > 1e-12
    out = np.full_like(y_true, np.nan, dtype=float)
    out[ok] = 200.0 * np.abs(y_true[ok] - y_pred[ok]) / denom[ok]
    return float(np.nanmean(out))

def wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.sum(np.abs(y_true)))
    if denom < 1e-12: return float("nan")
    return 100.0 * float(np.sum(np.abs(y_true - y_pred))) / denom

def coverage(actual: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    hit = (actual >= lo) & (actual <= hi)
    return 100.0 * float(np.mean(hit))

def _bands_from_resid(resid: np.ndarray, fallback_sd: float):
    if resid.size >= 8:
        q = np.nanpercentile(resid, [2.5, 97.5])
        if np.ndim(q) == 0:
            sd = float(np.nanstd(resid, ddof=1)); return -1.96*sd, 1.96*sd
        return float(q[0]), float(q[1])
    sd = float(np.nanstd(resid, ddof=1)) if resid.size>1 else float(fallback_sd)
    return -1.96*sd, 1.96*sd

def ets_forecast(ts: pd.Series, h: int=24, seasonal: bool=True):
    """Robust ETS with graceful fallbacks; returns 24m horizon for short/mid/long views."""
    # clean
    s = ts.copy()
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index, errors="coerce")
    s = s.dropna().sort_index()
    if s.index.has_duplicates:
        s = s.groupby(s.index).mean()
    s = s.asfreq("M")
    s = s.dropna()
    if len(s) == 0:
        idx_f = pd.date_range(pd.Timestamp.today().normalize() + pd.offsets.MonthEnd(1), periods=h, freq="M")
        f = pd.Series([np.nan]*h, index=idx_f)
        return f, f, f, {"sMAPE": np.nan, "WMAPE": np.nan, "RMSE": np.nan, "Coverage@95": np.nan}

    use_season = seasonal and len(s) >= 36
    fitted = None
    # progressive fallback
    for tr, sez in [("add", use_season), ("add", False), (None, False)]:
        try:
            m = ExponentialSmoothing(s, trend=tr, seasonal=("add" if sez else None), seasonal_periods=(12 if sez else None))
            fitted = m.fit(optimized=True, use_brute=True)
            break
        except Exception:
            continue

    if fitted is None:
        # naive
        last = float(s.iloc[-1])
        idx_f = pd.date_range(s.index[-1] + pd.offsets.MonthEnd(1), periods=h, freq="M")
        f = pd.Series(last, index=idx_f)
        sd = float(np.nanstd(s.values, ddof=1)) if len(s.values)>1 else 0.0
        lo, hi = f - 1.96*sd, f + 1.96*sd
        return f, lo, hi, {"sMAPE": np.nan, "WMAPE": np.nan, "RMSE": np.nan, "Coverage@95": np.nan}

    f_vals = fitted.forecast(h)
    idx_f = pd.date_range(s.index[-1] + pd.offsets.MonthEnd(1), periods=h, freq="M")
    f = pd.Series(f_vals.values, index=idx_f)

    resid = (s - fitted.fittedvalues.reindex(s.index)).dropna().values
    qlo, qhi = _bands_from_resid(resid, fallback_sd=np.nanstd(s.values, ddof=1) if len(s.values)>1 else 0.0)
    lo = f + qlo; hi = f + qhi

    # Backtest 12m if enough points
    scores = {"sMAPE": np.nan, "WMAPE": np.nan, "RMSE": np.nan, "Coverage@95": np.nan}
    if len(s) >= 36:
        train, test = s.iloc[:-12], s.iloc[-12:]
        fitted_bt = None
        for tr, sez in [("add", (seasonal and len(train)>=36)), ("add", False), (None, False)]:
            try:
                m2 = ExponentialSmoothing(train, trend=tr, seasonal=("add" if sez else None),
                                          seasonal_periods=(12 if sez else None))
                fitted_bt = m2.fit(optimized=True, use_brute=True)
                break
            except Exception:
                continue
        if fitted_bt is not None:
            fc_bt = fitted_bt.forecast(12)
            resid_bt = (train - fitted_bt.fittedvalues.reindex(train.index)).dropna().values
            qlo_bt, qhi_bt = _bands_from_resid(resid_bt, fallback_sd=np.nanstd(train.values, ddof=1))
            lo_bt, hi_bt = fc_bt + qlo_bt, fc_bt + qhi_bt
            scores = {
                "sMAPE": smape(test.values, fc_bt.values),
                "WMAPE": wmape(test.values, fc_bt.values),
                "RMSE": float(np.sqrt(np.nanmean((test.values - fc_bt.values)**2))),
                "Coverage@95": coverage(test.values, lo_bt.values, hi_bt.values),
            }
    return f, lo, hi, scores

# ----------------------------
# Plot & embed helpers
# ----------------------------
def _save_and_embed(fig_name: str) -> str:
    path = OUT_DIR / fig_name
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return embed_png(path)

def plot_series(ts: pd.Series, title: str, y_label: str, fig_name: str) -> str:
    plt.figure(figsize=(8.6, 4.4))
    plt.plot(ts.index, ts.values, marker="o")
    plt.title(title); plt.xlabel("Date"); plt.ylabel(y_label)
    return _save_and_embed(fig_name)

def plot_forecast(ts: pd.Series, f: pd.Series, lo: pd.Series, hi: pd.Series,
                  title: str, y_label: str, fig_name: str, show_horizons: Tuple[int,int,int]=(3,12,24)) -> str:
    plt.figure(figsize=(8.6, 4.4))
    plt.plot(ts.index, ts.values, label="History")
    idx_f = f.index
    plt.plot(idx_f, f.values, linestyle="--", marker="o", label="Forecast")
    plt.fill_between(idx_f, lo.values, hi.values, alpha=0.18, label="≈95% band")
    # horizon markers
    for h in show_horizons:
        if len(idx_f) >= h:
            plt.axvline(idx_f[h-1], color="gray", linewidth=1, linestyle=":")
    plt.title(title); plt.xlabel("Date"); plt.ylabel(y_label); plt.legend()
    return _save_and_embed(fig_name)

# ----------------------------
# Sanity & explanations
# ----------------------------
def sanity_panel(df: pd.DataFrame, value_col: str, date_col: str, unit_label: str) -> str:
    dfx = df.dropna(subset=[date_col]).sort_values(date_col)
    last = dfx[[date_col, value_col]].dropna().tail(1)
    last_dt = last[date_col].iloc[0] if not last.empty else None
    nn = dfx[value_col].notna().mean()*100.0 if len(dfx) else 0.0
    zeros = (dfx[value_col].fillna(0)==0).mean()*100.0 if len(dfx) else 0.0
    last_str = last_dt.date().isoformat() if hasattr(last_dt, "date") else str(last_dt)
    return small(f"Sanity ✓ — unit: {unit_label} · last: {last_str} · non-null: {nn:.1f}% · zeros: {zeros:.1f}% · rows: {len(dfx):,}")

def static_explain(what: str, how: str, forecast_note: str = "", extra: str = "") -> str:
    parts = [f"<b>What this shows:</b> {what}", f"<b>How to read:</b> {how}"]
    if forecast_note: parts.append(f"<b>Forecast:</b> {forecast_note}")
    if extra: parts.append(f"<b>Context:</b> {extra}")
    return "<div style='margin:10px 0 14px 0'>" + "<br/>".join(parts) + "</div>"

# ----------------------------
# AI explainer (optional)
# ----------------------------
def _ai_chat(system_prompt: str, user_prompt: str) -> str:
    if not (USE_AI and OPENAI_API_KEY): return ""
    try:
        import requests
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": OPENAI_MODEL,
            "temperature": 0.25,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=45)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return ""

def _summarize_series(s: Optional[pd.Series], k: int=24) -> Dict[str,Any]:
    if s is None or s.empty: return {}
    s2 = s.dropna()
    yoy = None
    try:
        if len(s2) >= 13:
            yoy = 100.0*(s2.iloc[-1]-s2.iloc[-13]) / max(1e-9, abs(s2.iloc[-13]))
    except Exception:
        pass
    tail = s2.tail(k)
    return {
        "last_date": str(s2.index[-1].date()) if hasattr(s2.index[-1],"date") else str(s2.index[-1]),
        "last_value": float(s2.iloc[-1]),
        "min_lastK": float(np.nanmin(tail.values)) if len(tail) else None,
        "max_lastK": float(np.nanmax(tail.values)) if len(tail) else None,
        "yoy_pct": float(yoy) if yoy is not None else None,
        "count": int(len(s2)),
    }

def ai_explain(section_title: str, freq_unit: str,
               history: Optional[pd.Series], forecast: Optional[Tuple[pd.Series,pd.Series,pd.Series,dict]],
               extras: Optional[Dict[str,Any]]=None) -> str:
    if not (USE_AI and OPENAI_API_KEY): return ""
    hist = _summarize_series(history)
    fc_stats = {}
    if forecast is not None:
        f, lo, hi, scores = forecast
        fc_stats = {
            "h": len(f),
            "short_3m": float(f.iloc[min(2, len(f)-1)]) if len(f) else None,
            "mid_12m": float(f.iloc[min(11, len(f)-1)]) if len(f) > 11 else None,
            "long_24m": float(f.iloc[min(23, len(f)-1)]) if len(f) > 23 else None,
            "band_last": [float(lo.iloc[-1]), float(hi.iloc[-1])] if len(f) else None,
            "scores": {k: (None if (isinstance(v,float) and (np.isnan(v) or np.isinf(v))) else float(v)) for k,v in (scores or {}).items()}
        }
    sys_msg = (
        "You are an economist writing for busy executives. "
        "Provide 6–9 sentences. Start with what the chart measures and why it matters. "
        "Explain frequency and units plainly, define acronyms (PPI, ETS). "
        "State current level and YoY, then short (3m), mid (12m), long (24m) outlook from the forecast. "
        "Interpret the uncertainty band and Coverage@95 calibration (too narrow/wide). "
        "Add light geopolitical/business context (tariffs, sanctions, elections, energy, transport shocks) without speculation. "
        "End with one actionable takeaway."
    )
    user_msg = (
        f"Section: {section_title}\n"
        f"Freq/Units: {freq_unit}\n"
        f"History: {json.dumps(hist)}\n"
        f"Forecast: {json.dumps(fc_stats)}\n"
        f"Extras: {json.dumps(extras or {})}\n"
        "Write in clear business English."
    )
    out = _ai_chat(sys_msg, user_msg)
    return f"<div style='margin:10px 0 14px 0'>{out}</div>" if out else ""

# ----------------------------
# Partner deep-dives
# ----------------------------
def partner_top_table(c: pd.DataFrame, source_key: str, top_n: int=8):
    df = c[c["source"].eq(source_key)].dropna(subset=["date","value_usd","partner_label"]).copy()
    if df.empty: return "", 0, pd.DataFrame()
    recent = df[df["date"] > (df["date"].max() - pd.DateOffset(months=12))]
    grp = recent.groupby("partner_label", as_index=False)["value_usd"].sum().sort_values("value_usd", ascending=False)
    total = grp["value_usd"].sum()
    grp["share"] = grp["value_usd"] / max(total, 1e-9)
    hhi = int(round((grp["share"]**2).sum() * 10_000))
    top = grp.head(top_n)
    # bar
    plt.figure(figsize=(8.6, 4.4))
    plt.bar(top["partner_label"], top["value_usd"])
    plt.title(f"Top {top_n} Partners (Last 12m) — {'Imports' if 'import' in source_key else 'Exports'}")
    plt.ylabel("USD"); plt.xticks(rotation=30, ha="right")
    img = _save_and_embed(f"{source_key}_partners_top.png")
    # table
    tbl = "<table border='1' cellpadding='4' cellspacing='0'><tr><th>Partner</th><th>Last 12m (USD)</th><th>Share</th></tr>"
    for _, r in top.iterrows():
        tbl += f"<tr><td>{r['partner_label']}</td><td>{human_usd(r['value_usd'])}</td><td>{pct(r['share']*100)}</td></tr>"
    tbl += "</table>"
    return img + "<br/>" + tbl, hhi, grp

def named_countries_block(c: pd.DataFrame, source_key: str, title: str):
    df = c[c["source"].eq(source_key)].dropna(subset=["date","value_usd","partner_label"]).copy()
    if df.empty: return ""
    recent = df[df["date"] > (df["date"].max() - pd.DateOffset(months=12))]
    rows = []
    for nm in FOCUS_PARTNERS:
        sub = recent[recent["partner_label"].str.strip().str.casefold()==nm.casefold()]
        val = float(sub["value_usd"].sum()) if not sub.empty else np.nan
        rows.append((nm, val))
    tbl = "<table border='1' cellpadding='4' cellspacing='0'><tr><th>Country</th><th>Last 12m (USD)</th></tr>"
    for nm, val in rows:
        tbl += f"<tr><td>{nm}</td><td>{human_usd(val)}</td></tr>"
    tbl += "</table>"
    return H(4, title) + tbl

# ----------------------------
# Build report
# ----------------------------
def build_report():
    df   = load_main()
    wits = prepare_wits(df)
    cen  = prepare_census(df)
    bls  = prepare_bls(df)

    html = []
    html.append("<html><head><meta charset='utf-8'><title>SSC Report</title>")
    html.append("<style>body{font-family:system-ui,Segoe UI,Roboto,Arial,sans-serif;max-width:1120px;margin:24px auto;padding:0 16px;line-height:1.45} img{max-width:100%}</style>")
    html.append("</head><body>")
    html.append(H(1, "U.S. Trade & Supply Chain — Executive Dashboard"))

    # ---------- WITS (Annual) ----------
    if not wits.empty:
        wt = wits[wits["indicator"].isin(["MPRT-TRD-VL","XPRT-TRD-VL"])].copy()
        wt = wt[(wt["partner"].astype(str).str.upper()=="WLD") & (wt["product"].astype(str).str.lower()=="total")]
        wt["kind"] = np.where(wt["indicator"].eq("MPRT-TRD-VL"), "Imports", "Exports")
        g = wt.groupby(["year","kind"], as_index=False)["value_usd"].sum().dropna()
        if not g.empty:
            wide = g.pivot(index="year", columns="kind", values="value_usd").sort_index()
            html.append(H(2, "WITS — Annual Totals (USD)"))
            html.append(small("Frequency: Annual · Unit: USD (converted from 'US$ Mil') · Coverage: Goods (nominal)"))

            for col in wide.columns:
                ts = wide[col].dropna()
                if ts.empty: continue
                ts2 = pd.Series(ts.values, index=pd.to_datetime(ts.index.astype(str) + "-12-31"))
                img = plot_series(ts2, f"WITS — {col} (Annual, USD)", "USD", f"wits_{col.lower()}.png")
                html.append(img)
                html.append(static_explain(
                    what=f"U.S. {col.lower()} of goods by calendar year (WITS).",
                    how="Each point is a full-year total in U.S. dollars; not monthly.",
                    extra="Policy shifts (tariffs, FTAs), sanctions, energy and logistics shocks can move annual totals."
                ))
                html.append(ai_explain(
                    section_title=f"WITS — {col} (Annual)",
                    freq_unit="Annual, USD",
                    history=ts2, forecast=None,
                    extras={"source":"WITS (converted from US$ Mil)"}
                ))
            # last valid bullets
            if not wide.dropna(how="all").empty:
                last_year = wide.dropna(how="all").index.max()
                bullets = []
                for col in wide.columns:
                    val = wide.loc[last_year, col]
                    if pd.notna(val):
                        bullets.append(f"{col} {int(last_year)}: {human_usd(val)}")
                if bullets:
                    html.append(bullet_list(bullets))

    # ---------- Census: Imports & Exports ----------
    if not cen.empty:
        imp_raw = cen[cen["source"].eq("census_imports")].set_index("date")["value_usd"]
        exp_raw = cen[cen["source"].eq("census_exports")].set_index("date")["value_usd"]
        imports_m = to_monthly_unique(imp_raw, how="sum")
        exports_m = to_monthly_unique(exp_raw, how="sum")

        # Imports
        if not imports_m.empty:
            html.append(H(2, "Census — Goods Imports (Monthly, USD)"))
            html.append(sanity_panel(cen[cen["source"].eq("census_imports")], "value_usd","date","USD"))
            html.append(plot_series(imports_m, "U.S. Goods Imports — Monthly (USD)", "USD", "census_imports_hist.png"))
            f, lo, hi, scores = ets_forecast(imports_m, h=24, seasonal=True)
            html.append(plot_forecast(imports_m, f, lo, hi, "Imports — ETS Forecast (3/12/24 months)", "USD", "census_imports_fc.png"))
            html.append(bullet_list([
                f"Latest: {human_usd(imports_m.iloc[-1])}",
                f"sMAPE (12m OOS): {pct(scores['sMAPE'])}",
                f"WMAPE: {pct(scores['WMAPE'])}",
                f"RMSE: {human_usd(scores['RMSE'])}",
                f"Coverage@95: {pct(scores['Coverage@95'])}",
            ]))
            html.append(static_explain(
                what="Monthly total dollar value of U.S. goods imports across all partners.",
                how="Up = more imports in nominal USD. Not adjusted for inflation.",
                forecast_note="Bands show ≈95% empirical uncertainty. Vertical dotted lines mark 3m/12m/24m horizons.",
                extra="Risks: tariff policy (e.g., 301 actions), shipping routes (Red Sea), energy prices, and election-year uncertainty."
            ))
            html.append(ai_explain(
                section_title="Census — U.S. Goods Imports (Monthly)",
                freq_unit="Monthly, USD",
                history=imports_m,
                forecast=(f, lo, hi, scores),
                extras={"source":"U.S. Census HS timeseries"}
            ))

            # Imports partners (top + named)
            img, hhi_imp, grp_imp = partner_top_table(cen, "census_imports", top_n=8)
            html.append(H(3, "Imports — Partner Concentration"))
            html.append(img)
            html.append(small(f"HHI: {hhi_imp} (≈1,500–2,500 = moderate; >2,500 = high)"))
            html.append(named_countries_block(cen, "census_imports", "Named Partners — Imports (Last 12 months)"))
            html.append(ai_explain(
                section_title="Imports — Partner Deep-Dive",
                freq_unit="Aggregated last 12 months, USD",
                history=None, forecast=None,
                extras={"hhi":hhi_imp, "top_partners": grp_imp.head(5).to_dict(orient="records")}
            ))

        # Exports
        if not exports_m.empty:
            html.append(H(2, "Census — Goods Exports (Monthly, USD)"))
            html.append(sanity_panel(cen[cen["source"].eq("census_exports")], "value_usd","date","USD"))
            html.append(plot_series(exports_m, "U.S. Goods Exports — Monthly (USD)", "USD", "census_exports_hist.png"))
            f, lo, hi, scores = ets_forecast(exports_m, h=24, seasonal=True)
            html.append(plot_forecast(exports_m, f, lo, hi, "Exports — ETS Forecast (3/12/24 months)", "USD", "census_exports_fc.png"))
            html.append(bullet_list([
                f"Latest: {human_usd(exports_m.iloc[-1])}",
                f"sMAPE (12m OOS): {pct(scores['sMAPE'])}",
                f"WMAPE: {pct(scores['WMAPE'])}",
                f"RMSE: {human_usd(scores['RMSE'])}",
                f"Coverage@95: {pct(scores['Coverage@95'])}",
            ]))
            html.append(static_explain(
                what="Monthly dollar value of U.S. goods exports across all partners.",
                how="Up = more exports in nominal USD.",
                forecast_note="ETS 3/12/24-month outlook with empirical ≈95% bands; Coverage@95 indicates calibration.",
                extra="Context: global growth, partner demand (Mexico, Canada, EU/UK, Japan) and export controls/sanctions shape the outlook."
            ))
            html.append(ai_explain(
                section_title="Census — U.S. Goods Exports (Monthly)",
                freq_unit="Monthly, USD",
                history=exports_m,
                forecast=(f, lo, hi, scores),
                extras={"source":"U.S. Census HS timeseries"}
            ))

            img, hhi_exp, grp_exp = partner_top_table(cen, "census_exports", top_n=8)
            html.append(H(3, "Exports — Partner Concentration"))
            html.append(img)
            html.append(small(f"HHI: {hhi_exp} (≈1,500–2,500 = moderate; >2,500 = high)"))
            html.append(named_countries_block(cen, "census_exports", "Named Partners — Exports (Last 12 months)"))
            html.append(ai_explain(
                section_title="Exports — Partner Deep-Dive",
                freq_unit="Aggregated last 12 months, USD",
                history=None, forecast=None,
                extras={"hhi":hhi_exp, "top_partners": grp_exp.head(5).to_dict(orient="records")}
            ))

    # ---------- BLS PPI ----------
    if not bls.empty:
        for sid, sub in bls.groupby("series_id"):
            raw = sub.set_index("date")["value"]
            ts  = to_monthly_unique(raw, how="mean")
            if ts.empty: continue
            html.append(H(2, f"BLS PPI — {sid} (Monthly Index)"))
            html.append(sanity_panel(sub, "value","date","Index"))
            html.append(plot_series(ts, f"PPI {sid} — Monthly (Index)", "Index", f"bls_{sid}_hist.png"))
            f, lo, hi, scores = ets_forecast(ts, h=24, seasonal=True)
            html.append(plot_forecast(ts, f, lo, hi, f"PPI {sid} — ETS Forecast (3/12/24 months)", "Index", f"bls_{sid}_fc.png"))
            html.append(bullet_list([
                f"sMAPE (12m OOS): {pct(scores['sMAPE'])}",
                f"WMAPE: {pct(scores['WMAPE'])}",
                f"RMSE: {scores['RMSE']:.2f}",
                f"Coverage@95: {pct(scores['Coverage@95'])}",
            ]))
            html.append(static_explain(
                what="Producer Price Index (PPI): prices received by domestic producers.",
                how="Index (base=100 at a reference period). Up = higher producer prices.",
                forecast_note="3/12/24-month ETS outlook with ≈95% band; calibration via Coverage@95.",
                extra="Drivers: energy, supply bottlenecks, FX; persistent PPI pressure can foreshadow CPI."
            ))
            html.append(ai_explain(
                section_title=f"BLS PPI — {sid}",
                freq_unit="Monthly, Index",
                history=ts, forecast=(f, lo, hi, scores),
                extras={"acronyms":{"PPI":"Producer Price Index","ETS":"Exponential Smoothing"}}
            ))

    # ---------- Methodology ----------
    html.append(H(2, "Methodology"))
    html.append(bullet_list([
        "<b>Sources:</b> WITS (annual, converted from “US$ Mil” to USD), U.S. Census HS timeseries (monthly USD), BLS PPI (monthly index).",
        "<b>Frequencies:</b> WITS=Annual; Census & BLS=Monthly (nominal).",
        "<b>Model:</b> ETS/Holt-Winters (additive trend; seasonality=12 when ≥36 months).",
        "<b>Uncertainty:</b> ≈95% bands from empirical residual quantiles (fallback ±1.96σ).",
        "<b>Backtest:</b> Last 12 months OOS; we report sMAPE, WMAPE, RMSE, Coverage@95.",
        "<b>Partner concentration:</b> Herfindahl-Hirschman Index (HHI) using last-12-month shares.",
        "<b>Sanity checks:</b> Units, last date, non-null %, zeros %, and row counts printed above charts."
    ]))
    html.append(H(3, "Glossary"))
    html.append(bullet_list([
        "<b>ETS:</b> Exponential smoothing time-series model (Holt-Winters).",
        "<b>sMAPE:</b> Symmetric MAPE (robust to near-zero values).",
        "<b>WMAPE:</b> Weighted MAPE (|error| as share of total actuals).",
        "<b>Coverage@95:</b> Share of backtest points inside the 95% band — ~95% is well-calibrated.",
        "<b>HHI:</b> Sum of squared shares × 10,000; higher = more concentration."
    ]))

    html.append("</body></html>")
    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(html))
    print(f"✅ Report built → {HTML_PATH}")

if __name__ == "__main__":
    build_report()
