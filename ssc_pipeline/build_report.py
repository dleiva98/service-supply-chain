# ssc_pipeline/build_report.py
# Executive, trustworthy trade dashboard with optional GPT narratives.
# - WITS (annual USD, scaled from "US$ Mil")
# - Census imports/exports (monthly USD, de-aggregated)
# - BLS PPI (monthly index)
# - ETS/Holt-Winters forecasts, robust bands, calibrated Coverage@95
# - Partner deep-dive with HHI
# - Sanity panels + clear static explanations
# - OPTIONAL GPT explainer (enable with SSC_USE_AI=1 and OPENAI_API_KEY)

import os
import io
import json
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # CI/headless safe
import matplotlib.pyplot as plt

from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ----------------------------
# Paths / environment toggles
# ----------------------------
ROOT = Path(os.environ.get("SSC_ROOT", ".")).resolve()
OUT_DIR = ROOT / "ssc_pipeline"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = ROOT / "main.csv"
HTML_PATH = OUT_DIR / "Auto_Report.html"

# Optional AI layer (set these in GitHub Actions env/secrets)
USE_AI = os.getenv("SSC_USE_AI", "0") == "1"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # set via repo vars if preferred

# -------------
# Constants
# -------------
# WITS headline values are “in US$ Mil” → convert to USD
WITS_TO_USD = 1_000_000

# -------------
# Formatting
# -------------
def human_usd(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    n = float(x)
    sgn = "-" if n < 0 else ""
    n = abs(n)
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

def bullet_list(items) -> str:
    return "<ul>" + "".join([f"<li>{it}</li>" for it in items]) + "</ul>"

# ----------------------------
# Load & normalize main.csv
# ----------------------------
def load_main() -> pd.DataFrame:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"main.csv not found at {CSV_PATH} — run the update-data workflow first.")
    df = pd.read_csv(CSV_PATH, low_memory=False)
    if "Value" in df.columns and "value" not in df.columns:
        df = df.rename(columns={"Value": "value"})
    return df

def prepare_wits(df: pd.DataFrame) -> pd.DataFrame:
    w = df[df["source"].isin(["wits_trade", "wits_tariff"])].copy()
    if w.empty: return w
    # Convert trade values from "US$ Mil" to USD
    is_trd = w["indicator"].astype(str).str.contains("TRD-VL", case=False, na=False)
    w.loc[is_trd, "value_usd"] = pd.to_numeric(w.loc[is_trd, "value"], errors="coerce") * WITS_TO_USD
    # WITS is annual
    if "time" in w.columns:
        w["year"] = pd.to_numeric(w["time"], errors="coerce").astype("Int64")
    return w

def prepare_census(df: pd.DataFrame) -> pd.DataFrame:
    c = df[df["source"].isin(["census_imports", "census_exports"])].copy()
    if c.empty: return c

    name_col = "CTY_NAME" if "CTY_NAME" in c.columns else ("partner_name" if "partner_name" in c.columns else ("partner" if "partner" in c.columns else None))
    code_col = "CTY_CODE" if "CTY_CODE" in c.columns else ("partner_code" if "partner_code" in c.columns else ("partner" if "partner" in c.columns else None))
    if name_col is None: name_col = "partner"
    if code_col is None: code_col = "partner"

    upnames = c[name_col].astype(str).str.upper()
    bad = upnames.str.contains("TOTAL") | upnames.str.contains("ALL COUNTRIES") | upnames.eq("WLD")
    bad = bad | c[code_col].astype(str).str.upper().isin(["ALL","WLD","000","TOT","0"])
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

    # Keep friendly partner label for deep-dive
    c["partner_label"] = c[name_col].astype(str)
    return c

def prepare_bls(df: pd.DataFrame) -> pd.DataFrame:
    b = df[df["source"].eq("bls_ppi")].copy()
    if b.empty: return b
    b["month"] = b["period"].astype(str).str.extract(r"M(\d{2})").astype(float)
    b["year"]  = pd.to_numeric(b["year"], errors="coerce")
    b["date"]  = pd.PeriodIndex(year=b["year"].astype(int), month=b["month"].astype(int), freq="M").to_timestamp("M")
    b["value"] = pd.to_numeric(b["value"], errors="coerce")
    b = b.dropna(subset=["date","value"]).sort_values("date")
    return b

# ----------------------------
# Time-series helpers
# ----------------------------
def to_monthly_unique(series: pd.Series, how: str = "sum") -> pd.Series:
    s = series.copy()
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index, errors="coerce")
    s = s.dropna().sort_index()

    if s.index.has_duplicates:
        if how == "mean":
            s = s.groupby(s.index).mean()
        else:
            s = s.groupby(s.index).sum()

    if how == "mean":
        s = s.resample("M").mean()
    else:
        s = s.resample("M").sum()

    return s

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

def ets_forecast(ts: pd.Series, h: int = 12, seasonal: bool = True):
    """
    Robust ETS/Holt-Winters forecaster with graceful fallbacks.

    - Cleans the series to a monthly grid and drops NAs.
    - Uses additive trend; seasonality=12 ONLY if >=36 points after cleaning.
    - If fitting fails (or series too short), falls back to:
        Holt → SimpleExpSmoothing → Naïve (last value repeated).
    - 95% bands from empirical residual quantiles when available; otherwise ±1.96*sd.
    - Backtest on last 12 months only if we have >=36 points post-clean.

    Returns: (f, lo, hi, scores) where scores has keys sMAPE, WMAPE, RMSE, Coverage@95.
    """
    # ---- Clean & regularize ----
    s = ts.copy()
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index, errors="coerce")
    s = s.dropna().sort_index()

    # if duplicates → aggregate (mean; for dollar totals you already pass aggregated series)
    if s.index.has_duplicates:
        s = s.groupby(s.index).mean()

    # put on a monthly grid
    s = s.resample("M").mean()
    s = s.dropna()

    # If basically empty, return a flat forecast with NaN scores
    if len(s) == 0:
        idx_f = pd.date_range(pd.Timestamp.today().normalize() + pd.offsets.MonthEnd(1), periods=h, freq="M")
        f = pd.Series([np.nan]*h, index=idx_f)
        return f, f, f, {"sMAPE": float("nan"), "WMAPE": float("nan"), "RMSE": float("nan"), "Coverage@95": float("nan")}

    # Helper to build simple bands from sd
    def bands_from_sd(base: pd.Series, sd: float):
        return base - 1.96*sd, base + 1.96*sd

    # Decide seasonality
    use_season = (seasonal and len(s) >= 36)
    season_periods = 12 if use_season else None

    # Try full ETS
    f = lo = hi = None
    scores = {"sMAPE": float("nan"), "WMAPE": float("nan"),
              "RMSE": float("nan"), "Coverage@95": float("nan")}
    fitted = None

    # Fit with progressive fallback
    try:
        model = ExponentialSmoothing(s, trend="add",
                                     seasonal=("add" if use_season else None),
                                     seasonal_periods=season_periods)
        fitted = model.fit(optimized=True, use_brute=True)
    except Exception:
        # Try Holt (no seasonality)
        try:
            model = ExponentialSmoothing(s, trend="add", seasonal=None)
            fitted = model.fit(optimized=True, use_brute=True)
        except Exception:
            # Try simple exponential smoothing (no trend/season)
            try:
                model = ExponentialSmoothing(s, trend=None, seasonal=None)
                fitted = model.fit(optimized=True, use_brute=True)
            except Exception:
                fitted = None

    # Forecast + bands
    if fitted is not None:
        f_vals = fitted.forecast(h)
        # ensure index continues monthly after last observed
        idx_f = pd.date_range(s.index[-1] + pd.offsets.MonthEnd(1), periods=h, freq="M")
        f = pd.Series(f_vals.values, index=idx_f)

        # Residuals for bands
        resid = (s - fitted.fittedvalues.reindex(s.index)).dropna().values
        if resid.size >= 8:
            q = np.nanpercentile(resid, [2.5, 97.5])
            if np.ndim(q) == 0:
                sd = float(np.nanstd(resid, ddof=1))
                qlo, qhi = -1.96*sd, 1.96*sd
            else:
                qlo, qhi = float(q[0]), float(q[1])
        else:
            sd = float(np.nanstd(resid, ddof=1)) if resid.size > 1 else float(np.nanstd(s.values, ddof=1))
            qlo, qhi = -1.96*sd, 1.96*sd

        lo = f + qlo
        hi = f + qhi

        # Backtest (12m) if enough history
        if len(s) >= 36:
            train = s.iloc[:-12]
            test  = s.iloc[-12:]
            # Mirror fallback strategy for backtest
            fitted_bt = None
            for cfg in [("add", use_season and len(train) >= 36),
                        ("add", False),
                        (None, False)]:
                tr, sez = cfg
                try:
                    m2 = ExponentialSmoothing(train, trend=tr,
                                              seasonal=("add" if sez else None),
                                              seasonal_periods=(12 if sez else None))
                    fitted_bt = m2.fit(optimized=True, use_brute=True)
                    break
                except Exception:
                    continue
            if fitted_bt is not None:
                fc_bt = fitted_bt.forecast(12)
                resid_bt = (train - fitted_bt.fittedvalues.reindex(train.index)).dropna().values
                if resid_bt.size >= 8:
                    q_bt = np.nanpercentile(resid_bt, [2.5, 97.5])
                    if np.ndim(q_bt) == 0:
                        sd_bt = float(np.nanstd(resid_bt, ddof=1)); qlo_bt, qhi_bt = -1.96*sd_bt, 1.96*sd_bt
                    else:
                        qlo_bt, qhi_bt = float(q_bt[0]), float(q_bt[1])
                else:
                    sd_bt = float(np.nanstd(resid_bt, ddof=1)) if resid_bt.size>1 else float(np.nanstd(train.values, ddof=1))
                    qlo_bt, qhi_bt = -1.96*sd_bt, 1.96*sd_bt

                lo_bt = fc_bt + qlo_bt
                hi_bt = fc_bt + qhi_bt

                scores = {
                    "sMAPE": smape(test.values, fc_bt.values),
                    "WMAPE": wmape(test.values, fc_bt.values),
                    "RMSE": float(np.sqrt(np.nanmean((test.values - fc_bt.values)**2))),
                    "Coverage@95": coverage(test.values, lo_bt.values, hi_bt.values),
                }
    else:
        # Final fallback: naive forecast (repeat last value)
        last = float(s.iloc[-1])
        idx_f = pd.date_range(s.index[-1] + pd.offsets.MonthEnd(1), periods=h, freq="M")
        f = pd.Series(last, index=idx_f)
        sd = float(np.nanstd(s.values, ddof=1)) if len(s.values) > 1 else 0.0
        lo, hi = bands_from_sd(f, sd)

    return f, lo, hi, scores

# ----------------------------
# Plot helpers
# ----------------------------
def save_line(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

def plot_series(ts: pd.Series, title: str, y_label: str, fig_name: str) -> Path:
    plt.figure(figsize=(8.6, 4.4))
    plt.plot(ts.index, ts.values, marker="o")
    plt.title(title); plt.xlabel("Date"); plt.ylabel(y_label)
    path = OUT_DIR / fig_name
    save_line(path); return path

def plot_forecast(ts: pd.Series, f: pd.Series, lo: pd.Series, hi: pd.Series,
                  title: str, y_label: str, fig_name: str) -> Path:
    plt.figure(figsize=(8.6, 4.4))
    plt.plot(ts.index, ts.values, label="History")
    idx_f = pd.date_range(ts.index[-1] + pd.offsets.MonthEnd(1), periods=len(f), freq="M")
    plt.plot(idx_f, f.values, linestyle="--", marker="o", label="Forecast (12m)")
    plt.fill_between(idx_f, lo.values, hi.values, alpha=0.18, label="≈95% band")
    plt.title(title); plt.xlabel("Date"); plt.ylabel(y_label); plt.legend()
    path = OUT_DIR / fig_name
    save_line(path); return path

# ----------------------------
# Sanity panel & Explanations
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
    return "<div style='margin:8px 0 14px 0'>" + "<br/>".join(parts) + "</div>"

# ----------------------------
# Optional GPT explainer
# ----------------------------
def _ai_chat(system_prompt: str, user_prompt: str) -> str:
    if not (USE_AI and OPENAI_API_KEY):
        return ""
    try:
        import requests
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": OPENAI_MODEL,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=45)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return ""

def ai_explain_chart(section_title: str, freq_unit: str, series: Optional[pd.Series],
                     forecast: Optional[Tuple[pd.Series,pd.Series,pd.Series,dict]],
                     extras: Optional[Dict[str,Any]] = None) -> str:
    if not (USE_AI and OPENAI_API_KEY): return ""

    def sum_stats(s: Optional[pd.Series], k=24):
        if s is None or s.empty: return {}
        s2 = s.dropna()
        tail = s2.tail(k)
        yoy = None
        try:
            if len(s2) >= 13 and (getattr(s2.index, "freqstr", None) in ("M","ME") or True):
                yoy = 100.0 * (s2.iloc[-1] - s2.iloc[-13]) / max(1e-9, abs(s2.iloc[-13]))
        except Exception:
            pass
        return {
            "last_date": str(s2.index[-1].date()) if hasattr(s2.index[-1], "date") else str(s2.index[-1]),
            "last_value": float(s2.iloc[-1]),
            "min_lastK": float(np.nanmin(tail.values)) if len(tail) else None,
            "max_lastK": float(np.nanmax(tail.values)) if len(tail) else None,
            "yoy_pct": float(yoy) if yoy is not None else None,
            "count": int(len(s2)),
            "non_null_pct": float(s.notna().mean()*100.0) if s is not None else None
        }

    hist = sum_stats(series)
    fc_stats = {}
    if forecast is not None:
        try:
            f, lo, hi, scores = forecast
            fc_stats = {
                "h": int(len(f)),
                "fc_last": float(f.iloc[-1]),
                "band_last": [float(lo.iloc[-1]), float(hi.iloc[-1])],
                "scores": {k: (None if (isinstance(v,float) and (np.isnan(v) or np.isinf(v))) else float(v)) for k,v in (scores or {}).items()}
            }
        except Exception:
            fc_stats = {}

    sys_msg = (
        "You are an economist writing a concise, executive narrative (4–7 sentences). "
        "Explain acronyms (PPI, ETS), state frequency & units, and provide one clear takeaway. "
        "Discuss near-term outlook and risks with light geopolitical context (tariffs, sanctions, elections, conflicts) "
        "but do not speculate beyond what the data implies. "
        "If Coverage@95 is far from 95, mention calibration caveat. Keep it factual and readable."
    )
    user_msg = (
        f"Section: {section_title}\n"
        f"Frequency & units: {freq_unit}\n"
        f"History: {json.dumps(hist)}\n"
        f"Forecast: {json.dumps(fc_stats)}\n"
        f"Extras: {json.dumps(extras or {})}\n"
        "Write for a non-technical business reader."
    )
    out = _ai_chat(sys_msg, user_msg)
    return f"<div style='margin:8px 0 14px 0'>{out}</div>" if out else ""

# ----------------------------
# Partner deep-dive (Census)
# ----------------------------
def partner_deep_dive(cen: pd.DataFrame, source_key: str, title: str, top_n: int = 8):
    """
    Build a YoY/top-partner table and plot for imports or exports.
    Assumes cen already de-aggregated (TOTAL/ALL removed).
    """
    df = cen[cen["source"].eq(source_key)].dropna(subset=["date","value_usd","partner_label"]).copy()
    if df.empty:
        return "", ""

    # Last 12 months aggregate by partner
    last_date = df["date"].max()
    year_ago = (last_date - pd.DateOffset(years=1)).normalize() + pd.offsets.MonthEnd(0)
    recent = df[df["date"] > (last_date - pd.DateOffset(months=12))]
    grp = recent.groupby("partner_label", as_index=False)["value_usd"].sum().sort_values("value_usd", ascending=False)

    # HHI (share^2 * 10_000)
    total = grp["value_usd"].sum()
    grp["share"] = grp["value_usd"] / max(total, 1e-9)
    hhi = int(round((grp["share"]**2).sum() * 10_000))

    top = grp.head(top_n).copy()

    # Simple bar chart
    plt.figure(figsize=(8.6, 4.4))
    plt.bar(top["partner_label"], top["value_usd"])
    plt.title(f"{title} — Top {top_n} (Last 12m)")
    plt.ylabel("USD")
    plt.xticks(rotation=30, ha="right")
    fig_path = OUT_DIR / f"{source_key}_partners_top.png"
    save_line(fig_path)

    # HTML table for top partners
    tbl = "<table border='1' cellpadding='4' cellspacing='0'><tr><th>Partner</th><th>Last 12m (USD)</th><th>Share</th></tr>"
    for _, r in top.iterrows():
        tbl += f"<tr><td>{r['partner_label']}</td><td>{human_usd(r['value_usd'])}</td><td>{pct(r['share']*100)}</td></tr>"
    tbl += "</table>"

    expl = static_explain(
        what=f"Top trade partners for the U.S. by {('imports' if 'import' in source_key else 'exports')} in the last 12 months.",
        how="Bars show each partner’s dollar total; the table includes last-12-month value and share of total.",
        forecast_note="Not a forecast; use this to spot concentration risk.",
        extra=f"Concentration (HHI): {hhi} (≈1,500–2,500 implies moderate concentration; >2,500 high)."
    )
    return f"<img src='{fig_path.name}'/>" + expl + tbl, hhi

# ----------------------------
# Build the report
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

    # Title
    html.append(H(1, "U.S. Trade & Supply Chain — Executive Dashboard"))

    # ===== WITS (Annual) =====
    if not wits.empty:
        wt = wits[wits["indicator"].isin(["MPRT-TRD-VL","XPRT-TRD-VL"])].copy()
        wt = wt[(wt["partner"].astype(str).str.upper()=="WLD") & (wt["product"].astype(str).str.lower()=="total")]
        if not wt.empty:
            wt["kind"] = np.where(wt["indicator"].eq("MPRT-TRD-VL"), "Imports", "Exports")
            g = wt.groupby(["year","kind"], as_index=False)["value_usd"].sum().dropna()
            wide = g.pivot(index="year", columns="kind", values="value_usd").sort_index()

            html.append(H(2, "WITS — Annual Totals (USD)"))
            html.append(small("Frequency: Annual · Unit: USD (converted from 'US$ Mil') · Coverage: Goods (nominal)"))

            for col in wide.columns:
                ts = wide[col].dropna()
                if ts.empty: continue
                # use end-of-year dates for plotting
                ts2 = pd.Series(ts.values, index=pd.to_datetime(ts.index.astype(str) + "-12-31"))
                plot_series(ts2, f"WITS — {col} (Annual, USD)", "USD", f"wits_{col.lower()}.png")
                html.append(f"<img src='wits_{col.lower()}.png'/>")
                # Explanations
                html.append(static_explain(
                    what=f"U.S. {col.lower()} of goods by calendar year (WITS).",
                    how="Each point is a full-year total in U.S. dollars; not monthly.",
                    extra="Policy shifts (tariffs, FTAs), sanctions, and global shocks (pandemics, conflicts) can move annual totals."
                ))
                html.append(ai_explain_chart(
                    section_title=f"WITS — {col} (Annual)",
                    freq_unit="Annual, USD",
                    series=ts2, forecast=None,
                    extras={"source":"WITS (converted from US$ Mil)"}
                ))

            if not wide.empty:
                last_year = wide.index.max()
                bullets = [f"{col} {int(last_year)}: {human_usd(wide.loc[last_year,col])}" for col in wide.columns]
                html.append(bullet_list(bullets))

    # ===== Census (Monthly) =====
    if not cen.empty:
        imp_raw = cen[cen["source"].eq("census_imports")].set_index("date")["value_usd"]
        exp_raw = cen[cen["source"].eq("census_exports")].set_index("date")["value_usd"]
        imports_m = to_monthly_unique(imp_raw, how="sum")
        exports_m = to_monthly_unique(exp_raw, how="sum")

        # Imports
        if not imports_m.empty:
            html.append(H(2, "Census — Goods Imports (Monthly, USD)"))
            html.append(sanity_panel(cen[cen["source"].eq("census_imports")], "value_usd","date","USD"))
            plot_series(imports_m, "U.S. Goods Imports — Monthly (USD)", "USD", "census_imports_hist.png")
            html.append("<img src='census_imports_hist.png'/>")
            f, lo, hi, scores = ets_forecast(imports_m, h=12, seasonal=True)
            plot_forecast(imports_m, f, lo, hi, "Imports — 12-Month ETS Forecast", "USD", "census_imports_fc.png")
            html.append("<img src='census_imports_fc.png'/>")
            html.append(bullet_list([
                f"Latest level: {human_usd(imports_m.iloc[-1])}",
                f"sMAPE (12-mo OOS): {pct(scores['sMAPE'])}",
                f"WMAPE: {pct(scores['WMAPE'])}",
                f"RMSE: {human_usd(scores['RMSE'])}",
                f"Coverage@95: {pct(scores['Coverage@95'])}",
            ]))
            html.append(static_explain(
                what="Monthly total dollar value of U.S. goods imports across all partners.",
                how="Up = more imports in USD. Values are nominal (not inflation-adjusted).",
                forecast_note="Dashed line shows ETS forecast for 12 months; band is an empirical ≈95% interval. Coverage@95 near 95% indicates well-calibrated uncertainty.",
                extra="Watch for policy risks (Section 301 tariffs, antidumping actions), supply disruptions (Red Sea, port congestion), and election-year uncertainty."
            ))
            html.append(ai_explain_chart(
                section_title="Census — U.S. Goods Imports (Monthly)",
                freq_unit="Monthly, USD",
                series=imports_m,
                forecast=(f, lo, hi, scores),
                extras={"source":"U.S. Census HS timeseries","coverage95":scores.get("Coverage@95")}
            ))

            # Partner deep-dive (imports)
            block, hhi_imp = partner_deep_dive(cen, "census_imports", "Imports — Partner Deep-Dive")
            if block:
                html.append(H(3, "Imports — Partner Deep-Dive (Last 12 months)"))
                html.append(block)
                html.append(ai_explain_chart(
                    section_title="Imports — Partner Deep-Dive",
                    freq_unit="Aggregated last 12 months, USD",
                    series=None, forecast=None,
                    extras={"hhi":hhi_imp, "note":"Higher HHI = more concentrated exposure; monitor top partners' policy shifts."}
                ))

        # Exports
        if not exports_m.empty:
            html.append(H(2, "Census — Goods Exports (Monthly, USD)"))
            html.append(sanity_panel(cen[cen["source"].eq("census_exports")], "value_usd","date","USD"))
            plot_series(exports_m, "U.S. Goods Exports — Monthly (USD)", "USD", "census_exports_hist.png")
            html.append("<img src='census_exports_hist.png'/>")
            f, lo, hi, scores = ets_forecast(exports_m, h=12, seasonal=True)
            plot_forecast(exports_m, f, lo, hi, "Exports — 12-Month ETS Forecast", "USD", "census_exports_fc.png")
            html.append("<img src='census_exports_fc.png'/>")
            html.append(bullet_list([
                f"Latest level: {human_usd(exports_m.iloc[-1])}",
                f"sMAPE (12-mo OOS): {pct(scores['sMAPE'])}",
                f"WMAPE: {pct(scores['WMAPE'])}",
                f"RMSE: {human_usd(scores['RMSE'])}",
                f"Coverage@95: {pct(scores['Coverage@95'])}",
            ]))
            html.append(static_explain(
                what="Monthly total dollar value of U.S. goods exports across all partners.",
                how="Up = more exports in USD. Values are nominal.",
                forecast_note="12-month ETS forecast with empirical ≈95% band. If coverage is below 95%, bands are likely too narrow and should be treated cautiously.",
                extra="Geopolitics (export controls, sanctions) and partner growth (e.g., Mexico, Canada, EU, UK, Japan) shape the near-term outlook."
            ))
            html.append(ai_explain_chart(
                section_title="Census — U.S. Goods Exports (Monthly)",
                freq_unit="Monthly, USD",
                series=exports_m,
                forecast=(f, lo, hi, scores),
                extras={"source":"U.S. Census HS timeseries","coverage95":scores.get("Coverage@95")}
            ))

            # Partner deep-dive (exports)
            block, hhi_exp = partner_deep_dive(cen, "census_exports", "Exports — Partner Deep-Dive")
            if block:
                html.append(H(3, "Exports — Partner Deep-Dive (Last 12 months)"))
                html.append(block)
                html.append(ai_explain_chart(
                    section_title="Exports — Partner Deep-Dive",
                    freq_unit="Aggregated last 12 months, USD",
                    series=None, forecast=None,
                    extras={"hhi":hhi_exp, "note":"Concentration suggests where demand or policy shocks matter most."}
                ))

    # ===== BLS PPI (Monthly Index) =====
    if not bls.empty:
        for sid, sub in bls.groupby("series_id"):
            raw = sub.set_index("date")["value"]
            ts  = to_monthly_unique(raw, how="mean")  # index → mean for duplicates
            if ts.empty: continue

            html.append(H(2, f"BLS PPI — {sid} (Monthly Index)"))
            html.append(sanity_panel(sub, "value","date","Index"))
            plot_series(ts, f"PPI {sid} — Monthly (Index)", "Index", f"bls_{sid}_hist.png")
            html.append(f"<img src='bls_{sid}_hist.png'/>")
            f, lo, hi, scores = ets_forecast(ts, h=12, seasonal=True)
            plot_forecast(ts, f, lo, hi, f"PPI {sid} — 12-Month ETS Forecast", "Index", f"bls_{sid}_fc.png")
            html.append(f"<img src='bls_{sid}_fc.png'/>")
            html.append(bullet_list([
                f"sMAPE (12-mo OOS): {pct(scores['sMAPE'])}",
                f"WMAPE: {pct(scores['WMAPE'])}",
                f"RMSE: {scores['RMSE']:.2f}",
                f"Coverage@95: {pct(scores['Coverage@95'])}",
            ]))
            html.append(static_explain(
                what="Producer Price Index (PPI): upstream prices received by domestic producers — a key input-cost signal.",
                how="Index (base=100 at a reference period). Up = higher producer prices.",
                forecast_note="12-month ETS forecast with ≈95% band. Calibration indicated by Coverage@95.",
                extra="Supply shocks, energy prices, and currency shifts influence PPI; persistent increases often precede consumer inflation."
            ))
            html.append(ai_explain_chart(
                section_title=f"BLS PPI — {sid}",
                freq_unit="Monthly, Index",
                series=ts,
                forecast=(f, lo, hi, scores),
                extras={"acronyms":{"PPI":"Producer Price Index","ETS":"Exponential Smoothing"}}
            ))

    # ===== Methodology & Glossary =====
    html.append(H(2, "Methodology"))
    html.append(bullet_list([
        "<b>Data sources:</b> WITS (annual totals, converted from “US$ Mil” to USD), U.S. Census HS timeseries (monthly USD), BLS PPI (monthly index).",
        "<b>Frequencies:</b> WITS = annual; Census & BLS = monthly.",
        "<b>Models:</b> ETS/Holt-Winters with additive trend; seasonality=12 when ≥36 months of history.",
        "<b>Uncertainty:</b> ≈95% bands from empirical residual quantiles (fallback ±1.96σ if thin history).",
        "<b>Backtest:</b> Last 12 months OOS when history allows; metrics reported below each chart.",
        "<b>Metrics:</b> sMAPE (scale-free), WMAPE (weighted), RMSE (level error), Coverage@95 (calibration).",
        "<b>Partner analysis:</b> Top partners over the last 12 months; concentration via Herfindahl-Hirschman Index (HHI).",
        "<b>Quality guardrails:</b> Sanity panels list units, last date, non-null %, zeros %, and row count."
    ]))
    html.append(H(3, "Glossary"))
    html.append(bullet_list([
        "<b>ETS / Holt-Winters:</b> Exponential smoothing time-series model with optional seasonality.",
        "<b>sMAPE:</b> Symmetric Mean Absolute Percentage Error — robust when values approach zero.",
        "<b>WMAPE:</b> Weighted MAPE — absolute error as a share of total actuals.",
        "<b>Coverage@95:</b> Share of backtest points captured by the 95% band; ~95% indicates well-calibrated uncertainty.",
        "<b>HHI:</b> Sum of squared shares × 10,000; higher = more concentration (risk)."
    ]))

    # Close HTML
    html.append("</body></html>")
    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(html))
    print(f"✅ Report built → {HTML_PATH}")

if __name__ == "__main__":
    build_report()
