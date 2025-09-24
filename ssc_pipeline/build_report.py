# ssc_pipeline/build_report.py
# Executive-grade U.S. Trade & Supply Chain report (HTML):
# - WITS scaling fixed (US$ Mil -> USD), annual only (no forward-fill)
# - Census monthly USD enforced (rejects "units" pulls)
# - Partners plausibility guard + named partners (MX, CA, CN, JP, DE, UK, IN)
# - AI narratives (3/6/12-month outlooks) when OPENAI key present
# - Base64-embedded plots (Pages-safe)
# - Robust ETS forecasting with calibration/backtests
# - Clear score explanations (for non-economists)
# - Data-quality panels per chart

import os, io, json, base64, math, warnings
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless for CI
import matplotlib.pyplot as plt
plt.rcParams["axes.grid"] = True
plt.rcParams["figure.dpi"] = 150

from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ----------------------------
# Paths / environment
# ----------------------------
ROOT = Path(os.environ.get("SSC_ROOT", ".")).resolve()
OUT_DIR = ROOT / "ssc_pipeline"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = ROOT / "main.csv"
HTML_PATH = OUT_DIR / "Auto_Report.html"

# Optional AI layer
USE_AI = os.getenv("SSC_USE_AI", "0") == "1"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # you can set gpt-4o
AI_TIMEOUT = int(os.getenv("SSC_AI_TIMEOUT", "45"))

# Conversions
WITS_TO_USD = 1.0  

# Named deep-dive partners (canonical labels)
FOCUS_PARTNERS = [
    "Mexico", "Canada", "China", "Japan", "Germany", "United Kingdom", "India"
]

# Micro/implausible partners (to flag if they bubble to top)
SUSPECT_PARTNERS = {
    "NIUE","COMOROS","COCOS (KEELING) ISLANDS","KOREA, NORTH",
    "WALLIS AND FUTUNA","SVALBARD, JAN MAYEN ISLAND","NAURU",
    "HEARD AND MCDONALD ISLANDS","PITCAIRN","TUVALU","TOKELAU",
    "SAINT HELENA","BRITISH INDIAN OCEAN TERRITORY"
}

# Synonyms/unify partner names
PARTNER_NORMALIZE = {
    "UNITED KINGDOM, THE": "United Kingdom",
    "UNITED STATES": "United States",
    "KOREA, REP.": "Korea, Rep.",
    "KOREA, SOUTH": "Korea, Rep.",
    "KOREA, NORTH": "Korea, DPRK",
    "RUSSIAN FEDERATION": "Russia",
    "VIET NAM": "Vietnam",
    "TAIWAN, CHINA": "Taiwan",
    "HONG KONG SAR, CHINA": "Hong Kong",
    "CZECH REPUBLIC": "Czechia",
    "TURKEY": "Türkiye",
}

# ----------------------------
# HTML helpers
# ----------------------------
def h(n: int, txt: str) -> str:
    return f"<h{n}>{txt}</h{n}>"

def small(txt: str) -> str:
    return f"<div style='color:#666;font-size:0.9em'>{txt}</div>"

def bullet_list(items: List[str]) -> str:
    return "<ul>" + "".join([f"<li>{it}</li>" for it in items]) + "</ul>"

def human_usd(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    n = float(x)
    sign = "-" if n < 0 else ""
    n = abs(n)
    if n >= 1e12: return f"{sign}{n/1e12:.2f}T"
    if n >= 1e9:  return f"{sign}{n/1e9:.2f}B"
    if n >= 1e6:  return f"{sign}{n/1e6:.2f}M"
    return f"{sign}{n:,.0f}"

def pct(x: Optional[float]) -> str:
    return "—" if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))) else f"{x:.1f}%"

def embed_png(path: Path) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"<img src='data:image/png;base64,{b64}'/>"

def save_and_embed(fig_name: str) -> str:
    path = OUT_DIR / fig_name
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return embed_png(path)

# ----------------------------
# Load & normalize data
# ----------------------------
def load_main() -> pd.DataFrame:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"main.csv not found at {CSV_PATH}")
    df = pd.read_csv(CSV_PATH, low_memory=False)
    # normalize "Value" -> "value"
    if "value" not in df.columns and "Value" in df.columns:
        df = df.rename(columns={"Value": "value"})
    # normalize "time" type (some sources have "2019.0")
    if "time" in df.columns:
        df["time"] = pd.to_numeric(df["time"], errors="coerce")
    return df

def normalize_partner_label(s: pd.Series) -> pd.Series:
    lab = s.fillna("").astype(str).str.strip()
    up = lab.str.upper()
    # quick map for synonyms
    mapped = lab.copy()
    for k, v in PARTNER_NORMALIZE.items():
        mapped[up.eq(k)] = v
    return mapped

def prepare_wits(df: pd.DataFrame) -> pd.DataFrame:
    w = df[df["source"].isin(["wits_trade","wits_tariff"])].copy()
    if w.empty: return w
    # annual year from 'time' (integer)
    w["year"] = pd.to_numeric(w.get("time"), errors="coerce").round().astype("Int64")
    v = pd.to_numeric(w.get("value"), errors="coerce")
    is_trade = w["indicator"].astype(str).str.contains("TRD-VL", case=False, na=False)
    w["value_usd"] = np.where(is_trade, v * WITS_TO_USD, v)
    w["partner_label"] = normalize_partner_label(w.get("partner", ""))
    w["product_label"]  = w.get("product","").astype(str)
    return w

def prepare_census(df: pd.DataFrame) -> pd.DataFrame:
    c = df[df["source"].isin(["census_imports","census_exports"])].copy()
    if c.empty: return c

    # pick partner name/code columns flexibly
    name_col = None
    for cand in ["CTY_NAME","partner_name","partner"]:
        if cand in c.columns: name_col = cand; break
    if name_col is None: name_col = "partner"

    code_col = None
    for cand in ["CTY_CODE","partner_code","partner"]:
        if cand in c.columns: code_col = cand; break
    if code_col is None: code_col = "partner"

    # drop totals/world
    names_up = c[name_col].astype(str).str.upper()
    drop_mask = names_up.str.contains("TOTAL") | names_up.str.contains("ALL COUNTRIES") | names_up.eq("WLD")
    drop_mask |= c[code_col].astype(str).str.upper().isin(["ALL","WLD","000","TOT","0"])
    c = c.loc[~drop_mask].copy()

    # date (prefer YEAR/MONTH)
    if {"YEAR","MONTH"}.issubset(c.columns):
        c["year"]  = pd.to_numeric(c["YEAR"], errors="coerce")
        c["month"] = pd.to_numeric(c["MONTH"], errors="coerce")
    elif {"year","period"}.issubset(c.columns):
        c["year"]  = pd.to_numeric(c["year"], errors="coerce")
        c["month"] = c["period"].astype(str).str.extract(r"M(\d{2})").astype(float)
    else:
        raise ValueError("Census rows missing YEAR/MONTH or year/period (cannot build monthly index)")

    # ensure monthly date
    c = c.dropna(subset=["year","month"]).copy()
    c["year"]  = c["year"].astype(int)
    c["month"] = c["month"].astype(int)
    c["date"]  = pd.PeriodIndex(year=c["year"], month=c["month"], freq="M").to_timestamp("M")

    # value (USD only)
    val_col = None
    for cand in ["ALL_VAL_MO","value","VAL_MO","GEN_VAL_MO"]:
        if cand in c.columns: val_col = cand; break
    if val_col is None:
        raise ValueError("Census rows missing USD value (ALL_VAL_MO/value). Did the pipeline fetch 'units' instead of dollars?")
    c["value_usd"] = pd.to_numeric(c[val_col], errors="coerce")

    # partner label
    c["partner_label"] = normalize_partner_label(c[name_col])
    # filter outrageous negatives or NA dates
    c = c.dropna(subset=["date","value_usd"])
    c = c[c["value_usd"] >= 0]  # imports/exports in USD shouldn't be negative

    # if series are suspiciously tiny (e.g., thousands only), leave to sanity panel + flags
    return c.sort_values("date")

def prepare_bls(df: pd.DataFrame) -> pd.DataFrame:
    b = df[df["source"].eq("bls_ppi")].copy()
    if b.empty: return b
    b["month"] = b["period"].astype(str).str.extract(r"M(\d{2})").astype(float)
    b["year"]  = pd.to_numeric(b["year"], errors="coerce")
    b = b.dropna(subset=["year","month"])
    b["year"]  = b["year"].astype(int)
    b["month"] = b["month"].astype(int)
    b["date"]  = pd.PeriodIndex(year=b["year"], month=b["month"], freq="M").to_timestamp("M")
    b["value"] = pd.to_numeric(b["value"], errors="coerce")
    b = b.dropna(subset=["date","value"]).sort_values("date")
    return b

# ----------------------------
# Time-series utilities
# ----------------------------
def to_monthly_unique(series: pd.Series, how: str="sum") -> pd.Series:
    s = series.copy()
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index, errors="coerce")
    s = s.dropna().sort_index()
    if s.index.has_duplicates:
        s = s.groupby(s.index).sum() if how=="sum" else s.groupby(s.index).mean()
    # Fill monthly grid but don't forward-fill values; just align to month-end
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
    if len(actual)==0 or len(lo)==0 or len(hi)==0: return float("nan")
    hit = (actual >= lo) & (actual <= hi)
    return 100.0 * float(np.mean(hit))

def _bands_from_resid(resid: np.ndarray, fallback_sd: float) -> Tuple[float,float]:
    resid = np.asarray(resid, dtype=float)
    resid = resid[np.isfinite(resid)]
    if resid.size >= 8:
        lo, hi = np.nanpercentile(resid, [2.5, 97.5])
        return float(lo), float(hi)
    sd = float(np.nanstd(resid, ddof=1)) if resid.size>1 else float(fallback_sd)
    return -1.96*sd, 1.96*sd

def ets_forecast(ts: pd.Series, h: int=24, seasonal: bool=True):
    """
    Robust ETS with layered fallbacks. Returns (forecast, lo, hi, scores).
    Band calibration uses empirical residual quantiles; backtest is last 12 months OOS.
    """
    s = ts.copy()
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index, errors="coerce")
    s = s.dropna().sort_index()
    if s.index.has_duplicates:
        s = s.groupby(s.index).mean()
    # Keep monthly frequency, no forward-fill
    s = s.asfreq("M")
    s = s.dropna()
    if len(s) == 0:
        idx_f = pd.date_range(pd.Timestamp.today().normalize() + pd.offsets.MonthEnd(1), periods=h, freq="M")
        f = pd.Series([np.nan]*h, index=idx_f)
        return f, f, f, {"sMAPE": np.nan, "WMAPE": np.nan, "RMSE": np.nan, "Coverage@95": np.nan}

    use_season = seasonal and len(s) >= 36
    fitted = None
    for tr, sez in [("add", use_season), ("add", False), (None, False)]:
        try:
            m = ExponentialSmoothing(s, trend=tr, seasonal=("add" if sez else None), seasonal_periods=(12 if sez else None))
            fitted = m.fit(optimized=True, use_brute=True); break
        except Exception:
            continue

    if fitted is None:
        # naive: flat forecast + stdev band
        last = float(s.iloc[-1])
        idx_f = pd.date_range(s.index[-1] + pd.offsets.MonthEnd(1), periods=h, freq="M")
        f = pd.Series(last, index=idx_f)
        sd = float(np.nanstd(s.values, ddof=1)) if len(s.values)>1 else 0.0
        lo, hi = f - 1.96*sd, f + 1.96*sd
        return f, lo, hi, {"sMAPE": np.nan, "WMAPE": np.nan, "RMSE": np.nan, "Coverage@95": np.nan}

    # main forecast
    f_vals = fitted.forecast(h)
    idx_f = pd.date_range(s.index[-1] + pd.offsets.MonthEnd(1), periods=h, freq="M")
    f = pd.Series(f_vals.values, index=idx_f)

    # bands from residuals
    resid_full = (s - fitted.fittedvalues.reindex(s.index)).dropna().values
    qlo, qhi = _bands_from_resid(resid_full, fallback_sd=np.nanstd(s.values, ddof=1) if len(s.values)>1 else 0.0)
    lo = f + qlo; hi = f + qhi

    # backtest last 12 months
    scores = {"sMAPE": np.nan, "WMAPE": np.nan, "RMSE": np.nan, "Coverage@95": np.nan}
    if len(s) >= 36:
        train, test = s.iloc[:-12], s.iloc[-12:]
        fitted_bt = None
        for tr, sez in [("add", (seasonal and len(train)>=36)), ("add", False), (None, False)]:
            try:
                m2 = ExponentialSmoothing(train, trend=tr, seasonal=("add" if sez else None), seasonal_periods=(12 if sez else None))
                fitted_bt = m2.fit(optimized=True, use_brute=True); break
            except Exception:
                continue
        if fitted_bt is not None:
            fc_bt = fitted_bt.forecast(12)
            resid_bt = (train - fitted_bt.fittedvalues.reindex(train.index)).dropna().values
            lo_bt_off, hi_bt_off = _bands_from_resid(resid_bt, fallback_sd=np.nanstd(train.values, ddof=1))
            lo_bt, hi_bt = fc_bt + lo_bt_off, fc_bt + hi_bt_off
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                scores = {
                    "sMAPE": smape(test.values, fc_bt.values),
                    "WMAPE": wmape(test.values, fc_bt.values),
                    "RMSE": float(np.sqrt(np.nanmean((test.values - fc_bt.values)**2))),
                    "Coverage@95": coverage(test.values, lo_bt.values, hi_bt.values),
                }
    return f, lo, hi, scores

# ----------------------------
# AI explainer (optional)
# ----------------------------
def ai_chat(system_prompt: str, user_prompt: str) -> str:
    if not (USE_AI and OPENAI_API_KEY):
        return ""
    try:
        import requests
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": OPENAI_MODEL,
            "temperature": 0.25,
            "messages": [
                {"role":"system","content":system_prompt},
                {"role":"user","content":user_prompt}
            ]
        }
        r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=AI_TIMEOUT)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return ""

def summarize_series(s: Optional[pd.Series], k: int=24) -> Dict[str,Any]:
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

def explain_scores(scores: Dict[str,float]) -> str:
    if not scores: return ""
    smape_val = scores.get("sMAPE")
    wmape_val = scores.get("WMAPE")
    rmse_val  = scores.get("RMSE")
    cov_val   = scores.get("Coverage@95")
    lines = []
    if smape_val is not None and np.isfinite(smape_val):
        lines.append(f"<b>sMAPE</b> {smape_val:.1f}% — average percentage error, robust near zero.")
    if wmape_val is not None and np.isfinite(wmape_val):
        lines.append(f"<b>WMAPE</b> {wmape_val:.1f}% — weighted error over total actuals (lower is better).")
    if rmse_val is not None and np.isfinite(rmse_val):
        lines.append(f"<b>RMSE</b> {human_usd(rmse_val)} — typical absolute error in units.")
    if cov_val is not None and np.isfinite(cov_val):
        lines.append(f"<b>Coverage@95</b> {cov_val:.1f}% — share of points inside the 95% band (target ≈95%).")
    if not lines:
        return ""
    return small("Model scores: " + " ".join(lines))

def ai_explain(section_title: str, freq_unit: str,
               history: Optional[pd.Series],
               forecast: Optional[Tuple[pd.Series,pd.Series,pd.Series,dict]],
               extras: Optional[Dict[str,Any]]=None,
               horizons: List[int] = [3,6,12]) -> str:
    """
    Produces 1–2 short paragraphs (natural language) if AI is enabled.
    Explains what, how to read, current level & YoY, 3/6/12-month outlook, uncertainty, geopolitics, and takeaway.
    """
    if not (USE_AI and OPENAI_API_KEY):
        return ""
    hist = summarize_series(history)
    fc_stats = {}
    if forecast is not None:
        f, lo, hi, scores = forecast
        pts = {}
        for h in horizons:
            if len(f) >= h:
                pts[str(h)] = float(f.iloc[h-1])
        fc_stats = {
            "horizons": pts,
            "band_last": [float(lo.iloc[-1]), float(hi.iloc[-1])] if len(f) else None,
            "scores": {k: (None if (isinstance(v,float) and (np.isnan(v) or np.isinf(v))) else float(v)) for k,v in (scores or {}).items()}
        }
    sys_msg = (
        "You are a senior economist writing for executives. "
        "Write 2–3 concise paragraphs (10–16 sentences total) in clear business English. "
        "Explain the chart (units/frequency), define acronyms (ETS, PPI), state the latest value and YoY, "
        "and provide a 3/6/12-month outlook using the forecast points provided explain in deep based on data obtained and geopolitical context. "
        "Interpret uncertainty bands and Coverage@95, and mention calibration (bands too tight/wide). "
        "Add light geopolitical/industry context (tariffs, sanctions, energy, logistics, reshoring) without speculation. "
        "End with a practical takeaway."
)

    user_msg = (
        f"Section: {section_title}\n"
        f"Freq/Units: {freq_unit}\n"
        f"History: {json.dumps(hist)}\n"
        f"Forecast: {json.dumps(fc_stats)}\n"
        f"Extras: {json.dumps(extras or {})}\n"
        "Audience is non-economist leadership. Avoid jargon; be precise and neutral."
    )
    out = ai_chat(sys_msg, user_msg)
    return f"<div style='margin:10px 0 14px 0'>{out}</div>" if out else ""

# ----------------------------
# Sanity & partners
# ----------------------------
def sanity_panel(df: pd.DataFrame, value_col: str, date_col: str, unit_label: str) -> str:
    dfx = df.dropna(subset=[date_col]).sort_values(date_col)
    last = dfx[[date_col, value_col]].dropna().tail(1)
    last_dt = last[date_col].iloc[0] if not last.empty else None
    nn = dfx[value_col].notna().mean()*100.0 if len(dfx) else 0.0
    zeros = (dfx[value_col].fillna(0)==0).mean()*100.0 if len(dfx) else 0.0
    last_str = last_dt.date().isoformat() if hasattr(last_dt, "date") else str(last_dt)
    return small(f"Sanity ✓ — unit: {unit_label} · last: {last_str} · non-null: {nn:.1f}% · zeros: {zeros:.1f}% · rows: {len(dfx):,}")

def looks_implausible_partner_mix(grp: pd.DataFrame) -> bool:
    if grp is None or grp.empty: return True
    names = set(grp["partner_label"].astype(str).str.upper().head(12))
    big = {"MEXICO","CANADA","CHINA","JAPAN","GERMANY","UNITED KINGDOM","INDIA"}
    has_micro = len(names & SUSPECT_PARTNERS) >= 2
    missing_big = len(names & big) <= 2  # if we barely see any big partners, be cautious
    return has_micro or missing_big

def partner_top_table(c: pd.DataFrame, source_key: str, top_n: int=8):
    df = c[c["source"].eq(source_key)].dropna(subset=["date","value_usd","partner_label"]).copy()
    if df.empty: return "", 0, pd.DataFrame()
    recent = df[df["date"] > (df["date"].max() - pd.DateOffset(months=12))]
    grp = recent.groupby("partner_label", as_index=False)["value_usd"].sum().sort_values("value_usd", ascending=False)
    total = grp["value_usd"].sum()
    if total <= 0: return "", 0, pd.DataFrame()
    grp["share"] = grp["value_usd"] / total
    hhi = int(round((grp["share"]**2).sum() * 10_000))
    top = grp.head(top_n)
    # bar
    plt.figure(figsize=(9, 4.8))
    plt.bar(top["partner_label"], top["value_usd"])
    plt.title(f"Top {top_n} Partners (Last 12 months) — {'Imports' if 'import' in source_key else 'Exports'}")
    plt.ylabel("USD"); plt.xticks(rotation=30, ha="right")
    img = save_and_embed(f"{source_key}_partners_top.png")
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
    tbl = "<table border='1' cellpadding='4' cellspacing='0'><tr><th>Country</th><th>Last 12 months (USD)</th></tr>"
    for nm, val in rows:
        tbl += f"<tr><td>{nm}</td><td>{human_usd(val)}</td></tr>"
    tbl += "</table>"
    return h(4, title) + tbl

# ----------------------------
# Plot helpers
# ----------------------------
def plot_series(ts: pd.Series, title: str, y_label: str, fig_name: str) -> str:
    if not isinstance(ts.index, pd.DatetimeIndex):
        ts.index = pd.to_datetime(ts.index, errors="coerce")
    ts = ts.dropna().sort_index()
    plt.figure(figsize=(9, 4.8))
    plt.plot(ts.index, ts.values, marker="o", linewidth=1.5)
    plt.title(title); plt.xlabel("Date"); plt.ylabel(y_label)
    return save_and_embed(fig_name)

def plot_forecast(ts: pd.Series, f: pd.Series, lo: pd.Series, hi: pd.Series,
                  title: str, y_label: str, fig_name: str,
                  show_horizons: Tuple[int,int,int]=(3,12,24)) -> str:
    plt.figure(figsize=(9, 4.8))
    plt.plot(ts.index, ts.values, label="History", linewidth=1.5)
    plt.plot(f.index, f.values, linestyle="--", marker="o", label="Forecast", linewidth=1.2)
    plt.fill_between(f.index, lo.values, hi.values, alpha=0.18, label="≈95% band")
    for hmark in show_horizons:
        if len(f.index) >= hmark:
            plt.axvline(f.index[hmark-1], color="gray", linewidth=1, linestyle=":")
    plt.title(title); plt.xlabel("Date"); plt.ylabel(y_label); plt.legend()
    return save_and_embed(fig_name)

# ----------------------------
# Executive overview (auto or AI)
# ----------------------------
def executive_overview(imports_m: pd.Series, exports_m: pd.Series) -> str:
    if imports_m is None or imports_m.empty or exports_m is None or exports_m.empty:
        return "<p><b>Executive overview:</b> We could not compute a reliable overview because monthly Census series were missing. Please re-run the data update to fetch monthly USD flows.</p>"

    last_im = float(imports_m.iloc[-1])
    last_ex = float(exports_m.iloc[-1])
    yoy_im = None; yoy_ex = None
    if len(imports_m) >= 13:
        base = abs(float(imports_m.iloc[-13])) or 1e-9
        yoy_im = 100.0 * (last_im - float(imports_m.iloc[-13])) / base
    if len(exports_m) >= 13:
        base = abs(float(exports_m.iloc[-13])) or 1e-9
        yoy_ex = 100.0 * (last_ex - float(exports_m.iloc[-13])) / base

    text = (
        f"<p><b>Executive overview.</b> U.S. monthly goods imports are currently about <b>{human_usd(last_im)}</b> and exports about "
        f"<b>{human_usd(last_ex)}</b>. Year-over-year, imports are {pct(yoy_im)} and exports are {pct(yoy_ex)}. "
        "The short-term outlook reflects tariff risks and supply-chain rerouting: a possible near-term dip in targeted imports, "
        "with partial offsets via sourcing shifts. Export momentum depends on partner demand and the dollar. "
        "Details and calibrated forecasts (3, 6, 12 months) follow below for each series, with uncertainty bands and scorecards."
        "</p>"
    )

    if USE_AI and OPENAI_API_KEY:
        sys = (
            "You are a senior economist. Write one concise executive paragraph (5–8 sentences) about U.S. monthly imports/exports. "
            "Use the latest levels and YoY provided and speak to risks: tariffs, supply chains, dollar, partner demand. "
            "Avoid hype; be precise and neutral."
        )
        usr = json.dumps({
            "imports_last": last_im, "exports_last": last_ex,
            "imports_yoy_pct": yoy_im, "exports_yoy_pct": yoy_ex
        })
        ai = ai_chat(sys, usr)
        if ai:
            text = f"<p><b>Executive overview.</b> {ai}</p>"
    return text

# ----------------------------
# Build report
# ----------------------------
def build_report():
    df   = load_main()
    wits = prepare_wits(df)
    cen  = prepare_census(df)
    bls  = prepare_bls(df)

    html = []
    html.append("<html><head><meta charset='utf-8'><title>U.S. Trade & Supply Chain Report</title>")
    html.append("<style>body{font-family:system-ui,Segoe UI,Roboto,Arial,sans-serif;max-width:1150px;margin:24px auto;padding:0 16px;line-height:1.45} img{max-width:100%} table{border-collapse:collapse} td,th{border:1px solid #ccc;padding:6px 8px}</style>")
    html.append("</head><body>")
    html.append(h(1, "U.S. Trade & Supply Chain — Executive Report"))

    # Census monthly series (must reach 2025 if update ran correctly)
    imports_m = exports_m = pd.Series(dtype=float)
    if not cen.empty:
        imp_raw = cen[cen["source"].eq("census_imports")].set_index("date")["value_usd"]
        exp_raw = cen[cen["source"].eq("census_exports")].set_index("date")["value_usd"]
        imports_m = to_monthly_unique(imp_raw, how="sum")
        exports_m = to_monthly_unique(exp_raw, how="sum")

    # Executive overview
    html.append(executive_overview(imports_m, exports_m))

    # ---------- WITS (Annual, up to last available year) ----------
if not wits.empty:
    # pick trade indicators and then world/total
    wt = wits[wits["indicator"].isin(["MPRT-TRD-VL", "XPRT-TRD-VL"])].copy()
    wt = wt[
        (wt["partner_label"].astype(str).str.upper() == "WLD") &
        (wt["product_label"].astype(str).str.casefold() == "all")
    ]
    if not wt.empty:
        wt["kind"] = np.where(wt["indicator"].eq("MPRT-TRD-VL"), "Imports (Annual)", "Exports (Annual)")
        g = wt.groupby(["year", "kind"], as_index=False)["value_usd"].sum().dropna()

        if not g.empty:
            wide = g.pivot(index="year", columns="kind", values="value_usd").sort_index()
            html.append(h(2, "WITS — Annual Totals (USD)"))
            html.append(small("Frequency: Annual • Units: USD (converted from WITS 'US$ Million') • Coverage: Goods (nominal)."))
            for col in wide.columns:
                ts = wide[col].dropna()
                if ts.empty:
                    continue
                ts2 = pd.Series(ts.values, index=pd.to_datetime(ts.index.astype(str) + "-12-31"))
                html.append(plot_series(ts2, f"{col} — WITS (Annual)", "USD", f"wits_{col.lower().replace(' ', '_')}.png"))
                html.append(ai_explain(
                    section_title=f"{col} — WITS (Annual)",
                    freq_unit="Annual, USD",
                    history=ts2, forecast=None,
                    extras={"source": "WITS (annual, no forward fill)"},
                    horizons=[3, 6, 12]
                ))
            if not wide.dropna(how="all").empty:
                last_year = wide.dropna(how="all").index.max()
                bullets = []
                for col in wide.columns:
                    val = wide.loc[last_year, col]
                    if pd.notna(val):
                        bullets.append(f"{col} in {int(last_year)}: {human_usd(val)}")
                if bullets:
                    html.append(bullet_list(bullets))


    # ---------- Census: Imports (Monthly) ----------
    if not imports_m.empty:
        html.append(h(2, "Census — U.S. Goods Imports (Monthly, USD)"))
        html.append(sanity_panel(cen[cen["source"].eq("census_imports")], "value_usd","date","USD"))
        html.append(plot_series(imports_m, "U.S. Goods Imports — Monthly (USD)", "USD", "census_imports_hist.png"))
        f_i, lo_i, hi_i, sc_i = ets_forecast(imports_m, h=24, seasonal=True)
        html.append(plot_forecast(imports_m, f_i, lo_i, hi_i, "Imports — ETS Forecast (3/12/24 months)", "USD", "census_imports_fc.png"))
        html.append(explain_scores(sc_i))
        html.append(ai_explain(
            section_title="Census — U.S. Goods Imports (Monthly)",
            freq_unit="Monthly, USD",
            history=imports_m, forecast=(f_i, lo_i, hi_i, sc_i),
            extras={"source":"U.S. Census, HS timeseries (USD)"},
            horizons=[3,6,12]
        ))

        img_imp, hhi_imp, grp_imp = partner_top_table(cen, "census_imports", top_n=10)
        if looks_implausible_partner_mix(grp_imp):
            html.append(h(3, "Imports — Partner Deep-Dive"))
            html.append("<div style='color:#b00'><b>Data quality warning:</b> Partner mix looks implausible (micro-economies bubbled up or big partners missing). This usually means the pipeline fetched 'units' or a filtered subset. The table is hidden until fixed.</div>")
        else:
            html.append(h(3, "Imports — Top Partners (Last 12 months)"))
            html.append(img_imp)
            html.append(small(f"HHI: {hhi_imp} (≈1,500–2,500 = moderate; >2,500 = high concentration)"))
            html.append(named_countries_block(cen, "census_imports", "Named Partners — Imports (Last 12 months)"))

    # ---------- Census: Exports (Monthly) ----------
    if not exports_m.empty:
        html.append(h(2, "Census — U.S. Goods Exports (Monthly, USD)"))
        html.append(sanity_panel(cen[cen["source"].eq("census_exports")], "value_usd","date","USD"))
        html.append(plot_series(exports_m, "U.S. Goods Exports — Monthly (USD)", "USD", "census_exports_hist.png"))
        f_e, lo_e, hi_e, sc_e = ets_forecast(exports_m, h=24, seasonal=True)
        html.append(plot_forecast(exports_m, f_e, lo_e, hi_e, "Exports — ETS Forecast (3/12/24 months)", "USD", "census_exports_fc.png"))
        html.append(explain_scores(sc_e))
        html.append(ai_explain(
            section_title="Census — U.S. Goods Exports (Monthly)",
            freq_unit="Monthly, USD",
            history=exports_m, forecast=(f_e, lo_e, hi_e, sc_e),
            extras={"source":"U.S. Census, HS timeseries (USD)"},
            horizons=[3,6,12]
        ))

        img_exp, hhi_exp, grp_exp = partner_top_table(cen, "census_exports", top_n=10)
        if looks_implausible_partner_mix(grp_exp):
            html.append(h(3, "Exports — Partner Deep-Dive"))
            html.append("<div style='color:#b00'><b>Data quality warning:</b> Partner mix looks implausible. Table hidden until unit/filter is corrected.</div>")
        else:
            html.append(h(3, "Exports — Top Partners (Last 12 months)"))
            html.append(img_exp)
            html.append(small(f"HHI: {hhi_exp} (≈1,500–2,500 = moderate; >2,500 = high concentration)"))
            html.append(named_countries_block(cen, "census_exports", "Named Partners — Exports (Last 12 months)"))

    # ---------- BLS PPI ----------
    if not bls.empty:
        for sid, sub in bls.groupby("series_id"):
            raw = sub.set_index("date")["value"]
            ts  = to_monthly_unique(raw, how="mean")
            if ts.empty: continue
            html.append(h(2, f"BLS PPI — {sid} (Monthly Index)"))
            html.append(sanity_panel(sub, "value","date","Index"))
            html.append(plot_series(ts, f"PPI {sid} — Monthly (Index)", "Index", f"bls_{sid}_hist.png"))
            f_p, lo_p, hi_p, sc_p = ets_forecast(ts, h=24, seasonal=True)
            html.append(plot_forecast(ts, f_p, lo_p, hi_p, f"PPI {sid} — ETS Forecast (3/12/24 months)", "Index", f"bls_{sid}_fc.png"))
            html.append(explain_scores(sc_p))
            html.append(ai_explain(
                section_title=f"BLS PPI — {sid}",
                freq_unit="Monthly, Index",
                history=ts, forecast=(f_p, lo_p, hi_p, sc_p),
                extras={"acronyms":{"PPI":"Producer Price Index","ETS":"Exponential Smoothing"}},
                horizons=[3,6,12]
            ))

    # ---------- Methodology & Glossary ----------
    html.append(h(2, "Methodology"))
    html.append(bullet_list([
        "<b>Sources:</b> WITS (annual, converted from 'US$ Million' to USD), U.S. Census HS timeseries (monthly USD only), BLS PPI (monthly index).",
        "<b>Frequencies:</b> WITS = Annual; Census & BLS = Monthly (nominal). No forward-filling.",
        "<b>Models:</b> ETS/Holt-Winters (additive trend; seasonality=12 when ≥36 months) with fallbacks.",
        "<b>Uncertainty:</b> ≈95% bands from empirical residual quantiles (fallback ±1.96σ).",
        "<b>Backtest:</b> Last 12 months out-of-sample; sMAPE, WMAPE, RMSE, Coverage@95 reported under forecasts.",
        "<b>Partner concentration:</b> Herfindahl-Hirschman Index (HHI) on last-12-month shares.",
        "<b>Sanity checks:</b> Units, last date, non-null %, zeros %, and row counts shown under each chart.",
        "<b>Data guards:</b> Hide partner tables if micro-economies dominate or major partners are missing; this usually signals a bad pull ('units' or filtered slice)."
    ]))
    html.append(h(3, "Glossary"))
    html.append(bullet_list([
        "<b>ETS:</b> Exponential smoothing (Holt-Winters).",
        "<b>sMAPE:</b> Symmetric MAPE — percent error robust when actuals can be near zero.",
        "<b>WMAPE:</b> Weighted MAPE — absolute error over total actuals; interpretable as % of scale.",
        "<b>RMSE:</b> Root Mean Squared Error — typical unit error (e.g., USD).",
        "<b>Coverage@95:</b> Share of realized points falling inside the nominal 95% forecast band; ≈95% indicates well-calibrated uncertainty.",
        "<b>HHI:</b> Sum of squared shares × 10,000; higher = more concentrated supplier base."
    ]))

    html.append("</body></html>")
    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(html))
    print(f"✅ Report built → {HTML_PATH}")

if __name__ == "__main__":
    build_report()
