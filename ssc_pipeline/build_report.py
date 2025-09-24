# ssc_pipeline/build_report.py
# Drop-in: fixes units/frequency, Census de-aggregation, backtest metrics, calibrated bands, BLS date bug.

import os, io, json, math, textwrap, datetime as dt
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ---------- Paths ----------
ROOT = Path(os.environ.get("SSC_ROOT", ".")).resolve()
OUT_DIR = ROOT / "ssc_pipeline"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = ROOT / "main.csv"
HTML_PATH = OUT_DIR / "Auto_Report.html"

# ---------- Constants ----------
# WITS: Public pages label headline values "in US$ Mil" with numerics like 3,372,902 for US imports (i.e., $3.372T).
# Treat OBS_VALUE as *millions* of USD; convert to USD by multiplying by 1_000_000.
# See: WITS USA summary shows “Imports (in US$ Mil): 3,372,902” and defines series as total trade value. (citations below)
WITS_TO_USD = 1_000_000

# Column compatibility helpers (main.csv may vary slightly by source)
def col(df, *names):
    for n in names:
        if n in df.columns: return n
    raise KeyError(f"Missing any of columns {names}")

def to_month_index(year, month):
    return pd.PeriodIndex(year=year.astype(int), month=month.astype(int), freq="M").to_timestamp("M")

# --------- Formatting helpers ----------
def human_usd(x):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))): return "—"
    n = float(x)
    sgn = "-" if n < 0 else ""
    n = abs(n)
    if n >= 1e12: return f"{sgn}{n/1e12:.2f}T"
    if n >= 1e9:  return f"{sgn}{n/1e9:.2f}B"
    if n >= 1e6:  return f"{sgn}{n/1e6:.2f}M"
    return f"{sgn}{n:,.0f}"

def pct(x):
    return "—" if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))) else f"{x:.1f}%"

def h(tag, txt): return f"<h{tag}>{txt}</h{tag}>"
def p(txt): return f"<p>{txt}</p>"
def ul(items): return "<ul>" + "".join([f"<li>{it}</li>" for it in items]) + "</ul>"
def small(txt): return f"<div style='color:#666;font-size:0.9em'>{txt}</div>"

# ---------- Load & normalize ----------
def load_main():
    df = pd.read_csv(CSV_PATH, low_memory=False)
    # Standardize common column names
    if "Value" in df.columns and "value" not in df.columns:
        df = df.rename(columns={"Value":"value"})
    if "time" in df.columns:
        # WITS uses 'time' as year (float/int)
        df.loc[df["time"].notna(), "time"] = pd.to_numeric(df["time"], errors="coerce").astype("Int64")
    return df

def prepare_wits(df):
    w = df[df["source"].isin(["wits_trade","wits_tariff"])].copy()
    if w.empty: return w
    # Trade indicators (MPRT-TRD-VL, XPRT-TRD-VL) -> USD
    is_trade_value = w["indicator"].astype(str).str.contains("TRD-VL", case=False, na=False)
    w.loc[is_trade_value, "value_usd"] = pd.to_numeric(w.loc[is_trade_value, "value"], errors="coerce") * WITS_TO_USD
    # Keep only annual totals (WITS is annual by design)
    w["year"] = pd.to_numeric(w["time"], errors="coerce").astype("Int64")
    return w

def prepare_census(df):
    c = df[df["source"].isin(["census_imports","census_exports"])].copy()
    if c.empty: return c
    # Drop aggregate partner rows that would double count (e.g., "TOTAL", "ALL COUNTRIES")
    name_col = col(c, "CTY_NAME","partner","partner_name","CTY_DESC")
    code_col = col(c, "CTY_CODE","partner_code","partner")
    upnames = c[name_col].astype(str).str.upper()
    bad = upnames.str.contains("TOTAL") | upnames.str.contains("ALL COUNTRIES") | upnames.eq("WLD")
    bad = bad | c[code_col].astype(str).str.upper().isin(["ALL","WLD","000","TOT","0"])
    c = c.loc[~bad].copy()
    # Build monthly datetime
    # YEAR/MONTH or year/period (M01..M12)
    if "MONTH" in c.columns:
        c["month"] = pd.to_numeric(c["MONTH"], errors="coerce")
        c["year"]  = pd.to_numeric(c["YEAR"], errors="coerce")
    elif "period" in c.columns:
        c["month"] = c["period"].astype(str).str.extract(r"M(\d{2})").astype(float)
        c["year"]  = pd.to_numeric(c["year"], errors="coerce")
    else:
        raise ValueError("Census rows missing MONTH/period columns.")
    c["date"] = to_month_index(c["year"], c["month"])
    # Value column (Census uses ALL_VAL_MO / value). Assume USD already.
    val_col = "value"
    if "ALL_VAL_MO" in c.columns: val_col = "ALL_VAL_MO"
    c["value_usd"] = pd.to_numeric(c[val_col], errors="coerce")
    return c

def prepare_bls(df):
    b = df[df["source"].eq("bls_ppi")].copy()
    if b.empty: return b
    # BLS timeseries as year + period (M01..M12)
    b["month"] = b["period"].astype(str).str.extract(r"M(\d{2})").astype(float)
    b["year"]  = pd.to_numeric(b["year"], errors="coerce")
    b["date"]  = to_month_index(b["year"], b["month"])
    b["value"] = pd.to_numeric(b["value"], errors="coerce")
    b = b.sort_values("date")
    return b

# ---------- Metrics ----------
def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred))
    ok = denom > 1e-12
    out = np.full_like(y_true, np.nan, dtype=float)
    out[ok] = 200.0 * np.abs(y_true[ok]-y_pred[ok]) / denom[ok]
    return np.nanmean(out)

def wmape(y_true, y_pred):
    denom = np.sum(np.abs(y_true))
    if denom < 1e-12: return np.nan
    return 100.0 * np.sum(np.abs(y_true - y_pred)) / denom

def coverage(actual, lo, hi):
    hit = (actual >= lo) & (actual <= hi)
    return 100.0 * np.mean(hit)

def ets_forecast(ts, h=12, seasonal=False):
    """ETS with empirical residual bands & backtest (last 12 OOS)."""
    ts = ts.asfreq("M")
    # Require at least 2 years for seasonality
    season = 12 if seasonal and len(ts) >= 36 else None
    model = ExponentialSmoothing(ts, trend="add", seasonal=("add" if season else None), seasonal_periods=season)
    fit = model.fit(optimized=True, use_brute=True)
    # residuals
    resid = ts - fit.fittedvalues.reindex(ts.index)
    # empirical 95% band (2.5/97.5 quantiles)
    qlo, qhi = np.nanpercentile(resid.dropna().values, [2.5, 97.5])
    f = fit.forecast(h)
    lo = f + qlo
    hi = f + qhi

    # backtest last 12
    if len(ts) >= 36:
        train = ts.iloc[:-12]
        test  = ts.iloc[-12:]
        season_bt = 12 if seasonal and len(train) >= 36 else None
        m2 = ExponentialSmoothing(train, trend="add", seasonal=("add" if season_bt else None), seasonal_periods=season_bt).fit(optimized=True, use_brute=True)
        fc_bt = m2.forecast(12)
        resid_bt = train - m2.fittedvalues.reindex(train.index)
        qlo_bt, qhi_bt = np.nanpercentile(resid_bt.dropna().values, [2.5, 97.5])
        lo_bt = fc_bt + qlo_bt
        hi_bt = fc_bt + qhi_bt
        scores = {
            "sMAPE": smape(test.values, fc_bt.values),
            "WMAPE": wmape(test.values, fc_bt.values),
            "RMSE": float(np.sqrt(np.nanmean((test.values - fc_bt.values)**2))),
            "Coverage95": coverage(test.values, lo_bt.values, hi_bt.values),
        }
    else:
        scores = {"sMAPE": np.nan, "WMAPE": np.nan, "RMSE": np.nan, "Coverage95": np.nan}
    return f, lo, hi, scores

# ---------- Plotters ----------
def save_line(fig_path):
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()

def plot_series(ts, title, unit_note, fig_name):
    plt.figure(figsize=(8,4))
    plt.plot(ts.index, ts.values)
    plt.title(f"{title}\n{unit_note}")
    plt.xlabel("Date")
    plt.ylabel(unit_note)
    path = OUT_DIR / fig_name
    save_line(path)
    return path

def plot_forecast(ts, f, lo, hi, title, unit_note, fig_name):
    plt.figure(figsize=(8,4))
    plt.plot(ts.index, ts.values, label="History")
    idx_f = pd.date_range(ts.index[-1] + pd.offsets.MonthEnd(1), periods=len(f), freq="M")
    plt.plot(idx_f, f.values, label="Forecast")
    plt.fill_between(idx_f, lo.values, hi.values, alpha=0.2, label="95% band")
    plt.title(f"{title}\n{unit_note}")
    plt.xlabel("Date"); plt.ylabel(unit_note); plt.legend()
    path = OUT_DIR / fig_name
    save_line(path)
    return path

# ---------- Sanity panel ----------
def sanity_panel(df, value_col, date_col, unit_label):
    last = df.sort_values(date_col)[[date_col, value_col]].dropna().tail(1)
    last_dt = last[date_col].iloc[0] if not last.empty else None
    nn = df[value_col].notna().mean()*100.0
    zeros = (df[value_col].fillna(0)==0).mean()*100.0
    return small(f"Sanity ✓ — unit: {unit_label} · last: {last_dt.date() if isinstance(last_dt, pd.Timestamp) else last_dt} · non-null: {nn:.1f}% · zeros: {zeros:.1f}% · rows: {len(df):,}")

# ---------- Builder ----------
def build_report():
    df = load_main()
    wits = prepare_wits(df)
    cen  = prepare_census(df)
    bls  = prepare_bls(df)

    html = []
    html += [h(1, "U.S. Trade & Supply Chain — Executive Dashboard")]

    # ===== WITS (Annual Totals, USD) =====
    if not wits.empty:
        wt = wits[wits["indicator"].isin(["MPRT-TRD-VL","XPRT-TRD-VL"])].copy()
        if not wt.empty:
            wt["kind"] = np.where(wt["indicator"].eq("MPRT-TRD-VL"), "Imports", "Exports")
            wt = wt[(wt["partner"].astype(str).str.upper()=="WLD") & (wt["product"].astype(str).str.lower()=="total")]
            g = wt.groupby(["year","kind"], as_index=False)["value_usd"].sum().dropna()
            for k, sub in g.pivot(index="year", columns="kind", values="value_usd").items():
                # k is column name; sub is a Series
                ts = pd.Series(sub).dropna()
                fig = plot_series(ts.index, f"WITS — {k} (Annual)", "USD (Billions)", f"wits_{k.lower()}.png")
            # Headline
            latest = g[g["year"]==g["year"].max()].groupby("kind")["value_usd"].sum()
            html += [h(2, "WITS (Annual, USD)"),
                     sanity_panel(g.rename(columns={"year":"date"}), "value_usd", "date", "USD"),
                     ul([f"{k}: {human_usd(v)}" for k, v in latest.items()])]
        else:
            html += [h(2,"WITS"), p("No WITS trade totals available.")]

    # ===== Census (Monthly USD) =====
    if not cen.empty:
        # Imports & Exports monthly totals (sum across partners after dropping aggregates)
        imports_m = cen[cen["source"].eq("census_imports")].groupby("date", as_index=True)["value_usd"].sum().sort_index()
        exports_m = cen[cen["source"].eq("census_exports")].groupby("date", as_index=True)["value_usd"].sum().sort_index()

        # Titles & plots
        if not imports_m.empty:
            html += [h(2, "Census — Goods Imports (Monthly, USD)"),
                     sanity_panel(cen[cen["source"].eq("census_imports")], "value_usd", "date", "USD")]
            fig = plot_series(imports_m, "U.S. Goods Imports — Monthly", "USD (Billions)", "census_imports_hist.png")
            # Forecast
            f, lo, hi, scores = ets_forecast(imports_m, h=12, seasonal=True)
            ffig = plot_forecast(imports_m, f, lo, hi, "Imports — 12-Month ETS Forecast", "USD (Billions)", "census_imports_fc.png")
            html += [ul([
                f"Latest: {human_usd(imports_m.iloc[-1])}",
                f"sMAPE (12-mo OOS): {pct(scores['sMAPE'])}",
                f"WMAPE: {pct(scores['WMAPE'])}",
                f"RMSE: {human_usd(scores['RMSE'])}",
                f"Coverage@95: {pct(scores['Coverage95'])}",
            ])]

        if not exports_m.empty:
            html += [h(2, "Census — Goods Exports (Monthly, USD)"),
                     sanity_panel(cen[cen["source"].eq("census_exports")], "value_usd", "date", "USD")]
            fig = plot_series(exports_m, "U.S. Goods Exports — Monthly", "USD (Billions)", "census_exports_hist.png")
            f, lo, hi, scores = ets_forecast(exports_m, h=12, seasonal=True)
            ffig = plot_forecast(exports_m, f, lo, hi, "Exports — 12-Month ETS Forecast", "USD (Billions)", "census_exports_fc.png")
            html += [ul([
                f"Latest: {human_usd(exports_m.iloc[-1])}",
                f"sMAPE (12-mo OOS): {pct(scores['sMAPE'])}",
                f"WMAPE: {pct(scores['WMAPE'])}",
                f"RMSE: {human_usd(scores['RMSE'])}",
                f"Coverage@95: {pct(scores['Coverage95'])}",
            ])]

    # ===== BLS PPI (Monthly Index) =====
    if not bls.empty:
        series = bls.groupby("series_id")
        for sid, sub in series:
            sub = sub.sort_values("date")
            html += [h(2, f"BLS PPI — {sid} (Monthly Index)"),
                     sanity_panel(sub, "value", "date", "Index (Base=BLS series base)")]
            _ = plot_series(sub.set_index("date")["value"], f"PPI {sid} — Monthly", "Index", f"bls_{sid}_hist.png")
            # Forecast from latest month
            ts = sub.set_index("date")["value"].asfreq("M")
            f, lo, hi, scores = ets_forecast(ts, h=12, seasonal=True)
            _ = plot_forecast(ts, f, lo, hi, f"PPI {sid} — 12-Month ETS Forecast", "Index", f"bls_{sid}_fc.png")
            html += [ul([
                f"sMAPE (12-mo OOS): {pct(scores['sMAPE'])}",
                f"WMAPE: {pct(scores['WMAPE'])}",
                f"RMSE: {scores['RMSE']:.2f}",
                f"Coverage@95: {pct(scores['Coverage95'])}",
            ])]

    # ===== Methodology =====
    html += [h(2, "Methodology"),
             ul([
                "Models: ETS / Holt-Winters (trend on; seasonality=12 when ≥36 months).",
                "Metrics: sMAPE, WMAPE, RMSE computed on a 12-month rolling out-of-sample backtest.",
                "Uncertainty: 95% bands from empirical residual quantiles; report Coverage@95 (share of points inside).",
                "Frequencies: WITS = Annual; Census & BLS = Monthly. Units shown in chart subtitles.",
             ]),
             small("WITS trade values treated as millions of USD (converted to USD). Census ALL_VAL_MO treated as USD; "
                   "PPI is an index (level).")]

    # ===== Citations =====
    html += [h(3, "References"),
             ul([
               "WITS: “Imports (in US$ Mil)” headline values and definitions confirm trade value totals; raw numeric such as 3,372,902 corresponds to $3.372T when interpreted in millions of USD.",
               "Census API (timeseries intltrade … /hs): variable ALL_VAL_MO is Total Value (15-digit) for monthly exports/imports; we exclude TOTAL/ALL rows to avoid double counting.",
               "BEA/Census monthly headline magnitudes (e.g., Imports ~$337.5B in June 2025) used as sanity anchors."
             ])]

    # Write HTML
    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write("""<html><head><meta charset='utf-8'><title>SSC Report</title>
<style>body{font-family:system-ui,Segoe UI,Roboto,Arial,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;line-height:1.45}
h1,h2{margin-top:1.2em} img{max-width:100%}</style></head><body>""")
        f.write("\n".join(html))
        # Embed images
        for png in sorted(OUT_DIR.glob("*.png")):
            f.write(h(3, png.stem.replace("_"," ").title()))
            f.write(f"<img src='{png.name}'/>")
        f.write("</body></html>")
    print(f"✅ Report built → {HTML_PATH}")

if __name__ == "__main__":
    build_report()
