# ssc_pipeline/ssc_pipeline/sources.py
# Robust data sources for SSC pipeline
# - WITS SDMX (annual): imports/exports totals, MFN tariffs
# - Census HS timeseries (monthly, USD) by partner
# - BLS PPI monthly index
# - Retries, backoff, and safer parsing
# - Drops "Total for all countries" at source level
# - Scales WITS trade values to USD here (so do NOT scale again downstream)

import os
import time
import datetime as dt
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
import requests

# ---------------------------------------------------------------------
# HTTP helpers with retry/backoff
# ---------------------------------------------------------------------
DEF_TIMEOUT = 60
MAX_RETRIES = 4
BACKOFF_SEC = 1.5

def _get(url: str, params=None, headers=None, timeout=DEF_TIMEOUT):
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            # Retry on 429/5xx
            if 500 <= r.status_code < 600 or r.status_code == 429:
                raise requests.HTTPError(f"{r.status_code} {r.reason}", response=r)
            r.raise_for_status()
            return r
        except Exception as e:
            last_err = e
            time.sleep(BACKOFF_SEC * attempt)
    if last_err:
        raise last_err

def _get_json(url: str, params=None, headers=None, timeout=DEF_TIMEOUT):
    r = _get(url, params=params, headers=headers, timeout=timeout)
    try:
        return r.json()
    except Exception as e:
        snippet = (r.text or "")[:400]
        raise ValueError(f"JSON parse error for {url} | params={params} | snippet={snippet}") from e

# ---------------------------------------------------------------------
# WITS (World Bank) SDMX v2.1
# Notes:
#   - Trade values are typically reported in **thousands of USD**.
#   - We convert to **USD** here using WITS_TRADE_SCALE (default 1_000.0).
#   - If your manual cross-check shows the feed is in **millions** instead,
#     set env WITS_TRADE_SCALE=1000000.
# ---------------------------------------------------------------------
WITS_TRADE_SCALE = float(os.environ.get("WITS_TRADE_SCALE", "1000"))

def _wits_sdmx_to_rows(js: dict, source: str) -> List[dict]:
    """Convert WITS SDMX JSON to flat rows with annual observations."""
    try:
        series_dict = js["data"]["dataSets"][0]["series"]
        dims = js["data"]["structure"]["dimensions"]["series"]
        time_dim = js["data"]["structure"]["dimensions"]["observation"][0]["values"]
    except KeyError as e:
        raise ValueError("WITS SDMX shape changed") from e

    dim_names = [d["id"] for d in dims]
    out = []
    for s_key, s_val in series_dict.items():
        idx = list(map(int, s_key.split(":")))
        meta = {dim_names[i].lower(): dims[i]["values"][idx[i]]["id"] for i in range(len(idx))}
        for obs_idx, obs in s_val.get("observations", {}).items():
            t = int(obs_idx)
            year = int(time_dim[t]["id"])
            val = float(obs[0])
            out.append({
                "freq": "A",
                "reporter": meta.get("reporter", "USA"),
                "partner":  meta.get("partner", "WLD"),
                "product":  meta.get("product", "all"),
                "indicator": meta.get("indicator", ""),
                "time": str(year),
                "Value": val,    # raw numeric; we will scale for trade below
                "source": source,
            })
    return out

def wits_trade_data() -> pd.DataFrame:
    """
    Annual US↔World totals, goods imports/exports, in USD after scaling.
    """
    base = "https://wits.worldbank.org/API/V1/SDMX/V21/rest/data/df_wits_tradestats_trade"
    keys = [
        "A.USA.WLD.all.MPRT-TRD-VL",  # imports
        "A.USA.WLD.all.XPRT-TRD-VL",  # exports
    ]
    frames = []
    for key in keys:
        url = f"{base}/{key}/?format=JSON"
        js = _get_json(url)
        rows = _wits_sdmx_to_rows(js, "wits_trade")
        frames.extend(rows)

    df = pd.DataFrame(frames)
    if df.empty:
        return df

    # Scale trade values (TRD-VL) to USD; keep tariffs as-is (not in this function)
    is_trade = df["indicator"].astype(str).str.contains("TRD-VL", case=False, na=False)
    df.loc[is_trade, "Value"] = pd.to_numeric(df.loc[is_trade, "Value"], errors="coerce") * WITS_TRADE_SCALE

    # Minimal cleaning
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df.dropna(subset=["Value"])
    return df.reset_index(drop=True)

def wits_tariff_data() -> pd.DataFrame:
    """
    Annual US MFN tariffs (percent).
    """
    base = "https://wits.worldbank.org/API/V1/SDMX/V21/rest/data/df_wits_tariff"
    keys = [
        "A.USA.WLD.all.MFN-WGHTD-AVRG",  # weighted average
        "A.USA.WLD.all.MFN-SMPL-AVRG",   # simple average
    ]
    frames = []
    for key in keys:
        url = f"{base}/{key}/?format=JSON"
        js = _get_json(url)
        rows = _wits_sdmx_to_rows(js, "wits_tariff")
        frames.extend(rows)

    df = pd.DataFrame(frames)
    if df.empty:
        return df
    # Tariffs are already in percent; no scaling
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df.dropna(subset=["Value"])
    return df.reset_index(drop=True)

# ---------------------------------------------------------------------
# US Census (Timeseries International Trade, HS)
# Goal:
#   - Pull monthly **USD** values by partner for imports and exports.
#   - Keep YEAR, MONTH, CTY_CODE, CTY_NAME, ALL_VAL_MO.
#   - Drop totals/world rows at source to avoid bogus “top partners”.
# Notes:
#   - If your manual cross-check shows values are in thousands, set
#     env CENSUS_VALUE_SCALE=1000 (default = 1).
# ---------------------------------------------------------------------
CENSUS_VALUE_SCALE = float(os.environ.get("CENSUS_VALUE_SCALE", "1"))
CENSUS_API_KEY = os.environ.get("CENSUS_API_KEY", "")

def _census_timeseries(side: str, y0: int = 2018, y1: Optional[int] = None) -> pd.DataFrame:
    """
    side: 'imports' or 'exports'
    Returns monthly rows by country with ALL_VAL_MO (USD after scaling).
    """
    assert side in ("imports", "exports")
    if y1 is None:
        today = dt.date.today()
        y1 = today.year

    base = f"https://api.census.gov/data/timeseries/intltrade/{side}/hs"
    frames: List[pd.DataFrame] = []

    for y in range(y0, y1 + 1):
        params = {
            "get": "CTY_CODE,CTY_NAME,ALL_VAL_MO,YEAR,MONTH",
            "time": f"from {y}-01 to {y}-12",
        }
        if CENSUS_API_KEY:
            params["key"] = CENSUS_API_KEY

        try:
            js = _get_json(base, params=params)
        except Exception as e:
            print(f"[CENSUS] {side} {y} failed: {e} | skipping")
            continue

        if not isinstance(js, list) or not js:
            print(f"[CENSUS] {side} {y} empty payload | skipping")
            continue

        header, *rows = js
        df = pd.DataFrame(rows, columns=header)

        # Clean & type coercions
        df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce")
        df["MONTH"] = pd.to_numeric(df["MONTH"], errors="coerce")
        df["ALL_VAL_MO"] = pd.to_numeric(df["ALL_VAL_MO"], errors="coerce") * CENSUS_VALUE_SCALE
        df["CTY_CODE"] = df["CTY_CODE"].astype(str)
        df["CTY_NAME"] = df["CTY_NAME"].astype(str)

        # Drop non-country totals/world
        up = df["CTY_NAME"].str.upper()
        bad = (
            up.str.contains("TOTAL", na=False) |
            up.str.contains("ALL COUNTRIES", na=False) |
            df["CTY_CODE"].isin(["0", "000", "WLD", "ALL"])
        )
        df = df.loc[~bad].copy()

        # Build standard columns used downstream
        df["freq"] = "M"
        df["reporter"] = "USA"
        df["partner"] = df["CTY_CODE"]
        df["partner_name"] = df["CTY_NAME"]
        df["product"] = "Total"
        df["indicator"] = "ALL_VAL_MO_USD"
        df["time"] = df["YEAR"].astype(int).astype(str) + "-" + df["MONTH"].astype(int).astype(str).str.zfill(2)
        df["source"] = f"census_{side}"

        frames.append(df[[
            "freq","reporter","partner","partner_name","product","indicator",
            "time","ALL_VAL_MO","source","YEAR","MONTH","CTY_CODE","CTY_NAME"
        ]])

        # Gentle throttle to be nice to the API
        time.sleep(0.3)

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=[
        "freq","reporter","partner","partner_name","product","indicator",
        "time","ALL_VAL_MO","source","YEAR","MONTH","CTY_CODE","CTY_NAME"
    ])

    # Rename value column to Value (pipeline standard), keep originals too
    if "ALL_VAL_MO" in out.columns:
        out["Value"] = out["ALL_VAL_MO"]
    return out.reset_index(drop=True)

def census_monthly_by_country() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (imports_df, exports_df) with monthly USD by partner."""
    imp = _census_timeseries("imports")
    exp = _census_timeseries("exports")
    return imp, exp

# ---------------------------------------------------------------------
# BLS PPI
# - Monthly index, consistent base.
# - Uses API key if present, and an explicit start year for stability.
# ---------------------------------------------------------------------
def bls_series(series_id: str = "WPUFD4", start_year: int = 2010) -> pd.DataFrame:
    url = f"https://api.bls.gov/publicAPI/v2/timeseries/data/{series_id}"
    params = {
        "startyear": str(start_year),
        "endyear": str(dt.date.today().year),
    }
    bls_key = os.environ.get("BLS_API_KEY", "")
    if bls_key:
        params["registrationKey"] = bls_key

    js = _get_json(url, params=params)
    try:
        rows = js["Results"]["series"][0]["data"]
    except Exception as e:
        raise ValueError(f"BLS shape changed for {series_id}") from e

    data = []
    for r in rows:
        yr = int(r["year"])
        per = r["period"]
        if not per.startswith("M"):
            # Skip annual summary rows like 'M13'
            continue
        mo = int(per.replace("M",""))
        val = float(r["value"])
        data.append({
            "year": yr,
            "period": f"M{mo:02d}",
            "value": val,
            "series_id": series_id
        })

    df = pd.DataFrame(sorted(data, key=lambda x: (x["year"], x["period"])))
    return df.reset_index(drop=True)
