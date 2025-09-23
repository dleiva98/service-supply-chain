# ssc_pipeline/ssc_pipeline/sources.py
import os
import time
import json
import math
import datetime as dt
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
import requests

# ---------------------------
# HTTP helpers
# ---------------------------
DEF_TIMEOUT = 60

def _get(url, params=None, headers=None, timeout=DEF_TIMEOUT):
    r = requests.get(url, params=params, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r

def _get_json(url, params=None, headers=None, timeout=DEF_TIMEOUT):
    r = _get(url, params=params, headers=headers, timeout=timeout)
    try:
        return r.json()
    except Exception as e:
        # Better debugging snippet
        snippet = (r.text or "")[:400]
        raise ValueError(f"JSON parse error for {url} | params={params} | snippet={snippet}") from e

# ---------------------------
# WITS (World Bank) SDMX v2.1
# Notes:
#  - Trade values come in **thousands of USD**.
#  - We scale to **USD** (× 1,000) here so the rest of the pipeline is consistent.
# ---------------------------
def wits_trade_data() -> pd.DataFrame:
    # Annual US ↔ World totals, imports/exports (goods)
    # Key pattern: A.{reporter}.{partner}.{product}.{indicator}
    base = "https://wits.worldbank.org/API/V1/SDMX/V21/rest/data/df_wits_tradestats_trade"
    keys = [
        "A.USA.WLD.all.MPRT-TRD-VL",  # imports
        "A.USA.WLD.all.XPRT-TRD-VL",  # exports
    ]
    frames = []
    for key in keys:
        url = f"{base}/{key}/?format=JSON"
        js = _get_json(url)
        # SDMX JSON → table
        try:
            data = js["data"]["dataSets"][0]["series"]
            dims = js["data"]["structure"]["dimensions"]["series"]
            time_dim = js["data"]["structure"]["dimensions"]["observation"][0]["values"]
        except KeyError as e:
            raise ValueError(f"WITS SDMX shape changed for {url}") from e

        # Decode dimension positions
        dim_names = [d["id"] for d in dims]
        # Build rows
        for s_key, s_val in data.items():
            # s_key e.g. "0:0:0:0:0"
            idx = list(map(int, s_key.split(":")))
            meta = {dim_names[i].lower(): dims[i]["values"][idx[i]]["id"] for i in range(len(idx))}
            for obs_idx, obs in s_val.get("observations", {}).items():
                t = int(obs_idx)
                obs_val = obs[0]
                year = int(time_dim[t]["id"])
                frames.append({
                    "freq": "A",
                    "reporter": meta.get("reporter","USA"),
                    "partner":  meta.get("partner","WLD"),
                    "product":  meta.get("product","all"),
                    "indicator": meta.get("indicator",""),
                    "time": str(year),
                    "Value": float(obs_val) * 1_000.0,  # **scale to USD**
                    "source": "wits_trade",
                })
    df = pd.DataFrame(frames)
    return df

def wits_tariff_data() -> pd.DataFrame:
    # Annual US MFN averages (percent)
    base = "https://wits.worldbank.org/API/V1/SDMX/V21/rest/data/df_wits_tariff"
    keys = [
        "A.USA.WLD.all.MFN-WGHTD-AVRG",  # weighted
        "A.USA.WLD.all.MFN-SMPL-AVRG",   # simple
    ]
    frames = []
    for key in keys:
        url = f"{base}/{key}/?format=JSON"
        js = _get_json(url)
        try:
            data = js["data"]["dataSets"][0]["series"]
            dims = js["data"]["structure"]["dimensions"]["series"]
            time_dim = js["data"]["structure"]["dimensions"]["observation"][0]["values"]
        except KeyError as e:
            raise ValueError(f"WITS Tariff SDMX shape changed for {url}") from e

        dim_names = [d["id"] for d in dims]
        for s_key, s_val in data.items():
            idx = list(map(int, s_key.split(":")))
            meta = {dim_names[i].lower(): dims[i]["values"][idx[i]]["id"] for i in range(len(idx))}
            for obs_idx, obs in s_val.get("observations", {}).items():
                year = int(time_dim[int(obs_idx)]["id"])
                frames.append({
                    "freq": "A",
                    "reporter": meta.get("reporter","USA"),
                    "partner":  meta.get("partner","WLD"),
                    "product":  meta.get("product","all"),
                    "indicator": meta.get("indicator",""),
                    "time": str(year),
                    "Value": float(obs[0]),  # percent, no scaling
                    "source": "wits_tariff",
                })
    return pd.DataFrame(frames)

# ---------------------------
# US Census (Timeseries International Trade, HS)
# Goal:
#  - Pull **monthly total goods** values in **USD** for all partners.
#  - Use the same variables for imports and exports to avoid asymmetry.
#  - Columns: YEAR, MONTH, CTY_CODE, CTY_NAME, ALL_VAL_MO
# Notes:
#  - The API returns dollars, but some docs say "thousands". We treat ALL_VAL_MO as **USD** only after
#    cross-checking. If you confirm it is thousands, change SCALE_CENSUS below to 1_000.0.
# ---------------------------
SCALE_CENSUS = 1.0  # set to 1_000.0 if your manual cross-check shows 'thousands of USD'

def _census_timeseries(side: str, y0: int = 2018, y1: Optional[int] = None) -> pd.DataFrame:
    """
    side: 'imports' or 'exports'
    Returns monthly rows by country with ALL_VAL_MO.
    """
    assert side in ("imports", "exports")
    if y1 is None:
        today = dt.date.today()
        y1 = today.year

    base = f"https://api.census.gov/data/timeseries/intltrade/{side}/hs"
    # Query by full range using 'time=from YYYY-MM to YYYY-MM'
    frames = []
    # We loop year-by-year to keep the URL short and avoid API edge cases.
    for y in range(y0, y1 + 1):
        params = {
            "get": "CTY_CODE,CTY_NAME,ALL_VAL_MO,YEAR,MONTH",
            "time": f"from {y}-01 to {y}-12",
        }
        try:
            js = _get_json(base, params=params)
        except Exception as e:
            # Skip bad years but continue the series
            print(f"[CENSUS] {side} {y} failed: {e} | skipping")
            continue

        # First row is header
        header, *rows = js
        df = pd.DataFrame(rows, columns=header)
        # Clean types
        df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce")
        df["MONTH"] = pd.to_numeric(df["MONTH"], errors="coerce")
        df["ALL_VAL_MO"] = pd.to_numeric(df["ALL_VAL_MO"], errors="coerce") * SCALE_CENSUS
        df["CTY_CODE"] = df["CTY_CODE"].astype(str)
        df["CTY_NAME"] = df["CTY_NAME"].astype(str)
        df = df.dropna(subset=["YEAR","MONTH"])
        df["time"] = df["YEAR"].astype(int).astype(str) + "-" + df["MONTH"].astype(int).astype(str).str.zfill(2)
        df["freq"] = "M"
        df["reporter"] = "USA"
        df["partner"] = df["CTY_CODE"]   # keep code, but keep name for labels
        df["partner_name"] = df["CTY_NAME"]
        df["product"] = "Total"
        df["indicator"] = "ALL_VAL_MO_USD"
        df.rename(columns={"ALL_VAL_MO":"Value"}, inplace=True)
        df["source"] = f"census_{side}"
        frames.append(df[["freq","reporter","partner","partner_name","product","indicator","time","Value","source","YEAR","MONTH"]])

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=[
        "freq","reporter","partner","partner_name","product","indicator","time","Value","source","YEAR","MONTH"
    ])
    return out

def census_monthly_by_country() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (imports_df, exports_df)."""
    imp = _census_timeseries("imports")
    exp = _census_timeseries("exports")
    return imp, exp

# ---------------------------
# BLS PPI
# Use a single consistent series (e.g., WPUFD4 – Finished goods), monthly index.
# ---------------------------
def bls_series(series_id: str = "WPUFD4") -> pd.DataFrame:
    """Return PPI monthly index with YEAR, MONTH and numeric 'value'."""
    url = f"https://api.bls.gov/publicAPI/v2/timeseries/data/{series_id}"
    params = {"registrationKey": os.environ.get("BLS_API_KEY","")}
    js = _get_json(url, params=params)
    try:
        rows = js["Results"]["series"][0]["data"]
    except Exception as e:
        raise ValueError(f"BLS shape changed for {series_id}") from e
    data = []
    for r in rows:
        # Keep numeric months, same base; BLS returns strings
        yr = int(r["year"])
        mo = int(r["period"].replace("M",""))
        val = float(r["value"])
        data.append({"year": yr, "period": f"M{mo:02d}", "value": val, "series_id": series_id})
    df = pd.DataFrame(data)
    return df
