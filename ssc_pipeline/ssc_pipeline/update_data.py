# ssc_pipeline/ssc_pipeline/update_data.py
# Robust updater for main.csv:
# - Keeps ALL useful columns (no destructive slicing)
# - Validates Census monthly USD (ALL_VAL_MO/GEN_VAL_MO/VAL_MO) and YEAR/MONTH
# - Normalizes WITS value column
# - Ensures BLS has series_id/year/period/value and builds a YYYY-MM 'time' helper
# - Writes a concise ingestion summary by source

import os
import datetime as dt
import pandas as pd

from .sources import (
    wits_trade_data,
    wits_tariff_data,
    census_monthly_by_country,
    bls_series,
)

USD_CANDIDATES = ["ALL_VAL_MO", "GEN_VAL_MO", "VAL_MO"]  # monthly USD value fields

def _ensure_census_monthly_usd(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Validate Census frame:
    - Has YEAR & MONTH (ints)
    - Has at least one USD value column (ALL_VAL_MO/GEN_VAL_MO/VAL_MO)
    - Partner columns kept (CTY_NAME / CTY_CODE if present)
    - Add/verify 'source' and do minimal cleaning
    """
    if df is None or df.empty:
        raise ValueError(f"{label}: empty frame")

    # YEAR/MONTH
    have_year_month = {"YEAR", "MONTH"}.issubset(df.columns)
    have_year_period = {"year", "period"}.issubset(df.columns)

    if not (have_year_month or have_year_period):
        raise ValueError(f"{label}: missing YEAR/MONTH or year/period columns")

    df = df.copy()

    if have_year_month:
        df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce").astype("Int64")
        df["MONTH"] = pd.to_numeric(df["MONTH"], errors="coerce").astype("Int64")
    else:
        # fall back to year/period like M01..M12
        df["YEAR"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        df["MONTH"] = (
            df["period"].astype(str).str.extract(r"M(\d{2})")[0].astype(float).astype("Int64")
        )

    # USD value column
    usd_col = None
    for c in USD_CANDIDATES + ["value"]:
        if c in df.columns:
            usd_col = c
            break
    if usd_col is None:
        raise ValueError(f"{label}: no monthly USD value column (expected one of {USD_CANDIDATES})")

    # Sanity: must be non-negative numeric
    df[usd_col] = pd.to_numeric(df[usd_col], errors="coerce")
    df = df.dropna(subset=["YEAR", "MONTH", usd_col])
    df = df[df[usd_col] >= 0]

    # Partner labels (keep whatever is present)
    if "CTY_NAME" not in df.columns and "partner_name" in df.columns:
        df["CTY_NAME"] = df["partner_name"]
    if "CTY_CODE" not in df.columns and "partner_code" in df.columns:
        df["CTY_CODE"] = df["partner_code"]

    # Build a helper 'time' (YYYY-MM) string; report uses YEAR/MONTH anyway
    df["time"] = (
        df["YEAR"].astype(str) + "-" + df["MONTH"].astype(str).str.zfill(2)
    )

    # Standardize value_usd column name for downstream use
    df["value_usd"] = df[usd_col]

    # Ensure 'source'
    if "source" not in df.columns or df["source"].isna().all():
        df["source"] = label

    # Minimal sort
    # NOTE: do not drop CTY_* columns — the report needs them
    return df.sort_values(["YEAR", "MONTH"]).reset_index(drop=True)


def _normalize_wits(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Normalize WITS frames:
    - ensure 'value' column exists (rename 'Value' -> 'value' if needed)
    - carry 'source'
    """
    if df is None or df.empty:
        raise ValueError(f"{label}: empty frame")
    df = df.copy()
    if "value" not in df.columns and "Value" in df.columns:
        df["value"] = pd.to_numeric(df["Value"], errors="coerce")
    elif "value" in df.columns:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
    else:
        # allow empty value if tariff indicators (handled in report) but keep column
        df["value"] = pd.NA

    if "source" not in df.columns or df["source"].isna().all():
        df["source"] = label

    # 'time' for WITS is typically a year number; keep as-is if present
    if "time" in df.columns:
        df["time"] = pd.to_numeric(df["time"], errors="coerce")

    return df.reset_index(drop=True)


def _normalize_bls(ppi: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure BLS has series_id, year, period, value, and a helper 'time' (YYYY-MM).
    """
    if ppi is None or ppi.empty:
        raise ValueError("BLS: empty frame")

    ppi = ppi.copy()
    # expected columns: series_id, year, period, value
    # coerce numerics where needed
    if "year" in ppi.columns:
        ppi["year"] = pd.to_numeric(ppi["year"], errors="coerce").astype("Int64")
    if "value" in ppi.columns:
        ppi["value"] = pd.to_numeric(ppi["value"], errors="coerce")

    # Normalize 'period' like M01..M12
    if "period" not in ppi.columns:
        raise ValueError("BLS: missing 'period'")
    # Build helper time
    ppi["time"] = (
        ppi["year"].astype(str) + "-" +
        ppi["period"].astype(str).str.replace("M", "", regex=False).astype(int).astype(str).str.zfill(2)
    )

    # Ensure source
    ppi["source"] = "bls_ppi"

    return ppi.reset_index(drop=True)


def run(csv_path: str = "main.csv"):
    frames = []

    # -------- WITS annual --------
    try:
        w_tr = _normalize_wits(wits_trade_data(), "wits_trade")
        frames.append(w_tr)
    except Exception as e:
        print("[WITS trade] skipped:", e)

    try:
        w_tf = _normalize_wits(wits_tariff_data(), "wits_tariff")
        frames.append(w_tf)
    except Exception as e:
        print("[WITS tariff] skipped:", e)

    # -------- Census monthly (imports & exports, USD only) --------
    try:
        imp, exp = census_monthly_by_country()
        imp = _ensure_census_monthly_usd(imp, "census_imports")
        exp = _ensure_census_monthly_usd(exp, "census_exports")
        frames.extend([imp, exp])
    except Exception as e:
        print("[Census] skipped:", e)

    # -------- BLS PPI --------
    try:
        ppi = bls_series("WPUFD4")
        ppi = _normalize_bls(ppi)
        frames.append(ppi)
    except Exception as e:
        print("[BLS] skipped:", e)

    if not frames:
        raise RuntimeError("No data sources succeeded.")

    # Concatenate WITHOUT dropping useful columns
    df = pd.concat(frames, ignore_index=True, sort=False)

    # Stamp
    df["pulled_at_utc"] = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    # Light normalization: keep both 'Value' and 'value' if present, report handles it
    # Ensure 'time' exists (some rows may not have it; that's OK for annual WITS)
    if "time" not in df.columns:
        df["time"] = pd.NA

    # Write all columns (preserve CTY_NAME/CTY_CODE/YEAR/MONTH/value_usd/etc.)
    out = os.path.abspath(csv_path)
    df.to_csv(out, index=False)

    # Ingestion summary
    by_src = df["source"].astype(str).value_counts(dropna=False).to_string()
    print(f"✅ Data updated. Rows: {len(df):,} → {out}\nSources breakdown:\n{by_src}")
