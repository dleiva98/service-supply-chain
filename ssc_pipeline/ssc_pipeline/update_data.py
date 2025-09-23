# ssc_pipeline/ssc_pipeline/update_data.py
import os
import datetime as dt
import pandas as pd

from .sources import wits_trade_data, wits_tariff_data, census_monthly_by_country, bls_series

def run(csv_path: str = "main.csv"):
    frames = []

    # WITS annual
    try:
        w_tr = wits_trade_data()
        frames.append(w_tr)
    except Exception as e:
        print("[WITS trade] skipped:", e)

    try:
        w_tf = wits_tariff_data()
        frames.append(w_tf)
    except Exception as e:
        print("[WITS tariff] skipped:", e)

    # Census monthly (imports & exports)
    try:
        imp, exp = census_monthly_by_country()
        frames.extend([imp, exp])
    except Exception as e:
        print("[Census] skipped:", e)

    # BLS PPI
    try:
        ppi = bls_series("WPUFD4")
        ppi["source"] = "bls_ppi"
        frames.append(ppi.rename(columns={"year":"year","period":"period","value":"value"}))
    except Exception as e:
        print("[BLS] skipped:", e)

    if not frames:
        raise RuntimeError("No data sources succeeded.")

    df = pd.concat(frames, ignore_index=True)
    df["pulled_at_utc"] = dt.datetime.utcnow().replace(microsecond=0).isoformat()+"Z"

    # Normalize ‘time’ column everywhere
    if "time" not in df.columns:
        df["time"] = None
    # For BLS rows, build YYYY-MM from year/period
    bls_mask = df["source"].eq("bls_ppi")
    if bls_mask.any():
        tmp = df.loc[bls_mask]
        # ensure ‘period’ like M01..M12
        tmp["time"] = tmp["year"].astype(int).astype(str) + "-" + tmp["period"].str.replace("M","", regex=False).astype(int).astype(str).str.zfill(2)
        df.loc[bls_mask, "time"] = tmp["time"]

    # Column ordering for consistency
    cols = ["freq","reporter","partner","partner_name","product","indicator","time","Value","value","source","pulled_at_utc","year","period"]
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[cols]

    # Write
    out = os.path.abspath(csv_path)
    df.to_csv(out, index=False)
    print(f"✅ Data updated. Rows: {len(df):,} → {out}")
