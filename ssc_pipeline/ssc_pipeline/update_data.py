
import os, json, pandas as pd
from datetime import datetime
from typing import Dict, Any, List
from .sources import wits_trade_data, wits_tariff_data, census_monthly_by_country, bls_series

ROOT = os.environ.get("SSC_ROOT", "/content/drive/MyDrive/SSC" if os.path.exists("/content/drive/MyDrive/SSC") else ".")
OUT_PATH = os.path.join(ROOT, "main.csv")
CONFIG = os.path.join(os.path.dirname(__file__), "..", "config.yaml")

def load_config(path: str) -> Dict[str, Any]:
    try:
        import yaml
        return yaml.safe_load(open(path, "r", encoding="utf-8").read())
    except Exception:
        try:
            return json.loads(open(path, "r", encoding="utf-8").read())
        except Exception:
            return {}

import json

def _hashable(v):
    if isinstance(v, (list, dict)):
        try:
            return json.dumps(v, sort_keys=True)
        except Exception:
            return str(v)
    return v


def upsert_csv(df_new: pd.DataFrame, out_path: str, join_keys: List[str]) -> pd.DataFrame:
    if os.path.exists(out_path):
        df_old = pd.read_csv(out_path)
        df = pd.concat([df_old, df_new], ignore_index=True)
        for c in join_keys:
            if c in df.columns:
                df[c] = df[c].map(_hashable)
        df = df.drop_duplicates(subset=join_keys, keep="last")
    else:
        df = df_new
        for c in join_keys:
            if c in df.columns:
                df[c] = df[c].map(_hashable)
    df.to_csv(out_path, index=False)
    return df

def run():
    cfg = load_config(CONFIG)
    reporters = cfg.get("reporters", ["usa"])
    partners  = cfg.get("partners", ["wld","can","mex","chn","jpn","deu"])
    years     = cfg.get("years", [2019, 2020, 2021, 2022, 2023, 2024, 2025])

    trade_indicators  = cfg.get("wits_trade_indicators",  ["MPRT-TRD-VL","XPRT-TRD-VL"])
    tariff_indicators = cfg.get("wits_tariff_indicators", ["MFN-WGHTD-AVRG","MFN-SMPL-AVRG"])

    bls_ids  = cfg.get("bls_series", ["WPUFD4","WPSID612"])
    bls_start= cfg.get("bls_start_year", 2010)
    bls_end  = cfg.get("bls_end_year", datetime.utcnow().year)

    # 1) WITS (trade + tariff)
    df_trade  = wits_trade_data(reporters, partners, trade_indicators, years)
    df_tariff = wits_tariff_data(reporters, partners, tariff_indicators, years)

    # 2) Census (mensual)
    census_start = max(2013, years[0] if years else 2013)
    census_end   = max(years) if years else datetime.utcnow().year
    df_imp = census_monthly_by_country(census_start, census_end, dataset="imports")
    df_exp = census_monthly_by_country(census_start, census_end, dataset="exports")
    if not df_imp.empty: df_imp["source"] = "census_imports"
    if not df_exp.empty: df_exp["source"] = "census_exports"

    # 3) BLS (PPI servicios)
    df_bls = bls_series(bls_ids, bls_start, bls_end)
    if not df_bls.empty: df_bls["source"] = "bls_ppi"

    frames = [x for x in [df_trade, df_tariff, df_imp, df_exp, df_bls] if x is not None and not x.empty]
    if not frames:
        print("No se pudo descargar ningún dataset. Revisa red/endpoint/config.")
        return

    pulled_at = datetime.utcnow().isoformat()
    for f in frames:
        f["pulled_at_utc"] = pulled_at

    union = pd.concat(frames, ignore_index=True, sort=False)
    join_keys = [c for c in ["time","year","Year","reporter","partner","indicator","product","series_id","CTY_CODE","CTY_NAME"] if c in union.columns]
    if join_keys:
        for c in join_keys:
            if c in union.columns:
                union[c] = union[c].map(_hashable)
        union = upsert_csv(union, OUT_PATH, join_keys)
    else:
        union.to_csv(OUT_PATH, index=False)

    print(f"✅ Data actualizada. Filas: {len(union)} | Guardado en: {OUT_PATH}")

if __name__ == "__main__":
    run()
