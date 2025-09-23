
import time, json, requests, pandas as pd
from typing import List, Dict, Any
from .utils import cache_get, cache_put
from .sdmx import flatten_wits_sdmx

DEFAULT_TIMEOUT = 60

# =========================
# WITS SDMX REST datasets
# =========================
WITS_TRADE  = "https://wits.worldbank.org/API/V1/SDMX/V21/rest/data/df_wits_tradestats_trade"
WITS_TARIFF = "https://wits.worldbank.org/API/V1/SDMX/V21/rest/data/df_wits_tradestats_tariff"

def _get_json(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Descarga con Accept JSON; si el servidor devuelve XML/HTML o vacío, retorna payload vacío
    en formato SDMX para que el pipeline continúe sin romperse.
    """
    headers = {"Accept": "application/json, text/json;q=0.9, */*;q=0.1"}
    cached = cache_get(url, params, ttl_seconds=86400)
    if cached is not None:
        return cached
    resp = requests.get(url, params=params, headers=headers, timeout=DEFAULT_TIMEOUT)
    resp.raise_for_status()
    try:
        data = resp.json()
    except Exception:
        snippet = (resp.text or "")[:300].replace("\\n", " ")
        print(f"[WITS] Respuesta no JSON (saltando): {url} params={params} | inicio: {snippet}")
        return {"dataSets": []}  # estructura vacía entendida por el aplanador
    else:
        cache_put(url, params, data)
        time.sleep(0.25)
        return data

def _sdmx_key(freq: str, reporter: str, partner: str, product: str, indicator: str) -> str:
    return f"{freq}.{reporter}.{partner}.{product}.{indicator}"

def _read_wits_product_from_config() -> str:
    """
    Lee 'wits_product' desde config.yaml (mismo paquete); por defecto 'total'.
    """
    import os
    try:
        import yaml
        cfg_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
        cfg = yaml.safe_load(open(cfg_path, 'r', encoding='utf-8'))
        return str(cfg.get('wits_product', 'total')).strip() or 'total'
    except Exception:
        return 'total'

def wits_trade_data(reporters: List[str], partners: List[str], indicators: List[str], years: List[int]) -> pd.DataFrame:
    frames = []
    y0, y1 = min(years), max(years)
    product_default = _read_wits_product_from_config()
    for r in reporters:
        r = r.lower()
        for p in (partners or ["wld"]):
            p = p.lower()
            for ind in indicators:
                key = _sdmx_key("A", r, p, product_default, ind)
                url = f"{WITS_TRADE}/{key}"
                try:
                    js = _get_json(url, {"startperiod": y0, "endperiod": y1, "format": "JSON"})
                except Exception as e:
                    print(f"[WITS] Error en {url}: {e}")
                    js = {"dataSets": []}
                df = flatten_wits_sdmx(js)
                if not df.empty:
                    # Garantiza metadatos mínimos si el payload no los trae
                    for col, val in [("indicator", ind), ("reporter", r), ("partner", p), ("product", product_default)]:
                        if col not in df.columns:
                            df[col] = val
                    df["source"] = "wits_trade"
                    frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def wits_tariff_data(reporters: List[str], partners: List[str], indicators: List[str], years: List[int]) -> pd.DataFrame:
    frames = []
    y0, y1 = min(years), max(years)
    product_default = _read_wits_product_from_config()
    for r in reporters:
        r = r.lower()
        for p in (partners or ["wld"]):
            p = p.lower()
            for ind in indicators:
                key = _sdmx_key("A", r, p, product_default, ind)
                url = f"{WITS_TARIFF}/{key}"
                try:
                    js = _get_json(url, {"startperiod": y0, "endperiod": y1, "format": "JSON"})
                except Exception as e:
                    print(f"[WITS] Error en {url}: {e}")
                    js = {"dataSets": []}
                df = flatten_wits_sdmx(js)
                if not df.empty:
                    for col, val in [("indicator", ind), ("reporter", r), ("partner", p), ("product", product_default)]:
                        if col not in df.columns:
                            df[col] = val
                    df["source"] = "wits_tariff"
                    frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

# =========================
# U.S. Census Time Series
# =========================


def census_monthly_by_country(year_from: int, year_to: int, country_codes=None, dataset="imports", variables=None) -> pd.DataFrame:
    """
    Time Series International Trade (HS):
      - 'time' es PREDICATE-ONLY (va como parámetro, no en get).
      - Variables de valor total:
          * exports/hs:  ALL_VAL_MO, ALL_VAL_YR
          * imports/hs:  GEN_VAL_MO, GEN_VAL_YR   (no existe ALL_VAL_* en imports)
      - Pedimos por rangos anuales (time=from YYYY-01 to YYYY-12); si falla, por meses.
      - Armonizamos en la salida: para imports creamos ALL_VAL_MO/YR = GEN_VAL_MO/YR.
    """
    if dataset not in ("imports","exports"):
        raise ValueError("dataset must be 'imports' or 'exports'")
    base = f"https://api.census.gov/data/timeseries/intltrade/{dataset}/hs"

    # Variables por dataset (NO incluir 'time' en get; 'time' es predicate-only)
    if variables is None:
        if dataset == "exports":
            variables = ["CTY_CODE","CTY_NAME","ALL_VAL_MO","ALL_VAL_YR","YEAR","MONTH"]
        else:  # imports
            variables = ["CTY_CODE","CTY_NAME","GEN_VAL_MO","GEN_VAL_YR","YEAR","MONTH"]

    frames = []
    for year in range(year_from, year_to+1):
        params = {"get": ",".join(variables), "time": f"from {year}-01 to {year}-12"}
        if country_codes:
            params["CTY_CODE"] = ",".join(country_codes)
        try:
            resp = requests.get(base, params=params, timeout=DEFAULT_TIMEOUT)
            resp.raise_for_status()
            js = resp.json()
            if isinstance(js, list) and js:
                cols = js[0]
                for row in js[1:]:
                    tmp = pd.DataFrame([row], columns=cols)
                    frames.append(tmp)
        except requests.HTTPError as e:
            # Fallback mensual si no acepta el rango
            any_ok = False
            for m in range(1, 13):
                mm = f"{m:02d}"
                p2 = {"get": ",".join(variables), "time": f"{year}-{mm}"}
                if country_codes:
                    p2["CTY_CODE"] = ",".join(country_codes)
                try:
                    r2 = requests.get(base, params=p2, timeout=DEFAULT_TIMEOUT)
                    r2.raise_for_status()
                    js2 = r2.json()
                    if isinstance(js2, list) and js2:
                        cols2 = js2[0]
                        for row in js2[1:]:
                            tmp2 = pd.DataFrame([row], columns=cols2)
                            frames.append(tmp2)
                            any_ok = True
                except Exception:
                    pass
            if not any_ok:
                snippet = ""
                try:
                    snippet = (resp.text or "")[:180].replace("\n"," ")
                except Exception:
                    pass
                print(f"[CENSUS] {e} | sin datos para year={year} | detalle: {snippet}")
        except Exception as e:
            print(f"[CENSUS] error inesperado year={year}: {e}")

        time.sleep(0.15)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)

    # Tipos numéricos
    for c in ("ALL_VAL_MO","ALL_VAL_YR","GEN_VAL_MO","GEN_VAL_YR"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # Construir 'time' = YYYY-MM
    if "YEAR" in out.columns and "MONTH" in out.columns:
        out["time"] = out["YEAR"].astype(str).str.zfill(4) + "-" + out["MONTH"].astype(str).str.zfill(2)

    # Armonizar nombres: en imports creamos ALL_VAL_* desde GEN_VAL_*
    if dataset == "imports":
        if "GEN_VAL_MO" in out.columns and "ALL_VAL_MO" not in out.columns:
            out["ALL_VAL_MO"] = out["GEN_VAL_MO"]
        if "GEN_VAL_YR" in out.columns and "ALL_VAL_YR" not in out.columns:
            out["ALL_VAL_YR"] = out["GEN_VAL_YR"]

    # Renombrar columnas de partner para consistencia
    out = out.rename(columns={"CTY_CODE":"partner", "CTY_NAME":"partner_name"})

    return out


    out = pd.concat(frames, ignore_index=True)
    # Normalizaciones y tipos
    for c in ("ALL_VAL_MO","ALL_VAL_YR"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    # Construir columna 'time' tipo YYYY-MM
    if "YEAR" in out.columns and "MONTH" in out.columns:
        out["time"] = out["YEAR"].astype(str).str.zfill(4) + "-" + out["MONTH"].astype(str).str.zfill(2)
    out = out.rename(columns={"CTY_CODE":"partner", "CTY_NAME":"partner_name"})
    return out


# =========================
# BLS PPI (servicios)
# =========================
BLS_BASE = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
def bls_series(series_ids: List[str], start_year: int, end_year: int) -> pd.DataFrame:
    payload = {"seriesid": series_ids, "startyear": str(start_year), "endyear": str(end_year)}
    key = {"payload": json.dumps(payload, sort_keys=True)}
    cached = cache_get(BLS_BASE, key, ttl_seconds=86400)
    if cached is None:
        resp = requests.post(BLS_BASE, json=payload, timeout=DEFAULT_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        cache_put(BLS_BASE, key, data)
        time.sleep(0.2)
    else:
        data = cached
    frames = []
    for s in data.get("Results", {}).get("series", []):
        sid = s.get("seriesID")
        df = pd.DataFrame(s.get("data", []))
        if not df.empty:
            df["series_id"] = sid
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
