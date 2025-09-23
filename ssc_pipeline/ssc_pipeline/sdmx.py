
from typing import Dict, Any, List
import pandas as pd

def _get_dim_values(struct: Dict[str, Any], axis: str) -> List[Dict[str, Any]]:
    return struct.get("dimensions", {}).get(axis, [])

def _value_of(dim: Dict[str, Any], idx: int) -> str:
    vals = dim.get("values", [])
    if 0 <= idx < len(vals):
        v = vals[idx]
        return str(v.get("id") or v.get("name") or "")
    return ""

def flatten_wits_sdmx(payload: Dict[str, Any]) -> pd.DataFrame:
    try:
        datasets = payload.get("dataSets", [])
        if not datasets:
            data = payload.get("data")
            if isinstance(data, list) and data:
                df = pd.DataFrame(data)
                if "OBS_VALUE" in df.columns: df.rename(columns={"OBS_VALUE":"Value"}, inplace=True)
                if "TimePeriod" in df.columns: df.rename(columns={"TimePeriod":"time"}, inplace=True)
                return df
            return pd.DataFrame()

        series_dict = datasets[0].get("series", {})
        struct = payload.get("structure", {})
        s_dims = _get_dim_values(struct, "series")
        o_dims = _get_dim_values(struct, "observation")
        time_dim = o_dims[0] if o_dims else {}

        rows = []
        for s_key, s_val in series_dict.items():
            idxs = [int(x) for x in s_key.split(":")]
            vals = [ _value_of(s_dims[i], idxs[i]) if i < len(s_dims) else "" for i in range(len(idxs)) ]
            # Orden típico WITS: FREQ, REPORTER, PARTNER, PRODUCT, INDICATOR
            freq      = vals[0] if len(vals)>0 else ""
            reporter  = vals[1] if len(vals)>1 else ""
            partner   = vals[2] if len(vals)>2 else ""
            product   = vals[3] if len(vals)>3 else ""
            indicator = vals[4] if len(vals)>4 else ""

            obs = s_val.get("observations", {})
            for o_idx, o_val in obs.items():
                value = None
                if isinstance(o_val, list) and o_val:
                    value = o_val[0]
                elif isinstance(o_val, (int, float)):
                    value = o_val
                try:
                    t = _value_of(time_dim, int(o_idx))
                except Exception:
                    t = str(o_idx)
                rows.append({
                    "freq": freq, "reporter": reporter, "partner": partner,
                    "product": product, "indicator": indicator,
                    "time": t, "Value": value
                })
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()
