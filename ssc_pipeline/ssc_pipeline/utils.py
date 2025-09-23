
import os, time, json, hashlib
from typing import Dict, Any, Optional

CACHE_DIR = os.environ.get("SSC_CACHE_DIR", ".ssc_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def _hash_key(url: str, params: Optional[Dict[str, Any]]=None) -> str:
    payload = url + ("|" + json.dumps(params, sort_keys=True) if params else "")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def cache_get(url: str, params: Optional[Dict[str, Any]]=None, ttl_seconds: int = 3600):
    key = _hash_key(url, params)
    path = os.path.join(CACHE_DIR, key + ".json")
    if os.path.exists(path) and (time.time() - os.path.getmtime(path)) < ttl_seconds:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def cache_put(url: str, params: Optional[Dict[str, Any]], data: Any):
    key = _hash_key(url, params)
    path = os.path.join(CACHE_DIR, key + ".json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        pass
