#!/usr/bin/env python3
# Safe, cached weighting helpers (Windows-friendly)
import os, re, math, datetime as dt
from functools import lru_cache

NOW = dt.datetime.now()
_TS_RE = re.compile(r"__(\d{8})__(\d{6})")

@lru_cache(maxsize=1024)
def parse_ts_from_name(name):
    if not name: return None
    m = _TS_RE.search(str(name))
    if not m: return None
    try:
        return dt.datetime.strptime("".join(m.groups()), "%Y%m%d%H%M%S")
    except Exception:
        return None

@lru_cache(maxsize=2048)
def file_age_days(path_or_name, now=None):
    now = now or NOW
    s = str(path_or_name or "")
    try:
        # If a real path, prefer mtime
        if (os.path.isabs(s) or os.path.sep in s or "/" in s or "\\\\" in s) and os.path.exists(s):
            mtime = dt.datetime.fromtimestamp(os.path.getmtime(s))
            return max((now - mtime).total_seconds()/86400.0, 0.0)
    except Exception:
        pass
    # Else parse __YYYYMMDD__HHMMSS from the name
    ts = parse_ts_from_name(s)
    if ts:
        return max((now - ts).total_seconds()/86400.0, 0.0)
    # Fallback: treat as fresh
    return 0.0

def recency_weight(age_days, half_life_days=30.0):
    if half_life_days <= 0:
        return 1.0
    return math.pow(0.5, age_days/half_life_days)

def route_bias(name_or_path, speed=None, mapping=None, default=1.0):
    mapping = mapping or {"neighborhood":1.5, "inbound":1.2, "outbound":1.2, "highway":1.1}
    s = (name_or_path or "").lower()
    for key, w in mapping.items():
        if key in s:
            try: return float(w)
            except: return default
    if speed is not None:
        try:
            sp = float(speed)
            if sp <= 30: return float(mapping.get("neighborhood", default))
            if sp >= 60: return float(mapping.get("highway", default))
        except Exception:
            pass
    return float(default)

def combined_weight(file_name, speed, half_life_days=30.0, route_map=None, now=None):
    now = now or NOW
    if half_life_days <= 0:
        rw = 1.0
    else:
        age = file_age_days(file_name, now=now)
        rw  = recency_weight(age, half_life_days=half_life_days)
    rb  = route_bias(file_name, speed=speed, mapping=route_map or {})
    return float(rw * rb)
