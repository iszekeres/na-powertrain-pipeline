"""
TPS phase helper for NA_Trans pipeline.

Based on TPS heatmap from 2025-11-15 (shortthree1 log):
- Median TPS (moving, brake-off, fwd) ~14.9%
- 25–75% ≈ 8.2–22.8%
- Most real driving is 8–23% throttle.

Phases:
  IDLE        :   0–4%
  FEATHER     :   4–8%
  CRUISE      :   8–18%
  MILD_ACCEL  :  18–25%
  STRONG_ACCEL:  25–35%
  HEAVY       :  >35%
"""

CRUISE_MIN = 8.0
CRUISE_MAX = 18.0
MILD_MAX   = 25.0
KICKDOWN_MIN = 25.0
CORNER_EXIT_MIN = 12.0


def _to_float(x):
    try:
        return float(x)
    except Exception:
        return None


def tps_phase(tps: float) -> str:
    x = _to_float(tps)
    if x is None:
        return "UNKNOWN"
    if x < 4.0:
        return "IDLE"
    if x < 8.0:
        return "FEATHER"
    if x < CRUISE_MAX:
        return "CRUISE"
    if x < MILD_MAX:
        return "MILD_ACCEL"
    if x < 35.0:
        return "STRONG_ACCEL"
    return "HEAVY"


def is_cruise_band(tps: float) -> bool:
    """Main city/neighborhood cruise band (8–18%)."""
    x = _to_float(tps)
    if x is None:
        return False
    return (CRUISE_MIN <= x < CRUISE_MAX)


def is_cruise_or_mild(tps: float) -> bool:
    """Cruise or mild-accel (8–25%)."""
    x = _to_float(tps)
    if x is None:
        return False
    return (CRUISE_MIN <= x <= MILD_MAX)


def is_kickdown_band(tps: float) -> bool:
    """Where real kickdown requests live (>=25%)."""
    x = _to_float(tps)
    if x is None:
        return False
    return x >= KICKDOWN_MIN


def is_corner_exit_band(tps: float) -> bool:
    """Corner exit should have at least ~12% TPS."""
    x = _to_float(tps)
    if x is None:
        return False
    return x >= CORNER_EXIT_MIN

