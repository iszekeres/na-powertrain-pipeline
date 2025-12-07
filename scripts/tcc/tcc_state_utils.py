import pandas as pd

SLIP_LOCK_MAX = 50.0

def classify_tcc_state_psi(abs_slip, psi):
    """
    psi-aware TCC state:
      - If psi is NaN: return None (discard)
      - If psi == 0: OPEN
      - If psi > 0:
          LOCKED  if |slip| <= 50
          PARTIAL if |slip| > 50
    """
    if pd.isna(psi):
        return None
    if psi == 0:
        return "OPEN"
    if abs_slip <= SLIP_LOCK_MAX:
        return "LOCKED"
    return "PARTIAL"
