from pathlib import Path

path = Path("tools/comfort_weakspot_scanner.py")
data = path.read_text()
old = """    dfc = dfc.dropna(
        subset=[
            "time_s",
            "speed_mph",
            "gear_actual__canon",
            "throttle_pct",
            "engine_rpm__canon",
            "tcc_locked_built",
            "tcc_slip_fused",
        ]
    )
    if dfc.empty:
"""
new = """    dfc = dfc.dropna(
        subset=[
            "time_s",
            "speed_mph",
            "gear_actual__canon",
            "throttle_pct",
            "engine_rpm__canon",
            "tcc_locked_built",
            "tcc_slip_fused",
        ]
    )
    dfc = dfc.reset_index(drop=True)
    if dfc.empty:
"""
if old not in data:
    raise SystemExit("pattern not found for drop block")
path.write_text(data.replace(old, new, 1))
