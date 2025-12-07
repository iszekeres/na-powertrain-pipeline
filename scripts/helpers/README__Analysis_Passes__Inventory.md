# Analysis Scripts Inventory (Tahoe L83 / 6L80)

Buckets and counts focus on the transmission + TCC tooling currently in this repo. Paths are workspace‐relative.

## Bucket summary
- TCC / EC³ slip & psi: key tools (8) + many auxiliary table builders
- Shift quality / harshness / latency: key tools (4)
- Driver intent / kickdown / cruise: key tools (5)
- Torque surfaces / physics / road load / torque error: key tools (6)
- ABS / TCS / traction: key tools (2)
- Corner / chassis gates: key tools (2)
- “Super” orchestrators (highway packs): key tools (3)
- Fuel / MPG overlays: key tools (1)

## TCC / EC³ slip & psi
- `tcc_truth_with_psi.py`  
  Purpose: per‑gear LOCKED/PARTIAL/OPEN time, slip, psi, slip*psi integral.  
  Inputs: prepped NOCRUISE CSVs. Outputs: `tcc_truth_by_gear_with_psi.csv`. Strict columns (gear, time_s, slip, psi).
- `tcc_speed_binned_with_psi.py`  
  Purpose: speed‑binned (5 mph) TCC slip+psi by gear/state with slip*psi.  
  Inputs: prepped NOCRUISE. Outputs: `tcc_speed_gear_state_stats*.csv`. Strict columns (speed, gear, slip, psi, time).
- `classify_tcc_coupling_modes.py`  
  Purpose: classify bins (HYDRO_COUPLED / HARD_LOCK / EC3_PARTIAL_* / etc.) from speed‑binned slip+psi.  
  Inputs: speed‑binned CSV above. Outputs: `...__classified*.csv`.
- `tcc_highres_surface__burblock3.py`  
  Purpose: 1 mph × 1% pedal lock/partial/open fractions; psi‑aware state via `tcc_state_utils.py`; smoothed surface.  
  Inputs: raw log (burblock5, edit LOG_PATH to reuse). Outputs: `tcc_highres_surface__*.csv` (+ `__SMOOTHED`).
- `tcc_triggers_burblock3.py`  
  Purpose: TCC state transitions in gears 4/5, 38–52 mph with 1 s pre/post context; psi‑aware state.  
  Inputs: raw log. Outputs: `tcc_4_5_38_52_transitions__*.csv`, `...state_vs_pedal__*.csv`.
- `tcc_light_accel_decision_surface__burblock3.py`  
  Purpose: light‑accel decisions (stay locked / unlock / downshift / upshift) over 1 mph × 1% grid; psi‑aware lock.  
  Inputs: raw log. Outputs: `tcc_light_accel_decision_map__*.csv`.
- `tcc_fuel_economy__burblock3.py`  
  Purpose: MPG vs speed/gear/pedal/TCC (psi‑aware state).  
  Inputs: raw log. Outputs: `fuel_global_summary__*.txt`, `fuel_vs_speed*__*.csv`.
- `tcc_state_utils.py`  
  Purpose: psi‑aware state classifier (NaN→drop, psi==0→OPEN, psi>0 & slip<=50→LOCKED else PARTIAL). Used by the above.

## Shift quality / harshness / latency
- `highway_intent_and_harshness.py`  
  Purpose: harshness event detection + heatmap from prepped logs; also intent annotation mode.  
  Inputs: prepped dir (`--do-harshness`), or episodes/DD/torque surface (`--do-intent`). Outputs: shift events, heatmap.
- `highway_harshness_deepdive.py`  
  Purpose: occupancy merge into harshness heatmap; TCC overlay merge.  
  Inputs: harshness heatmap, prepped dir, optional TCC truth. Outputs: `...with_occupancy.csv`, `...tcc_overlay.csv`.
- `highway_LAT_lite__NOCRUISE.py`  
  Purpose: LAT‑lite shift latency summary for 4–5–6 windows. Inputs: prepped NOCRUISE or episodes dir. Outputs: LAT CSV.
- `shift_latency_pass_weighted.py` (and `__TAHOE`)  
  Purpose: build shift latency tables from logs; used in table passes.

## Driver intent / kickdown / cruise
- `build_intent_episodes__NOCRUISE.py`  
  Purpose: detect driver intent episodes from prepped NOCRUISE logs. Outputs: `ALL__intent_episodes__*_NOCRUISE.csv`.
- `highway_intent_and_harshness.py --do-intent`  
  Purpose: annotate intent episodes with driver demand + torque surface; flag should_have_downshifted.
- `highway_overlay_map__NOCRUISE.py`  
  Purpose: merge occupancy + torque deficit + intent frustration into one overlay map (speed/pedal bins).
- `cruise_pedal_scan_dfcofree.py`  
  Purpose: DFCO‑free cruise pedal scan from raw log (uses speed/brake/pedal/throttle).
- `pedal_tps_usage__NOCRUISE.py`  
  Purpose: summarize pedal vs TPS usage from NOCRUISE logs.

## Torque surfaces / physics / road load / torque error
- `highway_physics_torque.py`  
  Purpose: build physics torque surfaces (wheel/engine) and hybrid surfaces; downshift gains; summaries; ZIP pack.  
  Inputs: prepped dir. Outputs: physics/hybrid surfaces, summaries, zip.
- `highway_torque_surface.py`  
  Purpose: legacy ECM torque surface builder (air/spark/KR) by gear/speed. Inputs: prepped. Outputs: torque surfaces.
- `highway_super_analysis.py`  
  Purpose: occupancy, torque deficit, virtual schedule sim, intent frustration, TCC heat. Inputs: prepped, torque surface, optional episodes/schedule. Outputs: highway_super_analysis__*/ CSV pack.
- `torque_error_surface_from_log.py`  
  Purpose: physics vs ECM torque error surfaces (gear×RPM/speed), TCC locked. Inputs: raw log. Outputs: `torque_error_surface__*`.
- `torque_error_surface_v2_with_roadload.py`  
  Purpose: same as above but with road‑load forces (roll + aero). Inputs: raw log. Outputs: v2 surfaces.
- `highway_intent_and_harshness.py --run-torque-deficit` (in super)  
  Purpose: torque deficit integral (“pain map”) from torque surface vs actual gear.

## ABS / TCS / traction
- `abs_tcs_heavy_throttle_scan.py`  
  Purpose: scan heavy throttle segments for ABS/TCS activation. Inputs: logs. Outputs: CSV summary.
- `tcc_triggers_burblock3.py` (context)  
  Notes: includes TCS fields in transition context (tcs_request/system/desired torque) for traction-related unlocks.

## Corner / chassis gated
- `corner_exit_pass_weighted.py` (and shim variants)  
  Purpose: identify corner exit zones with chassis gates; used in table passes. Inputs: prepped. Outputs: pass TSVs.
- `corner_chassis_pass.py` / `corner_debug_scan.py`  
  Purpose: chassis‑aware corner handling; diagnostics for steering/lat_g/yaw gates.

## “Super” orchestrators (highway packs)
- `highway_trans_MAX_analysis.py`  
  Purpose: prepped log builder + MAX pack (shift/TCC events, drag segments, usage summaries). Inputs: CLEAN_FULL. Outputs: prepped + MAX CSVs/ZIP.
- `highway_super_analysis.py`  
  Purpose: run occupancy/torque deficit/virtual sim/intent frustration/TCC heat; writes timestamped output dir.
- `highway_harshness_deepdive.py`  
  Purpose: (listed above) overlays on harshness heatmap.

## Fuel / MPG overlays
- `tcc_fuel_economy__burblock3.py`  
  Purpose: MPG vs speed/gear/pedal/TCC (psi‑aware). Inputs: raw log. Outputs: fuel_vs_speed* CSVs + summary TXT.

## Mapping to four requested analysis ideas
- TCC wear / durability (psi+slip+torque/temp):  
  Implemented by `tcc_truth_with_psi.py`, `tcc_speed_binned_with_psi.py`, `classify_tcc_coupling_modes.py`, `tcc_highres_surface__burblock3.py`, `tcc_triggers_burblock3.py`, `tcc_light_accel_decision_surface__burblock3.py`, `tcc_fuel_economy__burblock3.py` (psi-aware lock fractions). Could be enhanced by adding torque/temp overlays, but core plumbing exists.
- ABS/TCS vs torque:  
  Partially covered by `abs_tcs_heavy_throttle_scan.py` and TCS fields in `tcc_triggers_burblock3.py` context windows. A dedicated traction‑vs‑torque map (speed/pedal/gear with TCS torque requests) is not present yet → gap.
- Corner exit with chassis gates:  
  Partially implemented via `corner_exit_pass_weighted*.py`, `corner_chassis_pass.py`, `corner_debug_scan.py` (table-pass oriented). No standalone analytics report; would need a new summarizer for shift/TCC behavior on corner exit → gap.
- Torque sanity vs physics/hybrid surfaces:  
  Implemented by `highway_physics_torque.py` (physics/hybrid surfaces), `torque_error_surface_from_log.py`, `torque_error_surface_v2_with_roadload.py` (per-log error surfaces), and super-analysis torque deficit. Coverage is good; could add a simple “lie detector” map per gear×speed comparing delivered torque to physics/hybrid to flag outliers.

## Notes / reminders
- Most highway tools expect prepped or prepped_NOCRUISE logs with canonical columns (speed_mph, pedal_pct, throttle_pct, gear_actual__canon, time_s, brake).
- Psi-aware TCC state: use `tcc_state_utils.py` classifier (NaN psi dropped, psi==0 → OPEN, psi>0: <=50 rpm LOCKED, >50 PARTIAL).
- Heavy outputs are already organized under `newlogs/burblock5_analysis_NORMAL` and `newlogs/output/...`; trimmed bundles exclude large raw logs.

README updated to capture the current analysis tooling landscape. Use this as the quick reference before adding new scripts or rerunning passes.
