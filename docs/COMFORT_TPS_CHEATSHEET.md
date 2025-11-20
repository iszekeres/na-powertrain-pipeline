# Comfort Mode TPS Cheat Sheet (from test1/test2 logs)

Context:

- Logs: `test1` and `test2` FULL cleans (`__trans_focus__clean_FULL__*.csv`)
- Mode: **Comfort / Normal** only (`mode_is_pattern_a == 0`)
- Gears: forward gears 1–6
- Signals: `throttle_pct`, `speed_mph`, `gear_actual__canon`
- Phases (derived from d(speed_mph)/dt):
  - `cruise`: near-constant speed
  - `accel`: speeding up
  - `decel`: slowing (ignored for this cheat sheet)

The goal is to have a quick mental map from **TPS %** to **“what kind of driving this usually is”** in comfort mode.

---

## TPS bands (comfort mode)

These bands are approximate and based on patterns seen in the test logs; they are not hard limits.

### 0–5% TPS — Coast / micro corrections

- Situation:
  - Foot barely on the pedal or fully off.
  - Used to maintain speed on a slight downhill or when the truck is already rolling.
- Feel:
  - Engine doing almost no work.
  - Good for maximum fuel economy but can feel a bit lazy if you stay here too long.

### 5–10% TPS — Very light cruise

- Situation:
  - Neighborhood / light city streets or very gentle highway cruise.
  - 2nd–3rd gear: low-speed rolling (around neighborhood speeds).
  - 4th–6th gear: lower-speed cruise or gentle highway.
- Feel:
  - “I’m moving, but I’m not in a hurry.”
  - Smooth, soft response.

### 10–20% TPS — Normal cruise (main comfort band)

- Situation:
  - This is the primary **comfort-mode cruise** band.
  - 3rd gear: typical city speeds.
  - 4th gear: suburban / 30–40 mph zones.
  - 5th gear: ~40 mph and up.
  - 6th gear: ~50–55+ mph highway cruising.
- Feel:
  - Feels “normal” and relaxed.
  - The engine has enough torque to hold speed on gentle grades.

### 20–30% TPS — Mild hill / gentle acceleration

- Situation:
  - Slight uphill on the highway or brisker city driving.
  - 3rd–4th gear: stronger neighborhood / town accel.
  - 5th–6th gear: adding a bit more torque on the highway.
- Feel:
  - Noticeable acceleration, but not aggressive.
  - Good for “I want to pick up speed a bit” without waking the whole neighborhood.

### 30–40% TPS — Decent accel / merge behavior

- Situation:
  - Merging, short on-ramps, or passing slower traffic without going full send.
  - 2nd–3rd gear: strong local accel.
  - 4th–6th gear: highway merge / passing behavior.
- Feel:
  - Clearly “I’m asking for power now.”
  - Still in comfort mode, so it’s not as aggressive as Pattern A, but this is the upper half of “normal driving.”

### 40–60% TPS — Strong accel / passing in comfort

- Situation:
  - Committed passing, short gaps, or needing to get moving.
  - Uses more of the torque band but still under the comfort pattern.
- Feel:
  - Pulls hard for a comfort mode.
  - If this is used frequently, Pattern A might be a better match for that driving.

### 60–100% TPS — Heavy accel / near-WOT (comfort)

- Situation:
  - Rare in comfort mode in the current logs.
  - When it appears, it’s “I mashed it but forgot I’m not in Pattern A.”
- Feel:
  - The truck will still go, but the true “aggressive WOT” behavior is better represented in **Pattern A** (performance mode).

---

## How to use this when tuning

When editing **Comfort (Normal) mode** shift and TCC tables:

- Treat TPS rows approximately as:
  - **10–20%** → “true cruise” behavior.
  - **20–30%** → “normal accel & mild hills.”
  - **30–40%** → “firm accel / merges.”
  - **40–60%+** → “strong accel / passing.”

- If you don’t like how it feels at:
  - Light throttle on the highway → adjust 10–20% & 20–30% rows in 5th/6th.
  - City accel from a stop → adjust 10–20% & 20–30% rows in 2nd–4th.
  - Merges/passes in comfort → look at 30–40% & 40–60% rows in 3rd–6th.

Pattern A remains the “rowdy” mode; comfort should stay smooth and low-rpm, using these bands as anchors for behavior.
