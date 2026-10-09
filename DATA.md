# DATA.md — Raw Data Coverage, Quality Findings, and Cleaning Decisions

---

## Dataset Overview

**REFIT Electrical Load Measurements Dataset**
- 19 UK households (Houses 1–21, skipping 12 and 14)
- Collection period: October 2013 – June 2015 (~20 months)
- Sampling interval: 8 seconds
- Channels per house: whole-house aggregate + 9 individual appliance sub-meters (IAMs)
- House 1 washing machine: Appliance 5 channel

**Two-part structure with different missing-value conventions:**

| Part | Period | Missing value encoding |
|---|---|---|
| Part 1 | Oct 2013 – Apr 2014 | Zero (0) = missing |
| Part 2 | Apr 2014 – Jun 2015 | NaN = missing |

This distinction is critical: in Part 1, genuine zero-watt readings (house asleep at
3am) and missing readings are encoded identically. Part 2 uses an unambiguous NaN.
All model training uses Part 2 only.

---

## Raw Data Coverage — House 1

| Metric | Value |
|---|---|
| Total raw rows (both parts) | 8,533,035 |
| Out-of-order timestamps | 2 |
| Duplicate timestamps | 956,000 |
| Impossible IAM readings (> 4,000W) | 3,350 |
| Aggregate clamp spikes removed | 45+ |
| 1-min bins after resample | 920,031 |
| Short gaps filled (≤ 30 min, linear) | 349 gaps |
| SARIMA-imputed (30 min – 24 h) | 1,862 minutes |
| Outages flagged (> 24 h, not filled) | 12 gaps, 1,845 hours total |

---

## Cleaning Pipeline — Seven Rules

### R1: Sort by timestamp
Raw rows are not guaranteed to be chronological. Out-of-order rows (2 found in
House 1) cause negative time differences that break rolling calculations and
resampling. Sort applied first before any other operation.

### R2: Remove duplicate timestamps
956,000 duplicate timestamps found in House 1. Duplicates arise from overlapping
collection batches when the IAM hardware uploads buffered readings. Keeping any
duplicate value is arbitrary; duplicates are dropped, retaining only the first
occurrence per timestamp.

### R3: Remove physically impossible readings

**IAM channels**: individual appliance readings above 4,000W are physically
impossible for UK domestic plug-socket appliances (13A fuse × 230V = 2,990W max).
The fridge sub-meter showed a spike to ~900W — a domestic fridge compressor draws
100–150W maximum. Removed.

**Aggregate channel**: values above 20,000W are not achievable in a UK domestic
supply (100A main fuse × 230V = 23,000W theoretical max, but practical household
peak is 6–8 kW). Readings above 20,000W are sensor artefacts, removed.

Threshold selection is grounded in UK electrical standards (BS 7671) and product
specifications, not tuned to the data distribution.

### R4: Resample 8-second → 1-minute
Mean aggregation within each 1-minute bin. Bins where more than 50% of the
constituent 8-second readings are missing (NaN or zero in Part 1) are treated as
missing at the 1-minute level. This produces 920,031 1-minute bins for House 1.

**Why 1-minute?** The assignment specifies 1-minute resolution. Additionally, the
NILM model operates at 1-minute intervals: coarser resolution loses cycle-phase
texture, finer resolution is unnecessary for whole-cycle energy accounting.

### R5: Fill short gaps (≤ 30 min) with linear interpolation
349 gaps of 1–30 minutes were found in House 1. These typically arise from IAM
connectivity dropouts. Linear interpolation between the boundary values is
appropriate: over 30 minutes, household consumption transitions smoothly without
sudden structure that SARIMA would introduce. These gaps are flagged with
`impute_source = 'linear'`.

### R6: Fill medium gaps (30 min – 24 h) with SARIMA
1,862 minutes imputed via SARIMA(2,1,2)(1,1,1,1440). The seasonal period of 1,440
(minutes per day) captures the daily usage cycle. For a 3-hour outage at 6am, linear
interpolation would produce a flat line; SARIMA generates a realistic consumption
trace that follows the expected early-morning ramp-up pattern. These gaps are flagged
with `impute_source = 'sarima'`.

**Limitation**: SARIMA forecasts are plausible but not ground truth. Any appliance
events that genuinely occurred during a SARIMA-imputed window are not captured.

### R7: Flag outages (> 24 h) — not filled
12 outage gaps found in House 1, totalling 1,845 hours (~77 days). These are
excluded from all downstream analysis. Filling gaps of this duration would require
extrapolating consumption patterns across multiple days — too uncertain to be useful
and potentially misleading for model training.

**Important**: outages are not missing at random. They cluster in winter months
(higher consumption periods, more grid instability). Any analysis of seasonal
patterns in House 1 data should account for this systematic exclusion.

---

## Cleaning Decisions and Justifications

### Why SARIMA instead of forward-fill or zero-fill?

Forward-filling would copy the last known value across the gap, creating an
unrealistic flat period. Zero-filling would suggest the house had no consumption
during the gap, which is false and would teach the model that zero aggregate periods
are normal at unexpected times. SARIMA generates values consistent with the household's
established temporal patterns.

### Why not impute outages (> 24 h)?

Beyond 24 hours, the uncertainty in SARIMA forecasts compounds to the point where
imputed values would bear little relationship to actual consumption. The model is
better served by seeing no data than by seeing fabricated data presented as real.
Additionally, 1,845 hours of outages across House 1 represents 10.6% of the total
time span — imputing this would materially change the training distribution.

### Why retain Part 2 only for modelling?

In Part 1 (Oct 2013 – Apr 2014), zeros encode missing values. There is no reliable
way to distinguish genuine zero-watt readings (all appliances off) from missing
readings. Training on Part 1 would expose the model to thousands of consecutive
zero-aggregate periods that are actually data gaps, teaching it that zero is a common
aggregate state. Part 2 uses NaN unambiguously. We use Part 2 for all model training.

Part 1 data is used in the Section 1 cleaning analysis and the before/after
visualisation to demonstrate the cleaning pipeline, but not for model training.

---

## Remaining Concerns

**Systematic outage bias**: the 12 outages in House 1 are not evenly distributed
across seasons. Winter months have more gaps. This means the test set over-represents
spring and autumn consumption patterns relative to a full year. Model performance
in winter (when WM usage may differ — more hot washes for cold-weather clothing)
may be lower than reported metrics suggest.

**Part 1 zero ambiguity in training houses**: we use Part 2 only for model training,
but Houses 2–19 also have Part 1 data with the same zero-ambiguity problem. Training
only on Part 2 limits the training window to April 2014 – June 2015 (14 months)
and excludes all October 2013 – April 2014 data across all houses.

**IAM channel quality**: IAMs are plug-level monitors that measure current at the
socket. They do not account for standby draw from appliances with permanent physical
connections (hardwired appliances). In a minority of REFIT households, some appliance
channels show systematic offsets or occasional step-changes that suggest sensor drift
rather than real consumption changes. These were not addressed beyond the impossible-
value filter in Rule R3.

**SARIMA stationarity assumption**: SARIMA assumes that the consumption series is
stationary after differencing. Long-term trends (occupancy changes, new appliance
purchase, seasonal shifts) within a 14-month Part 2 window may violate this
assumption. Imputed values during long trends may not reflect the actual level of
consumption at the time of the gap.
