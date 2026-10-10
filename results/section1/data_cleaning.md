# Section 1 — Raw Data Cleaning: House 1

**Author:** Vijay Rameshkumar  
**Dataset:** REFIT Electrical Load Measurements (University of Strathclyde)  
**Scope:** House 1 raw data only — Part 1 + Part 2 combined  
**Output:** `data/processed/house1_clean_1min.parquet`

---

## 1. What the Assignment Asks For

The assignment specifies three concrete deliverables for Section 1:

1. **Inspect** — check timestamp parsing and ordering, duplicate timestamps, missing readings, irregular intervals, and impossible/extreme values; summarise the findings.
2. **Build** — a documented pipeline that sorts readings, removes duplicates with a stated rule, handles short missing gaps, and flags longer outages rather than filling them without justification.
3. **Plot** — aggregate power and at least two appliance channels before and after cleaning.

The key instruction is *"you are not expected to reproduce every specialised correction in the official REFIT cleaning pipeline."* The goal is a manageable, reproducible pipeline with clear reasoning — not a perfect reconstruction.

---

## 2. Mental Model of the Data

### 2.1 How REFIT was collected

REFIT monitored 20 UK households (Loughborough area) from October 2013 to June 2015 using an 8-second polling gateway. The gateway logged a new row **only when a sensor value changed**, so intervals vary from 1 s (rapid change) to hours (nothing changed). This event-driven design means the raw data is:

- **Irregular in time**: you cannot assume consecutive rows are 8 seconds apart.
- **Multi-sensor asynchronous**: the aggregate clamp and 9 individual appliance monitors (IAMs) each update independently. One real "moment" may produce 9 separate rows, each 1–2 seconds apart.
- **Sparse by design**: long stable periods (overnight, during outages) produce very few rows.

### 2.2 Part 1 vs Part 2 — the missing-value encoding problem

House 1 comes in two files with a ~4-month overlap (June–October 2014):

| File | Period | Missing encoding |
|---|---|---|
| Part 1 | Oct 2013 – Oct 2014 | `0` — cannot distinguish from genuine zero load |
| Part 2 | Jun 2014 – Jul 2015 | `NaN` — distinguishable from genuine zero |

This is the most subtle data quality issue in Section 1. A `0` reading in Part 1 on the fridge channel could mean the fridge is off (real zero) or the sensor was disconnected (missing). We **cannot resolve this ambiguity** for Part 1. The pipeline treats Part 1 zeros as genuine and notes this as a remaining concern.

### 2.3 The clamp vs IAM signal mismatch

The aggregate is measured by a current clamp at the consumer unit; individual appliances are measured by plug-in IAMs. They are not on the same circuit and their sampling is not synchronised. A consequence:

- Clamp artifacts (firmware overflow, EMI) appear only in the aggregate — they are `2^16 - 1 = 65,535` W uint16 overflows.
- IAM artifacts appear only in appliance channels — typically `98,301` W readings (likely `2^17 - 1` in a different firmware variant).
- When we cross-validate: during aggregate spike events (>5,000 W), the sum of all 9 IAMs is only **73 W median**. A 136× mismatch is definitive hardware artifact evidence — not real consumption.

### 2.4 Why resampling to 1-minute is necessary

The raw data is at ~8-second irregular intervals. Downstream tasks (cycle detection, NILM modelling) need a regular time axis. One-minute bins are the right choice:

- Fine enough to resolve washing machine cycle phases (~5–15 min per phase).
- Coarse enough to reduce noise from multi-sensor polling jitter.
- Matches the resolution required by Sections 2 and 3 of the assignment.

Each 1-minute bin is filled by the **mean** of all valid 8-second readings in that minute (after removing spikes). If a bin has no valid readings, it is NaN — a gap.

---

## 3. Issues Found in the Raw Data

The pipeline runs a full detection pass before touching the data. Below is the complete findings summary from House 1.

| Finding | Count | Details |
|---|---|---|
| Total rows (Part 1 + Part 2) | 8,518,763 | Oct 2013 – Jul 2015 |
| Rows out of time order | 2 | Sort applied (R1) |
| Duplicate Unix timestamps (same aggregate) | 738,219 | Sensor-polling collision; keep first (R2) |
| Duplicate Unix timestamps (different aggregate) | 217,376 | Ambiguous; rolling-median tie-break (R2) |
| Aggregate spikes > 15,000 W | 186 | Rolling MAD context filter (R3/R4) |
| IAM spikes > 4,000 W | 3,950 | Hard physics threshold (R3); IAMs cannot exceed this |
| Gaps 5–30 min (short outage) | 362 | Linear interpolation (R5) |
| Gaps 30 min – 24 h (SARIMA-fillable) | 16 | SARIMA time-series imputation (R6) |
| Gaps > 24 h (flag only) | 12 | Outage flag; no imputation (R7) |
| Largest gap | 998.4 h | ~6 weeks; sensor disconnected |
| Flatline runs ≥ 30 min (per appliance) | Multiple | Flagged as `flatline_suspect`; not removed (R8) |
| Part 1 zero-vs-missing ambiguity | Unresolvable | Remaining concern noted in output |

---

## 4. Pipeline Decisions and Rationale

### R1 — Sort by Unix timestamp
Raw data has 2 out-of-order rows. Sorting ensures the gap analysis and rolling window operations produce correct results.

### R2 — Duplicate Unix timestamp resolution
The same Unix second can have multiple rows because different sensors (aggregate, fridge, WM) each update independently and may land in the same second.

- **Same aggregate value**: trivially identical reads; keep first.
- **Different aggregate value**: genuinely ambiguous (sensor updated mid-poll). We keep the reading **closest to the rolling 11-point median** of the aggregate. This preserves the most representative reading rather than arbitrarily taking first or last.

**Why not take the mean?** The mean of a legitimate reading and a spiked reading would produce a partially-contaminated value. The rolling-median tie-break is robust to the spike.

### R3 — Physics-based spike filter (IAMs)
Any IAM reading above 4,000 W is physically implausible for a domestic plug-in appliance. These are set to NaN. The threshold matches the official REFIT cleaning pipeline and is supported by the UK domestic appliance maximum (a 13A socket at 230V = 2,990 W; the 4,000 W limit provides a safe margin for multi-socket adaptors).

### R4 — Context-aware spike filter (Aggregate)
The aggregate clamp is not capped by the official clean pipeline. We apply a **rolling MAD (Median Absolute Deviation) filter**:

1. Compute a 45-minute rolling median of the aggregate.
2. Compute rolling MAD of the deviation from that median.
3. Flag readings more than 5× MAD above the rolling median (when MAD > 10 W to avoid false positives during stable periods).

This is context-aware — a spike is only flagged if it deviates substantially from local behaviour. A genuine large load (oven + WM + shower simultaneously) will not be flagged because the baseline will shift with it over time.

**Cross-validation result**: at the 650 spike events (>5,000 W) in November 2013, the sum of all 9 IAMs was 73 W median vs 9,964 W aggregate median — a 136× ratio. The spikes are confirmed clamp artifacts, not real consumption.

### R5 — Short gap fill (5–30 min, linear interpolation)
Gaps of 5–30 minutes are unlikely to contain significant structural change — the household is probably mid-activity. Linear interpolation between the last known value and the next known value is appropriate and conservative. These are labelled `linear_med` in the output.

**Why not SARIMA for all gaps?** SARIMA fitting takes ~15 s per gap. With 349 short gaps, that would add ~87 minutes of compute for fills that are statistically indistinguishable from linear. The 30-minute threshold is where diurnal structure starts to matter.

### R6 — Medium/long gap fill (30 min – 24 h, SARIMA)
Gaps of 30 minutes or longer span enough time that the household's diurnal pattern matters. A flat linear fill from 165 W (midnight) to 280 W (afternoon) would miss the expected morning energy ramp. SARIMA captures this pattern.

**Model selection**: `auto_arima` with a two-pass approach (Pass 1 on all available data → remove residual outliers → Pass 2 refit) found orders `(2,1,0) × (1,0,1,24)` with daily seasonality. The anchor-point fix ensures sub-hourly gaps are covered by prepending the last known reading before forecasting.

**Result**: 16 SARIMA fills, 1,862 total imputed minutes. The largest fill (Nov 10, 2014, 777 min / 13 h) correctly reproduced the diurnal pattern — SARIMA forecast ended at ~300 W, actual signal resumed at ~280 W (7% error at the boundary).

### R7 — Long gap flag (>24 h, outage)
Gaps exceeding 24 hours cannot be reliably imputed. SARIMA uncertainty compounds over multi-day horizons. These rows are set to `outage = 1` and left as NaN. The pipeline correctly identified 12 such gaps, the largest being 998 h.

### R8 — Flatline detection
Runs of ≥30 consecutive minutes at exactly the same value on any channel are flagged as `flatline_suspect`. This catches the forward-fill artifact introduced by the official clean pipeline (all missing values were forward-filled in the clean data). Flagged rows are **not removed** — they may be genuine (a device held exactly constant) — but downstream models should exclude or down-weight them.

### R9 — Hierarchical cross-check (validation only)
We compute the ratio of the aggregate to the sum of all 9 IAMs at every minute. The median ratio is ~4.7× in normal operation — expected, because lights, sockets, and unmonitored devices are not captured by the 9 IAMs. We use this cross-check to **validate** the anomaly detection (spikes appear only in the aggregate, not in IAMs) but do **not** reconcile the two signals. Hierarchical reconciliation would pull the correct IAM readings toward the bogus aggregate spike values.

---

## 5. Output Statistics

| Metric | Value |
|---|---|
| Total 1-minute rows | 920,031 |
| Known (sensor readings) | 913,939 (99.3%) |
| Linear fills (5–30 min gaps) | 4,230 min (349 gaps) |
| SARIMA fills (30 min–24 h gaps) | 1,862 min (16 gaps) |
| Outage flagged (>24 h gaps) | 110,705 min (1,845 h across 12 gaps) |
| Flatline suspect | 114,415 rows |
| Issues flagged | 8,938 rows |
| Largest gap | 998.4 h (not imputed) |

---

## 6. Validation Plots

Four plots saved to `figures/`:

| Figure | What it shows |
|---|---|
| `C1_pipeline_overview.png` | Bin disposition at each pipeline stage (left) and row count funnel (right) — 8.5M raw rows → 920k 1-min bins, with imputation breakdown |
| `C1_quality_breakdown.png` | Three-panel daily timeline: imputation source (observed / linear / SARIMA), quality flags (outage / flatline), and WM ON activity |
| `C1_before_after.png` | One representative week — aggregate with quality flags overlaid (top) vs cleaned aggregate + WM disaggregation (bottom) |
| `C1_sarima_zoom.png` | Zoom on longest SARIMA gap — observed boundary transitions and imputed region highlighted |

---

## 7. Remaining Concerns

1. **Part 1 zero-vs-missing ambiguity**: In Part 1 (Oct 2013 – Oct 2014), a `0` on any appliance channel cannot be distinguished from a genuine zero load vs a missing sensor reading. This affects ~4.4M rows. Training a NILM model on Part 1 data risks learning "zero = off" when "zero = sensor dropout."

2. **Flatline artifacts from official clean pipeline**: The official clean REFIT data forward-fills all missing values. This means any long gap in clean data becomes a flat line. Models trained on clean data may learn flatline patterns as legitimate appliance behaviour. The `flatline_suspect` flag addresses this.

3. **BST/UTC ambiguity in raw timestamps**: Raw timestamps are local UK time (BST in summer, GMT in winter). We work in raw local time throughout Section 1 and note the UTC shift. Downstream models should use consistent timezone handling.

4. **SARIMA uncertainty**: SARIMA imputation is probabilistic. The 1,862 imputed minutes carry forecast uncertainty not reflected in the final output. They should be excluded from NILM model training and from evaluation windows.

5. **Unmonitored loads**: The 9 IAMs cover only 9 appliances. The aggregate always includes unmonitored loads (lighting, socket devices, EV chargers if present). The median unmonitored fraction is ~79% of aggregate, varying widely by time of day.
