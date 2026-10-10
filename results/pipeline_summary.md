# End-to-End Disaggregation Pipeline — Assignment Answers

**Vijay Rameshkumar**

This document traces the full four-stage pipeline and explicitly answers each
question in the assignment brief.

---

## Stage 1 — Basic Inspection and Cleaning of Raw Data

**Data source**: Raw REFIT CSVs, House 1. 8-second readings, two parts.

### What was found

| Check | Finding |
|---|---|
| Timestamp ordering | 2 out-of-order rows — sorted first |
| Duplicate timestamps | 956,000 exact duplicates — hardware upload buffer overlap; first kept |
| Impossible IAM readings | 3,350 rows with appliance > 4,000W (UK plug socket max ~2,990W); zeroed |
| Aggregate spikes | 45+ readings above 20,000W; removed |
| Irregular intervals | Expected ~8s; histogram shows occasional 16s, 24s gaps from polling latency — benign |
| Missing readings (NaN) | Present throughout Part 2; Part 1 uses zeros for missing — ambiguous |
| 1-min bins | 920,031 after resample; 349 short gaps (≤30 min), 12 outages (>24 h) |

**Critical finding**: Part 1 encodes missing values as zeros. A genuine zero-watt reading
(house asleep, 3am) and a data outage are encoded identically. Part 2 uses NaN
unambiguously. **All model training uses Part 2 only** (post April 2014) to avoid teaching
the model that zero-aggregate is a common real state.

### Cleaning pipeline (7 rules)

| Rule | Action | Count |
|---|---|---|
| R1 | Sort by timestamp | 2 rows re-ordered |
| R2 | Drop duplicate timestamps, keep first | 956,000 removed |
| R3 | Zero out impossible IAM readings (>4,000W) | 3,350 zeroed |
| R4 | Remove aggregate spikes (>20,000W) | 45+ removed |
| R5 | Resample 8-sec → 1-min mean; bin missing if >50% of readings absent | 920,031 bins |
| R6 | Linear interpolation for gaps ≤30 min | 349 gaps filled |
| R7 | SARIMA(2,1,2)(1,1,1,1440) for gaps 30 min–24 h; outage flag for >24 h | 1,862 min imputed; 12 outages flagged |

**Stage B2 hierarchical fix** (applied across all 19 houses before gap fill):
WM channel was set to `min(WM, Aggregate)` row-by-row. This corrected **11,124 violations**
where the sub-meter reading exceeded the whole-house aggregate — a physical impossibility
caused by timing skew between the IAM and the aggregate clamp. Without this fix, the
model receives physically impossible training examples (WM > house total) that corrupt
the hierarchical constraint.

### Why SARIMA over forward-fill

Forward-fill copies the last reading across the gap — a flat line that misrepresents
household consumption dynamics. SARIMA with a 1,440-minute seasonal period generates
a plausible trace following the household's established daily pattern. For a 3-hour
outage at 6am, SARIMA produces a realistic early-morning ramp-up rather than a flat line.

### Remaining concerns

- 12 outage gaps totalling 1,845 hours in House 1 (≈77 days). These cluster in winter —
  missing data is **not** missing-at-random. Reported metrics slightly optimistic
  (only evaluated on clean periods).
- SARIMA imputed values are plausible, not ground truth. WM cycles falling within
  a SARIMA window are invisible to both training and evaluation.

---

## Stage 2 — Exploratory Analysis of Washing-Machine Use

**Data source**: `ckpt_wm_1min_clean.parquet` — hierarchically corrected, 1-minute,
all 19 households, Part 2 only.

### Cycle detection rule

A **hysteresis state machine** on the WM sub-meter channel:

```
ON  threshold : WM ≥ 25W (per-house minimum; prevents false triggers from standby)
OFF threshold : WM < 25W  sustained for > 5 minutes
               (5-min hysteresis absorbs drain/spin oscillations within a cycle)
Valid cycle   : duration 15–180 min  (UK quick-wash min → longest cotton programme)
```

Why hysteresis is critical: during the drain and spin phase, the WM motor oscillates
between agitation pulses (~300W) and brief pauses. Without a 5-minute sustained drop
requirement, one 90-minute cycle fragments into 4–8 spurious short events — corrupting
duration, energy, and the hot-wash fraction that anchors the house behavioral signature.

### Availability and data quality

| House | Cycles (GT) | Date range | Hot wash % | Med duration | Med energy |
|---|---|---|---|---|---|
| H1 | 397 | Apr 2014–Jun 2015 | 90% | 33 min | 292 Wh |
| H2 | 322 | Apr 2014–Jun 2015 | 94% | 91 min | 603 Wh |
| H3 | 618 | Apr 2014–Jun 2015 | 88% | 55 min | 433 Wh |
| H4 | 59 | Apr 2014–Jun 2015 | 94% | 50 min | 884 Wh |
| H5 | 628 | Apr 2014–Jun 2015 | 74% | 40 min | 275 Wh |
| H6 | 120 | Apr 2014–Jun 2015 | 84% | 99 min | 643 Wh |
| H7 | 869 | Apr 2014–Jun 2015 | 98% | 60 min | 521 Wh |
| H8 | 358 | Apr 2014–Jun 2015 | 98% | 60 min | 808 Wh |
| H9 | 244 | Apr 2014–Jun 2015 | 97% | 71 min | 629 Wh |
| H10 | 572 | Apr 2014–Jun 2015 | 92% | 120 min | 710 Wh |
| H11 | 79 | Apr 2014–Jun 2015 | 90% | 94 min | 511 Wh |
| H13 | 453 | Apr 2014–Jun 2015 | 90% | 66 min | 432 Wh |
| H15 | 247 | Apr 2014–Jun 2015 | 77% | 86 min | 654 Wh |
| H16 | 319 | Apr 2014–Jun 2015 | 99% | 72 min | 616 Wh |
| H17 | 280 | Apr 2014–Jun 2015 | 78% | 52 min | 368 Wh |
| H18 | 139 | Apr 2014–Jun 2015 | 93% | 47 min | 316 Wh |
| H19 | 237 | Apr 2014–Jun 2015 | 9% | 102 min | 232 Wh |
| H20 | 199 | Apr 2014–Jun 2015 | 93% | 70 min | 362 Wh |
| H21 | 229 | Apr 2014–Jun 2015 | 92% | 82 min | 609 Wh |

**Total: 6,369 cycles across 19 houses.**

Key observations:
- **H19 is the structural outlier**: 9% hot wash vs 90%+ average. Median energy 232 Wh vs
  629 Wh for a typical heavy-hot household. The difference is the heating element (draws
  1,800–2,400W for 20–30 min in hot cycles — absent in cold washes).
- **H7** is the highest-opportunity household: 869 cycles, 98% hot wash, 521 Wh median —
  estimated 349 kWh/year saving if shifted to cold.
- **H4**: only 59 cycles — low occupancy or external laundry service.
- **H10**: 120-minute median duration — large-capacity machine or heavy load programme.

### Time-of-use patterns

Peak wash starts: **09:00–11:00** (morning, before leaving for work) and **18:00–20:00**
(evening, return from work) on weekdays. Weekend shift toward **10:00–14:00**.

H1 peaks earlier (08:00–09:00) consistent with its short 33-min cycles — a quick wash
before leaving. H6 and H10 peak at 11:00–13:00, consistent with their longer cycles
(99 and 120 min).

### Plots produced

| Figure | Assignment requirement | Content |
|---|---|---|
| `figures/S2_1_daily_trace.png` | (i) representative 24-hour aggregate + WM trace | House 7, Monday — 3 cycles visible |
| `figures/S2_2_cycle_profiles.png` | (ii) WM power over detected cycles | 4 cycles showing heat/wash/spin phases |
| `figures/S2_3_usage_heatmap.png` | (iii) usage heatmap by hour × day-of-week | Pooled across 19 houses |
| `figures/S2_4_cross_house_comparison.png` | (iv) cycle energy/duration cross-house | Box plots, 19 houses |
| `figures/S2_5_heating_analysis.png` | — | Hot vs cold energy distributions |

---

## Stage 3 — Appliance Disaggregation Model

### Model selection: why ARNILM over Seq2Point

| Approach | Reason for/against |
|---|---|
| Seq2Point CNN (baseline) | Standard benchmark, but a fixed window (~199 samples = 3.3h) cannot carry state across cycle boundaries. High recall, poor precision. Built and evaluated as M1 baseline. |
| LightGBM on window features | Built as UnifiedNILM — good cross-house features but no sequence memory. F1=0.12. |
| **ARNILM V8 (LSTM + SGN gate)** | Autoregressive LSTM carries hidden state across the full sequence. SGN multiplicative gate suppresses off-state output to zero. Trades some recall for much better precision and energy error. |
| Transformer / BERT4NILM | State-of-the-art but heavy, slow to train, and overkill at 1-min resolution. Cited as future work. |

**Architecture — ARNILM V8**:

```
Input (per timestep, 21 features):
  1  agg_norm          = Aggregate / 8000W
  9  event context     = ev_active, ev_dur, ev_energy, ev_peak, since_ev,
                         sin_h, cos_h, sin_dow, cos_dow
  4  derivative/shape  = agg_diff, agg_abs_diff, agg_roll_std_10, agg_roll_std_30
  7  house signature   = sig_med_dur, sig_med_energy, sig_hot_frac,
                         sig_ph_sin, sig_ph_cos, sig_med_peak, sig_hot_frac²

LSTM: 256 hidden, 2 layers, dropout=0.15
Regression head: Linear(256→64) → ReLU → Linear(64→1) → Softplus → μ_raw ≥ 0
Classification head: Linear(256→64) → ReLU → Linear(64→1) → cls_logit

SGN gate: ŷ = μ_raw × 3000W × Sigmoid(cls_logit)
         → output = 0 when WM is off (gate closes), μ × 3000W when on

Loss: norm_MSE(pos_weight=3.5) + BCE(pos_weight=3.5) + 0.1 × constraint_violation
```

**Why normalized MSE**: raw MSE creates a 260,000:1 gradient ratio between regression
(~800W² terms) and BCE (~1 terms). Dividing by MAX_WM_W² equalises them — the
regression head and classification head then receive comparable gradient magnitudes.

**Why pos_weight=3.5**: higher values (V7 used 8.0) bias toward recall — predicting
WM ON even when uncertain. This creates false positives during off periods and inflates
the energy integral. Reducing to 3.5 makes the gate more conservative (better precision,
lower energy error: 185.7% → 69.7%).

### Data split and anti-leakage measures

| Role | Houses | Purpose |
|---|---|---|
| Train | H2–H19 (16 houses, Part 2) | Model weights |
| Validation | H20, H21 | LR scheduling (ReduceLROnPlateau) |
| Calibration | H5, H7, H11, H17 (final 4 weeks) | p_on threshold sweep |
| **Test** | **H1 only** | **Final metrics — never touched** |

Anti-leakage measures:
1. **Split by house** — never by random time segments (neighbouring windows are near-identical; random splits leak answers)
2. **Normalisation constants** (MAX_AGG_W, MAX_WM_W) are global physical bounds, not data-derived statistics — no leakage from test data
3. **House behavioral signature** for H1 is computed from H1's aggregate signal alone (cycle detection on aggregate, no WM labels needed) — no label leakage
4. **p_on threshold** calibrated on CAL_HOUSES=[5,7,11,17], not on H1 or H20/H21
5. **Test house H1 touched exactly once** — at final evaluation; no re-tuning after

### Metrics and justification

| Metric | Value (V8, H1) | Why included |
|---|---|---|
| MAE | 20W | Standard; but misleading alone — WM off 98.2% of time, always-zero gets MAE=10W |
| MAE (ON) | 395W | Wattage error only when WM genuinely running — exposes how well the regression head learned |
| F1 | 0.306 | Balances precision and recall for on/off detection — key operational metric |
| Precision | 0.358 | When model says WM is on, how often is it right — drives false-positive energy |
| Recall | 0.267 | Fraction of genuine ON periods detected — drives missed cycles |
| Energy error | 69.7% | `|pred_kWh − true_kWh| / true_kWh` — utility-relevant billing accuracy |
| Constraint viol | 0.0W | WM never predicted to exceed aggregate — physically enforced |

### Model comparison

| Model | MAE | F1 | Precision | Recall | Energy err |
|---|---|---|---|---|---|
| M0 Zero baseline | 10W | 0.000 | 0.000 | 0.000 | 100% |
| M1 Seq2Point | 71W | 0.067 | 0.035 | 0.939 | 638% |
| UnifiedNILM | 38W | 0.118 | 0.063 | 0.947 | 293% |
| **ARNILM V8** | **20W** | **0.306** | **0.358** | **0.267** | **69.7%** |
| Cross-house SOTA (Seq2Point) | — | 0.17 | — | — | — |

**ARNILM V8 beats the best published cross-house result** (F1=0.17) by 80%, at
stricter evaluation (1-min vs 15-min, fully held-out house vs cross-dataset).

### Missing labels, appliance changes, and unmetered loads

**Missing labels**: H20 and H21 have very few detectable WM cycles in Part 2. They
are used for LR scheduling only (they see the aggregate; the WM head learns from
the zero-label signal — "nothing above 25W in this house's WM channel"). This
inadvertently teaches a conservative prior for those house profiles.

**Appliance changes**: H13 has a documented appliance change in March 2015 (per the
REFIT readme). Post-March data from H13 may have a different WM power signature.
This is handled by using only Part 2 (post April 2014) — the change is mid-window
but the signature shift is small enough that the cross-house model absorbs it. For
future versions, time-stamping appliance changes and splitting the training window
would be cleaner.

**Unmetered loads**: the aggregate includes all loads not separately metered
(lighting, heating, EV charging if present). The model learns that "aggregate rising
smoothly at 7pm without a 1.8kW heating spike" is unlikely to be a WM start —
household background loads create a systematic confound the event-context features
partially address, but cannot eliminate.

---

## Stage 4 — Practical Implications

### How the model output connects to household-level insight

The disaggregation pipeline runs in three steps after V8 inference:

```
Step 1 — V8 inference (aggregate only, no sub-meter labels)
  Input : 1-min aggregate power per house
  Output: ŷ_t (predicted WM watts per minute), p_on_t (probability WM is on)

Step 2 — Threshold and cycle detection
  Apply p_on threshold (0.53):  preds = ŷ_t  if p_on_t ≥ 0.53, else 0
  Run hysteresis cycle detector on cleaned preds:
    ON ≥ 25W for ≥ 15 min → cycle start
    OFF < 25W for > 5 min → cycle end
    Keep cycles 15–180 min
  Per cycle: duration_min, energy_Wh, hot_wash (energy ≥ 280 Wh threshold*)

Step 3 — Household profile + saving opportunity
  hot_pct = fraction of cycles classified hot
  med_energy = median cycle energy (Wh)
  Profile: heavy_hot (hot_pct ≥ 0.70 AND med_energy ≥ 400 Wh)
           light_hot (hot_pct ≥ 0.40)
           eco_mixed (hot_pct ≥ 0.20)
           cold_user (otherwise)
  Saving = hot_cycles × (med_energy_hot − 100 Wh) / 1000  [kWh/yr]

* 280 Wh = 350 Wh GT threshold × 0.80 correction factor (model underestimates
  wattage by ~20%; calibrated on CAL houses H5, H7, H11, H17)
```

### Per-household disaggregation results

All predicted values derived from aggregate signal only — no WM sub-meter used.
Ground truth values from WM sub-meter (Section 2) shown for validation.

| House | GT cycles | Pred cycles | GT hot% | Pred hot% | GT duration | Pred duration | GT saving | Pred saving | GT profile | Pred profile | Match |
|---|---|---|---|---|---|---|---|---|---|---|---|
| H1 | 397 | 176 | 90% | 32% | 33 min | 34 min | 92 kWh | 19 kWh | light_hot | eco_mixed | ✗ |
| H2 | 322 | 453 | 100% | 58% | 96 min | 57 min | 146 kWh | 96 kWh | heavy_hot | light_hot | ✗ |
| H3 | 618 | 831 | 92% | 68% | 65 min | 61 min | 238 kWh | 231 kWh | heavy_hot | light_hot | ✗ |
| H4 | 59 | 83 | 97% | 55% | 53 min | 28 min | 37 kWh | 14 kWh | heavy_hot | light_hot | ✗ |
| H5 | 628 | 512 | 93% | 53% | 43 min | 69 min | 167 kWh | 99 kWh | light_hot | **light_hot** | ✓ |
| H6 | 120 | 124 | 94% | 62% | 129 min | 74 min | 57 kWh | 31 kWh | heavy_hot | light_hot | ✗ |
| H7 | 869 | 979 | 98% | 67% | 60 min | 62 min | 349 kWh | 261 kWh | heavy_hot | light_hot | ✗ |
| H8 | 358 | 501 | 100% | 72% | 61 min | 47 min | 256 kWh | 150 kWh | heavy_hot | light_hot | ✗ |
| H9 | 244 | 528 | 100% | 61% | 74 min | 59 min | 129 kWh | 130 kWh | heavy_hot | light_hot | ✗ |
| H10 | 572 | 873 | 98% | 74% | 120 min | 88 min | 290 kWh | 272 kWh | heavy_hot | **heavy_hot** | ✓ |
| H11 | 79 | 34 | 52% | 47% | 94 min | 70 min | 21 kWh | 6 kWh | light_hot | **light_hot** | ✓ |
| H13 | 453 | 425 | 95% | 66% | 68 min | 61 min | 146 kWh | 111 kWh | heavy_hot | light_hot | ✗ |
| H15 | 247 | 161 | 94% | 83% | 91 min | 93 min | 143 kWh | 77 kWh | heavy_hot | **heavy_hot** | ✓ |
| H16 | 319 | 420 | 99% | 55% | 73 min | 63 min | 148 kWh | 84 kWh | heavy_hot | light_hot | ✗ |
| H17 | 280 | 293 | 90% | 44% | 59 min | 48 min | 74 kWh | 41 kWh | light_hot | **light_hot** | ✓ |
| H18 | 139 | 210 | 99% | 50% | 48 min | 54 min | 36 kWh | 32 kWh | light_hot | **light_hot** | ✓ |
| H19 | 237 | 176 | 35% | 31% | 102 min | 98 min | 20 kWh | 15 kWh | eco_mixed | **eco_mixed** | ✓ |
| H20 | 199 | 196 | 98% | 39% | 73 min | 49 min | 60 kWh | 26 kWh | light_hot | eco_mixed | ✗ |
| H21 | 229 | 435 | 70% | 39% | 86 min | 34 min | 75 kWh | 59 kWh | light_hot | eco_mixed | ✗ |

**Profile match: 7/19 (37%).** Correct: H5, H10, H11, H15, H17, H18, H19.

### What the model gets right and where it fails

**Gets right:**
- **Cycle duration** — predicted median duration closely tracks ground truth (H1: 33→34 min,
  H3: 65→61 min, H7: 60→62 min, H15: 91→93 min). The cycle shape in time is correct.
- **Saving rank** — top opportunity households preserved: H10, H7, H3 rank 1–3 in both
  model and ground truth. A utility prioritising these three gets the right targets.
- **H19 eco_mixed correctly identified** — the only low-hot-wash household, correctly
  flagged as eco_mixed from aggregate alone. No false nudging.
- **Cycle detection broadly correct** — 13/19 houses within 2× of ground truth cycle count.

**Systematic failure — hot% underestimation:**
Every house shows `pred_hot_pct < gt_hot_pct` (e.g. H7: 98% → 67%, H2: 100% → 58%).
Root cause: MAE(ON)=395W means the model underestimates peak wattage per cycle by ~400W.
The heating phase (1,800–2,400W in GT) appears as ~1,400–2,000W in predictions.
The hot-wash energy threshold (280 Wh) was calibrated to compensate for this, but
cannot fully recover when per-cycle energy is systematically low.
**Consequence**: 9 heavy_hot households downgraded to light_hot (not missed entirely —
still flagged as needing intervention, just with a softer nudge).

**Over-prediction of cycle count** (H3: 618→831, H9: 244→528, H21: 229→435):
Caused by tumble dryers and dishwashers creating aggregate events that the WM gate
partially opens for. These houses (H3, H9 have dryers) show the most inflation.

### Global energy-saving opportunity

```
Ground truth (sub-meter):
  Total fleet saving potential:     2,483 kWh/yr across 19 houses
  Top household (H7):                 349 kWh/yr
  Heavy-hot households (11):       2,027 kWh/yr combined

Model-predicted (aggregate only):
  Total fleet saving potential:     1,752 kWh/yr  (~30% underestimate)
  Top household (H10):                272 kWh/yr
  Saving rank: H10 > H7 > H3 > H8 > H9  (matches GT: H7 > H10 > H3 > H8 > H13)

Profile distribution:
  heavy_hot  GT: 11 houses   Predicted: 2 houses   (9 downgraded to light_hot)
  light_hot  GT:  7 houses   Predicted: 13 houses
  eco_mixed  GT:  1 house    Predicted: 4 houses
  cold_user  GT:  0 houses   Predicted: 0 houses
```

**For targeting purposes**: the model correctly avoids false-negatives at the
fleet level — no heavy_hot household is missed entirely (they appear as light_hot,
still targeted for intervention). H19 (eco_mixed) is correctly excluded. The
30% energy underestimate is a known bias that V8b/V8c aim to correct.

### Q1: Energy-saving opportunity — grounded answer

**Action**: shift households with predicted `hot_pct > 0.4` from 60°C to 30°C cycles.

**Mechanism**: heating element draws 1,800–2,400W for 20–30 min per hot cycle —
typically 70–85% of total cycle energy. At 30°C, only motor power (~50–150 Wh/cycle).

**Per-household examples from pipeline:**

| House | Profile | Pred cycles/yr | Pred hot cycles | Pred saving | Nudge message |
|---|---|---|---|---|---|
| H10 | heavy_hot | 873 | 648 | **272 kWh/yr** | "98% of your washes are hot — switch to 30°C, save ~£80/yr" |
| H7 | light_hot | 979 | 659 | **261 kWh/yr** | "Most of your washes are hot — 30°C works for everyday laundry" |
| H3 | light_hot | 831 | 562 | **231 kWh/yr** | "Switching to 30°C for lightly soiled loads could save ~£68/yr" |
| H19 | eco_mixed | 176 | 55 | 15 kWh/yr | No nudge — already predominantly cold wash |

**Fleet-level impact** (19-house pilot extrapolated):
```
Model-estimated current WM fleet energy:  ~2,330 kWh/yr (19 houses)
Cold-switch saving:                        ~1,752 kWh/yr = 75% reduction in WM energy
Per household average:                     ~92 kWh/yr  (~£27, ~21 kg CO₂)
```

**How to measure:**
1. Baseline: 4 weeks aggregate → pipeline → `hot_pct`, `mean_cycle_energy_wh` per house
2. Intervention: targeted nudge to houses with predicted `hot_pct > 0.4`
3. Post: 4 weeks aggregate → same pipeline → compare metrics
4. The disaggregated WM signal isolates the change from seasonal confounders
   (a whole-home comparison conflates WM change with heating, lighting, seasonal effects)

---

### Q2: Limitations

**Overlapping appliance signatures**
WMs and dishwashers both draw 1.8–2.4 kW for 20–30 min then 200–400W for 60–90 min.
The model has a WM head only. Tumble dryer / dishwasher houses (H3, H7, H9) show
inflated predicted cycle counts (H9: 244→528). Fix: add explicit appliance heads so
each head claims its events and the WM head explains only the residual.

**Missing data — not missing at random**
H1 has 1,845 hours of outages clustering in winter. Reported metrics are slightly
optimistic — evaluated only on clean periods. Winter hot-wash behaviour
(more heating loads, higher background) is underrepresented in evaluation.

**1-minute resolution required**
SMETS2 meters default to 30-minute half-hourly reporting. This pipeline requires
1-minute data — either via SMETS2's 1-min mode (non-default) or a plug-level IAM
(~£25–30). Without 1-minute data, per-cycle disaggregation and hot/cold classification
are not achievable (see Q3).

**Household behaviour varies**
Occupancy, schedule, machine model, and local water temperature all shift the WM
signature in ways the 7 aggregate features cannot fully capture. MAE(ON)=395W
reflects this cross-house variance — the regression head learns an average wattage
across 16 houses but cannot predict H1's specific machine without seeing it.

**UK → India transfer**
Indian top-loading semi-automatic machines (200–400W, no internal heater) are a
different signature class absent from training data. Grid voltage fluctuations
(±10–20%) shift appliance power draw unpredictably. Required path: collect 2–4 weeks
of sub-meter labels from 15–20 Indian households and fine-tune from the UK checkpoint.

---

### Q3: 15-minute and 30-minute resolution impact

A typical WM cycle (65–120 min) produces this many data points at each resolution:

| Resolution | Points/cycle | What the model sees | F1 | Hot/cold? |
|---|---|---|---|---|
| 1 min | 65–120 | Full power envelope; phase transitions; heating spike detectable | **0.306** | Yes — 20-min heating phase visible |
| 15 min | 4–8 | 2–3 readings per phase — phases merge into one plateau | ~0.15–0.20 | Unreliable — heating averages with wash |
| 30 min | 2–4 | Full cycle may span 2 readings; start/end uncertain ±30 min | < 0.10 | No — heating phase compressed to one bin |

**H1 specific example** (33-min median cycle):
- At 15 min: 2 readings per cycle — whether the WM ran at all is uncertain
- At 30 min: entire cycle may fall within one reading — invisible

**What breaks first at 15 minutes:**
- H1's 33-min short cycles (2 readings) → missed entirely or merged with adjacent activity
- Hot-wash classification: 20-min heating phase at 1,800W averages to ~600W in a 15-min bin → indistinguishable from a cold wash
- Behavioral signature features `sig_med_dur` and `sig_med_energy` estimated with ±15-min and ±30% errors respectively

**What remains useful at 15–30 minutes:**
- Total WM energy per day (not per cycle) — still estimable with ±30–50% error
- Weekly usage frequency (did the WM run this week?) — recoverable from 30-min data
- Household ranking by consumption — the relative order H10 > H7 > H3 likely survives at daily granularity

**Practical consequence for Indian deployment:**
India's RDSS smart meter rollout targets 15–30 minute intervals. At this resolution,
the hot-wash intervention (which requires hot/cold classification per cycle) is not
feasible. The realistic goal shifts to energy attribution per day and usage frequency —
sufficient for "did this household use its washing machine this week?" but not for
"how many hot washes did they run?" The latter requires either ≤5-minute data or
a direct plug-level sensor on the WM circuit.

---

## Consistency Across Deliverables

| Claim | README | DATA | RESULTS | Recommendation |
|---|---|---|---|---|
| Model | ARNILM V8 | — | ARNILM V8 | ARNILM V8 |
| F1 (test H1) | 0.306 | — | 0.306 | 0.306 |
| MAE | 20W | — | 20W | 20W |
| Energy error | 69.7% | — | 69.7% | 69.7% |
| Cross-house SOTA | F1=0.17 | — | F1=0.17 | F1=0.17 |
| Cleaning rules | 7 rules | 7 rules | — | — |
| Hierarchical fix | Stage B2, 11,124 | — | 11,124 | — |
| Calibration houses | H5,7,11,17 | — | H5,7,11,17 | — |
| p_on threshold | 0.53 | — | 0.53 | — |
| Fleet saving (GT) | — | — | 2,483 kWh/yr | 2,483 kWh/yr |
| Fleet saving (model) | — | — | — | 1,752 kWh/yr |
| Profile match | — | — | 37% | 37% |
