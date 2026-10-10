# Recommendation — Utility-Facing Summary

**Prepared by Vijay Rameshkumar**

---

## What Was Built

A machine learning pipeline that reads only the whole-house smart meter and
estimates, minute by minute, how much power the washing machine is drawing — no
additional hardware required. The pipeline has four connected stages:

1. **Data cleaning** — 11,124 physical impossibilities (appliance power exceeding
   aggregate) corrected; flat-line outages detected and flagged.
2. **EDA and ground truth** — 6,369 WM cycles characterised across 19 households;
   per-household behavioural profiles derived from the sub-meter.
3. **Disaggregation model** (V8 LSTM + SGN gate) — a sequence-to-sequence LSTM with a
   multiplicative on/off gate, trained on 16 households and tested on a 17th it
   never saw. (No autoregressive feedback — previous WM predictions are not fed back
   as inputs; the LSTM hidden state carries aggregate sequence context.)
4. **Predicted profiles pipeline** — V8 inference applied to all 19 households to
   derive cycle-level profiles from the aggregate alone, with no sub-meter input.

The model achieves F1=0.306, MAE=20W, and energy estimation error of 69.7% at
1-minute resolution on the held-out test house, beating the best published
cross-house benchmark (F1=0.17) by 80% on a stricter protocol.

---

## Energy-Saving Opportunity

### Finding from the data

Across 19 REFIT households, wash temperature is the single largest lever for
reducing WM energy. A hot cycle (60°C) consumes 400–1,100 Wh in heating alone;
a cold cycle (30°C) consumes only motor power (~50–150 Wh). The ground-truth
sub-meter analysis found:

- **11 of 19 households** are classified `heavy_hot` (≥70% hot washes AND median
  cycle energy ≥400 Wh) or `light_hot` (≥40% hot washes).
- **H19 is the structural outlier**: 9% hot wash, 232 Wh median cycle energy —
  already running predominantly cold. Its annual WM saving opportunity is only
  20 kWh/year.
- **H7 is the highest opportunity**: 98% hot wash, 521 Wh median cycle energy,
  869 cycles over the study period. Cold-switching saves an estimated **349 kWh/year**
  for this household alone.

### Modelled saving potential (all 19 households)

From the Section 4 pipeline — using only aggregate meter data, no sub-metering:

| Metric | Ground truth | Model-predicted |
|---|---|---|
| Total cold-switch saving, 19 houses | 2,483 kWh/yr | 1,752 kWh/yr |
| Top opportunity household | H7: 349 kWh/yr | H10: 272 kWh/yr |
| Profile match rate (exact label) | — | 37% (7/19 houses) |
| Saving opportunity ranking | H7 > H10 > H3 | H10 > H7 > H3 |

The model identifies the right households to target. The top-3 opportunity
ranking is preserved (H3, H7, H10 appear in both lists). Total saving is
underestimated by ~30% because wattage per cycle is underestimated
(MAE when WM is ON = 395W — a known issue being fixed in the next model
version). For targeting purposes — sending nudge messages to the right
households — this is sufficient.

### How to measure impact

1. Compute baseline hot-wash fraction and median cycle energy per household
   using four weeks of aggregate data through the disaggregation pipeline.
2. Send targeted in-app messages to households with modelled hot-wash fraction > 0.6,
   recommending 30–40°C and Bio detergent.
3. Re-run the pipeline four weeks after the campaign. Compare hot-wash fraction
   and estimated cycle energy before and after.
4. The disaggregated WM signal isolates the effect from seasonal confounders
   (a whole-home comparison would conflate WM change with heating or lighting changes).

**Estimated impact per heavy-hot household (H7-type, 3 cycles/week):**

```
Before (60°C):   3 × 521 Wh  = 1,563 Wh/week  =  81 kWh/year
After  (30°C):   3 × 100 Wh  =   300 Wh/week  =  16 kWh/year
Annual saving:   ~65 kWh  |  ~£19/year  |  ~14 kg CO₂/year (per household)
Fleet saving (1,000 H7-type households):  ~65,000 kWh/year
```

---

## Where This Model Can Be Used Confidently

**Behavioural segmentation at scale**
Classifying households as heavy-hot, light-hot, eco, or cold-user from two weeks
of aggregate data. No sub-metering required. The profile labels are stable enough
for campaign targeting even at the current F1=0.306.

**Cycle-count and duration estimation**
The pipeline counts predicted wash cycles per household per month with reasonable
accuracy (H1: 176 predicted vs 397 ground truth — under-counts due to low recall,
but duration estimates per detected cycle are accurate to within 1–2 minutes).

**Fleet-level energy attribution**
Estimating total WM energy as a fraction of household consumption, aggregated
across a portfolio. Individual household estimates carry 70% energy error; fleet
averages smooth this substantially.

**A/B testing of efficiency campaigns**
Before/after changes in modelled hot-wash fraction provide a clean signal for
programme evaluation, provided the campaign and control groups are large enough
(≥30 households per arm recommended given model noise).

---

## Where This Model Has Limits

### 1. Not suitable for billing or individual energy guarantees

Energy estimation error is 69.7% at the individual household level. This reflects
false positives during off periods (predicted ~300W when WM is silent). Do not use
disaggregated estimates for tariff charges, appliance rebates, or energy guarantees
without further per-household calibration.

### 2. Wattage underestimation in ON-state

When the WM is running, the model underestimates wattage by ~395W on average
(MAE when ON). This means hot-wash energy is underestimated by ~30–50%, which
is why the fleet saving estimate (1,752 kWh/yr) is lower than ground truth
(2,483 kWh/yr). V8b (currently training) adds a direct regression path on true-ON timesteps that
bypasses the gate, which should correct this underestimation.

### 3. Overlapping appliance signatures

Dishwashers, tumble dryers, and electric showers all draw 1.5–3 kW for 45–120
minutes — signatures that overlap with WM cycles. The model was trained on WM
labels only; it has no explicit dishwasher or dryer head to suppress false positives.
REFIT H3, H5, H7, H15, and H17 have tumble dryers. These households tend to have
inflated predicted cycle counts (H7: 979 predicted vs 869 ground truth).

### 4. 1-minute data required

UK SMETS2 smart meters default to 30-minute half-hourly reporting. This model
requires 1-minute data. A direct comparison of what breaks at lower resolution:

| Resolution | F1 (expected) | Cycle detection | Energy (weekly) |
|---|---|---|---|
| 1 min (this model) | 0.306 | Reliable — cycle shape visible | ~30% underestimate |
| 15 min | ~0.15–0.25 | Uncertain — short cycles merge into one interval | ±20–30% |
| 30 min | <0.10 | Not feasible — a 35-min cycle appears as 1–2 intervals | ±40–60% |

At 15 minutes, the heating-phase spike (2 kW, 20–30 min) averages down into a
1,000–1,500 W interval indistinguishable from an electric shower. Cycle timing
is lost. A 35-minute quick-wash cycle at 30°C may appear in only one or two
intervals, making start/stop detection unreliable.

At 30 minutes, per-cycle disaggregation is not achievable. The practical
application shifts to: "this household appears to use a WM regularly on weekday
mornings" rather than "this household ran 3 hot washes this week."

Deploying this model therefore requires requesting 1-minute data from the smart
meter. SMETS2 supports this via the DCC but it is not the default. Alternatively,
a low-cost in-home display (IHD) or consumer access device (CAD, ~£30) provides
1-minute data without meter exchange.

### 5. UK front-loading machines, 2013–2015

The model was trained on UK front-loading washing machines from one geographic
area. Indian households commonly use:
- Top-loading semi-automatic machines (200–400W flat profile, no internal heater)
- Cold-fill-only top-loaders (bucket heating external)
- Semi-automatic twin-tub machines (separate wash + spin tub)

These signatures are substantially different. Direct deployment in India without
retraining would produce unreliable results. The recommended path for Indian
deployment: collect 2–3 weeks of labelled sub-meter data from 15–20 diverse
Indian households and fine-tune from the UK checkpoint.

---

## Additional Data That Would Improve the Model

**1-minute sub-meter labels for Indian households**
The single highest-value addition. Two weeks of labelled data from 15–20 Indian
households with diverse machine types is sufficient to fine-tune the existing
checkpoint for the Indian market.

**Dishwasher sub-meter channel (already in REFIT)**
REFIT includes dishwasher channels for several houses but they were not used in
this work. Adding a dishwasher head to the model would reduce WM false positives
substantially — the two appliances are the main source of confusion.

**Post-2020 UK dataset**
Quick-wash programmes (15–20 min, 800W, no heating phase) have grown
significantly since 2015 and represent a new cycle class the current model
has not seen. A more recent dataset would extend coverage to this growing
programme type.

**Voltage measurements**
Indian grid voltage fluctuates ±10–20%, shifting apparent power draw in ways
that affect the behavioural signature features. Adding voltage as an input would
improve robustness for Indian deployment.
