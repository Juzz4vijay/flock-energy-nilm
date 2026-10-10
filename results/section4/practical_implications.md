# Section 4 — Practical Implications

---

## 1. Energy-Saving Opportunity: Hot Wash Reduction

### What the data shows

The house behavioral signature computed in Section 3 includes `sig_hot_frac` — the
fraction of wash cycles run at hot temperature (inferred from cycle energy: hot washes
draw more energy per cycle than cold/warm). Across the 19 REFIT households:

| Household | Hot wash fraction | Median cycle energy |
|---|---|---|
| H19 | 9% | 242 Wh |
| H5 | 74% | 275 Wh |
| H15 | 77% | 654 Wh |
| H1 | 91% | 304 Wh |
| H7, H8, H16 | 98–99% | 526–808 Wh |

H19, the outlier at 9% hot wash, consumes 242 Wh per cycle median — compared to
654–808 Wh for the high hot-wash households. The energy consumed is not just higher
per cycle: it draws at 1800–2400W for 20–30 minutes during the heating phase, placing
a large instantaneous load on the grid.

A UK front-loader at 60°C consumes roughly 1.0–1.5 kWh per cycle. The same machine
at 30°C consumes 0.2–0.3 kWh. The difference is almost entirely the heating element.

### The practical action

**Target households running > 80% hot washes and shift them to 30–40°C for lightly
soiled loads.**

Modern detergents (Bio, Persil, Ariel) are formulated to work effectively at 30°C.
The only loads that genuinely require 60°C+ are heavily soiled items and bedding for
allergy sufferers. For typical weekly household laundry — work clothes, casual wear,
bedding maintenance — 30–40°C achieves equivalent cleaning.

For a household running two 60°C cycles per week (H8 profile: 808 Wh median, 98% hot):

```
Current:  2 cycles × 1.1 kWh  =  2.2 kWh/week  =  114 kWh/year
30°C:     2 cycles × 0.25 kWh =  0.5 kWh/week  =   26 kWh/year

Annual saving per household:  ~88 kWh  (~£26 at UK 2024 rates)
CO₂ reduction (UK grid, 233g/kWh):  ~20 kg CO₂/year
```

Scaled across the 12 high hot-wash households in REFIT (>80% hot):

```
Aggregate annual saving:  ~1,056 kWh  ~£312  ~240 kg CO₂
```

Across the 27 million UK households with a washing machine (assuming similar
hot-wash prevalence), the national saving would be approximately 2.4 TWh/year.

### How impact would be measured

The disaggregation model provides the measurement instrument directly:

1. **Baseline**: run model on aggregate data for 4 weeks before intervention.
   Record `cycles_per_week`, `mean_energy_wh`, `hot_frac` per household.

2. **Intervention**: send targeted in-app nudge to high-hot-wash households
   (those with `sig_hot_frac > 0.8`). Message: "Your washing machine uses 4×
   more energy on hot cycles. Switching to 30°C saves you ~£26/year."

3. **Post-intervention**: repeat measurement 4 weeks later. Compare
   `mean_energy_wh` and `hot_frac` before and after.

4. **Attribution**: the disaggregated WM signal isolates the change.
   Without NILM, only the total household consumption is visible — confounded
   by seasonal effects, occupancy changes, and other appliances. NILM makes
   the measurement precise and appliance-specific.

Weeks where the model detects fewer than 3 complete cycles are excluded from the
before/after comparison — too few cycles to give a reliable energy estimate.

---

## 2. Limitations

### Overlapping appliance signatures

Washing machines and dishwashers share almost identical power profiles: both have
a heating phase (1.8–2.4 kW, 20–30 min) followed by agitation/rinse (200–400W,
40–60 min). The aggregate signal cannot distinguish them without sub-metering.

In our model, the event context features (ev_dur, ev_peak) partially discriminate
them — a dishwasher cycle is typically shorter (total 90–120 min vs WM 65–180 min)
and runs more frequently through the day. But households that own both and run them
concurrently create a superimposed signal the model has no way to decompose with a
single appliance head.

The practical consequence: precision is limited in households with dishwashers.
The model learns from training houses that some long-duration aggregate events are
dishwashers (WM sub-meter = 0 despite an aggregate pattern resembling WM), but
generalisation to unseen households is imperfect. Adding a dishwasher head would
resolve this directly.

### Missing data and outages

The REFIT dataset has 12 outage gaps in House 1 alone, totalling 1,845 hours — 77
full days of missing data. These are not random gaps: outages tend to cluster in
winter months (grid instability, meter reboots) and are more likely during periods
of high consumption. Missing data is therefore not missing-at-random; it is
correlated with the signal we are trying to model.

Our handling (SARIMA imputation for gaps < 24h, flag-only for longer gaps) is
conservative. The imputed values are plausible but not ground truth — any WM cycles
that fall entirely within a gap are invisible to both training and evaluation. The
reported metrics are therefore slightly optimistic: we only evaluate on periods where
we have clean data.

### 1-minute resolution

Every published NILM benchmark uses 6–8 second resolution. At that resolution, a
single wash cycle produces 600–800 data points — enough to see the thermal cycling
oscillations in the heating phase, the sharp power drop at agitation start, and each
individual rinse cycle pulse. These are highly discriminating shape features.

At 1 minute, the same cycle produces 65–180 data points. The thermal cycling averages
out. The sharp transitions become single-point inflections. The multi-step rinse
sequence collapses into a flat plateau. This is a fundamental information loss — not
a data quality problem that better cleaning can fix.

The practical implication for smart meter deployments: UK SMETS2 smart meters report
at 30-minute intervals by default (half-hourly settlement). The 1-minute data in
REFIT comes from individual appliance monitoring devices (IAMs) deployed for the
research study, not from the household meter itself. Real smart meter deployments
operate at 15–30× coarser resolution than REFIT, making WM disaggregation
substantially harder than our results suggest.

### Household behaviour

REFIT was collected from 2013 to 2015 from 19 UK households recruited through a
university. Several behavioural biases are likely:

- **Selection bias**: participants willing to install monitoring hardware and share
  energy data are more likely to be energy-conscious. Their washing patterns may
  differ from the general population (potentially already lower hot-wash fraction
  than average UK households).
- **Temporal drift**: wash patterns recorded in 2013–2015 reflect appliance
  models, detergent formulations, and household compositions from that period.
  The growth of quick-wash programmes (15–30 min, 800–1200W, no heating phase)
  since 2015 has changed the distribution of cycle signatures the model would
  need to handle.
- **Occupancy effects**: households with young children, elderly residents, or
  home workers have systematically different wash frequency and timing. The model
  learns from the 18 training households but cannot guarantee coverage of all
  occupancy types.
- **Seasonal variation**: the REFIT collection window (Oct 2013 – Jun 2015) is
  biased toward autumn, winter, and spring. Summer washing behaviour — lighter
  loads, shorter cycles, more cold washes — is underrepresented.

### Applying UK results to India

Several structural differences make direct transfer of a UK-trained model to Indian
households unreliable:

**Appliance hardware**: the UK market in 2013–2015 was dominated by front-loading
European machines (Beko, Hotpoint, Bosch, Indesit) with standardised power profiles.
India's market is a mixture of top-loading semi-automatic machines (common in
lower-income households, manual water filling, no heating element — 200–400W flat
profile) and fully automatic front/top-loaders. The power signatures are
fundamentally different. Our model has never seen a top-loading semi-automatic cycle
and would fail to disaggregate it.

**Voltage and grid stability**: India operates at 230V 50Hz (same nominal as UK)
but with significant voltage fluctuations (±10% is common, ±20% during peak hours
in rural areas). Appliance power draw scales with V², so a 10% voltage drop reduces
heating element power by ~20%. The cycle energy and peak-power features that anchor
our behavioral signature would shift unpredictably across the day.

**Data availability**: REFIT's 1-minute resolution comes from dedicated IAM hardware.
India's smart meter rollout (Advanced Metering Infrastructure under RDSS scheme) is
ongoing but targets 15–30 minute intervals. Disaggregation at this resolution is
near-impossible for a washing machine (see Section 2.3 below).

**Usage patterns**: Indian washing behaviour differs systematically — higher frequency
of cold-water-only cycles (particularly in warm climates where cold water is
sufficient), more frequent small loads (water conservation), and different time-of-use
distribution (early morning washing before household members leave for work). The
temporal features (hour_sin/cos, dow_sin/cos) and behavioral signature (hot_frac,
cycle timing) would all shift significantly.

**Transfer path**: the correct approach for India is not to deploy a UK-trained model
directly, but to:
1. Collect 2–4 weeks of aggregate data from Indian households
2. Run cycle detection to identify WM-like events in the aggregate
3. Compute the 7 behavioral signature features from those cycles
4. Use the features to find the closest matching training household in the
   feature space and fine-tune from that starting point
5. Re-evaluate on locally labelled cycles

This is feasible using the continuous behavioral signature design — the model does
not need a new cluster assignment, just a new 7-vector input that reflects Indian
appliance behaviour. But without Indian training data, generalisation will be poor.

---

## 3. Resolution Impact: 15-minute and 30-minute Data

### What resolution means for a WM cycle

A washing machine cycle at typical UK household settings lasts 65–120 minutes.

```
Resolution    Points per cycle    What is visible
──────────────────────────────────────────────────────────────────
8 sec         ~540 – 900         Full shape: thermal cycling, phase transitions,
                                  rinse pulses, spin profile — textbook signal
1 min         65 – 120           Overall envelope visible, phase transitions
                                  detectable, thermal cycling averaged out
15 min        4 – 8              2–3 readings in heating phase, 2–3 in
                                  agitation/rinse — phases indistinguishable
30 min        2 – 4              Entire cycle may fall within 2 readings.
                                  Only total energy is partially recoverable
```

### Quantitative impact

The literature provides a direct data point: Seq2Point's F1 for washing machine on
REFIT drops from 0.27 (within-house, 1-min) to 0.17 (cross-dataset, 15-min) — a 37%
relative decline from an already modest baseline. For a stronger model:

| Model | F1 at 1-min | Expected F1 at 15-min | Expected F1 at 30-min |
|---|---|---|---|
| Seq2Point | 0.27 | 0.17 (observed, cross-dataset) | < 0.10 |
| **ARNILM V8 (ours)** | **0.306** | ~0.15–0.22 (estimated) | < 0.10 |
| SGN | 0.76 | ~0.35–0.45 (estimated) | ~0.15 |

The ARNILM V8 estimate is based on the same proportional degradation observed for
Seq2Point (37% relative drop from 1-min to 15-min). The LSTM's longer context gives
some robustness but the fundamental problem remains — cycle shape information is lost. But the fundamental problem remains: at 15 minutes, a cycle heating phase
and agitation phase each produce at most 2 data points. The discriminating shape
information is gone.

### What changes technically at 15–30 minutes

**Event detection fails**: the threshold-based event detector (80W, minimum 15 min)
relies on consecutive 1-minute readings above threshold. At 15 minutes, a 3-minute
kettle event either falls entirely within one bin (averaged down) or spans two bins
(partially captured). Event duration and energy features lose their discriminating
power.

**Behavioral signature degrades**: cycle detection itself becomes unreliable. Short
cycles (H1: 33-min median) at 15-minute resolution produce only 2 readings per cycle.
Start and end boundaries are uncertain to ±15 minutes. The `sig_med_dur` and
`sig_med_energy` features that anchor house-level generalisation are estimated with
large errors.

**Energy error becomes the primary metric**: at 30-minute resolution, cycle-level
detection is essentially impossible. The only useful output is total WM energy per
30-minute window — a much coarser quantity than per-minute wattage. Energy estimation
error would dominate all other metrics.

### Practical consequence for smart meter deployments

UK SMETS2 meters default to 30-minute half-hourly data. The REFIT data was collected
using plug-level IAMs specifically because the household meter is too coarse. Any
real-world NILM deployment at smart meter resolution (15–30 min) needs to accept that
washing machine disaggregation at per-cycle granularity is not achievable. The
realistic goal shifts to:

- **Energy attribution** (per day, not per cycle): was the washing machine responsible
  for X% of today's consumption? Achievable at 15-minute resolution.
- **Usage frequency** (cycles per week, not cycle start time): recoverable from 30-min
  data by detecting sustained elevated loads during likely wash windows.
- **Hot vs cold classification**: not recoverable at 30-minute resolution — the heating
  phase is too short relative to the bin size.

For per-cycle, per-phase, behavioural insight — the kind that enables the hot-wash
intervention described in Section 1 — 1-minute or finer resolution is a hard
requirement. Smart meter deployments should pair SMETS2 half-hourly data with a small
number of plug-level IAMs on high-impact appliances (WM, dryer, EV charger) to get
the resolution needed for actionable disaggregation.
