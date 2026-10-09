# RESULTS.md — EDA, Model Results, Validation, and Interpretation

---

## Washing Machine EDA — Key Findings

### Cycle detection across 19 households

Cycles were detected from the WM sub-meter channel using a hysteresis state machine:
- Power ≥ 25W for ≥ 15 consecutive minutes = cycle start
- Power < 25W for > 5 consecutive minutes = cycle end (hysteresis prevents drain-phase fragmentation)
- Cycles shorter than 15 min or longer than 180 min discarded (grounded in UK product specs)

| House | Cycles | Median duration | Median energy | Hot wash % |
|---|---|---|---|---|
| H1 | 415 | 33 min | 304 Wh | 91% |
| H2 | 342 | 91 min | 603 Wh | 94% |
| H4 | 62 | 50 min | 884 Wh | 94% |
| H5 | 730 | 40 min | 275 Wh | 74% |
| H6 | 136 | 99 min | 643 Wh | 84% |
| H7 | 892 | 58 min | 526 Wh | 99% |
| H8 | 366 | 60 min | 808 Wh | 98% |
| H9 | 252 | 71 min | 629 Wh | 97% |
| H10 | 616 | 113 min | 694 Wh | 92% |
| H11 | 78 | 94 min | 511 Wh | 90% |
| H13 | 459 | 66 min | 432 Wh | 90% |
| H15 | 265 | 86 min | 654 Wh | 77% |
| H16 | 323 | 72 min | 616 Wh | 99% |
| H17 | 311 | 52 min | 368 Wh | 78% |
| H18 | 139 | 47 min | 316 Wh | 93% |
| H19 | 257 | 101 min | 242 Wh | 9% |
| H20 | 205 | 70 min | 362 Wh | 93% |
| H21 | 251 | 82 min | 609 Wh | 92% |

**Key observations:**
- H19 is the structural outlier: 9% hot wash vs 90%+ average. Median cycle energy
  242 Wh vs 629 Wh for a typical high-hot-wash household. The difference is the
  heating element — cold cycles draw only motor power (~300W), hot cycles add the
  heating element (~2000W for 20–30 min).
- H4 has only 62 cycles (vs 892 for H7) — the household appears to use a laundry
  service or have low occupancy.
- H10's 113-minute median duration suggests a large-capacity machine or regular
  heavy-load washes.

### Time-of-use patterns

Peak washing activity across REFIT households: 09:00–11:00 and 18:00–20:00 on
weekdays, with a shift toward 10:00–14:00 on weekends. The morning peak aligns
with household routine (before leaving for work); the evening peak aligns with
return from work.

H1 has the earliest peak (08:00–09:00), consistent with a short-cycle household
(33-min cycles — quick wash before leaving). H6 and H10 peak later (11:00–13:00),
consistent with longer cycles run midday.

### Plots produced

| Figure | Description |
|---|---|
| `figures/S2_0_coverage_quality.png` | Data coverage and gap distribution per house |
| `figures/S2_1_daily_trace.png` | Representative WM daily power trace |
| `figures/S2_2_cycle_profiles.png` | Cycle duration and energy distributions |
| `figures/S2_3_usage_heatmap.png` | Hour × day-of-week usage heatmap |
| `figures/S2_4_cross_house_comparison.png` | Cross-household comparison of key metrics |
| `figures/S2_5_heating_analysis.png` | Hot vs cold wash energy distributions |

---

## Model Results

### Validation approach

**Split: Leave-House-1-Out (LOHO)**
- Train: Houses 2–19 (Part 2 data only, post April 2014)
- Validation: Houses 20 and 21 (held out during training for early stopping and threshold calibration)
- Test: House 1 entirely — zero data leakage

House 1 characteristics in test set: 670,257 timesteps, WM ON 1.8% of the time.

**Why LOHO**: within-house time splits give optimistic results because the model
has seen the household's specific appliance signatures, noise floor, and background
load during training. LOHO simulates real deployment where the model encounters an
entirely new household.

**Training balance**: ON/OFF timesteps were balanced 50/50 during training (10,000
each per house) to prevent the model learning to always predict zero (true prevalence
is 1.8% ON). This creates a calibration mismatch addressed by threshold calibration
on the validation set.

### Model comparison — House 1 test set

| Model | MAE | RMSE | mae_on | F1 | Precision | Recall | Energy Err | Constraint Viol |
|---|---|---|---|---|---|---|---|---|
| M0 Zero baseline | 10W | 132W | 565W | 0.00 | 0.00 | 0.00 | 100% | 0.0W |
| M1 Seq2Point | 71W | 189W | 249W | 0.07 | 0.03 | 0.94 | 638% | 0.0W |
| UnifiedNILM | 38W | 156W | 271W | 0.12 | 0.06 | 0.95 | 293% | 0.0W |
| **ARNILM (40 ep)** | **8W** | **94W** | 409W | **0.64** | **0.54** | **0.79** | **64%** | **0.0W** |

**Constraint_viol_W = 0.0 for all models**: the hierarchical constraint (WM ≤
aggregate) is never violated, enforced by soft penalty in training loss and hard
clip at inference.

### Reading the results

**M0 (zero baseline)**: MAE=10W because the WM is off 98.2% of the time. A model
that always predicts zero is trivially correct on most timesteps. Any useful model
must beat this. ARNILM achieves MAE=8W — better than zero.

**M1 (Seq2Point)**: high recall (0.94) but near-zero precision (0.03). The model
detects almost every genuine ON event but also fires constantly on OFF periods — it
has learned to predict "always on" as a strategy. This is the failure mode of a CNN
window model at 1-minute resolution: the raw window has insufficient discriminating
signal.

**UnifiedNILM**: adding 32 engineered features (rolling aggregate stats, event context,
house signature) improves F1 from 0.07 to 0.12 and reduces energy error from 638% to
293%. But precision remains poor (0.06) — the model still over-predicts.

**ARNILM**: LSTM autoregressive architecture processes the full sequence without
a fixed window constraint. The hidden state carries context from the entire observed
history. F1 jumps to 0.64, precision to 0.54, energy error collapses to 64%. The MAE
of 8W is now genuinely useful — predictions are more accurate than silence.

### Training dynamics (ARNILM)

| Epoch | Train NLL | Val NLL |
|---|---|---|
| 1 | 6.34 | 5.57 |
| 15 | 4.44 | 4.84 |
| 18 | 4.44 | 4.60 | ← first LR decay event |
| 21 | 3.89 | 4.09 | ← second LR decay event |
| 30 | 4.12 | 3.85 | ← third LR decay event |
| 40 | 3.64 | 3.44 | ← still improving |

Three ReduceLROnPlateau decay events each unlocked a new convergence level. Val NLL
was still declining at epoch 40 — the model has not converged and would continue
improving with additional training.

### Comparison against REFIT published benchmarks

| Model | F1 | MAE | Resolution | Split | Source |
|---|---|---|---|---|---|
| Seq2Point on REFIT | 0.27 | 28W | 1-min | within-house | NILMBENCH2026 |
| BERT4NILM (no denoise) | 0.33 | — | 1-min | within-house | Yue et al. 2020 |
| Seq2Point NILMBench2026 | 0.42 | — | 1-min | within-house | NILMBENCH2026 |
| BERT4NILM (denoised) | 0.64 | — | 1-min | within-house | Yue et al. 2020 |
| SGN | 0.76 | 14W | 1-min | within-house | NILMBENCH2026 |
| Seq2Point cross-dataset | 0.17 | — | 15-min | cross-house | Springer 2025 |
| **ARNILM (ours)** | **0.64** | **8W** | **1-min** | **cross-house LOHO** | this work |

### Plots produced

| Figure | Description |
|---|---|
| `figures/S3_0_training_curves.png` | M1 and UnifiedNILM training convergence |
| `figures/S3_1_predicted_vs_actual_day.png` | M1 vs UnifiedNILM vs ground truth, 1 day |
| `figures/S3_2_metric_comparison.png` | Bar chart: M0 vs M1 vs UnifiedNILM |
| `figures/S3_3_scatter.png` | Actual vs predicted scatter, M1 and UnifiedNILM |
| `figures/S3b_0_ar_lstm_training.png` | ARNILM 40-epoch convergence curve |
| `figures/S3b_1_ar_lstm_day.png` | ARNILM prediction with uncertainty ribbon |
| `figures/S3b_2_all_model_comparison.png` | All four models compared |

---

## Failure Analysis

### Why precision is limited

The root cause is **single-appliance modelling**. With only a WM head, the model
must classify every aggregate event as WM or not-WM by itself, without explicitly
accounting for what the other appliances are doing. Dishwashers run 90–120 min cycles
at 1.5–2.0 kW — indistinguishable from a WM cycle in the aggregate. The model has
learned this partially from training examples (dishwasher events where WM sub-meter=0),
but generalisation to an unseen household's dishwasher is imperfect.

The fix: add a dishwasher head to the model. Once the dishwasher head claims its
events, the WM head only fires on residual load. Precision would improve substantially.

### Why energy error remains at 64%

The model predicts cycle timing well (recall=0.79) but undershoots peak wattage
during the heating phase. The heating phase draws 1800–2400W for 20–30 minutes —
at 1-minute resolution, a single minute's reading captures 20–100W of the thermal
cycling oscillation rather than the sustained plateau. The model learns a smoothed
representation of the heating phase rather than its true peak. More training epochs
and a lower learning rate would improve this.

### Why mae_on (409W) is high despite good F1

mae_on measures the error specifically on timesteps where the WM is genuinely running.
The model detects cycle presence well (F1=0.64) but the wattage prediction within a
cycle is still imprecise. The heating phase is systematically underestimated (see
above), and the agitation/rinse phase (200–400W) is more accurately predicted. The
high mae_on reflects the heating-phase prediction gap, not a detection failure.

### Train/test distribution mismatch

Training uses 50/50 ON/OFF balance; House 1 test is 1.8% ON. The model is calibrated
for a world where WM is on half the time. Post-hoc threshold calibration (found
threshold=10W on validation set) partially corrects this, but the underlying
probability estimates remain miscalibrated. A class-weighted loss using the true
1.8% prior would address this fundamentally.
