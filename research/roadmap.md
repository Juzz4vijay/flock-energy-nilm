# Strategic Roadmap: REFIT Cleaning and Washing-Machine Disaggregation

> Working plan for the Flock Energy take-home. It sets out what to build, which approaches to compare, the order to work in, and the time to give each part.

---

## 1. The strategy in one paragraph

Build a small, reproducible pipeline with **three models of increasing complexity** (a rule-based baseline, a gradient-boosted model, and a Seq2Point CNN). Evaluate all three on **houses the models never saw in training**. Add one experiment that most candidates won't run: **degrade the input to 15- and 30-minute resolution and measure what breaks**. That experiment answers the brief's resolution question with evidence instead of opinion, and it is directly relevant to Indian smart meters. Spend the remaining effort on clear write-ups and honest failure analysis, because the brief says reasoning matters more than coverage.

---

## 2. What the evaluators are scoring

| Signal | How to show it |
|---|---|
| Careful data handling | Explicit, stated rules for duplicates, gaps and outliers, with counts before and after |
| Sound validation | House-level split, no leakage, and an explanation of *why* |
| Fit-for-purpose metrics | Metrics a utility cares about (energy per day or cycle), not just MAE |
| Judgement | A simple baseline next to the deep model, so its gain is proven |
| Honesty | A failure-analysis section showing where the model breaks |
| Reproducibility | One command runs everything, from the extracted data to the figures |
| Business sense | A one-page recommendation a utility manager could act on |

---

## 3. Know the data first (facts that shape the plan)

From the official cleaned-data readme:

- **20 houses**, labelled 1–21 with **House 14 skipped**, from the Loughborough area, about 2013–2015.
- Each house has **1 aggregate clamp + 9 appliance plugs (IAMs)**. **Active power only, in watts, every ~8 seconds.**
- Format: `DATETIME, UNIX TIMESTAMP (UTC), Aggregate, Appliance1 … Appliance9, Issues`.
- **The cleaned data already has corrections applied**, so know what they hide:
  - Daylight-saving time corrected, and appliance columns re-aligned when plugs were moved.
  - **Missing values are forward-filled.** A long outage therefore appears as a **flat line**, not as NaN. Detect long constant runs and treat them as missing. This is a strong point to raise in DATA.md.
  - IAM spikes above 4000 W were replaced with zeros.
  - `Issues = 1` where the appliance readings sum to more than the aggregate. **Drop these rows from training and evaluation.**
- Data was only logged when the load changed, and the sensors were **not synchronised** (the polling took 6–8 s). This is fine at 1-minute resolution, but worth stating as a limitation.
- Notable outage period: **February 2014**.

### Washing-machine column by house (check against the readme)

| House | WM column | Watch out for |
|---|---|---|
| 1 | Appliance5 | Also has a **washer-dryer** (Appliance4) |
| 2 | Appliance2 | |
| 3 | Appliance6 | Tumble dryer present |
| 4 | Appliance4 **and** Appliance5 | **Two** washing machines |
| 5 | Appliance3 | Tumble dryer present |
| 6 | Appliance2 | |
| 7 | Appliance5 | Tumble dryer present |
| 8 | Appliance4 | Also has a **washer-dryer** (Appliance3); WM has no make or model listed |
| 9 | Appliance3 | Also has a **washer-dryer** (Appliance2) |
| 10 | Appliance5 | |
| 11 | Appliance3 | |
| 12 | none | No washing machine listed. Exclude |
| 13 | Appliance3 | **Appliance changed 25 Mar 2015** |
| 15 | Appliance3 | Tumble dryer present |
| 16 | Appliance5 | |
| 17 | Appliance4 | Tumble dryer present |
| 18 | Appliance5 | Also has a **washer-dryer** (Appliance4) |
| 19 | Appliance2 | |
| 20 | Appliance4 | Tumble dryer present |
| 21 | Appliance3 | |

**Action:** don't trust this table blindly. In the EDA, plot a few cycles per house and confirm each one looks like a washing machine: a ~2 kW heating block near the start, a long low-power wash, and spin bursts at the end. Flag any house where it doesn't. This "label audit" is cheap and shows good judgement.

**Houses with a washer-dryer or tumble dryer are the hard cases**, because those appliances have overlapping signatures. They make a natural failure-analysis section.

---

## 4. Approach options (from the literature)

### 4.1 Disaggregation models

| # | Approach | Idea | Pros | Cons | Verdict |
|---|---|---|---|---|---|
| A | **Rule-based baseline** | Find ~1.8–2.5 kW rises in the aggregate that last about 10–30 min, then low-power activity for 60+ min | Interpretable, no training, fast | Confused by kettles, showers, heaters and dryers | **Build it.** It's the floor every model must beat |
| B | **Mean / always-off baseline** | Predict 0 W, or the training mean | Trivial | Shows how "easy" MAE is when the machine is mostly off | **Build it.** It takes 5 minutes and exposes misleading metrics |
| C | **LightGBM on window features** | Rolling mean, std, max, diff and quantiles of the aggregate over several windows, plus hour and weekday | Fast on CPU, strong, explainable with feature importance | Hand-made features, limited sense of the cycle's shape | **Build it.** Strong, cheap middle ground |
| D | **Seq2Point CNN** (Zhang et al., 2018) | A window of aggregate power predicts the appliance's power at the **midpoint** | Standard REFIT benchmark, simple, well cited, published house splits exist | Needs a GPU for speed (CPU works if small), a black box | **Main model** |
| E | Seq2Seq / UNet / **TCN** | Predict the whole output window, with dilated convolutions for long context | Smoother output, long receptive field | More tuning | Optional, if time remains |
| F | **Transformers** (BERT4NILM, ELECTRIcity) | Bidirectional attention over the sequence | State of the art on some benchmarks | Heavy, slow to train, overkill in 6 hours | **Cite, don't build.** Mention as future work |
| G | FHMM / combinatorial (via NILMTK) | Classic probabilistic models | Historical baseline | Weak on multi-state appliances like washing machines | Skip; mention only |
| H | **Graph signal processing** (Strathclyde group) | Training-free, unsupervised disaggregation | Needs no labels, suits low-resolution data | Harder to implement in the time | Cite as the route for unlabelled Indian data |

**Why Seq2Point is the main model:**
1. It is the most widely reproduced deep NILM model on REFIT, so your numbers can be compared with published work.
2. A washing-machine cycle is a long, multi-phase shape (heat, wash, rinse, spin). A CNN over a 2–4 hour window sees the whole cycle context, which point-wise features miss.
3. It is small enough to train on a laptop at 1-minute resolution.

**Adapting it to 1-minute data:** the original model uses 599 samples at 8 s (about 80 min). At 1 min, a window of **~199–255 samples (3.3–4.3 h)** covers a full cycle plus context. Treat the window length as a stated assumption and test 2 values if time allows.

### 4.2 Ideas to adopt (each needs a citation)

- **Standard house split for the washing machine** from the Seq2Point transfer-learning repository: train on houses 2, 5, 7, 9, 15, 16 and 17; validate on house 18; test on house 8. Using it makes your results comparable with the literature.
- **Normalisation:** standardise the aggregate using training statistics only; scale appliance power by a fixed cap (about 2500 W).
- **Activation thresholds** from Kelly & Knottenbelt (2015, Neural NILM) as a starting point: "on" above about 20 W, a minimum on-duration, and a minimum off-gap to merge pauses within one cycle. Tune these on the validation house only.
- **Multi-task idea** (on/off classification plus power regression), as in subtask-gated networks. Optional: multiply predicted power by predicted on-probability to suppress false positives during off periods.
- **Very-low-frequency NILM** (Strathclyde): appliances whose cycles last 30 minutes or more can still be partly disaggregated at low resolution. This motivates the resolution experiment.

---

## 5. Validation design

### 5.1 Splits

| Scenario | Train | Validate | Test | Purpose |
|---|---|---|---|---|
| **Unseen houses (main)** | 2, 5, 7, 9, 15, 16, 17 | 18 | 8 | Generalisation to new homes, which is what a utility needs |
| Second unseen test (recommended) | same | same | one clean house, such as 2 swapped out for 19 | Checks that one test house isn't luck; house 8 is a hard case with its washer-dryer |
| Seen house (optional) | First ~70% of time in the training houses | Next 10% | Last 20% | Shows the gap between seen and unseen houses |

**Optional, stronger version:** leave-one-house-out cross-validation with LightGBM, which is cheap enough to run per house. It reports a spread of results, not one number.

### 5.2 Avoiding leakage (state each point in the README)

1. **The split is by house**, never by random minutes. Neighbouring windows are almost identical, so random splits leak the answer into the test set.
2. **Normalisation statistics, thresholds and hyperparameters** come from training or validation houses only.
3. In time-based splits, **no window crosses a split boundary**; leave a gap of one window length.
4. **The test house is touched once**, at the end. No re-tuning after seeing test results.
5. `Issues = 1` rows and detected flat-line outages are excluded **before** windowing.

### 5.3 Metrics (and why)

A washing machine is **off for over 90% of minutes**. A model predicting 0 W everywhere gets a deceptively low MAE. So report a balanced set:

| Metric | What it answers | Why include it |
|---|---|---|
| MAE (W), overall **and when on** | Point accuracy | The standard metric; splitting it by on/off exposes the "predict zero" trap |
| SAE (signal aggregate error) | Is total energy over the period right? | Utilities care about energy, not instantaneous watts |
| Daily energy error (kWh/day, absolute and %) | Billing-style accuracy | Directly useful for consumer feedback |
| Precision, recall and **F1 for the on-state** | Does it know *when* the machine runs? | Needed for time-of-use and load-shifting advice |
| **Cycle-level detection** (cycles found or missed, false cycles) and per-cycle energy error | Does it count washes correctly? | Ties the model to the EDA and to the business use |

Compare **all models (B, A, C, D) in one table** on the same test house.

---

## 6. Phased roadmap with time budget

Target: about 6 focused hours. Each phase ends with something committed to the repo.

### Phase 0 — Setup and data loading (0.5 h)
- Repo skeleton (see §8), `requirements.txt`, `config.yaml` (paths, house list, thresholds, seeds).
- Download script and extraction with `py7zr`.
- **Efficient loader:** read only the needed columns (Unix, Aggregate, WM column(s), Issues) and resample to 1 minute. **Cache each house as Parquet.** This turns a multi-GB load into seconds on re-runs.
- **Exit check:** a Parquet file per house, with the row count and date range logged.

### Phase 1 — Raw data cleaning, House 1 (1 h)
- **Inspect first:** print the headers and first rows of the raw file (don't assume the cleaned-file layout).
- **Checks, each with a count in a summary table:**
  - Timestamp parsing (UNIX vs. local time, the daylight-saving shift) and whether rows are ordered.
  - Exact duplicate rows vs. duplicate timestamps with different values.
  - Irregular intervals: histogram of time differences (expect ~6–8 s).
  - Missing readings and gap-length distribution.
  - Impossible values: negatives, appliance > aggregate, appliance > 4 kW, aggregate > ~15 kW (a typical UK household supply limit).
- **Pipeline (stated rules):**
  1. Sort by timestamp.
  2. Duplicates: keep the **first** occurrence (or the mean, stating the choice).
  3. Set impossible values to NaN, with the rule and count logged.
  4. Resample to 1-minute **mean**.
  5. Gaps of **≤ 5 min**: linear interpolation (short relative to any appliance cycle).
  6. Gaps **> 5 min**: leave as NaN and set an `outage` flag column. Report outage count and total hours.
- **Plots:** aggregate plus 2 appliances (for example the washing machine and fridge), before and after, over the same busy week. Add a zoomed day.
- **Exit check:** `DATA.md` section drafted with the summary table.

### Phase 2 — Washing-machine EDA, cleaned data (1.25 h)
- **Coverage table per house:** start and end dates, % missing minutes (including flat-line runs), % zero or off minutes, cycles detected.
- **Cycle rule (state it):** on above **20 W**; merge on-segments separated by **< 10 min**; keep cycles lasting **≥ 20 min** and using ≥ 0.05 kWh. Tune by eye on a few houses and say so.
- **Per cycle:** start time, duration, peak W, energy kWh, and whether there is a heating phase (any minute above ~1.5 kW).
- **The four required plots:**
  1. A representative 24 h: aggregate and washing machine overlaid.
  2. 3–4 detected cycles, showing the heat, wash and spin phases.
  3. Heatmap of cycle starts by hour × weekday (pooled, plus a few houses).
  4. Box plots of energy per cycle and duration across houses.
- **Label audit:** flag houses whose "washing machine" doesn't look like one.
- **Business hook:** share of cycle energy spent in the heating phase (this feeds Part 4).
- **Exit check:** plots saved under `figures/`; `RESULTS.md` EDA section drafted with a 1–2 line interpretation per plot.

### Phase 3 — Models and evaluation (2 h)
1. **Data prep (20 min):** build windows from the cached Parquet; drop `Issues` and outage minutes; normalise with training statistics.
2. **Baselines B and A (20 min):** always-off and rule-based.
3. **LightGBM, C (25 min):** rolling features at 5, 15, 30, 60 and 120 minutes, plus hour and weekday. Early stopping on house 18.
4. **Seq2Point, D (45 min):** PyTorch, 5 convolutional layers and 1 dense layer, window ~199. Adam, MSE loss, early stopping on validation. Subsample "off" windows during training to balance classes (state this). Fixed seeds.
5. **Evaluate (10 min):** one metrics table for all models on test house 8; actual vs. predicted plots over 3 days with cycles.
- **Exit check:** `results/metrics.csv` and comparison figures.

### Phase 4 — Resolution experiment (0.5 h, high value)
- Downsample the test and training aggregate to **15 and 30 minutes** (mean power per interval). Retrain LightGBM, the cheap model, at each resolution.
- Report daily energy error and cycle-detection F1 at **1, 15 and 30 minutes**.
- **Expected story:** daily or weekly energy estimates degrade gracefully; minute-level timing and cycle counting collapse. Overlapping loads, like a shower or kettle in the same interval, become inseparable.
- This **directly answers** the brief's question with your own numbers, and it maps to Indian smart meters, which report at 15–30 minutes.

### Phase 5 — Failure analysis and write-ups (1 h)
- **Failure analysis:** show 3 concrete failure examples:
  - False positive from a tumble dryer, kettle or shower (high-power overlap).
  - Missed cycle from a low-temperature (no heating) wash.
  - Appliance change or label error (house 13 after March 2015 if used, or any house flagged in the audit).
- **Discussion:** missing labels, appliance changes, and unmetered loads (the aggregate includes many appliances without plugs; the model must learn that "other" load isn't the washing machine).
- Write `README.md`, `DATA.md`, `RESULTS.md` and `Recommendation.md` (templates in §8).
- **Final check:** fresh clone, `pip install -r requirements.txt`, then `python run_all.py`.

### Model roadmap

| Version | Description | Status | Key result |
|---|---|---|---|
| V1–V4b | Rule-based → Seq2Point CNN baselines | Done | F1=0.14–0.21 |
| V5–V6 | LSTM + SGN gate (basic) | Done | F1=0.27 |
| V7 | +normalized MSE + 4 shape features | Done | MAE=29.5W, F1=0.293, Energy_err=185.7% |
| V8 | V7 + pos_weight=3.5 (precision bias) | Done | MAE=20W, F1=0.306, Energy_err=69.7% — **beats cross-house SOTA** |
| V8b | V8 + decoupled regression (direct MSE on true-ON timesteps) | **Training** | Target: MAE(ON) ↓ from 395W |
| V9 | V8 + LSTM + Multi-head Self-Attention (8 heads) + residual LayerNorm | Killed at ep 20/80 (val_loss=0.206) | Add attention back in V10 |
| V10 | V8b + Multi-head Self-Attention (best of V8b + V9) | Planned | — |

#### V9 architecture (reference for V10)
- `PositionalEncoding` (sinusoidal, max_len=SEQ_LEN)
- `nn.MultiheadAttention(256, 8, dropout=0.1, batch_first=True)` over LSTM output
- Residual `x = x + attn_out` + `nn.LayerNorm(256)`
- AdamW + CosineAnnealingLR, LR=5e-4, BATCH_SIZE=128
- Killed at epoch 20 — val_loss trending down (0.206); GPU freed for V8b

#### V10 plan (future)
- Start from V8b once decoupled regression fix is validated (MAE(ON) target < 200W)
- Add self-attention block after LSTM: captures long-range cycle context (heating→wash→spin phases ~60–90 min apart)
- Retain SGN gate and normalized MSE from V8b
- Expected gain: improved F1 (better cycle boundary detection) with V8b's corrected wattage

### If you have extra time (in priority order)
1. Leave-one-house-out cross-validation with LightGBM, to show the spread across houses.
2. TCN or UNet sequence-to-sequence model as a second deep model.
3. On/off gating (multi-task) on Seq2Point.
4. Window-length sensitivity check (for example 127 vs. 255).

### If you run short (cut in this order)
1. Drop the second test house and the seen-house scenario.
2. Shrink Seq2Point training (fewer epochs or a subsample of houses). Say so in the README.
3. **Never cut:** baselines, the leakage explanation, failure analysis, or Recommendation.md.

---

## 7. Part 4 content plan (Practical implications)

### Energy-saving opportunity: wash temperature
- **Evidence from your EDA:** the share of each cycle's energy spent in the ~2 kW heating phase (typically most of a hot wash's energy).
- **Action:** utility messaging nudging households towards 30 °C or eco programmes, triggered when disaggregation detects frequent heated cycles.
- **How to measure the impact:** compare the change in heating-phase energy per cycle in nudged vs. control households (difference-in-differences), using disaggregated cycle energy as the outcome. Run for at least 8 weeks so weekly laundry habits average out.
- **Secondary:** load shifting to off-peak hours, measured as the change in the share of cycles starting during peak hours. Relevant to time-of-day tariffs.

### Limitations to discuss
- **Overlapping signatures:** washer-dryers, tumble dryers, dishwashers, kettles, showers and heaters all draw 1.5–3 kW.
- **Missing data:** forward-filled gaps in the cleaned data, the February 2014 outage, and plug sensors that may be unplugged.
- **1-minute resolution:** no transients, no reactive power. The machine's spin and motor details are averaged.
- **Household behaviour:** occupancy ranges from 1 to 6, so habits vary a lot; 20 homes in one UK town is a small, unrepresentative sample.
- **UK to India transfer:**
  - **Appliances:** many Indian homes use top-loading or semi-automatic machines, often **without internal heaters**. Their signature is smaller and harder to separate.
  - **Load mix:** air-conditioners, water heaters (geysers), water pumps and inverter/battery backup dominate; little of this exists in REFIT.
  - **Supply:** voltage fluctuation and outages alter power readings.
  - **Metering:** India's RDSS smart meters typically log at 15–30 minutes, so minute-level models won't apply directly.
  - **Conclusion:** retrain on local labelled data, or use unsupervised or low-frequency methods (approach H).

### 15- and 30-minute resolution
Answer with your Phase 4 numbers. Expected: total energy over days or weeks is still estimable, but individual cycle detection, timing and peak power are lost; models suited to this are behaviour or time-of-use models, not shape-based CNNs.

---

## 8. Repository layout

```
refit-nilm/
├── README.md              # what, how to run, assumptions, decisions, trade-offs, AI tools used, next steps
├── DATA.md                # raw coverage, quality findings, cleaning rules, remaining concerns
├── RESULTS.md             # EDA, model table, validation, failure analysis, interpretation
├── Recommendation.md      # one page for a utility audience
├── requirements.txt
├── config.yaml            # paths, houses, WM columns, thresholds, seeds
├── run_all.py             # runs every step in order
├── src/
│   ├── load.py            # 7z extract, column selection, 1-min resample, Parquet cache
│   ├── clean_raw.py       # Part 1
│   ├── cycles.py          # cycle detection (shared by EDA and evaluation)
│   ├── eda.py             # Part 2
│   ├── features.py        # LightGBM features
│   ├── models/
│   │   ├── baselines.py
│   │   ├── lgbm.py
│   │   └── seq2point.py
│   ├── evaluate.py        # metrics and plots
│   └── resolution.py      # Phase 4
├── notebooks/             # optional, thin notebooks calling src/ for walkthroughs
├── figures/
└── results/
```

**Note:** the data itself goes in `.gitignore`. The README explains how to download it.

### Recommendation.md outline (one page)
1. **What the model can do:** estimate washing-machine energy per day or week and detect most cycles from 1-minute whole-home data, with accuracy figures from the test house.
2. **Suitable uses:** aggregate consumer insights, programme targeting, campaign evaluation.
3. **Not suitable for:** billing, per-cycle guarantees, or 15–30 minute meters without retraining.
4. **Energy-saving opportunity:** the wash-temperature nudge and how to measure it.
5. **Data that would increase confidence:** local sub-metered pilot homes in India, appliance surveys, higher-frequency (≤1 min) data from consumer access devices, and reactive power.

---

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Data files are large and slow to load | Read only the needed columns; cache as Parquet once |
| Seq2Point is slow on a CPU | Smaller window, fewer filters, subsampled off-windows, fewer epochs; LightGBM as a fallback main model |
| Wrong appliance column for a house | Label audit plots in the EDA; exclude any doubtful house |
| Test house 8 is unrepresentative (washer-dryer) | Add a second test house; discuss it explicitly |
| Over-tuning to the test set | Freeze everything on house 18 before touching house 8 |
| Running out of time | Follow the cut order in §6 |

---

## 10. AI-use disclosure (for README)

The brief asks which AI tools you used. Be specific: for example, "Claude was used to research NILM approaches and draft the project plan; all code was written, run and verified by me, and all numbers come from my own runs."

---

## 11. References

1. Murray, D., Stankovic, L., & Stankovic, V. (2017). An electrical load measurements dataset of United Kingdom households from a two-year longitudinal study. *Scientific Data*, 4, 160122. https://doi.org/10.1038/sdata.2016.122
2. REFIT cleaned data readme (CLEAN_READ_ME_081116.txt). University of Strathclyde. https://pure.strath.ac.uk/ws/portalfiles/portal/62090183/CLEAN_READ_ME_081116.txt
3. Zhang, C., Zhong, M., Wang, Z., Goddard, N., & Sutton, C. (2018). Sequence-to-point learning with neural networks for non-intrusive load monitoring. *AAAI*.
4. D'Incecco, M., Squartini, S., & Zhong, M. (2019). Transfer learning for non-intrusive load monitoring. Code and REFIT house splits: https://github.com/MingjunZhong/transferNILM
5. Kelly, J., & Knottenbelt, W. (2015). Neural NILM: Deep neural networks applied to energy disaggregation. *ACM BuildSys*.
6. Yue, Z., et al. (2020). BERT4NILM: A bidirectional transformer model for non-intrusive load monitoring. *NILM Workshop 2020*.
7. Sykiotis, S., et al. (2022). ELECTRIcity: An efficient transformer for non-intrusive load monitoring. *Sensors*, 22(8), 2926.
8. Barber, J., et al. (2020). Lightweight non-intrusive load monitoring employing pruned sequence-to-point learning. *NILM Workshop 2020*.
9. Zhao, B., Stankovic, L., & Stankovic, V. Blind non-intrusive appliance load monitoring using graph-based signal processing. University of Strathclyde.
10. Altrabalsi, H., Stankovic, V., Liao, J., & Stankovic, L. Low-complexity energy disaggregation using appliance load modelling. *AIMS Energy*.
11. Mollel, R. S., et al. (2026). Uncovering hidden demand flexibility using NILM: a case for Southern Africa – Namibia. *EEDAL*. https://strathprints.strath.ac.uk/90523
