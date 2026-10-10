# Problem Understanding, Literature Review & Solution Rationale

> Purpose: build a shared mental model of the problem before writing any code.  
> Date: 2026-10-08

---

## 1. What Problem Are We Actually Solving?

### The Core Challenge

A household has **one electricity meter** at the entry point. It measures total power drawn at every moment — what is called the **aggregate signal**. Behind that meter, dozens of appliances are running at different times: a fridge cycling on and off, a washing machine running a full programme, a kettle boiling for 3 minutes.

**NILM (Non-Intrusive Load Monitoring)** asks: *given only the aggregate signal, can you estimate how much each appliance is using, without installing a sensor on every device?*

This is the "non-intrusive" part — you are not permitted to touch individual appliances. You only ever see the sum.

### Why It Is Hard

1. **The signal is a superposition.** At any given second, the aggregate = fridge + washing machine + TV + everything else. Inverting a sum without knowing which terms are active is fundamentally under-determined.

2. **Appliances overlap.** A washing machine's heating phase (~2 kW) looks identical in the aggregate to an electric shower, a kettle, or a tumble dryer. Without temporal context (the heating lasted 12 minutes, not 3), you cannot tell them apart.

3. **Appliances have multiple states.** A washing machine is not simply "on" or "off". It cycles through: fill → heat → wash → rinse → spin → drain. Each phase has a different power level. A model must capture the *shape over time*, not just a single power value.

4. **Labels are imperfect.** Even in REFIT — a carefully curated research dataset — sensors were moved, appliances were changed mid-study, and labels were lost. In real deployments there are no labels at all.

5. **The baseline trap.** A washing machine is off for roughly 90–95% of every minute in the dataset. A model that **always predicts zero** achieves a very low MAE and is completely useless. This makes naive evaluation metrics actively misleading.

### Mathematical Formulation

Let `y(t)` = aggregate power at time `t`.  
Let `x_i(t)` = power of appliance `i` at time `t`.  
Let `u(t)` = "unmetered" background (appliances without sensors).

Then:  `y(t) = Σ x_i(t) + u(t)`

The goal is to estimate `x_wm(t)` — the washing-machine power time series — from `y(t)` alone, where `u(t)` is unknown and treated as noise.

This is a **noisy, partially observed, single-channel blind source separation** problem. Deep learning approaches treat it as supervised regression: learn a mapping `f(y[t-w : t+w]) → x_wm(t)` from labelled training examples.

---

## 2. The REFIT Dataset — What It Is and What Was Done to It

### The Raw Data (what we have in `data/raw/`)

- Logged by a gateway device that **polled sensors every 6–8 seconds**.
- Logged a row only **when a value changed** — not on a fixed schedule.  
  → Result: intervals of 1–15 s are common; gaps up to hours exist when nothing changed.
- Sensors were **not synchronised** — each sensor updated independently. One poll cycle can produce 9 separate rows, each 1–2 s apart.
- Each house comes as **Part1 + Part2**:
  - Part1: missing sensors encoded as **0** (cannot distinguish from genuine zero load).
  - Part2: missing sensors encoded as **NaN** (distinguishable, introduced mid-study).
- Timestamps are in **local UK time** — BST (UTC+1) in summer, GMT (UTC+0) in winter. The Unix timestamp is also in local time, not UTC.

### The Official REFIT Cleaning Pipeline (from CLEAN_README.md)

Five transformations were applied, and each one **hides something**:

| Step | What was done | What it hides / why it matters |
|---|---|---|
| **1. Daylight-saving correction** | Both the datetime string AND the Unix timestamp were shifted to UTC | The raw Unix timestamps are local-time. Any time-zone-naive merge of raw + clean will be wrong by 3600 s. Verified: clean Unix = raw Unix − 3600 during BST. |
| **2. Appliance column re-alignment** | When a household moved a plug sensor, the column was corrected retroactively to always represent the intended appliance | We cannot detect these corrections from data alone. The label is trustworthy but the raw column was a different appliance during some period. |
| **3. Forward-fill of NaN** | All missing values were forward-filled | A long outage (e.g. Feb 2014) becomes a **flat line** at the last seen value — indistinguishable from genuine constant load unless you detect the run-length. This is the most dangerous artifact in the clean data for model training. |
| **4. Spike removal (IAM channels)** | Values > 4,000 W on any appliance channel were replaced with **zero** | Short genuine peak loads above 4 kW are lost. For washing machines (max ~2,700 W) this is fine. For aggregate (not capped), spikes can still reach 11 kW. |
| **5. Issues flag** | `Issues = 1` when Σ(appliances) > Aggregate — a physical impossibility | These rows must be excluded from training and evaluation. Using them trains the model on self-contradictory data. |

### What the Clean Data Looks Like After These Steps

```
Time (UTC)     Aggregate  App1  App2  App3  App4  App5  ... App9  Issues
2013-10-09     523        74    0     69    0     0     ... 1     0
2013-10-09     526        75    0     69    0     0     ... 1     0
(~8s interval, irregular)
```

Still at 8-second irregular resolution. **Resampling to 1-minute is our job**, not done by the official pipeline.

---

## 3. The Washing Machine Signal — What It Actually Looks Like

A European front-loading washing machine runs a programme that lasts **60–120 minutes** and has four distinct power phases:

```
Power (W)
2500 |    ████                              ██ ████
2000 |    ████                              ████████
1500 |
1000 |
 500 |         █████████████████████████  ██        ███
   0 |____|____|____|____|____|____|____|____|____|___
       0   10   20   30   40   50   60   70   80   90 min

Phase:  [Heat] [---------- Wash/Rinse --------] [Spin]
```

- **Heating phase** (~10–20 min): draws 1,800–2,500 W to heat water. This is by far the most energy-intensive phase and is directly temperature-sensitive. A 60°C wash has a long, high-power heating phase; a 30°C wash has a shorter one or none.
- **Wash/rinse phase** (~40–70 min): intermittent low power (50–300 W) from drum rotation and water pump cycling. May include additional heating pulses for a rinse.
- **Spin phase** (~10–15 min): 300–800 W bursts from the motor. Power is jagged and irregular.

**Why this shape matters for modelling:**
- The full cycle spans 60–120 minutes. A model must see the **whole shape in context** to recognise it — you cannot identify a washing machine cycle from a 5-minute window alone.
- The heating phase is what most models latch onto. But a cold-water or eco wash skips it, making the signature much harder to separate from background noise.
- At 1-minute resolution, the 8-second spin bursts average out — you lose the individual spin pulses but the cycle envelope remains intact.

**Key practical implication:** at 15- or 30-minute resolution, the heating phase becomes a single wide bar. You can still estimate total energy but you cannot reliably detect cycle boundaries or count cycles. This is the core answer to Flock's India question.

---

## 4. Key Literature — What Each Paper Contributes

### 4.1 Hart (1992) — "Nonintrusive Appliance Load Monitoring" (NILM origin paper)
- **Idea**: detect step changes (edges) in power and match them to known appliance power signatures stored in a library.
- **Approach**: event-driven; only uses instantaneous power. Works at very high frequency (kHz).
- **Why it matters here**: baseline intuition. A washing machine's start and end are large edges (+2 kW at start, −2 kW at end). But the wash phase is not a simple step — it drifts. Event-based methods fail at 1-minute resolution where transients are averaged away.
- **What we take**: the idea of a **rule-based baseline** — detect ~2 kW rises and falls in the aggregate.

### 4.2 Kelly & Knottenbelt (2015) — "Neural NILM" (BuildSys)
- **Idea**: apply deep neural networks (LSTM, denoising autoencoder, regression network) to energy disaggregation.
- **Key contribution**: first to show deep learning beats classical FHMM/CO approaches on UKDALE and REDD.
- **Activation thresholds for washing machine**: on if > 20 W, minimum on-duration 30 s, minimum off-gap 30 s. (Note: our data shows 20 W may be too low — see House 5 standby finding.)
- **What we take**: the general framing of NILM as supervised regression on windows; the 20 W threshold as a starting point to tune.

### 4.3 Zhang et al. (2018) — "Seq2Point Learning" (AAAI)
- **Idea**: instead of predicting the whole output window (Seq2Seq), predict only the **midpoint** of the output window corresponding to the input window. This single-point prediction is simpler and trains faster.
- **Architecture**: 5 1D convolutional layers (filters: 30-30-40-50-50, kernel sizes: 10-8-6-5-5), one dense layer. Input window = 599 samples at 8 s ≈ 80 minutes.
- **Key results**: up to 83% MAE reduction over prior work on UKDALE/REDD.
- **Why it works for washing machines**: the wide receptive field (599 samples ≈ full cycle) lets the CNN see the entire programme shape — heat, wash, spin — in context.
- **What we adapt**: at 1-minute resolution, 599 samples = 10 hours (too wide). Use **~199 samples ≈ 3.3 hours** to cover one full cycle with context. The architecture (5 conv layers) stays the same.
- **What we take**: this is our main deep model. The house-level train/test split and normalisation scheme come from this lineage.

### 4.4 D'Incecco et al. (2019) — "Transfer Learning for NILM" (IEEE Trans. Smart Grid)
- **Key contribution**: established a **canonical REFIT house split** for the washing machine:
  - Train: 2, 5, 7, 9, 15, 16, 17
  - Validate: 18
  - Test: **8**
- **Why this split**: it tests generalisation to unseen homes (the only thing a utility cares about). House 8 is intentionally hard — it has a washer-dryer alongside the WM, creating overlapping signatures.
- **What we take**: adopt this split exactly, so results are comparable to published benchmarks.

### 4.5 Precioso & Gómez-Ullate (2020) — "NILM as Regression vs. Classification" (arXiv:2010.16050)
- **Core finding**: models trained as regression (predict Watts) and evaluated with MAE will look good even if they predict near-zero everywhere — because WM is off >90% of the time.
- **The fix**: apply a threshold to the predicted signal to get on/off states, then compute F1. But the threshold matters enormously — a bad threshold turns a decent regression into a terrible classifier.
- **What we take**: always report both MAE (split into on/off periods) AND F1 with a stated threshold. Justify the threshold using validation data only.

### 4.6 Murray et al. (2017) — "REFIT Dataset Paper" (Scientific Data)
- **Key numbers**: 20 houses, ~8 s sampling, Oct 2013–Jul 2015, 1.19 billion readings, 250,000+ appliance uses monitored.
- **What we cite**: when stating what dataset we used and its scope.

### 4.7 Zhao et al. (2020) — Strathclyde — "Non-intrusive disaggregation for very low-rate smart meter data"
- **Directly relevant to India**: shows that at 15–30 min resolution, shape-based methods (like Seq2Point) fail because the cycle envelope is compressed to 2–4 samples. Behaviour-based approaches (what time of day does this household do laundry?) become the only viable route.
- **What we cite**: in the resolution experiment and in the India limitations section.

---

## 5. The Official REFIT Cleaning Pipeline vs. What We Do

This comparison is important because the brief says *"you are not expected to reproduce every specialised correction."* We need to know what we are and are not doing.

```
OFFICIAL REFIT PIPELINE                    OUR RAW CLEANING (Section 1)
───────────────────────────────            ──────────────────────────────────
1. BST → UTC correction                    We identify and document the offset.
   (both datetime + Unix shifted)          We do NOT re-correct (we use clean data
                                           for modelling; raw cleaning is diagnostic).

2. Appliance column re-alignment           We CANNOT reproduce — we have no record
   (plug moved mid-study → column          of when plugs were moved. We accept the
   corrected retroactively)                clean data's corrections as ground truth.

3. Forward-fill NaN                        We detect long flat-line runs (> 5 min)
   (all missing → filled with last value)  in the clean data and flag them as outages.
                                           We do NOT fill them further.

4. Spike removal (IAM > 4 kW → 0)         We apply our own threshold: any IAM
                                           reading > 4 kW set to NaN (not zero),
                                           log count, don't silently drop.

5. Issues flag (Σ apps > aggregate → 1)   We exclude Issues=1 rows from all
                                           modelling and evaluation.

OUR ADDITIONS (not in official pipeline)
─────────────────────────────────────────
6. Deduplicate: same-second rows → keep first (or mean), log count and choice.
7. Resample to 1-minute mean.
8. Gaps ≤ 5 min → linear interpolation (shorter than any appliance cycle).
9. Gaps > 5 min → leave as NaN, set outage flag.
10. Per-house: detect flat-line runs > 30 min in cleaned WM channel → flag as forward-fill artifact.
```

The key insight: **the official cleaning hides outages behind flat lines.** Our job is to re-detect them. A Parquet cache with an `outage` flag column per row is the clean deliverable.

---

## 6. Why Each Model in the Stack Exists

### Always-off baseline (predict 0 W everywhere)
- Achieves low overall MAE because WM is off 90–95% of the time.
- Exists only to **expose the baseline trap** and anchor the evaluation table.
- Any model that does not clearly beat this is not useful.

### Rule-based baseline
- Find moments in the aggregate where power rises by 1,500–2,500 W and stays elevated for ≥ 30 minutes, then falls again.
- Interpretable, deployable without training data, fast.
- Fails when: a kettle (3 min) + dishwasher overlap mimics the WM start; or when the WM runs at night during low-activity periods where the signal is cleaner.
- **Why include it**: sets the floor for what domain knowledge alone can achieve. If Seq2Point barely beats it, the deep model is not worth the complexity.

### LightGBM on rolling features
- Features: rolling mean, std, max, diff, and quantiles of the aggregate over 5, 15, 30, 60, 120-min windows. Hour of day, day of week.
- Captures temporal patterns and typical usage times without needing to see the whole cycle shape.
- Trains in seconds on a laptop; early stopping on the validation house gives a robust model.
- **Why include it**: strong, cheap, explainable. Feature importance shows which temporal scale matters most. If it nearly matches Seq2Point, deep learning is unnecessary for this resolution.

### Seq2Point CNN
- The window centred on the target point captures the full cycle shape: heating phase before midpoint, spin phase after midpoint.
- The 5 conv layers learn to recognise the characteristic power envelope of a washing machine across different households automatically — no hand-crafted features.
- **Why it is appropriate**: the washing machine is a multi-phase, multi-hour appliance. The CNN's wide receptive field is the right tool. A point-in-time model (LightGBM) misses the cycle shape; a sequence model captures it.
- **The trade-off**: black box, requires GPU or time, needs sufficient labelled data. For 20 houses, training is manageable.

---

## 7. Why the India / Resolution Question Matters to Flock

Flock Energy's customers are Indian distribution utilities. Smart meters deployed under India's Revamped Distribution Sector Scheme (RDSS) report at **15–30 minute intervals** — not 1 minute.

At 1-minute resolution:  
- A washing machine cycle of 90 minutes = 90 data points. The heating, wash, and spin phases are all visible.

At 15-minute resolution:  
- The same 90-minute cycle = **6 data points**. The heating phase collapses into one or two intervals mixed with other load. The spin bursts disappear. Cycle boundaries are ambiguous.

At 30-minute resolution:  
- 3 data points. Individual cycle detection is impossible. You can at best estimate that *sometime in this half-hour, the WM was probably running.*

**The transfer gap for Indian homes is even wider:**
- Many Indian washing machines are **top-loaders or semi-automatics** — they often have no internal heater (water is heated separately by a geyser). The 2 kW heating signature that Seq2Point learns to recognise does not exist.
- The Indian load mix is dominated by ACs, geysers, water pumps, and inverter/battery backup — none of which appear in REFIT. The model's sense of "background noise" is completely wrong.

**The honest answer for Recommendation.md:**  
A Seq2Point model trained on REFIT is not deployable to Indian 15/30-minute smart meter data. For Indian deployment, the viable paths are: (a) collect a small labelled pilot dataset in India and fine-tune, or (b) use unsupervised methods that work at low frequency (usage-pattern clustering, Strathclyde's graph-signal approach) and do not depend on a 2 kW heating signature.

---

## 8. Intuitive Solution Path (Informed by Research)

**The approach that makes both engineering and business sense:**

```
Step 1 — Understand before modelling
  Raw data → clean with documented rules → 1-min Parquet cache
  Outage flag, Issues exclusion, forward-fill detection
  This step answers: "do you know what is in your data?"

Step 2 — Understand the appliance before predicting it
  EDA on all WM houses → cycles, timing, energy, seasonal patterns
  Heating-phase fraction per house → energy-saving hook
  Label audit → which houses look like WM? which don't?
  This step answers: "do you know what you are trying to predict?"

Step 3 — Build a ladder of models
  Always-off → Rule-based → LightGBM → Seq2Point
  Each one must beat the previous to justify its complexity.
  Evaluate on ONE unseen house (House 8), using metrics that cannot be gamed:
    - F1 (on/off) with stated threshold
    - SAE (total energy)
    - Daily kWh error
    - Cycle-level detection F1

Step 4 — Be honest about failure
  Show 3 concrete failure cases: dryer confusion, cold wash miss, label error.
  This is not weakness — it is the most useful output for a deployment decision.

Step 5 — Answer the India question with evidence
  Downsample to 15 and 30 min. Retrain LightGBM. Measure cycle F1 collapse.
  State what would need to change for Indian deployment.
  This is the differentiating deliverable.
```

---

## 9. Summary: What Makes This Solution Defensible

| Decision | Reasoning |
|---|---|
| 1-minute resample using mean | Consistent with published REFIT NILM benchmarks; mean is appropriate for power (energy-preserving) |
| House-level train/test split | Leakage-free; adjacent minutes are nearly identical, random splits would inflate accuracy |
| Standard split (train 2,5,7,9,15,16,17 / val 18 / test 8) | Reproducible comparison with published benchmarks; House 8 is a hard, realistic test |
| 50 W on-threshold (not 20 W) | Empirical: House 5 standby is 1–5 W; 20 W threshold captures noise, not cycles |
| SAE + daily kWh as primary metrics | These are what a utility cares about (billing, reporting) — not instantaneous watts |
| Exclude Issues=1 rows | Physical impossibility; including them trains the model on corrupted supervision signal |
| Flag flat-line runs > 30 min as outage | Official pipeline forward-fills silently; we make this visible in our features |
| Seq2Point window = 199 samples (~3.3 h) | Covers one full WM cycle with context on both sides; 599-sample original is 80 min at 8 s |

---

## 10. References

1. Hart, G. W. (1992). Nonintrusive appliance load monitoring. *Proceedings of the IEEE*, 80(12), 1870–1891.
2. Kelly, J., & Knottenbelt, W. (2015). Neural NILM: Deep neural networks applied to energy disaggregation. *ACM BuildSys*. https://arxiv.org/abs/1507.06594
3. Zhang, C. et al. (2018). Sequence-to-point learning with neural networks for non-intrusive load monitoring. *AAAI*. https://arxiv.org/abs/1612.09106
4. D'Incecco, M. et al. (2019). Transfer learning for non-intrusive load monitoring. *IEEE Trans. Smart Grid*. https://arxiv.org/abs/1902.08835
5. Murray, D. et al. (2017). An electrical load measurements dataset of UK households from a two-year longitudinal study. *Scientific Data*, 4, 160122. https://strathprints.strath.ac.uk/58873/
6. Precioso, D. & Gómez-Ullate, D. (2020). NILM as a regression versus classification problem: the importance of thresholding. arXiv:2010.16050. https://arxiv.org/abs/2010.16050
7. Zhao, B. et al. (2020). Non-intrusive load disaggregation solutions for very low-rate smart meter data. *Applied Energy*. https://strathprints.strath.ac.uk/72013/
8. REFIT Cleaned Data README, University of Strathclyde. https://pure.strath.ac.uk/ws/portalfiles/portal/62090183/CLEAN_READ_ME_081116.txt
9. Survey: NILM algorithms and techniques. arXiv:1703.00785. https://arxiv.org/abs/1703.00785
10. Gradient Boosting for multi-label appliance state classification in NILM using low-frequency data. *Media Elektrik Journal* (2023). https://journal.unm.ac.id/index.php/mediaelektrik/article/view/9169
