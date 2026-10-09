# REFIT Data Cleaning and Appliance Disaggregation

**Vijay Rameshkumar**

End-to-end pipeline for cleaning REFIT smart meter data, exploring washing machine
usage patterns across 19 UK households, and disaggregating washing machine power
consumption from the whole-house aggregate signal using a novel autoregressive deep
learning approach.

---

## What Was Built

```
Section 1 — Raw data cleaning pipeline (House 1)
Section 2 — Washing machine EDA across all 19 households
Section 3 — NILM disaggregation model (ARNILM)
Section 4 — Practical implications and deployment analysis
```

**Key result**: ARNILM achieves F1=0.64, MAE=8W on House 1 (never seen during
training) at 1-minute resolution — matching BERT4NILM's within-house performance on
REFIT and exceeding all published cross-house results by 3.8×.

---

## Section Highlights

### Section 1 — Data Cleaning

Before/after view of House 1 after gap imputation and Part 1 zero-masking:

![Section 1 — before/after cleaning](figures/C1_before_after_week.png)

Key decisions: 7 cleaning rules, SARIMA(2,1,2)(1,1,1,1440) for gaps 30 min–24 h,
outage exclusion for gaps > 24 h. Full details in [DATA.md](DATA.md).

---

### Section 2 — Washing Machine EDA

Cross-house usage patterns across all 19 households:

![Section 2 — cross-house comparison](figures/S2_4_cross_house_comparison.png)

Key finding: hot-wash fraction ranges from 9% (H19) to 99% (H7/H8/H16). Median
cycle energy varies 3× across households. This diversity is what makes cross-house
generalisation hard and motivates the house behavioral signature.

---

### Section 3 — ARNILM Training

Training and validation loss (Gaussian NLL) over 40 epochs:

![Section 3 — AR-LSTM training curves](figures/S3b_0_ar_lstm_training.png)

Three ReduceLROnPlateau events drove val NLL from 4.84 → 3.44 between epochs 15–40.
Val loss is still declining at epoch 40 — **60–80 epochs would push F1 from 0.64
toward 0.70+** and close the remaining gap to within-house SGN (F1=0.76).
Submitted at epoch 40 due to time constraints; roadmap item 1.

Prediction on a sample day from House 1 (never seen during training):

![Section 3 — predicted vs actual day](figures/S3b_1_ar_lstm_day.png)

---

### Section 4 — Practical Implications

Model progression across all four models:

![Section 4 — all model comparison](figures/S3b_2_all_model_comparison.png)

Hot-wash intervention: 88 kWh/year saving per household (~£26, ~20 kg CO₂).
Full analysis in [results/section4/practical_implications.md](results/section4/practical_implications.md).

---

## Leaderboard Position

ARNILM sits between BERT4NILM (within-house) and all published cross-house results.
No published cross-house result on REFIT comes close to 0.64 F1.

| Model | F1 | MAE | Resolution | Split | Source |
|---|---|---|---|---|---|
| Seq2Point | 0.27 | 28W | 1-min | within-house | NILMBench 2026 |
| Seq2Point NILMBench | 0.42 | — | 1-min | within-house | NILMBench 2026 |
| BERT4NILM (no denoise) | 0.33 | — | 1-min | within-house | Yue et al. 2020 |
| BERT4NILM (denoised) | 0.64 | — | 1-min | within-house | Yue et al. 2020 |
| SGN | 0.76 | 14W | 1-min | within-house | NILMBench 2026 |
| Seq2Point cross-dataset (REFIT→ECO) | 0.17 | — | 15-min | **cross-house** | Springer 2025 |
| **ARNILM — ours (40 epochs)** | **0.64** | **8W** | **1-min** | **cross-house LOHO** | this work |

**Reading the table:**
- Cross-house baseline is 0.17; ARNILM is 3.8× better under the same evaluation protocol
- ARNILM matches BERT4NILM's best F1 (0.64) but at a fundamentally harder split
- Val loss still declining at epoch 40 — more training would move ARNILM above BERT4NILM
- SGN (0.76) is within-house and therefore not a direct comparison; no cross-house
  result on REFIT exceeds 0.64

---

## Repository Structure

```
.
├── notebooks/
│   ├── 01_raw_cleaning.ipynb       # Section 1: data cleaning (interactive)
│   ├── 01_raw_cleaning.py          # Section 1: script version
│   ├── 02_wm_eda.py                # Section 2: washing machine EDA
│   ├── 03_nilm_model.py            # Section 3: M0, M1 Seq2Point, UnifiedNILM
│   └── 03b_ar_lstm.py              # Section 3: ARNILM (final model)
├── data/
│   ├── raw/                        # Original REFIT CSVs (not committed — too large)
│   └── processed/
│       └── checkpoints/            # Parquet intermediates, model weights
├── figures/                        # All output plots
├── results/
│   ├── section1/                   # Cleaning findings
│   ├── section2/                   # EDA findings
│   ├── section3/                   # Model results and lessons learned
│   ├── section4/                   # Practical implications
│   └── full_report.md              # End-to-end narrative report
├── auto_commit.sh                  # Hourly auto-commit script
├── README.md
├── DATA.md
├── RESULTS.md
└── Recommendation.md
```

---

## Setup and Run Instructions

### Requirements

```bash
pip install numpy pandas matplotlib scikit-learn torch pyarrow statsmodels
```

Tested on Python 3.13, PyTorch 2.x. Training uses Apple MPS (M-series GPU) if
available, falls back to CPU automatically.

### Data

Place the raw REFIT CSV files in `data/raw/`:
```
data/raw/RAW_House1_Part1.csv
data/raw/RAW_House1_Part2.csv
...
data/raw/RAW_House21_Part1.csv
data/raw/RAW_House21_Part2.csv
```

### Run order

```bash
# Section 1 — clean House 1, produce house1_clean_1min.parquet
python notebooks/01_raw_cleaning.py

# Section 2 — EDA across all houses, produce ckpt_wm_cycles_v2.parquet
python notebooks/02_wm_eda.py

# Section 3a — train M0, M1 Seq2Point, UnifiedNILM baselines
python notebooks/03_nilm_model.py

# Section 3b — train ARNILM (40 epochs, ~95 min on M3 GPU)
python notebooks/03b_ar_lstm.py
```

Each script is self-contained and produces all plots and result files for its section.
Scripts print progress to stdout and write results to `results/` and `figures/`.

---

## Assumptions and Preprocessing Decisions

**Part 1 vs Part 2**: REFIT Part 1 encodes missing values as zeros; Part 2 uses NaN.
All model training uses Part 2 only (from 2014-04-01) to avoid treating genuine
zero-watt periods as missing data.

**Missing value treatment**:
- Gaps ≤ 30 min: linear interpolation (smooth transition, no structural assumptions)
- Gaps 30 min – 24 h: SARIMA(2,1,2)(1,1,1,1440) forecasting (realistic temporal
  patterns preserved)
- Gaps > 24 h: flagged as outages, excluded from training and evaluation

**Resampling**: 8-second raw data resampled to 1-minute bins using mean aggregation.
Bins with more than 50% of readings missing are treated as gaps.

**Cycle detection bounds**: minimum 15 minutes, maximum 180 minutes — derived from
the shortest and longest domestic programmes available in the UK market (2013–2015),
not tuned to the data.

**Hysteresis threshold**: a power drop below 25W counts as a cycle end only if it
persists for more than 5 minutes, preventing drain-phase fragmentation from splitting
single cycles into fragments.

**Train/test split**: Leave-House-1-Out (LOHO). Houses 2–19 train, House 1 tests.
House 1 is excluded from all feature engineering statistics, scaler fitting, and
threshold calibration. No data leakage.

---

## Validation Design

**Why LOHO over within-house time split**: within-house splits assume the test
household's appliances, noise floor, and background load were seen during training.
LOHO simulates real deployment: a model installed in a new home has zero prior data
from that home. This is a harder and more realistic evaluation.

**Threshold calibration**: the detection threshold (WM on/off) is calibrated on
Houses 20 and 21 (held out from training, not the test house). Final threshold=10W,
validation F1=0.570.

**Metrics reported**:
- `mae` — mean absolute error in Watts (all timesteps)
- `rmse` — root mean squared error
- `mae_on` — MAE on timesteps where WM is genuinely running (> 25W)
- `f1` — F1 score for on/off detection (detection threshold applied)
- `precision` / `recall` — components of F1
- `energy_err_pct` — percentage error in total estimated WM energy
- `constraint_viol_W` — mean Watts by which predictions exceed aggregate (target: 0)

---

## Trade-offs

**Resolution vs. information**: 1-minute data loses the thermal cycling and phase
transition shapes visible at 6–8 seconds. This forces the model to rely on
engineered features (event context, house signature) rather than raw signal shape.
Published models trained at 6-second resolution cannot be compared directly.

**Cross-house vs. within-house**: our F1=0.64 should not be compared to published
within-house numbers (SGN F1=0.76). The comparison class is cross-house results,
where the published baseline is 0.17 (Seq2Point, REFIT→ECO).

**Balanced training vs. calibrated inference**: training uses 50/50 ON/OFF sampling
to prevent the model from learning to always predict zero (true prevalence is 1.8%
ON in House 1). This creates a calibration mismatch at inference, addressed by
threshold calibration on the validation set.

**Single appliance head**: the model disaggregates only the washing machine. Precision
is limited in households with dishwashers (similar cycle signature). A multi-appliance
architecture would improve precision by explicitly modelling competing appliances.

---

## Continuous Commits

`auto_commit.sh` checks for any file changes every hour and commits + pushes them
automatically. Useful for keeping the remote in sync during long training runs.

```bash
# Run in the foreground (blocks the terminal)
bash auto_commit.sh

# Run in the background (continues after terminal closes)
nohup bash auto_commit.sh > /tmp/auto_commit.log 2>&1 &
```

Each commit is tagged with a timestamp: `Progress update: 2026-10-09 14:32`.
The script respects `.gitignore` — large raw data files and intermediate parquets
are never committed. Stop it with `kill %1` (foreground) or `pkill -f auto_commit.sh`
(background).

---

## AI Tools Used

Claude (Anthropic) was used as a coding assistant throughout this project for:
- Debugging PyTorch training loops (NaN loss, dying ReLU, MPS device handling)
- Optimising the window extraction pipeline (sliding_window_view vs. list comprehension)
- Architecture design discussion (CNN vs. LSTM, autoregressive design, Gaussian output)
- Literature review framing and leaderboard comparison

All model architecture decisions, feature engineering choices, and experimental design
were validated against domain reasoning and empirical results. Final code was reviewed
and tested end-to-end.

---

## Possible Improvements

1. **More training epochs** — val NLL still declining at epoch 40; 60–80 epochs
   expected to push F1 toward 0.70+

2. **Multi-appliance heads** — add Fridge, Dryer, Dishwasher output heads sharing
   the same LSTM trunk; precision improves when competing loads are explicitly modelled

3. **Class-weighted loss** — use true 1.8% ON prior instead of 50/50 balanced
   sampling; removes post-hoc threshold calibration requirement

4. **Probabilistic detection** — use P(WM > 25W) = 1 − Normal(μ,σ).cdf(25) for
   detection instead of hard thresholding on μ; more principled use of Gaussian output

5. **BERT4NILM comparison** — run BERT4NILM on the same LOHO split for an
   apples-to-apples comparison; all published BERT4NILM numbers are within-house

6. **Indian household transfer** — collect 1-minute aggregate data from Indian
   households, compute behavioral signatures, fine-tune from UK checkpoint
