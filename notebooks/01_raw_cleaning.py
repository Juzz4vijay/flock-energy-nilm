# %% [markdown]
# # Section 1 — Raw Data Inspection and Cleaning: House 1
#
# **Pipeline:** Detect → Plan → Build → Validate
#
# **Checkpoint design:** each expensive stage saves to `data/processed/checkpoints/`.
# Re-running a cell skips if checkpoint exists — set `FORCE_RERUN = True` to override.
#
# House 1 appliance map:
# - Appliance1: Fridge | Appliance2: Freezer 1 | Appliance3: Freezer 2
# - Appliance4: Washer-Dryer | **Appliance5: Washing Machine** ← target
# - Appliance6: Dishwasher | Appliance7: Computer | Appliance8: TV | Appliance9: Heater

# %%
import warnings
warnings.filterwarnings('ignore')

import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from pathlib import Path

RAW_DIR  = Path('/Volumes/T7/Projects/FlockEnergy/data/raw')
FIG_DIR  = Path('/Volumes/T7/Projects/FlockEnergy/figures')
OUT_DIR  = Path('/Volumes/T7/Projects/FlockEnergy/data/processed')
CKPT_DIR = Path('/Volumes/T7/Projects/FlockEnergy/data/processed/checkpoints')
for d in [FIG_DIR, OUT_DIR, CKPT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

IAM_SPIKE_THRESHOLD = 4000
AGG_SPIKE_THRESHOLD = 15000
AGG_ROLL_W          = 75      # ~10 min at 8s polling
AGG_MAD_MULT        = 5
SHORT_GAP_MIN       = 5
LONG_GAP_H          = 24
OUTAGE_FLATLINE_MIN = 30
SARIMA_SEASONAL     = 24
ANOMALY_SIGMA       = 3
APPLIANCE_COLS      = [f'Appliance{i}' for i in range(1, 10)]
ALWAYS_ON_COLS      = ['Appliance1', 'Appliance2', 'Appliance3']

FORCE_RERUN = False   # set True to recompute all checkpoints from scratch
print('Setup complete.')

# %% [markdown]
# ---
# ## PHASE 1 — DETECT

# %%
# ── Stage 1: Load + sort (checkpoint) ─────────────────────────────────────
CKPT_RAW = CKPT_DIR / 'ckpt_raw_sorted.parquet'

if CKPT_RAW.exists() and not FORCE_RERUN:
    print('Loading raw checkpoint ...')
    raw_sorted = pd.read_parquet(CKPT_RAW)
    raw_sorted['Time'] = pd.to_datetime(raw_sorted['Time'])
    print(f'  {len(raw_sorted):,} rows  |  {raw_sorted["Time"].iloc[0]} → {raw_sorted["Time"].iloc[-1]}')
else:
    print('Loading RAW House 1 Part 1 ...')
    p1 = pd.read_csv(RAW_DIR / 'RAW_House1_Part1.csv', low_memory=False)
    print(f'  Part1: {len(p1):,} rows')
    print('Loading RAW House 1 Part 2 ...')
    p2 = pd.read_csv(RAW_DIR / 'RAW_House1_Part2.csv', low_memory=False)
    print(f'  Part2: {len(p2):,} rows')

    raw = pd.concat([p1, p2], ignore_index=True)
    raw['Time'] = pd.to_datetime(raw['Time'])
    raw_sorted = raw.sort_values('Unix').reset_index(drop=True)
    raw_sorted.to_parquet(CKPT_RAW)
    print(f'  Checkpoint saved → {CKPT_RAW.name}')

print(f'Combined: {len(raw_sorted):,} rows | '
      f'{raw_sorted["Time"].iloc[0].date()} → {raw_sorted["Time"].iloc[-1].date()}')

# %%
# ── Time diff distribution ─────────────────────────────────────────────────
diffs_s = raw_sorted['Unix'].diff().dropna()
out_of_order_raw = int((raw_sorted['Unix'].diff() < 0).sum())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('House 1 Raw — Time Difference Distribution', fontsize=13, fontweight='bold')

ax = axes[0]
ax.hist(diffs_s[diffs_s <= 120], bins=60, color='steelblue', edgecolor='none', alpha=0.8)
ax.axvline(8, color='red', linestyle='--', lw=1.5, label='Expected poll: 8s')
ax.set_xlabel('Seconds'); ax.set_ylabel('Count'); ax.set_title('Zoomed: 0–120 s')
ax.legend(fontsize=9); ax.set_yscale('log')

ax = axes[1]
bins_f  = [0,1,2,5,8,15,30,60,120,300,900,1800,3600,86400,float(diffs_s.max())+1]
counts, edges = np.histogram(diffs_s, bins=bins_f)
labels = [f'{int(edges[i])}–{int(edges[i+1])}s' for i in range(len(counts))]
ax.barh(range(len(counts)), counts, color='steelblue', alpha=0.8)
ax.set_yticks(range(len(counts))); ax.set_yticklabels(labels, fontsize=8)
ax.set_xscale('log'); ax.set_xlabel('Count (log)'); ax.set_title('Full range')

plt.tight_layout()
fig.savefig(FIG_DIR / 'C1_time_diff_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

gap_inv = {
    '0 s (dup)':               int((diffs_s == 0).sum()),
    '1–15 s (normal poll)':    int(((diffs_s > 0) & (diffs_s <= 15)).sum()),
    '15 s–5 min (minor)':      int(((diffs_s > 15) & (diffs_s <= 300)).sum()),
    '5–30 min (short outage)': int(((diffs_s > 300) & (diffs_s <= 1800)).sum()),
    '30 min–1 h':              int(((diffs_s > 1800) & (diffs_s <= 3600)).sum()),
    '1–24 h (SARIMA range)':   int(((diffs_s > 3600) & (diffs_s <= 86400)).sum()),
    '>24 h (flag only)':       int((diffs_s > 86400).sum()),
}
print(f'\nOut of order rows (raw): {out_of_order_raw}')
print('Gap inventory:')
for k, v in gap_inv.items():
    print(f'  {k:<28} {v:>8,}')
print(f'  Largest gap: {diffs_s.max()/3600:.1f} h')

# %%
# ── Duplicate analysis ─────────────────────────────────────────────────────
dup_mask      = raw_sorted.duplicated('Unix', keep=False)
dup_rows      = raw_sorted[dup_mask].copy()
dup_rows['Aggregate'] = pd.to_numeric(dup_rows['Aggregate'], errors='coerce')
same_val_unix = int(dup_rows.groupby('Unix')['Aggregate'].nunique().eq(1).sum())
diff_val_unix = int(dup_rows.groupby('Unix')['Aggregate'].nunique().gt(1).sum())

print(f'Duplicate Unix timestamps: {dup_rows["Unix"].nunique():,} unique timestamps with >1 row')
print(f'  Same Aggregate value → keep first:       {same_val_unix:,}')
print(f'  Different Aggregate value → rolling-med: {diff_val_unix:,}')

# %%
# ── Spike / impossible value scan ─────────────────────────────────────────
findings = {}
for col in ['Aggregate'] + APPLIANCE_COLS:
    s   = pd.to_numeric(raw_sorted[col], errors='coerce')
    thr = IAM_SPIKE_THRESHOLD if col != 'Aggregate' else AGG_SPIKE_THRESHOLD
    findings[col] = {'neg': int((s < 0).sum()), 'spikes': int((s > thr).sum()),
                     'max': float(s.max()), 'thr': thr}

print(f'{"Channel":<14} {"Negatives":>10} {"Spikes":>8} {"Max W":>10} {"Threshold":>10}')
print('-' * 56)
for col, d in findings.items():
    flag = ' ←' if d['spikes'] > 0 else ''
    print(f'{col:<14} {d["neg"]:>10,} {d["spikes"]:>8,} {d["max"]:>10.0f} {d["thr"]:>10}{flag}')

# %%
# ── Flat-line detection on always-on channels ─────────────────────────────
tmp = raw_sorted.drop_duplicates('Unix').set_index('Time').copy()
for col in ALWAYS_ON_COLS:
    tmp[col] = pd.to_numeric(tmp[col], errors='coerce')
tmp_1min = tmp[ALWAYS_ON_COLS].resample('1min').mean()

for col in ALWAYS_ON_COLS:
    s = tmp_1min[col].dropna()
    runs = (s != s.shift()).cumsum()
    rl   = s.groupby(runs).count()
    long = rl[rl >= OUTAGE_FLATLINE_MIN]
    print(f'{col}: {len(long)} flat runs ≥ {OUTAGE_FLATLINE_MIN} min (longest: {int(rl.max())} min)')

# %% [markdown]
# ---
# ## PHASE 2 — PLAN
#
# | Rule | Decision | Justification |
# |------|----------|---------------|
# | R1 | Sort by Unix | Resampling requires ordered index |
# | R2a | Dup Unix, same value → keep first | Values identical |
# | R2b | Dup Unix, diff value → keep row closest to rolling median | Local context over position |
# | R3a | IAM > 4,000 W → NaN | UK socket fused at 13A × 230V ≈ 3 kW |
# | R3b | Aggregate > 15,000 W → NaN | Conservative UK supply limit |
# | R3c | Aggregate > 5× local MAD above rolling median → NaN | Catches context-dependent glitches |
# | R4 | Resample to 1-min mean | Mean preserves energy; NaN-aware |
# | R5 | Gaps ≤ 5 min → linear interpolation | Shorter than any appliance cycle |
# | R6 | Gaps 5 min–24 h → SARIMA (aggregate only) | Appliances never imputed |
# | R7 | Gaps > 24 h → NaN + outage=1 | Beyond one daily cycle = extrapolation |
# | R8 | Flat-line > 30 min on always-on → flatline_suspect | REFIT forward-fill artifact |
# | R9 | Σ(appliances) > Aggregate → issues_flag | Hierarchical check, validation only |

# %% [markdown]
# ---
# ## PHASE 3 — BUILD
# ### Stage A: Sort + Dedup (checkpoint)

# %%
CKPT_DEDUP = CKPT_DIR / 'ckpt_deduped.parquet'

if CKPT_DEDUP.exists() and not FORCE_RERUN:
    print('Loading dedup checkpoint ...')
    df = pd.read_parquet(CKPT_DEDUP)
    df['Time'] = pd.to_datetime(df['Time'])
    print(f'  {len(df):,} rows')
else:
    df = raw_sorted.copy()
    before = len(df)
    df['_agg'] = pd.to_numeric(df['Aggregate'], errors='coerce')

    # Rolling median for tie-breaking — vectorised, no Python loop
    WINDOW = 37
    df['_roll_med'] = df['_agg'].rolling(WINDOW, min_periods=3, center=True).median()

    # Distance of each row's Aggregate from its local rolling median.
    # NaN aggregate → infinite distance so non-NaN rows are always preferred.
    df['_dist'] = (df['_agg'] - df['_roll_med']).abs()
    df.loc[df['_agg'].isna(), '_dist'] = np.inf

    # For each Unix timestamp keep the single row closest to local median.
    # This handles both same-value (distance tie → first wins via stable sort)
    # and differing-value duplicates in one vectorised step.
    keep_idx = df.groupby('Unix')['_dist'].idxmin()
    df = df.loc[keep_idx].drop(columns=['_agg', '_roll_med', '_dist'])
    df = df.sort_values('Unix').reset_index(drop=True)

    removed = before - len(df)
    print(f'R2 Dedup: {removed:,} rows removed  |  {len(df):,} rows remain')
    df.to_parquet(CKPT_DEDUP)
    print(f'  Checkpoint saved → {CKPT_DEDUP.name}')

# %% [markdown]
# ### Stage B: Physics Filter + Resample to 1-min (checkpoint)

# %%
CKPT_1MIN = CKPT_DIR / 'ckpt_1min.parquet'

if CKPT_1MIN.exists() and not FORCE_RERUN:
    print('Loading 1-min checkpoint ...')
    df_1min = pd.read_parquet(CKPT_1MIN)
    print(f'  {len(df_1min):,} rows  |  NaN Aggregate: {df_1min["Aggregate"].isna().sum():,}')
else:
    d = df.copy()

    # R3a/b: Hard limits on IAM channels and aggregate
    spike_counts = {}
    for col in APPLIANCE_COLS:
        d[col] = pd.to_numeric(d[col], errors='coerce')
        mask = (d[col] > IAM_SPIKE_THRESHOLD) | (d[col] < 0)
        spike_counts[col] = int(mask.sum())
        d.loc[mask, col] = np.nan

    agg = pd.to_numeric(d['Aggregate'], errors='coerce')
    hard_mask = (agg > AGG_SPIKE_THRESHOLD) | (agg < 0)
    spike_counts['Aggregate_hard'] = int(hard_mask.sum())
    agg = agg.where(~hard_mask)

    # R3c: Context-aware spike removal — vectorised rolling MAD
    roll_med = agg.rolling(AGG_ROLL_W, min_periods=10, center=True).median()
    roll_mad = (agg - roll_med).abs().rolling(AGG_ROLL_W, min_periods=10, center=True).median()
    mad_mask = (agg > roll_med + AGG_MAD_MULT * roll_mad) & (roll_mad > 10)
    spike_counts['Aggregate_mad'] = int(mad_mask.sum())
    agg = agg.where(~mad_mask)
    d['Aggregate'] = agg

    print('R3 Physics filter:')
    for col, n in spike_counts.items():
        if n > 0:
            print(f'  {col}: {n:,} → NaN')

    # R4: Resample to 1-min mean
    d['Time'] = pd.to_datetime(d['Time'])
    d = d.set_index('Time')
    cols_r  = ['Aggregate'] + APPLIANCE_COLS
    df_1min = d[cols_r].resample('1min').mean()
    print(f'R4 Resample: → {len(df_1min):,} 1-min rows')
    print(f'   NaN bins: {df_1min["Aggregate"].isna().sum():,}')

    df_1min.to_parquet(CKPT_1MIN)
    print(f'  Checkpoint saved → {CKPT_1MIN.name}')

# %% [markdown]
# ### Stage C: Gap Classification

# %%
agg_series = df_1min['Aggregate'].copy()
gap_mask   = agg_series.isna()
gap_starts_idx = gap_mask & ~gap_mask.shift(1, fill_value=False)
gap_ends_idx   = gap_mask & ~gap_mask.shift(-1, fill_value=False)

gap_lengths = []
for s, e in zip(gap_starts_idx[gap_starts_idx].index, gap_ends_idx[gap_ends_idx].index):
    gap_lengths.append((s, e, int((e - s).total_seconds() / 60) + 1))

SARIMA_MIN_GAP = 30  # gaps shorter than this get linear interp; SARIMA only for ≥30 min

short_gaps    = [(s,e,l) for s,e,l in gap_lengths if l <= SHORT_GAP_MIN]
lin_med_gaps  = [(s,e,l) for s,e,l in gap_lengths if SHORT_GAP_MIN < l < SARIMA_MIN_GAP]
sarima_gaps   = [(s,e,l) for s,e,l in gap_lengths if SARIMA_MIN_GAP <= l <= LONG_GAP_H*60]
long_gaps     = [(s,e,l) for s,e,l in gap_lengths if l > LONG_GAP_H*60]
med_gaps      = lin_med_gaps + sarima_gaps  # kept for backward compat in plot cell

print(f'Gap classification at 1-min resolution:')
print(f'  Short  (≤ {SHORT_GAP_MIN} min)         → linear:       {len(short_gaps):>5,}')
print(f'  Linear (5–{SARIMA_MIN_GAP} min)         → linear:       {len(lin_med_gaps):>5,}')
print(f'  SARIMA (≥{SARIMA_MIN_GAP} min – {LONG_GAP_H}h)  → SARIMA fit:   {len(sarima_gaps):>5,}')
print(f'  Long   (> {LONG_GAP_H}h)            → outage flag:  {len(long_gaps):>5,}')
if med_gaps:
    print('  Medium gaps:')
    for s, e, l in med_gaps:
        print(f'    {s}  →  {e}  ({l} min)')

# %% [markdown]
# ### Stage D: SARIMA Two-Pass Fit (checkpoint — fits once, reuses orders)

# %%
CKPT_SARIMA = CKPT_DIR / 'ckpt_sarima_orders.pkl'
agg_1h      = df_1min['Aggregate'].resample('1h').mean()

if CKPT_SARIMA.exists() and not FORCE_RERUN:
    print('Loading SARIMA orders from checkpoint ...')
    with open(CKPT_SARIMA, 'rb') as f:
        sarima_order, sarima_seas = pickle.load(f)
    print(f'  Orders: {sarima_order} × {sarima_seas}')
else:
    train_end    = agg_1h.index[0] + pd.Timedelta(days=60)
    train_series = agg_1h.loc[:train_end].dropna()
    print(f'Training window: {train_series.index[0].date()} → {train_series.index[-1].date()} ({len(train_series)} obs)')

    print('Pass 1: auto_arima ...')
    model_p1 = auto_arima(
        train_series, seasonal=True, m=SARIMA_SEASONAL,
        start_p=1, start_q=1, max_p=2, max_q=2,
        start_P=0, start_Q=0, max_P=1, max_Q=1,
        d=None, D=None, information_criterion='aic',
        stepwise=True, suppress_warnings=True, error_action='ignore',
        max_iter=30, n_jobs=1
    )
    print(f'  Pass 1: {model_p1.order} × {model_p1.seasonal_order}')

    resid_p1   = train_series - pd.Series(model_p1.predict_in_sample(), index=train_series.index)
    clean_mask = np.abs(resid_p1) <= ANOMALY_SIGMA * resid_p1.std()
    train_clean = train_series[clean_mask]
    print(f'  Statistical outliers removed: {(~clean_mask).sum()}')

    print('Pass 2: refit on cleaned data ...')
    model_p2 = auto_arima(
        train_clean, seasonal=True, m=SARIMA_SEASONAL,
        start_p=1, start_q=1, max_p=2, max_q=2,
        start_P=0, start_Q=0, max_P=1, max_Q=1,
        d=None, D=None, information_criterion='aic',
        stepwise=True, suppress_warnings=True, error_action='ignore',
        max_iter=30, n_jobs=1
    )
    sarima_order = model_p2.order
    sarima_seas  = model_p2.seasonal_order
    print(f'  Pass 2: {sarima_order} × {sarima_seas}')

    with open(CKPT_SARIMA, 'wb') as f:
        pickle.dump((sarima_order, sarima_seas), f)
    print(f'  Checkpoint saved → {CKPT_SARIMA.name}')

# %% [markdown]
# ### Stage E: Impute + Flag + Final Output (checkpoint)

# %%
CKPT_FINAL = CKPT_DIR / 'ckpt_final.parquet'

if CKPT_FINAL.exists() and not FORCE_RERUN:
    print('Loading final checkpoint ...')
    df_1min = pd.read_parquet(CKPT_FINAL)
    print(f'  {len(df_1min):,} rows  |  cols: {df_1min.columns.tolist()}')
else:
    df_1min['impute_source'] = 'known'
    agg_imputed = df_1min['Aggregate'].copy()
    n_sarima_filled = 0

    # R5b: Linear interpolation for 5–30 min gaps on aggregate (indistinguishable from SARIMA at this scale)
    for start, end, length_min in lin_med_gaps:
        gap_range = pd.date_range(start=start, end=end, freq='1min').intersection(df_1min.index)
        agg_imputed.loc[gap_range] = agg_imputed.loc[gap_range].interpolate(method='linear')
        df_1min.loc[gap_range, 'impute_source'] = 'linear_med'
    print(f'R5b Linear (5–30 min gaps): {len(lin_med_gaps)} gaps filled')

    # R6: SARIMA imputation for gaps ≥ 30 min only (daily seasonality matters here)
    for start, end, length_min in sarima_gaps:
        known_before = agg_1h.loc[:start].dropna()
        if len(known_before) < 48:
            continue
        try:
            sm_fit = SARIMAX(
                known_before.values, order=sarima_order, seasonal_order=sarima_seas,
                enforce_stationarity=False, enforce_invertibility=False
            ).fit(disp=False, maxiter=50)
            n_hours   = int(np.ceil(length_min / 60)) + 2
            fc        = sm_fit.get_forecast(steps=n_hours).predicted_mean.clip(0)
            fc_idx    = pd.date_range(start=known_before.index[-1] + pd.Timedelta(hours=1),
                                       periods=n_hours, freq='1h')
            # Anchor: prepend last known hourly value so gaps that start within
            # the first forecast hour (sub-hourly start) get covered by interpolation.
            anchor_t   = known_before.index[-1]
            anchor_v   = float(known_before.iloc[-1])
            combined   = pd.Series(
                np.concatenate([[anchor_v], np.asarray(fc)]),
                index=pd.DatetimeIndex([anchor_t]).append(fc_idx)
            )
            fc_1min   = combined.resample('1min').interpolate('linear')
            gap_range = pd.date_range(start=start, end=end, freq='1min')
            fill_idx  = gap_range.intersection(fc_1min.index)
            agg_imputed.loc[fill_idx] = fc_1min.loc[fill_idx].values
            df_1min.loc[fill_idx, 'impute_source'] = 'sarima'
            n_sarima_filled += len(fill_idx)
            print(f'  SARIMA filled {len(fill_idx)} min: {start} → {end}')
        except Exception as e:
            print(f'  SARIMA failed for {start}: {e}')

    df_1min['Aggregate'] = agg_imputed
    print(f'R6 SARIMA total: {n_sarima_filled:,} min filled across {len(sarima_gaps)} gaps')

    # R5: Linear interpolation for short gaps (all channels)
    cols_r   = ['Aggregate'] + APPLIANCE_COLS
    n_before = int(df_1min[cols_r].isna().sum().sum())
    df_1min[cols_r] = df_1min[cols_r].interpolate(
        method='linear', limit=SHORT_GAP_MIN, limit_direction='forward')
    print(f'R5 Linear: {n_before - df_1min[cols_r].isna().sum().sum():,} NaN cells filled')

    # R7: Outage flag for long gaps and remaining NaN
    df_1min['outage'] = 0
    for start, end, _ in long_gaps:
        idx = pd.date_range(start=start, end=end, freq='1min').intersection(df_1min.index)
        df_1min.loc[idx, 'outage'] = 1
    df_1min.loc[df_1min['Aggregate'].isna(), 'outage'] = 1
    print(f'R7 Outage: {df_1min["outage"].sum():,} min ({df_1min["outage"].sum()/60:.1f}h)')

    # R8: Flat-line on always-on channels (only flag non-zero runs — exclude genuine zeros)
    df_1min['flatline_suspect'] = 0
    for col in ALWAYS_ON_COLS:
        s    = df_1min[col].ffill()
        runs = (s != s.shift()).cumsum()
        flat = (s.groupby(runs).transform('count') >= OUTAGE_FLATLINE_MIN) & (s > 0)
        df_1min.loc[flat, 'flatline_suspect'] = 1

    # R9: Hierarchical check
    app_sum = df_1min[APPLIANCE_COLS].sum(axis=1, skipna=True)
    df_1min['issues_flag'] = (app_sum > df_1min['Aggregate']).astype(int)

    df_1min.to_parquet(CKPT_FINAL)
    print(f'  Checkpoint saved → {CKPT_FINAL.name}')

# %%
# ── Copy to final output ──────────────────────────────────────────────────
import shutil
out_path = OUT_DIR / 'house1_clean_1min.parquet'
shutil.copy(CKPT_FINAL, out_path)

print('\n' + '='*65)
print('BUILD COMPLETE — house1_clean_1min.parquet')
print('='*65)
print(f'  Rows:             {len(df_1min):,}')
print(f'  Known readings:   {(df_1min["impute_source"]=="known").sum():,}')
print(f'  SARIMA imputed:   {(df_1min["impute_source"]=="sarima").sum():,}')
print(f'  Outage minutes:   {df_1min["outage"].sum():,}')
print(f'  Flatline suspect: {df_1min["flatline_suspect"].sum():,}')
print(f'  Issues flag:      {df_1min["issues_flag"].sum():,}')
print('='*65)

# %% [markdown]
# ---
# ## PHASE 4 — VALIDATE

# %%
# ── Build raw_before_1min for BEFORE side ─────────────────────────────────
raw_before = raw_sorted.drop_duplicates('Unix').set_index('Time').copy()
for col in ['Aggregate', 'Appliance5', 'Appliance1']:
    raw_before[col] = pd.to_numeric(raw_before[col], errors='coerce')
raw_before_1min = raw_before[['Aggregate','Appliance5','Appliance1']].resample('1min').mean()
print(f'raw_before_1min: {len(raw_before_1min):,} rows')

# %%
# ── Plot 1: Before vs After — Nov 1–8 2013 ───────────────────────────────
week_start, week_end = '2013-11-01', '2013-11-09'
before_week = raw_before_1min.loc[week_start:week_end]
after_week  = df_1min.loc[week_start:week_end]

channels = [
    ('Aggregate',  'Whole-house Aggregate',  '#2c7bb6'),
    ('Appliance5', 'Washing Machine (App5)', '#d7191c'),
    ('Appliance1', 'Fridge (App1)',          '#1a9641'),
]

fig, axes = plt.subplots(3, 2, figsize=(16, 10), sharex='col')
fig.suptitle('House 1 — Before vs After Cleaning  |  Nov 1–8 2013',
             fontsize=13, fontweight='bold')

for ri, (col, label, color) in enumerate(channels):
    ax = axes[ri, 0]
    ax.plot(before_week.index, before_week[col], color=color, lw=0.6, alpha=0.8)
    ax.set_ylabel(f'{label}\n(W)', fontsize=9)
    if ri == 0: ax.set_title('BEFORE (raw, 1-min)', fontweight='bold')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=8)

    ax = axes[ri, 1]
    ymax = float(after_week[col].quantile(0.999)) * 1.2 if after_week[col].notna().any() else 200
    outage_mask = after_week['outage'] == 1
    sarima_mask = after_week['impute_source'] == 'sarima'
    ax.fill_between(after_week.index, 0, ymax, where=outage_mask,
                    alpha=0.2, color='gray', label='Outage')
    ax.fill_between(after_week.index, 0, ymax, where=sarima_mask,
                    alpha=0.3, color='orange', label='SARIMA imputed')
    ax.plot(after_week.index, after_week[col], color=color, lw=0.6, alpha=0.8)
    ax.set_ylim(0, ymax)
    if ri == 0:
        ax.set_title('AFTER (cleaned, 1-min)', fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=8)

plt.tight_layout()
fig.savefig(FIG_DIR / 'C1_before_after_week.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: C1_before_after_week.png')

# %%
# ── Plot 2: Zoomed outage + SARIMA forecast overlay ───────────────────────
zoom_start, zoom_end = '2013-11-07 06:00', '2013-11-08 18:00'
before_zoom = raw_before_1min.loc[zoom_start:zoom_end, 'Aggregate']
after_zoom  = df_1min.loc[zoom_start:zoom_end]

# Find actual first NaN in aggregate within zoom window
agg_1h = df_1min['Aggregate'].resample('1h').mean()
gap_in_zoom = [s for s, e, l in (med_gaps + long_gaps)
               if pd.Timestamp(zoom_start) <= s <= pd.Timestamp(zoom_end)]
fc_origin = gap_in_zoom[0] if gap_in_zoom else pd.Timestamp('2013-11-07 11:00')

has_fc = False
try:
    known_hz = agg_1h.loc[:fc_origin].dropna()
    print(f'Fitting SARIMA for plot: {len(known_hz)} hourly obs up to {known_hz.index[-1]}')
    if len(known_hz) >= 48:
        sm_fit = SARIMAX(
            known_hz.values, order=sarima_order, seasonal_order=sarima_seas,
            enforce_stationarity=False, enforce_invertibility=False
        ).fit(disp=False, maxiter=50)
        n_h    = int((pd.Timestamp(zoom_end) - known_hz.index[-1]).total_seconds() / 3600) + 1
        fc_res = sm_fit.get_forecast(steps=n_h)
        fc_idx = pd.date_range(start=known_hz.index[-1] + pd.Timedelta(hours=1),
                                periods=n_h, freq='1h')
        fc_zoom = pd.Series(fc_res.predicted_mean.clip(0), index=fc_idx)
        fc_ci   = fc_res.conf_int(alpha=0.1)
        fc_lo   = pd.Series(fc_ci.iloc[:,0].clip(0).values, index=fc_idx)
        fc_hi   = pd.Series(fc_ci.iloc[:,1].clip(0).values, index=fc_idx)
        has_fc  = True
        print(f'Forecast OK: {n_h} hours, W range {fc_zoom.min():.0f}–{fc_zoom.max():.0f}')
except Exception as e:
    print(f'SARIMA forecast skipped: {e}')

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
fig.suptitle('House 1 Aggregate — Zoomed: Nov 7–8 Outage', fontsize=12, fontweight='bold')

ax1.plot(before_zoom.index, before_zoom.values, color='steelblue', lw=0.8, label='Raw (1-min)')
ax1.set_ylabel('Power (W)'); ax1.set_title('BEFORE'); ax1.legend(fontsize=9)

ymax2 = 4000
ax2.fill_between(after_zoom.index, 0, ymax2,
                 where=(after_zoom['outage']==1), alpha=0.15, color='gray', label='Outage window')
ax2.fill_between(after_zoom.index, 0, ymax2,
                 where=(after_zoom['impute_source']=='sarima'), alpha=0.2, color='orange',
                 label='SARIMA imputed')
ax2.plot(after_zoom.index, after_zoom['Aggregate'], color='steelblue', lw=0.8,
         label='Cleaned aggregate')
if has_fc:
    ax2.plot(fc_zoom.index, fc_zoom.values, color='darkorange', lw=2, linestyle='--',
             label='SARIMA forecast')
    ax2.fill_between(fc_zoom.index, fc_lo.values, fc_hi.values,
                     alpha=0.2, color='orange', label='90% CI')
ax2.set_ylabel('Power (W)'); ax2.set_ylim(0, ymax2)
ax2.set_title('AFTER: cleaned + SARIMA forecast overlay')
ax2.legend(fontsize=9, loc='upper right')
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d %H:%M'))
ax2.xaxis.set_major_locator(mdates.HourLocator(interval=6))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=8)

plt.tight_layout()
fig.savefig(FIG_DIR / 'C1_zoom_outage.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: C1_zoom_outage.png')

# %%
# ── Plot 3: Anomaly detection overlay ────────────────────────────────────
diag_start, diag_end = '2013-10-09', '2013-10-23'
raw_diag = raw_before_1min.loc[diag_start:diag_end, 'Aggregate']

ROLL_W  = 30
r_med   = raw_diag.rolling(ROLL_W, center=True).median()
r_mad   = (raw_diag - r_med).abs().rolling(ROLL_W, center=True).median()
r_upper = r_med + ANOMALY_SIGMA * r_mad
r_lower = (r_med - ANOMALY_SIGMA * r_mad).clip(lower=0)
anom    = raw_diag[(raw_diag > r_upper) | (raw_diag < r_lower)]

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(raw_diag.index, raw_diag.values, color='steelblue', lw=0.5, alpha=0.7, label='Raw (1-min)')
ax.plot(r_med.index, r_med.values, color='navy', lw=1.2, label='Rolling median')
ax.fill_between(r_med.index, r_lower, r_upper, alpha=0.2, color='navy',
                label=f'±{ANOMALY_SIGMA}×MAD envelope')
ax.scatter(anom.index, anom.values, color='red', s=20, zorder=5,
           label=f'Anomalies flagged: {len(anom)}')
ax.set_ylabel('Power (W)')
ax.set_title('House 1 — Rolling Median + MAD Anomaly Detection  |  Oct 9–23 2013')
ax.legend(fontsize=9)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=8)
plt.tight_layout()
fig.savefig(FIG_DIR / 'C1_anomaly_detection.png', dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved: C1_anomaly_detection.png  |  {len(anom)} anomalies flagged')
