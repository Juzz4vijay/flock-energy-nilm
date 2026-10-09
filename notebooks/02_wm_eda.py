# %% [markdown]
# # Section 2 — Washing Machine EDA Across Households
#
# **Data source**: Official CLEAN REFIT dataset (University of Strathclyde)
# **Coverage**: 19 households with confirmed washing machine channels (House 12 excluded — no WM)
# **Resolution**: 1-minute bins (resampled from 8-second raw)
#
# **Cycle detection thresholds — grounded in 2013–2015 UK market data**:
# - ON threshold  : ≥ 100 W sustained (below = standby/off per EU ErP 2013)
# - Min ON streak : ≥ 3 consecutive 1-min bins (avoids transient noise)
# - Min cycle     : ≥ 15 min  (shortest verified quick-wash sold in UK 2013–2015: Bosch SpeedPerfect)
# - Max cycle     : ≤ 180 min (no domestic programme exceeds 3 h)
# - Heating phase : any 1-min bin ≥ 1 800 W within a cycle → hot wash (40–60 °C)

# %%
import warnings; warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_CLEAN  = Path('data/clean')
CKPT_DIR    = Path('data/processed/checkpoints')
FIG_DIR     = Path('figures')
FIG_DIR.mkdir(exist_ok=True)

FORCE_RERUN = False   # set True to redo all stages from scratch

# ── WM channel map (19 houses; House 12 has no WM) ───────────────────────────
WM_MAP = {
    1:'Appliance5', 2:'Appliance2', 3:'Appliance6', 4:'Appliance4', 5:'Appliance3',
    6:'Appliance2', 7:'Appliance5', 8:'Appliance4', 9:'Appliance3',10:'Appliance5',
   11:'Appliance3',13:'Appliance3',15:'Appliance3',16:'Appliance5',17:'Appliance4',
   18:'Appliance5',19:'Appliance2',20:'Appliance4',21:'Appliance3',
}

# ── Cycle detection parameters ────────────────────────────────────────────────
ON_THRESH_W   = 100    # W — minimum power to consider machine ON
MIN_ON_STREAK = 3      # consecutive 1-min bins to confirm ON state
MIN_CYCLE_MIN = 15     # minutes — shortest valid cycle (Bosch quick-wash 2013–15)
MAX_CYCLE_MIN = 180    # minutes — longest possible domestic programme
HEAT_THRESH_W = 1800   # W — heating element active → hot wash

print('Setup complete.')
print(f'Houses with WM: {sorted(WM_MAP.keys())}')
print(f'Cycle detection: ON≥{ON_THRESH_W}W × {MIN_ON_STREAK}min streak | '
      f'cycle {MIN_CYCLE_MIN}–{MAX_CYCLE_MIN} min | heat≥{HEAT_THRESH_W}W')

# %% [markdown]
# ---
# ## STAGE 0 — Load all WM channels, resample to 1-min

# %%
CKPT_WM = CKPT_DIR / 'ckpt_wm_1min.parquet'

if CKPT_WM.exists() and not FORCE_RERUN:
    print(f'Loading checkpoint: {CKPT_WM}')
    wm_all = pd.read_parquet(CKPT_WM)
    print(f'  Loaded: {wm_all.shape}  |  houses: {wm_all["house"].nunique()}')
else:
    frames = []
    for house, col in sorted(WM_MAP.items()):
        fname = DATA_CLEAN / f'CLEAN_House{house}.csv'
        print(f'  Loading House {house:2d} ({col}) ...', end=' ')
        df = pd.read_csv(fname, usecols=['Time', 'Aggregate', col, 'Issues'],
                         parse_dates=['Time'])
        df = df.rename(columns={col: 'WM'})
        df['WM']       = pd.to_numeric(df['WM'],       errors='coerce')
        df['Aggregate']= pd.to_numeric(df['Aggregate'],errors='coerce')
        df = df.set_index('Time').sort_index()

        # Clean REFIT uses 0 for missing — keep 0 (machine off) but mark issues
        # Resample to 1-min mean (handles multiple 8-sec readings per minute)
        wm_1min = df[['WM','Aggregate']].resample('1min').mean()
        wm_1min['house'] = house
        wm_1min['issues'] = df['Issues'].resample('1min').max()

        frames.append(wm_1min)
        print(f'{len(wm_1min):,} rows  |  WM max={df["WM"].max():.0f}W')

    wm_all = pd.concat(frames)
    wm_all.to_parquet(CKPT_WM)
    print(f'\nSaved: {CKPT_WM}  |  shape: {wm_all.shape}')

# ── Coverage summary ──────────────────────────────────────────────────────────
print('\n=== Coverage Summary ===')
print(f'{"House":>6}  {"WM col":>12}  {"Start":>12}  {"End":>12}  {"Days":>6}  {"WM col"}')
for h, grp in wm_all.groupby('house'):
    days = (grp.index.max() - grp.index.min()).days
    col  = WM_MAP[h]
    print(f'  {h:>4}  {col:>12}  {grp.index.min().date()}  '
          f'{grp.index.max().date()}  {days:>6}d')

# %% [markdown]
# ---
# ## STAGE 1 — Cycle detection (dynamic per-house thresholds + hysteresis)
#
# **Why dynamic thresholds?**
# A fixed 100 W ON threshold fragments hot-wash cycles at the drain phase (~50–80 W).
# Each machine has its own minimum active power — we derive it from its power distribution.
#
# **Threshold derivation:**
# - ON threshold  = 20th percentile of non-zero readings (min 60 W)  → "clearly active"
# - OFF threshold = 5th percentile × 0.6 (min 25 W)                  → "below drain pump"
#
# **Hysteresis state machine:**
# - ENTER cycle : power ≥ on_thresh for ≥ 3 consecutive minutes
# - STAY in cycle: power ≥ off_thresh (bridges the 50–80 W drain phase)
# - EXIT cycle  : power < off_thresh for ≥ 5 consecutive minutes

# %%
CKPT_CYCLES = CKPT_DIR / 'ckpt_wm_cycles_v2.parquet'

def find_house_thresholds(wm_series: pd.Series) -> tuple:
    """
    Derive per-house ON/OFF thresholds from the machine's power histogram.
    Returns (on_thresh_w, off_thresh_w).
    """
    active = wm_series[wm_series > 5].dropna()
    if len(active) < 200:
        return 100, 50   # fallback for sparse data
    p05 = float(np.percentile(active, 5))
    p20 = float(np.percentile(active, 20))
    on_thresh  = int(max(round(p20 / 10) * 10, 60))   # round to 10 W, floor 60 W
    off_thresh = int(max(round(p05 * 0.6 / 5) * 5, 25))  # round to 5 W, floor 25 W
    return on_thresh, off_thresh


def detect_cycles_v2(wm_series: pd.Series, house: int,
                     on_thresh: int, off_thresh: int) -> pd.DataFrame:
    """
    Hysteresis-based cycle detection.
    ENTER when power >= on_thresh for MIN_ON_STREAK consecutive minutes.
    EXIT  when power <  off_thresh for OFF_PATIENCE consecutive minutes.
    """
    OFF_PATIENCE = 5   # minutes below off_thresh required to exit cycle
    s   = wm_series.fillna(0).copy()
    idx = s.index
    n   = len(idx)

    # Confirmed entry mask: on_thresh streak of MIN_ON_STREAK
    raw_on   = (s >= on_thresh).astype(int)
    streaked = raw_on.rolling(MIN_ON_STREAK, min_periods=MIN_ON_STREAK).min()
    conf     = (streaked == 1)
    for lag in range(1, MIN_ON_STREAK):
        conf = conf | conf.shift(-lag).fillna(False)
    confirmed_entry = conf & (raw_on == 1)

    above_off = (s >= off_thresh).values  # numpy array for speed

    cycles = []
    i = 0
    while i < n:
        if not confirmed_entry.iloc[i]:
            i += 1
            continue

        # Cycle start confirmed — extend via hysteresis
        start_pos  = i
        low_streak = 0
        j = i + 1
        while j < n:
            if not above_off[j]:
                low_streak += 1
                if low_streak >= OFF_PATIENCE:
                    break
            else:
                low_streak = 0
            j += 1

        # End position: last active minute before the exit low-streak
        if low_streak >= OFF_PATIENCE:
            end_pos = j - OFF_PATIENCE   # last minute before the low streak
        else:
            end_pos = j - 1              # series ended while still active

        end_pos = max(end_pos, start_pos)
        dur     = end_pos - start_pos + 1

        if MIN_CYCLE_MIN <= dur <= MAX_CYCLE_MIN:
            seg             = s.iloc[start_pos:end_pos + 1]
            energy_wh       = seg.sum() / 60.0
            peak_w          = seg.max()
            heat_mins       = int((seg >= HEAT_THRESH_W).sum())
            heat_energy_wh  = seg[seg >= HEAT_THRESH_W].sum() / 60.0
            hot_wash        = heat_mins >= 3

            cycles.append({
                'house'         : house,
                'on_thresh'     : on_thresh,
                'off_thresh'    : off_thresh,
                'start'         : idx[start_pos],
                'end'           : idx[end_pos],
                'duration_min'  : dur,
                'energy_wh'     : round(energy_wh, 3),
                'peak_w'        : peak_w,
                'heat_mins'     : heat_mins,
                'heat_energy_wh': round(heat_energy_wh, 3),
                'hot_wash'      : hot_wash,
                'hour_start'    : idx[start_pos].hour,
                'dow_start'     : idx[start_pos].dayofweek,
            })

        i = end_pos + 1   # advance past this cycle — no overlaps

    return pd.DataFrame(cycles)


if CKPT_CYCLES.exists() and not FORCE_RERUN:
    print(f'Loading checkpoint: {CKPT_CYCLES}')
    cycles_all = pd.read_parquet(CKPT_CYCLES)
else:
    print(f'{"House":>6}  {"ON thresh":>10}  {"OFF thresh":>11}  {"Cycles":>7}')
    print('-' * 42)
    cycle_frames = []
    for h, grp in wm_all.groupby('house'):
        wm_s = grp['WM']
        on_t, off_t = find_house_thresholds(wm_s)
        cdf  = detect_cycles_v2(wm_s, h, on_t, off_t)
        print(f'  {h:>4}  {on_t:>8}W  {off_t:>9}W  {len(cdf):>7}')
        cycle_frames.append(cdf)
    cycles_all = pd.concat(cycle_frames, ignore_index=True)
    cycles_all.to_parquet(CKPT_CYCLES)
    print(f'\nSaved: {CKPT_CYCLES}  |  total cycles: {len(cycles_all):,}')

print(f'\n=== Cycle Summary ===')
print(cycles_all.groupby('house')[['duration_min','energy_wh','hot_wash']].agg(
    cycles=('duration_min','count'),
    dur_median=('duration_min','median'),
    energy_median=('energy_wh','median'),
    hot_pct=('hot_wash', lambda x: f'{100*x.mean():.0f}%')
).to_string())

# %% [markdown]
# ---
# ## STAGE 2 — Plots

# %%
# ── Helper: pick a representative WM day for a house ─────────────────────────
def pick_wm_day(house: int, n_cycles: int = 2) -> pd.Timestamp:
    """Return date with exactly n_cycles WM uses — good for illustrative plot."""
    hc = cycles_all[cycles_all['house'] == house].copy()
    hc['date'] = hc['start'].dt.date
    by_day = hc.groupby('date').size()
    matches = by_day[by_day == n_cycles]
    if len(matches) == 0:
        matches = by_day[by_day >= 1]
    return pd.Timestamp(matches.index[len(matches)//2])  # middle of dataset

# %%
# ── Plot S2-1: 24-hour aggregate + WM trace (House 1, typical 2-wash day) ────
print('Generating Plot S2-1: 24h aggregate + WM trace ...')

h1_day  = pick_wm_day(1, n_cycles=2)
day_s   = h1_day
day_e   = h1_day + pd.Timedelta('1D') - pd.Timedelta('1min')

h1_data = wm_all[wm_all['house'] == 1].loc[day_s:day_e]
h1_cyc  = cycles_all[(cycles_all['house']==1) &
                     (cycles_all['start'].dt.date == h1_day.date())]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                                gridspec_kw={'hspace': 0.06, 'height_ratios': [2, 1]})
fig.suptitle(f'House 1 — Typical WM Day  |  {h1_day.strftime("%d %b %Y")}',
             fontsize=13, fontweight='bold')

ax1.plot(h1_data.index, h1_data['Aggregate'], color='#2c7bb6', lw=0.9,
         label='Whole-house aggregate')
for _, cyc in h1_cyc.iterrows():
    ax1.axvspan(cyc['start'], cyc['end'], alpha=0.18, color='darkorange')
ax1.set_ylabel('Power (W)')
ax1.set_title('Whole-house aggregate  (orange bands = WM cycles)', fontsize=9,
              color='dimgray', loc='left')
ax1.set_ylim(bottom=0)
ax1.legend(fontsize=8, loc='upper right')

ax2.fill_between(h1_data.index, 0, h1_data['WM'], color='#d7191c',
                 alpha=0.75, label='Washing machine (App5)')
for _, cyc in h1_cyc.iterrows():
    label = f"{'Hot' if cyc['hot_wash'] else 'Cold'} wash\n{cyc['duration_min']} min\n{cyc['energy_wh']:.0f} Wh"
    mid   = cyc['start'] + (cyc['end'] - cyc['start']) / 2
    ax2.annotate(label, xy=(mid, cyc['peak_w'] * 0.5),
                 ha='center', va='center', fontsize=7.5,
                 bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))
ax2.set_ylabel('WM Power (W)')
ax2.set_ylim(bottom=0)
ax2.legend(fontsize=8, loc='upper right')
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax2.xaxis.set_major_locator(mdates.HourLocator(interval=2))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0, ha='center', fontsize=8)
ax2.set_xlabel('Time of day')

plt.savefig(FIG_DIR / 'S2_1_daily_trace.png', dpi=150, bbox_inches='tight')
print(f'  Saved S2_1_daily_trace.png  |  {len(h1_cyc)} cycles shown')

# %%
# ── Plot S2-2: WM power over detected cycles — aligned from cycle start ───────
print('Generating Plot S2-2: Cycle power profiles (aligned) ...')

# Show up to 80 cycles from House 1 overlaid + median envelope
h1_cyc_sorted = cycles_all[cycles_all['house'] == 1].copy()
h1_wm         = wm_all[wm_all['house'] == 1]['WM']

# Sample up to 80 cycles
sample_cyc = h1_cyc_sorted.sample(min(80, len(h1_cyc_sorted)), random_state=42)

fig, ax = plt.subplots(figsize=(13, 6))

profiles = []
for _, cyc in sample_cyc.iterrows():
    seg = h1_wm.loc[cyc['start']:cyc['end']].values
    t   = np.arange(len(seg))
    color = '#d94e4e' if cyc['hot_wash'] else '#4e9fd9'
    ax.plot(t, seg, color=color, lw=0.5, alpha=0.25)
    # Pad/truncate to MAX_CYCLE_MIN for median
    padded = np.full(MAX_CYCLE_MIN, np.nan)
    padded[:len(seg)] = seg
    profiles.append(padded)

# Median envelope
profiles_arr = np.array(profiles)
med_profile  = np.nanmedian(profiles_arr, axis=0)
t_med        = np.arange(len(med_profile))
valid        = ~np.isnan(med_profile)
ax.plot(t_med[valid], med_profile[valid], color='black', lw=2.0,
        label='Median cycle', zorder=5)

# Legend patches
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(color='#d94e4e', alpha=0.6, label='Hot wash (≥1,800W heating phase)'),
    Patch(color='#4e9fd9', alpha=0.6, label='Cold/warm wash'),
    plt.Line2D([0],[0], color='black', lw=2, label='Median cycle'),
], fontsize=9, loc='upper right')

ax.set_xlabel('Minutes from cycle start')
ax.set_ylabel('WM Power (W)')
ax.set_title('House 1 — WM Power Profiles Across All Detected Cycles\n'
             'Red = hot wash (heating phase ≥1,800W) | Blue = cold/warm wash',
             fontsize=11, fontweight='bold')
ax.set_ylim(bottom=0)
ax.set_xlim(0, MAX_CYCLE_MIN)
ax.axhline(HEAT_THRESH_W, color='red', lw=0.8, ls='--', alpha=0.5)
ax.text(1, HEAT_THRESH_W+30, f'{HEAT_THRESH_W}W heating threshold', fontsize=8, color='red')

plt.savefig(FIG_DIR / 'S2_2_cycle_profiles.png', dpi=150, bbox_inches='tight')
print(f'  Saved S2_2_cycle_profiles.png  |  {len(sample_cyc)} cycles overlaid')

# %%
# ── Plot S2-3: WM usage heatmap — hour of day × day of week ──────────────────
print('Generating Plot S2-3: Usage heatmap (hour × day-of-week) ...')

# Aggregate across ALL houses
dow_labels = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
heatmap    = np.zeros((24, 7))

for _, cyc in cycles_all.iterrows():
    heatmap[cyc['hour_start'], cyc['dow_start']] += 1

# Normalize to % of total cycles
heatmap_pct = 100.0 * heatmap / heatmap.sum()

fig, ax = plt.subplots(figsize=(10, 7))
im = ax.imshow(heatmap_pct, aspect='auto', cmap='YlOrRd', origin='upper',
               vmin=0, vmax=heatmap_pct.max())
cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label('% of all cycles', fontsize=9)

ax.set_xticks(range(7)); ax.set_xticklabels(dow_labels)
ax.set_yticks(range(0, 24, 2))
ax.set_yticklabels([f'{h:02d}:00' for h in range(0, 24, 2)])
ax.set_xlabel('Day of week')
ax.set_ylabel('Hour of day (local UK time)')
ax.set_title('Washing Machine Usage — All 19 Houses\nHeat = % of detected cycles starting in that hour/day slot',
             fontsize=11, fontweight='bold')

# annotate peak cell
peak_h, peak_d = np.unravel_index(heatmap_pct.argmax(), heatmap_pct.shape)
ax.add_patch(plt.Rectangle((peak_d-0.5, peak_h-0.5), 1, 1,
             fill=False, edgecolor='black', lw=2))
ax.text(peak_d, peak_h, f'{heatmap_pct[peak_h,peak_d]:.1f}%\npeak',
        ha='center', va='center', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig(FIG_DIR / 'S2_3_usage_heatmap.png', dpi=150, bbox_inches='tight')
print(f'  Saved S2_3_usage_heatmap.png')

# %%
# ── Plot S2-4: Cycle energy & duration comparison across houses ───────────────
print('Generating Plot S2-4: Cross-house cycle energy & duration ...')

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('Washing Machine Cycles — Cross-House Comparison\n(Official CLEAN REFIT data, 19 households)',
             fontsize=12, fontweight='bold')

houses_sorted = sorted(cycles_all['house'].unique())
labels = [f'H{h}' for h in houses_sorted]

# Energy boxplot
ax = axes[0]
data_e = [cycles_all[cycles_all['house']==h]['energy_wh'].values for h in houses_sorted]
bp = ax.boxplot(data_e, tick_labels=labels, patch_artist=True,
                medianprops=dict(color='black', lw=1.5),
                flierprops=dict(marker='.', ms=3, alpha=0.3),
                whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8))
for patch, h in zip(bp['boxes'], houses_sorted):
    hot_pct = cycles_all[cycles_all['house']==h]['hot_wash'].mean()
    patch.set_facecolor(plt.cm.RdYlGn_r(hot_pct * 0.8 + 0.1))
    patch.set_alpha(0.75)
ax.set_xlabel('House')
ax.set_ylabel('Cycle Energy (Wh)')
ax.set_title('Cycle Energy Distribution per House\n(box colour = hot-wash fraction: red=high, green=low)',
             fontsize=9)
ax.grid(axis='y', alpha=0.4)
ax.axhline(500, color='gray', lw=0.7, ls='--', alpha=0.5)
ax.text(0.5, 510, '500 Wh ref', fontsize=7, color='gray')

# Duration boxplot
ax = axes[1]
data_d = [cycles_all[cycles_all['house']==h]['duration_min'].values for h in houses_sorted]
bp2 = ax.boxplot(data_d, tick_labels=labels, patch_artist=True,
                 medianprops=dict(color='black', lw=1.5),
                 flierprops=dict(marker='.', ms=3, alpha=0.3),
                 whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8))
for patch in bp2['boxes']:
    patch.set_facecolor('#4e9fd9'); patch.set_alpha(0.65)
ax.set_xlabel('House')
ax.set_ylabel('Cycle Duration (min)')
ax.set_title('Cycle Duration Distribution per House\n'
             f'Min threshold: {MIN_CYCLE_MIN} min (Bosch quick-wash) | Max: {MAX_CYCLE_MIN} min',
             fontsize=9)
ax.axhline(MIN_CYCLE_MIN, color='green', lw=0.8, ls='--', alpha=0.6)
ax.axhline(MAX_CYCLE_MIN, color='red',   lw=0.8, ls='--', alpha=0.6)
ax.text(0.5, MIN_CYCLE_MIN+2, f'{MIN_CYCLE_MIN}-min lower bound', fontsize=7, color='green')
ax.grid(axis='y', alpha=0.4)

plt.tight_layout()
plt.savefig(FIG_DIR / 'S2_4_cross_house_comparison.png', dpi=150, bbox_inches='tight')
print(f'  Saved S2_4_cross_house_comparison.png')

# %%
# ── Plot S2-5 (bonus): Heating-phase energy fraction per house ────────────────
print('Generating Plot S2-5 (bonus): Hot vs cold wash energy breakdown ...')

heat_summary = cycles_all.groupby('house').apply(lambda g: pd.Series({
    'total_cycles'    : len(g),
    'hot_pct'         : 100.0 * g['hot_wash'].mean(),
    'median_energy_hot' : g[g['hot_wash']]['energy_wh'].median() if g['hot_wash'].any() else 0,
    'median_energy_cold': g[~g['hot_wash']]['energy_wh'].median() if (~g['hot_wash']).any() else 0,
})).reset_index()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Washing Machine Heating Analysis — All 19 Houses\n'
             '(Heating phase ≥1,800W × ≥3 min → hot wash)',
             fontsize=12, fontweight='bold')

# Hot wash % per house
colors = plt.cm.RdYlGn_r(heat_summary['hot_pct'].values / 100.0)
bars = ax1.bar(heat_summary['house'].astype(str), heat_summary['hot_pct'],
               color=colors, edgecolor='white', linewidth=0.5)
ax1.set_xlabel('House'); ax1.set_ylabel('Hot wash cycles (%)')
ax1.set_title('Hot Wash Fraction per House\n(red = more hot washes → more energy)', fontsize=9)
ax1.set_ylim(0, 100)
ax1.axhline(heat_summary['hot_pct'].mean(), color='black', lw=1.2, ls='--',
            label=f'Mean = {heat_summary["hot_pct"].mean():.0f}%')
ax1.legend(fontsize=8)
ax1.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, heat_summary['hot_pct']):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{val:.0f}%', ha='center', va='bottom', fontsize=7)

# Energy comparison: hot vs cold median
x = np.arange(len(heat_summary))
w = 0.38
ax2.bar(x - w/2, heat_summary['median_energy_hot'],  width=w, color='#d94e4e',
        alpha=0.8, label='Hot wash median (Wh)')
ax2.bar(x + w/2, heat_summary['median_energy_cold'], width=w, color='#4e9fd9',
        alpha=0.8, label='Cold/warm wash median (Wh)')
ax2.set_xticks(x)
ax2.set_xticklabels(heat_summary['house'].astype(str))
ax2.set_xlabel('House'); ax2.set_ylabel('Median cycle energy (Wh)')
ax2.set_title('Hot vs Cold Wash Energy per House\n(gap = energy saving from switching to cold)', fontsize=9)
ax2.legend(fontsize=8)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_DIR / 'S2_5_heating_analysis.png', dpi=150, bbox_inches='tight')
print(f'  Saved S2_5_heating_analysis.png')

# %% [markdown]
# ---
# ## Summary

# %%
print('\n' + '='*70)
print('SECTION 2 COMPLETE')
print('='*70)
total = len(cycles_all)
hot   = cycles_all['hot_wash'].sum()
print(f'  Total cycles detected (19 houses): {total:,}')
print(f'  Hot washes (≥1,800W heating):      {hot:,}  ({100*hot/total:.1f}%)')
print(f'  Median cycle duration:             {cycles_all["duration_min"].median():.0f} min')
print(f'  Median cycle energy:               {cycles_all["energy_wh"].median():.0f} Wh')
print(f'  Energy saving hot→cold (median):   '
      f'{(cycles_all[cycles_all["hot_wash"]]["energy_wh"].median() - cycles_all[~cycles_all["hot_wash"]]["energy_wh"].median()):.0f} Wh/cycle')
print()
print('  Figures saved:')
for fig_name in ['S2_1_daily_trace','S2_2_cycle_profiles','S2_3_usage_heatmap',
                  'S2_4_cross_house_comparison','S2_5_heating_analysis']:
    print(f'    figures/{fig_name}.png')
print('='*70)
