# %% [markdown]
# # Section 2 — Enhanced Plots
# Addresses all assignment requirements:
#   S2_0 : data availability + quality summary (date range, coverage, cycle count per house)
#   S2_1 : 24-h aggregate + WM trace  (clipped y-axis, phases annotated)
#   S2_2 : representative cycle profiles from 4 contrasting houses (2×2 grid)
#   S2_3 : usage patterns — hour-of-day bars + day-of-week bars + heatmap (3-panel)
#   S2_4 : cross-house comparison — energy + duration + peak power (3-panel boxplots)

# %%
import warnings; warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from pathlib import Path

DATA_CLEAN = Path('data/clean')
CKPT_DIR   = Path('data/processed/checkpoints')
FIG_DIR    = Path('figures')
FIG_DIR.mkdir(exist_ok=True)

WM_MAP = {
    1:'Appliance5', 2:'Appliance2', 3:'Appliance6', 4:'Appliance4', 5:'Appliance3',
    6:'Appliance2', 7:'Appliance5', 8:'Appliance4', 9:'Appliance3',10:'Appliance5',
   11:'Appliance3',13:'Appliance3',15:'Appliance3',16:'Appliance5',17:'Appliance4',
   18:'Appliance5',19:'Appliance2',20:'Appliance4',21:'Appliance3',
}
MIN_CYCLE_MIN = 15
MAX_CYCLE_MIN = 180
HEAT_THRESH_W = 1800

# ── Load checkpoints ──────────────────────────────────────────────────────────
print('Loading checkpoints ...')
cycles_all = pd.read_parquet(CKPT_DIR / 'ckpt_wm_cycles_v2.parquet')
wm_all     = pd.read_parquet(CKPT_DIR / 'ckpt_wm_1min.parquet')
print(f'  Cycles: {len(cycles_all):,}  |  WM 1-min rows: {len(wm_all):,}')

# ─────────────────────────────────────────────────────────────────────────────
# S2_0 — Data availability & quality summary
# ─────────────────────────────────────────────────────────────────────────────
print('\nGenerating S2_0: Data availability & quality summary ...')

houses = sorted(WM_MAP.keys())

summary = []
for h in houses:
    sub = wm_all[wm_all['house'] == h]['WM']
    cyc = cycles_all[cycles_all['house'] == h]
    total_rows  = len(sub)
    nonzero_pct = 100.0 * (sub > 5).sum() / total_rows if total_rows > 0 else 0
    zero_pct    = 100.0 - nonzero_pct
    start = sub.index.min()
    end   = sub.index.max()
    n_cyc = len(cyc)
    hot_p = 100.0 * cyc['hot_wash'].mean() if n_cyc > 0 else 0
    # rough missing: NaN bins after resample
    nan_pct = 100.0 * sub.isna().sum() / total_rows
    summary.append(dict(house=h, start=start, end=end,
                        days=(end-start).days,
                        active_pct=nonzero_pct, zero_pct=zero_pct,
                        nan_pct=nan_pct, cycles=n_cyc, hot_pct=hot_p))

df_sum = pd.DataFrame(summary)

# Global timeline reference
global_start = df_sum['start'].min()
global_end   = df_sum['end'].max()
span_days    = (global_end - global_start).days

fig, (ax_main, ax_bar) = plt.subplots(
    1, 2, figsize=(15, 7),
    gridspec_kw={'width_ratios': [3, 1]},
)
fig.suptitle('REFIT Washing Machine — Coverage & Data Quality per House\n'
             '(Official CLEAN data · 19 households · 1-min resolution)',
             fontsize=12, fontweight='bold')

cmap = plt.cm.RdYlGn
y_pos = range(len(houses))

for i, row in df_sum.iterrows():
    y = list(houses).index(row['house'])
    x0 = (row['start'] - global_start).days
    w  = (row['end']   - row['start']).days
    color = cmap(row['active_pct'] / 100.0)
    ax_main.barh(y, w, left=x0, height=0.65,
                 color=color, alpha=0.82, edgecolor='white', linewidth=0.5)
    # Annotate with cycle count
    ax_main.text(x0 + w + 2, y,
                 f"{row['cycles']} cycles\n{row['hot_pct']:.0f}% hot",
                 va='center', fontsize=6.5, color='#333')

ax_main.set_yticks(list(range(len(houses))))
ax_main.set_yticklabels([f'H{h}' for h in houses], fontsize=8)
ax_main.set_xlabel('Days since first recording (Oct 2013)', fontsize=8)

# x-axis as dates
tick_months = pd.date_range(global_start, global_end, freq='3ME')
ax_main.set_xticks([(t - global_start).days for t in tick_months])
ax_main.set_xticklabels([t.strftime("%b'%y") for t in tick_months],
                         fontsize=7, rotation=30, ha='right')
ax_main.set_title('Date coverage  (colour = % time WM is active > 5 W)',
                  fontsize=9, loc='left')
ax_main.grid(axis='x', alpha=0.3)
ax_main.invert_yaxis()

# Colour bar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 100))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax_main, fraction=0.02, pad=0.01)
cbar.set_label('% active readings (>5W)', fontsize=7)

# Right panel: cycle count bar
colors_bar = [cmap(r['active_pct'] / 100.0) for _, r in df_sum.iterrows()]
ax_bar.barh(list(range(len(houses))), df_sum['cycles'].values,
            color=colors_bar, alpha=0.82, edgecolor='white', linewidth=0.5)
ax_bar.set_yticks(list(range(len(houses))))
ax_bar.set_yticklabels([])
ax_bar.set_xlabel('Detected cycles', fontsize=8)
ax_bar.set_title('Cycle\ncount', fontsize=8)
ax_bar.grid(axis='x', alpha=0.3)
ax_bar.invert_yaxis()
for i, row in df_sum.iterrows():
    y = list(houses).index(row['house'])
    ax_bar.text(row['cycles'] + 3, y, str(int(row['cycles'])),
                va='center', fontsize=6.5)

# Detection rule box
rule_txt = (
    "Cycle detection rule:\n"
    "• ON  : per-house p20 of active readings (min 60 W)\n"
    "• OFF : per-house p05 × 0.6 (min 25 W)\n"
    "• Entry: ON sustained ≥ 3 consecutive min\n"
    "• Exit : OFF for ≥ 5 consecutive min (bridges drain phase)\n"
    "• Valid cycle: 15 – 180 min duration\n"
    "• Hot wash: any 1-min bin ≥ 1 800 W for ≥ 3 min"
)
fig.text(0.01, 0.01, rule_txt, fontsize=6.8,
         va='bottom', ha='left',
         bbox=dict(boxstyle='round,pad=0.4', fc='#f5f5f5', ec='#ccc', alpha=0.9))

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig(FIG_DIR / 'S2_0_coverage_quality.png', dpi=150, bbox_inches='tight')
print('  Saved S2_0_coverage_quality.png')

# ─────────────────────────────────────────────────────────────────────────────
# S2_1 — 24-hour aggregate + WM trace  (clipped aggregate axis)
# ─────────────────────────────────────────────────────────────────────────────
print('Generating S2_1: 24h aggregate + WM trace ...')

def pick_wm_day(house, n_cycles=2):
    cyc = cycles_all[cycles_all['house'] == house]
    by_day = cyc.groupby(cyc['start'].dt.date).size()
    matches = by_day[by_day == n_cycles]
    if len(matches) == 0:
        matches = by_day[by_day >= 1]
    return pd.Timestamp(matches.index[len(matches) // 2])

h1_day  = pick_wm_day(1, n_cycles=2)
day_s   = h1_day
day_e   = h1_day + pd.Timedelta('1D') - pd.Timedelta('1min')
h1_data = wm_all[wm_all['house'] == 1].loc[day_s:day_e]
h1_cyc  = cycles_all[(cycles_all['house'] == 1) &
                     (cycles_all['start'].dt.date == h1_day.date())]

# Clip aggregate: 99th percentile of real load (excludes clamp spikes)
agg_clip = float(np.percentile(
    wm_all[wm_all['house'] == 1]['Aggregate'].dropna(), 99.5))
agg_clip = min(agg_clip, 6000)   # never clip below 6 kW

fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True,
                         gridspec_kw={'hspace': 0.06,
                                      'height_ratios': [2.5, 1.2, 0.5]})
fig.suptitle(f'House 1 — Representative Washing Machine Day  |  '
             f'{h1_day.strftime("%d %b %Y")}',
             fontsize=13, fontweight='bold')

# Panel 1: whole-house aggregate (clipped)
ax = axes[0]
agg = h1_data['Aggregate'].clip(upper=agg_clip)
ax.plot(h1_data.index, agg, color='#2c7bb6', lw=0.9, label='Whole-house aggregate')
for _, cyc in h1_cyc.iterrows():
    ax.axvspan(cyc['start'], cyc['end'], alpha=0.15, color='darkorange')
ax.set_ylabel('Power (W)', fontsize=8)
ax.set_title(f'Whole-house aggregate (clipped at {agg_clip:.0f} W — clamp spikes above excluded)',
             fontsize=8, color='dimgray', loc='left')
ax.set_ylim(0, agg_clip * 1.05)
ax.legend(fontsize=8, loc='upper right')
ax.grid(axis='y', alpha=0.3)

# Panel 2: WM channel with phase annotations
ax = axes[1]
ax.fill_between(h1_data.index, 0, h1_data['WM'],
                color='#d7191c', alpha=0.75, label='Washing machine')
ax.axhline(HEAT_THRESH_W, color='#d7191c', lw=0.7, ls='--', alpha=0.5)
ax.text(h1_data.index[2], HEAT_THRESH_W + 30,
        f'{HEAT_THRESH_W}W — heating threshold', fontsize=7, color='#d7191c')

for _, cyc in h1_cyc.iterrows():
    lbl = (f"{'Hot' if cyc['hot_wash'] else 'Cold'} wash\n"
           f"{cyc['duration_min']} min · {cyc['energy_wh']:.0f} Wh\n"
           f"peak {cyc['peak_w']:.0f} W")
    mid = cyc['start'] + (cyc['end'] - cyc['start']) / 2
    ax.annotate(lbl, xy=(mid, cyc['peak_w'] * 0.45),
                ha='center', va='center', fontsize=7,
                bbox=dict(boxstyle='round,pad=0.25', fc='white', alpha=0.8))

ax.set_ylabel('WM Power (W)', fontsize=8)
ax.set_ylim(0, 2700)
ax.legend(fontsize=8, loc='upper right')
ax.grid(axis='y', alpha=0.3)

# Panel 3: WM on/off state bar (thin)
ax = axes[2]
wm_on = (h1_data['WM'] > 25).astype(float)
ax.fill_between(h1_data.index, 0, wm_on,
                step='mid', color='#d7191c', alpha=0.6)
ax.set_ylim(0, 1.5)
ax.set_yticks([])
ax.set_title('WM on/off state', fontsize=7, color='dimgray', loc='left')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
ax.set_xlabel('Time of day', fontsize=9)

plt.savefig(FIG_DIR / 'S2_1_daily_trace.png', dpi=150, bbox_inches='tight')
print(f'  Saved S2_1_daily_trace.png  ({len(h1_cyc)} cycles)')

# ─────────────────────────────────────────────────────────────────────────────
# S2_2 — Representative cycle profiles from 4 contrasting houses (2×2)
# ─────────────────────────────────────────────────────────────────────────────
print('Generating S2_2: Representative cycle profiles (4 houses) ...')

# Houses chosen for contrast:
#   H1  — typical hot-wash house, 91% hot
#   H5  — mostly cold/warm washes, 74% hot (lowest outside H19)
#   H8  — highest median energy (808 Wh), 98% hot
#   H19 — cold-wash outlier, only 9% hot
SHOWCASE = [1, 5, 8, 19]
HOUSE_LABELS = {
    1:  'House 1 — Typical hot-wash\n(91% hot · 33 min median)',
    5:  'House 5 — Mixed temp washes\n(74% hot · 40 min median)',
    8:  'House 8 — High energy machine\n(98% hot · 808 Wh median)',
   19:  'House 19 — Cold-wash household\n(9% hot · 101 min median)',
}

fig, axes = plt.subplots(2, 2, figsize=(14, 9),
                         gridspec_kw={'hspace': 0.38, 'wspace': 0.25})
fig.suptitle('WM Power Profiles — Representative Cycles from 4 Contrasting Houses\n'
             'Each trace = one full detected cycle  |  '
             'Red = hot wash (≥1800W heating)  |  Blue = cold/warm wash',
             fontsize=11, fontweight='bold')

for ax, h in zip(axes.flat, SHOWCASE):
    wm_h   = wm_all[wm_all['house'] == h]['WM']
    cyc_h  = cycles_all[cycles_all['house'] == h].copy()

    # Pick 6 hot + 4 cold (or whatever's available), deterministic
    hot_cyc  = cyc_h[cyc_h['hot_wash']].sample(
                   min(6, cyc_h['hot_wash'].sum()), random_state=7)
    cold_cyc = cyc_h[~cyc_h['hot_wash']].sample(
                   min(4, (~cyc_h['hot_wash']).sum()), random_state=7)
    show_cyc = pd.concat([hot_cyc, cold_cyc]).sort_values('start')

    for _, cyc in show_cyc.iterrows():
        seg   = wm_h.loc[cyc['start']:cyc['end']].values
        t     = np.arange(len(seg))
        color = '#d94e4e' if cyc['hot_wash'] else '#4e9fd9'
        ax.plot(t, seg, color=color, lw=0.8, alpha=0.55)

    # Median from cycles ≤ 120 min (avoid distortion from a few very long ones)
    good = cyc_h[cyc_h['duration_min'] <= 120]
    if len(good) >= 5:
        max_len = int(good['duration_min'].max())
        mats = []
        for _, cyc in good.iterrows():
            seg = wm_h.loc[cyc['start']:cyc['end']].values[:max_len]
            pad = np.full(max_len, np.nan)
            pad[:len(seg)] = seg
            mats.append(pad)
        med = np.nanmedian(np.array(mats), axis=0)
        # Only plot where at least 10% of cycles contribute
        counts = np.sum(~np.isnan(np.array(mats)), axis=0)
        med[counts < max(3, len(good) * 0.10)] = np.nan
        valid = ~np.isnan(med)
        ax.plot(np.arange(max_len)[valid], med[valid],
                color='black', lw=2.0, label='Median', zorder=5)

    ax.axhline(HEAT_THRESH_W, color='#d94e4e', lw=0.7, ls='--', alpha=0.5)
    ax.set_title(HOUSE_LABELS[h], fontsize=8.5, fontweight='bold')
    ax.set_xlabel('Minutes from cycle start', fontsize=8)
    ax.set_ylabel('WM Power (W)', fontsize=8)
    ax.set_xlim(0, 130)
    ax.set_ylim(0, 2700)
    ax.grid(alpha=0.25)

    # Mini legend
    handles = []
    if cyc_h['hot_wash'].any():
        handles.append(mpatches.Patch(color='#d94e4e', alpha=0.6, label='Hot wash'))
    if (~cyc_h['hot_wash']).any():
        handles.append(mpatches.Patch(color='#4e9fd9', alpha=0.6, label='Cold/warm'))
    handles.append(plt.Line2D([0],[0], color='black', lw=2, label='Median'))
    ax.legend(handles=handles, fontsize=7, loc='upper right')

plt.savefig(FIG_DIR / 'S2_2_cycle_profiles.png', dpi=150, bbox_inches='tight')
print('  Saved S2_2_cycle_profiles.png')

# ─────────────────────────────────────────────────────────────────────────────
# S2_3 — Usage patterns: hour-of-day + day-of-week bars + heatmap (3-panel)
# ─────────────────────────────────────────────────────────────────────────────
print('Generating S2_3: Usage patterns (3-panel) ...')

dow_labels   = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
hour_counts  = np.zeros(24)
dow_counts   = np.zeros(7)
heatmap      = np.zeros((24, 7))

for _, cyc in cycles_all.iterrows():
    h, d = cyc['hour_start'], cyc['dow_start']
    hour_counts[h]   += 1
    dow_counts[d]    += 1
    heatmap[h, d]    += 1

hour_pct = 100.0 * hour_counts / hour_counts.sum()
dow_pct  = 100.0 * dow_counts  / dow_counts.sum()
heat_pct = 100.0 * heatmap     / heatmap.sum()

fig = plt.figure(figsize=(15, 9))
gs  = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.30,
                        width_ratios=[1.6, 1], height_ratios=[1, 1])
ax_hour = fig.add_subplot(gs[0, 0])
ax_dow  = fig.add_subplot(gs[1, 0])
ax_heat = fig.add_subplot(gs[:, 1])

fig.suptitle('Washing Machine Usage Patterns — All 19 Houses · 6 776 Detected Cycles',
             fontsize=12, fontweight='bold')

# Hour-of-day bar chart
peak_h = int(np.argmax(hour_pct))
colors_h = ['#d94e4e' if i == peak_h else '#4e9fd9' for i in range(24)]
ax_hour.bar(range(24), hour_pct, color=colors_h, alpha=0.8, edgecolor='white')
ax_hour.set_xticks(range(0, 24, 2))
ax_hour.set_xticklabels([f'{h:02d}:00' for h in range(0, 24, 2)], fontsize=7.5)
ax_hour.set_xlabel('Hour of day (local UK time)', fontsize=8)
ax_hour.set_ylabel('% of all cycles', fontsize=8)
ax_hour.set_title('When do people wash?  (by hour of day)', fontsize=9, fontweight='bold')
ax_hour.text(peak_h, hour_pct[peak_h] + 0.05,
             f'Peak\n{hour_pct[peak_h]:.1f}%', ha='center', fontsize=7.5,
             color='#d94e4e', fontweight='bold')
ax_hour.grid(axis='y', alpha=0.3)
# Morning peak band
ax_hour.axvspan(6.5, 10.5, alpha=0.07, color='darkorange',
                label='Morning peak (7–10am)')
ax_hour.legend(fontsize=7.5)

# Day-of-week bar chart
peak_d = int(np.argmax(dow_pct))
colors_d = ['#d94e4e' if i == peak_d else '#4e9fd9' for i in range(7)]
ax_dow.bar(range(7), dow_pct, color=colors_d, alpha=0.8, edgecolor='white')
ax_dow.set_xticks(range(7))
ax_dow.set_xticklabels(dow_labels, fontsize=9)
ax_dow.set_xlabel('Day of week', fontsize=8)
ax_dow.set_ylabel('% of all cycles', fontsize=8)
ax_dow.set_title('Which days are busiest?', fontsize=9, fontweight='bold')
ax_dow.text(peak_d, dow_pct[peak_d] + 0.05,
            f'Peak\n{dow_pct[peak_d]:.1f}%', ha='center', fontsize=7.5,
            color='#d94e4e', fontweight='bold')
ax_dow.grid(axis='y', alpha=0.3)
# Equal usage reference line
ax_dow.axhline(100/7, color='gray', lw=0.8, ls='--', alpha=0.5,
               label=f'Equal distribution ({100/7:.1f}%)')
ax_dow.legend(fontsize=7.5)

# Heatmap (hour × day)
im = ax_heat.imshow(heat_pct, aspect='auto', cmap='YlOrRd',
                    origin='upper', vmin=0, vmax=heat_pct.max())
cbar = fig.colorbar(im, ax=ax_heat, fraction=0.04, pad=0.02)
cbar.set_label('% of all cycles', fontsize=8)
ax_heat.set_xticks(range(7)); ax_heat.set_xticklabels(dow_labels, fontsize=8)
ax_heat.set_yticks(range(0, 24, 2))
ax_heat.set_yticklabels([f'{h:02d}:00' for h in range(0, 24, 2)], fontsize=7.5)
ax_heat.set_xlabel('Day of week', fontsize=8)
ax_heat.set_ylabel('Hour of day', fontsize=8)
ax_heat.set_title('Joint distribution\n(hour × day)', fontsize=9, fontweight='bold')

ph, pd_ = np.unravel_index(heat_pct.argmax(), heat_pct.shape)
ax_heat.add_patch(plt.Rectangle((pd_-0.5, ph-0.5), 1, 1,
                 fill=False, edgecolor='black', lw=2))
ax_heat.text(pd_, ph, f'{heat_pct[ph,pd_]:.1f}%\npeak',
             ha='center', va='center', fontsize=8, fontweight='bold')

plt.savefig(FIG_DIR / 'S2_3_usage_heatmap.png', dpi=150, bbox_inches='tight')
print('  Saved S2_3_usage_heatmap.png')

# ─────────────────────────────────────────────────────────────────────────────
# S2_4 — Cross-house comparison: energy + duration + peak power (3 panels)
# ─────────────────────────────────────────────────────────────────────────────
print('Generating S2_4: Cross-house comparison (energy + duration + peak power) ...')

houses_s = sorted(cycles_all['house'].unique())
labels   = [f'H{h}' for h in houses_s]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Washing Machine Cycles — Cross-House Comparison\n'
             '(Official CLEAN REFIT data · 19 households · 6 776 cycles)',
             fontsize=12, fontweight='bold')

PANEL_CFG = [
    ('energy_wh',    'Cycle Energy (Wh)',        'Cycle Energy per House',      '#e05a2b', 500,  '500 Wh ref'),
    ('duration_min', 'Cycle Duration (min)',      'Cycle Duration per House',    '#4e9fd9', None, None),
    ('peak_w',       'Peak Power per Cycle (W)',  'Peak Power per House',        '#7b5ea7', 1800, '1800W heating threshold'),
]

for ax, (col, ylabel, title, color, ref, ref_lbl) in zip(axes, PANEL_CFG):
    data = [cycles_all[cycles_all['house'] == h][col].dropna().values
            for h in houses_s]
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True,
                    medianprops=dict(color='black', lw=1.8),
                    flierprops=dict(marker='.', ms=2.5, alpha=0.25),
                    whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8))

    for patch, h in zip(bp['boxes'], houses_s):
        hot_pct = cycles_all[cycles_all['house'] == h]['hot_wash'].mean()
        # Colour by hot-wash fraction (energy/peak) or uniform (duration)
        if col == 'duration_min':
            patch.set_facecolor(color); patch.set_alpha(0.6)
        else:
            patch.set_facecolor(plt.cm.RdYlGn_r(hot_pct * 0.8 + 0.1))
            patch.set_alpha(0.75)

    if ref is not None:
        ax.axhline(ref, color='gray', lw=0.8, ls='--', alpha=0.55)
        ax.text(0.5, ref + (ax.get_ylim()[1] * 0.01 if ax.get_ylim()[1] > 0 else 10),
                ref_lbl, fontsize=7, color='gray')

    ax.set_xlabel('House', fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title +
                 ('\n(box colour = hot-wash fraction: red=high, green=low)'
                  if col != 'duration_min' else ''),
                 fontsize=8.5)
    ax.tick_params(axis='x', labelsize=7.5)
    ax.grid(axis='y', alpha=0.35)

    # Median annotation on each box
    for i, h in enumerate(houses_s):
        med = cycles_all[cycles_all['house'] == h][col].median()
        ax.text(i + 1, med, f'{med:.0f}',
                ha='center', va='bottom', fontsize=5.5, color='black',
                fontweight='bold')

# Duration-specific: add bounds lines
ax_d = axes[1]
ax_d.axhline(MIN_CYCLE_MIN, color='green', lw=0.8, ls='--', alpha=0.6)
ax_d.axhline(MAX_CYCLE_MIN, color='red',   lw=0.8, ls='--', alpha=0.6)
ax_d.text(0.5, MIN_CYCLE_MIN + 1, f'{MIN_CYCLE_MIN}-min lower bound (Bosch quick-wash)',
          fontsize=6.5, color='green')

plt.tight_layout()
plt.savefig(FIG_DIR / 'S2_4_cross_house_comparison.png', dpi=150, bbox_inches='tight')
print('  Saved S2_4_cross_house_comparison.png')

# ─────────────────────────────────────────────────────────────────────────────
# S2_5 — Hot vs cold heating analysis (unchanged — already good)
# ─────────────────────────────────────────────────────────────────────────────
print('Generating S2_5: Hot vs cold wash energy breakdown ...')

heat_summary = cycles_all.groupby('house').apply(lambda g: pd.Series({
    'total_cycles'       : len(g),
    'hot_pct'            : 100.0 * g['hot_wash'].mean(),
    'median_energy_hot'  : g[g['hot_wash']]['energy_wh'].median() if g['hot_wash'].any() else 0,
    'median_energy_cold' : g[~g['hot_wash']]['energy_wh'].median() if (~g['hot_wash']).any() else 0,
}), include_groups=False).reset_index()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Washing Machine Heating Analysis — All 19 Houses\n'
             '(Heating phase ≥1,800W × ≥3 min → hot wash)',
             fontsize=12, fontweight='bold')

cmap2 = plt.cm.RdYlGn_r
colors = cmap2(heat_summary['hot_pct'].values / 100.0)
bars   = ax1.bar(heat_summary['house'].astype(str), heat_summary['hot_pct'],
                 color=colors, edgecolor='white', linewidth=0.5)
ax1.set_xlabel('House'); ax1.set_ylabel('Hot wash cycles (%)')
ax1.set_title('Hot Wash Fraction per House', fontsize=9)
ax1.set_ylim(0, 105)
ax1.axhline(heat_summary['hot_pct'].mean(), color='black', lw=1.2, ls='--',
            label=f"Cohort mean = {heat_summary['hot_pct'].mean():.0f}%")
ax1.legend(fontsize=8); ax1.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, heat_summary['hot_pct']):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{val:.0f}%', ha='center', va='bottom', fontsize=7)

x = np.arange(len(heat_summary)); w = 0.38
ax2.bar(x - w/2, heat_summary['median_energy_hot'],  width=w, color='#d94e4e',
        alpha=0.8, label='Hot wash median (Wh)')
ax2.bar(x + w/2, heat_summary['median_energy_cold'], width=w, color='#4e9fd9',
        alpha=0.8, label='Cold/warm wash median (Wh)')
ax2.set_xticks(x); ax2.set_xticklabels(heat_summary['house'].astype(str))
ax2.set_xlabel('House'); ax2.set_ylabel('Median cycle energy (Wh)')
ax2.set_title('Hot vs Cold Wash Energy per House\n'
              '(gap = energy saving from switching to cold)', fontsize=9)
ax2.legend(fontsize=8); ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_DIR / 'S2_5_heating_analysis.png', dpi=150, bbox_inches='tight')
print('  Saved S2_5_heating_analysis.png')

# ─────────────────────────────────────────────────────────────────────────────
print('\n' + '='*70)
print('ENHANCED PLOTS COMPLETE')
print('='*70)
for f in ['S2_0_coverage_quality','S2_1_daily_trace','S2_2_cycle_profiles',
          'S2_3_usage_heatmap','S2_4_cross_house_comparison','S2_5_heating_analysis']:
    print(f'  figures/{f}.png')
print('='*70)
