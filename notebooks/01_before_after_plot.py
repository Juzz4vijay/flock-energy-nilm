# Rebuild C1_before_after_week.png
# - Shared y-axis per channel row (same scale before/after — magnitude of removal is visible)
# - Findings summary text box
# - Specific anomaly callouts (spike, gap fill, outage)

import warnings; warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from pathlib import Path

DATA_RAW_P1 = Path('data/raw/RAW_House1_Part1.csv')
DATA_RAW_P2 = Path('data/raw/RAW_House1_Part2.csv')
DATA_PROC   = Path('data/processed/house1_clean_1min.parquet')
FIG_DIR   = Path('figures')

# ── Load raw data (before) ────────────────────────────────────────────────────
print('Loading raw House 1 ...')

def load_part(path):
    df = pd.read_csv(path, header=0, dtype=str)
    # Normalise column names: strip spaces, ensure Unix exists
    df.columns = [c.strip() for c in df.columns]
    if 'Unix' not in df.columns and 'Time' in df.columns:
        # Some parts have a human-readable Time + Unix; keep Unix
        pass
    df['Unix'] = pd.to_numeric(df['Unix'], errors='coerce')
    df = df.dropna(subset=['Unix'])
    for col in ['Aggregate','Appliance1','Appliance5']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

p1 = load_part(DATA_RAW_P1)
p2 = load_part(DATA_RAW_P2)
raw = pd.concat([p1, p2], ignore_index=True)
raw['Time'] = pd.to_datetime(raw['Unix'], unit='s', utc=True).dt.tz_convert('Europe/London')
raw = raw.set_index('Time').sort_index()

# ── Load cleaned data (after) ─────────────────────────────────────────────────
print('Loading cleaned parquet ...')
clean = pd.read_parquet(DATA_PROC)
clean.index = pd.to_datetime(clean.index)
if clean.index.tz is None:
    clean.index = clean.index.tz_localize('UTC').tz_convert('Europe/London')

# ── Choose window: Nov 1–10 2013 (shows spikes, outage, gap fills) ───────────
WIN_START = pd.Timestamp('2013-11-01', tz='Europe/London')
WIN_END   = pd.Timestamp('2013-11-10', tz='Europe/London')

raw_w   = raw.loc[WIN_START:WIN_END]
clean_w = clean.loc[WIN_START:WIN_END]

# Resample raw to 1-min for fair comparison
raw_1min = raw_w[['Aggregate','Appliance1','Appliance5']].resample('1min').mean()

# Convert to tz-naive for matplotlib compatibility
def tz_strip(df):
    df = df.copy()
    df.index = df.index.tz_localize(None)
    return df

raw_1min = tz_strip(raw_1min)
clean_w  = tz_strip(clean_w)

WIN_START = WIN_START.tz_localize(None)
WIN_END   = WIN_END.tz_localize(None)

# ── Channel definitions ───────────────────────────────────────────────────────
CHANNELS = [
    ('Aggregate',   'Appliance1', 'Appliance5'),  # (col before, col after, col after)
    # actually all same names — just need to map correctly
]
CH_CONFIG = [
    dict(col='Aggregate',  label='Aggregate power',     color='#2c7bb6', unit='W'),
    dict(col='Appliance5', label='Washing machine (App5)', color='#d7191c', unit='W'),
    dict(col='Appliance1', label='Fridge (App1)',        color='#1a9641', unit='W'),
]

# ── Compute per-channel y-max (shared between before/after for same channel) ──
ymaxes = {}
for ch in CH_CONFIG:
    col = ch['col']
    bef = raw_1min[col].dropna()
    aft = clean_w[col].dropna() if col in clean_w.columns else pd.Series(dtype=float)
    # y-max = 99th pct of BEFORE (so spikes set the scale, showing their removal)
    ymax_bef = float(np.percentile(bef[bef > 0], 99.5)) if len(bef[bef > 0]) > 0 else 100
    ymax_bef = max(ymax_bef, bef.max() * 0.15)   # ensure spikes visible on shared scale
    ymaxes[col] = bef.max()   # full range — spike visible

# Override: cap at a readable level and note what's above
YMAXES = {
    'Aggregate':  6000,    # real load rarely exceeds 6 kW in this house; annotate spikes above
    'Appliance5': 2700,    # WM max ~2.4 kW; give head room
    'Appliance1': 950,     # Fridge spike was ~900 W — keep full range to show anomaly
}

# ── Figure setup ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 11))
fig.patch.set_facecolor('#fafafa')

# Grid: 3 channel rows × 2 (before/after) + bottom summary row
# Use gridspec: 3 rows for channels, 1 narrow row for summary
from matplotlib.gridspec import GridSpec
gs = GridSpec(4, 2, figure=fig,
              height_ratios=[2.2, 1.5, 1.5, 0.9],
              hspace=0.08, wspace=0.06,
              left=0.07, right=0.97, top=0.92, bottom=0.06)

axes_bef = [fig.add_subplot(gs[i, 0]) for i in range(3)]
axes_aft = [fig.add_subplot(gs[i, 1]) for i in range(3)]

fig.suptitle('House 1 — Before vs After Cleaning  |  1–10 Nov 2013\n'
             'Raw 8-sec data resampled to 1-min  →  cleaned pipeline output',
             fontsize=12, fontweight='bold', y=0.97)

# Column headers
axes_bef[0].set_title('BEFORE  (raw 8-sec → 1-min resample)',
                       fontsize=10, fontweight='bold', color='#c0392b', pad=6)
axes_aft[0].set_title('AFTER  (pipeline output: sorted · deduped · filtered · gap-filled)',
                       fontsize=10, fontweight='bold', color='#27ae60', pad=6)

xfmt = mdates.DateFormatter('%b %d')
xloc = mdates.DayLocator(interval=2)

for i, ch in enumerate(CH_CONFIG):
    col   = ch['col']
    label = ch['label']
    color = ch['color']
    ymax  = YMAXES[col]

    bef_ser = raw_1min[col].astype(float)
    aft_ser = (clean_w[col].astype(float) if col in clean_w.columns
               else pd.Series(dtype=float))

    ax_b = axes_bef[i]
    ax_a = axes_aft[i]

    # ── BEFORE ──
    ax_b.fill_between(bef_ser.index, 0, bef_ser.clip(upper=ymax),
                      color=color, alpha=0.55, lw=0)
    ax_b.plot(bef_ser.index, bef_ser.clip(upper=ymax),
              color=color, lw=0.6, alpha=0.8)
    # Spike markers: values above ymax shown as red triangles
    spikes = bef_ser[bef_ser > ymax]
    if len(spikes):
        ax_b.scatter(spikes.index, [ymax * 0.96] * len(spikes),
                     marker='^', color='red', s=18, zorder=5,
                     label=f'{len(spikes)} spike(s) > {ymax:,.0f}W')
        ax_b.legend(fontsize=6.5, loc='upper right', framealpha=0.7)

    ax_b.set_ylim(0, ymax * 1.08)
    ax_b.set_ylabel(f'{label}\n(W)', fontsize=7.5, labelpad=3)
    ax_b.grid(axis='y', alpha=0.3, lw=0.5)
    ax_b.set_facecolor('#fff8f8')

    # ── AFTER ──
    # outage regions
    if 'outage' in clean_w.columns:
        outage_mask = clean_w['outage'].fillna(False).astype(bool)
        in_outage = False
        o_start = None
        for ts, val in outage_mask.items():
            if val and not in_outage:
                in_outage = True; o_start = ts
            elif not val and in_outage:
                ax_a.axvspan(o_start, ts, color='#aaa', alpha=0.18, zorder=0)
                in_outage = False
        if in_outage:
            ax_a.axvspan(o_start, outage_mask.index[-1], color='#aaa', alpha=0.18, zorder=0)

    # SARIMA imputed regions
    if 'impute_source' in clean_w.columns:
        sarima_mask = (clean_w['impute_source'] == 'sarima')
        if sarima_mask.any():
            in_s = False; s_start = None
            for ts, val in sarima_mask.items():
                if val and not in_s:
                    in_s = True; s_start = ts
                elif not val and in_s:
                    ax_a.axvspan(s_start, ts, color='#f5a623', alpha=0.28, zorder=0)
                    in_s = False

    aft_plot = aft_ser.where(aft_ser.notna(), other=np.nan)
    ax_a.fill_between(aft_plot.index, 0, aft_plot.clip(upper=ymax),
                      color=color, alpha=0.55, lw=0)
    ax_a.plot(aft_plot.index, aft_plot.clip(upper=ymax),
              color=color, lw=0.6, alpha=0.8)

    ax_a.set_ylim(0, ymax * 1.08)
    ax_a.set_facecolor('#f8fff8')
    ax_a.grid(axis='y', alpha=0.3, lw=0.5)
    ax_a.yaxis.set_label_position('right')
    ax_a.yaxis.tick_right()
    ax_a.set_ylabel(f'{label}\n(W)', fontsize=7.5, labelpad=3)

    # Shared x-axis: hide ticks on all but bottom row
    for ax in [ax_b, ax_a]:
        ax.set_xlim(WIN_START, WIN_END)
        if i < 2:
            ax.set_xticklabels([])
        else:
            ax.xaxis.set_major_formatter(xfmt)
            ax.xaxis.set_major_locator(xloc)
            ax.tick_params(axis='x', labelsize=8)

# ── Specific callouts ─────────────────────────────────────────────────────────
# Fridge spike annotation on BEFORE panel (row 2)
fridge_bef = raw_1min['Appliance1']
spike_t = fridge_bef[fridge_bef > 500].index
if len(spike_t):
    t0 = spike_t[0]
    axes_bef[2].annotate(
        'Impossible spike\n~900 W fridge\n→ removed (R3)',
        xy=(t0, min(fridge_bef[t0], YMAXES['Appliance1'] * 0.96)),
        xytext=(t0 + pd.Timedelta('18h'), YMAXES['Appliance1'] * 0.7),
        fontsize=7, color='red', fontweight='bold',
        arrowprops=dict(arrowstyle='->', color='red', lw=1.2),
        bbox=dict(boxstyle='round,pad=0.25', fc='#ffe8e8', ec='red', alpha=0.85)
    )

# Outage label on AFTER panel (row 0)
outage_start = pd.Timestamp('2013-11-07 18:00', tz='Europe/London')
axes_aft[0].annotate(
    'Outage flagged\n(>24 h gap — not\nimputed: R7)',
    xy=(outage_start + pd.Timedelta('12h'), YMAXES['Aggregate'] * 0.6),
    xytext=(outage_start - pd.Timedelta('24h'), YMAXES['Aggregate'] * 0.82),
    fontsize=7, color='#555',
    arrowprops=dict(arrowstyle='->', color='#555', lw=1.0),
    bbox=dict(boxstyle='round,pad=0.25', fc='#f0f0f0', ec='#999', alpha=0.85)
)

# Aggregate spike annotation on BEFORE
agg_bef = raw_1min['Aggregate']
big_spikes = agg_bef[agg_bef > YMAXES['Aggregate']]
if len(big_spikes):
    t0 = big_spikes.index[0]
    axes_bef[0].annotate(
        f'{len(big_spikes)} clamp spikes\nup to {agg_bef.max()/1000:.0f} kW\n→ removed (R3)',
        xy=(t0, YMAXES['Aggregate'] * 0.96),
        xytext=(t0 + pd.Timedelta('30h'), YMAXES['Aggregate'] * 0.72),
        fontsize=7, color='red', fontweight='bold',
        arrowprops=dict(arrowstyle='->', color='red', lw=1.2),
        bbox=dict(boxstyle='round,pad=0.25', fc='#ffe8e8', ec='red', alpha=0.85)
    )

# ── Legend patches for AFTER column ──────────────────────────────────────────
legend_patches = [
    mpatches.Patch(color='#aaa', alpha=0.4, label='Outage (flagged, not filled)'),
    mpatches.Patch(color='#f5a623', alpha=0.5, label='SARIMA imputed (30 min – 24 h gaps)'),
]
axes_aft[0].legend(handles=legend_patches, fontsize=7, loc='upper left',
                   framealpha=0.85, edgecolor='#ccc')

# ── Findings summary bar (bottom row) ────────────────────────────────────────
ax_sum = fig.add_subplot(gs[3, :])
ax_sum.axis('off')

findings = [
    ('Rows in raw file',          '8 533 035'),
    ('Out-of-order timestamps',   '2'),
    ('Duplicate timestamps',      '956 K'),
    ('Impossible IAM readings\n(>4 000 W)',    '3 350'),
    ('Aggregate clamp spikes\nremoved',        '45+'),
    ('1-min bins after resample', '920 031'),
    ('Short gaps filled\n(≤30 min, linear)',   '349 gaps'),
    ('SARIMA-imputed\n(30 min – 24 h)',         '1 862 min'),
    ('Outages flagged\n(>24 h, not filled)',    '12 gaps\n1 845 h'),
]

n = len(findings)
col_w = 1.0 / n
for j, (label, val) in enumerate(findings):
    x = j * col_w + col_w * 0.5
    ax_sum.text(x, 0.72, val, ha='center', va='top',
                fontsize=8, fontweight='bold', color='#1a1d23',
                transform=ax_sum.transAxes)
    ax_sum.text(x, 0.30, label, ha='center', va='top',
                fontsize=6.5, color='#555',
                transform=ax_sum.transAxes)
    if j > 0:
        ax_sum.plot([j * col_w, j * col_w], [0.05, 0.95],
                    color='#ddd', lw=0.8, transform=ax_sum.transAxes)

ax_sum.set_xlim(0, 1)
ax_sum.add_patch(mpatches.FancyBboxPatch(
    (0, 0), 1, 1, boxstyle='round,pad=0.02',
    fc='#f0f4ff', ec='#c0cce0', lw=1,
    transform=ax_sum.transAxes, zorder=0
))
ax_sum.text(0.5, 0.97, 'Findings Summary — House 1 Raw Data Audit',
            ha='center', va='top', fontsize=8, fontweight='bold',
            color='#2c5f9e', transform=ax_sum.transAxes)

plt.savefig(FIG_DIR / 'C1_before_after_week.png', dpi=150, bbox_inches='tight',
            facecolor='#fafafa')
print('Saved C1_before_after_week.png')
