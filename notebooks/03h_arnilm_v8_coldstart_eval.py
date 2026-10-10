"""
ARNILM V8 — Cold-start corrected evaluation
============================================
Methodological fix: household signature for H1 is computed from the first 14 days
of Part 2 data only (calibration window). Inference and metrics run on the remaining
period. This validates the claimed deployment scenario: 14-day aggregate-only
cold-start → freeze signature → predict on new data.

Also reports pre-clipping vs post-clipping constraint violations.
"""
import warnings; warnings.filterwarnings('ignore')
import sys, os, json
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score

CKPT_DIR = Path('data/processed/checkpoints')
RES_DIR  = Path('results/section3'); RES_DIR.mkdir(exist_ok=True)

PART2_START   = pd.Timestamp('2014-04-01')
TEST_HOUSE    = 1
CAL_HOUSES    = [5, 7, 11, 17]
CAL_WEEKS     = 4

ON_THRESH_W   = 25.0
MAX_WM_W      = 3000.0
MAX_AGG_W     = 8000.0
SEQ_LEN       = 720
COLDSTART_DAYS = 14   # days used for H1 signature; eval runs on the remainder

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── Reproduce V8 feature engineering ──────────────────────────────────────────

SIG_COLS = ['sig_med_dur', 'sig_med_energy', 'sig_agg_high_energy_frac',
            'sig_ph_sin',  'sig_ph_cos',     'sig_med_peak',
            'sig_agg_high_energy_frac2']
N_DYN = 13; N_STATIC = 7; N_INPUT = 1 + N_DYN + N_STATIC  # 21

def detect_agg_cycles(agg, thresh_on=300.0, thresh_off=80.0,
                      dur_min=15, dur_max=180, peak_min=400.0, hyst_min=5):
    events = []
    state = 'IDLE'; ev_start = 0; drop_start = 0; cum_e = 0.0; pk = 0.0
    for i in range(len(agg)):
        v = float(agg[i]) if np.isfinite(agg[i]) else 0.0
        if state == 'IDLE':
            if v >= thresh_on:
                ev_start = i; cum_e = 0.0; pk = 0.0; state = 'ACTIVE'
        elif state == 'ACTIVE':
            if v < thresh_off:
                drop_start = i; state = 'COOLING'
            else:
                cum_e += v / 60.0; pk = max(pk, v)
        elif state == 'COOLING':
            if v >= thresh_off:
                state = 'ACTIVE'; cum_e += v / 60.0; pk = max(pk, v)
            elif (i - drop_start) >= hyst_min:
                dur = drop_start - ev_start
                if dur_min <= dur <= dur_max and pk >= peak_min:
                    events.append(dict(
                        duration_min=dur, energy_wh=cum_e, peak_w=pk,
                        # renamed: high_energy_event replaces hot_wash
                        high_energy_event=(cum_e >= 350.0),
                        hour_start=i // 60 % 24))
                state = 'IDLE'
    return pd.DataFrame(events) if events else pd.DataFrame(
        columns=['duration_min', 'energy_wh', 'peak_w', 'high_energy_event', 'hour_start'])


def house_sig_from_agg(agg_series):
    """Compute aggregate-event behavioral signature from agg_series only."""
    cyc = detect_agg_cycles(agg_series.values)
    if len(cyc) < 3:
        return {c: 0.0 for c in SIG_COLS}
    ph = float(cyc['hour_start'].mode().iloc[0])
    # renamed: sig_agg_high_energy_frac (not 'hot_frac' — derived from aggregate only)
    high_e_frac = float(cyc['high_energy_event'].mean())
    return dict(
        sig_med_dur                = float(cyc['duration_min'].median()) / 180.0,
        sig_med_energy             = float(cyc['energy_wh'].median())    / 800.0,
        sig_agg_high_energy_frac   = high_e_frac,
        sig_ph_sin                 = float(np.sin(2 * np.pi * ph / 24)),
        sig_ph_cos                 = float(np.cos(2 * np.pi * ph / 24)),
        sig_med_peak               = float(cyc['peak_w'].median()) / MAX_WM_W,
        sig_agg_high_energy_frac2  = high_e_frac ** 2,
    )


def detect_events(agg_vals, thresh=80.0):
    n = len(agg_vals)
    ev_active = np.zeros(n, np.float32); ev_dur  = np.zeros(n, np.float32)
    ev_energy = np.zeros(n, np.float32); ev_peak = np.zeros(n, np.float32)
    since_ev  = np.zeros(n, np.float32)
    in_ev = False; ev_start = 0; cum_e = 0.0; pk = 0.0; last_end = -1
    for i in range(n):
        v = float(agg_vals[i]) if np.isfinite(agg_vals[i]) else 0.0
        if v >= thresh:
            if not in_ev:
                in_ev = True; ev_start = i; cum_e = 0.0; pk = 0.0
            cum_e += v / 60.0; pk = max(pk, v)
            ev_active[i] = 1.0; ev_dur[i] = float(i - ev_start + 1) / 180.0
            ev_energy[i] = cum_e / 500.0; ev_peak[i] = pk / MAX_WM_W
        else:
            if in_ev:
                in_ev = False; last_end = i
            since_ev[i] = float(i - last_end) / (24 * 60) if last_end >= 0 else 1.0
    return np.stack([ev_active, ev_dur, ev_energy, ev_peak, since_ev], axis=1)


def build_dyn_covariates(df_h):
    agg  = df_h['Aggregate'].values.astype(np.float32)
    n    = len(agg)
    ev   = detect_events(agg)
    hour = (df_h.index.hour + df_h.index.minute / 60.0).values
    dow  = df_h.index.dayofweek.astype(float).values
    temporal = np.stack([
        np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24),
        np.sin(2*np.pi*dow/7),   np.cos(2*np.pi*dow/7),
    ], axis=1).astype(np.float32)
    agg_diff = np.zeros(n, np.float32)
    agg_diff[1:] = (agg[1:] - agg[:-1]) / MAX_AGG_W
    agg_abs_diff = np.abs(agg_diff)
    agg_s = pd.Series(agg)
    roll10 = agg_s.rolling(10, min_periods=1).std().fillna(0).values.astype(np.float32)
    agg_roll_std_10 = roll10 / MAX_AGG_W
    roll30 = agg_s.rolling(30, min_periods=1).std().fillna(0).values.astype(np.float32)
    agg_roll_std_30 = roll30 / MAX_AGG_W
    deriv = np.stack([agg_diff, agg_abs_diff, agg_roll_std_10, agg_roll_std_30], axis=1)
    return np.concatenate([ev, temporal, deriv], axis=1)


# ── Model (identical to V8) ────────────────────────────────────────────────────

class ARNILM(nn.Module):
    def __init__(self, n_input=N_INPUT, hidden=256, n_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(n_input, hidden, n_layers, batch_first=True, dropout=0.2)
        self.reg_head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1), nn.Softplus())
        self.cls_head = nn.Sequential(
            nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, agg, dyn, sig, hidden=None):
        agg_n = (agg / MAX_AGG_W).unsqueeze(-1)
        sig_e = sig.unsqueeze(1).expand(-1, agg.size(1), -1)
        x = torch.cat([agg_n, dyn, sig_e], dim=-1)
        h, hidden = self.lstm(x, hidden)
        mu_raw    = self.reg_head(h).squeeze(-1)
        cls_logit = self.cls_head(h).squeeze(-1)
        p_on      = torch.sigmoid(cls_logit)
        y_hat     = mu_raw * MAX_WM_W * p_on
        return y_hat, p_on, cls_logit, hidden


# ── Load data and checkpoint ──────────────────────────────────────────────────

print('Loading data ...')
wm_all = pd.read_parquet(CKPT_DIR / 'ckpt_wm_1min_clean.parquet')
wm_all.index = pd.to_datetime(wm_all.index)
wm_all['WM']        = wm_all['WM'].astype(float).fillna(0).clip(0, MAX_WM_W)
wm_all['Aggregate'] = wm_all['Aggregate'].astype(float).fillna(0).clip(0)
wm_part2 = wm_all[wm_all.index >= PART2_START].copy()
print(f'  Part2: {len(wm_part2):,} rows  |  {wm_part2["house"].nunique()} houses')

print('\nLoading V8 checkpoint ...')
model = ARNILM().to(DEVICE)
model.load_state_dict(torch.load(CKPT_DIR / 'nilm_ar_lstm.pt', map_location=DEVICE))
model.eval()
print('  Checkpoint loaded.')

config = json.loads((CKPT_DIR / 'nilm_ar_lstm_v8_config.json').read_text())
P_ON_THRESHOLD = config['p_on_threshold']
print(f'  p_on threshold (from training calibration): {P_ON_THRESHOLD}')

# ── Build training-house signatures (unchanged — full Part 2 is correct for them) ──

print('\nComputing house signatures ...')
house_sigs = {}
for h in sorted(wm_part2['house'].unique()):
    sub = wm_part2[wm_part2['house'] == h]
    if h == TEST_HOUSE:
        # COLD-START FIX: use only the first COLDSTART_DAYS days for H1 signature
        h1_start = sub.index[0]
        h1_cal_end = h1_start + pd.Timedelta(days=COLDSTART_DAYS)
        cal_sub = sub[sub.index < h1_cal_end]
        house_sigs[h] = house_sig_from_agg(cal_sub['Aggregate'])
        print(f'  H{h:2d} [COLD-START]: sig from {h1_start.date()} to {h1_cal_end.date()} '
              f'({len(cal_sub):,} rows = {COLDSTART_DAYS} days)')
    else:
        house_sigs[h] = house_sig_from_agg(sub['Aggregate'])

# ── Build H1 eval data (post-calibration window only) ─────────────────────────

h1_sub = wm_part2[wm_part2['house'] == TEST_HOUSE]
h1_start = h1_sub.index[0]
h1_eval_start = h1_start + pd.Timedelta(days=COLDSTART_DAYS)
h1_eval = h1_sub[h1_sub.index >= h1_eval_start]

print(f'\nH1 eval window: {h1_eval_start.date()} → {h1_sub.index[-1].date()} '
      f'({len(h1_eval):,} timesteps)')
print(f'  (Full Part2: {len(h1_sub):,}  |  Excluded first {COLDSTART_DAYS} days for cold-start: '
      f'{len(h1_sub) - len(h1_eval):,})')

agg_eval = h1_eval['Aggregate'].values.astype(np.float32)
wm_eval  = h1_eval['WM'].values.astype(np.float32)
dyn_eval = build_dyn_covariates(h1_eval)
sig_eval  = np.array([house_sigs[TEST_HOUSE][c] for c in SIG_COLS], dtype=np.float32)

# ── Inference ─────────────────────────────────────────────────────────────────

print('\nRunning inference on H1 eval window ...')
chunk = 10000
T = len(agg_eval)
preds_raw = np.zeros(T, np.float32)   # before clip — for constraint violation stats
p_ons     = np.zeros(T, np.float32)
hidden = None
with torch.no_grad():
    for s in range(0, T, chunk):
        e   = min(s + chunk, T)
        ac  = torch.FloatTensor(agg_eval[s:e]).unsqueeze(0).to(DEVICE)
        dc  = torch.FloatTensor(dyn_eval[s:e]).unsqueeze(0).to(DEVICE)
        sc  = torch.FloatTensor(sig_eval).unsqueeze(0).to(DEVICE)
        yh, po, _, hidden = model(ac, dc, sc, hidden)
        preds_raw[s:e] = yh[0].cpu().numpy()
        p_ons[s:e]     = po[0].cpu().numpy()
        hidden = tuple(hh.detach() for hh in hidden)

# Constraint violations — measure BEFORE clipping
violations = np.clip(preds_raw - agg_eval, 0, None)
viol_rate  = float((violations > 0).mean() * 100)
viol_mean  = float(violations.mean())
viol_max   = float(violations.max())

preds_clipped = np.clip(preds_raw, 0, agg_eval)   # hard constraint

# ── Metrics ───────────────────────────────────────────────────────────────────

true_on  = (wm_eval >= ON_THRESH_W).astype(int)
pred_on  = (p_ons >= P_ON_THRESHOLD).astype(int)

mae      = float(np.mean(np.abs(preds_clipped - wm_eval)))
rmse     = float(np.sqrt(np.mean((preds_clipped - wm_eval)**2)))
on_mask  = true_on == 1
mae_on   = float(np.mean(np.abs(preds_clipped[on_mask] - wm_eval[on_mask]))) if on_mask.sum() > 0 else float('nan')
f1       = float(f1_score(true_on, pred_on, zero_division=0))
prec     = float(precision_score(true_on, pred_on, zero_division=0))
rec      = float(recall_score(true_on, pred_on, zero_division=0))
true_kwh = float(wm_eval.sum() / 60000)
pred_kwh = float(preds_clipped.sum() / 60000)
energy_err = abs(pred_kwh - true_kwh) / max(true_kwh, 1e-6) * 100

# ── Report ────────────────────────────────────────────────────────────────────

print('\n' + '='*72)
print('ARNILM V8 — Cold-start corrected evaluation (14-day signature)')
print('='*72)
print(f'  Cold-start window:  {COLDSTART_DAYS} days of aggregate only')
print(f'  Eval timesteps:     {T:,}  ({T/60/24:.1f} days)')
print(f'  WM ON fraction:     {true_on.mean()*100:.2f}%')
print()
print(f'  MAE:            {mae:.2f} W          [V8 full-period: 20.0 W]')
print(f'  RMSE:           {rmse:.2f} W         [V8 full-period: 119 W]')
print(f'  MAE (ON):       {mae_on:.2f} W        [V8 full-period: 395 W]')
print(f'  F1:             {f1:.4f}             [V8 full-period: 0.306]')
print(f'  Precision:      {prec:.4f}')
print(f'  Recall:         {rec:.4f}')
print(f'  Energy error:   {energy_err:.1f}%          [V8 full-period: 69.7%]')
print()
print('  Physical constraint violations (WM > Aggregate):')
print(f'    Before clipping:  rate={viol_rate:.2f}%  mean={viol_mean:.3f} W  max={viol_max:.1f} W')
print(f'    After clipping:   rate=0.00%  mean=0.000 W  max=0.0 W')
print('='*72)

# Side-by-side comparison table
print('\nComparison: original evaluation vs 14-day cold-start corrected')
print(f'  {"Metric":<20} {"V8 (full-period)":<22} {"V8 (14-day cold-start)"}')
print(f'  {"-"*62}')
print(f'  {"MAE":<20} {"20.0 W":<22} {mae:.1f} W')
print(f'  {"RMSE":<20} {"119 W":<22} {rmse:.0f} W')
print(f'  {"MAE (ON)":<20} {"395 W":<22} {mae_on:.0f} W')
print(f'  {"F1":<20} {"0.306":<22} {f1:.3f}')
print(f'  {"Precision":<20} {"0.358":<22} {prec:.3f}')
print(f'  {"Recall":<20} {"0.267":<22} {rec:.3f}')
print(f'  {"Energy error":<20} {"69.7%":<22} {energy_err:.1f}%')

results = {
    'model': 'ARNILM_V8_coldstart14d',
    'coldstart_days': COLDSTART_DAYS,
    'eval_timesteps': T,
    'mae': round(mae, 2),
    'rmse': round(rmse, 2),
    'mae_on': round(mae_on, 2),
    'f1': round(f1, 4),
    'precision': round(prec, 4),
    'recall': round(rec, 4),
    'energy_err_pct': round(energy_err, 1),
    'constraint_viol_rate_pct_before_clip': round(viol_rate, 4),
    'constraint_viol_mean_W_before_clip': round(viol_mean, 4),
    'constraint_viol_max_W_before_clip': round(viol_max, 2),
    'constraint_viol_after_clip': 0.0,
}
import json
out = RES_DIR / 'metrics_ar_lstm_v8_coldstart.json'
out.write_text(json.dumps(results, indent=2))
print(f'\nResults saved → {out}')
