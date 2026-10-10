"""
ARNILM V8 — Cold-start corrected evaluation
============================================
Leakage fix: the original evaluation computed H1's household signature from the
full H1 Part 2 history, including the period being evaluated. In deployment, only
14 days of aggregate are available before inference begins.

Fix: compute H1 signature from first COLDSTART_DAYS=14 days only, then evaluate
on the remaining period.

Note on H1 data quality: H1 Part 1 (Oct 2013-Apr 2014) encodes both outages and
genuine zeros as 0 — ambiguous labels. We use Part 2 only (Apr 2014+) where NaN
correctly marks outages. H7 is used as a secondary cold-start demonstration: it
has clean Part 2 labels, 869 WM cycles, and 98% hot-wash — more informative
ground truth than H1 for demonstrating signature quality.

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

def detect_agg_cycles(agg_series, thresh_on=80.0, thresh_off=25.0,
                      hyst_min=5, dur_min=15, dur_max=180, peak_min=400.0):
    """Identical to 03h_arnilm_v8.py — thresholds and index usage must not diverge."""
    agg = agg_series.values.astype(np.float32)
    idx = agg_series.index
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
                    events.append(dict(duration_min=dur, energy_wh=cum_e, peak_w=pk,
                                       high_energy_event=(cum_e >= 350.0),
                                       hour_start=idx[ev_start].hour))
                state = 'IDLE'
    return pd.DataFrame(events) if events else pd.DataFrame(
        columns=['duration_min', 'energy_wh', 'peak_w', 'high_energy_event', 'hour_start'])


def house_sig_from_agg(agg_series):
    """Compute aggregate-event behavioral signature from agg_series only."""
    cyc = detect_agg_cycles(agg_series)  # pass Series — needed for timestamp-based hour_start
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
            if last_end >= 0:
                since_ev[i] = min(float(i - last_end), 240.0) / 240.0
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


# ── Model (must match checkpoint — NILM_LSTM_V7 from 03h_arnilm_v8.py) ────────

class NILM_LSTM_V7(nn.Module):
    def __init__(self, n_input=N_INPUT, hidden=256, n_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(n_input, hidden, n_layers,
                            batch_first=True, dropout=0.15)
        self.mu_head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Softplus())
        self.cls_head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(),
            nn.Linear(64, 1))

    def forward(self, agg, dyn, sig, hidden=None):
        B, T = agg.shape
        agg_n = (agg / MAX_AGG_W).unsqueeze(-1)
        sig_e = sig.unsqueeze(1).expand(-1, T, -1)
        inp   = torch.cat([agg_n, dyn, sig_e], dim=-1)
        out, hidden = self.lstm(inp, hidden)
        mu_raw    = self.mu_head(out).squeeze(-1)
        cls_logit = self.cls_head(out).squeeze(-1)
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
model = NILM_LSTM_V7().to(DEVICE)
model.load_state_dict(torch.load(CKPT_DIR / 'nilm_ar_lstm_v8.pt', map_location=DEVICE))
model.eval()
print('  Checkpoint loaded.')

config = json.loads((CKPT_DIR / 'nilm_ar_lstm_v8_config.json').read_text())
P_ON_THRESHOLD = config['p_on_threshold']
print(f'  p_on threshold (from training calibration): {P_ON_THRESHOLD}')

# ── House signatures ──────────────────────────────────────────────────────────
# Training houses: full Part 2 history (correct — they have sub-meter labels).
# H1 (test house): two signatures computed for the controlled A/B comparison:
#   sig_full  = full Part 2 history (control)
#   sig_cold  = first COLDSTART_DAYS only (cold-start deployment simulation)
# Both are evaluated on the SAME held-out period (post day-14) so the only
# variable between experiments A and B is the signature, not the eval window.

print('\nComputing house signatures ...')
house_sigs = {}
for h in sorted(wm_part2['house'].unique()):
    sub = wm_part2[wm_part2['house'] == h]
    house_sigs[h] = house_sig_from_agg(sub['Aggregate'])

h1_sub   = wm_part2[wm_part2['house'] == TEST_HOUSE]
h1_start = h1_sub.index[0]
h1_eval_start = h1_start + pd.Timedelta(days=COLDSTART_DAYS)

# Full-history H1 signature (control — Experiment A)
sig_h1_full = house_sigs[TEST_HOUSE]

# 14-day cold-start H1 signature (Experiment B)
h1_cal_sub  = h1_sub[h1_sub.index < h1_eval_start]
sig_h1_cold = house_sig_from_agg(h1_cal_sub['Aggregate'])

print(f'  H1 full-history sig: dur={sig_h1_full["sig_med_dur"]*180:.0f}min  '
      f'high_e={sig_h1_full["sig_agg_high_energy_frac"]:.0%}  '
      f'peak={sig_h1_full["sig_med_peak"]*MAX_WM_W:.0f}W')
print(f'  H1 14-day cold-start: dur={sig_h1_cold["sig_med_dur"]*180:.0f}min  '
      f'high_e={sig_h1_cold["sig_agg_high_energy_frac"]:.0%}  '
      f'peak={sig_h1_cold["sig_med_peak"]*MAX_WM_W:.0f}W')

# ── H1 eval window (identical for both experiments) ───────────────────────────

h1_eval = h1_sub[h1_sub.index >= h1_eval_start]
print(f'\nH1 eval window: {h1_eval_start.date()} → {h1_sub.index[-1].date()} '
      f'({len(h1_eval):,} timesteps = {len(h1_eval)/60/24:.1f} days)')
print(f'  Excluded first {COLDSTART_DAYS} days ({len(h1_cal_sub):,} rows) from scoring in both experiments.')

agg_eval = h1_eval['Aggregate'].values.astype(np.float32)
wm_eval  = h1_eval['WM'].values.astype(np.float32)
dyn_eval = build_dyn_covariates(h1_eval)

# Calibration window data — fed through LSTM for warm-up (no labels used)
agg_cal = h1_cal_sub['Aggregate'].values.astype(np.float32)
dyn_cal = build_dyn_covariates(h1_cal_sub)

# ── Inference helper with LSTM warm-up ────────────────────────────────────────
# Both experiments warm up using the same 14-day calibration window, so the
# LSTM hidden state at eval-window start is matched. The only variable between
# Exp A and B is the static household signature.

def infer_h1(sig_dict, label, chunk=10000):
    sig_vec = np.array([sig_dict[c] for c in SIG_COLS], dtype=np.float32)
    # Warm-up: process calibration window to build LSTM context
    hidden = None
    with torch.no_grad():
        for s in range(0, len(agg_cal), chunk):
            e  = min(s + chunk, len(agg_cal))
            ac = torch.FloatTensor(agg_cal[s:e]).unsqueeze(0).to(DEVICE)
            dc = torch.FloatTensor(dyn_cal[s:e]).unsqueeze(0).to(DEVICE)
            sc = torch.FloatTensor(sig_vec).unsqueeze(0).to(DEVICE)
            _, _, _, hidden = model(ac, dc, sc, hidden)
            hidden = tuple(hh.detach() for hh in hidden)
    print(f'  [{label}] warm-up done ({len(agg_cal):,} timesteps). Running eval ...')
    # Eval: carry warm-up hidden state into the scored window
    T_ = len(agg_eval)
    preds_raw_ = np.zeros(T_, np.float32)
    p_ons_     = np.zeros(T_, np.float32)
    with torch.no_grad():
        for s in range(0, T_, chunk):
            e  = min(s + chunk, T_)
            ac = torch.FloatTensor(agg_eval[s:e]).unsqueeze(0).to(DEVICE)
            dc = torch.FloatTensor(dyn_eval[s:e]).unsqueeze(0).to(DEVICE)
            sc = torch.FloatTensor(sig_vec).unsqueeze(0).to(DEVICE)
            yh, po, _, hidden = model(ac, dc, sc, hidden)
            preds_raw_[s:e] = yh[0].cpu().numpy()
            p_ons_[s:e]     = po[0].cpu().numpy()
            hidden = tuple(hh.detach() for hh in hidden)
    print(f'  [{label}] inference done.')
    return preds_raw_, p_ons_

print(f'\nRunning controlled A/B inference on H1 eval window (with LSTM warm-up) ...')
preds_raw_full, pons_full = infer_h1(sig_h1_full, 'Exp A: full-history sig')
preds_raw_cold, pons_cold = infer_h1(sig_h1_cold, 'Exp B: 14-day cold-start sig')

# Constraint violations — measure BEFORE clipping (cold-start experiment)
violations = np.clip(preds_raw_cold - agg_eval, 0, None)
viol_rate  = float((violations > 0).mean() * 100)
viol_mean  = float(violations.mean())
viol_max   = float(violations.max())
viol_mean_among_violating = viol_mean / (viol_rate / 100) if viol_rate > 0 else 0.0

preds_full_clipped = np.clip(preds_raw_full, 0, agg_eval)
preds_cold_clipped = np.clip(preds_raw_cold, 0, agg_eval)

# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(preds, p_ons_, wm_gt, threshold):
    true_on = (wm_gt >= ON_THRESH_W).astype(int)
    pred_on = (p_ons_ >= threshold).astype(int)
    mae_    = float(np.mean(np.abs(preds - wm_gt)))
    rmse_   = float(np.sqrt(np.mean((preds - wm_gt) ** 2)))
    on_m    = true_on == 1
    mae_on_ = float(np.mean(np.abs(preds[on_m] - wm_gt[on_m]))) if on_m.sum() > 0 else float('nan')
    f1_     = float(f1_score(true_on, pred_on, zero_division=0))
    prec_   = float(precision_score(true_on, pred_on, zero_division=0))
    rec_    = float(recall_score(true_on, pred_on, zero_division=0))
    true_kwh_ = float(wm_gt.sum() / 60000)
    pred_kwh_ = float(preds.sum() / 60000)
    ee_     = abs(pred_kwh_ - true_kwh_) / max(true_kwh_, 1e-6) * 100
    return dict(mae=round(mae_,2), rmse=round(rmse_,2), mae_on=round(mae_on_,2),
                f1=round(f1_,4), precision=round(prec_,4), recall=round(rec_,4),
                energy_err_pct=round(ee_,1))

m_full = compute_metrics(preds_full_clipped, pons_full, wm_eval, P_ON_THRESHOLD)
m_cold = compute_metrics(preds_cold_clipped, pons_cold, wm_eval, P_ON_THRESHOLD)

# ── Report ────────────────────────────────────────────────────────────────────

print('\n' + '='*72)
print('ARNILM V8 — Controlled A/B cold-start comparison')
print('Both experiments evaluated on identical timestamps (post-14-day window)')
print('='*72)
print(f'  Eval window: {h1_eval_start.date()} → {h1_sub.index[-1].date()}  '
      f'({len(h1_eval):,} timesteps = {len(h1_eval)/60/24:.1f} days)')
print(f'  WM ON fraction: {(wm_eval >= ON_THRESH_W).mean()*100:.2f}%')
print()
w = 24
print(f'  {"Metric":<{w}} {"Exp A: full-history sig":<{w}} {"Exp B: 14-day cold-start"}')
print(f'  {"-"*70}')
for k, label in [('mae','MAE'), ('rmse','RMSE'), ('mae_on','MAE (ON)'),
                 ('f1','F1'), ('precision','Precision'), ('recall','Recall'),
                 ('energy_err_pct','Energy error %')]:
    print(f'  {label:<{w}} {str(m_full[k]):<{w}} {m_cold[k]}')
print()
print('  Physical constraint violations (cold-start, before clipping):')
print(f'    Rate:             {viol_rate:.2f}% of timesteps')
print(f'    Mean (all ts):    {viol_mean:.4f} W')
print(f'    Mean (violating): {viol_mean_among_violating:.2f} W')
print(f'    Max:              {viol_max:.1f} W')
print(f'    After clipping:   0.00% / 0.0 W')
print('='*72)

results = {
    'model': 'ARNILM_V8_coldstart14d',
    'test_house': TEST_HOUSE,
    'coldstart_days': COLDSTART_DAYS,
    'eval_window_start': str(h1_eval_start.date()),
    'eval_timesteps': len(h1_eval),
    'experiment_A_full_history_sig': m_full,
    'experiment_B_coldstart_14d_sig': m_cold,
    'constraint_viol_rate_pct_before_clip': round(viol_rate, 4),
    'constraint_viol_mean_all_ts_W': round(viol_mean, 4),
    'constraint_viol_mean_violating_W': round(viol_mean_among_violating, 2),
    'constraint_viol_max_W_before_clip': round(viol_max, 2),
    'constraint_viol_after_clip': 0.0,
}
out = RES_DIR / 'metrics_ar_lstm_v8_coldstart.json'
out.write_text(json.dumps(results, indent=2))
print(f'\nH1 results saved → {out}')

# ── H7 signature sensitivity experiment (within-house temporal holdout) ───────
# H7 is a TRAINING house — this is NOT an unseen-household cold-start test.
# It is a within-house temporal holdout: training sequences exclude the last 4
# weeks; the model has seen H7's aggregate patterns but not this exact period.
# Purpose: isolate the effect of signature quality by comparing full-history
# vs 14-day signature on identical timestamps. H7 has clean Part 2 labels
# (869 cycles, 98% hot-wash) making it a reliable sensitivity benchmark.
# Do not use H7's result as evidence of cross-house cold-start generalisation;
# that evidence comes from H1 only.

print('\n' + '='*72)
print('H7 signature sensitivity experiment (within-house temporal holdout)')
print('NOTE: H7 is a training house — this measures signature sensitivity,')
print('      not cold-start generalisation to an unseen household.')
print('='*72)

H7 = 7
h7_sub = wm_part2[wm_part2['house'] == H7]

# Calibration / eval window: last 4 weeks (same as CAL_WEEKS logic in training)
CAL_WEEKS = 4
cal_len = min(CAL_WEEKS * 7 * 24 * 60, len(h7_sub) // 4)
h7_eval = h7_sub.iloc[-cal_len:]

# Signature A: full history (training used this)
sig_h7_full = house_sigs[H7]

# Signature B: 14-day cold-start (simulates deployment on a new house)
h7_cal_end = h7_sub.index[0] + pd.Timedelta(days=COLDSTART_DAYS)
h7_cal_sub = h7_sub[h7_sub.index < h7_cal_end]
sig_h7_cold = house_sig_from_agg(h7_cal_sub['Aggregate'])

print(f'  H7 eval window:      last {cal_len:,} timesteps ({cal_len/60/24:.0f} days)')
print(f'  Full-history sig:    dur={sig_h7_full["sig_med_dur"]*180:.0f}min  high_e={sig_h7_full["sig_agg_high_energy_frac"]:.0%}  peak={sig_h7_full["sig_med_peak"]*3000:.0f}W')
print(f'  14-day cold-start:   dur={sig_h7_cold["sig_med_dur"]*180:.0f}min  high_e={sig_h7_cold["sig_agg_high_energy_frac"]:.0%}  peak={sig_h7_cold["sig_med_peak"]*3000:.0f}W')

agg_h7 = h7_eval['Aggregate'].values.astype(np.float32)
wm_h7  = h7_eval['WM'].values.astype(np.float32)
dyn_h7 = build_dyn_covariates(h7_eval)

def run_inference(agg, dyn, sig_dict, chunk=10000):
    sig_vec = np.array([sig_dict[c] for c in SIG_COLS], dtype=np.float32)
    T_ = len(agg)
    preds = np.zeros(T_, np.float32); p_ons_ = np.zeros(T_, np.float32)
    hidden = None
    with torch.no_grad():
        for s in range(0, T_, chunk):
            e = min(s + chunk, T_)
            ac = torch.FloatTensor(agg[s:e]).unsqueeze(0).to(DEVICE)
            dc = torch.FloatTensor(dyn[s:e]).unsqueeze(0).to(DEVICE)
            sc = torch.FloatTensor(sig_vec).unsqueeze(0).to(DEVICE)
            yh, po, _, hidden = model(ac, dc, sc, hidden)
            preds[s:e] = yh[0].cpu().numpy()
            p_ons_[s:e] = po[0].cpu().numpy()
            hidden = tuple(hh.detach() for hh in hidden)
    return np.clip(preds, 0, agg), p_ons_

def score(preds, p_ons_, wm_gt, agg_gt, threshold):
    true_on = (wm_gt >= ON_THRESH_W).astype(int)
    pred_on = (p_ons_ >= threshold).astype(int)
    mae_  = float(np.mean(np.abs(preds - wm_gt)))
    f1_   = float(f1_score(true_on, pred_on, zero_division=0))
    on_m  = true_on == 1
    mae_on_ = float(np.mean(np.abs(preds[on_m] - wm_gt[on_m]))) if on_m.sum() > 0 else float('nan')
    true_kwh_ = float(wm_gt.sum() / 60000)
    pred_kwh_ = float(preds.sum() / 60000)
    ee_ = abs(pred_kwh_ - true_kwh_) / max(true_kwh_, 1e-6) * 100
    return dict(mae=round(mae_,2), f1=round(f1_,3), mae_on=round(mae_on_,1), energy_err=round(ee_,1))

preds_full, pons_full = run_inference(agg_h7, dyn_h7, sig_h7_full)
preds_cold, pons_cold = run_inference(agg_h7, dyn_h7, sig_h7_cold)

sc_full = score(preds_full, pons_full, wm_h7, agg_h7, P_ON_THRESHOLD)
sc_cold = score(preds_cold, pons_cold, wm_h7, agg_h7, P_ON_THRESHOLD)

print(f'\n  {"Metric":<18} {"Full-history sig":<22} {"14-day cold-start sig"}')
print(f'  {"-"*58}')
for k in ['mae', 'f1', 'mae_on', 'energy_err']:
    print(f'  {k:<18} {str(sc_full[k]):<22} {sc_cold[k]}')

h7_results = {'full_sig': sc_full, 'coldstart_sig': sc_cold,
              'h7_14d_sig': {k: round(v,4) if isinstance(v,float) else v
                             for k,v in sig_h7_cold.items()}}
(RES_DIR / 'metrics_ar_lstm_v8_coldstart_h7demo.json').write_text(json.dumps(h7_results, indent=2))
print(f'\nH7 demo results saved → {RES_DIR}/metrics_ar_lstm_v8_coldstart_h7demo.json')
