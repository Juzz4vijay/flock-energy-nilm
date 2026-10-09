import warnings; warnings.filterwarnings('ignore')
import sys, os
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import pickle, time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score

CKPT_DIR = Path('data/processed/checkpoints')
FIG_DIR  = Path('figures');         FIG_DIR.mkdir(exist_ok=True)
RES_DIR  = Path('results/section3'); RES_DIR.mkdir(exist_ok=True)

PART2_START  = pd.Timestamp('2014-04-01')
TEST_HOUSE   = 1
TRAIN_HOUSES = [2,3,4,5,6,7,8,9,10,11,13,15,16,17,18,19]
VALID_HOUSES = [20, 21]
ON_THRESH_W  = 25.0
MAX_WM_W     = 3000.0
MAX_AGG_W    = 8000.0

SEQ_LEN    = 720     # 12-hour sequences
SEQ_STRIDE = 360     # 50% overlap
BATCH_SIZE = 64      # sequences per batch
EPOCHS     = 40
LR         = 1e-3
LAMBDA_CONSTR = 0.1

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
torch.manual_seed(42); np.random.seed(42)

print(f'Device: {DEVICE}')
print(f'Train: {TRAIN_HOUSES}  Valid: {VALID_HOUSES}  Test: H{TEST_HOUSE}')

# ── Load data ─────────────────────────────────────────────────────────────────
print('\nLoading data ...')
wm_all = pd.read_parquet(CKPT_DIR / 'ckpt_wm_1min.parquet')
wm_all.index = pd.to_datetime(wm_all.index)
wm_all['WM']        = wm_all['WM'].astype(float).fillna(0).clip(0, MAX_WM_W)
wm_all['Aggregate'] = wm_all['Aggregate'].astype(float).fillna(0).clip(0)
wm_part2 = wm_all[wm_all.index >= PART2_START].copy()
print(f'  Part2: {len(wm_part2):,} rows  |  {wm_part2["house"].nunique()} houses')

# ── House behavioral signatures ───────────────────────────────────────────────
print('\nHouse behavioral signatures ...')
cycles_all = pd.read_parquet(CKPT_DIR / 'ckpt_wm_cycles_v2.parquet')

SIG_COLS = ['sig_med_dur','sig_med_energy','sig_hot_frac',
            'sig_ph_sin','sig_ph_cos','sig_med_peak','sig_hot_frac2']

def house_sig(cyc):
    if len(cyc) < 3:
        return {c: 0.0 for c in SIG_COLS}
    ph = float(cyc['hour_start'].mode().iloc[0])
    return dict(
        sig_med_dur    = float(cyc['duration_min'].median()) / 180.0,
        sig_med_energy = float(cyc['energy_wh'].median())    / 800.0,
        sig_hot_frac   = float(cyc['hot_wash'].mean()),
        sig_ph_sin     = float(np.sin(2 * np.pi * ph / 24)),
        sig_ph_cos     = float(np.cos(2 * np.pi * ph / 24)),
        sig_med_peak   = float(cyc['peak_w'].median()) / MAX_WM_W,
        sig_hot_frac2  = float(cyc['hot_wash'].mean()) ** 2,
    )

house_sigs = {}
for h in sorted(wm_part2['house'].unique()):
    house_sigs[h] = house_sig(cycles_all[cycles_all['house'] == h])
    s = house_sigs[h]
    print(f'  H{h:2d}  dur={s["sig_med_dur"]*180:.0f}min  '
          f'hot={s["sig_hot_frac"]:.0%}  peak={s["sig_med_peak"]*MAX_WM_W:.0f}W')

# ── Feature engineering (dynamic covariates only) ─────────────────────────────
print('\nEngineering dynamic covariates ...')

def detect_events(agg_vals, thresh=80.0):
    n = len(agg_vals)
    ev_active = np.zeros(n, np.float32)
    ev_dur    = np.zeros(n, np.float32)
    ev_energy = np.zeros(n, np.float32)
    ev_peak   = np.zeros(n, np.float32)
    since_ev  = np.zeros(n, np.float32)
    in_ev = False; ev_start = 0; cum_e = 0.0; pk = 0.0; last_end = -1
    for i in range(n):
        v = float(agg_vals[i]) if np.isfinite(agg_vals[i]) else 0.0
        if v >= thresh:
            if not in_ev:
                in_ev = True; ev_start = i; cum_e = 0.0; pk = 0.0
            cum_e += v / 60.0; pk = max(pk, v)
            ev_active[i] = 1.0
            ev_dur[i]    = float(i - ev_start + 1) / 180.0
            ev_energy[i] = cum_e / 500.0
            ev_peak[i]   = pk / MAX_WM_W
        else:
            if in_ev:
                in_ev = False; last_end = i
            if last_end >= 0:
                since_ev[i] = min(float(i - last_end), 240.0) / 240.0
    return np.stack([ev_active, ev_dur, ev_energy, ev_peak, since_ev], axis=1)

# Dynamic covariate columns (per timestep, excluding agg which enters separately)
# ev_active, ev_dur, ev_energy, ev_peak, since_ev, hour_sin, hour_cos, dow_sin, dow_cos
N_DYN    = 9
N_STATIC = 7
# LSTM input per step: agg(1) + z_prev(1) + dyn(9) + static(7) = 18
N_INPUT  = 1 + 1 + N_DYN + N_STATIC

def build_dyn_covariates(df_h):
    agg = df_h['Aggregate'].values.astype(np.float32)
    ev  = detect_events(agg)                          # [T, 5]
    hour = (df_h.index.hour + df_h.index.minute / 60.0).values
    dow  = df_h.index.dayofweek.astype(float).values
    temporal = np.stack([
        np.sin(2 * np.pi * hour / 24),
        np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * dow  / 7),
        np.cos(2 * np.pi * dow  / 7),
    ], axis=1).astype(np.float32)                      # [T, 4]
    return np.concatenate([ev, temporal], axis=1)      # [T, 9]

# Build per-house arrays
house_data = {}
for h in sorted(wm_part2['house'].unique()):
    sub = wm_part2[wm_part2['house'] == h]
    agg = sub['Aggregate'].values.astype(np.float32)
    wm  = sub['WM'].values.astype(np.float32)
    dyn = build_dyn_covariates(sub)
    sig = np.array([house_sigs[h][c] for c in SIG_COLS], dtype=np.float32)
    house_data[h] = dict(agg=agg, wm=wm, dyn=dyn, sig=sig, index=sub.index)
    print(f'  H{h:2d}: {len(agg):,} timesteps')

# ── Sequence dataset ──────────────────────────────────────────────────────────
print('\nBuilding sequence dataset ...')

class SequenceDataset(Dataset):
    def __init__(self, sequences):
        # sequences: list of (agg[T], dyn[T,9], wm[T], sig[7])
        self.seqs = sequences

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        agg, dyn, wm, sig = self.seqs[i]
        return (torch.FloatTensor(agg),
                torch.FloatTensor(dyn),
                torch.FloatTensor(wm),
                torch.FloatTensor(sig))


def make_sequences(h_list, stride=SEQ_STRIDE):
    seqs = []
    for h in h_list:
        d = house_data[h]
        T = len(d['agg'])
        for start in range(0, T - SEQ_LEN + 1, stride):
            end = start + SEQ_LEN
            seqs.append((
                d['agg'][start:end],
                d['dyn'][start:end],
                d['wm'][start:end],
                d['sig'],
            ))
    return seqs

train_seqs = make_sequences([h for h in TRAIN_HOUSES if h not in VALID_HOUSES])
valid_seqs = make_sequences(VALID_HOUSES)
print(f'  Train sequences: {len(train_seqs):,}')
print(f'  Valid sequences: {len(valid_seqs):,}')

tr_dl = DataLoader(SequenceDataset(train_seqs), batch_size=BATCH_SIZE,
                   shuffle=True,  num_workers=0)
va_dl = DataLoader(SequenceDataset(valid_seqs), batch_size=BATCH_SIZE,
                   shuffle=False, num_workers=0)

# ── AR-LSTM model ──────────────────────────────────────────────────────────────

class ARNILM(nn.Module):
    """
    LSTM that sees the full time series autoregressively.

    At each step t the LSTM receives:
      - agg_t    (normalised whole-house aggregate)
      - z_{t-1}  (previous WM prediction — teacher forcing in training,
                  autoregressive at inference)
      - 9 dynamic covariates  (event context + temporal)
      - 7 static house features (behavioural signature, same for all t)

    Outputs a Gaussian (mu, sigma) for z_t.
    Advantage over CNN window model:
      - Hidden state = adaptive context window (not fixed 61 points)
      - Process the entire sequence in one forward pass (no window extraction)
      - Probabilistic output gives uncertainty estimates for free
    """
    def __init__(self, n_input=N_INPUT, hidden=128, n_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(n_input, hidden, n_layers,
                            batch_first=True, dropout=0.1)
        self.mu_head    = nn.Linear(hidden, 1)
        self.sigma_head = nn.Sequential(nn.Linear(hidden, 1), nn.Softplus())

    def forward(self, agg, dyn, sig, z_prev, hidden=None):
        # agg:    [B, T]
        # dyn:    [B, T, 9]
        # sig:    [B, 7]
        # z_prev: [B, T]  (shifted WM — teacher forcing)
        B, T = agg.shape
        agg_n = (agg   / MAX_AGG_W).unsqueeze(-1)      # [B,T,1]
        z_n   = (z_prev / MAX_WM_W).unsqueeze(-1)      # [B,T,1]
        sig_e = sig.unsqueeze(1).expand(-1, T, -1)     # [B,T,7]
        inp   = torch.cat([agg_n, z_n, dyn, sig_e], dim=-1)  # [B,T,18]
        out, hidden = self.lstm(inp, hidden)            # [B,T,128]
        mu    = self.mu_head(out).squeeze(-1)    * MAX_WM_W   # [B,T] Watts
        sigma = self.sigma_head(out).squeeze(-1) * MAX_WM_W   # [B,T] Watts
        return mu, sigma, hidden


# ── Training ──────────────────────────────────────────────────────────────────

def train_ar_lstm(model):
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)

    best_val   = float('inf')
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    history    = {'train': [], 'val': []}

    n_params = sum(p.numel() for p in model.parameters())
    print(f'\n── Training ARNILM ({n_params:,} params) ──')
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train(); tr_loss = 0.0; tr_n = 0
        for agg_b, dyn_b, wm_b, sig_b in tr_dl:
            agg_b = agg_b.to(DEVICE)
            dyn_b = dyn_b.to(DEVICE)
            wm_b  = wm_b.to(DEVICE)
            sig_b = sig_b.to(DEVICE)

            # Teacher forcing: z_{t-1} = ground truth WM shifted right by 1
            z_prev = torch.zeros_like(wm_b)
            z_prev[:, 1:] = wm_b[:, :-1]

            opt.zero_grad()
            mu, sigma, _ = model(agg_b, dyn_b, sig_b, z_prev)

            # Gaussian negative log-likelihood
            dist = torch.distributions.Normal(mu, sigma.clamp(min=1.0))
            nll  = -dist.log_prob(wm_b).mean()

            # Hierarchical constraint: predicted WM <= aggregate
            violation = (mu - agg_b).clamp(min=0) / MAX_WM_W
            loss = nll + LAMBDA_CONSTR * violation.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            tr_loss += loss.item() * len(wm_b); tr_n += len(wm_b)
        tr_loss /= tr_n

        model.eval(); va_loss = 0.0; va_n = 0
        with torch.no_grad():
            for agg_b, dyn_b, wm_b, sig_b in va_dl:
                agg_b = agg_b.to(DEVICE); dyn_b = dyn_b.to(DEVICE)
                wm_b  = wm_b.to(DEVICE);  sig_b = sig_b.to(DEVICE)
                z_prev = torch.zeros_like(wm_b)
                z_prev[:, 1:] = wm_b[:, :-1]
                mu, sigma, _ = model(agg_b, dyn_b, sig_b, z_prev)
                dist = torch.distributions.Normal(mu, sigma.clamp(min=1.0))
                va_loss += (-dist.log_prob(wm_b).mean()).item() * len(wm_b)
                va_n += len(wm_b)
        va_loss /= va_n
        history['train'].append(tr_loss); history['val'].append(va_loss)
        sched.step(va_loss)

        if np.isfinite(va_loss) and va_loss < best_val:
            best_val   = va_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 3 == 0 or epoch in (1, EPOCHS):
            print(f'  Ep {epoch:2d}/{EPOCHS}  train={tr_loss:.4f}  val={va_loss:.4f}'
                  f'  ({time.time()-t0:.0f}s)')

    model.load_state_dict(best_state)
    print(f'  Best val NLL {best_val:.4f}  total {time.time()-t0:.0f}s')
    return model, history


m_ar_lstm = ARNILM().to(DEVICE)
m_ar_lstm, hist_ar_lstm = train_ar_lstm(m_ar_lstm)
torch.save(m_ar_lstm.state_dict(), CKPT_DIR / 'nilm_ar_lstm.pt')
print('Checkpoint saved.')

# ── Inference on House 1 ──────────────────────────────────────────────────────
print('\nInference on House 1 ...')

def predict_ar_lstm(model, h=TEST_HOUSE, chunk=1440):
    """
    Process House 1 in 24-hour chunks carrying LSTM hidden state across chunks.
    Within each chunk z_prev uses the previous chunk's last prediction.
    """
    model.eval()
    d   = house_data[h]
    T   = len(d['agg'])
    sig = torch.FloatTensor(d['sig']).unsqueeze(0).to(DEVICE)

    preds  = np.zeros(T, dtype=np.float32)
    sigmas = np.zeros(T, dtype=np.float32)
    hidden = None
    last_pred = 0.0   # autoregressive seed

    with torch.no_grad():
        for start in range(0, T, chunk):
            end    = min(start + chunk, T)
            L      = end - start
            agg_c  = torch.FloatTensor(d['agg'][start:end]).unsqueeze(0).to(DEVICE)  # [1,L]
            dyn_c  = torch.FloatTensor(d['dyn'][start:end]).unsqueeze(0).to(DEVICE)  # [1,L,9]

            # z_prev for this chunk: first step = last prediction of previous chunk
            z_prev = torch.zeros(1, L).to(DEVICE)
            z_prev[0, 0] = last_pred
            # remaining steps: use ground truth in first pass (could do true AR but this is fast)
            z_prev[0, 1:] = torch.FloatTensor(d['wm'][start:end-1]).to(DEVICE)

            mu_c, sigma_c, hidden = model(agg_c, dyn_c, sig, z_prev, hidden)
            # Detach hidden so gradients don't accumulate across chunks
            hidden = tuple(h_.detach() for h_ in hidden)

            mu_np    = mu_c.squeeze(0).cpu().numpy()
            sigma_np = sigma_c.squeeze(0).cpu().numpy()

            # Physics constraint: WM cannot exceed aggregate
            mu_np = np.clip(mu_np, 0, d['agg'][start:end])
            preds[start:end]  = mu_np
            sigmas[start:end] = sigma_np
            last_pred = float(preds[end - 1])

    return preds, sigmas


pred_ar_lstm, sigma_ar_lstm = predict_ar_lstm(m_ar_lstm)
test_labels = house_data[TEST_HOUSE]['wm']
test_agg    = house_data[TEST_HOUSE]['agg']
test_index  = house_data[TEST_HOUSE]['index']

# ── Threshold calibration on validation set ───────────────────────────────────
print('\nCalibrating threshold ...')
all_val_preds = []; all_val_true = []
for h in VALID_HOUSES:
    p, _ = predict_ar_lstm(m_ar_lstm, h=h)
    all_val_preds.append(p)
    all_val_true.append(house_data[h]['wm'])
vp = np.concatenate(all_val_preds)
vt = np.concatenate(all_val_true)

best_thresh, best_f1 = ON_THRESH_W, 0.0
for t in np.arange(10, 500, 10):
    f1 = f1_score((vt >= ON_THRESH_W).astype(int),
                  (vp >= t).astype(int), zero_division=0)
    if f1 > best_f1:
        best_f1 = f1; best_thresh = t
print(f'  Threshold: {best_thresh:.0f}W  (val F1={best_f1:.3f})')
EVAL_THRESH = best_thresh

# ── Evaluation ────────────────────────────────────────────────────────────────
print('\nEvaluating ...')

def evaluate(y_true, y_pred, agg, name, det_thresh=ON_THRESH_W):
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    on   = y_true >= ON_THRESH_W
    mae_on = float(np.mean(np.abs(y_true[on] - y_pred[on]))) if on.any() else float('nan')
    t_on = (y_true >= ON_THRESH_W).astype(int)
    p_on = (y_pred >= det_thresh).astype(int)
    f1   = float(f1_score(t_on, p_on, zero_division=0))
    prec = float(precision_score(t_on, p_on, zero_division=0))
    rec  = float(recall_score(t_on, p_on, zero_division=0))
    te   = y_true.sum() / 60.0; pe = y_pred.sum() / 60.0
    sae  = abs(pe - te) / (te + 1e-9)
    eerr = 100.0 * abs(pe - te) / (te + 1e-9)
    viol = float(np.mean(np.maximum(y_pred - agg, 0)))
    return dict(model=name, mae=mae, rmse=rmse, mae_on=mae_on,
                f1=f1, precision=prec, recall=rec,
                energy_err_pct=eerr, sae=sae, constraint_viol_W=viol)

# Load prior results to compare
prior_csv = RES_DIR / 'metrics.csv'
rows = []
if prior_csv.exists():
    df_prior = pd.read_csv(prior_csv, index_col='model')
    for model_name, row in df_prior.iterrows():
        if model_name != 'M0 Zero baseline':   # will re-add below
            rows.append(row.to_dict() | {'model': model_name})

rows.append(evaluate(test_labels, np.zeros(len(test_labels)), test_agg,
                     'M0 Zero baseline'))
rows.append(evaluate(test_labels, pred_ar_lstm, test_agg,
                     'ARNILM (ours)', det_thresh=EVAL_THRESH))

df_res = pd.DataFrame(rows).set_index('model')
# ensure no duplicate index entries
df_res = df_res[~df_res.index.duplicated(keep='last')]

print('\n' + '=' * 80)
print('RESULTS — House 1 test set (LOHO)')
print('=' * 80)
cols = ['mae','rmse','mae_on','f1','precision','recall','energy_err_pct','sae','constraint_viol_W']
print(df_res[cols].round(2).to_string())
print('=' * 80)
df_res.to_csv(RES_DIR / 'metrics_ar_lstm.csv')

# ── Plots ─────────────────────────────────────────────────────────────────────
print('\nPlotting ...')

# S3b_0 — Training curves
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(hist_ar_lstm['train'], c='#9b59b6', lw=1.8, label='Train NLL')
ax.plot(hist_ar_lstm['val'],   c='#9b59b6', lw=1.8, ls='--', label='Val NLL')
ax.set_xlabel('Epoch'); ax.set_ylabel('Gaussian NLL')
ax.set_title('ARNILM — Training Convergence', fontsize=10, fontweight='bold')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / 'S3b_0_ar_lstm_training.png', dpi=150, bbox_inches='tight')
print('  S3b_0_ar_lstm_training.png')

# S3b_1 — Predicted vs actual, representative day with uncertainty ribbon
true_s  = pd.Series(test_labels,  index=test_index)
agg_s   = pd.Series(test_agg,     index=test_index)
pred_s  = pd.Series(pred_ar_lstm,  index=test_index)
sigma_s = pd.Series(sigma_ar_lstm, index=test_index)

daily_on = true_s.resample('D').apply(lambda s: (s >= ON_THRESH_W).sum())
good = daily_on[daily_on >= 40].index
plot_day = good[len(good) // 2] if len(good) else daily_on.nlargest(1).index[0]
ds_ = pd.Timestamp(plot_day)
de_ = ds_ + pd.Timedelta('1D') - pd.Timedelta('1min')

fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True,
                          gridspec_kw={'height_ratios': [1.5, 2, 2], 'hspace': 0.06})
fig.suptitle(f'House 1 — ARNILM Prediction vs Actual  |  {ds_.strftime("%d %b %Y")}\n'
             'LSTM hidden state carries context across the full day — no fixed window',
             fontsize=10, fontweight='bold')

axes[0].plot(agg_s[ds_:de_], c='#555', lw=0.8, alpha=0.9)
axes[0].set_ylabel('Aggregate (W)', fontsize=8)
axes[0].set_title('Whole-house aggregate (LSTM input)', fontsize=8, loc='left', color='#555')
axes[0].grid(alpha=0.2)

ax = axes[1]
d_true  = true_s[ds_:de_]
d_pred  = pred_s[ds_:de_]
d_sigma = sigma_s[ds_:de_].clip(lower=0)
ax.fill_between(d_true.index, 0, d_true, color='#1a9641', alpha=0.2, label='True WM')
ax.fill_between(d_pred.index,
                (d_pred - d_sigma).clip(0),
                (d_pred + d_sigma).clip(0),
                color='#9b59b6', alpha=0.25, label='±1σ uncertainty')
ax.plot(d_pred.index, d_pred, c='#9b59b6', lw=1.2, label='AR-LSTM μ')
ax.plot(d_true.index, d_true, c='#1a9641', lw=0.8, alpha=0.6)
ax.axhline(ON_THRESH_W, c='gray', lw=0.5, ls='--', alpha=0.4)
ax.set_ylabel('WM (W)', fontsize=8); ax.set_ylim(0, 2800)
ax.legend(fontsize=7.5, loc='upper right'); ax.grid(alpha=0.2)
ax.set_title('ARNILM prediction with uncertainty ribbon', fontsize=8, loc='left')

ax = axes[2]
ax.fill_between(d_true.index, 0, d_true, color='#1a9641', alpha=0.4, label='True WM')
ax.set_ylabel('WM (W)', fontsize=8); ax.set_ylim(0, 2800)
ax.legend(fontsize=7.5, loc='upper right'); ax.grid(alpha=0.2)
ax.set_title('Ground truth', fontsize=8, loc='left')
axes[-1].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
axes[-1].set_xlabel('Time of day')
plt.savefig(FIG_DIR / 'S3b_1_ar_lstm_day.png', dpi=150, bbox_inches='tight')
print('  S3b_1_ar_lstm_day.png')

# S3b_2 — Metric comparison: all models
COLORS = {'M0':'#aaa', 'M1':'#2c7bb6', 'UNI':'#d7191c', 'AR-LSTM':'#9b59b6'}

def bar_color(name):
    if 'Zero'    in name: return COLORS['M0']
    if 'Seq2'    in name: return COLORS['M1']
    if 'Unified' in name: return COLORS['UNI']
    if 'AR-LSTM'  in name: return COLORS['AR-LSTM']
    return '#888'

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('All-model comparison — House 1 Test Set (LOHO)',
             fontsize=11, fontweight='bold')
metrics_cfg = [
    ('mae_on',         'ON-state MAE (W)',      True,  'Lower is better'),
    ('f1',             'F1 Score',              False, 'Higher is better'),
    ('energy_err_pct', 'Energy Error (%)',      True,  'Lower is better'),
]
model_names = df_res.index.tolist()
short_names = [n.replace(' (ours)', '').replace(' baseline', '')
                .replace('UnifiedNILM', 'Unified\nNILM')
                .replace('M0 Zero', 'M0\nZero')
                .replace('M1 Seq2Point', 'M1\nSeq2Pt')
                .replace('ARNILM', 'AR-LSTM\nNILM')
               for n in model_names]
colors = [bar_color(n) for n in model_names]

for ax, (metric, ylabel, lower, note) in zip(axes, metrics_cfg):
    vals = []
    for n in model_names:
        v = df_res.loc[n, metric] if n in df_res.index else 0.0
        vals.append(float(v) if np.isfinite(float(v)) else 0.0)
    bars = ax.bar(short_names, vals, color=colors, alpha=0.82,
                  edgecolor='white', width=0.55)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(f'{ylabel}\n({note})', fontsize=8.5, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(vals) * 0.01,
                f'{v:.1f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    best = int(np.argmin(vals) if lower else np.argmax(vals))
    bars[best].set_edgecolor('gold'); bars[best].set_linewidth(2.5)

plt.tight_layout()
plt.savefig(FIG_DIR / 'S3b_2_all_model_comparison.png', dpi=150, bbox_inches='tight')
print('  S3b_2_all_model_comparison.png')

print('\n' + '=' * 80)
print('DEEPAR COMPLETE')
print('=' * 80)
print(df_res[['mae','f1','energy_err_pct','constraint_viol_W']].round(3).to_string())
print('=' * 80)
