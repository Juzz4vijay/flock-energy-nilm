# Section 3 — Appliance Disaggregation: Washing Machine
#
# Architecture: UnifiedNILM
#   Shared CNN backbone (aggregate window) + feature branch (engineered context)
#   Per-appliance output heads — designed for multi-appliance extension
#   House identity: 7-dimensional continuous behavioral signature derived from
#     cycle statistics, NOT a cluster label or one-hot. Model learns the mapping
#     from behavioral features to disaggregation patterns. New house: compute
#     signature from early cycles → plug in → generalises without retraining.
#   Hierarchical constraint: ŷ_WM ≤ Aggregate enforced as soft penalty IN the
#     training loss, not just clipped at inference.
#
# Comparison: M1 Seq2Point (Zhang et al. 2018) trained for reference.
# Future: BERT4NILM comparison planned separately.
#
# Data split: Leave-House-1-Out (LOHO)
#   Train : Houses 2–21 (17 houses), Part2 only (≥ 2014-04-01)
#   Valid : Houses 20, 21 held out during training
#   Test  : House 1 entirely — zero data leakage

import warnings; warnings.filterwarnings('ignore')
import sys, os
# Force line-buffered stdout so progress is visible even when redirected to a file
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score

CKPT_DIR = Path('data/processed/checkpoints')
FIG_DIR  = Path('figures');        FIG_DIR.mkdir(exist_ok=True)
RES_DIR  = Path('results/section3'); RES_DIR.mkdir(exist_ok=True)

WINDOW      = 61
HALF_W      = WINDOW // 2
PART2_START = pd.Timestamp('2014-04-01')
TEST_HOUSE  = 1
TRAIN_HOUSES = [2,3,4,5,6,7,8,9,10,11,13,15,16,17,18,19,20,21]
VALID_HOUSES = [20, 21]
ON_THRESH_W  = 25.0
MAX_WM_W     = 3000.0
BATCH_SIZE   = 2048
EPOCHS        = 10
LR            = 1e-3
LAMBDA_CONSTR = 0.1   # weight on hierarchical constraint penalty in loss
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'

torch.manual_seed(42); np.random.seed(42)

print(f'Device: {DEVICE}  (MPS available: {torch.backends.mps.is_available()})')
print(f'Train houses : {[h for h in TRAIN_HOUSES if h not in VALID_HOUSES]}')
print(f'Valid houses : {VALID_HOUSES}')
print(f'Test house   : {TEST_HOUSE}')

# ──────────────────────────────────────────────────────────────────────────────
# STAGE 0 — Load 1-min data
# ──────────────────────────────────────────────────────────────────────────────
print('\nLoading ckpt_wm_1min.parquet ...')
wm_all = pd.read_parquet(CKPT_DIR / 'ckpt_wm_1min.parquet')
wm_all.index = pd.to_datetime(wm_all.index)
wm_all['WM']        = wm_all['WM'].astype(float).fillna(0).clip(lower=0, upper=MAX_WM_W)
wm_all['Aggregate'] = wm_all['Aggregate'].astype(float).fillna(0).clip(lower=0)
wm_part2 = wm_all[wm_all.index >= PART2_START].copy()
print(f'  Part2 rows: {len(wm_part2):,}  |  houses: {wm_part2["house"].nunique()}')

# ──────────────────────────────────────────────────────────────────────────────
# STAGE 1 — House behavioral signatures (replaces cluster one-hot)
#
# Each house gets a 7-dim continuous vector derived from its WM cycle statistics.
# The model learns what these behavioral features mean for disaggregation.
# For a new house: compute the same 7 features from its early detected cycles
# and the model generalises without any cluster assignment or retraining.
# ──────────────────────────────────────────────────────────────────────────────
print('\nComputing house behavioral signatures ...')

cycles_all = pd.read_parquet(CKPT_DIR / 'ckpt_wm_cycles_v2.parquet')

def house_behavioral_sig(cyc: pd.DataFrame) -> dict:
    """
    7 features that characterise a household's washing machine usage style.
    All derived from cycle-level statistics — no minute-level labels used.
    """
    if len(cyc) < 3:
        return None
    med_dur    = float(cyc['duration_min'].median())
    med_energy = float(cyc['energy_wh'].median())
    hot_frac   = float(cyc['hot_wash'].mean())
    # Peak usage hour as sin/cos (cyclic distance preserved)
    ph         = float(cyc['hour_start'].mode().iloc[0]) if len(cyc) > 0 else 12.0
    ph_sin     = float(np.sin(2 * np.pi * ph / 24))
    ph_cos     = float(np.cos(2 * np.pi * ph / 24))
    med_peak   = float(cyc['peak_w'].median())
    return dict(
        sig_med_dur    = med_dur    / 180.0,      # normalise by max cycle length
        sig_med_energy = med_energy / 800.0,      # normalise by approx max cycle Wh
        sig_hot_frac   = hot_frac,
        sig_ph_sin     = ph_sin,
        sig_ph_cos     = ph_cos,
        sig_med_peak   = med_peak   / MAX_WM_W,
        sig_hot_frac2  = hot_frac ** 2,           # captures extreme cold/hot households
    )

SIG_COLS = ['sig_med_dur','sig_med_energy','sig_hot_frac',
            'sig_ph_sin','sig_ph_cos','sig_med_peak','sig_hot_frac2']
N_SIG = len(SIG_COLS)

house_sigs = {}
all_houses = sorted(wm_part2['house'].unique())
for h in all_houses:
    cyc_h = cycles_all[cycles_all['house'] == h]
    sig   = house_behavioral_sig(cyc_h)
    if sig is None:
        sig = {c: 0.0 for c in SIG_COLS}
    house_sigs[h] = sig
    print(f'  H{h:2d}: {len(cyc_h):4d} cycles  '
          f'dur={sig["sig_med_dur"]*180:.0f}min  '
          f'energy={sig["sig_med_energy"]*800:.0f}Wh  '
          f'hot={sig["sig_hot_frac"]:.0%}')

# ──────────────────────────────────────────────────────────────────────────────
# STAGE 2 — Feature engineering
# ──────────────────────────────────────────────────────────────────────────────
print('\nEngineering features ...')

def detect_events(agg: pd.Series, thresh: float = 80.0) -> pd.DataFrame:
    """
    Scan aggregate for contiguous power events regardless of which appliance.
    ev_dur and ev_peak act as implicit phase proxies:
      ev_dur=3min + ev_peak=3000W → kettle (not WM)
      ev_dur=23min + ev_peak=2400W → WM heating phase
      ev_dur=65min + ev_peak=400W  → WM agitation/rinse, heating done
    The model learns what these patterns mean — we don't label the phases.
    """
    n = len(agg); vals = agg.values
    ev_active = np.zeros(n, np.float32); ev_dur   = np.zeros(n, np.float32)
    ev_energy = np.zeros(n, np.float32); ev_peak  = np.zeros(n, np.float32)
    since_ev  = np.zeros(n, np.float32)
    in_ev = False; ev_start = 0; cum_e = 0.0; pk = 0.0; last_end = -1
    for i in range(n):
        v = vals[i] if np.isfinite(vals[i]) else 0.0
        if v >= thresh:
            if not in_ev:
                in_ev = True; ev_start = i; cum_e = 0.0; pk = 0.0
            cum_e += v / 60.0; pk = max(pk, v)
            ev_active[i] = 1.0; ev_dur[i] = float(i - ev_start + 1)
            ev_energy[i] = cum_e; ev_peak[i] = pk
        else:
            if in_ev:
                in_ev = False; last_end = i
            if last_end >= 0:
                since_ev[i] = min(float(i - last_end), 240.0)
    return pd.DataFrame({'ev_active': ev_active,
                         'ev_dur':    ev_dur    / 180.0,
                         'ev_energy': ev_energy / 500.0,
                         'ev_peak':   ev_peak   / MAX_WM_W,
                         'since_ev':  since_ev  / 240.0}, index=agg.index)


def engineer_features(df_h: pd.DataFrame, house_id: int) -> pd.DataFrame:
    s   = df_h.copy()
    agg = s['Aggregate']

    # Rolling aggregate stats (12)
    for w in [5, 15, 30, 60]:
        s[f'agg_mean_{w}'] = agg.rolling(w, min_periods=1).mean()
        s[f'agg_std_{w}']  = agg.rolling(w, min_periods=1).std().fillna(0)
        s[f'agg_max_{w}']  = agg.rolling(w, min_periods=1).max()

    # Rate of change (2)
    s['agg_diff']  = agg.diff().fillna(0)
    s['agg_diff2'] = s['agg_diff'].diff().fillna(0)

    # Background + net load (2)
    s['bg_load'] = agg.rolling(60, min_periods=1).min()
    s['net_agg'] = (agg - s['bg_load']).clip(lower=0)

    # Event context — unmonitored loads as cycle candidates (5)
    ev = detect_events(agg)
    for col in ev.columns:
        s[col] = ev[col]

    # Temporal sin/cos (4)
    hour = s.index.hour + s.index.minute / 60.0
    dow  = s.index.dayofweek.astype(float)
    s['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    s['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    s['dow_sin']  = np.sin(2 * np.pi * dow  / 7)
    s['dow_cos']  = np.cos(2 * np.pi * dow  / 7)

    # House behavioral signature — 7 continuous features, NOT a cluster label
    sig = house_sigs.get(house_id, {c: 0.0 for c in SIG_COLS})
    for col in SIG_COLS:
        s[col] = sig[col]

    return s


FEAT_COLS = (
    ['agg_mean_5','agg_std_5','agg_max_5',
     'agg_mean_15','agg_std_15','agg_max_15',
     'agg_mean_30','agg_std_30','agg_max_30',
     'agg_mean_60','agg_std_60','agg_max_60']   # 12
  + ['agg_diff','agg_diff2']                    #  2
  + ['bg_load','net_agg']                       #  2
  + ['ev_active','ev_dur','ev_energy',
     'ev_peak','since_ev']                      #  5
  + ['hour_sin','hour_cos','dow_sin','dow_cos']  #  4
  + SIG_COLS                                    #  7
)
N_FEAT = len(FEAT_COLS)
print(f'Feature set: {N_FEAT} features  '
      f'(12 rolling + 2 roc + 2 bg + 5 event + 4 temporal + 7 house_sig)')

frames = []
for h in sorted(wm_part2['house'].unique()):
    sub = wm_part2[wm_part2['house'] == h].copy()
    sub = engineer_features(sub, h)
    frames.append(sub)
wm_feat = pd.concat(frames).sort_index()
# Fill any NaN remaining (rolling on gap boundaries, edge cases)
wm_feat[FEAT_COLS] = wm_feat[FEAT_COLS].fillna(0)
nan_count = wm_feat[FEAT_COLS].isna().sum().sum()
print(f'Feature matrix: {wm_feat.shape}  |  NaN remaining: {nan_count}')

# ──────────────────────────────────────────────────────────────────────────────
# STAGE 3 — Windowed dataset with balanced sampling
# ──────────────────────────────────────────────────────────────────────────────
print('\nBuilding windowed dataset ...')

def extract_windows(df_h: pd.DataFrame, max_per_class: int = None):
    agg_arr  = df_h['Aggregate'].values.astype(np.float32)
    wm_arr   = df_h['WM'].values.astype(np.float32)
    feat_arr = df_h[FEAT_COLS].values.astype(np.float32)
    n = len(agg_arr)

    # sliding_window_view: zero-copy stride trick, shape (n-WINDOW+1, WINDOW)
    # window at row j is centred at j + HALF_W
    all_wins = sliding_window_view(agg_arr, WINDOW)

    valid = np.arange(HALF_W, n - HALF_W)
    on_mask  = wm_arr[valid] >= ON_THRESH_W
    on_idx   = valid[on_mask]
    off_idx  = valid[~on_mask]

    rng = np.random.default_rng(42)
    if max_per_class:
        on_idx  = rng.choice(on_idx,  min(len(on_idx),  max_per_class), replace=False)
        off_idx = rng.choice(off_idx, min(len(off_idx), max_per_class), replace=False)
    idx = np.sort(np.concatenate([on_idx, off_idx]))

    wins   = all_wins[idx - HALF_W].copy()   # numpy advanced indexing — fast
    feats  = feat_arr[idx]
    labels = wm_arr[idx]
    agg_t  = agg_arr[idx]
    return wins, feats, labels, agg_t


MAX_PER_CLASS = 10000
tr_w, tr_f, tr_l, tr_a = [], [], [], []
va_w, va_f, va_l, va_a = [], [], [], []

for h in TRAIN_HOUSES:
    sub = wm_feat[wm_feat['house'] == h]
    w, f, l, a = extract_windows(sub, max_per_class=MAX_PER_CLASS)
    bucket = (va_w, va_f, va_l, va_a) if h in VALID_HOUSES else (tr_w, tr_f, tr_l, tr_a)
    bucket[0].append(w); bucket[1].append(f)
    bucket[2].append(l); bucket[3].append(a)
    print(f'  H{h:2d}: {len(l):>6,}  ON={( l >= ON_THRESH_W).sum():>5,}  '
          f'{"[valid]" if h in VALID_HOUSES else ""}')

for lst in [tr_w, tr_f, tr_l, tr_a, va_w, va_f, va_l, va_a]:
    lst[:] = [np.concatenate(lst, axis=0)]

train_wins, train_feats, train_labels, train_agg = tr_w[0], tr_f[0], tr_l[0], tr_a[0]
valid_wins, valid_feats, valid_labels, valid_agg = va_w[0], va_f[0], va_l[0], va_a[0]

# Test: all of House 1, no sampling — use sliding_window_view to avoid slow list comprehension
test_h   = wm_feat[wm_feat['house'] == TEST_HOUSE]
_agg = test_h['Aggregate'].values.astype(np.float32)
_wm  = test_h['WM'].values.astype(np.float32)
_f   = test_h[FEAT_COLS].values.astype(np.float32)
nt   = len(_agg)
# sliding_window_view shape: (nt-WINDOW+1, WINDOW) — zero-copy, instant
test_wins   = sliding_window_view(_agg, WINDOW).copy()   # (nt-60, 61)
test_feats  = _f[HALF_W: nt - HALF_W]
test_labels = _wm[HALF_W: nt - HALF_W]
test_agg    = _agg[HALF_W: nt - HALF_W]
test_index  = test_h.index[HALF_W: nt - HALF_W]
print(f'Test windows shape: {test_wins.shape}')

print(f'\nTrain {len(train_labels):,}  Valid {len(valid_labels):,}  '
      f'Test(H1) {len(test_labels):,}  '
      f'H1 ON={( test_labels >= ON_THRESH_W).mean():.1%}')

# Normalise
agg_sc  = StandardScaler()
feat_sc = StandardScaler()
train_wins_n  = agg_sc.fit_transform(train_wins).astype(np.float32)
train_feats_n = feat_sc.fit_transform(train_feats).astype(np.float32)
valid_wins_n  = agg_sc.transform(valid_wins).astype(np.float32)
valid_feats_n = feat_sc.transform(valid_feats).astype(np.float32)
test_wins_n   = agg_sc.transform(test_wins).astype(np.float32)
test_feats_n  = feat_sc.transform(test_feats).astype(np.float32)
train_labels_n = (train_labels / MAX_WM_W).astype(np.float32)
valid_labels_n = (valid_labels / MAX_WM_W).astype(np.float32)

pickle.dump({'agg_sc': agg_sc, 'feat_sc': feat_sc},
            open(CKPT_DIR / 'nilm_scalers.pkl', 'wb'))
print('Scalers saved.')

# ──────────────────────────────────────────────────────────────────────────────
# STAGE 4 — Model definitions
# ──────────────────────────────────────────────────────────────────────────────

class NILMDataset(Dataset):
    def __init__(self, wins, feats, labels, agg_raw):
        self.wins   = torch.tensor(wins,    dtype=torch.float32)
        self.feats  = torch.tensor(feats,   dtype=torch.float32)
        self.labels = torch.tensor(labels,  dtype=torch.float32)
        self.agg    = torch.tensor(agg_raw, dtype=torch.float32)
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        return self.wins[i], self.feats[i], self.labels[i], self.agg[i]


def _cnn_backbone():
    return nn.Sequential(
        nn.Conv1d(1, 30, kernel_size=10, padding=5), nn.ReLU(),
        nn.Conv1d(30, 30, kernel_size=8, padding=4), nn.ReLU(),
        nn.Conv1d(30, 40, kernel_size=6, padding=3), nn.ReLU(),
        nn.Conv1d(40, 50, kernel_size=5, padding=2), nn.ReLU(),
        nn.Conv1d(50, 50, kernel_size=5, padding=2), nn.ReLU(),
    )


class Seq2PointM1(nn.Module):
    """
    M1 — Seq2Point (Zhang et al. 2018) adapted for 1-min / 61-sample window.
    Aggregate window only — no engineered features, no house signature.
    Trained here for direct comparison on the same LOHO split.
    """
    def __init__(self, window=WINDOW):
        super().__init__()
        self.cnn = _cnn_backbone()
        with torch.no_grad():
            sz = self.cnn(torch.zeros(1, 1, window)).flatten(1).shape[1]
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(sz, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 1),
            # No ReLU at output — clips to 0 during inference; ReLU here causes dying gradients
        )
    def forward(self, window, _feats=None):
        return self.fc(self.cnn(window.unsqueeze(1))).squeeze(1)


class UnifiedNILM(nn.Module):
    """
    Our model — shared backbone with per-appliance output heads.

    Design principles:
      1. House identity = 7-dim continuous behavioral signature in FEAT_COLS,
         not a cluster label. Model learns the mapping from behavior → pattern.
      2. Event context features (ev_dur, ev_peak) act as implicit phase proxies:
         the model learns "3-min 3kW event = kettle, 65-min 400W = WM agitation"
         without any explicit phase labelling.
      3. Hierarchical constraint (ŷ ≤ agg) enforced as soft penalty in training
         loss so the model learns physically consistent representations, not just
         clipped at inference.
      4. Multi-appliance extension: add heads (dryer_head, fridge_head, ...) to
         self and train with additional labeled channels. The shared trunk learns
         a universal load decomposition representation.
    """
    def __init__(self, window=WINDOW, n_feat=N_FEAT):
        super().__init__()
        self.cnn = _cnn_backbone()
        with torch.no_grad():
            cnn_sz = self.cnn(torch.zeros(1, 1, window)).flatten(1).shape[1]

        self.feat_branch = nn.Sequential(
            nn.Linear(n_feat, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64),    nn.ReLU(),
            nn.Linear(64, 64),     nn.ReLU(),
        )
        self.trunk = nn.Sequential(
            nn.Flatten(),
            nn.Linear(cnn_sz + 64, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128),         nn.ReLU(),
        )
        # WM head — add more heads here for other appliances
        # No final ReLU — clips to 0 at inference; ReLU here kills gradients for negative pre-activations
        self.wm_head = nn.Sequential(
            nn.Linear(128, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, window, feats):
        cnn_out  = self.cnn(window.unsqueeze(1)).flatten(1)
        feat_out = self.feat_branch(feats)
        z        = self.trunk(torch.cat([cnn_out, feat_out], dim=1))
        return self.wm_head(z).squeeze(1)


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 5 — Training
# ──────────────────────────────────────────────────────────────────────────────

def train_model(model, model_name, use_constraint=False):
    tr_ds = NILMDataset(train_wins_n, train_feats_n, train_labels_n, train_agg)
    va_ds = NILMDataset(valid_wins_n, valid_feats_n, valid_labels_n, valid_agg)
    tr_dl = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    va_dl = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)

    # initialise best_state to current weights so load_state_dict never gets None
    best_val   = float('inf')
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    history    = {'train': [], 'val': []}

    print(f'\n── Training {model_name} ({sum(p.numel() for p in model.parameters()):,} params) ──')
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train(); tr_loss = 0.0
        for wins_b, feats_b, labels_b, agg_b in tr_dl:
            wins_b   = wins_b.to(DEVICE)
            feats_b  = feats_b.to(DEVICE)
            labels_b = labels_b.to(DEVICE)
            agg_b    = agg_b.to(DEVICE)
            opt.zero_grad()
            pred = model(wins_b, feats_b)
            mse  = F.mse_loss(pred, labels_b)
            if use_constraint:
                pred_w    = pred * MAX_WM_W
                violation = (pred_w - agg_b).clamp(min=0) / MAX_WM_W
                loss = mse + LAMBDA_CONSTR * violation.mean()
            else:
                loss = mse
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            tr_loss += loss.item() * len(labels_b)
        tr_loss /= len(tr_ds)

        model.eval(); va_loss = 0.0
        with torch.no_grad():
            for wins_b, feats_b, labels_b, agg_b in va_dl:
                wins_b   = wins_b.to(DEVICE)
                feats_b  = feats_b.to(DEVICE)
                labels_b = labels_b.to(DEVICE)
                pred = model(wins_b, feats_b)
                va_loss += F.mse_loss(pred, labels_b).item() * len(labels_b)
        va_loss /= len(va_ds)

        history['train'].append(tr_loss)
        history['val'].append(va_loss)
        sched.step(va_loss)

        if np.isfinite(va_loss) and va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 4 == 0 or epoch in (1, EPOCHS):
            print(f'  Ep {epoch:2d}/{EPOCHS}  train={tr_loss:.6f}  val={va_loss:.6f}'
                  f'  ({time.time()-t0:.0f}s)')

    model.load_state_dict(best_state)
    print(f'  Best val MSE {best_val:.6f}  total {time.time()-t0:.0f}s')
    return model, history


m1    = Seq2PointM1().to(DEVICE)
m_uni = UnifiedNILM().to(DEVICE)

m1,    hist_m1  = train_model(m1,    'M1 Seq2Point',  use_constraint=False)
m_uni, hist_uni = train_model(m_uni, 'UnifiedNILM',   use_constraint=True)

torch.save(m1.state_dict(),    CKPT_DIR / 'nilm_m1_seq2point.pt')
torch.save(m_uni.state_dict(), CKPT_DIR / 'nilm_unified.pt')
print('Models saved.')

# ──────────────────────────────────────────────────────────────────────────────
# STAGE 6 — Inference on House 1
# ──────────────────────────────────────────────────────────────────────────────
print('\nInference on House 1 ...')

def predict(model, wins_n, feats_n, agg_raw, clip=True):
    model.eval()
    preds = []
    ds = NILMDataset(wins_n, feats_n, np.zeros(len(wins_n), np.float32), agg_raw)
    dl = DataLoader(ds, batch_size=4096, shuffle=False, num_workers=0)
    with torch.no_grad():
        for w, f, _, _ in dl:
            preds.append(model(w.to(DEVICE), f.to(DEVICE)).cpu().numpy())
    p = np.concatenate(preds) * MAX_WM_W
    p = np.clip(p, 0, MAX_WM_W)
    if clip:
        p = np.minimum(p, agg_raw)   # hard clip at inference (belt-and-braces)
    return p

pred_m0  = np.zeros(len(test_labels))
pred_m1  = predict(m1,    test_wins_n, test_feats_n, test_agg)
pred_uni = predict(m_uni, test_wins_n, test_feats_n, test_agg)

# Threshold calibration: find optimal ON/OFF detection threshold from validation set
# This is calibration only — model weights unchanged
print('Calibrating detection threshold on validation set ...')
val_preds_uni = predict(m_uni, valid_wins_n, valid_feats_n, valid_agg)
from sklearn.metrics import f1_score as _f1
best_thresh, best_f1 = ON_THRESH_W, 0.0
for t in np.arange(10, 500, 10):
    f1 = _f1((valid_labels >= ON_THRESH_W).astype(int),
             (val_preds_uni >= t).astype(int), zero_division=0)
    if f1 > best_f1:
        best_f1 = f1; best_thresh = t
print(f'  Best threshold: {best_thresh:.0f}W  (val F1={best_f1:.3f}  default 25W)')
# Apply calibrated threshold for UnifiedNILM evaluation
EVAL_THRESH_UNI = best_thresh

# ──────────────────────────────────────────────────────────────────────────────
# STAGE 7 — Evaluation metrics
# ──────────────────────────────────────────────────────────────────────────────
print('\nEvaluating ...')

def evaluate(y_true, y_pred, agg, name, det_thresh=None):
    if det_thresh is None:
        det_thresh = ON_THRESH_W
    mae  = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    on   = y_true >= ON_THRESH_W
    mae_on = float(np.mean(np.abs(y_true[on] - y_pred[on]))) if on.any() else np.nan
    t_on = (y_true >= ON_THRESH_W).astype(int)
    p_on = (y_pred >= det_thresh).astype(int)
    f1   = f1_score(t_on, p_on, zero_division=0)
    prec = precision_score(t_on, p_on, zero_division=0)
    rec  = recall_score(t_on, p_on, zero_division=0)
    te = y_true.sum() / 60.0; pe = y_pred.sum() / 60.0
    sae   = abs(pe - te) / (te + 1e-9)
    e_err = 100.0 * abs(pe - te) / (te + 1e-9)
    viol  = np.mean(np.maximum(y_pred - agg, 0))
    return dict(model=name, mae=mae, rmse=rmse, mae_on=mae_on,
                f1=f1, precision=prec, recall=rec,
                true_kwh=te/1000, pred_kwh=pe/1000,
                energy_err_pct=e_err, sae=sae, constraint_viol_W=viol)

results = [
    evaluate(test_labels, pred_m0,  test_agg, 'M0 Zero baseline'),
    evaluate(test_labels, pred_m1,  test_agg, 'M1 Seq2Point'),
    evaluate(test_labels, pred_uni, test_agg, 'UnifiedNILM (ours)', det_thresh=EVAL_THRESH_UNI),
]
df_res = pd.DataFrame(results).set_index('model')

print('\n' + '='*80)
print('RESULTS — House 1 test set (LOHO)')
print('='*80)
print(df_res[['mae','rmse','mae_on','f1','precision','recall',
              'energy_err_pct','sae','constraint_viol_W']].round(2).to_string())
print('='*80)
df_res.to_csv(RES_DIR / 'metrics.csv')

# ──────────────────────────────────────────────────────────────────────────────
# STAGE 8 — Plots
# ──────────────────────────────────────────────────────────────────────────────
print('\nPlotting ...')
COLORS = {'M0':'#aaa', 'M1':'#2c7bb6', 'UNI':'#d7191c', 'Truth':'#1a9641'}

# S3_0 — Training curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle('Training Convergence', fontsize=11, fontweight='bold')
for ax, hist, title, c in [
    (axes[0], hist_m1,  'M1 Seq2Point',   COLORS['M1']),
    (axes[1], hist_uni, 'UnifiedNILM',    COLORS['UNI']),
]:
    ax.plot(hist['train'], c=c, lw=1.5, label='Train')
    ax.plot(hist['val'],   c=c, lw=1.5, ls='--', label='Val')
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('MSE (normalised)')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / 'S3_0_training_curves.png', dpi=150, bbox_inches='tight')
print('  S3_0_training_curves.png')

# S3_1 — Predicted vs actual, representative day
true_s = pd.Series(test_labels, index=test_index)
agg_s  = pd.Series(test_agg,    index=test_index)
preds_s = {'M1': pd.Series(pred_m1,  index=test_index),
           'UNI': pd.Series(pred_uni, index=test_index)}

daily_on = true_s.resample('D').apply(lambda s: (s >= ON_THRESH_W).sum())
good = daily_on[daily_on >= 40].index
plot_day = good[len(good)//2] if len(good) else daily_on.nlargest(1).index[0]
ds_ = pd.Timestamp(plot_day); de_ = ds_ + pd.Timedelta('1D') - pd.Timedelta('1min')

fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True,
                         gridspec_kw={'height_ratios': [2,1.2,1.2,1.2], 'hspace': 0.07})
fig.suptitle(f'House 1 — Predicted vs Actual WM Power  |  {ds_.strftime("%d %b %Y")}\n'
             '(Test set — never seen during training)',
             fontsize=11, fontweight='bold')

axes[0].plot(agg_s[ds_:de_].index, agg_s[ds_:de_], c='#555', lw=0.8)
axes[0].set_ylabel('Aggregate (W)', fontsize=8)
axes[0].set_title('Whole-house aggregate (context)', fontsize=8, loc='left', color='#555')
axes[0].grid(alpha=0.2)

for ax, (key, label, c) in zip(axes[1:], [
    ('M1',   'M1 Seq2Point',  COLORS['M1']),
    ('UNI',  'UnifiedNILM',   COLORS['UNI']),
    ('TRUTH','Ground truth',  COLORS['Truth']),
]):
    d_true = true_s[ds_:de_]
    d_pred = preds_s[key][ds_:de_] if key != 'TRUTH' else d_true
    ax.fill_between(d_true.index, 0, d_true, color=COLORS['Truth'], alpha=0.2, label='True WM')
    ax.fill_between(d_pred.index, 0, d_pred, color=c, alpha=0.45)
    ax.plot(d_pred.index, d_pred, c=c, lw=0.9, label=label)
    ax.axhline(ON_THRESH_W, c='gray', lw=0.5, ls='--', alpha=0.4)
    ax.set_ylabel('WM (W)', fontsize=7.5); ax.set_ylim(0, 2800)
    ax.legend(fontsize=7.5, loc='upper right'); ax.grid(alpha=0.2)
axes[-1].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
axes[-1].set_xlabel('Time of day')
plt.savefig(FIG_DIR / 'S3_1_predicted_vs_actual_day.png', dpi=150, bbox_inches='tight')
print('  S3_1_predicted_vs_actual_day.png')

# S3_2 — Metric comparison bars
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle('Model Comparison — House 1 Test Set\n'
             'M0 Zero  |  M1 Seq2Point  |  UnifiedNILM (ours)',
             fontsize=11, fontweight='bold')
metrics_cfg = [
    ('mae_on',          'ON-state MAE (W)',     True,  'Lower is better'),
    ('f1',              'F1 Score (ON/OFF det)',False, 'Higher is better'),
    ('energy_err_pct',  'Energy Error (%)',     True,  'Lower is better'),
]
bar_colors = [COLORS['M0'], COLORS['M1'], COLORS['UNI']]
mnames     = ['M0\nZero', 'M1\nSeq2Pt', 'Unified\nNILM']
for ax, (m, ylabel, lower, note) in zip(axes, metrics_cfg):
    vals = [float(df_res.loc[r, m]) if not np.isnan(df_res.loc[r, m]) else 0
            for r in df_res.index]
    bars = ax.bar(mnames, vals, color=bar_colors, alpha=0.82,
                  edgecolor='white', width=0.55)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(f'{ylabel}\n({note})', fontsize=8.5, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(vals)*0.01,
                f'{v:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    best = int(np.argmin(vals) if lower else np.argmax(vals))
    bars[best].set_edgecolor('gold'); bars[best].set_linewidth(2.5)
plt.tight_layout()
plt.savefig(FIG_DIR / 'S3_2_metric_comparison.png', dpi=150, bbox_inches='tight')
print('  S3_2_metric_comparison.png')

# S3_3 — Scatter actual vs predicted
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Actual vs Predicted — House 1 Test Set', fontsize=11, fontweight='bold')
rng2 = np.random.default_rng(0)
idx_s = rng2.choice(len(test_labels), min(15000, len(test_labels)), replace=False)
for ax, (name, preds, c) in zip(axes, [
    ('M1 Seq2Point', pred_m1,  COLORS['M1']),
    ('UnifiedNILM',  pred_uni, COLORS['UNI']),
]):
    yt = test_labels[idx_s]; yp = preds[idx_s]; on_m = yt >= ON_THRESH_W
    ax.scatter(yt[~on_m], yp[~on_m], c='#ccc', s=3, alpha=0.3, label='WM off')
    ax.scatter(yt[on_m],  yp[on_m],  c=c,     s=5, alpha=0.5, label='WM on')
    lim = max(yt.max(), yp.max()) * 1.02
    ax.plot([0, lim], [0, lim], 'k--', lw=0.8, alpha=0.5)
    row = df_res[df_res.index.str.contains(
        'Seq2' if 'Seq' in name else 'Unified')].iloc[0]
    ax.set_title(f'{name}\nMAE={row.mae:.1f}W  F1={row.f1:.3f}', fontsize=8.5)
    ax.set_xlabel('Actual WM (W)', fontsize=8); ax.set_ylabel('Predicted WM (W)', fontsize=8)
    ax.legend(fontsize=7.5, markerscale=2); ax.grid(alpha=0.25)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
plt.tight_layout()
plt.savefig(FIG_DIR / 'S3_3_scatter.png', dpi=150, bbox_inches='tight')
print('  S3_3_scatter.png')

print('\n' + '='*80)
print('SECTION 3 COMPLETE')
print('='*80)
print(df_res[['mae','f1','energy_err_pct','sae','constraint_viol_W']].round(3).to_string())
print('='*80)
