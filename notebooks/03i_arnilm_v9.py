import warnings; warnings.filterwarnings('ignore')
import sys, os
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import time, math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score

CKPT_DIR = Path('data/processed/checkpoints')
FIG_DIR  = Path('figures');          FIG_DIR.mkdir(exist_ok=True)
RES_DIR  = Path('results/section3'); RES_DIR.mkdir(exist_ok=True)

PART2_START  = pd.Timestamp('2014-04-01')
TEST_HOUSE   = 1
TRAIN_HOUSES = [2,3,4,5,6,7,8,9,10,11,13,15,16,17,18,19]
VALID_HOUSES = [20, 21]
CAL_HOUSES   = [5, 7, 11, 17]
CAL_WEEKS    = 4

ON_THRESH_W   = 25.0
MAX_WM_W      = 3000.0
MAX_AGG_W     = 8000.0
SEQ_LEN       = 720
SEQ_STRIDE    = 360
BATCH_SIZE    = 128        # smaller: attention is memory-heavier
EPOCHS        = 80
LR            = 5e-4       # lower LR: attention layers benefit from slower convergence
LAMBDA_CONSTR = 0.1
LAMBDA_CLS    = 1.0
WM_ON_WEIGHT  = 3.5        # V8 setting: precision-balanced

# V9 changes vs V8:
#   Replace LSTM-only encoder with LSTM + Multi-head Self-Attention
#   Each timestep attends to all 720 positions → captures full WM cycle pattern
#   Positional encoding added so attention knows where in the window it is
#   N_HEADS=8, ATT_DIM=256 (same as LSTM hidden)
#
#   Why attention helps:
#   LSTM: processes left-to-right, forgets cycle shape by end of 12h window
#   Attention: minute 600 can directly attend to minute 30 (cycle start signature)
#              → model learns global WM cycle pattern, not just local transitions

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42); np.random.seed(42)

print(f'Device: {DEVICE}')
print(f'Architecture: ARNILM v9 — LSTM + Multi-head Self-Attention + SGN gate')
print(f'  V9 adds: Multi-head attention over LSTM output (n_heads=8)')
print(f'  Keeps:   norm-MSE, 4 shape features (N_INPUT=21), pos_weight=3.5')
print(f'  Target:  F1 > 0.42  (Seq2Point NILMBench benchmark)')
print(f'Train: {TRAIN_HOUSES}  Valid: {VALID_HOUSES}  Cal: {CAL_HOUSES}  Test: H{TEST_HOUSE}')

# ── Load data ──────────────────────────────────────────────────────────────────
print('\nLoading data ...')
wm_all = pd.read_parquet(CKPT_DIR / 'ckpt_wm_1min_clean.parquet')
wm_all.index = pd.to_datetime(wm_all.index)
wm_all['WM']        = wm_all['WM'].astype(float).fillna(0).clip(0, MAX_WM_W)
wm_all['Aggregate'] = wm_all['Aggregate'].astype(float).fillna(0).clip(0)
wm_part2 = wm_all[wm_all.index >= PART2_START].copy()
print(f'  Part2: {len(wm_part2):,} rows  |  {wm_part2["house"].nunique()} houses')

# ── House behavioral signatures ────────────────────────────────────────────────
print('\nHouse behavioral signatures ...')
SIG_COLS = ['sig_med_dur','sig_med_energy','sig_hot_frac',
            'sig_ph_sin','sig_ph_cos','sig_med_peak','sig_hot_frac2']

def detect_agg_cycles(agg_series, thresh_on=80.0, thresh_off=25.0,
                      hyst_min=5, dur_min=15, dur_max=180, peak_min=400.0):
    agg = agg_series.values.astype(np.float32); idx = agg_series.index
    events = []
    state = 'IDLE'; ev_start = 0; drop_start = 0; cum_e = 0.0; pk = 0.0
    for i in range(len(agg)):
        v = float(agg[i]) if np.isfinite(agg[i]) else 0.0
        if state == 'IDLE':
            if v >= thresh_on: ev_start = i; cum_e = 0.0; pk = 0.0; state = 'ACTIVE'
        elif state == 'ACTIVE':
            if v < thresh_off: drop_start = i; state = 'COOLING'
            else: cum_e += v / 60.0; pk = max(pk, v)
        elif state == 'COOLING':
            if v >= thresh_off: state = 'ACTIVE'; cum_e += v / 60.0; pk = max(pk, v)
            elif (i - drop_start) >= hyst_min:
                dur = drop_start - ev_start
                if dur_min <= dur <= dur_max and pk >= peak_min:
                    events.append(dict(duration_min=dur, energy_wh=cum_e, peak_w=pk,
                                       hot_wash=(cum_e >= 350.0), hour_start=idx[ev_start].hour))
                state = 'IDLE'
    return pd.DataFrame(events) if events else pd.DataFrame(
        columns=['duration_min','energy_wh','peak_w','hot_wash','hour_start'])

def house_sig_from_agg(agg_series):
    cyc = detect_agg_cycles(agg_series)
    if len(cyc) < 3: return {c: 0.0 for c in SIG_COLS}
    ph = float(cyc['hour_start'].mode().iloc[0]); hot_frac = float(cyc['hot_wash'].mean())
    return dict(sig_med_dur=float(cyc['duration_min'].median())/180.0,
                sig_med_energy=float(cyc['energy_wh'].median())/800.0,
                sig_hot_frac=hot_frac,
                sig_ph_sin=float(np.sin(2*np.pi*ph/24)),
                sig_ph_cos=float(np.cos(2*np.pi*ph/24)),
                sig_med_peak=float(cyc['peak_w'].median())/MAX_WM_W,
                sig_hot_frac2=hot_frac**2)

house_sigs = {}
for h in sorted(wm_part2['house'].unique()):
    sub = wm_part2[wm_part2['house'] == h]
    house_sigs[h] = house_sig_from_agg(sub['Aggregate'])
    s = house_sigs[h]
    print(f'  H{h:2d}  dur={s["sig_med_dur"]*180:.0f}min  hot={s["sig_hot_frac"]:.0%}  peak={s["sig_med_peak"]*MAX_WM_W:.0f}W')

# ── Dynamic covariates (same as V7/V8: N_DYN=13) ──────────────────────────────
print('\nEngineering covariates ...')

def detect_events(agg_vals, thresh=80.0):
    n = len(agg_vals)
    ev_active=np.zeros(n,np.float32); ev_dur=np.zeros(n,np.float32)
    ev_energy=np.zeros(n,np.float32); ev_peak=np.zeros(n,np.float32)
    since_ev=np.zeros(n,np.float32)
    in_ev=False; ev_start=0; cum_e=0.0; pk=0.0; last_end=-1
    for i in range(n):
        v = float(agg_vals[i]) if np.isfinite(agg_vals[i]) else 0.0
        if v >= thresh:
            if not in_ev: in_ev=True; ev_start=i; cum_e=0.0; pk=0.0
            cum_e+=v/60.0; pk=max(pk,v)
            ev_active[i]=1.0; ev_dur[i]=float(i-ev_start+1)/180.0
            ev_energy[i]=cum_e/500.0; ev_peak[i]=pk/MAX_WM_W
        else:
            if in_ev: in_ev=False; last_end=i
            if last_end>=0: since_ev[i]=min(float(i-last_end),240.0)/240.0
    return np.stack([ev_active,ev_dur,ev_energy,ev_peak,since_ev],axis=1)

N_DYN=13; N_STATIC=7; N_INPUT=1+N_DYN+N_STATIC  # 21
print(f'  N_INPUT={N_INPUT}')

def build_dyn_covariates(df_h):
    agg=df_h['Aggregate'].values.astype(np.float32); n=len(agg)
    ev=detect_events(agg)
    hour=(df_h.index.hour+df_h.index.minute/60.0).values
    dow=df_h.index.dayofweek.astype(float).values
    temporal=np.stack([np.sin(2*np.pi*hour/24),np.cos(2*np.pi*hour/24),
                       np.sin(2*np.pi*dow/7),np.cos(2*np.pi*dow/7)],axis=1).astype(np.float32)
    agg_diff=np.zeros(n,np.float32); agg_diff[1:]=(agg[1:]-agg[:-1])/MAX_AGG_W
    agg_abs_diff=np.abs(agg_diff)
    agg_s=pd.Series(agg)
    roll10=agg_s.rolling(10,min_periods=1).std().fillna(0).values.astype(np.float32)/MAX_AGG_W
    roll30=agg_s.rolling(30,min_periods=1).std().fillna(0).values.astype(np.float32)/MAX_AGG_W
    deriv=np.stack([agg_diff,agg_abs_diff,roll10,roll30],axis=1)
    return np.concatenate([ev,temporal,deriv],axis=1)

house_data={}; cal_split_idx={}
for h in sorted(wm_part2['house'].unique()):
    sub=wm_part2[wm_part2['house']==h]
    agg=sub['Aggregate'].values.astype(np.float32)
    wm=sub['WM'].values.astype(np.float32)
    dyn=build_dyn_covariates(sub)
    sig=np.array([house_sigs[h][c] for c in SIG_COLS],dtype=np.float32)
    house_data[h]=dict(agg=agg,wm=wm,dyn=dyn,sig=sig,index=sub.index)
    if h in CAL_HOUSES:
        cal_len=min(CAL_WEEKS*7*24*60,len(agg)//4)
        cal_split_idx[h]=len(agg)-cal_len
    print(f'  H{h:2d}: {len(agg):,} timesteps'+
          (f'  [cal from {cal_split_idx[h]:,}]' if h in CAL_HOUSES else ''))

# ── Dataset ────────────────────────────────────────────────────────────────────
print('\nBuilding sequences ...')

class SequenceDataset(Dataset):
    def __init__(self, sequences): self.seqs = sequences
    def __len__(self): return len(self.seqs)
    def __getitem__(self, i):
        agg,dyn,wm,sig = self.seqs[i]
        return (torch.FloatTensor(agg),torch.FloatTensor(dyn),
                torch.FloatTensor(wm), torch.FloatTensor(sig))

def make_sequences(h_list, stride=SEQ_STRIDE, end_idx_map=None):
    seqs=[]
    for h in h_list:
        d=house_data[h]; T=len(d['agg'])
        max_end=end_idx_map.get(h,T) if end_idx_map else T
        for start in range(0, max_end-SEQ_LEN+1, stride):
            end=start+SEQ_LEN
            if end>max_end: break
            seqs.append((d['agg'][start:end],d['dyn'][start:end],d['wm'][start:end],d['sig']))
    return seqs

def make_cal_sequences(stride=SEQ_STRIDE):
    seqs=[]
    for h in CAL_HOUSES:
        d=house_data[h]; T=len(d['agg']); s0=cal_split_idx[h]
        for start in range(s0,T-SEQ_LEN+1,stride):
            seqs.append((d['agg'][start:start+SEQ_LEN],d['dyn'][start:start+SEQ_LEN],
                         d['wm'][start:start+SEQ_LEN],d['sig']))
    return seqs

train_end_map={h:cal_split_idx[h] for h in CAL_HOUSES}
train_seqs=make_sequences(TRAIN_HOUSES,end_idx_map=train_end_map)
valid_seqs=make_sequences(VALID_HOUSES)
cal_seqs=make_cal_sequences()
print(f'  Train:{len(train_seqs):,}  Valid:{len(valid_seqs):,}  Cal:{len(cal_seqs):,}')

tr_dl=DataLoader(SequenceDataset(train_seqs),batch_size=BATCH_SIZE,shuffle=True, num_workers=4,pin_memory=True)
va_dl=DataLoader(SequenceDataset(valid_seqs),batch_size=BATCH_SIZE,shuffle=False,num_workers=4,pin_memory=True)

# ── ARNILM v9 — LSTM + Multi-head Self-Attention + SGN gate ───────────────────
#
# Encoder:
#   LSTM [T, 256] → captures local sequential patterns (ramp-up, within-cycle transitions)
#   MultiheadAttention [T, 256] → each timestep attends to all T=720 positions
#   → model learns: "if minute 30 looks like WM start, minute 80 (heating phase) should too"
#
# Decoder (SGN gate, same as V6-V8):
#   mu_head:  regression — how many watts?
#   cls_head: classification gate — is WM on?
#   y_hat = Softplus(mu_raw) × MAX_WM_W × Sigmoid(cls_logit)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=SEQ_LEN, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, T, d_model]

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])

class NILM_LSTM_Attention_V9(nn.Module):
    def __init__(self, n_input=N_INPUT, hidden=256, n_layers=2, n_heads=8):
        super().__init__()
        # LSTM encoder: local sequential context
        self.lstm = nn.LSTM(n_input, hidden, n_layers,
                            batch_first=True, dropout=0.15)
        # Positional encoding so attention knows timestep position
        self.pos_enc = PositionalEncoding(hidden, max_len=SEQ_LEN)
        # Multi-head self-attention: global cycle pattern recognition
        self.attn = nn.MultiheadAttention(hidden, n_heads,
                                          dropout=0.1, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden)
        # Feed-forward after attention (standard Transformer block)
        self.ff = nn.Sequential(
            nn.Linear(hidden, hidden * 2), nn.GELU(),
            nn.Linear(hidden * 2, hidden), nn.Dropout(0.1)
        )
        self.ff_norm = nn.LayerNorm(hidden)
        # SGN gate heads (same as V6-V8)
        self.mu_head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Softplus()
        )
        self.cls_head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, agg, dyn, sig, hidden=None):
        B, T = agg.shape
        agg_n = (agg / MAX_AGG_W).unsqueeze(-1)
        sig_e = sig.unsqueeze(1).expand(-1, T, -1)
        inp   = torch.cat([agg_n, dyn, sig_e], dim=-1)   # [B, T, 21]

        # LSTM: local sequential context
        lstm_out, hidden = self.lstm(inp, hidden)          # [B, T, 256]

        # Positional encoding + self-attention: global context
        x = self.pos_enc(lstm_out)
        attn_out, _ = self.attn(x, x, x)                  # [B, T, 256]
        x = self.attn_norm(lstm_out + attn_out)            # residual

        # Feed-forward
        ff_out = self.ff(x)
        x = self.ff_norm(x + ff_out)                       # residual

        # SGN gate
        mu_raw    = self.mu_head(x).squeeze(-1)            # [B, T]
        cls_logit = self.cls_head(x).squeeze(-1)           # [B, T]
        p_on      = torch.sigmoid(cls_logit)
        y_hat     = mu_raw * MAX_WM_W * p_on               # [B, T] watts

        return y_hat, p_on, cls_logit, hidden

# ── Loss (same as V7/V8: normalized MSE) ──────────────────────────────────────
def loss_fn(y_hat, p_on, cls_logit, agg_b, wm_b):
    w = torch.where(wm_b >= ON_THRESH_W,
                    torch.full_like(wm_b, WM_ON_WEIGHT),
                    torch.ones_like(wm_b))
    mse = (w * ((y_hat - wm_b) / MAX_WM_W) ** 2).sum() / w.sum()
    y_on = (wm_b >= ON_THRESH_W).float()
    bce  = F.binary_cross_entropy_with_logits(
        cls_logit, y_on,
        pos_weight=torch.tensor(WM_ON_WEIGHT, device=DEVICE))
    violation = (y_hat - agg_b).clamp(min=0) / MAX_WM_W
    return mse + LAMBDA_CLS * bce + LAMBDA_CONSTR * violation.mean()

# ── Training ───────────────────────────────────────────────────────────────────
model = NILM_LSTM_Attention_V9().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'\n── Training NILM_LSTM_Attention_V9 ({n_params:,} params) ──')
print(f'   LSTM + Multi-head Self-Attention + SGN gate')

opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)

best_val=float('inf')
best_state={k:v.clone() for k,v in model.state_dict().items()}
history={'train':[],'val':[]}
t0=time.time()

for epoch in range(1, EPOCHS+1):
    model.train(); tr_loss=0.0; tr_n=0
    for agg_b,dyn_b,wm_b,sig_b in tr_dl:
        agg_b=agg_b.to(DEVICE); dyn_b=dyn_b.to(DEVICE)
        wm_b=wm_b.to(DEVICE);   sig_b=sig_b.to(DEVICE)
        opt.zero_grad()
        y_hat,p_on,cls_logit,_=model(agg_b,dyn_b,sig_b)
        loss=loss_fn(y_hat,p_on,cls_logit,agg_b,wm_b)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step()
        tr_loss+=loss.item()*len(wm_b); tr_n+=len(wm_b)
    tr_loss/=tr_n

    model.eval(); va_loss=0.0; va_n=0
    with torch.no_grad():
        for agg_b,dyn_b,wm_b,sig_b in va_dl:
            agg_b=agg_b.to(DEVICE); dyn_b=dyn_b.to(DEVICE)
            wm_b=wm_b.to(DEVICE);   sig_b=sig_b.to(DEVICE)
            y_hat,p_on,cls_logit,_=model(agg_b,dyn_b,sig_b)
            loss=loss_fn(y_hat,p_on,cls_logit,agg_b,wm_b)
            va_loss+=loss.item()*len(wm_b); va_n+=len(wm_b)
    va_loss/=va_n
    history['train'].append(tr_loss); history['val'].append(va_loss)
    sched.step()

    if va_loss<best_val:
        best_val=va_loss
        best_state={k:v.clone() for k,v in model.state_dict().items()}

    if epoch%5==0 or epoch in (1, EPOCHS):
        print(f'  Ep {epoch:2d}/{EPOCHS}  train={tr_loss:.4f}  val={va_loss:.4f}  ({time.time()-t0:.0f}s)')

model.load_state_dict(best_state)
torch.save(model.state_dict(), CKPT_DIR/'nilm_ar_lstm_v9.pt')
print(f'  Best val loss {best_val:.4f}  |  checkpoint saved.')

# ── Inference ──────────────────────────────────────────────────────────────────
print('\nInference on House 1 ...')

def predict(mdl, h=TEST_HOUSE, chunk=2000):
    # attention is O(T²) memory — use smaller chunks
    mdl.eval()
    d=house_data[h]; T=len(d['agg'])
    preds=np.zeros(T,np.float32); p_ons=np.zeros(T,np.float32)
    hidden=None
    with torch.no_grad():
        for s in range(0,T,chunk):
            e=min(s+chunk,T)
            ac=torch.FloatTensor(d['agg'][s:e]).unsqueeze(0).to(DEVICE)
            dc=torch.FloatTensor(d['dyn'][s:e]).unsqueeze(0).to(DEVICE)
            sc=torch.FloatTensor(d['sig']).unsqueeze(0).to(DEVICE)
            yh,po,_,hidden=mdl(ac,dc,sc,hidden)
            preds[s:e]=yh[0].cpu().numpy()
            p_ons[s:e]=po[0].cpu().numpy()
            hidden=tuple(hh.detach() for hh in hidden)
    preds=np.clip(preds,0,d['agg'])
    return preds,p_ons

pred_wm,pred_pon=predict(model)
test_labels=house_data[TEST_HOUSE]['wm']
test_agg=house_data[TEST_HOUSE]['agg']
print('  Inference done.')

# ── Calibrate ──────────────────────────────────────────────────────────────────
print('\nCalibrating p_on threshold ...')
all_pon=[]; all_true=[]
for h in CAL_HOUSES:
    _,po=predict(model,h=h)
    s0=cal_split_idx[h]
    all_pon.append(po[s0:]); all_true.append(house_data[h]['wm'][s0:])
cpo=np.concatenate(all_pon); ct=np.concatenate(all_true)
best_pt,best_f1=0.5,0.0
for pt in np.arange(0.03,0.95,0.02):
    f=f1_score((ct>=ON_THRESH_W).astype(int),(cpo>=pt).astype(int),zero_division=0)
    if f>best_f1: best_f1,best_pt=f,float(pt)
print(f'  Best p_on >= {best_pt:.2f}  (cal F1={best_f1:.3f})')

# ── Evaluate ───────────────────────────────────────────────────────────────────
print('\nEvaluating ...')
true_on=(test_labels>=ON_THRESH_W).astype(int)
pred_on=(pred_pon>=best_pt).astype(int)
mae=float(np.mean(np.abs(pred_wm-test_labels)))
rmse=float(np.sqrt(np.mean((pred_wm-test_labels)**2)))
on_mask=true_on==1
mae_on=float(np.mean(np.abs(pred_wm[on_mask]-test_labels[on_mask]))) if on_mask.sum()>0 else float('nan')
f1=float(f1_score(true_on,pred_on,zero_division=0))
prec=float(precision_score(true_on,pred_on,zero_division=0))
rec=float(recall_score(true_on,pred_on,zero_division=0))
true_kwh=float(test_labels.sum()/60000)
pred_kwh=float(pred_wm.sum()/60000)
energy_err=abs(pred_kwh-true_kwh)/max(true_kwh,1e-6)*100
viol_w=float(np.mean(np.clip(pred_wm-test_agg,0,None)))

print('\n'+'='*72)
print('RESULTS — House 1 test set (cross-house LOHO)')
print('='*72)
print(f'  MAE:            {mae:.2f} W')
print(f'  RMSE:           {rmse:.2f} W')
print(f'  MAE (ON):       {mae_on:.2f} W')
print(f'  F1:             {f1:.4f}')
print(f'  Precision:      {prec:.4f}')
print(f'  Recall:         {rec:.4f}')
print(f'  Energy error:   {energy_err:.1f}%')
print(f'  Constraint viol:{viol_w:.3f} W')
print('='*72)
print(f'V8 baseline: MAE=20W  F1=0.306  Energy_err=69.7%')
print(f'Target:      F1>0.42  (Seq2Point NILMBench benchmark)')

results_df=pd.DataFrame([{
    'model':'ARNILM v9 (LSTM + MHA + SGN gate)',
    'mae':round(mae,2),'rmse':round(rmse,2),'mae_on':round(mae_on,2),
    'f1':round(f1,4),'precision':round(prec,4),'recall':round(rec,4),
    'energy_err_pct':round(energy_err,1),'constraint_viol_W':round(viol_w,3),
}])
results_df.to_csv(RES_DIR/'metrics_ar_lstm_v9.csv',index=False)
print(f'Saved → {RES_DIR}/metrics_ar_lstm_v9.csv')

# ── Plots ──────────────────────────────────────────────────────────────────────
fig,axes=plt.subplots(1,2,figsize=(14,4))
axes[0].plot(history['train'],label='Train',color='steelblue')
axes[0].plot(history['val'],  label='Val',  color='tomato')
axes[0].set_title('ARNILM v9 Training (LSTM + MHA + norm-MSE)')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
axes[0].legend(); axes[0].grid(alpha=0.3)

n_day=1440; offset=len(test_labels)-2*n_day
axes[1].plot(test_labels[offset:offset+n_day],label='Actual WM',alpha=0.8,lw=1)
axes[1].plot(pred_wm[offset:offset+n_day],    label='V9 Predicted',alpha=0.8,lw=1,ls='--')
axes[1].set_title(f'ARNILM v9 H1  (F1={f1:.3f}  MAE={mae:.1f}W)')
axes[1].set_xlabel('Minute'); axes[1].set_ylabel('WM Power (W)')
axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
plt.tight_layout()
fig.savefig(FIG_DIR/'S3i_0_arnilm_v9.png',dpi=150,bbox_inches='tight')
plt.close()

print('\nARNILM v9 COMPLETE')
