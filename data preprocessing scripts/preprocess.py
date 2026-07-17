"""
03_preprocess.py
Extract 3-SECOND REM mini-epochs from PSG with per-mini RSWA labels, matching the
SINBAR / McCarter (Leclair-Visonneau et al., JCSM 2024) scoring unit so that the
CNN's per-mini output aggregates directly to the guideline "any"-activity density.

Unit = 3-s mini-epoch (600 samples @ 200Hz), taken only from valid REM 30-s epochs.
Per-mini label (chin EMG, background-corrected uV — Ferri rolling-min):
  is_phasic : a burst > PHASIC_UV within the mini
  is_tonic  : sustained > TONIC_UV for >= TONIC_FRAC of the mini
  labels    : "any" = phasic OR tonic   <-- CNN target (McCarter/SINBAR "any")

Subject-level diagnosis = "any" density = % of REM 3-s mini-epochs labelled any.
  Compared to the SINBAR/McCarter chin "any" cutoff (~6.5-21.6% across studies;
  Frauscher 2012 = 18.2%; Leclair-Visonneau/McCarter 2024 = 6.5% optimized / 14% max-spec).
Also stores the whole-night Ferri REM Atonia Index (RAI) as an independent readout.

Output: data/processed/workshop_data.h5
  {control|rbd}/{subj}/signals   (n_mini, 3, 600) float32  [EEG,EOG,EMG] filtered uV
                                  (physical microvolts; globally normalised at load, NOT per-subject)
  {control|rbd}/{subj}/labels    (n_mini,) int8   any = phasic OR tonic  (CNN target)
  {control|rbd}/{subj}/is_phasic (n_mini,) int8
  {control|rbd}/{subj}/is_tonic  (n_mini,) int8
  {control|rbd}/{subj}/n_mini    scalar
  per-subject attrs: any/phasic/tonic density, dx, rai_rem

Run: python scripts/03_preprocess.py
"""

import mne
import os
import numpy as np
import h5py
from scipy.ndimage import minimum_filter1d
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_DIR = Path(os.environ.get('RBD_DATA', 'rbd_workshop/data')) / 'raw'
OUT_DIR = Path(os.environ.get('RBD_DATA', 'rbd_workshop/data')) / 'processed'
OUT_DIR.mkdir(parents=True, exist_ok=True)
H5_PATH = OUT_DIR / 'workshop_data.h5'

# ── Constants ─────────────────────────────────────────────────────────────────
TARGET_SFREQ  = 200
EPOCH_SEC     = 30                                # AASM sleep epoch (for staging validity)
MINI_SEC      = 3                                 # SINBAR/AASM scoring unit = CNN window
MINI_SAMPLES  = TARGET_SFREQ * MINI_SEC           # 600

# EMG filtering (Ferri) + rolling-min noise correction
EMG_LFREQ, EMG_HFREQ = 1.0, 100.0
NOTCH_FREQ           = 50.0                        # CAP = Italian mains 50 Hz
NOISE_WIN_SEC        = 60                          # Ferri 2010 rolling-min window
ATONIA_UV            = 1.0                         # <=1uV bin (subject RAI numerator)
UNCERTAIN_UV         = 2.0                         # 1-2uV bin excluded from RAI denom
RAI_ABNORMAL         = 0.8                         # Ferri RAI subject cutoff (reporting)

# Per-3-s-mini activity thresholds (on background-corrected uV; calibrated to SINBAR)
TONIC_UV   = 2.0    # tonic: sustained > 2uV for >= TONIC_FRAC of the mini
TONIC_FRAC = 0.5
PHASIC_UV  = 3.0    # phasic: a burst > 3uV within the mini
# label "any" = phasic OR tonic

# Subject "any"-density diagnostic cutoff (% of 3-s REM minis). Chin "any" literature
# range ~6.5-21.6% (Frauscher 2012 18.2%; Leclair-Visonneau/McCarter 2024 6.5-14%).
SINBAR_ANY_CUT = 18.0

# ── Subject lists ─────────────────────────────────────────────────────────────
CONTROLS = ['n1','n2','n3','n4','n5','n8','n10','n11']   # n4,n8 = mylohyoid (milo), 200Hz
RBDS     = ['rbd1','rbd2','rbd3','rbd4','rbd5','rbd6','rbd7',
            'rbd8','rbd9','rbd10','rbd12','rbd13','rbd14','rbd15',
            'rbd16','rbd17','rbd18','rbd19','rbd20','rbd21','rbd22']

# ── Channel lookup ────────────────────────────────────────────────────────────
def find_channel(ch_names, candidates):
    for c in candidates:
        if c in ch_names:
            return c
    return None

def get_channels(raw):
    chs = raw.ch_names
    emg = find_channel(chs, ['EMG1-EMG2','EMG-EMG','EMG','CHIN-0','CHIN1','milo'])
    eeg = find_channel(chs, ['C4-A1','C4-P4','Fp2-F4','F4-C4'])
    eog = find_channel(chs, ['ROC-LOC','EOG dx','EOG-L','EOG sin'])
    return emg, eeg, eog

# ── Staging parser ────────────────────────────────────────────────────────────
NREM_STAGES = {'SLEEP-S1','SLEEP-S2','SLEEP-S3','SLEEP-S4'}
REM_STAGE   = 'SLEEP-REM'

def parse_staging(txt_path):
    stages = []
    with open(txt_path, 'r', errors='ignore') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            event    = next((p for p in parts if p.startswith('SLEEP-')), None)
            time_str = next((p for p in parts if ':' in p and len(p) == 8), None)
            if event is None or time_str is None:
                continue
            h, m, s = time_str.split(':')
            stages.append((int(h)*3600 + int(m)*60 + int(s), event))
    return stages

def to_relative(stages, rec_start_sec):
    out = []
    for onset_abs, stage in stages:
        rel = onset_abs - rec_start_sec
        if rel < -3600:
            rel += 86400
        out.append((rel, stage))
    return out

# ── Signal processing ─────────────────────────────────────────────────────────
def bandpass(data_uv, sfreq, lfreq, hfreq, notch=None):
    hfreq = min(hfreq, sfreq/2 - 1)
    out = mne.filter.filter_data(data_uv.astype(np.float64), sfreq, lfreq, hfreq,
                                 verbose=False)
    if notch is not None and notch < sfreq/2:
        out = mne.filter.notch_filter(out, sfreq, [notch], verbose=False)
    return out

def resample_to_target(x, sfreq):
    return mne.filter.resample(x, up=TARGET_SFREQ, down=int(round(sfreq)), verbose=False)

def corrected_1s_amps(emg_filt_uv, sfreq):
    """Ferri: rectify -> mean per 1-s -> subtract rolling-min(60). Used for subject RAI."""
    sps  = int(round(sfreq))
    rect = np.abs(emg_filt_uv)
    n_sec = rect.size // sps
    mini = rect[:n_sec*sps].reshape(n_sec, sps).mean(axis=1)
    return np.clip(mini - minimum_filter1d(mini, size=NOISE_WIN_SEC, mode='nearest'), 0, None)

def rai(corrected_uv):
    """Ferri REM Atonia Index over a set of corrected 1-s amplitudes."""
    if corrected_uv.size == 0:
        return np.nan
    p_atonic = np.mean(corrected_uv <= ATONIA_UV)
    p_uncert = np.mean((corrected_uv > ATONIA_UV) & (corrected_uv <= UNCERTAIN_UV))
    denom = 1.0 - p_uncert
    return float(p_atonic / denom) if denom > 1e-9 else np.nan

# ── Per-subject processing ──────────────────────────────────────────────────────
def process_subject(subj, h5_group):
    edf_path = RAW_DIR / f'{subj}.edf'
    txt_path = RAW_DIR / f'{subj}.txt'

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    sfreq = raw.info['sfreq']
    emg_ch, eeg_ch, eog_ch = get_channels(raw)
    if emg_ch is None:
        print(f"  {subj}: no EMG — skipping"); raw.close(); return None

    def sig_uv(ch):
        if ch is None: return None
        return raw.get_data(picks=[raw.ch_names.index(ch)])[0] * 1e6   # V -> uV

    emg_f = bandpass(sig_uv(emg_ch), sfreq, EMG_LFREQ, EMG_HFREQ, notch=NOTCH_FREQ)
    eeg_f = bandpass(sig_uv(eeg_ch), sfreq, 0.5, 40.0) if eeg_ch else None
    eog_f = bandpass(sig_uv(eog_ch), sfreq, 0.5, 20.0) if eog_ch else None

    corr = corrected_1s_amps(emg_f, sfreq)             # native 1-s corrected amps (RAI)
    n_sec_native = corr.size

    emg = resample_to_target(emg_f, sfreq)
    eeg = resample_to_target(eeg_f, sfreq) if eeg_f is not None else None
    eog = resample_to_target(eog_f, sfreq) if eog_f is not None else None
    lens = [len(emg)] + [len(s) for s in (eeg, eog) if s is not None]
    n_sec = min([n_sec_native] + [L // TARGET_SFREQ for L in lens])
    emg = emg[:n_sec*TARGET_SFREQ]
    if eeg is not None: eeg = eeg[:n_sec*TARGET_SFREQ]
    if eog is not None: eog = eog[:n_sec*TARGET_SFREQ]

    # ── Staging -> per-second stage map + valid REM 30-s epoch onsets ────────
    stages_abs = parse_staging(txt_path)
    if not stages_abs:
        print(f"  {subj}: no staging — skipping"); raw.close(); return None
    rec = raw.info['meas_date']
    rec_sec = rec.hour*3600 + rec.minute*60 + rec.second if rec else 0
    stages_rel = sorted(to_relative(stages_abs, rec_sec))
    raw.close()

    sec_stage = np.empty(n_sec, dtype=object); sec_stage[:] = ''
    valid_rem = []
    for i, (onset, stage) in enumerate(stages_rel):
        s0 = int(round(onset))
        if s0 < 0 or s0 + EPOCH_SEC > n_sec:
            continue
        sec_stage[s0:s0+EPOCH_SEC] = stage
        if stage == REM_STAGE and i+1 < len(stages_rel) \
           and stages_rel[i+1][0] - onset == EPOCH_SEC:
            valid_rem.append(s0)

    nrem_mask = np.array([s in NREM_STAGES for s in sec_stage])
    rem_mask  = np.array([s == REM_STAGE  for s in sec_stage])

    # ── Keep PHYSICAL microvolts — NO per-subject normalisation ──────────────
    # The per-mini label is defined on ABSOLUTE background-corrected uV (see below /
    # Module 1), so rescaling each subject by its own statistics would erase the very
    # amplitude the label depends on. Worse, NREM tone is confounded with disease
    # (higher in RBD), so per-subject scaling shrinks RSWA in the sickest patients —
    # the same trap we avoided in the rule by using absolute uV. We therefore store
    # filtered uV and let the notebook apply ONE global scale shared by all subjects.
    emg_z = emg                                   # chin EMG, filtered microvolts (no per-subject scaling)
    eeg_z = eeg                                   # EEG, filtered microvolts (or None)
    eog_z = eog                                   # EOG, filtered microvolts (or None)

    # ── Corrected 0.1-s bins (for per-mini phasic/tonic on the atonia scale) ──
    rect = np.abs(emg)
    dsp  = TARGET_SFREQ // 10                                     # 0.1s = 20 samples
    n_ds = len(rect) // dsp
    a01  = rect[:n_ds*dsp].reshape(n_ds, dsp).mean(axis=1)
    a01c = np.clip(a01 - minimum_filter1d(a01, NOISE_WIN_SEC * 10), 0, None)
    BINS = MINI_SEC * 10                                         # 30 (0.1-s bins per mini)

    # ── Build 3-s REM mini-epochs + per-mini labels (any = phasic OR tonic) ───
    signals, any_list, phasic_list, tonic_list = [], [], [], []
    for s0 in valid_rem:                                        # valid 30-s REM epochs
        for k in range(0, EPOCH_SEC, MINI_SEC):                 # 10 minis per epoch
            ms = s0 + k                                         # mini start (sec)
            a = ms * TARGET_SFREQ
            b = a + MINI_SAMPLES
            if b > len(emg_z):
                continue
            bins = a01c[ms*10: ms*10 + BINS]
            if bins.size < BINS:
                continue
            is_phasic = bool(bins.max() > PHASIC_UV)
            is_tonic  = bool(np.mean(bins > TONIC_UV) >= TONIC_FRAC)
            ch = []
            for sig in (eeg_z, eog_z, emg_z):
                ch.append(sig[a:b] if sig is not None else np.zeros(MINI_SAMPLES))
            signals.append(np.stack(ch, axis=0).astype(np.float32))
            any_list.append(int(is_phasic or is_tonic))
            phasic_list.append(int(is_phasic))
            tonic_list.append(int(is_tonic))

    if not signals:
        print(f"  {subj}: no valid REM mini-epochs"); return None

    signals    = np.stack(signals, axis=0)
    labels_any = np.array(any_list,    dtype=np.int8)
    phasic_arr = np.array(phasic_list, dtype=np.int8)
    tonic_arr  = np.array(tonic_list,  dtype=np.int8)
    n_mini = len(labels_any)

    any_density    = float(labels_any.mean() * 100)             # == guideline "any" %
    phasic_density = float(phasic_arr.mean() * 100)
    tonic_density  = float(tonic_arr.mean() * 100)
    rai_rem  = rai(corr[rem_mask])
    rai_nrem = rai(corr[nrem_mask])
    dx = 'RBD' if any_density > SINBAR_ANY_CUT else 'ctrl'

    print(f"  {subj:8s}: EMG={emg_ch:9s} | {n_mini:5d} minis ({len(valid_rem):3d} REM ep) | "
          f"any {any_density:5.1f}% (P{phasic_density:4.1f}/T{tonic_density:4.1f}) "
          f"-> {dx:4s} | RAI={rai_rem:.3f}")

    grp = h5_group.create_group(subj)
    grp.create_dataset('signals',   data=signals, compression='gzip')
    grp.create_dataset('labels',    data=labels_any)
    grp.create_dataset('is_phasic', data=phasic_arr)
    grp.create_dataset('is_tonic',  data=tonic_arr)
    grp.create_dataset('n_mini',    data=n_mini)
    grp.attrs.update(emg_ch=emg_ch or 'missing', eeg_ch=eeg_ch or 'missing',
                     eog_ch=eog_ch or 'missing', n_rem_epochs=len(valid_rem),
                     any_density=any_density, phasic_density=phasic_density,
                     tonic_density=tonic_density, dx=dx,
                     rai_rem=float(rai_rem), rai_nrem=float(rai_nrem))

    return dict(subj=subj, n_mini=n_mini, any=any_density, dx=dx, rai=rai_rem)

# ── Main ────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("="*70)
    print(f"Preprocessing @ {TARGET_SFREQ}Hz | 3-s REM mini-epochs | "
          f"label: any = phasic(>{PHASIC_UV}uV) OR tonic(>{TONIC_UV}uV,>={TONIC_FRAC})")
    print("="*70)

    stats = {'control': [], 'rbd': []}
    with h5py.File(H5_PATH, 'w') as h5:
        h5.attrs.update(sfreq=TARGET_SFREQ, mini_sec=MINI_SEC, mini_samples=MINI_SAMPLES,
                        channels=['EEG','EOG','EMG'],
                        label='any = phasic OR tonic per 3-s REM mini-epoch (CNN target)',
                        tonic_uv=TONIC_UV, phasic_uv=PHASIC_UV,
                        sinbar_any_cut=SINBAR_ANY_CUT)
        for label, subjects in [('control', CONTROLS), ('rbd', RBDS)]:
            print(f"\n{label.upper()}")
            grp = h5.create_group(label)
            for subj in subjects:
                r = process_subject(subj, grp)
                if r: stats[label].append(r)

    # ── Verification: subject "any" density + guideline diagnosis ────────────
    print("\n" + "="*70)
    print(f"VERIFICATION — subject 'any' density (%) and rule-based DX (cut {SINBAR_ANY_CUT}%)")
    print("="*70)
    for label in ('control', 'rbd'):
        print(f"\n{label.upper()} ({len(stats[label])} subj):   "
              f"{'subj':>8s} {'any%':>7s} {'RAI':>7s} {'DX':>5s}")
        for r in sorted(stats[label], key=lambda x: x['any']):
            print(f"  {'':6s} {r['subj']:>8s} {r['any']:6.1f}% {r['rai']:7.3f} {r['dx']:>5s}")

    ctrl, rbd = stats['control'], stats['rbd']
    fp = [r['subj'] for r in ctrl if r['dx'] == 'RBD']
    fn = [r['subj'] for r in rbd  if r['dx'] == 'ctrl']
    if ctrl and rbd:
        sens = (len(rbd) - len(fn)) / len(rbd) * 100
        spec = (len(ctrl) - len(fp)) / len(ctrl) * 100
        print(f"\nRule-based DX (any density > {SINBAR_ANY_CUT}%): "
              f"sens {sens:.0f}% ({len(rbd)-len(fn)}/{len(rbd)}), "
              f"spec {spec:.0f}% ({len(ctrl)-len(fp)}/{len(ctrl)})")
        print(f"  false+ (control->RBD): {fp or 'none'}")
        print(f"  false- (RBD->control): {fn or 'none'}")

    tot = sum(r['n_mini'] for g in stats.values() for r in g)
    print(f"\nTotal 3-s minis: {tot} | Output: {H5_PATH} "
          f"({H5_PATH.stat().st_size/1e6:.1f} MB)")
    print("Done. Paste VERIFICATION. (CNN classifies 3-s minis; aggregate = guideline 'any' density.)")
