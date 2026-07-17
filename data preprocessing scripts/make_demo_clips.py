"""
05_make_demo_clips.py
Cut short, REM-containing EDF clips from two USED subjects for the Module 1
hands-on demo (run preprocessing live + visualize), so we don't ship the npz.

Output (small, ~5 MB each): data/processed/
  clip_control.edf + clip_control.txt   (from n1,  chin = EMG1-EMG2)
  clip_rbd.edf     + clip_rbd.txt        (from rbd2, chin = EMG1-EMG2)

Each clip keeps the 3 workshop channels at native rate, plus a matching staging
.txt whose clock times are re-derived from the exported clip so the existing
preprocessing time logic lines up exactly.

Needs an EDF exporter:  pip install edfio   (if export raises ImportError)
Run: python scripts/05_make_demo_clips.py
"""

import mne
import os
import numpy as np
from pathlib import Path
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

RAW_DIR = Path(os.environ.get('RBD_DATA', 'rbd_workshop/data')) / 'raw'
OUT_DIR = Path(os.environ.get('RBD_DATA', 'rbd_workshop/data')) / 'processed'
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEMO = [('control', 'n1'), ('rbd', 'rbd2')]     # used subjects; both chin = EMG1-EMG2
WINDOW_SEC = 1800                                # 30-min clip
PRE_REM_SEC = 300                                # start 5 min before the first REM
REM_STAGE = 'SLEEP-REM'

EMG_C = ['EMG1-EMG2', 'EMG-EMG', 'EMG', 'CHIN-0', 'CHIN1', 'milo']
EEG_C = ['C4-A1', 'C4-P4', 'Fp2-F4', 'F4-C4']
EOG_C = ['ROC-LOC', 'EOG dx', 'EOG-L', 'EOG sin']


def find(chs, cands):
    return next((c for c in cands if c in chs), None)


def parse_staging(txt):
    out = []
    with open(txt, 'r', errors='ignore') as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 5:
                continue
            ev = next((x for x in p if x.startswith('SLEEP-')), None)
            t  = next((x for x in p if ':' in x and len(x) == 8), None)
            if ev and t:
                h, m, s = t.split(':')
                out.append((int(h)*3600 + int(m)*60 + int(s), ev))
    return out


def make_clip(tag, subj):
    raw = mne.io.read_raw_edf(RAW_DIR / f'{subj}.edf', preload=True, verbose=False)
    sfreq = raw.info['sfreq']
    emg = find(raw.ch_names, EMG_C); eeg = find(raw.ch_names, EEG_C); eog = find(raw.ch_names, EOG_C)
    keep = [c for c in (eeg, eog, emg) if c]
    raw.pick(keep)

    md = raw.info['meas_date']
    rec_sec = (md.hour*3600 + md.minute*60 + md.second) if md else 0
    stages = []
    for oa, st in parse_staging(RAW_DIR / f'{subj}.txt'):
        rel = oa - rec_sec
        if rel < -3600:
            rel += 86400
        stages.append((rel, st))
    stages.sort()

    rem = [r for r, s in stages if s == REM_STAGE and 0 <= r <= raw.times[-1]]
    if not rem:
        print(f"  {subj}: no REM found — skipping"); raw.close(); return
    t0 = max(0.0, rem[0] - PRE_REM_SEC)
    t1 = min(raw.times[-1], t0 + WINDOW_SEC)
    raw.crop(tmin=t0, tmax=t1)

    out_edf = OUT_DIR / f'clip_{tag}.edf'
    out_txt = OUT_DIR / f'clip_{tag}.txt'
    try:
        mne.export.export_raw(out_edf, raw, fmt='edf', overwrite=True)
    except Exception as e:
        print(f"  {subj}: EDF export failed ({e}). Try: pip install edfio")
        raw.close(); return
    raw.close()

    # Re-read the exported clip and derive staging clock times from ITS meas_date,
    # so preprocessing (rel = clock - clip_start) indexes the clip signal correctly.
    clip = mne.io.read_raw_edf(out_edf, preload=False, verbose=False)
    cs = clip.info['meas_date']
    clip_len = clip.n_times / clip.info['sfreq']
    ch_names = clip.ch_names
    clip.close()

    n = 0
    with open(out_txt, 'w') as f:
        f.write("Sleep Stage\tPosition\tTime [hh:mm:ss]\tEvent\tDuration[s]\n")
        for rel, st in stages:
            if t0 <= rel < t0 + clip_len:
                sec = rel - t0
                if cs is not None:
                    clock = (cs + timedelta(seconds=float(sec))).strftime('%H:%M:%S')
                else:
                    h = int(sec // 3600) % 24; m = int(sec % 3600 // 60); s = int(sec % 60)
                    clock = f"{h:02d}:{m:02d}:{s:02d}"
                f.write(f"{st}\tSUPINE\t{clock}\t{st}\t30\n")
                n += 1

    mb = out_edf.stat().st_size / 1e6
    n_rem = sum(1 for r, s in stages if s == REM_STAGE and t0 <= r < t0 + clip_len)
    print(f"  {tag:8s} ({subj}): {out_edf.name} {mb:.1f}MB | channels={ch_names} | "
          f"{n} staged epochs ({n_rem} REM) in clip")


if __name__ == '__main__':
    print("Cutting Module-1 demo clips (REM-containing, 3 channels, native rate)...")
    for tag, subj in DEMO:
        make_clip(tag, subj)
    print("\nUpload clip_control.* and clip_rbd.* instead of the preview .npz.")
