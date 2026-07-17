# Data preparation scripts

A slim, self-contained pipeline that regenerates the workshop dataset from the public
**PhysioNet CAP Sleep Database** — provided for transparency and reproducibility.
Participants do **not** need these: the notebook downloads the prepared data from the
GitHub Release automatically.

All scripts read the data directory from the `RBD_DATA` environment variable
(default `rbd_workshop/data`), which holds `raw/` (downloads) and `processed/` (outputs).

## Requirements

```bash
pip install mne h5py numpy scipy edfio
```

## Steps

Run these from the repository root (the folder name has spaces, so keep the quotes):

```bash
export RBD_DATA=./rbd_workshop/data                        # where to store raw + processed data

bash   "data preprocessing scripts/download_cap.sh"        # 1) download the CAP records into $RBD_DATA/raw
python "data preprocessing scripts/preprocess.py"          # 2) -> $RBD_DATA/processed/workshop_data.h5 (3-s REM minis + labels)
python "data preprocessing scripts/shrink_for_upload.py"   # 3) -> workshop_data_fp16.h5 (signals cast to float16)
python "data preprocessing scripts/make_demo_clips.py"     # 4) -> clip_control.* / clip_rbd.* (Module 1 demo clips)
```

Then attach `workshop_data_fp16.h5` and the four `clip_*` files to the GitHub Release
(tag `data-v1`) — the notebook downloads them from there.

## What the pipeline does

- Filters the chin EMG (1–100 Hz band-pass + 50 Hz notch), resamples every channel to
  200 Hz, finds valid REM 30-second epochs and splits them into **3-second mini-epochs**
  (600 samples).
- Labels each mini on the background-corrected (rolling-min) µV chin EMG: `is_phasic`
  (a burst > 3 µV), `is_tonic` (> 2 µV for ≥ 50 % of the mini), and `labels` =
  **"any" = phasic OR tonic** (the CNN target).
- Stores `signals (n_mini, 3, 600)` `[EEG, EOG, EMG]` in **physical microvolts** (the
  notebook standardizes globally at load — not per subject), plus per-subject
  `any_density` and the Ferri REM Atonia Index.

Roster used: controls `n1, n2, n3, n4, n5, n8, n10, n11`; RBD `rbd1–rbd10, rbd12–rbd22`
(see the top of `preprocess.py` for the exact subject lists and thresholds).

## Data license

PhysioNet CAP Sleep Database — Open Data Commons Attribution License (ODC-By). Please
cite PhysioNet and the CAP Sleep Database when redistributing the data or derived files.
