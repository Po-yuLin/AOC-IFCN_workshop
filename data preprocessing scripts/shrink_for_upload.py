"""
04_shrink_for_upload.py
Shrink the processed data for distribution — run LOCALLY, before uploading.

Casts the float32 `signals` to float16 (precision is far more than enough for a
z-scored CNN input) and rewrites the preview .npz files with compression + float16.
Produces *_fp16 copies alongside the originals; upload the *_fp16 files.

In the Colab notebook, cast signals back to float32 when loading them into tensors.

Run: python scripts/04_shrink_for_upload.py
"""

import h5py
import numpy as np
import os
from pathlib import Path

OUT = Path(os.environ.get('RBD_DATA', 'rbd_workshop/data')) / 'processed'
SRC_H5 = OUT / 'workshop_data.h5'
DST_H5 = OUT / 'workshop_data_fp16.h5'


def _copy(sg, dg):
    for k, v in sg.attrs.items():
        dg.attrs[k] = v
    for key, item in sg.items():
        if isinstance(item, h5py.Group):
            _copy(item, dg.create_group(key))
        else:
            arr = item[()]
            if key == 'signals':                       # only the big array shrinks
                dg.create_dataset(key, data=arr.astype(np.float16), compression='gzip')
            else:
                dg.create_dataset(key, data=arr)
            for ak, av in item.attrs.items():
                dg[key].attrs[ak] = av


if __name__ == '__main__':
    with h5py.File(SRC_H5, 'r') as s, h5py.File(DST_H5, 'w') as d:
        _copy(s, d)
    print(f"h5      : {os.path.getsize(SRC_H5)/1e6:7.1f} MB  ->  "
          f"{os.path.getsize(DST_H5)/1e6:7.1f} MB (fp16)")

    for tag in ('control', 'rbd'):
        p = OUT / f'preview_{tag}.npz'
        if not p.exists():
            continue
        data = np.load(p, allow_pickle=True)
        out = {k: (data[k].astype(np.float16) if data[k].dtype.kind == 'f' else data[k])
               for k in data.files}
        q = OUT / f'preview_{tag}_fp16.npz'
        np.savez_compressed(q, **out)
        print(f"preview_{tag}: {os.path.getsize(p)/1e6:7.1f} MB  ->  "
              f"{os.path.getsize(q)/1e6:7.1f} MB")

    print("\nUpload the *_fp16 files. Notebook loads signals then casts to float32.")
