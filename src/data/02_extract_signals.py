"""
Stage 2 - Waveform extraction, zero-padding removal and signal QC (Methods II-D).

Extracts the cohort's exams either from the 18 CODE-15% HDF5 parts
(data/exams_part*.hdf5) or, if those are absent, from pre-extracted per-part archives
(data/cohort_tracings/exams_part*.npz, each holding 'exam_id' and 'tracings').
Records shorter than 10.24 s are zero-padded to 4,096 samples; the padding is removed
before any filtering. QC keeps the two standard lengths (2,934 samples = 7.33 s,
4,096 samples = 10.24 s); empty and non-standard records are excluded.

Note: in the manuscript analysis the archive part exams_part0.hdf5 was not extracted, so
its records (about 6% of the cohort) are absent. Set SKIP_PARTS = {} to include all parts.

Input : outputs/cohort_raw.csv, data/exams_part*.hdf5 or data/cohort_tracings/*.npz
Output: outputs/cohort_signals.npz, outputs/signal_bounds.csv, outputs/cohort_clean.csv
"""
import sys, glob, os
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from common import DATA, OUT

SKIP_PARTS = {"exams_part0"}          # reproduces the manuscript analysis (see note above)
STANDARD_LENGTHS = {2934, 4096}


def strip_zero_padding(tracing):
    nz = np.where(np.any(tracing != 0, axis=1))[0]
    if len(nz) == 0:
        return -1, -1
    return int(nz[0]), int(nz[-1])


def iter_parts():
    h5 = sorted(glob.glob(str(DATA / "exams_part*.hdf5")))
    if h5:
        import h5py
        for path in h5:
            name = os.path.basename(path).split(".")[0]
            if name in SKIP_PARTS:
                continue
            with h5py.File(path, "r") as f:
                yield name, np.array(f["exam_id"]).astype(int), f["tracings"]
    else:
        for path in sorted(glob.glob(str(DATA / "cohort_tracings" / "exams_part*.npz"))):
            name = os.path.basename(path).split(".")[0]
            if name in SKIP_PARTS:
                continue
            z = np.load(path)
            yield name, z["exam_id"].astype(int), z["tracings"]


def main():
    cohort = pd.read_csv(OUT / "cohort_raw.csv")
    target = set(cohort.exam_id.astype(int))
    ids, tracings = [], []
    for name, file_ids, tr in iter_parts():
        idx = np.sort(np.where(np.isin(file_ids, list(target)))[0])
        if len(idx):
            ids.append(file_ids[idx]); tracings.append(np.asarray(tr[idx, :, :], dtype=np.float32))
        print(f"  {name}: {len(idx)} records")
    ids = np.concatenate(ids); tracings = np.concatenate(tracings)
    np.savez_compressed(OUT / "cohort_signals.npz", exam_id=ids, tracings=tracings)

    rows = []
    for i, e in enumerate(ids):
        s, t = strip_zero_padding(tracings[i])
        rows.append((int(e), s, t, (t - s + 1) if s >= 0 else 0))
    b = pd.DataFrame(rows, columns=["exam_id", "start", "end", "length"])
    b.to_csv(OUT / "signal_bounds.csv", index=False)

    avail = cohort[cohort.exam_id.isin(ids)]
    ok = set(b[b.length.isin(STANDARD_LENGTHS)].exam_id)
    clean = avail[avail.exam_id.isin(ok)].merge(b, on="exam_id")
    clean.to_csv(OUT / "cohort_clean.csv", index=False)
    print(f"\nWaveforms found: {len(avail)} of {len(cohort)} cohort records")
    print("  by split:", avail.split.value_counts().to_dict())
    print(f"Excluded at QC (empty or non-standard length): {len(avail) - len(clean)}")
    print("  after QC by split:", clean.split.value_counts().to_dict())


if __name__ == "__main__":
    main()
