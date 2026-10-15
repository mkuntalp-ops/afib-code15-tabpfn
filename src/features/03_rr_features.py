"""
Stage 3 - Beat-to-beat variability features from lead II (Methods II-D, II-E).

All QC-passed records are written; the signal-quality rule (at least six detected R peaks,
i.e. five RR intervals) is applied at the matching stage through n_R.

Input : outputs/cohort_signals.npz, outputs/cohort_clean.csv
Output: outputs/rr_features.csv
"""
import sys
sys.path.insert(0, "src"); sys.path.insert(0, "src/features")
from multiprocessing import Pool
import pandas as pd
from scipy.signal import filtfilt
from common import OUT
import ecg_features as F


def work(item):
    exam_id, seg = item
    x = filtfilt(F.BB, F.BA, seg[:, F.LEAD_II])
    return dict(exam_id=exam_id, dur_s=len(x) / F.FS, **F.rr_features(x))


if __name__ == "__main__":
    items = F.load_segments(OUT)
    with Pool(8) as p:
        rows = p.map(work, items, chunksize=16)
    pd.DataFrame(rows).to_csv(OUT / "rr_features.csv", index=False)
    print(f"-> rr_features.csv ({len(rows)} records)")
