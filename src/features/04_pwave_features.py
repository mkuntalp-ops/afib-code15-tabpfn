"""
Stage 4 - Atrial conduction and depolarisation features, 12 leads (Methods II-E, II-F).

Input : outputs/cohort_signals.npz, outputs/cohort_clean.csv
Output: outputs/pwave_features.csv (per-lead PR_k, Pamp_k, Pdur_k and record-level PR_med,
        PR_max, Pamp_V1, PTFV1_proxy, Pdur_med, Pdisp, n_lead_ok)
"""
import sys
sys.path.insert(0, "src"); sys.path.insert(0, "src/features")
from multiprocessing import Pool
import pandas as pd
from common import OUT
import ecg_features as F


def work(item):
    exam_id, seg = item
    return dict(exam_id=exam_id, **F.pwave_features(seg))


if __name__ == "__main__":
    items = F.load_segments(OUT)
    with Pool(8) as p:
        rows = p.map(work, items, chunksize=8)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "pwave_features.csv", index=False)
    print(f"-> pwave_features.csv ({len(df)} records); PR_med coverage {df.PR_med.notna().mean():.3f}, "
          f"Pamp_V1 coverage {df.Pamp_V1.notna().mean():.3f}")
