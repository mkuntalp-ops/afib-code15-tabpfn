"""
Stage 5 - Classic P-terminal force in V1 (Morris index; Methods II-E, Table III).

Input : outputs/cohort_signals.npz, outputs/cohort_clean.csv
Output: outputs/ptfv1_true.csv
"""
import sys
sys.path.insert(0, "src"); sys.path.insert(0, "src/features")
from multiprocessing import Pool
import pandas as pd
from common import OUT
import ecg_features as F


def work(item):
    exam_id, seg = item
    return {"exam_id": exam_id, "PTFV1_true": F.ptfv1_classic(seg)}


if __name__ == "__main__":
    items = F.load_segments(OUT)
    with Pool(8) as p:
        rows = p.map(work, items, chunksize=16)
    pd.DataFrame(rows).to_csv(OUT / "ptfv1_true.csv", index=False)
    print(f"-> ptfv1_true.csv ({len(rows)} records)")
