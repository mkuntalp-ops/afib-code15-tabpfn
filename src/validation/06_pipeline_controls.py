"""
Stage 6 - Pipeline checks: positive and negative controls (Methods II-H, Results III-O).

Positive control: AF-labelled records (sentinel_af) vs sinus-rhythm records (cases and
controls), RR features only. Tests that the pipeline detects a signal that must be present.
Negative control: in the seed-42 matched cohort, cases vs controls with RR features only;
share of case records scoring above the 95th percentile of control scores, and share of
records with a delineated P wave in fewer than three leads (contamination by AF would make
both large).

Input : outputs/cohort_clean.csv, rr_features.csv, pwave_features.csv, matched_seed42.csv
Output: outputs/06_pipeline_controls.json
"""
import sys
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from common import OUT, oof_pred, save_json

RR_FEATURES = ["meanRR", "SDNN", "RMSSD", "n_RR", "dur_s"]


def main():
    cc = pd.read_csv(OUT / "cohort_clean.csv")[["exam_id", "patient_id", "split"]]
    rr = pd.read_csv(OUT / "rr_features.csv")
    d = cc.merge(rr, on="exam_id")
    d = d[(d.n_R >= 6) & d.split.isin(["sentinel_af", "vaka", "kontrol"])].copy()
    d["split"] = np.where(d.split == "sentinel_af", "sentinel_af", "sinus")
    y, p = oof_pred(d, RR_FEATURES, n_repeats=5, label="sentinel_af")
    res = {"positive_control": {"n_AF_records": int(y.sum()), "n_sinus": int((y == 0).sum()),
                                "AUROC": float(roc_auc_score(y, p))}}
    m = pd.read_csv(OUT / "matched_seed42.csv")
    y, p = oof_pred(m, RR_FEATURES)
    thr = np.percentile(p[y == 0], 95)
    res["negative_control"] = {"AUROC_RR_only": float(roc_auc_score(y, p)),
                               "case_frac_above_ctrl95": float((p[y == 1] > thr).mean()),
                               "P_wave_in_lt3_leads": {g: float((m[m.split == g].n_lead_ok < 3).mean())
                                                       for g in ["vaka", "kontrol"]}}
    save_json(res, "06_pipeline_controls.json"); print(res)


if __name__ == "__main__":
    main()
