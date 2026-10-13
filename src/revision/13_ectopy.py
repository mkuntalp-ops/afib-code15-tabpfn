"""
Revision 13 - Approximate ectopy quantification and RMSSD without ectopic intervals
(Methods II-I, Results III-O). Premature beat: RR_k < 0.80 x median of up to four
neighbouring RR values, followed by RR_{k+1} > 1.10 x that median (short-long pattern).

Input : outputs/cohort_signals.npz, cohort_clean.csv, matched_seed42.csv, feature tables
Output: outputs/13_ectopy.json, outputs/13_ectopy_records.csv
"""
import sys
sys.path.insert(0, "src"); sys.path.insert(0, "src/features")
from multiprocessing import Pool
import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy.signal import filtfilt
from sklearn.metrics import roc_auc_score
from common import OUT, FINAL3, load_pool, match_once, cv_auc, save_json
import ecg_features as F


def rr_series(item):
    exam_id, seg = item
    x = filtfilt(F.BB, F.BA, seg[:, F.LEAD_II])
    try:
        rp = nk.ecg_peaks(x, sampling_rate=F.FS)[1]["ECG_R_Peaks"]
    except Exception:
        rp = np.array([])
    rr = np.diff(rp) / F.FS * 1000
    return exam_id, rr[(rr > 300) & (rr < 2000)]


def flag_ectopic(rr):
    n = len(rr); flag = np.zeros(n, bool)
    for k in range(n - 1):
        nb = [rr[j] for j in (k - 2, k - 1, k + 1, k + 2) if 0 <= j < n]
        if len(nb) >= 2:
            ref = np.median(nb)
            flag[k] = rr[k] < 0.80 * ref and rr[k + 1] > 1.10 * ref
    return flag


def rmssd_excluding(rr, flag):
    if len(rr) < 3:
        return np.nan
    d = np.diff(rr); use = d[~(flag[:-1] | flag[1:])]
    return float(np.sqrt(np.mean(use ** 2))) if len(use) else np.nan


def main():
    pool = load_pool()
    items = [it for it in F.load_segments(OUT) if it[0] in set(pool.exam_id)]
    with Pool(8) as p:
        series = p.map(rr_series, items, chunksize=16)
    rows = []
    for e, rr in series:
        if len(rr) >= 5:
            fl = flag_ectopic(rr)
            rows.append(dict(exam_id=e, n_flag=int(fl.sum()), RMSSD_noect=rmssd_excluding(rr, fl)))
    E = pd.DataFrame(rows); E.to_csv(OUT / "13_ectopy_records.csv", index=False)
    P = pool.merge(E, on="exam_id", how="left"); P["has_ectopy"] = P.n_flag.fillna(0) > 0
    res = {"pool_has_ectopy": {g: float(s.has_ectopy.mean()) for g, s in P.groupby("split")}}
    m = pd.read_csv(OUT / "matched_seed42.csv").merge(E, on="exam_id", how="left")
    m["has_ectopy"] = m.n_flag.fillna(0) > 0
    m["RMSSD_noect"] = m.RMSSD_noect.fillna(m.RMSSD)
    res["matched_has_ectopy"] = {g: float(s.has_ectopy.mean()) for g, s in m.groupby("split")}
    res["matched_RMSSD_median"] = {g: float(s.RMSSD.median()) for g, s in m.groupby("split")}
    res["matched_RMSSD_noect_median"] = {g: float(s.RMSSD_noect.median()) for g, s in m.groupby("split")}
    res["AUROC"] = {"RMSSD": cv_auc(m, ["RMSSD"]), "RMSSD_noect": cv_auc(m, ["RMSSD_noect"]),
                    "Final3": cv_auc(m, FINAL3), "Final3_noect": cv_auc(m, ["RMSSD_noect", "PR_med", "Pamp_V1"])}
    free = set(m[~m.has_ectopy].exam_id)
    _, pairs = match_once(pool, 42)
    both = [(c, k) for c, k in pairs if c in free and k in free]
    sub = m[m.exam_id.isin([x for pr in both for x in pr])]
    res["ectopy_free_pairs"] = {"pairs": len(both), "RMSSD": cv_auc(sub, ["RMSSD"]), "Final3": cv_auc(sub, FINAL3)}
    save_json(res, "13_ectopy.json"); print(res)


if __name__ == "__main__":
    main()
