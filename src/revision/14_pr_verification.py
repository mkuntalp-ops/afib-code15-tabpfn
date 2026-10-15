"""
Revision 14 - Verification of the PR and heart-rate measurements (Methods II-G, Results III-S,
Fig. 7): device-label AUROCs, inter-lead agreement, a second delineator (NeuroKit2 CWT, lead
II, aligned to each R peak) and the full analysis repeated with the CWT PR interval.

Input : outputs/cohort_signals.npz, cohort_clean.csv, feature tables, data/exams.csv
Output: outputs/14_pr_verification.json, outputs/14_pr_cwt.csv
"""
import sys
sys.path.insert(0, "src"); sys.path.insert(0, "src/features")
from multiprocessing import Pool
import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy.signal import filtfilt
from sklearn.metrics import roc_auc_score
from common import OUT, EXAMS, SEEDS, FINAL3, ATR, load_pool, match_once, cv_auc, boot_auc_ci, icc21, save_json
import ecg_features as F


def pr_aligned(w, rp):
    on = np.array(w["ECG_P_Onsets"], float); ron = np.array(w["ECG_R_Onsets"], float); prs = []
    for p_ in rp:
        co = on[(~np.isnan(on)) & (on < p_) & (on > p_ - 0.4 * F.FS)]
        cr = ron[(~np.isnan(ron)) & (ron < p_) & (ron > p_ - 0.15 * F.FS)]
        if len(co) and len(cr):
            pr = (cr[-1] - co[-1]) / F.FS * 1000
            if 60 < pr < 400:
                prs.append(pr)
    return float(np.median(prs)) if len(prs) >= 2 else np.nan


def both_methods(item):
    exam_id, seg = item
    x = filtfilt(F.BB, F.BA, seg[:, F.LEAD_II]); out = {"exam_id": exam_id}
    try:
        rp = nk.ecg_peaks(x, sampling_rate=F.FS)[1]["ECG_R_Peaks"]
    except Exception:
        return out
    for method in ["dwt", "cwt"]:
        try:
            out[f"PR_{method}"] = pr_aligned(nk.ecg_delineate(x, rp, sampling_rate=F.FS, method=method)[1], rp)
        except Exception:
            out[f"PR_{method}"] = np.nan
    return out


def main():
    res = {}
    pool = load_pool()
    pw = pd.read_csv(OUT / "pwave_features.csv")
    leads = [f"PR_{i}" for i in range(12)]
    full = pw.dropna(subset=leads)
    res["inter_lead"] = {"n_all12": int(len(full)), "ICC21": float(icc21(full[leads].values)),
                         "abs_dev_median": float(np.nanmedian(pw[leads].sub(pw[leads].median(axis=1), axis=0).abs().values)),
                         "abs_dev_p90": float(np.nanpercentile(pw[leads].sub(pw[leads].median(axis=1), axis=0).abs().values, 90)),
                         "leads_measured_median": float(pw[leads].notna().sum(axis=1).median())}
    items = [it for it in F.load_segments(OUT) if it[0] in set(pool.exam_id)]
    with Pool(8) as p:
        M = pd.DataFrame(p.map(both_methods, items, chunksize=8))
    M.to_csv(OUT / "14_pr_cwt.csv", index=False)
    rng = np.random.default_rng(7)
    sample = M.loc[rng.choice(len(M), 400, replace=False)].dropna(subset=["PR_dwt", "PR_cwt"])
    d = sample.PR_dwt - sample.PR_cwt
    res["dwt_vs_cwt_leadII"] = {"n": int(len(sample)), "bias": float(d.mean()),
                                "LoA": [float(d.mean() - 1.96 * d.std(ddof=1)), float(d.mean() + 1.96 * d.std(ddof=1))],
                                "ICC21": float(icc21(sample[["PR_dwt", "PR_cwt"]].values)),
                                "spearman": float(sample[["PR_dwt", "PR_cwt"]].corr("spearman").iloc[0, 1])}
    ex = pd.read_csv(EXAMS, usecols=["exam_id", "1dAVb", "SB", "ST"])
    Q = pool.merge(ex, on="exam_id").merge(M[["exam_id", "PR_cwt"]], on="exam_id", how="left")
    q = Q.dropna(subset=["PR_med"]); y = q["1dAVb"].astype(int).values
    res["label_1dAVb"] = {"n": int(len(q)), "n_pos": int(y.sum()), "AUROC": float(roc_auc_score(y, q.PR_med)),
                          "CI": boot_auc_ci(y, q.PR_med.values, n=1000), "median_PR_pos": float(q[q["1dAVb"]].PR_med.median()),
                          "gt200_pos": float((q[q["1dAVb"]].PR_med > 200).mean())}
    qc = Q.dropna(subset=["PR_cwt"]); yc = qc["1dAVb"].astype(int).values
    res["label_1dAVb_cwt"] = {"AUROC": float(roc_auc_score(yc, qc.PR_cwt)), "median_PR_pos": float(qc[qc["1dAVb"]].PR_cwt.median()),
                              "gt200_pos": float((qc[qc["1dAVb"]].PR_cwt > 200).mean())}
    hr = 60000 / Q.meanRR
    res["label_SB"] = {"n_pos": int(Q.SB.sum()), "AUROC": float(roc_auc_score(Q.SB.astype(int), -hr))}
    res["label_ST"] = {"n_pos": int(Q.ST.sum()), "AUROC": float(roc_auc_score(Q.ST.astype(int), hr))}
    pool = pool.merge(M[["exam_id", "PR_cwt"]], on="exam_id", how="left")
    rows = []
    for s in SEEDS:
        m, _ = match_once(pool, s)
        rows.append(dict(Final3_dwt=cv_auc(m, FINAL3), Final3_cwt=cv_auc(m, ["RMSSD", "PR_cwt", "Pamp_V1"]),
                         Atrial_dwt=cv_auc(m, ATR), Atrial_cwt=cv_auc(m, ["PR_cwt", "Pamp_V1"])))
    S = pd.DataFrame(rows)
    res["cwt_sensitivity"] = {c: [float(S[c].mean()), float(S[c].std())] for c in S.columns}
    res["cwt_sensitivity"]["cwt_higher_seeds"] = int((S.Final3_cwt > S.Final3_dwt).sum())
    save_json(res, "14_pr_verification.json"); print(res)


if __name__ == "__main__":
    main()
