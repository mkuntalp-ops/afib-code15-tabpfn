"""
Stage 9 - PTB-XL cross-sectional replication (Methods II-M, Results III-R, Fig. 6).

Cohort file data/ptbxl_cohort.csv: ecg_id, patient_id, age, sex, file, af_oncesi, gun, split
(135 AF/atrial-flutter cases with an available sinus record, 540 age/sex-matched controls).
The join key is ecg_id (PTB-XL), not exam_id.

PTB-XL cross-sectional replication with features re-extracted by the same per-record
logic as the CODE-15% pipeline (r01_extract_features.py), rescaled to 500 Hz:
band-pass 0.5-40 Hz (3rd-order Butterworth, zero-phase), NeuroKit2 R peaks on lead II,
DWT delineation on all 12 leads, PR = median over beats (R onset - P onset) per lead then
median across leads, P amplitude in V1 = median filtered value at P peak, 40 ms pre-offset
V1 integral (20 samples at 500 Hz), classic Morris PTFV1.

Cohort: ptbxl_cohort.csv (135 AF/AFL cases, 540 age/sex-matched controls; built in the
original analysis). Inclusion rule as CODE-15%: >=6 detected R peaks on lead II.
Models: (1) direct transfer of the three-feature logistic model fitted on the CODE-15%
seed-42 matched cohort; (2) re-fitting inside PTB-XL with patient-grouped 5-fold CV, 10 repeats.
"""
import sys, json
sys.path.insert(0, "src")
import numpy as np, pandas as pd, wfdb, neurokit2 as nk, warnings
from multiprocessing import Pool
from scipy.signal import butter, filtfilt
from scipy.stats import mannwhitneyu
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

from common import DATA, OUT
FS = 500
V1, LEAD_II = 6, 1
W40 = int(round(0.040 * FS))      # 40 ms pre-offset window = 20 samples
BB, BA = butter(3, [0.5 / (FS / 2), 40 / (FS / 2)], btype="band")
FINAL3 = ["RMSSD", "PR_med", "Pamp_V1"]


def terminal_negative_area(seg):
    if len(seg) == 0 or not (seg < 0).any():
        return 0.0
    k = len(seg) - 1
    while k >= 0 and seg[k] >= 0:
        k -= 1
    j = k
    while j >= 0 and seg[j] < 0:
        j -= 1
    term = seg[j + 1: k + 1]
    return float(term.sum() / FS * 1000) if len(term) else 0.0


def extract(args):
    ecg_id, path = args
    sig, _ = wfdb.rdsamp(str(DATA / "ptbxl" / path))
    rec = {"ecg_id": int(ecg_id)}
    # RR features, lead II
    x2 = filtfilt(BB, BA, sig[:, LEAD_II])
    try:
        rp = nk.ecg_peaks(x2, sampling_rate=FS)[1]["ECG_R_Peaks"]
    except Exception:
        rp = np.array([])
    rr = np.diff(rp) / FS * 1000
    rr = rr[(rr > 300) & (rr < 2000)]
    rec["n_R"] = int(len(rp))
    if len(rr) >= 3:
        rec["meanRR"] = float(rr.mean()); rec["RMSSD"] = float(np.sqrt(np.mean(np.diff(rr) ** 2)))
    prs = []
    for lead in range(12):
        try:
            x = filtfilt(BB, BA, sig[:, lead])
            rpl = nk.ecg_peaks(x, sampling_rate=FS)[1]["ECG_R_Peaks"]
            if len(rpl) < 4:
                continue
            w = nk.ecg_delineate(x, rpl, sampling_rate=FS, method="dwt")[1]
            on = np.array(w["ECG_P_Onsets"], float); off = np.array(w["ECG_P_Offsets"], float)
            ron = np.array(w["ECG_R_Onsets"], float); pk = np.array(w["ECG_P_Peaks"], float)
            v = ~(np.isnan(on) | np.isnan(ron))
            if v.sum() >= 2:
                pr = float(np.median(ron[v] - on[v])) / FS * 1000
                if 60 < pr < 400:
                    prs.append(pr)
            if lead == V1:
                a = ~np.isnan(pk)
                if a.sum() >= 2:
                    rec["Pamp_V1"] = float(np.median(x[pk[a].astype(int)]))
                vv = ~(np.isnan(on) | np.isnan(off))
                if vv.sum() >= 2:
                    seg = [x[max(0, int(o) - W40): int(o)] for o in off[vv] if int(o) > W40]
                    if seg:
                        rec["PTFV1_proxy"] = float(np.median([t.sum() / FS * 1000 for t in seg]))
                    vals = [terminal_negative_area(x[int(o1):int(o2)])
                            for o1, o2 in zip(on[vv], off[vv]) if int(o2) > int(o1)]
                    if vals:
                        rec["PTFV1_true"] = float(np.median(vals))
        except Exception:
            continue
    if prs:
        rec["PR_med"] = float(np.median(prs))
    return rec


def boot_ci(y, p, n=2000, seed=42):
    rng = np.random.default_rng(seed); v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            v.append(roc_auc_score(y[i], p[i]))
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


if __name__ == "__main__":
    coh = pd.read_csv(DATA / "ptbxl_cohort.csv")
    with Pool(8) as p:
        rows = p.map(extract, list(zip(coh.ecg_id, coh.file)), chunksize=4)
    F = pd.DataFrame(rows)
    F.to_csv(OUT / "ptbxl_features.csv", index=False)
    d = coh.merge(F, on="ecg_id")
    res = {"n_cohort": int(len(coh)), "n_cases": int((coh.split == "vaka").sum()),
           "n_controls": int((coh.split == "kontrol").sum())}
    d = d[d.n_R >= 6].reset_index(drop=True)
    res["after_nR6"] = {"cases": int((d.split == "vaka").sum()), "controls": int((d.split == "kontrol").sum())}
    res["coverage"] = {f: float(d[f].notna().mean()) for f in ["RMSSD", "PR_med", "Pamp_V1", "PTFV1_proxy", "PTFV1_true"]}
    feats = {}
    for f in ["RMSSD", "PR_med", "Pamp_V1", "PTFV1_proxy", "PTFV1_true", "meanRR"]:
        a = d[d.split == "vaka"][f].dropna(); b = d[d.split == "kontrol"][f].dropna()
        feats[f] = {"case_median": float(a.median()), "ctrl_median": float(b.median()),
                    "MWU_p": float(mannwhitneyu(a, b).pvalue), "higher_in_cases": bool(a.median() > b.median())}
    res["features"] = feats
    y = (d.split == "vaka").astype(int).values
    # (1) direct transfer from CODE-15% seed-42 matched cohort
    m42 = pd.read_csv(OUT / "matched_seed42.csv")
    dev = make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(max_iter=2000))
    dev.fit(m42[FINAL3].values, (m42.split == "vaka").astype(int).values)
    p_t = dev.predict_proba(d[FINAL3].values)[:, 1]
    res["direct_transfer"] = {"AUROC": float(roc_auc_score(y, p_t)), "CI": boot_ci(y, p_t)}
    # (2) re-fit inside PTB-XL
    def refit(feats_, reps=10):
        X = d[feats_].values; P = np.zeros((reps, len(y)))
        for t in range(reps):
            for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=t).split(X, y, d.patient_id.values):
                mdl = make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(max_iter=2000)).fit(X[tr], y[tr])
                P[t, te] = mdl.predict_proba(X[te])[:, 1]
        return P.mean(0)
    p_r = refit(FINAL3)
    res["refit_PTBXL"] = {"AUROC": float(roc_auc_score(y, p_r)), "CI": boot_ci(y, p_r)}
    p_rr = refit(["meanRR", "RMSSD"])
    res["refit_RR_block"] = float(roc_auc_score(y, p_rr))
    res["refit_atrial_block"] = float(roc_auc_score(y, refit(["PR_med", "Pamp_V1"])))
    # restrict to the 57 cases with a sinus record before the first AF record (and their controls are not linked; report cases only)
    res["cases_with_prior_sinus"] = int(coh[(coh.split == "vaka") & (coh.af_oncesi)].shape[0])
    json.dump(res, open(OUT / "09_ptbxl.json", "w"), indent=2)
    print(json.dumps(res, indent=2))
