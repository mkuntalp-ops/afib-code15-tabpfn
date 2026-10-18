"""
Revision 15 - Table III (feature distributions), Table VI (coefficients), artefact
adjudication statistics, Stage-1 correlations, the effect of outcome-informed decisions
(nested Stage 2, model without V1 measure) and the unmatched-cohort analysis
(Results III-C, III-E, III-H, III-K, III-M; Tables I, III, VI).

Input : outputs/matched_seed42.csv and feature tables
Output: outputs/15_tables_reduction.json
"""
import sys
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr, pearsonr
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from common import OUT, SEEDS, FINAL3, FIVE, load_pool, match_once, oof_pred, cv_auc, boot_auc_ci, save_json


def hl(a, b):
    return float(np.median(np.subtract.outer(a, b)))


def main():
    res = {}
    rng = np.random.default_rng(42)
    pool = load_pool()
    m = pd.read_csv(OUT / "matched_seed42.csv")
    y = (m.split == "vaka").astype(int).values
    T3 = {}
    for f in ["RMSSD", "PR_med", "Pamp_V1", "PTFV1_true"]:
        a = m[m.split == "vaka"][f].dropna().values; b = m[m.split == "kontrol"][f].dropna().values
        bs = [hl(rng.choice(a, len(a)), rng.choice(b, len(b))) for _ in range(500)]
        T3[f] = {"case": float(np.median(a)), "ctrl": float(np.median(b)), "HL": hl(a, b),
                 "CI": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "p": float(mannwhitneyu(a, b).pvalue)}
    res["table3"] = T3
    res["PTFV1_p_by_seed"] = []
    for s in SEEDS:
        mm, _ = match_once(pool, s)
        res["PTFV1_p_by_seed"].append(float(mannwhitneyu(mm[mm.split == "vaka"].PTFV1_true.dropna(),
                                                         mm[mm.split == "kontrol"].PTFV1_true.dropna()).pvalue))
    X = m[FINAL3].values; C = []
    for t in range(10):
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=t).split(X, y, m.patient_id.values):
            C.append(make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(max_iter=2000)).fit(X[tr], y[tr])[-1].coef_[0])
    C = np.array(C)
    res["table6"] = {f: {"mean": float(C[:, i].mean()), "positive_folds": int((C[:, i] > 0).sum()), "n_folds": len(C)} for i, f in enumerate(FINAL3)}
    art = {}
    for f in ["Pdur_med", "Pdisp", "PR_med", "PTFV1_proxy"]:
        a = m[m.split == "vaka"][f].dropna(); b = m[m.split == "kontrol"][f].dropna()
        art[f"unstratified_{f}"] = {"case": float(a.median()), "ctrl": float(b.median()), "p": float(mannwhitneyu(a, b).pvalue)}
    med = m.Pamp_V1.median()
    for sname, sdf in [("low", m[m.Pamp_V1 <= med]), ("high", m[m.Pamp_V1 > med])]:
        for f in ["Pdur_med", "Pdisp", "PR_med", "PTFV1_proxy"]:
            a = sdf[sdf.split == "vaka"][f].dropna(); b = sdf[sdf.split == "kontrol"][f].dropna()
            art[f"{sname}_{f}_p"] = float(mannwhitneyu(a, b).pvalue)
    for sname, sdf in [("low", m[m.Pamp_V1 <= med]), ("high", m[m.Pamp_V1 > med])]:
        for f in ["Pdisp", "Pdur_med"]:
            a = sdf[sdf.split == "vaka"][f].dropna().values; b = sdf[sdf.split == "kontrol"][f].dropna().values
            bs = [hl(rng.choice(a, len(a)), rng.choice(b, len(b))) for _ in range(500)]
            art[f"{sname}_{f}_HL"] = {"HL": hl(a, b), "CI": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    res["artefact"] = art
    q = pool.dropna(subset=["PTFV1_proxy", "Pamp_V1"])
    res["spearman_proxy_amplitude"] = float(spearmanr(q.PTFV1_proxy, q.Pamp_V1)[0])
    res["pearson_nRR_meanRR"] = float(pearsonr(pool.n_RR, pool.meanRR)[0])
    qq = pool.dropna(subset=["SDNN", "RMSSD"]); res["pearson_SDNN_RMSSD"] = float(pearsonr(qq.SDNN, qq.RMSSD)[0])
    res["missing_final_features"] = {f: float(pool[f].isna().mean()) for f in FINAL3}

    nested, noV1, sel = [], [], []
    for s in SEEDS:
        mm, _ = match_once(pool, s)
        yy = (mm.split == "vaka").astype(int).values; XX = mm[FIVE].values; P = np.zeros((3, len(yy)))
        for t in range(3):
            for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=t).split(XX, yy, mm.patient_id.values):
                base = make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(max_iter=2000)).fit(XX[tr], yy[tr])
                pi = permutation_importance(base, XX[tr], yy[tr], scoring="roc_auc", n_repeats=10, random_state=0).importances_mean
                keep = [i for i in range(len(FIVE)) if pi[i] > 0.005] or list(range(len(FIVE)))
                sel.append([FIVE[i] for i in keep])
                mdl = make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(max_iter=2000)).fit(XX[tr][:, keep], yy[tr])
                P[t, te] = mdl.predict_proba(XX[te][:, keep])[:, 1]
        nested.append(roc_auc_score(yy, P.mean(0)))
        noV1.append(cv_auc(mm, ["meanRR", "RMSSD", "PR_med", "PR_max"]))
    res["nested_stage2"] = [float(np.mean(nested)), float(np.std(nested))]
    res["nested_selection_freq"] = {f: float(np.mean([f in s_ for s_ in sel])) for f in FIVE}
    res["noV1_four_feature"] = [float(np.mean(noV1)), float(np.std(noV1))]

    um = pool.dropna(subset=FINAL3).copy(); um["male"] = um.is_male.astype(int)
    U = {"n_cases": int((um.split == "vaka").sum()), "n_controls": int((um.split == "kontrol").sum()),
         "median_age": {g: float(s.age_at_index.median()) for g, s in um.groupby("split")}}
    preds = {}
    for name, feats in [("age_sex", ["age_at_index", "male"]), ("ECG3", FINAL3), ("combined", ["age_at_index", "male"] + FINAL3)]:
        yy, pp = oof_pred(um, feats, n_repeats=5); preds[name] = pp
        U[name] = {"AUROC": float(roc_auc_score(yy, pp)), "CI": boot_auc_ci(yy, pp, n=1000), "brier": float(brier_score_loss(yy, pp))}
    U["combined"]["sens_at_spec90"] = float((preds["combined"][yy == 1] > np.quantile(preds["combined"][yy == 0], .9)).mean())
    d = []
    for _ in range(2000):
        i = rng.integers(0, len(yy), len(yy))
        d.append(roc_auc_score(yy[i], preds["combined"][i]) - roc_auc_score(yy[i], preds["age_sex"][i]))
    U["increment"] = {"value": U["combined"]["AUROC"] - U["age_sex"]["AUROC"], "CI": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
    res["unmatched"] = U
    save_json(res, "15_tables_reduction.json"); print(res)


if __name__ == "__main__":
    main()
