"""
Stage 7 - Age/sex matching averaged over 11 realisations and the primary results
(Methods II-B, II-J, II-K; Results III-D, III-I, III-L).

For each seed in config/seeds.txt: 1:1 matching (common.match_once), then out-of-fold
AUROC for age+sex, the variability block, the atrial block, the final three-feature model
and the five-feature model, and the atrial increment (final - variability). Also reports
the operating points, Brier score and calibration of the final model, the atrial increment
by recorded interval (Table V) and the seed-42 cohort used by single-realisation displays.

Input : outputs/cohort_clean.csv, rr_features.csv, pwave_features.csv, ptfv1_true.csv
Output: outputs/matched_seed42.csv, outputs/07_seed_results.csv, outputs/07_primary.json
"""
import sys
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score, brier_score_loss
from common import OUT, SEEDS, FINAL3, RRB, ATR, FIVE, load_pool, match_once, oof_pred, save_json


def main():
    pool = load_pool()
    pool["dAge"] = pool.age_at_first_af - pool.age_at_index
    print("pool:", pool.split.value_counts().to_dict())
    rows, ops = [], []
    incr = {"all": [], ">=2y": [], ">=3y": []}
    for s in SEEDS:
        m, pairs = match_once(pool, s)
        if s == 42:
            m.to_csv(OUT / "matched_seed42.csv", index=False)
        r = {"seed": s, "pairs": len(pairs)}
        for name, f in [("AgeSex", ["age_at_index", "is_male"]), ("RR", RRB), ("Atrial", ATR),
                        ("Final3", FINAL3), ("Five", FIVE)]:
            y, p = oof_pred(m, f)
            r[name] = roc_auc_score(y, p)
            if name == "Final3":
                ops.append(dict(sens80=(p[y == 1] > np.quantile(p[y == 0], .8)).mean(),
                                sens90=(p[y == 1] > np.quantile(p[y == 0], .9)).mean(),
                                brier=brier_score_loss(y, p)))
                if s == 42:
                    lp = np.log(p / (1 - p)).reshape(-1, 1)
                    cal = sm.Logit(y, sm.add_constant(lp)).fit(disp=0)
                    seed42 = {"Final3": r[name], "brier": brier_score_loss(y, p),
                              "cal_intercept": float(cal.params[0]), "cal_slope": float(cal.params[1])}
        r["H2"] = r["Final3"] - r["RR"]
        rows.append(r)
        dd = dict(zip(m[m.split == "vaka"].exam_id, m[m.split == "vaka"].dAge))
        for name, thr in [("all", 1), (">=2y", 2), (">=3y", 3)]:
            keep = [(c, k) for c, k in pairs if dd[c] >= thr]
            sub = m[m.exam_id.isin([x for pr in keep for x in pr])]
            y, p1 = oof_pred(sub, FINAL3); _, p0 = oof_pred(sub, RRB)
            incr[name].append((len(keep), roc_auc_score(y, p1) - roc_auc_score(y, p0)))
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    R = pd.DataFrame(rows); R.to_csv(OUT / "07_seed_results.csv", index=False)
    O = pd.DataFrame(ops)
    res = {k: [float(R[k].mean()), float(R[k].std())] for k in ["AgeSex", "RR", "Atrial", "Final3", "Five", "H2"]}
    res["Final3_range"] = [float(R.Final3.min()), float(R.Final3.max())]
    res["H2_positive_seeds"] = int((R.H2 > 0).sum())
    res["sens80"] = [float(O.sens80.mean()), float(O.sens80.std())]
    res["sens90"] = [float(O.sens90.mean()), float(O.sens90.std())]
    res["brier"] = [float(O.brier.mean()), float(O.brier.std())]
    res["seed42"] = seed42
    res["table5_increment"] = {k: {"cases": v[0][0], "mean": float(np.mean([x[1] for x in v])),
                                   "sd": float(np.std([x[1] for x in v])),
                                   "positive": int(sum(x[1] > 0 for x in v))} for k, v in incr.items()}
    save_json(res, "07_primary.json")
    print(R[["AgeSex", "RR", "Atrial", "Final3", "Five", "H2"]].agg(["mean", "std"]).round(4))


if __name__ == "__main__":
    main()
