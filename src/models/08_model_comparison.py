"""
Stage 8 - Seven model classes on the five-feature set, seed-42 matched cohort (Methods II-J,
Table IV, Results III-D, III-J).

Logistic regression, SVM, kNN and random forest at library defaults; gradient boosting and
the multilayer perceptron tuned by inner 3-fold grid search. TabPFN (optional) runs without
hyperparameter search with the checkpoint given in TABPFN_CHECKPOINT (environment variable;
default 'tabpfn-v3-classifier-v3_default.ckpt'); it is skipped if tabpfn is not installed.
When TabPFN runs, paired bootstrap comparisons against gradient boosting and logistic
regression (non-inferiority margin -0.02, Benjamini-Hochberg) and CPU costs are reported.

Input : outputs/matched_seed42.csv
Output: outputs/08_models.json
"""
import os, sys, time
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from common import OUT, FIVE, FINAL3, oof_pred, boot_auc_ci, save_json


def gbdt():
    return GridSearchCV(HistGradientBoostingClassifier(random_state=42),
                        {"max_leaf_nodes": [4, 8, 15], "learning_rate": [0.03, 0.1], "min_samples_leaf": [20, 50]},
                        scoring="roc_auc", cv=3, n_jobs=-1)


def mlp():
    return GridSearchCV(MLPClassifier(max_iter=2000, random_state=42),
                        {"hidden_layer_sizes": [(16,), (32,), (64, 32)], "alpha": [0.1, 1.0, 10.0]},
                        scoring="roc_auc", cv=3, n_jobs=-1)


MODELS = {
    "Logistic regression": (lambda: LogisticRegression(max_iter=2000), 10),
    "SVM (RBF)": (lambda: SVC(probability=True, random_state=42), 10),
    "k-NN (k=50)": (lambda: KNeighborsClassifier(50), 10),
    "Random forest": (lambda: RandomForestClassifier(400, random_state=42), 10),
    "Gradient boosting (tuned)": (gbdt, 3),
    "MLP (tuned)": (mlp, 3),
}


def main():
    m = pd.read_csv(OUT / "matched_seed42.csv")
    res, preds = {}, {}
    for name, (fn, reps) in MODELS.items():
        y, p = oof_pred(m, FIVE, reps, model=fn)
        preds[name] = p
        res[name] = {"AUROC": float(roc_auc_score(y, p)), "CI": boot_auc_ci(y, p)}
        print(name, round(res[name]["AUROC"], 3), flush=True)
    for name, fn in [("LR_3feat", lambda: LogisticRegression(max_iter=2000)),
                     ("RF_3feat", lambda: RandomForestClassifier(400, random_state=42))]:
        y, p = oof_pred(m, FINAL3, 10, model=fn); res[name] = float(roc_auc_score(y, p))
    try:
        from tabpfn import TabPFNClassifier
        import tabpfn
        ckpt = os.environ.get("TABPFN_CHECKPOINT", "tabpfn-v3-classifier-v3_default.ckpt")
        tab = lambda: TabPFNClassifier(model_path=ckpt, device="cpu", random_state=0)
        t0 = time.time(); y, p = oof_pred(m, FIVE, 5, model=tab, scale=False)
        res["TabPFN"] = {"AUROC": float(roc_auc_score(y, p)), "CI": boot_auc_ci(y, p),
                         "version": tabpfn.__version__, "checkpoint": ckpt, "cv_seconds": time.time() - t0}
        y3, p3 = oof_pred(m, FINAL3, 5, model=tab, scale=False); res["TabPFN_3feat"] = float(roc_auc_score(y3, p3))
        rng = np.random.default_rng(42); diffs = {"tab_gb": [], "tab_lr": [], "gb_lr": []}
        pg, pl = preds["Gradient boosting (tuned)"], preds["Logistic regression"]
        for _ in range(2000):
            i = rng.integers(0, len(y), len(y))
            a, g, l = (roc_auc_score(y[i], q[i]) for q in (p, pg, pl))
            diffs["tab_gb"].append(a - g); diffs["tab_lr"].append(a - l); diffs["gb_lr"].append(g - l)
        pv = {}
        for k, v in diffs.items():
            v = np.array(v); pv[k] = float(min(1, 2 * min((v <= 0).mean(), (v >= 0).mean())))
            res[k] = {"diff_mean": float(v.mean()), "CI": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
                      "p_two_sided": pv[k], "non_inferior_-0.02": bool(np.percentile(v, 2.5) > -0.02)}
        order = sorted(pv, key=pv.get); q, prev = {}, 1.0
        for rank, k in reversed(list(enumerate(order, 1))):
            prev = min(prev, pv[k] * len(order) / rank); q[k] = prev
        for k in q:
            res[k]["BH_q"] = float(q[k])
    except ImportError:
        print("tabpfn not installed: TabPFN rows skipped")
    X = SimpleImputer().fit_transform(m[FIVE].values); y = (m.split == "vaka").astype(int).values
    cost = {}
    cands = {"Logistic regression": lambda: LogisticRegression(max_iter=2000),
             "Multilayer perceptron": lambda: MLPClassifier((32,), alpha=1.0, max_iter=2000, random_state=42),
             "Gradient boosting": lambda: HistGradientBoostingClassifier(max_leaf_nodes=8, learning_rate=0.1, random_state=42),
             "Random forest": lambda: RandomForestClassifier(400, random_state=42)}
    if "TabPFN" in res:
        cands["TabPFN"] = tab
    for name, mk in cands.items():
        f_, p_ = [], []
        for r_ in range(5):
            Xtr, Xte, ytr, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=r_)
            mdl = make_pipeline(StandardScaler(), mk()) if name != "TabPFN" else mk()
            t0 = time.perf_counter(); mdl.fit(Xtr, ytr); f_.append((time.perf_counter() - t0) * 1000)
            t0 = time.perf_counter(); mdl.predict_proba(Xte); p_.append((time.perf_counter() - t0) * 1000 / len(Xte))
        cost[name] = {"fit_ms": float(np.mean(f_)), "predict_ms_per_record": float(np.mean(p_))}
    res["cost_cpu"] = cost
    save_json(res, "08_models.json")


if __name__ == "__main__":
    main()
