"""Shared paths, loaders, matching and cross-validation helpers.

Run every script from the repository root, e.g. ``python src/data/01_build_cohort.py``.
Raw data live in ``data/`` (see README); all intermediate and final outputs are written
to ``outputs/``.
"""
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DATA = Path("data")
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
EXAMS = DATA / "exams.csv"

FS = 400                      # CODE-15% sampling rate
LEAD_II, V1 = 1, 6            # CODE-15% lead order: I, II, III, aVR, aVL, aVF, V1..V6
SEEDS = [42, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]   # config/seeds.txt
FINAL3 = ["RMSSD", "PR_med", "Pamp_V1"]
RRB = ["meanRR", "RMSSD"]
ATR = ["PR_med", "Pamp_V1"]
FIVE = ["meanRR", "RMSSD", "PR_med", "PR_max", "PTFV1_proxy"]


def save_json(obj, name):
    with open(OUT / name, "w") as f:
        json.dump(obj, f, indent=2, default=float)


def load_pool(include_ambiguous=False):
    """Case/control analysis pool: records passing signal QC with >=6 detected R peaks,
    merged with all features and the exam metadata needed downstream."""
    pool = pd.read_csv(OUT / "cohort_clean.csv")
    splits = ["vaka", "kontrol"] + (["vaka_belirsiz"] if include_ambiguous else [])
    pool = pool[pool.split.isin(splits)][["exam_id", "patient_id", "split", "age_at_index", "age_at_first_af"]]
    rr = pd.read_csv(OUT / "rr_features.csv")
    pw = pd.read_csv(OUT / "pwave_features.csv")
    pt = pd.read_csv(OUT / "ptfv1_true.csv")
    ex = pd.read_csv(EXAMS, usecols=["exam_id", "is_male", "timey", "death"])
    d = (pool.merge(rr, on="exam_id", how="inner")
             .merge(pw, on="exam_id", how="left")
             .merge(pt, on="exam_id", how="left")
             .merge(ex, on="exam_id", how="left"))
    return d[d.n_R >= 6].reset_index(drop=True)


def match_once(d, seed):
    """1:1 matching on sex and recorded age within 2 years (3 years if no candidate),
    without replacement; cases processed from oldest to youngest."""
    cases = d[d.split == "vaka"]
    controls = d[d.split == "kontrol"]
    rng = np.random.default_rng(seed)
    used, pairs = set(), []
    for _, case in cases.sort_values("age_at_index", ascending=False).iterrows():
        pool = controls[(~controls.exam_id.isin(used))
                        & (controls.is_male == case.is_male)
                        & (abs(controls.age_at_index - case.age_at_index) <= 2)]
        if pool.empty:
            pool = controls[(~controls.exam_id.isin(used))
                            & (abs(controls.age_at_index - case.age_at_index) <= 3)]
        if pool.empty:
            continue
        chosen = pool.iloc[int(rng.integers(len(pool)))]
        used.add(chosen.exam_id)
        pairs.append((case.exam_id, chosen.exam_id))
    ids = [a for a, _ in pairs] + [b for _, b in pairs]
    return d[d.exam_id.isin(ids)].reset_index(drop=True), pairs


def oof_pred(m, features, n_repeats=10, model=None, label="vaka", scale=True):
    """Out-of-fold predictions: StratifiedGroupKFold (5 folds) grouped on patient,
    repeated n_repeats times and averaged. Imputation and scaling are fitted in the folds."""
    y = (m.split == label).astype(int).values
    X = m[features].values
    P = np.zeros((n_repeats, len(y)))
    for t in range(n_repeats):
        skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=t)
        for tr, te in skf.split(X, y, groups=m.patient_id.values):
            clf = model() if model else LogisticRegression(max_iter=2000)
            mdl = make_pipeline(SimpleImputer(), StandardScaler(), clf) if scale else make_pipeline(SimpleImputer(), clf)
            mdl.fit(X[tr], y[tr])
            P[t, te] = mdl.predict_proba(X[te])[:, 1]
    return y, P.mean(0)


def cv_auc(m, features, n_repeats=10):
    y, p = oof_pred(m, features, n_repeats)
    return roc_auc_score(y, p)


def boot_auc_ci(y, p, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            v.append(roc_auc_score(y[i], p[i]))
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def icc21(X):
    """ICC(2,1), two-way random effects, absolute agreement, single measurement."""
    X = np.asarray(X, float); n, k = X.shape; gm = X.mean()
    msr = k * ((X.mean(1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((X.mean(0) - gm) ** 2).sum() / (k - 1)
    mse = ((X - X.mean(1, keepdims=True) - X.mean(0) + gm) ** 2).sum() / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)


def icc31(X):
    """ICC(3,1), two-way mixed effects, consistency, single measurement."""
    X = np.asarray(X, float); n, k = X.shape; gm = X.mean()
    msr = k * ((X.mean(1) - gm) ** 2).sum() / (n - 1)
    mse = ((X - X.mean(1, keepdims=True) - X.mean(0) + gm) ** 2).sum() / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse)
