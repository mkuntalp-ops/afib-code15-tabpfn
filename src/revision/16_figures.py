"""
Revision 16 - Manuscript figures 1-7 at printed size (3.4 or 6.8 in wide, fonts >= 7 pt).

Input : outputs/ (matched_seed42.csv, 09/14 outputs, cohort_signals.npz), data/ptbxl_cohort.csv
Output: outputs/figures/Fig1-Fig7 (PNG, 600 dpi)
"""
import sys, json
sys.path.insert(0, "src"); sys.path.insert(0, "src/features")
from common import *
import ecg_features as R
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_curve
import neurokit2 as nk
import json
from scipy.signal import filtfilt

FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "font.family": "DejaVu Sans"})
DPI = 600
m42 = pd.read_csv(OUT / "matched_seed42.csv")
ids_ = m42  # shorthand


# ---------------- Figure 1: cohort flow ----------------
def box(ax, x, y, w, h, text, fc="#ffffff", fs=6.2):
    p = FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
                       linewidth=0.6, edgecolor="#333333", facecolor=fc)
    ax.add_patch(p)
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, linespacing=1.3)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", lw=0.6, color="#333333", shrinkA=0, shrinkB=0))


_raw = pd.read_csv(OUT / "cohort_raw.csv"); _cl = pd.read_csv(OUT / "cohort_clean.csv")
_sig = set(pd.read_csv(OUT / "signal_bounds.csv").exam_id); _pool = load_pool()
_ex = pd.read_csv(EXAMS, usecols=["patient_id", "AF"])
N_AF = _ex.loc[_ex.AF == 1, "patient_id"].nunique()
N_CTRL_POOL = len(set(_ex.loc[_ex.AF == 0, "patient_id"]) - set(_ex.loc[_ex.AF == 1, "patient_id"]))
N_CLEAR = int((_raw.split == "vaka").sum()); N_SAME = int((_raw.split == "vaka_belirsiz").sum())
N_WAVE = int(((_raw.split == "vaka") & _raw.exam_id.isin(_sig)).sum()); N_QC = int((_cl.split == "vaka").sum())
N_CASE = int((_pool.split == "vaka").sum()); N_CS = int((_raw.split == "kontrol").sum()); N_CP = int((_pool.split == "kontrol").sum())
N_PAIRS = int((m42.split == "vaka").sum())
N_CW = int(((_raw.split == "kontrol") & _raw.exam_id.isin(_sig)).sum()); N_CQC = int((_cl.split == "kontrol").sum())
_um = _pool.dropna(subset=FINAL3)
N_UM_CASE, N_UM_CTRL = int((_um.split == "vaka").sum()), int((_um.split == "kontrol").sum())
_t = json.load(open(OUT / "12_temporal_followup.json"))
N_GE2Y = int((m42[m42.split == "vaka"].age_at_first_af - m42[m42.split == "vaka"].age_at_index >= 2).sum())
N_ECG2 = int(_t["controls_ge2_ecgs"]["pairs"])
fig, ax = plt.subplots(figsize=(7.16, 6.2))
ax.set_xlim(0, 1); ax.set_ylim(0.10, 1); ax.axis("off")
L, R_ = 0.25, 0.75
def seg(x0, y0, x1, y1):
    ax.plot([x0, x1], [y0, y1], color="#333333", lw=0.6)
box(ax, 0.50, 0.955, 0.52, 0.05, "CODE-15%: 233,770 patients, 345,779 ECGs")
arrow(ax, 0.40, 0.93, L, 0.895); arrow(ax, 0.60, 0.93, R_, 0.895)
box(ax, L, 0.86, 0.46, 0.07, f"Patients with ≥1 AF-labelled ECG: {N_AF:,}\n{N_AF - N_CLEAR - N_SAME:,} without a pre-AF sinus ECG excluded", fc="#f4f4f4")
box(ax, R_, 0.86, 0.46, 0.07, f"Controls: no AF label, ≥1 sinus ECG: {N_CTRL_POOL:,}")
arrow(ax, L, 0.825, L, 0.775); arrow(ax, R_, 0.825, R_, 0.775)
box(ax, L, 0.74, 0.46, 0.065, f"Sinus ECG before first AF ECG: {N_CLEAR + N_SAME:,}\n{N_CLEAR} with clear age ordering; {N_SAME} same recorded age")
box(ax, R_, 0.74, 0.46, 0.065, f"Random sample, 4 controls per case (seed 42): {N_CS:,}")
arrow(ax, R_, 0.707, 0.755, 0.645)
box(ax, 0.755, 0.605, 0.42, 0.085, f"Index = earliest sinus ECG\nwaveform available: {N_CW:,}; after QC: {N_CQC:,}\n≥6 R peaks: {N_CP:,}")
arrow(ax, 0.22, 0.707, 0.12, 0.645); arrow(ax, 0.28, 0.707, 0.39, 0.645)
box(ax, 0.12, 0.61, 0.20, 0.065, f"Same recorded age:\n{N_SAME} (excluded from\nprimary analysis)", fc="#f4f4f4")
box(ax, 0.39, 0.61, 0.27, 0.085, f"Clear ordering: {N_CLEAR}\nwaveform available: {N_WAVE}\nafter QC: {N_QC}")
arrow(ax, 0.39, 0.567, 0.39, 0.49)
box(ax, 0.39, 0.465, 0.40, 0.05, f"Cases with ≥6 R peaks: {N_CASE}\n(index = latest sinus ECG before AF)")
arrow(ax, 0.39, 0.44, 0.45, 0.385); arrow(ax, 0.755, 0.562, 0.68, 0.385)
box(ax, 0.55, 0.36, 0.46, 0.05, f"Analysis pool: {N_CASE} cases, {N_CP:,} controls", fc="#f4f4f4")
arrow(ax, 0.45, 0.335, 0.30, 0.285); arrow(ax, 0.65, 0.335, 0.76, 0.285)
box(ax, 0.28, 0.255, 0.46, 0.06, f"Matched cohort, 1:1 (sex, ±2 y; ±3 y fallback):\n{N_PAIRS} cases and {N_PAIRS} controls (seed 42; 11 seeds analysed)", fc="#e8f1fb")
box(ax, 0.77, 0.255, 0.40, 0.06, f"Unmatched analysis (complete feature data):\n{N_UM_CASE} cases, {N_UM_CTRL:,} controls (age, sex covariates)")
arrow(ax, 0.28, 0.225, 0.28, 0.185)
box(ax, 0.28, 0.155, 0.46, 0.06, f"Sensitivity analyses: interval ≥2 y ({N_GE2Y} cases);\ncontrols with ≥2 ECGs ({N_ECG2} pairs)")
fig.savefig(f"{FIG}/Fig1_cohort_flow.png", dpi=DPI, bbox_inches="tight"); plt.close(fig)

# ---------------- Figure 2: recorded interval, integer years ----------------
d = (m42[m42.split == "vaka"].age_at_first_af - m42[m42.split == "vaka"].age_at_index).astype(int)
cnt = d.value_counts().sort_index()
fig, ax = plt.subplots(figsize=(3.5, 2.3))
ax.bar(cnt.index, cnt.values, width=0.8, color="#4a7ab5", edgecolor="#2d4f7a", linewidth=0.4)
ax.set_xticks(range(1, int(cnt.index.max()) + 1))
ax.set_xlabel("Recorded age difference, index to first AF-labelled ECG (years)")
ax.set_ylabel("Cases (n)")
ax.text(0.97, 0.95, f"n = {int(cnt.sum())} cases\nmedian = {d.median():.0f} y\n≥2 y: {int((d>=2).sum())} ({(d>=2).mean()*100:.0f}%)",
        transform=ax.transAxes, ha="right", va="top", fontsize=6.5)
for s_ in ["top", "right"]:
    ax.spines[s_].set_visible(False)
fig.tight_layout(); fig.savefig(f"{FIG}/Fig2_interval.png", dpi=DPI); plt.close(fig)

# ---------------- seed-42 out-of-fold predictions ----------------
pred = {}
for name, feats in [("Age and sex", ["age_at_index", "is_male"]), ("Variability (RR block)", RRB),
                    ("Atrial block (PR, V1 amplitude)", ATR), ("Three-feature model", FINAL3)]:
    y, p = oof_pred(m42, feats)
    pred[name] = (y, p)

# ---------------- Figure 3: ROC ----------------
fig, ax = plt.subplots(figsize=(3.5, 3.1))
styles = ["-", "--", ":", "-"]
for (name, (y, p)), ls in zip(pred.items(), styles):
    fpr, tpr, _ = roc_curve(y, p)
    ax.plot(fpr, tpr, ls=ls, lw=1.0 if name != "Three-feature model" else 1.4, color="#1f3b63" if name == "Three-feature model" else None,
            label=f"{name} (AUROC {roc_auc_score(y, p):.3f})")
ax.plot([0, 1], [0, 1], color="#999999", lw=0.5, ls="-")
ax.set_xlabel("False positive rate (1 − specificity)"); ax.set_ylabel("True positive rate (sensitivity)")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1, frameon=False)
fig.set_size_inches(3.5, 3.9)
fig.tight_layout(); fig.savefig(f"{FIG}/Fig3_ROC.png", dpi=DPI); plt.close(fig)

# ---------------- Figure 4: calibration ----------------
y, p = pred["Three-feature model"]
qs = pd.qcut(p, 8, labels=False, duplicates="drop")
obs = pd.Series(y).groupby(qs).mean(); mp = pd.Series(p).groupby(qs).mean()
fig, ax = plt.subplots(figsize=(3.5, 3.0))
ax.plot([0, 1], [0, 1], color="#999999", lw=0.6)
ax.plot(mp.values, obs.values, "o-", ms=3.5, lw=1.0, color="#1f3b63")
ax.set_xlabel("Mean predicted probability of being a case (8 bins)")
ax.set_ylabel("Observed proportion of cases")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.text(0.03, 0.93, "Matched prevalence 50%:\nnot calibration to population incidence", fontsize=6.2, va="top")
fig.tight_layout(); fig.savefig(f"{FIG}/Fig4_calibration.png", dpi=DPI); plt.close(fig)

# ---------------- Figure 5: artefact adjudication by amplitude stratum ----------------
med_amp = m42.Pamp_V1.median()
strata = {"Low V1 P amplitude": m42[m42.Pamp_V1 <= med_amp], "High V1 P amplitude": m42[m42.Pamp_V1 > med_amp]}
feats5 = [("Pdur_med", "P duration (ms)"), ("Pdisp", "P dispersion (ms)"), ("PR_med", "PR interval (ms)"),
          ("PTFV1_proxy", "Pre-offset V1 integral (ms·mV)")]
fig, axes = plt.subplots(2, 4, figsize=(7.16, 3.8))
pvals = {}
for r, (sname, sdf) in enumerate(strata.items()):
    for c, (f, lab) in enumerate(feats5):
        ax = axes[r, c]
        a = sdf[sdf.split == "vaka"][f].dropna(); b = sdf[sdf.split == "kontrol"][f].dropna()
        pv = mannwhitneyu(a, b).pvalue if len(a) > 5 and len(b) > 5 else np.nan
        pvals[f"{sname}|{f}"] = float(pv)
        ax.boxplot([b, a], widths=0.5, showfliers=False, medianprops=dict(color="#1f3b63", lw=1))
        ax.set_xticks([1, 2]); ax.set_xticklabels(["Ctrl", "Case"])
        ax.set_title(f"{lab}" if r == 0 else "", fontsize=6.8)
        ax.text(0.5, 0.97, f"p={pv:.2g}" if pv >= 0.001 else f"p<0.001", transform=ax.transAxes, ha="center", va="top", fontsize=6)
        if c == 0:
            ax.set_ylabel(sname + "\n(n per group ≈ %d)" % (len(sdf) // 2))
fig.tight_layout(); fig.savefig(f"{FIG}/Fig5_artefact.png", dpi=DPI); plt.close(fig)

# ---------------- Figure 7: PR verification ----------------
b = pd.read_csv(OUT / "signal_bounds.csv").set_index("exam_id")
ex_id = int(m42[(m42.split == "vaka") & (m42.PR_med.notna()) & (m42.Pamp_V1 > 0.1)].exam_id.iloc[0])
_z = np.load(OUT / "cohort_signals.npz"); tr = _z["tracings"][list(_z["exam_id"]).index(ex_id)][None]
s0, e0 = int(b.loc[ex_id, "start"]), int(b.loc[ex_id, "end"])
xs = filtfilt(R.BB, R.BA, tr[0][s0:e0 + 1, R.LEAD_II])
_, info = nk.ecg_peaks(xs, sampling_rate=400); rp = info["ECG_R_Peaks"]
_, wd = nk.ecg_delineate(xs, rp, sampling_rate=400, method="dwt")
_, wc = nk.ecg_delineate(xs, rp, sampling_rate=400, method="cwt")
fig = plt.figure(figsize=(7.16, 3.3))
gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1])
ax = fig.add_subplot(gs[0, 0])
t = np.arange(len(xs)) / 400
t0_, t1_ = 0.5, 3.0
mask = (t >= t0_) & (t <= t1_)
ax.plot(t[mask], xs[mask], color="#222222", lw=0.7)
def mark(w, key, color, lab, marker):
    idx = np.array(w[key], float)
    idx = idx[~np.isnan(idx)].astype(int)
    idx = idx[(idx / 400 >= t0_) & (idx / 400 <= t1_)]
    ax.plot(idx / 400, xs[idx], marker, color=color, ms=3, label=lab)
on_ = np.array(wd["ECG_P_Onsets"], float); ro_ = np.array(wd["ECG_R_Onsets"], float)
for p_ in rp:
    co = on_[(~np.isnan(on_)) & (on_ < p_) & (on_ > p_ - 0.4 * 400)]
    cr = ro_[(~np.isnan(ro_)) & (ro_ < p_) & (ro_ > p_ - 0.15 * 400)]
    if len(co) and len(cr) and t0_ <= co[-1] / 400 <= t1_:
        ax.axvspan(co[-1] / 400, cr[-1] / 400, color="#f2c6cc", alpha=0.6, lw=0)
mark(wd, "ECG_P_Onsets", "#d1495b", "P onset", "^")
mark(wd, "ECG_P_Peaks", "#7a1f2b", "P peak", "v")
mark(wd, "ECG_P_Offsets", "#d1495b", "P offset", ">")
mark(wd, "ECG_R_Onsets", "#333333", "R onset", "s")
ax.plot([], [], color="#f2c6cc", lw=6, label="PR interval")
ax.legend(fontsize=5.8, ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2))
ax.set_xlabel("Time (s)"); ax.set_ylabel("Lead II (mV, band-passed)")
ax.legend(fontsize=5.6, ncol=2, frameon=False, loc="upper right")
ax.set_title("a) Example case record, lead II: DWT fiducial points used in the study", fontsize=7)
ax2 = fig.add_subplot(gs[0, 1])
_M = pd.read_csv(OUT / "14_pr_cwt.csv"); _rng = np.random.default_rng(7)   # same 400-record sample as stage 14
M = _M.loc[_rng.choice(len(_M), 400, replace=False)].dropna(subset=["PR_dwt", "PR_cwt"])
mean_ = (M.PR_dwt + M.PR_cwt) / 2; diff_ = M.PR_dwt - M.PR_cwt
ax2.scatter(mean_, diff_, s=5, color="#4a7ab5", alpha=0.6, lw=0)
md = diff_.mean(); sd = diff_.std(ddof=1)
for yv, ls in [(md, "-"), (md + 1.96 * sd, "--"), (md - 1.96 * sd, "--")]:
    ax2.axhline(yv, color="#333333", lw=0.6, ls=ls)
ax2.set_xlabel("Mean of DWT and CWT PR (ms)"); ax2.set_ylabel("DWT − CWT PR (ms)")
ax2.set_title(f"b) Agreement, lead II (n={len(M)})", fontsize=7)
ax2.text(0.02, 0.04, f"bias {md:.0f} ms; limits {md-1.96*sd:.0f} to {md+1.96*sd:.0f} ms", transform=ax2.transAxes, fontsize=6.2)
fig.tight_layout(); fig.savefig(f"{FIG}/Fig7_PR_verification.png", dpi=DPI); plt.close(fig)

save_json({"fig5_pvalues": pvals, "fig7_example_exam": ex_id}, "16_figure_stats.json")
print("figures written:", sorted(__import__("os").listdir(FIG)))

# ---------------- Figure 6: derivation vs PTB-XL replication (new features) ----------------
px = pd.read_csv(OUT / "ptbxl_features.csv").merge(
    pd.read_csv(DATA / "ptbxl_cohort.csv")[["ecg_id", "split"]], on="ecg_id")
px = px[px.n_R >= 6]
feats6 = [("RMSSD", "RMSSD (ms)"), ("PR_med", "PR interval (ms)"), ("Pamp_V1", "P amplitude, V1 (mV)")]
fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.5))
for ax, (f, lab) in zip(axes, feats6):
    data = [m42[m42.split == "kontrol"][f].dropna(), m42[m42.split == "vaka"][f].dropna(),
            px[px.split == "kontrol"][f].dropna(), px[px.split == "vaka"][f].dropna()]
    ax.boxplot(data, positions=[1, 2, 3.6, 4.6], widths=0.6, showfliers=False,
               medianprops=dict(color="#1f3b63", lw=1))
    ax.axvline(2.8, color="#999999", lw=0.6, ls="--")
    lo_, hi_ = ax.get_ylim(); ax.set_ylim(lo_, hi_ + 0.12 * (hi_ - lo_))
    ax.set_xticks([1, 2, 3.6, 4.6]); ax.set_xticklabels(["Ctrl", "Case", "Ctrl", "Case"])
    ax.set_title(lab, fontsize=7)
    for x0, (a, b) in [(1.5, (data[0], data[1])), (4.1, (data[2], data[3]))]:
        pv = mannwhitneyu(a, b).pvalue
        ax.text(x0, 0.98, "p<0.001" if pv < 0.001 else f"p={pv:.2g}", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=6)
    ax.text(1.5, -0.2, "CODE-15%", transform=ax.get_xaxis_transform(), ha="center", fontsize=6.5)
    ax.text(4.1, -0.2, "PTB-XL", transform=ax.get_xaxis_transform(), ha="center", fontsize=6.5)
fig.tight_layout(); fig.savefig(f"{FIG}/Fig6_replication.png", dpi=DPI); plt.close(fig)
print("Fig6 written")
