"""
Revision 18 - Manuscript Fig. 1 (Section II): pictorial diagram of how an AF-labelled ECG arises in
CODE-15% and which quantities are measured on the index sinus-rhythm ECG.

Schematic only: the waveforms are synthetic (sums of Gaussians), not patient data.
Facts in panel (a) follow Ribeiro et al., Nat. Commun. 2020 [15] and the CODE-15%
Zenodo record [16]; numbers in panel (b) follow the manuscript (551 matched pairs,
median recorded-age difference 2 years, matching within 2 years and on sex).

Input : none (schematic; no data are read)
Output: outputs/figures/Fig1_acquisition_labelling.png (600 dpi, 7.16 in wide)
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path("outputs") / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 7.5, "axes.linewidth": 0.6, "font.family": "DejaVu Sans"})
DPI = 600
INK, GREY, BLUE, RED = "#333333", "#f4f4f4", "#e8f1fb", "#C00000"


def box(ax, x, y, w, h, text, fc="#ffffff", fs=6.6):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.004,rounding_size=0.012",
                                linewidth=0.6, edgecolor=INK, facecolor=fc))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, linespacing=1.3)


def arrow(ax, x0, y0, x1, y1, style="-|>"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, lw=0.6, color=INK, shrinkA=0, shrinkB=0))


def g(t, mu, sd, a):
    return a * np.exp(-0.5 * ((t - mu) / sd) ** 2)


def beat_lead2(t):   # t in s relative to R peak
    return (g(t, -0.17, 0.022, 0.15) + g(t, -0.025, 0.008, -0.10) + g(t, 0, 0.010, 1.0)
            + g(t, 0.025, 0.009, -0.22) + g(t, 0.26, 0.045, 0.28))


def beat_v1(t):      # biphasic P (exaggerated for legibility): positive then terminal negative
    return (g(t, -0.185, 0.018, 0.22) + g(t, -0.125, 0.016, -0.16)
            + g(t, -0.02, 0.008, 0.22) + g(t, 0.01, 0.012, -0.80) + g(t, 0.25, 0.05, 0.12))


fig = plt.figure(figsize=(7.16, 5.0))

# ---------------- (a) recording and labelling ----------------
ax = fig.add_axes([0.0, 0.70, 1.0, 0.30]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.text(0.0, 0.98, "(a) Recording and labelling (CODE study [15]; CODE-15% sample [16])",
        fontsize=7.8, fontweight="bold", va="top")
ys, H = 0.50, 0.60
box(ax, 0.082, ys, 0.145, H, "Primary care\n12-lead ECG,\n7–10 s,\ntele-ECG device", fs=6.3)
box(ax, 0.255, ys, 0.165, H, "Electronic upload\nto the Telehealth\nNetwork of Minas\nGerais (TNMG)", fs=6.3)
box(ax, 0.46, ys + 0.165, 0.19, 0.27, "Cardiologist\ntext report", fc=GREY, fs=6.3)
box(ax, 0.46, ys - 0.165, 0.19, 0.27, "Automatic analysis\n(Uni-G statements,\nMinnesota codes)", fc=GREY, fs=6.1)
box(ax, 0.68, ys, 0.19, H, "Combined rules:\nreport and automatic\nanalysis agree;\nAF also requires a\ncardiologist AF report\nand high RR variability", fs=6.0)
box(ax, 0.90, ys, 0.19, H, "Exam-level label\nAF = 1 or AF = 0\n\n12 leads, 400 Hz,\n4,096 samples", fc=BLUE, fs=6.3)
arrow(ax, 0.155, ys, 0.172, ys)
arrow(ax, 0.338, ys + 0.08, 0.365, ys + 0.165); arrow(ax, 0.338, ys - 0.08, 0.365, ys - 0.165)
arrow(ax, 0.555, ys + 0.165, 0.585, ys + 0.08); arrow(ax, 0.555, ys - 0.165, 0.585, ys - 0.08)
arrow(ax, 0.775, ys, 0.805, ys)
ax.text(0.5, 0.04, "Labels were used as released, without re-adjudication. "
        "An AF label marks an exam whose tracing showed AF; it does not date AF onset.",
        ha="center", fontsize=6.3, style="italic")

# ---------------- (b) case and control definition ----------------
ax = fig.add_axes([0.0, 0.40, 1.0, 0.28]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.text(0.0, 0.99, "(b) How an AF-labelled ECG defines a case", fontsize=7.8, fontweight="bold", va="top")
YC, YK = 0.62, 0.14
for y, lab in [(YC, "Case"), (YK, "Matched\ncontrol")]:
    ax.text(0.0, y, lab, va="center", fontsize=6.8, fontweight="bold", linespacing=1.1)
    arrow(ax, 0.11, y, 0.99, y)
    ax.text(0.99, y - 0.06, "recorded age", ha="right", va="top", fontsize=6.0, color="#666666")
for x, fc, lab in [(0.22, "white", "earlier\nAF = 0 ECG"), (0.42, BLUE, "index ECG, AF = 0\n(measured)"),
                   (0.72, "#f8d7d7", "first AF-labelled ECG\n(reference age)")]:
    ax.plot(x, YC, "o", ms=6.5, mfc=fc, mec=INK, mew=0.6, zorder=3)
    ax.text(x, YC + 0.07, lab, ha="center", va="bottom", fontsize=6.2, linespacing=1.15)
arrow(ax, 0.425, YC - 0.12, 0.715, YC - 0.12, style="<|-|>")
ax.text(0.57, YC - 0.16, r"$d = a_{\mathrm{AF}} - a_i > 0$ years (median 2)", ha="center", va="top", fontsize=6.2)
ax.plot(0.42, YK, "o", ms=6.5, mfc=BLUE, mec=INK, mew=0.6, zorder=3)
ax.text(0.42, YK + 0.07, "index ECG: earliest AF = 0 ECG (measured)", ha="center", va="bottom", fontsize=6.2)
ax.text(0.78, YK + 0.07, "no AF-labelled ECG at any age;\nsame sex, recorded age within 2 y",
        ha="center", va="bottom", fontsize=6.2, linespacing=1.15)

# ---------------- (c) measurements ----------------
fs = 1000
ax = fig.add_axes([0.0, 0.02, 0.56, 0.34])
ax.text(0.0, 1.06, "(c) Measured on the index ECG: rhythm (lead II)", transform=ax.transAxes,
        fontsize=7.8, fontweight="bold", va="top")
rpk = np.array([0.45, 1.27, 2.02, 2.88, 3.66])
t = np.arange(0, 4.1, 1 / fs)
sig = sum(beat_lead2(t - r) for r in rpk)
ax.plot(t, sig, color="black", lw=0.7)
ax.plot(rpk, np.ones_like(rpk) * 1.06, "v", color=RED, ms=3.5)
for i in range(len(rpk) - 1):
    a, b = rpk[i], rpk[i + 1]
    arrow(ax, a, 1.22, b, 1.22, style="<|-|>")
    ax.text((a + b) / 2, 1.28, f"RR$_{i + 1}$", ha="center", fontsize=6.3)
ax.text(0.02, -0.40, "R peaks (NeuroKit2) → RR intervals (300–2,000 ms; ≥5 intervals)\n"
        r"mean RR $=\overline{RR_k}$;   RMSSD $=\sqrt{\overline{(RR_{k+1}-RR_k)^2}}$",
        fontsize=6.4, va="top", linespacing=1.5)
ax.set_xlim(0, 4.1); ax.set_ylim(-1.05, 1.55); ax.axis("off")

ax = fig.add_axes([0.60, 0.02, 0.40, 0.34])
ax.text(0.0, 1.06, "Atrial measures (one beat, lead V1)", transform=ax.transAxes,
        fontsize=7.8, fontweight="bold", va="top")
tb = np.arange(-0.30, 0.12, 1 / fs)
v = beat_v1(tb)
ax.axhline(0, color="#999999", lw=0.5, ls=":")
ax.plot(tb, v, color="black", lw=0.8)
p_on, p_pk, p_off, q_on = -0.225, -0.185, -0.095, -0.045
term = (tb > -0.160) & (tb < p_off) & (v < 0)
ax.fill_between(tb[term], v[term], 0, color=RED, alpha=0.35, lw=0)
ax.vlines([p_on, p_off, q_on], -0.50, 0.40, colors="#666666", lw=0.5, linestyles="--")
for x, lab in [(p_on, "P on"), (p_off, "P off"), (q_on, "QRS on")]:
    ax.text(x, 0.43, lab, ha="center", va="bottom", fontsize=6.0, color="#444444")
pa = beat_v1(np.array([p_pk]))[0]
ax.annotate("", xy=(p_pk, pa), xytext=(p_pk, 0), arrowprops=dict(arrowstyle="-|>", lw=0.6, color=INK))
ax.annotate("V1 P amplitude", xy=(p_pk, pa * 0.6), xytext=(-0.305, 0.30), fontsize=6.0,
            arrowprops=dict(arrowstyle="-", lw=0.4, color="#666666"))
ax.annotate("terminal negative\narea (mV·ms)", xy=(-0.125, -0.06), xytext=(-0.305, -0.33), fontsize=6.0,
            color=RED, linespacing=1.1, arrowprops=dict(arrowstyle="-", lw=0.4, color=RED))
arrow(ax, p_on, -0.58, q_on, -0.58, style="<|-|>")
ax.text((p_on + q_on) / 2, -0.66, "PR = QRS onset − P onset", ha="center", va="top", fontsize=6.2)
ax.text(-0.305, -0.86, "DWT delineation (NeuroKit2);\nmedians over beats and leads",
        fontsize=6.0, va="top", color="#444444", linespacing=1.15)
ax.set_xlim(-0.31, 0.11); ax.set_ylim(-1.05, 0.55); ax.axis("off")

fig.text(0.99, 0.0, "Schematic waveforms; not patient data.", ha="right", fontsize=5.8, color="#666666")
fig.savefig(OUT / "Fig1_acquisition_labelling.png", dpi=DPI, bbox_inches="tight")
print("saved", OUT / "Fig1_acquisition_labelling.png")
