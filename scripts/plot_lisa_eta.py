"""
Plot the LISA noise4a eta-sensitivity sweep for the manuscript appendix.

Three panels: median relative CI width on diagonal PSDs, coverage,
and matrix RIAE versus eta, with one curve per duration (1m, 6m, 1y).
Converged runs are solid lines with filled markers; tree-depth-saturated
runs extend the line as a dashed segment with open markers.

Output: figures/lisa_eta_sweep.pdf
"""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CSV = HERE.parent / "data" / "lisa_eta_sweep.csv"
OUT = HERE.parent / "figures" / "lisa_eta_sweep.pdf"

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.labelsize": 8,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
    }
)

df = pd.read_csv(CSV)
df["failed"] = df["failed"].astype(str).str.lower() == "true"

durations = ["1m", "6m", "1y"]
colors = {"1m": "tab:blue", "6m": "tab:orange", "1y": "tab:green"}
labels = {"1m": "1 month", "6m": "6 months", "1y": "1 year"}

ETA_CHOSEN = 0.5

fig, axes = plt.subplots(
    2,
    1,
    figsize=(3.25, 4.15),
    sharex=True,
    constrained_layout=True,
)

panels = [
    ("ci_width_rel_psd_diag_median", "Median rel. CI width\n(diag. PSD)"),
    ("riae", "Matrix RIAE"),
]

for ax, (col, ylabel) in zip(axes, panels):
    for dur in durations:
        sub = df[df["duration"] == dur].sort_values("eta_value")
        ax.plot(
            sub["eta_value"], sub[col],
            "-o", color=colors[dur], label=labels[dur],
            markersize=3.7, linewidth=1.1,
        )

    ax.axvline(ETA_CHOSEN, color="0.55", linestyle=":", linewidth=0.8, zorder=0)
    ax.set_xscale("log")
    # ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(ylabel)
    ax.set_xticks([0.01, 0.1, 1.0])
    ax.xaxis.set_major_formatter(
        mticker.LogFormatterSciNotation(base=10, labelOnlyBase=True)
    )
    ax.grid(True, which="major", ls=":", alpha=0.35)
    ax.grid(True, which="minor", ls=":", alpha=0.18)

# RIAE: zoom in to make the ~5% flatness legible
riae_all = df["riae"].values
riae_lo, riae_hi = riae_all.min(), riae_all.max()
pad = (riae_hi - riae_lo) * 0.6
axes[1].set_ylim(riae_lo - pad, riae_hi + pad)
axes[1].yaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
axes[1].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))

axes[0].legend(loc="upper right", frameon=False, handlelength=2.0)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=300)
print(f"wrote {OUT}")
