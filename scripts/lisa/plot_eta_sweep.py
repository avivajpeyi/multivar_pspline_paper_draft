"""
Plot the LISA noise4a η-sensitivity sweep for the manuscript appendix.

Three panels: median relative CI width on diagonal PSDs, coverage,
and matrix-RIAE versus η, with one curve per duration (1m, 6m, 1y).
Failed runs (max-treedepth saturation) are marked but not connected.

Output: ../../figures/lisa_eta_sweep.pdf
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CSV = HERE / "eta_sweep_noise4a.csv"
OUT = HERE.parent.parent / "figures" / "lisa_eta_sweep.pdf"

df = pd.read_csv(CSV)
df["failed"] = df["failed"].astype(str).str.lower() == "true"

durations = ["1m", "6m", "1y"]
colors = {"1m": "tab:blue", "6m": "tab:orange", "1y": "tab:green"}
labels = {"1m": "1 month", "6m": "6 months", "1y": "1 year"}

fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.2), sharex=True)

panels = [
    ("ci_width_rel_psd_diag_median", r"Median rel.\ CI width on diag.\ PSD"),
    ("coverage", "Coverage of 90\\% CI"),
    ("riae",   "Matrix RIAE"),
]

for ax, (col, ylabel) in zip(axes, panels):
    for dur in durations:
        sub = df[df["duration"] == dur].sort_values("eta_value")
        good = sub[~sub["failed"]]
        bad = sub[sub["failed"]]
        ax.plot(
            good["eta_value"], good[col],
            "-o", color=colors[dur], label=labels[dur], markersize=4, linewidth=1.4,
        )
        ax.plot(
            bad["eta_value"], bad[col],
            "x", color=colors[dur], markersize=8, markeredgewidth=1.5,
        )
    ax.set_xscale("log")
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

axes[1].axhline(0.9, color="grey", linestyle="--", linewidth=0.8)
axes[1].text(0.012, 0.92, "nominal", color="grey", fontsize=8)

axes[0].legend(loc="best", frameon=False, fontsize=9)

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
print(f"wrote {OUT}")
