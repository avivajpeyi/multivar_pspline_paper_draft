#!/usr/bin/env python3
"""
Standalone triangle-plot renderer for the LISA eta=0.5 comparison bundle.

This script intentionally avoids any project-specific imports. It only needs:
  - Python standard library
  - numpy
  - h5py
  - matplotlib

Example:
  python plot_triangle_from_h5.py \
      --input triangle_plot_data_eta0p5.h5 \
      --outdir .
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


DURATIONS = ("1m", "6m", "1y")
CHANNELS = ("X", "Y", "Z")
DURATION_COLORS = {
    "1m": "tab:blue",
    "6m": "tab:orange",
    "1y": "tab:green",
}
DURATION_LABELS = {
    "1m": "1 month",
    "6m": "6 months",
    "1y": "12 months",
}

FMIN_PLOT = 1e-4
FMAX_PLOT = 1e-1
PSD_FLOOR = 1e-50
FIG_DPI = 300
FIGSIZE = (7.2, 6.2)

POSTERIOR_FILL_ALPHA = 0.26
POSTERIOR_FILL_ALPHA_PSD = 0.34
WELCH_COLOR = "0.65"
WELCH_ALPHA = 0.6
WELCH_WIDTH = 1.05
WELCH_LINESTYLE = (0, (3, 2))  # short dashed
LABEL_FONT_SIZE = 15


plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "axes.linewidth": 0.9,
        "legend.fontsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    }
)


def _coherence_from_matrix(spectral_matrix: np.ndarray, i: int, j: int) -> np.ndarray:
    """Return magnitude coherence from a spectral matrix with shape `(F, 3, 3)`."""
    sii = np.maximum(np.asarray(spectral_matrix[:, i, i].real, dtype=np.float64), 0.0)
    sjj = np.maximum(np.asarray(spectral_matrix[:, j, j].real, dtype=np.float64), 0.0)
    denom = np.sqrt(sii * sjj)
    return np.clip(
        np.divide(
            np.abs(np.asarray(spectral_matrix[:, i, j], dtype=np.complex128)),
            denom,
            out=np.zeros_like(denom, dtype=np.float64),
            where=denom > 0.0,
        ),
        0.0,
        1.0,
    )


def _plot_reference_curve(
    ax: plt.Axes,
    freq: np.ndarray,
    values: np.ndarray,
    *,
    is_psd: bool,
    color: str,
    linewidth: float,
    alpha: float,
    zorder: int,
) -> None:
    """Draw one Welch reference curve."""
    if is_psd:
        ax.loglog(
            freq,
            np.maximum(np.asarray(values, dtype=np.float64), PSD_FLOOR),
            color=color,
            lw=linewidth,
            alpha=alpha,
            ls=WELCH_LINESTYLE,
            zorder=zorder,
        )
        return

    ax.semilogx(
        freq,
        np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0),
        color=color,
        lw=linewidth,
        alpha=alpha,
        ls=WELCH_LINESTYLE,
        zorder=zorder,
    )


def _plot_posterior_band(
    ax: plt.Axes,
    freq: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    *,
    is_psd: bool,
    color: str,
) -> None:
    """Plot a 90% posterior interval band."""
    low_arr = np.asarray(low, dtype=np.float64)
    high_arr = np.asarray(high, dtype=np.float64)

    if is_psd:
        ax.fill_between(
            freq,
            np.maximum(low_arr, PSD_FLOOR),
            np.maximum(high_arr, PSD_FLOOR),
            color=color,
            alpha=POSTERIOR_FILL_ALPHA_PSD,
            zorder=4,
        )
        return

    ax.fill_between(
        freq,
        np.clip(low_arr, 0.0, 1.0),
        np.clip(high_arr, 0.0, 1.0),
        color=color,
        alpha=POSTERIOR_FILL_ALPHA,
        zorder=4,
    )


def _panel_math_label(i: int, j: int) -> str:
    """Return a manuscript-style panel label."""
    if i == j:
        ch = CHANNELS[i]
        return rf"$S_{{{ch}{ch}}}$"
    return rf"$|C_{{{CHANNELS[i]}{CHANNELS[j]}}}|$"


def _set_psd_axis_limits(ax: plt.Axes, values: list[np.ndarray]) -> None:
    """Set robust PSD limits without letting very small values dominate."""
    positive: list[np.ndarray] = []
    for arr in values:
        flat = np.asarray(arr, dtype=np.float64).ravel()
        flat = flat[np.isfinite(flat) & (flat > 0.0)]
        if flat.size > 0:
            positive.append(flat)
    if not positive:
        return
    merged = np.concatenate(positive)
    ylo = max(float(np.quantile(merged, 0.01)) * 0.35, PSD_FLOOR)
    yhi = float(np.quantile(merged, 0.995)) * 4.0
    if yhi > ylo:
        ax.set_ylim(ylo, yhi)


def _make_legend_handles() -> list[object]:
    """Build the shared figure legend."""
    handles: list[object] = [
        Line2D(
            [0],
            [0],
            color=WELCH_COLOR,
            lw=WELCH_WIDTH,
            ls=WELCH_LINESTYLE,
            alpha=WELCH_ALPHA,
            label="Welch reference",
        )
    ]
    for duration in DURATIONS:
        handles.append(
            Patch(
                facecolor=DURATION_COLORS[duration],
                edgecolor=DURATION_COLORS[duration],
                alpha=POSTERIOR_FILL_ALPHA,
                label=f"{DURATION_LABELS[duration]} 90% CI",
            )
        )
    return handles


def _read_run(group: h5py.Group) -> dict[str, np.ndarray | str | None]:
    """Read one duration run from HDF5."""
    return {
        "noise": group.attrs["noise"],
        "duration": group.attrs["duration"],
        "eta": group.attrs["eta"],
        "freq": group["freq"][...],
        "q05": group["q05"][...],
        "q50": group["q50"][...],
        "q95": group["q95"][...],
        "coh_q05": group["coh_q05"][...] if "coh_q05" in group else None,
        "coh_q50": group["coh_q50"][...] if "coh_q50" in group else None,
        "coh_q95": group["coh_q95"][...] if "coh_q95" in group else None,
    }


def _decode_attr(value) -> str:
    """Decode an HDF5 string attr if necessary."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _load_bundle(h5_path: Path, noise_name: str) -> dict[str, object]:
    """Load one noise bundle from the shared HDF5 file."""
    with h5py.File(h5_path, "r") as h5:
        root_eta = _decode_attr(h5.attrs.get("eta", "eta0.5"))
        noise_group = h5[noise_name]
        ref_group = noise_group["reference"]
        runs_group = noise_group["runs"]
        runs = {
            duration: _read_run(runs_group[duration])
            for duration in DURATIONS
            if duration in runs_group
        }
        return {
            "eta": root_eta,
            "noise": noise_name,
            "psd_welch_freq": ref_group["psd_welch_freq"][...],
            "psd_welch_matrix": ref_group["psd_welch_matrix"][...],
            "coh_welch_freq": ref_group["coh_welch_freq"][...],
            "coh_welch_matrix": ref_group["coh_welch_matrix"][...],
            "runs": runs,
        }


def plot_triangle(bundle: dict[str, object], out_path: Path) -> None:
    """Render one triangle plot from an in-memory HDF5 bundle."""
    psd_welch_freq = np.asarray(bundle["psd_welch_freq"])
    psd_welch_matrix = np.asarray(bundle["psd_welch_matrix"])
    coh_welch_freq = np.asarray(bundle["coh_welch_freq"])
    coh_welch_matrix = np.asarray(bundle["coh_welch_matrix"])
    runs = bundle["runs"]

    fig, axes = plt.subplots(3, 3, figsize=FIGSIZE, constrained_layout=False)

    for i in range(3):
        for j in range(3):
            ax = axes[i, j]

            if i < j:
                ax.axis("off")
                continue

            is_psd = i == j
            if is_psd:
                welch_freq = psd_welch_freq
                welch_values = psd_welch_matrix[:, i, i].real
            else:
                welch_freq = coh_welch_freq
                welch_values = _coherence_from_matrix(coh_welch_matrix, i, j)

            _plot_reference_curve(
                ax,
                welch_freq,
                welch_values,
                is_psd=is_psd,
                color=WELCH_COLOR,
                linewidth=WELCH_WIDTH,
                alpha=WELCH_ALPHA,
                zorder=2,
            )

            for duration in DURATIONS:
                run = runs.get(duration)
                if run is None:
                    continue

                freq = np.asarray(run["freq"])
                q05 = np.asarray(run["q05"])
                q95 = np.asarray(run["q95"])
                coh_q05 = run["coh_q05"]
                coh_q95 = run["coh_q95"]

                if is_psd:
                    _plot_posterior_band(
                        ax,
                        freq,
                        np.maximum(q05[:, i, i], PSD_FLOOR),
                        np.maximum(q95[:, i, i], PSD_FLOOR),
                        is_psd=True,
                        color=DURATION_COLORS[duration],
                    )
                else:
                    if coh_q05 is not None and coh_q95 is not None:
                        low = np.asarray(coh_q05)[:, i, j]
                        high = np.asarray(coh_q95)[:, i, j]
                    else:
                        low = _coherence_from_matrix(q05, i, j)
                        high = _coherence_from_matrix(q95, i, j)
                    _plot_posterior_band(
                        ax,
                        freq,
                        low,
                        high,
                        is_psd=False,
                        color=DURATION_COLORS[duration],
                    )

            ax.set_xlim(FMIN_PLOT, FMAX_PLOT)
            ax.grid(True, which="major", ls=":", alpha=0.28, lw=0.6)
            ax.grid(False, which="minor")

            if is_psd:
                _set_psd_axis_limits(
                    ax,
                    [
                        psd_welch_matrix[:, i, i].real,
                        *[
                            np.maximum(np.asarray(run["q05"])[:, i, i], PSD_FLOOR)
                            for run in runs.values()
                        ],
                        *[
                            np.maximum(np.asarray(run["q95"])[:, i, i], PSD_FLOOR)
                            for run in runs.values()
                        ],
                    ],
                )
                if j == 0:
                    ax.set_ylabel("PSD [1/Hz]")
            else:
                ax.set_ylim(-0.01, 1.0)
                if j == 0:
                    ax.set_ylabel(r"$|C_{ij}(f)|$")

            ax.text(
                0.04,
                0.93,
                _panel_math_label(i, j),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=LABEL_FONT_SIZE,
            )

            if i == 2:
                ax.set_xlabel("Frequency [Hz]")
            else:
                ax.tick_params(labelbottom=False)

    fig.legend(
        handles=_make_legend_handles(),
        loc="upper right",
        ncol=1,
        bbox_to_anchor=(0.975, 0.975),
        frameon=False,
        columnspacing=0.9,
        handlelength=2.2,
        handletextpad=0.6,
        fontsize=11,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 1.0), h_pad=0.45, w_pad=0.45)
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to triangle_plot_data_eta0p5.h5",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("."),
        help="Directory for output PDF files",
    )
    parser.add_argument(
        "--noise",
        nargs="*",
        default=["noise_4a", "noise_5a"],
        choices=["noise_4a", "noise_5a"],
        help="Which noise bundles to plot",
    )
    return parser.parse_args()


def main() -> None:
    """Load the HDF5 bundle and render the requested triangle plots."""
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    for noise_name in args.noise:
        bundle = _load_bundle(args.input, noise_name)
        eta_label = str(bundle["eta"]).replace(".", "p")
        noise_tag = noise_name.replace("_", "")
        out_path = args.outdir / f"triangle_{noise_tag}_{eta_label}.pdf"
        plot_triangle(bundle, out_path)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
