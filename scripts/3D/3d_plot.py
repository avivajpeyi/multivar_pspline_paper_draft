from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import arviz as az
import matplotlib.pyplot as plt
import numpy as np

EPS = 1e-12
warnings.filterwarnings(
    "ignore", message="Attempt to set non-positive ylim on a log-scaled axis"
)


def _calculate_true_var_psd_hz(
    freqs_hz: np.ndarray,
    var_coeffs: np.ndarray,
    sigma: np.ndarray,
    *,
    fs: float = 1.0,
) -> np.ndarray:
    """Compute one-sided theoretical PSD matrix for VAR(p)."""
    freqs_hz = np.asarray(freqs_hz, dtype=np.float64)
    ar_order, n_channels, _ = var_coeffs.shape
    omega = 2.0 * np.pi * freqs_hz / float(fs)
    psd = np.empty((freqs_hz.shape[0], n_channels, n_channels), dtype=np.complex128)
    ident = np.eye(n_channels, dtype=np.complex128)

    for idx, w in enumerate(omega):
        a_f = ident.copy()
        for lag in range(1, ar_order + 1):
            a_f = a_f - var_coeffs[lag - 1] * np.exp(-1j * w * lag)
        h_f = np.linalg.inv(a_f)
        s_f = h_f @ sigma @ h_f.conj().T
        psd[idx] = (2.0 / float(fs)) * s_f

    if freqs_hz.size and np.isclose(freqs_hz[-1], fs / 2.0):
        psd[-1] = 0.5 * psd[-1]

    psd = 0.5 * (psd + np.swapaxes(psd.conj(), -1, -2))
    psd = np.where(np.abs(psd) < EPS, EPS, psd)
    return psd


def _nearest_percentile(values: np.ndarray, percentiles: np.ndarray, q: float) -> np.ndarray:
    idx = int(np.argmin(np.abs(percentiles - q)))
    return np.asarray(values[idx], dtype=np.float64)


def _resolve_default_idatas(repo_root: Path) -> list[Path]:
    base = repo_root / "docs/studies/multivar_psd/out_var3"

    candidates_off = sorted(base.glob("seed_*_*/inference_data.nc"))
    candidates_cg_off = sorted(base.glob("seed_*_*_cgOFF/inference_data.nc"))
    candidates_cg_on = sorted(base.glob("seed_*_*_cgNH*/inference_data.nc"))

    if candidates_cg_off and candidates_cg_on:
        return [candidates_cg_off[0], candidates_cg_on[0]]

    if candidates_off:
        return [candidates_off[0]]

    raise FileNotFoundError(
        "Could not find any inference_data.nc under docs/studies/multivar_psd/out_var3."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create paper plot from one or more 3D VAR(2) InferenceData files."
    )
    parser.add_argument(
        "--idata",
        type=str,
        nargs="+",
        default=None,
        help=(
            "One or more paths to inference_data.nc. "
            "Example: --idata off.nc on.nc"
        ),
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Optional labels matching --idata. "
            "Example: --labels 'No coarse' 'Coarse Nh=4'"
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="docs/manuscript/figures/var3_simulation_idata_overlay.png",
        help="Output figure path.",
    )
    parser.add_argument(
        "--with-true",
        action="store_true",
        help="Overlay theoretical true VAR(2) spectrum used in 3d_study.py.",
    )
    parser.add_argument(
        "--xmax",
        type=float,
        default=0.5,
        help="Upper x-limit in Hz for paper plot focus region.",
    )
    parser.add_argument(
        "--decimate",
        type=int,
        default=1,
        help="Plot every Nth frequency point (default 1 = no decimation).",
    )
    return parser.parse_args()


def _load_summary(idata_path: Path) -> dict:
    idata = az.from_netcdf(idata_path)
    if not hasattr(idata, "posterior_psd"):
        raise ValueError(f"{idata_path} has no posterior_psd group.")
    if "psd_matrix_real" not in idata.posterior_psd:
        raise ValueError(f"{idata_path} posterior_psd has no psd_matrix_real variable.")

    psd_group = idata.posterior_psd
    freq = np.asarray(psd_group.coords["freq"].values, dtype=np.float64)
    percentiles = np.asarray(psd_group.coords["percentile"].values, dtype=np.float64)
    psd_real = np.asarray(psd_group["psd_matrix_real"].values, dtype=np.float64)
    psd_imag = np.asarray(psd_group["psd_matrix_imag"].values, dtype=np.float64)

    summary = {
        "idata": idata,
        "freq": freq,
        "q05_real": _nearest_percentile(psd_real, percentiles, 5.0),
        "q50_real": _nearest_percentile(psd_real, percentiles, 50.0),
        "q95_real": _nearest_percentile(psd_real, percentiles, 95.0),
        "q05_imag": _nearest_percentile(psd_imag, percentiles, 5.0),
        "q50_imag": _nearest_percentile(psd_imag, percentiles, 50.0),
        "q95_imag": _nearest_percentile(psd_imag, percentiles, 95.0),
        "periodogram": None,
    }

    if hasattr(idata, "observed_data") and "periodogram" in idata.observed_data:
        summary["periodogram"] = np.asarray(idata.observed_data["periodogram"].values)

    return summary


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]

    if args.idata:
        idata_paths = []
        for p in args.idata:
            path = Path(p)
            if not path.is_absolute():
                path = repo_root / path
            idata_paths.append(path)
    else:
        idata_paths = _resolve_default_idatas(repo_root)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = repo_root / output_path

    summaries = [_load_summary(p) for p in idata_paths]

    if args.labels is None:
        if len(idata_paths) == 2:
            labels = ["No coarse-grain", "Coarse-grain"]
        else:
            labels = [p.parent.name for p in idata_paths]
    else:
        labels = list(args.labels)
        if len(labels) != len(idata_paths):
            raise ValueError("Number of --labels must match number of --idata paths.")

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    fill_alphas = [0.28, 0.22, 0.18, 0.15]
    line_widths = [1.8, 1.8, 1.5, 1.5]

    # Use first dataset for n_channels and periodogram.
    first = summaries[0]
    n_channels = first["q50_real"].shape[1]
    periodogram = first["periodogram"]

    true_psd_dense = None
    if args.with_true:
        a1 = np.diag([0.4, 0.3, 0.2])
        a2 = np.array(
            [
                [-0.2, 0.5, 0.0],
                [0.4, -0.1, 0.0],
                [0.0, 0.0, -0.1],
            ],
            dtype=np.float64,
        )
        var_coeffs = np.array([a1, a2], dtype=np.float64)
        sigma = np.array(
            [
                [0.25, 0.0, 0.08],
                [0.0, 0.25, 0.08],
                [0.08, 0.08, 0.25],
            ],
            dtype=np.float64,
        )

        max_freq_available = max(float(np.max(s["freq"])) for s in summaries)
        xmax = float(args.xmax) if args.xmax is not None else max_freq_available
        xmax = min(xmax, max_freq_available)
        freq_dense = np.linspace(0.0, xmax, 1200)
        freq_dense = freq_dense[freq_dense > 0.0]
        true_psd_dense = _calculate_true_var_psd_hz(freq_dense, var_coeffs, sigma, fs=1.0)
    else:
        freq_dense = None

    fig, axes = plt.subplots(
        n_channels,
        n_channels,
        figsize=(n_channels * 2.8, n_channels * 2.8),
        sharex=True,
        constrained_layout=False,
    )
    if n_channels == 1:
        axes = np.array([[axes]])

    all_re_candidates: list[np.ndarray] = []
    all_im_candidates: list[np.ndarray] = []
    re_obs_candidates: list[np.ndarray] = []
    im_obs_candidates: list[np.ndarray] = []

    global_xmin = min(float(np.min(s["freq"])) for s in summaries)
    global_xmax = max(float(np.max(s["freq"])) for s in summaries)
    x_min = global_xmin
    x_max = float(args.xmax) if args.xmax is not None else global_xmax
    x_max = min(x_max, global_xmax)

    for s in summaries:
        freq = s["freq"]
        x_mask = (freq >= x_min) & (freq <= x_max)
        if not np.any(x_mask):
            x_mask = np.ones_like(freq, dtype=bool)

        for i in range(n_channels):
            for j in range(n_channels):
                if i <= j:
                    all_re_candidates.extend(
                        [
                            s["q05_real"][:, i, j][x_mask],
                            s["q50_real"][:, i, j][x_mask],
                            s["q95_real"][:, i, j][x_mask],
                        ]
                    )
                else:
                    all_im_candidates.extend(
                        [
                            s["q05_imag"][:, i, j][x_mask],
                            s["q50_imag"][:, i, j][x_mask],
                            s["q95_imag"][:, i, j][x_mask],
                        ]
                    )

    if periodogram is not None:
        freq0 = first["freq"]
        x_mask0 = (freq0 >= x_min) & (freq0 <= x_max)
        for i in range(n_channels):
            for j in range(n_channels):
                if i <= j:
                    re_obs_candidates.append(np.real(periodogram[:, i, j])[x_mask0])
                else:
                    im_obs_candidates.append(np.imag(periodogram[:, i, j])[x_mask0])

    if true_psd_dense is not None:
        truth_mask = (freq_dense >= x_min) & (freq_dense <= x_max)
        for i in range(n_channels):
            for j in range(n_channels):
                if i <= j:
                    all_re_candidates.append(np.real(true_psd_dense[:, i, j])[truth_mask])
                else:
                    all_im_candidates.append(np.imag(true_psd_dense[:, i, j])[truth_mask])

    def _global_limits(candidates: list[np.ndarray], symmetric: bool) -> tuple[float, float]:
        vals = np.concatenate([np.ravel(c) for c in candidates if c.size])
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return (-1.0, 1.0) if symmetric else (0.0, 1.0)
        if symmetric:
            vmax = max(float(np.percentile(np.abs(vals), 99.0)), 1e-8)
            return -1.1 * vmax, 1.1 * vmax
        lo = float(np.percentile(vals, 1.0))
        hi = float(np.percentile(vals, 99.0))
        if hi <= lo:
            span = max(abs(lo), 1.0)
            return lo - 0.1 * span, hi + 0.1 * span
        pad = 0.08 * (hi - lo)
        return lo - pad, hi + pad

    re_ylim = _global_limits(
        re_obs_candidates if re_obs_candidates else all_re_candidates,
        symmetric=False,
    )
    im_ylim = _global_limits(
        im_obs_candidates if im_obs_candidates else all_im_candidates,
        symmetric=True,
    )

    for i in range(n_channels):
        for j in range(n_channels):
            ax = axes[i, j]

            if periodogram is not None:
                freq_obs = first["freq"]
                step = max(1, int(args.decimate))
                idx_obs = np.arange(0, freq_obs.size, step, dtype=int)
                if idx_obs[-1] != freq_obs.size - 1:
                    idx_obs = np.append(idx_obs, freq_obs.size - 1)

                obs_arr = (
                    np.real(periodogram[:, i, j])
                    if i <= j
                    else np.imag(periodogram[:, i, j])
                )
                ax.plot(
                    freq_obs[idx_obs],
                    obs_arr[idx_obs],
                    color="0.82",
                    lw=0.7,
                    alpha=0.9,
                    zorder=-10,
                    label="Periodogram" if (i == 0 and j == 0) else None,
                )

            if true_psd_dense is not None:
                truth_arr = (
                    np.real(true_psd_dense[:, i, j])
                    if i <= j
                    else np.imag(true_psd_dense[:, i, j])
                )
                ax.plot(
                    freq_dense,
                    truth_arr,
                    color="k",
                    lw=2.0,
                    ls="--",
                    zorder=2,
                    alpha=0.85,
                    label="True PSD" if (i == 0 and j == 0) else None,
                )

            for k, (s, label) in enumerate(zip(summaries, labels)):
                freq = s["freq"]
                step = max(1, int(args.decimate))
                idx = np.arange(0, freq.size, step, dtype=int)
                if idx[-1] != freq.size - 1:
                    idx = np.append(idx, freq.size - 1)

                if i <= j:
                    lower = s["q05_real"][:, i, j]
                    median = s["q50_real"][:, i, j]
                    upper = s["q95_real"][:, i, j]
                    ylabel = r"$\Re\{S_{%d%d}(f)\}$" % (i + 1, j + 1)
                else:
                    lower = s["q05_imag"][:, i, j]
                    median = s["q50_imag"][:, i, j]
                    upper = s["q95_imag"][:, i, j]
                    ylabel = r"$\Im\{S_{%d%d}(f)\}$" % (i + 1, j + 1)

                ax.fill_between(
                    freq[idx],
                    lower[idx],
                    upper[idx],
                    color=colors[k % len(colors)],
                    alpha=fill_alphas[k % len(fill_alphas)],
                    linewidth=0.0,
                    zorder=3 + k,
                    label=f"{label} 90% CI" if (i == 0 and j == 0) else None,
                )
                ax.plot(
                    freq[idx],
                    median[idx],
                    color=colors[k % len(colors)],
                    lw=line_widths[k % len(line_widths)],
                    zorder=6 + k,
                    alpha=0.95,
                    label=f"{label} median" if (i == 0 and j == 0) else None,
                )

            ax.set_xlim(x_min, x_max)
            if i <= j:
                ax.set_ylim(*re_ylim)
            else:
                ax.set_ylim(*im_ylim)
                ax.axhline(0.0, color="0.35", lw=0.7, alpha=0.7, zorder=2)
                ax.set_facecolor((0.96, 0.96, 0.96))

            panel_key = (i + 1, j + 1)
            if panel_key in {(1, 1), (2, 2)}:
                ax.set_ylim(0.0, 3.0)
            elif panel_key in {(1, 3), (2, 3)}:
                ax.set_ylim(-0.5, 1.0)
            elif panel_key == (3, 3):
                ax.set_ylim(0.0, 1.5)
            elif panel_key in {(3, 1), (3, 2)}:
                ax.set_ylim(-0.25, 0.5)

            ax.grid(alpha=0.25, linewidth=0.5)
            if i == n_channels - 1:
                ax.set_xlabel("Frequency (Hz)")
            if j == 0:
                ax.set_ylabel(ylabel)
            ax.set_title(f"({i+1},{j+1})", fontsize=9)

    handles, labels_out = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels_out,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.03),
            ncol=3,
            frameon=False,
        )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    if output_path.suffix.lower() == ".pdf":
        fig.savefig(output_path.with_suffix(".png"), dpi=220, bbox_inches="tight")

    print("Loaded InferenceData files:")
    for p in idata_paths:
        print(f"  - {p}")
    print(f"Saved figure: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())