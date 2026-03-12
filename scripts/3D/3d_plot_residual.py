from __future__ import annotations

import argparse
import json
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


def _interp_complex_matrix_to_freq(
    source_freq: np.ndarray,
    target_freq: np.ndarray,
    source_matrix: np.ndarray,
) -> np.ndarray:
    """Interpolate complex matrix-valued spectrum from source to target frequency grid."""
    source_freq = np.asarray(source_freq, dtype=np.float64)
    target_freq = np.asarray(target_freq, dtype=np.float64)
    source_matrix = np.asarray(source_matrix)

    if source_matrix.ndim != 3:
        raise ValueError(f"Expected matrix with shape (F, C, C); got {source_matrix.shape}.")
    if source_matrix.shape[0] != source_freq.size:
        raise ValueError(
            "Frequency and matrix length mismatch: "
            f"{source_freq.size} vs {source_matrix.shape[0]}."
        )

    n_target = int(target_freq.size)
    n_channels = int(source_matrix.shape[1])
    out = np.empty((n_target, n_channels, n_channels), dtype=np.complex128)
    for i in range(n_channels):
        for j in range(n_channels):
            re = np.interp(
                target_freq,
                source_freq,
                np.real(source_matrix[:, i, j]),
                left=np.nan,
                right=np.nan,
            )
            im = np.interp(
                target_freq,
                source_freq,
                np.imag(source_matrix[:, i, j]),
                left=np.nan,
                right=np.nan,
            )
            out[:, i, j] = re + 1j * im
    return out


def _resolve_default_idatas(repo_root: Path) -> list[Path]:
    base = repo_root / "docs/manuscript/scripts/3D/out_var3"

    candidates_cg_off_npz = sorted(
        base.glob("seed_*_*_cgOFF/posterior_ci_summary.npz")
    )
    candidates_cg_on_npz = sorted(
        base.glob("seed_*_*_cgNH*/posterior_ci_summary.npz")
    )
    candidates_any_new_npz = sorted(base.glob("seed_*_*/posterior_ci_summary.npz"))
    candidates_any_old_npz = sorted(base.glob("seed_*_*/compact_ci_curves.npz"))
    candidates_cg_off_nc = sorted(base.glob("seed_*_*_cgOFF/inference_data.nc"))
    candidates_cg_on_nc = sorted(base.glob("seed_*_*_cgNH*/inference_data.nc"))

    if candidates_cg_off_npz and candidates_cg_on_npz:
        return [candidates_cg_off_npz[0], candidates_cg_on_npz[0]]

    if candidates_cg_off_nc and candidates_cg_on_nc:
        return [candidates_cg_off_nc[0], candidates_cg_on_nc[0]]

    if candidates_any_new_npz:
        return [candidates_any_new_npz[0]]

    if candidates_any_old_npz:
        return [candidates_any_old_npz[0]]

    raise FileNotFoundError(
        "Could not find posterior_ci_summary.npz, compact_ci_curves.npz, "
        "or inference_data.nc under docs/manuscript/scripts/3D/out_var3."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create paper plot from one or more 3D VAR(2) outputs. "
            "Each input may be posterior_ci_summary.npz, compact_ci_curves.npz, "
            "or inference_data.nc."
        )
    )
    parser.add_argument(
        "--idata",
        type=str,
        nargs="+",
        default=None,
        help=(
            "One or more paths to posterior_ci_summary.npz, compact_ci_curves.npz, "
            "or inference_data.nc. "
            "Example: --idata off.nc on.npz"
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
        default="var3_simulation_residual_ci_overlay.png",
        help="Output figure path.",
    )
    parser.add_argument(
        "--with-true",
        action="store_true",
        help="Use theoretical true VAR(2) spectrum from 3d_study.py when truth is not in inputs.",
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


def _reconstruct_quantiles_from_compact(data) -> tuple[np.ndarray, ...]:
    """Rebuild full (F, P, P) quantile arrays from compact diag/offdiag format."""
    freq = np.asarray(data["freq"], dtype=np.float64)
    diag_q05 = np.asarray(data["psd_diag_q05"], dtype=np.float64)
    diag_q50 = np.asarray(data["psd_diag_q50"], dtype=np.float64)
    diag_q95 = np.asarray(data["psd_diag_q95"], dtype=np.float64)
    off_re_q05 = np.asarray(data["psd_offre_q05"], dtype=np.float64)
    off_re_q50 = np.asarray(data["psd_offre_q50"], dtype=np.float64)
    off_re_q95 = np.asarray(data["psd_offre_q95"], dtype=np.float64)
    off_im_q05 = np.asarray(data["psd_offim_q05"], dtype=np.float64)
    off_im_q50 = np.asarray(data["psd_offim_q50"], dtype=np.float64)
    off_im_q95 = np.asarray(data["psd_offim_q95"], dtype=np.float64)
    pairs = np.asarray(data["offdiag_pairs"], dtype=int)

    p = int(diag_q50.shape[1])
    f = int(freq.size)

    q05_real = np.zeros((f, p, p), dtype=np.float64)
    q50_real = np.zeros((f, p, p), dtype=np.float64)
    q95_real = np.zeros((f, p, p), dtype=np.float64)
    q05_imag = np.zeros((f, p, p), dtype=np.float64)
    q50_imag = np.zeros((f, p, p), dtype=np.float64)
    q95_imag = np.zeros((f, p, p), dtype=np.float64)

    diag_idx = np.arange(p)
    q05_real[:, diag_idx, diag_idx] = diag_q05
    q50_real[:, diag_idx, diag_idx] = diag_q50
    q95_real[:, diag_idx, diag_idx] = diag_q95

    for k, (i, j) in enumerate(pairs):
        q05_real[:, i, j] = off_re_q05[:, k]
        q50_real[:, i, j] = off_re_q50[:, k]
        q95_real[:, i, j] = off_re_q95[:, k]

        q05_real[:, j, i] = off_re_q05[:, k]
        q50_real[:, j, i] = off_re_q50[:, k]
        q95_real[:, j, i] = off_re_q95[:, k]

        q05_imag[:, j, i] = off_im_q05[:, k]
        q50_imag[:, j, i] = off_im_q50[:, k]
        q95_imag[:, j, i] = off_im_q95[:, k]

        q05_imag[:, i, j] = -off_im_q05[:, k]
        q50_imag[:, i, j] = -off_im_q50[:, k]
        q95_imag[:, i, j] = -off_im_q95[:, k]

    return freq, q05_real, q50_real, q95_real, q05_imag, q50_imag, q95_imag


def _load_summary_from_npz(npz_path: Path) -> dict:
    with np.load(npz_path, allow_pickle=False) as data:
        freq = np.asarray(data["freq"], dtype=np.float64)

        if all(
            key in data
            for key in (
                "psd_real_q05",
                "psd_real_q50",
                "psd_real_q95",
                "psd_imag_q05",
                "psd_imag_q50",
                "psd_imag_q95",
            )
        ):
            q05_real = np.asarray(data["psd_real_q05"], dtype=np.float64)
            q50_real = np.asarray(data["psd_real_q50"], dtype=np.float64)
            q95_real = np.asarray(data["psd_real_q95"], dtype=np.float64)
            q05_imag = np.asarray(data["psd_imag_q05"], dtype=np.float64)
            q50_imag = np.asarray(data["psd_imag_q50"], dtype=np.float64)
            q95_imag = np.asarray(data["psd_imag_q95"], dtype=np.float64)
        else:
            (
                freq,
                q05_real,
                q50_real,
                q95_real,
                q05_imag,
                q50_imag,
                q95_imag,
            ) = _reconstruct_quantiles_from_compact(data)

        periodogram = None
        if "periodogram_real" in data and "periodogram_imag" in data:
            periodogram = np.asarray(data["periodogram_real"], dtype=np.float64) + 1j * np.asarray(
                data["periodogram_imag"], dtype=np.float64
            )
        truth = None
        if "truth_real" in data and "truth_imag" in data:
            truth = np.asarray(data["truth_real"], dtype=np.float64) + 1j * np.asarray(
                data["truth_imag"], dtype=np.float64
            )

    metrics: dict[str, float] = {}
    metrics_path = npz_path.parent / "metrics_summary.json"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        for k, v in loaded.items():
            if isinstance(v, (int, float)):
                metrics[k] = float(v)

    return {
        "freq": freq,
        "q05_real": q05_real,
        "q50_real": q50_real,
        "q95_real": q95_real,
        "q05_imag": q05_imag,
        "q50_imag": q50_imag,
        "q95_imag": q95_imag,
        "periodogram": periodogram,
        "truth": truth,
        "metrics": metrics,
    }


def _load_summary(idata_path: Path) -> dict:
    if idata_path.suffix.lower() == ".npz":
        return _load_summary_from_npz(idata_path)

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

    attrs = getattr(idata, "attrs", {})
    metric_keys = (
        "lnz",
        "lnz_err",
        "riae_matrix",
        "coverage",
        "runtime",
        "ess_median",
        "ciw_psd_diag_mean",
        "ciw_psd_offdiag_mean",
        "ciw_coh_offdiag_mean",
    )
    metrics = {}
    for key in metric_keys:
        if key in attrs:
            try:
                metrics[key] = float(attrs[key])
            except Exception:
                pass

    summary = {
        "freq": freq,
        "q05_real": _nearest_percentile(psd_real, percentiles, 5.0),
        "q50_real": _nearest_percentile(psd_real, percentiles, 50.0),
        "q95_real": _nearest_percentile(psd_real, percentiles, 95.0),
        "q05_imag": _nearest_percentile(psd_imag, percentiles, 5.0),
        "q50_imag": _nearest_percentile(psd_imag, percentiles, 50.0),
        "q95_imag": _nearest_percentile(psd_imag, percentiles, 95.0),
        "periodogram": None,
        "truth": None,
        "metrics": metrics,
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
            # if not path.is_absolute():
            #     path = repo_root / path
            idata_paths.append(path)
    else:
        idata_paths = _resolve_default_idatas(repo_root)

    output_path = Path(args.output)
    # if not output_path.is_absolute():
    #     output_path = repo_root / output_path

    summaries = []
    for p in idata_paths:
        s = _load_summary(p)
        s["source_path"] = str(p)
        summaries.append(s)

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
    fill_alphas = [0.4, 0.4, 0.4, 0.4]

    # Use first dataset for channel dimensions.
    first = summaries[0]
    n_channels = first["q50_real"].shape[1]

    truth_summary = next((s for s in summaries if s.get("truth") is not None), None)
    if truth_summary is not None:
        freq_dense = truth_summary["freq"]
        true_psd_dense = truth_summary["truth"]
    elif args.with_true:
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
        true_psd_dense = None
        freq_dense = None

    residual_summaries = []
    for s in summaries:
        freq = np.asarray(s["freq"], dtype=np.float64)
        truth = s.get("truth")
        truth_freq = freq

        if truth is None:
            if true_psd_dense is None or freq_dense is None:
                raise ValueError(
                    "Residual plotting requires truth spectra. Provide npz files with "
                    "truth_real/truth_imag, or use --with-true to synthesize VAR(2) truth."
                )
            truth = true_psd_dense
            truth_freq = freq_dense

        if truth.shape[0] != freq.size or not np.array_equal(np.asarray(truth_freq), freq):
            truth_on_freq = _interp_complex_matrix_to_freq(truth_freq, freq, truth)
        else:
            truth_on_freq = np.asarray(truth, dtype=np.complex128)

        residual_summaries.append(
            {
                "freq": freq,
                "r05_real": s["q05_real"] - np.real(truth_on_freq),
                "r50_real": s["q50_real"] - np.real(truth_on_freq),
                "r95_real": s["q95_real"] - np.real(truth_on_freq),
                "r05_imag": s["q05_imag"] - np.imag(truth_on_freq),
                "r50_imag": s["q50_imag"] - np.imag(truth_on_freq),
                "r95_imag": s["q95_imag"] - np.imag(truth_on_freq),
            }
        )

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

    global_xmin = min(float(np.min(s["freq"])) for s in residual_summaries)
    global_xmax = max(float(np.max(s["freq"])) for s in residual_summaries)
    x_min = global_xmin
    x_max = float(args.xmax) if args.xmax is not None else global_xmax
    x_max = min(x_max, global_xmax)

    for s in residual_summaries:
        freq = s["freq"]
        x_mask = (freq >= x_min) & (freq <= x_max)
        if not np.any(x_mask):
            x_mask = np.ones_like(freq, dtype=bool)

        for i in range(n_channels):
            for j in range(n_channels):
                if i <= j:
                    all_re_candidates.extend(
                        [
                            s["r05_real"][:, i, j][x_mask],
                            s["r50_real"][:, i, j][x_mask],
                            s["r95_real"][:, i, j][x_mask],
                        ]
                    )
                else:
                    all_im_candidates.extend(
                        [
                            s["r05_imag"][:, i, j][x_mask],
                            s["r50_imag"][:, i, j][x_mask],
                            s["r95_imag"][:, i, j][x_mask],
                        ]
                    )

    def _global_limits(candidates: list[np.ndarray], symmetric: bool) -> tuple[float, float]:
        if not candidates:
            return (-1.0, 1.0) if symmetric else (0.0, 1.0)
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

    re_ylim = _global_limits(all_re_candidates, symmetric=True)
    im_ylim = _global_limits(all_im_candidates, symmetric=True)

    for i in range(n_channels):
        for j in range(n_channels):
            ax = axes[i, j]

            for k, (s, label) in enumerate(zip(residual_summaries, labels)):
                freq = s["freq"]
                step = max(1, int(args.decimate))
                idx = np.arange(0, freq.size, step, dtype=int)
                if idx[-1] != freq.size - 1:
                    idx = np.append(idx, freq.size - 1)

                if i <= j:
                    lower = s["r05_real"][:, i, j]
                    upper = s["r95_real"][:, i, j]
                    ylabel = r"$\Re\{S_{%d%d}(f)-S^{\mathrm{true}}_{%d%d}(f)\}$" % (
                        i + 1,
                        j + 1,
                        i + 1,
                        j + 1,
                    )
                else:
                    lower = s["r05_imag"][:, i, j]
                    upper = s["r95_imag"][:, i, j]
                    ylabel = r"$\Im\{S_{%d%d}(f)-S^{\mathrm{true}}_{%d%d}(f)\}$" % (
                        i + 1,
                        j + 1,
                        i + 1,
                        j + 1,
                    )

                ax.fill_between(
                    freq[idx],
                    lower[idx],
                    upper[idx],
                    color=colors[k % len(colors)],
                    alpha=fill_alphas[k % len(fill_alphas)],
                    linewidth=0.0,
                    zorder=3 + k,
                    label=f"{label} residual 90% CI" if (i == 0 and j == 0) else None,
                )

            ax.set_xlim(x_min, x_max)
            if i <= j:
                ax.set_ylim(*re_ylim)
            else:
                ax.set_ylim(*im_ylim)
                ax.set_facecolor((0.96, 0.96, 0.96))

            ax.axhline(0.0, color="0.35", lw=0.7, alpha=0.7, zorder=2)

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

    # output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    if output_path.suffix.lower() == ".pdf":
        fig.savefig(output_path.with_suffix(".png"), dpi=220, bbox_inches="tight")

    print("Loaded input files:")
    for p in idata_paths:
        print(f"  - {p}")
    print("")
    print("Run stats:")
    report_keys = [
        "lnz",
        "lnz_err",
        "riae_matrix",
        "coverage",
        "runtime",
        "ess_median",
        "ciw_psd_diag_mean",
        "ciw_psd_offdiag_mean",
        "ciw_coh_offdiag_mean",
    ]
    for label, summary in zip(labels, summaries):
        print(f"  [{label}]")
        metrics = summary.get("metrics", {})
        if not metrics:
            print("    (no metrics found)")
            continue
        for key in report_keys:
            if key in metrics and np.isfinite(metrics[key]):
                print(f"    {key}: {metrics[key]:.6g}")
    print(f"Saved figure: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
