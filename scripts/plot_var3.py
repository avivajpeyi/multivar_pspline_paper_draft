from __future__ import annotations

import argparse
import json
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import numpy as np

EPS = 1e-12
warnings.filterwarnings(
    "ignore", message="Attempt to set non-positive ylim on a log-scaled axis"
)

VAR3_BURN_IN = 512
VAR3_A1 = np.diag([0.4, 0.3, 0.2])
VAR3_A2 = np.array(
    [[-0.2, 0.5, 0.0], [0.4, -0.1, 0.0], [0.0, 0.0, -0.1]],
    dtype=np.float64,
)
VAR3_COEFFS = np.array([VAR3_A1, VAR3_A2], dtype=np.float64)
VAR3_SIGMA = np.array(
    [[0.25, 0.0, 0.08], [0.0, 0.25, 0.08], [0.08, 0.08, 0.25]],
    dtype=np.float64,
)


def _plain_log_tick(value: float, _pos: float) -> str:
    """Format log-scale ticks as plain decimals for manuscript figures."""
    if value <= 0 or not np.isfinite(value):
        return ""
    if value >= 1:
        if np.isclose(value, round(value)):
            return str(int(round(value)))
        return f"{value:.1f}".rstrip("0").rstrip(".")
    return f"{value:.2f}".rstrip("0").rstrip(".")


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
    psd = np.empty(
        (freqs_hz.shape[0], n_channels, n_channels), dtype=np.complex128
    )
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


def _simulate_var3_observed_data(n: int, seed: int) -> np.ndarray:
    """Regenerate the VAR3 data to recover legacy periodogram scale factors."""
    ar_order, n_channels, _ = VAR3_COEFFS.shape
    n_total = int(n) + VAR3_BURN_IN
    rng = np.random.default_rng(int(seed))
    noise = rng.multivariate_normal(
        np.zeros(n_channels), VAR3_SIGMA, size=n_total
    )
    x = np.zeros((n_total, n_channels), dtype=np.float64)
    for t in range(ar_order, n_total):
        state = noise[t].copy()
        for lag in range(1, ar_order + 1):
            state = state + VAR3_COEFFS[lag - 1] @ x[t - lag]
        x[t] = state
    return x[VAR3_BURN_IN:]


def _physical_periodogram_from_pipeline_raw(
    periodogram: np.ndarray | None,
    *,
    channel_stds: np.ndarray | None,
    scaling_factor: float | None,
) -> np.ndarray | None:
    """Undo the pipeline raw-periodogram scale and restore channel units."""
    if periodogram is None or channel_stds is None:
        return periodogram
    scale = float(1.0 if scaling_factor is None else scaling_factor)
    if not np.isfinite(scale) or scale <= 0.0:
        return periodogram
    stds = np.asarray(channel_stds, dtype=np.float64)
    if stds.ndim != 1 or periodogram.shape[-1] != stds.size:
        return periodogram
    factor = np.outer(stds, stds) / scale
    return (
        np.asarray(periodogram, dtype=np.complex128)
        * factor[None, :, :]
    )


def _legacy_var3_periodogram_to_physical(
    periodogram: np.ndarray | None,
    metrics: dict[str, float],
) -> np.ndarray | None:
    """Correct old VAR3 summaries that saved standardized raw periodograms."""
    if periodogram is None:
        return None
    if "seed" not in metrics or "N" not in metrics:
        return periodogram
    data = _simulate_var3_observed_data(
        int(metrics["N"]),
        int(metrics["seed"]),
    )
    return _physical_periodogram_from_pipeline_raw(
        periodogram,
        channel_stds=np.std(data, axis=0),
        scaling_factor=float(np.std(data) ** 2.0),
    )


def _nearest_percentile(
    values: np.ndarray, percentiles: np.ndarray, q: float
) -> np.ndarray:
    idx = int(np.argmin(np.abs(percentiles - q)))
    return np.asarray(values[idx], dtype=np.float64)


def _resolve_default_idatas(repo_root: Path) -> list[Path]:
    data_dir = repo_root / "docs/manuscript/data"

    cg_off = data_dir / "var3_K10_cgOFF" / "posterior_ci_summary.npz"
    cg_on = data_dir / "var3_K10_cgNH4" / "posterior_ci_summary.npz"
    if cg_off.exists() and cg_on.exists():
        return [cg_off, cg_on]

    candidates_overlay = sorted(data_dir.glob("var3_*/posterior_vi_overlay_summary.npz"))
    if candidates_overlay:
        return [candidates_overlay[0]]

    candidates_off = sorted(data_dir.glob("var3_*cgOFF/posterior_ci_summary.npz"))
    candidates_on = sorted(data_dir.glob("var3_*cgNH*/posterior_ci_summary.npz"))
    if candidates_off and candidates_on:
        return [candidates_off[0], candidates_on[0]]

    candidates_any = sorted(data_dir.glob("var3_*/posterior_ci_summary.npz"))
    if candidates_any:
        return [candidates_any[0]]

    raise FileNotFoundError(
        "Could not find posterior_ci_summary.npz under docs/manuscript/data/var3_*/."
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
        default="var3_simulation_idata_overlay.pdf",
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
                "nuts_real_q05",
                "nuts_real_q50",
                "nuts_real_q95",
                "nuts_imag_q05",
                "nuts_imag_q50",
                "nuts_imag_q95",
                "vi_real_q05",
                "vi_real_q50",
                "vi_real_q95",
                "vi_imag_q05",
                "vi_imag_q50",
                "vi_imag_q95",
            )
        ):
            periodogram = None
            periodogram_physical_scale = False
            if "periodogram_real" in data and "periodogram_imag" in data:
                periodogram = np.asarray(
                    data["periodogram_real"], dtype=np.float64
                ) + 1j * np.asarray(data["periodogram_imag"], dtype=np.float64)
                if "periodogram_physical_scale" in data:
                    periodogram_physical_scale = bool(
                        np.asarray(data["periodogram_physical_scale"]).item()
                    )
            truth = None
            if "truth_real" in data and "truth_imag" in data:
                truth = np.asarray(
                    data["truth_real"], dtype=np.float64
                ) + 1j * np.asarray(data["truth_imag"], dtype=np.float64)

            metrics: dict[str, float] = {}
            metrics_path = npz_path.parent / "metrics_summary.json"
            if metrics_path.exists():
                with open(metrics_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                for group_name in ("nuts", "vi"):
                    group = loaded.get(group_name, {})
                    for k, v in group.items():
                        if isinstance(v, (int, float)):
                            metrics[f"{group_name}_{k}"] = float(v)
                for k in ("seed", "N"):
                    v = loaded.get(k)
                    if isinstance(v, (int, float)):
                        metrics[k] = float(v)

            if not periodogram_physical_scale:
                periodogram = _legacy_var3_periodogram_to_physical(
                    periodogram, metrics
                )

            return {
                "kind": "vi_vs_nuts_overlay",
                "freq": freq,
                "nuts_q05_real": np.asarray(
                    data["nuts_real_q05"], dtype=np.float64
                ),
                "nuts_q50_real": np.asarray(
                    data["nuts_real_q50"], dtype=np.float64
                ),
                "nuts_q95_real": np.asarray(
                    data["nuts_real_q95"], dtype=np.float64
                ),
                "nuts_q05_imag": np.asarray(
                    data["nuts_imag_q05"], dtype=np.float64
                ),
                "nuts_q50_imag": np.asarray(
                    data["nuts_imag_q50"], dtype=np.float64
                ),
                "nuts_q95_imag": np.asarray(
                    data["nuts_imag_q95"], dtype=np.float64
                ),
                "vi_q05_real": np.asarray(
                    data["vi_real_q05"], dtype=np.float64
                ),
                "vi_q50_real": np.asarray(
                    data["vi_real_q50"], dtype=np.float64
                ),
                "vi_q95_real": np.asarray(
                    data["vi_real_q95"], dtype=np.float64
                ),
                "vi_q05_imag": np.asarray(
                    data["vi_imag_q05"], dtype=np.float64
                ),
                "vi_q50_imag": np.asarray(
                    data["vi_imag_q50"], dtype=np.float64
                ),
                "vi_q95_imag": np.asarray(
                    data["vi_imag_q95"], dtype=np.float64
                ),
                "periodogram": periodogram,
                "truth": truth,
                "metrics": metrics,
            }

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
            periodogram = np.asarray(
                data["periodogram_real"], dtype=np.float64
            ) + 1j * np.asarray(data["periodogram_imag"], dtype=np.float64)
        truth = None
        if "truth_real" in data and "truth_imag" in data:
            truth = np.asarray(
                data["truth_real"], dtype=np.float64
            ) + 1j * np.asarray(data["truth_imag"], dtype=np.float64)

    metrics: dict[str, float] = {}
    metrics_path = npz_path.parent / "metrics_summary.json"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        for k, v in loaded.items():
            if isinstance(v, (int, float)):
                metrics[k] = float(v)

    return {
        "kind": "posterior_summary",
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

    import arviz as az  # lazy: only required for the .nc loader

    idata = az.from_netcdf(idata_path)
    if not hasattr(idata, "posterior_psd"):
        raise ValueError(f"{idata_path} has no posterior_psd group.")
    if "psd_matrix_real" not in idata.posterior_psd:
        raise ValueError(
            f"{idata_path} posterior_psd has no psd_matrix_real variable."
        )

    psd_group = idata.posterior_psd
    freq = np.asarray(psd_group.coords["freq"].values, dtype=np.float64)
    percentiles = np.asarray(
        psd_group.coords["percentile"].values, dtype=np.float64
    )
    psd_real = np.asarray(
        psd_group["psd_matrix_real"].values, dtype=np.float64
    )
    psd_imag = np.asarray(
        psd_group["psd_matrix_imag"].values, dtype=np.float64
    )

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
        "kind": "posterior_summary",
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

    if (
        hasattr(idata, "observed_data")
        and "periodogram" in idata.observed_data
    ):
        periodogram = np.asarray(
            idata.observed_data["periodogram"].values
        )
        attrs = getattr(idata, "attrs", {}) or {}
        summary["periodogram"] = _physical_periodogram_from_pipeline_raw(
            periodogram,
            channel_stds=attrs.get("channel_stds"),
            scaling_factor=attrs.get("scaling_factor"),
        )

    return summary


def _plot_vi_vs_nuts_overlay(
    summary: dict, output_path: Path, xmax: float
) -> None:
    # Match the LISA-plot style for visual consistency across the manuscript.
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    freq = np.asarray(summary["freq"], dtype=np.float64)
    x_mask = (freq >= float(np.min(freq))) & (freq <= float(xmax))
    if not np.any(x_mask):
        x_mask = np.ones_like(freq, dtype=bool)

    freq_plot = freq[x_mask]
    truth_plot = (
        None
        if summary.get("truth") is None
        else np.asarray(summary["truth"])[x_mask]
    )
    periodogram_plot = (
        None
        if summary.get("periodogram") is None
        else np.asarray(summary["periodogram"])[x_mask]
    )

    n_channels = summary["nuts_q50_real"].shape[1]
    fig, axes = plt.subplots(
        n_channels,
        n_channels,
        figsize=(n_channels * 3.27, n_channels * 2.8),
        sharex=True,
        constrained_layout=False,
    )
    if n_channels == 1:
        axes = np.array([[axes]])

    empirical_kw = {
        "color": "0.65",
        "linewidth": 0.8,
        "alpha": 0.6,
        "linestyle": (0, (3, 2)),
        "zorder": 1,
    }
    nuts_fill_kw = {"color": "tab:blue", "alpha": 0.6, "zorder": 3}
    vi_fill_kw = {"color": "tab:orange", "alpha": 0.5, "zorder": 4, "linewidth": 0.0}
    vi_edge_kw = {"color": "tab:orange", "linewidth": 0, "alpha": 0.9, "zorder": 5}
    truth_kw = {
        "color": "black",
        "linewidth": 1.1,
        "linestyle": ":",
        "zorder": 7,
    }

    for i in range(n_channels):
        for j in range(n_channels):
            ax = axes[i, j]

            if i == j:
                nuts_low = summary["nuts_q05_real"][x_mask, i, j]
                nuts_high = summary["nuts_q95_real"][x_mask, i, j]
                vi_low = summary["vi_q05_real"][x_mask, i, j]
                vi_high = summary["vi_q95_real"][x_mask, i, j]
                if periodogram_plot is not None:
                    ax.plot(
                        freq_plot,
                        np.maximum(np.real(periodogram_plot[:, i, j]), EPS),
                        label="Periodogram" if (i, j) == (0, 0) else None,
                        **empirical_kw,
                    )
                ax.fill_between(
                    freq_plot,
                    np.maximum(nuts_low, EPS),
                    np.maximum(nuts_high, EPS),
                    label="NUTS 90% CI" if (i, j) == (0, 0) else None,
                    **nuts_fill_kw,
                )
                ax.fill_between(
                    freq_plot,
                    np.maximum(vi_low, EPS),
                    np.maximum(vi_high, EPS),
                    label="VI 90% CI" if (i, j) == (0, 0) else None,
                    **vi_fill_kw,
                )
                # ax.plot(freq_plot, np.maximum(vi_low, EPS), **vi_edge_kw)
                # ax.plot(freq_plot, np.maximum(vi_high, EPS), **vi_edge_kw)
                if truth_plot is not None:
                    ax.plot(
                        freq_plot,
                        np.maximum(np.real(truth_plot[:, i, j]), EPS),
                        label="Truth" if (i, j) == (0, 0) else None,
                        **truth_kw,
                    )
                ax.set_yscale("log")
                ax.yaxis.set_major_locator(
                    LogLocator(base=10.0, subs=(1.0, 2.0, 5.0))
                )
                ax.yaxis.set_major_formatter(FuncFormatter(_plain_log_tick))
                ax.yaxis.set_minor_formatter(NullFormatter())
            elif i < j:
                nuts_low = summary["nuts_q05_real"][x_mask, i, j]
                nuts_high = summary["nuts_q95_real"][x_mask, i, j]
                vi_low = summary["vi_q05_real"][x_mask, i, j]
                vi_high = summary["vi_q95_real"][x_mask, i, j]
                if periodogram_plot is not None:
                    ax.plot(
                        freq_plot,
                        np.real(periodogram_plot[:, i, j]),
                        **empirical_kw,
                    )
                ax.fill_between(freq_plot, nuts_low, nuts_high, **nuts_fill_kw)
                ax.fill_between(freq_plot, vi_low, vi_high, **vi_fill_kw)
                # ax.plot(freq_plot, vi_low, **vi_edge_kw)
                # ax.plot(freq_plot, vi_high, **vi_edge_kw)
                if truth_plot is not None:
                    ax.plot(
                        freq_plot, np.real(truth_plot[:, i, j]), **truth_kw
                    )
            else:
                nuts_low = summary["nuts_q05_imag"][x_mask, i, j]
                nuts_high = summary["nuts_q95_imag"][x_mask, i, j]
                vi_low = summary["vi_q05_imag"][x_mask, i, j]
                vi_high = summary["vi_q95_imag"][x_mask, i, j]
                if periodogram_plot is not None:
                    ax.plot(
                        freq_plot,
                        np.imag(periodogram_plot[:, i, j]),
                        **empirical_kw,
                    )
                ax.fill_between(freq_plot, nuts_low, nuts_high, **nuts_fill_kw)
                ax.fill_between(freq_plot, vi_low, vi_high, **vi_fill_kw)
                # ax.plot(freq_plot, vi_low, **vi_edge_kw)
                # ax.plot(freq_plot, vi_high, **vi_edge_kw)
                if truth_plot is not None:
                    ax.plot(
                        freq_plot, np.imag(truth_plot[:, i, j]), **truth_kw
                    )

            if i == n_channels - 1:
                ax.set_xlabel("Frequency [Hz]")
            else:
                ax.tick_params(labelbottom=False)
            ax.set_xlim(
                float(freq_plot[0]), min(float(xmax), float(freq_plot[-1]))
            )
            ax.grid(True, which="major", ls=":", alpha=0.35)
            ax.grid(True, which="minor", ls=":", alpha=0.18)

            # Math-style panel label, matching the LISA triangle plot.
            if i == j:
                panel_label = rf"$S_{{{i + 1}{j + 1}}}$"
            elif i < j:
                panel_label = rf"$\Re\{{S_{{{i + 1}{j + 1}}}\}}$"
            else:
                panel_label = rf"$\Im\{{S_{{{i + 1}{j + 1}}}\}}$"
            ax.text(
                0.04,
                0.93,
                panel_label,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=15,
                zorder=10,
            )

            # Per-panel y-axis labels on the leftmost column only.
            if j == 0:
                if i == j:
                    ax.set_ylabel(r"PSD [1/Hz]")
                else:
                    ax.set_ylabel(r"Cross-spectrum")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        # Figure-level legend in the upper-right corner outside the axes.
        fig.legend(
            handles,
            labels,
            loc="upper right",
            bbox_to_anchor=(0.975, 0.975),
            frameon=False,
            ncol=1,
            handlelength=2.2,
            handletextpad=0.6,
            columnspacing=0.9,
            fontsize=10,
        )

    fig.tight_layout(h_pad=0.55, w_pad=0.65, rect=(0.0, 0.0, 1.0, 1.0))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


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

    if (
        len(summaries) == 1
        and summaries[0].get("kind") == "vi_vs_nuts_overlay"
    ):
        _plot_vi_vs_nuts_overlay(
            summaries[0],
            output_path=output_path,
            xmax=float(args.xmax),
        )
        print("Loaded input file:")
        print(f"  - {idata_paths[0]}")
        print("")
        print(f"Saved VI vs NUTS overlay to {output_path}")
        return 0

    if args.labels is None:
        if len(idata_paths) == 2:
            labels = ["No coarse-grain", "Coarse-grain"]
        else:
            labels = [p.parent.name for p in idata_paths]
    else:
        labels = list(args.labels)
        if len(labels) != len(idata_paths):
            raise ValueError(
                "Number of --labels must match number of --idata paths."
            )

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    fill_alphas = [0.4, 0.4, 0.4, 0.4]
    line_widths = [1.8, 1.8, 1.5, 1.5]

    # Use first dataset for channel dimensions.
    first = summaries[0]
    n_channels = first["q50_real"].shape[1]
    periodogram = next(
        (s["periodogram"] for s in summaries if s["periodogram"] is not None),
        None,
    )

    truth_summary = next(
        (s for s in summaries if s.get("truth") is not None), None
    )
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
        xmax = (
            float(args.xmax) if args.xmax is not None else max_freq_available
        )
        xmax = min(xmax, max_freq_available)
        freq_dense = np.linspace(0.0, xmax, 1200)
        freq_dense = freq_dense[freq_dense > 0.0]
        true_psd_dense = _calculate_true_var_psd_hz(
            freq_dense, var_coeffs, sigma, fs=1.0
        )
    else:
        true_psd_dense = None
        freq_dense = None

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    fig, axes = plt.subplots(
        n_channels,
        n_channels,
        figsize=(n_channels * 3.27, n_channels * 2.8),
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
                    re_obs_candidates.append(
                        np.real(periodogram[:, i, j])[x_mask0]
                    )
                else:
                    im_obs_candidates.append(
                        np.imag(periodogram[:, i, j])[x_mask0]
                    )

    if true_psd_dense is not None:
        truth_mask = (freq_dense >= x_min) & (freq_dense <= x_max)
        for i in range(n_channels):
            for j in range(n_channels):
                if i <= j:
                    all_re_candidates.append(
                        np.real(true_psd_dense[:, i, j])[truth_mask]
                    )
                else:
                    all_im_candidates.append(
                        np.imag(true_psd_dense[:, i, j])[truth_mask]
                    )

    def _global_limits(
        candidates: list[np.ndarray], symmetric: bool
    ) -> tuple[float, float]:
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
                    color="0.65",
                    lw=0.8,
                    alpha=0.6,
                    ls=(0, (3, 2)),
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
                # ax.plot(
                #     freq[idx],
                #     median[idx],
                #     color=colors[k % len(colors)],
                #     lw=line_widths[k % len(line_widths)],
                #     zorder=6 + k,
                #     alpha=0.95,
                #     label=f"{label} median" if (i == 0 and j == 0) else None,
                # )

            ax.set_xlim(x_min, x_max)
            if i == j:
                # Log-y for diagonal PSDs; clamp floor to a small fraction
                # of the running max so the dynamic range is preserved.
                ax.set_yscale("log")
                ax.yaxis.set_major_locator(
                    LogLocator(base=10.0, subs=(1.0, 2.0, 5.0))
                )
                ax.yaxis.set_major_formatter(FuncFormatter(_plain_log_tick))
                ax.yaxis.set_minor_formatter(NullFormatter())
            elif i < j:
                # Real cross-spectrum: data-driven symmetric-ish ylim.
                ax.set_ylim(*re_ylim)
            else:
                # Imag cross-spectrum: symmetric-ish ylim around zero.
                ax.set_ylim(*im_ylim)
                ax.axhline(0.0, color="0.5", lw=0.6, alpha=0.6, zorder=2)

            ax.grid(True, which="major", ls=":", alpha=0.35)
            ax.grid(True, which="minor", ls=":", alpha=0.18)
            if i == n_channels - 1:
                ax.set_xlabel("Frequency [Hz]")
            else:
                ax.tick_params(labelbottom=False)
            if j == 0:
                if i == j:
                    ax.set_ylabel(r"PSD [1/Hz]")
                else:
                    ax.set_ylabel(r"Cross-spectrum")

            # Math-style panel label
            if i == j:
                panel_label = rf"$S_{{{i + 1}{j + 1}}}$"
            elif i < j:
                panel_label = rf"$\Re\{{S_{{{i + 1}{j + 1}}}\}}$"
            else:
                panel_label = rf"$\Im\{{S_{{{i + 1}{j + 1}}}\}}$"
            ax.text(
                0.04,
                0.93,
                panel_label,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=15,
                zorder=10,
            )

    handles, labels_out = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels_out,
            loc="upper right",
            bbox_to_anchor=(0.975, 0.975),
            frameon=False,
            ncol=1,
            handlelength=2.2,
            handletextpad=0.6,
            columnspacing=0.9,
            fontsize=10,
        )
    fig.tight_layout(h_pad=0.55, w_pad=0.65, rect=(0.0, 0.0, 1.0, 1.0))

    # output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")

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
