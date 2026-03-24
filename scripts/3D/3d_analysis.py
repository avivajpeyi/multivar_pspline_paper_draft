"""Fixed 3D VAR(2) analysis for manuscript figures.

This script runs a single multivariate PSD analysis with:
- large N
- Nb = 4
- coarse-graining Nh = 4

It saves:
- compact posterior summaries for NUTS and VI,
- a metrics JSON,
- a PSD overlay figure comparing VI and NUTS.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

import jax
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import numpy as np

from log_psplines.logger import logger, set_level
from log_psplines.mcmc import MultivariateTimeseries, run_mcmc

jax.config.update("jax_enable_x64", True)
set_level("INFO")

HERE = Path(__file__).resolve().parent
OUTDIR = HERE / "out_var3" / "large_N16384_Nb4_cg4"

SEED = 0
FS = 1.0
N = 16 * 1024
NB = 4
COARSE_NH = 4
STAGE1_VI_NH = 8
K = 50
BURN_IN = 512

TARGET_ACCEPT_PROB = 0.95
MAX_TREE_DEPTH = 14
INIT_FROM_VI = True
VI_STEPS = 300_000
VI_GUIDE = "lowrank:16"
VI_LR = 5e-4
VI_PSD_MAX_DRAWS = 256
N_SAMPLES = 4000
N_WARMUP = 4000
NUM_CHAINS = 4
ALPHA_DELTA = 1.0
BETA_DELTA = 1.0
KNOT_METHOD = "density"

XMAX = 0.5
EPS = 1e-12

A1 = np.diag([0.4, 0.3, 0.2])
A2 = np.array(
    [
        [-0.2, 0.5, 0.0],
        [0.4, -0.1, 0.0],
        [0.0, 0.0, -0.1],
    ],
    dtype=np.float64,
)
VAR_COEFFS = np.array([A1, A2], dtype=np.float64)

SIGMA = np.array(
    [
        [0.25, 0.0, 0.08],
        [0.0, 0.25, 0.08],
        [0.08, 0.08, 0.25],
    ],
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


def _log_var_coefficients() -> None:
    logger.info("Using VAR coefficients:")
    for lag, coeff in enumerate(VAR_COEFFS, start=1):
        logger.info(f"A{lag} =\n{np.array2string(coeff, precision=4)}")


def _companion_spectral_radius(var_coeffs: np.ndarray) -> float:
    """Return companion-matrix spectral radius for VAR(p) coefficients."""
    ar_order, n_channels, _ = var_coeffs.shape
    companion = np.zeros(
        (n_channels * ar_order, n_channels * ar_order),
        dtype=np.float64,
    )
    companion[:n_channels, : (n_channels * ar_order)] = np.hstack(var_coeffs)
    if ar_order > 1:
        companion[n_channels:, :-n_channels] = np.eye(
            n_channels * (ar_order - 1),
            dtype=np.float64,
        )
    eigvals = np.linalg.eigvals(companion)
    return float(np.max(np.abs(eigvals))) if eigvals.size else 0.0


def _simulate_var_process(
    n_samples: int,
    var_coeffs: np.ndarray,
    sigma: np.ndarray,
    seed: int,
    *,
    fs: float,
    burn_in: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate VAR(p): x_t = sum_k A_k x_{t-k} + eps_t."""
    ar_order, n_channels, _ = var_coeffs.shape
    n_total = int(n_samples) + int(burn_in)
    rng = np.random.default_rng(int(seed))
    noise = rng.multivariate_normal(np.zeros(n_channels), sigma, size=n_total)
    x = np.zeros((n_total, n_channels), dtype=np.float64)

    for t_idx in range(ar_order, n_total):
        state = noise[t_idx].copy()
        for lag in range(1, ar_order + 1):
            state = state + var_coeffs[lag - 1] @ x[t_idx - lag]
        x[t_idx] = state

    x = x[burn_in:]
    t = np.arange(x.shape[0], dtype=np.float64) / float(fs)
    return t, x


def _calculate_true_var_psd_hz(
    freqs_hz: np.ndarray,
    var_coeffs: np.ndarray,
    sigma: np.ndarray,
    *,
    fs: float,
) -> np.ndarray:
    """Compute one-sided theoretical PSD matrix S(f) on a Hz frequency grid."""
    freqs_hz = np.asarray(freqs_hz, dtype=np.float64)
    ar_order, n_channels, _ = var_coeffs.shape
    omega = 2.0 * np.pi * freqs_hz / float(fs)
    psd = np.empty(
        (freqs_hz.shape[0], n_channels, n_channels),
        dtype=np.complex128,
    )
    ident = np.eye(n_channels, dtype=np.complex128)

    for idx, w in enumerate(omega):
        a_f = ident.copy()
        for lag in range(1, ar_order + 1):
            a_f = a_f - var_coeffs[lag - 1] * np.exp(-1j * w * lag)
        h_f = np.linalg.inv(a_f)
        psd[idx] = (2.0 / float(fs)) * (h_f @ sigma @ h_f.conj().T)

    if freqs_hz.size and np.isclose(freqs_hz[-1], fs / 2.0):
        psd[-1] = 0.5 * psd[-1]

    psd = 0.5 * (psd + np.swapaxes(psd.conj(), -1, -2))
    return np.where(np.abs(psd) < EPS, EPS, psd)


def _extract_percentile_slice(
    values: np.ndarray,
    percentiles: np.ndarray,
    target: float,
) -> np.ndarray:
    idx = int(np.argmin(np.abs(percentiles - target)))
    return np.asarray(values[idx], dtype=np.float64)


def _extract_group_quantiles(psd_group) -> dict[str, np.ndarray]:
    """Return freq and 5/50/95 quantiles from posterior_psd-like groups."""
    if psd_group is None:
        raise ValueError("PSD group is missing.")
    if "psd_matrix_real" not in psd_group or "psd_matrix_imag" not in psd_group:
        raise ValueError(
            "PSD group must contain psd_matrix_real and psd_matrix_imag."
        )

    freq = np.asarray(psd_group.coords["freq"].values, dtype=np.float64)
    real = np.asarray(psd_group["psd_matrix_real"].values, dtype=np.float64)
    imag = np.asarray(psd_group["psd_matrix_imag"].values, dtype=np.float64)
    percentiles = np.asarray(
        psd_group["psd_matrix_real"].coords["percentile"].values,
        dtype=np.float64,
    )

    return {
        "freq": freq,
        "q05_real": _extract_percentile_slice(real, percentiles, 5.0),
        "q50_real": _extract_percentile_slice(real, percentiles, 50.0),
        "q95_real": _extract_percentile_slice(real, percentiles, 95.0),
        "q05_imag": _extract_percentile_slice(imag, percentiles, 5.0),
        "q50_imag": _extract_percentile_slice(imag, percentiles, 50.0),
        "q95_imag": _extract_percentile_slice(imag, percentiles, 95.0),
    }


def _interp_complex_matrix(
    target_freq: np.ndarray,
    source_freq: np.ndarray,
    source_values: np.ndarray,
) -> np.ndarray:
    """Interpolate complex (F, P, P) arrays onto a target frequency grid."""
    target_freq = np.asarray(target_freq, dtype=np.float64)
    source_freq = np.asarray(source_freq, dtype=np.float64)
    source_values = np.asarray(source_values, dtype=np.complex128)

    if source_values.shape[0] != source_freq.size:
        raise ValueError("source_values first dimension must match source_freq.")

    _, n_channels, _ = source_values.shape
    out = np.empty((target_freq.size, n_channels, n_channels), dtype=np.complex128)
    for i in range(n_channels):
        for j in range(n_channels):
            out[:, i, j] = np.interp(
                target_freq,
                source_freq,
                np.real(source_values[:, i, j]),
            ) + 1j * np.interp(
                target_freq,
                source_freq,
                np.imag(source_values[:, i, j]),
            )
    return out


def _maybe_interp_summary(
    summary: dict[str, np.ndarray],
    target_freq: np.ndarray,
) -> dict[str, np.ndarray]:
    """Interpolate summary quantiles onto target_freq when needed."""
    source_freq = np.asarray(summary["freq"], dtype=np.float64)
    target_freq = np.asarray(target_freq, dtype=np.float64)
    if np.array_equal(source_freq, target_freq):
        return summary

    def _interp_real(array: np.ndarray) -> np.ndarray:
        return np.real(
            _interp_complex_matrix(
                target_freq,
                source_freq,
                array.astype(np.complex128),
            )
        )

    return {
        "freq": target_freq,
        "q05_real": _interp_real(summary["q05_real"]),
        "q50_real": _interp_real(summary["q50_real"]),
        "q95_real": _interp_real(summary["q95_real"]),
        "q05_imag": _interp_real(summary["q05_imag"]),
        "q50_imag": _interp_real(summary["q50_imag"]),
        "q95_imag": _interp_real(summary["q95_imag"]),
    }


def _compute_ci_width_metrics_from_group(psd_group) -> dict[str, float]:
    """Compute CI-width summaries from a posterior_psd-like group."""
    if psd_group is None or "psd_matrix_real" not in psd_group:
        return {}

    real = np.asarray(psd_group["psd_matrix_real"].values, dtype=np.float64)
    percentiles = np.asarray(
        psd_group["psd_matrix_real"].coords.get(
            "percentile",
            np.arange(real.shape[0], dtype=float),
        ),
        dtype=np.float64,
    )
    if real.shape[0] < 2:
        return {}

    q05 = _extract_percentile_slice(real, percentiles, 5.0)
    q95 = _extract_percentile_slice(real, percentiles, 95.0)
    width_psd = np.maximum(q95 - q05, 0.0)

    p = width_psd.shape[1]
    diag_idx = np.arange(p)
    offdiag_mask = ~np.eye(p, dtype=bool)
    diag_width = width_psd[:, diag_idx, diag_idx]
    offdiag_width = width_psd[:, offdiag_mask]

    metrics = {
        "ciw_psd_diag_mean": float(np.mean(diag_width)),
        "ciw_psd_diag_median": float(np.median(diag_width)),
        "ciw_psd_diag_max": float(np.max(diag_width)),
        "ciw_psd_offdiag_mean": float(np.mean(offdiag_width)),
        "ciw_psd_offdiag_median": float(np.median(offdiag_width)),
        "ciw_psd_offdiag_max": float(np.max(offdiag_width)),
    }

    if "coherence" in psd_group:
        coherence = np.asarray(psd_group["coherence"].values, dtype=np.float64)
        coh_percentiles = np.asarray(
            psd_group["coherence"].coords.get(
                "percentile",
                np.arange(coherence.shape[0], dtype=float),
            ),
            dtype=np.float64,
        )
        if coherence.shape[0] >= 2:
            coh_q05 = _extract_percentile_slice(coherence, coh_percentiles, 5.0)
            coh_q95 = _extract_percentile_slice(coherence, coh_percentiles, 95.0)
            coh_width = np.maximum(coh_q95 - coh_q05, 0.0)
            coh_offdiag = coh_width[:, offdiag_mask]
            metrics["ciw_coh_offdiag_mean"] = float(np.mean(coh_offdiag))
            metrics["ciw_coh_offdiag_median"] = float(np.median(coh_offdiag))
            metrics["ciw_coh_offdiag_max"] = float(np.max(coh_offdiag))

    return metrics


def _scalar_attr(attrs: dict, *keys: str) -> float | None:
    for key in keys:
        if key not in attrs:
            continue
        try:
            value = float(attrs[key])
        except Exception:
            continue
        if np.isfinite(value):
            return value
    return None


def _build_metrics_summary(idata) -> dict[str, object]:
    attrs = dict(getattr(idata, "attrs", {}) or {})
    summary: dict[str, object] = {
        "config": {
            "seed": SEED,
            "N": N,
            "Nb": NB,
            "coarse_Nh": COARSE_NH,
            "stage1_vi_Nh": STAGE1_VI_NH,
            "K": K,
            "n_samples": N_SAMPLES,
            "n_warmup": N_WARMUP,
            "num_chains": NUM_CHAINS,
            "vi_steps": VI_STEPS,
            "vi_guide": VI_GUIDE,
        }
    }

    posterior_group = getattr(idata, "posterior_psd", None)
    vi_group = getattr(idata, "vi_posterior_psd", None)

    summary["nuts"] = {
        "lnz": _scalar_attr(attrs, "lnz"),
        "lnz_err": _scalar_attr(attrs, "lnz_err"),
        "riae": _scalar_attr(attrs, "riae_matrix", "riae"),
        "coverage": _scalar_attr(attrs, "coverage"),
        "runtime": _scalar_attr(attrs, "runtime"),
        "coarse_vi_attempted": _scalar_attr(attrs, "coarse_vi_attempted"),
        "coarse_vi_success": _scalar_attr(attrs, "coarse_vi_success"),
        "coarse_vi_nfreq": _scalar_attr(attrs, "coarse_vi_nfreq"),
        "coarse_vi_full_nfreq": _scalar_attr(attrs, "coarse_vi_full_nfreq"),
        "coarse_vi_target_nfreq": _scalar_attr(attrs, "coarse_vi_target_nfreq"),
        **_compute_ci_width_metrics_from_group(posterior_group),
    }
    summary["vi"] = {
        "riae": _scalar_attr(attrs, "vi_riae_vs_truth", "vi_riae"),
        "coverage": _scalar_attr(attrs, "vi_coverage_vs_truth", "vi_coverage"),
        "ci_width": _scalar_attr(
            attrs,
            "vi_ci_width_vs_truth",
            "vi_ci_width",
            "vi_ci_width_diag_mean",
        ),
        **_compute_ci_width_metrics_from_group(vi_group),
    }
    return summary


def _save_compact_summary(
    outdir: Path,
    *,
    nuts_summary: dict[str, np.ndarray],
    vi_summary: dict[str, np.ndarray],
    periodogram: np.ndarray | None,
    truth: np.ndarray,
) -> None:
    payload: dict[str, np.ndarray] = {
        "freq": nuts_summary["freq"],
        "nuts_real_q05": nuts_summary["q05_real"],
        "nuts_real_q50": nuts_summary["q50_real"],
        "nuts_real_q95": nuts_summary["q95_real"],
        "nuts_imag_q05": nuts_summary["q05_imag"],
        "nuts_imag_q50": nuts_summary["q50_imag"],
        "nuts_imag_q95": nuts_summary["q95_imag"],
        "vi_real_q05": vi_summary["q05_real"],
        "vi_real_q50": vi_summary["q50_real"],
        "vi_real_q95": vi_summary["q95_real"],
        "vi_imag_q05": vi_summary["q05_imag"],
        "vi_imag_q50": vi_summary["q50_imag"],
        "vi_imag_q95": vi_summary["q95_imag"],
        "truth_real": np.real(truth),
        "truth_imag": np.imag(truth),
    }
    if periodogram is not None:
        payload["periodogram_real"] = np.real(periodogram)
        payload["periodogram_imag"] = np.imag(periodogram)

    out_path = outdir / "posterior_vi_overlay_summary.npz"
    np.savez_compressed(out_path, **payload)
    logger.info(f"Saved summary arrays to {out_path}")


def _plot_vi_vs_nuts_overlay(
    outdir: Path,
    *,
    nuts_summary: dict[str, np.ndarray],
    vi_summary: dict[str, np.ndarray],
    truth: np.ndarray,
    periodogram: np.ndarray | None,
) -> None:
    freq = nuts_summary["freq"]
    x_mask = (freq >= float(np.min(freq))) & (freq <= XMAX)
    if not np.any(x_mask):
        x_mask = np.ones_like(freq, dtype=bool)

    freq_plot = freq[x_mask]
    truth_plot = truth[x_mask]
    periodogram_plot = None if periodogram is None else periodogram[x_mask]

    n_channels = nuts_summary["q50_real"].shape[1]
    fig, axes = plt.subplots(
        n_channels,
        n_channels,
        figsize=(n_channels * 3.0, n_channels * 3.0),
        sharex=True,
        constrained_layout=False,
    )
    if n_channels == 1:
        axes = np.array([[axes]])

    empirical_kw = {
        "color": "0.75",
        "linewidth": 0.8,
        "alpha": 0.8,
        "zorder": 1,
    }
    nuts_fill_kw = {
        "color": "tab:blue",
        "alpha": 0.20,
        "zorder": 2,
    }
    nuts_line_kw = {
        "color": "tab:blue",
        "linewidth": 1.8,
        "zorder": 4,
    }
    vi_fill_kw = {
        "color": "tab:orange",
        "alpha": 0.18,
        "zorder": 3,
    }
    vi_line_kw = {
        "color": "tab:orange",
        "linewidth": 1.6,
        "linestyle": "--",
        "zorder": 5,
    }
    truth_kw = {
        "color": "black",
        "linewidth": 1.2,
        "linestyle": ":",
        "zorder": 6,
    }

    for i in range(n_channels):
        for j in range(n_channels):
            ax = axes[i, j]
            nuts_mid_re = nuts_summary["q50_real"][x_mask, i, j]
            nuts_low_re = nuts_summary["q05_real"][x_mask, i, j]
            nuts_high_re = nuts_summary["q95_real"][x_mask, i, j]
            vi_mid_re = vi_summary["q50_real"][x_mask, i, j]
            vi_low_re = vi_summary["q05_real"][x_mask, i, j]
            vi_high_re = vi_summary["q95_real"][x_mask, i, j]

            nuts_mid_im = nuts_summary["q50_imag"][x_mask, i, j]
            nuts_low_im = nuts_summary["q05_imag"][x_mask, i, j]
            nuts_high_im = nuts_summary["q95_imag"][x_mask, i, j]
            vi_mid_im = vi_summary["q50_imag"][x_mask, i, j]
            vi_low_im = vi_summary["q05_imag"][x_mask, i, j]
            vi_high_im = vi_summary["q95_imag"][x_mask, i, j]

            if i == j:
                if periodogram_plot is not None:
                    ax.plot(
                        freq_plot,
                        np.maximum(np.real(periodogram_plot[:, i, j]), EPS),
                        label="Periodogram" if (i, j) == (0, 0) else None,
                        **empirical_kw,
                    )
                ax.fill_between(
                    freq_plot,
                    np.maximum(nuts_low_re, EPS),
                    np.maximum(nuts_high_re, EPS),
                    **nuts_fill_kw,
                )
                ax.plot(
                    freq_plot,
                    np.maximum(nuts_mid_re, EPS),
                    label="NUTS median" if (i, j) == (0, 0) else None,
                    **nuts_line_kw,
                )
                ax.fill_between(
                    freq_plot,
                    np.maximum(vi_low_re, EPS),
                    np.maximum(vi_high_re, EPS),
                    **vi_fill_kw,
                )
                ax.plot(
                    freq_plot,
                    np.maximum(vi_mid_re, EPS),
                    label="VI median" if (i, j) == (0, 0) else None,
                    **vi_line_kw,
                )
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
                if periodogram_plot is not None:
                    ax.plot(
                        freq_plot,
                        np.real(periodogram_plot[:, i, j]),
                        **empirical_kw,
                    )
                ax.fill_between(
                    freq_plot,
                    nuts_low_re,
                    nuts_high_re,
                    **nuts_fill_kw,
                )
                ax.plot(
                    freq_plot,
                    nuts_mid_re,
                    **nuts_line_kw,
                )
                ax.fill_between(
                    freq_plot,
                    vi_low_re,
                    vi_high_re,
                    **vi_fill_kw,
                )
                ax.plot(
                    freq_plot,
                    vi_mid_re,
                    **vi_line_kw,
                )
                ax.plot(
                    freq_plot,
                    np.real(truth_plot[:, i, j]),
                    **truth_kw,
                )
            else:
                if periodogram_plot is not None:
                    ax.plot(
                        freq_plot,
                        np.imag(periodogram_plot[:, i, j]),
                        **empirical_kw,
                    )
                ax.fill_between(
                    freq_plot,
                    nuts_low_im,
                    nuts_high_im,
                    **nuts_fill_kw,
                )
                ax.plot(
                    freq_plot,
                    nuts_mid_im,
                    **nuts_line_kw,
                )
                ax.fill_between(
                    freq_plot,
                    vi_low_im,
                    vi_high_im,
                    **vi_fill_kw,
                )
                ax.plot(
                    freq_plot,
                    vi_mid_im,
                    **vi_line_kw,
                )
                ax.plot(
                    freq_plot,
                    np.imag(truth_plot[:, i, j]),
                    **truth_kw,
                )

            if i == n_channels - 1:
                ax.set_xlabel("Frequency (Hz)")
            ax.set_xlim(float(freq_plot[0]), min(XMAX, float(freq_plot[-1])))
            ax.grid(alpha=0.2, linewidth=0.5)
            if i == j:
                panel_label = f"S{i + 1}{j + 1}"
            elif i < j:
                panel_label = f"Re[S{i + 1}{j + 1}]"
            else:
                panel_label = f"Im[S{i + 1}{j + 1}]"
            ax.text(
                0.03,
                0.96,
                panel_label,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=11,
                fontweight="semibold",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.85,
                    "pad": 1.5,
                },
                zorder=10,
            )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        axes[0, 0].legend(
            handles,
            labels,
            loc="upper right",
            fontsize=9,
            frameon=True,
            framealpha=0.9,
        )

    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.07,
        top=0.98,
        wspace=0.18,
        hspace=0.12,
    )
    fig.text(
        0.015,
        0.5,
        "Spectral density [1/Hz]",
        rotation=90,
        va="center",
        ha="center",
        fontsize=12,
    )

    out_path = outdir / "psd_vi_vs_nuts_overlay.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved overlay figure to {out_path}")


def run_analysis() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    _log_var_coefficients()
    spectral_radius = _companion_spectral_radius(VAR_COEFFS)
    logger.info(
        f"Stationarity check (companion spectral radius): {spectral_radius:.6f}"
    )
    if spectral_radius >= 1.0:
        raise ValueError(
            f"Non-stationary VAR coefficients (spectral radius={spectral_radius:.6f})."
        )

    t, data = _simulate_var_process(
        n_samples=N,
        var_coeffs=VAR_COEFFS,
        sigma=SIGMA,
        seed=SEED,
        fs=FS,
        burn_in=BURN_IN,
    )
    if not np.all(np.isfinite(data)):
        raise ValueError("Generated VAR samples contain non-finite values.")

    ts = MultivariateTimeseries(t=t, y=data)
    freq_true_hz = np.fft.rfftfreq(N, d=1.0 / FS)[1:]
    true_psd = _calculate_true_var_psd_hz(
        freq_true_hz,
        VAR_COEFFS,
        SIGMA,
        fs=FS,
    )

    logger.info(
        f"Running fixed analysis with seed={SEED}, N={N}, Nb={NB}, K={K}, coarse={COARSE_NH}"
    )
    idata = run_mcmc(
        data=ts,
        n_knots=K,
        degree=2,
        diffMatrixOrder=2,
        n_samples=N_SAMPLES,
        n_warmup=N_WARMUP,
        num_chains=NUM_CHAINS,
        outdir=str(OUTDIR),
        verbose=True,
        target_accept_prob=TARGET_ACCEPT_PROB,
        max_tree_depth=MAX_TREE_DEPTH,
        init_from_vi=INIT_FROM_VI,
        vi_steps=VI_STEPS,
        vi_guide=VI_GUIDE,
        vi_psd_max_draws=VI_PSD_MAX_DRAWS,
        vi_lr=VI_LR,
        Nb=NB,
        knot_kwargs={"method": KNOT_METHOD},
        coarse_grain_config={
            "enabled": True,
            "Nc": None,
            "Nh": COARSE_NH,
        },
        coarse_grain_config_vi={
            "enabled": True,
            "Nc": None,
            "Nh": STAGE1_VI_NH,
        },
        alpha_delta=ALPHA_DELTA,
        beta_delta=BETA_DELTA,
        compute_coherence_quantiles=True,
        true_psd=(freq_true_hz, true_psd),
    )

    posterior_group = getattr(idata, "posterior_psd", None)
    vi_group = getattr(idata, "vi_posterior_psd", None)
    if posterior_group is None:
        raise ValueError("Expected posterior_psd in inference output.")
    if vi_group is None:
        raise ValueError("Expected vi_posterior_psd in inference output.")

    nuts_summary = _extract_group_quantiles(posterior_group)
    vi_summary = _extract_group_quantiles(vi_group)
    vi_summary = _maybe_interp_summary(vi_summary, nuts_summary["freq"])

    periodogram = None
    if hasattr(idata, "observed_data") and "periodogram" in idata.observed_data:
        periodogram_da = idata.observed_data["periodogram"]
        periodogram = np.asarray(periodogram_da.values)
        if periodogram.shape[0] != nuts_summary["freq"].size:
            periodogram = _interp_complex_matrix(
                nuts_summary["freq"],
                np.asarray(periodogram_da.coords["freq"].values, dtype=np.float64),
                periodogram,
            )

    truth_on_posterior_grid = _interp_complex_matrix(
        nuts_summary["freq"],
        freq_true_hz,
        true_psd,
    )

    _save_compact_summary(
        OUTDIR,
        nuts_summary=nuts_summary,
        vi_summary=vi_summary,
        periodogram=periodogram,
        truth=truth_on_posterior_grid,
    )

    metrics = _build_metrics_summary(idata)
    metrics_path = OUTDIR / "metrics_summary.json"
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    logger.info(f"Saved metrics summary to {metrics_path}")

    _plot_vi_vs_nuts_overlay(
        OUTDIR,
        nuts_summary=nuts_summary,
        vi_summary=vi_summary,
        truth=truth_on_posterior_grid,
        periodogram=periodogram,
    )


def main() -> None:
    run_analysis()


if __name__ == "__main__":
    main()
