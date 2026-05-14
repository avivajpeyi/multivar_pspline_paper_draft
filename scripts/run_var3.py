"""Run the 3-channel VAR(2) inference and save a VI-vs-NUTS overlay summary.

This script generates synthetic VAR(2) data, runs NUTS with VI initialisation,
then saves a ``posterior_vi_overlay_summary.npz`` under
``docs/manuscript/data/var3_vi_vs_nuts/`` that ``plot_var3.py`` can render
directly as a VI vs NUTS comparison figure.

Typical OzStar invocation (> 30 min for large mode)::

    sbatch run_var3.slurm   # see the SLURM template below

Quick local test (small N, fewer samples)::

    python run_var3.py --mode short --seed 0 --n-samples 200 --n-warmup 200

SLURM template (save as run_var3.slurm next to this script)::

    #!/bin/bash
    #SBATCH --job-name=var3_vi_nuts
    #SBATCH --output=logs/var3_%j.out
    #SBATCH --error=logs/var3_%j.err
    #SBATCH --time=2:00:00
    #SBATCH --mem=16G
    #SBATCH --cpus-per-task=4
    module --force purge
    module load gcc/13.3.0 python/3.12.3
    PYTHON=/fred/oz303/avajpeyi/codes/LogPSplinePSD/.venv/bin/python
    "${PYTHON}" /fred/oz303/avajpeyi/codes/LogPSplinePSD/docs/manuscript/scripts/run_var3.py \\
        --mode large --seed 0 --outdir ../data/var3_vi_vs_nuts
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Literal

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from log_psplines.arviz_utils import (
    get_multivar_posterior_psd_quantiles,
    get_multivar_vi_psd_quantiles,
)
from log_psplines.datatypes import MultivariateTimeseries
from log_psplines.logger import set_level
from log_psplines.mcmc import run_mcmc
from log_psplines.pipeline.config import PipelineConfig
from log_psplines.preprocessing.coarse_grain import CoarseGrainConfig

set_level("INFO")

HERE = Path(__file__).resolve().parent
DEFAULT_OUTDIR = str(HERE.parent / "data" / "var3_vi_vs_nuts")

# VAR(2) model — identical to docs/studies/multivar_psd/3d_study.py
A1 = np.diag([0.4, 0.3, 0.2])
A2 = np.array(
    [[-0.2, 0.5, 0.0], [0.4, -0.1, 0.0], [0.0, 0.0, -0.1]],
    dtype=np.float64,
)
VAR_COEFFS = np.array([A1, A2], dtype=np.float64)
SIGMA = np.array(
    [[0.25, 0.0, 0.08], [0.0, 0.25, 0.08], [0.08, 0.08, 0.25]],
    dtype=np.float64,
)
FS = 1.0
BURN_IN = 512

MODE_CONFIG: dict[str, dict] = {
    "short": {"N": 2048, "Nb": 4, "Nh": None},
    "large": {"N": 16384, "Nb": 4, "Nh": 4},
}


def _simulate_var(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    ar_order = VAR_COEFFS.shape[0]
    n_channels = VAR_COEFFS.shape[1]
    n_total = n + BURN_IN
    rng = np.random.default_rng(seed)
    noise = rng.multivariate_normal(np.zeros(n_channels), SIGMA, size=n_total)
    x = np.zeros((n_total, n_channels), dtype=np.float64)
    for t in range(ar_order, n_total):
        state = noise[t].copy()
        for lag in range(1, ar_order + 1):
            state = state + VAR_COEFFS[lag - 1] @ x[t - lag]
        x[t] = state
    x = x[BURN_IN:]
    t = np.arange(x.shape[0], dtype=np.float64) / FS
    return t, x


def _true_psd(freq: np.ndarray) -> np.ndarray:
    omega = 2.0 * np.pi * freq / FS
    ar_order, p, _ = VAR_COEFFS.shape
    psd = np.empty((freq.size, p, p), dtype=np.complex128)
    ident = np.eye(p, dtype=np.complex128)
    for k, w in enumerate(omega):
        a_f = ident.copy()
        for lag in range(1, ar_order + 1):
            a_f -= VAR_COEFFS[lag - 1] * np.exp(-1j * w * lag)
        h = np.linalg.inv(a_f)
        psd[k] = (2.0 / FS) * (h @ SIGMA @ h.conj().T)
    if freq.size and np.isclose(freq[-1], FS / 2.0):
        psd[-1] *= 0.5
    psd = 0.5 * (psd + np.swapaxes(psd.conj(), -1, -2))
    return psd


def _extract_q(quantiles: dict, pct: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (real, imag) slice nearest to pct from a quantiles dict."""
    percentiles = np.asarray(quantiles["percentile"], dtype=np.float64)
    idx = int(np.argmin(np.abs(percentiles - pct)))
    sd = np.asarray(quantiles["spectral_density"], dtype=np.complex128)
    return sd[idx].real, sd[idx].imag


def _physical_periodogram_from_pipeline_raw(
    periodogram: np.ndarray,
    data: np.ndarray,
) -> np.ndarray:
    """Convert observed_data.periodogram back to original channel units."""
    channel_stds = np.std(data, axis=0)
    scaling_factor = float(np.std(data) ** 2.0)
    scale_matrix = np.outer(channel_stds, channel_stds) / scaling_factor
    return (
        np.asarray(periodogram, dtype=np.complex128)
        * scale_matrix[None, :, :]
    )


def run(
    *,
    seed: int = 0,
    mode: str = "large",
    K: int = 10,
    n_samples: int = 4000,
    n_warmup: int = 4000,
    num_chains: int = 4,
    vi_steps: int = 100_000,
    outdir: str = DEFAULT_OUTDIR,
    chain_method: Literal["parallel", "vectorized", "sequential"] = "parallel",
) -> None:
    cfg = MODE_CONFIG[mode]
    N: int = cfg["N"]
    Nb: int = cfg["Nb"]
    Nh: int | None = cfg["Nh"]

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"[run_var3] mode={mode} N={N} Nb={Nb} Nh={Nh} K={K} seed={seed}")

    t, data = _simulate_var(N, seed)
    ts = MultivariateTimeseries(t=t, y=data)

    freq_true = np.fft.rfftfreq(N, d=1.0 / FS)[1:]
    true_psd = _true_psd(freq_true)

    coarse_grain_config = (
        None if Nh is None else CoarseGrainConfig(enabled=True, Nc=None, Nh=int(Nh))
    )

    config = PipelineConfig(
        n_knots=K,
        degree=2,
        diffMatrixOrder=2,
        n_samples=n_samples,
        n_warmup=n_warmup,
        num_chains=num_chains,
        outdir="out_var",
        verbose=True,
        chain_method=chain_method,
        target_accept_prob=0.95,
        max_tree_depth=14,
        init_from_vi=True,
        only_vi=False,
        vi_steps=vi_steps,
        vi_guide="lowrank:16",
        vi_posterior_draws=256,
        vi_psd_max_draws=256,
        vi_lr=5e-4,
        vi_progress_bar=True,
        Nb=Nb,
        wishart_window=None,
        coarse_grain_config=coarse_grain_config,
        true_psd=(freq_true, true_psd),
        knot_kwargs=dict(method='density')
    )
    idata = run_mcmc(data=ts, config=config)

    print("[run_var3] Extracting NUTS quantiles …")
    nuts_q = get_multivar_posterior_psd_quantiles(idata)
    print("[run_var3] Extracting VI quantiles …")
    vi_q = get_multivar_vi_psd_quantiles(idata)

    freq = np.asarray(nuts_q["freq"], dtype=np.float64)

    nuts_q05_real, nuts_q05_imag = _extract_q(nuts_q, 5.0)
    nuts_q50_real, nuts_q50_imag = _extract_q(nuts_q, 50.0)
    nuts_q95_real, nuts_q95_imag = _extract_q(nuts_q, 95.0)

    vi_q05_real, vi_q05_imag = _extract_q(vi_q, 5.0)
    vi_q50_real, vi_q50_imag = _extract_q(vi_q, 50.0)
    vi_q95_real, vi_q95_imag = _extract_q(vi_q, 95.0)

    # True PSD on the inference frequency grid
    truth = _true_psd(freq)

    # Periodogram from idata if available
    periodogram_real = periodogram_imag = None
    obs = getattr(idata, "observed_data", None)
    if obs is not None and "periodogram" in obs:
        pg = np.asarray(obs["periodogram"].values, dtype=np.complex128)
        pg = _physical_periodogram_from_pipeline_raw(pg, data)
        periodogram_real = pg.real.astype(np.float64)
        periodogram_imag = pg.imag.astype(np.float64)

    npz_path = out_path / "posterior_vi_overlay_summary.npz"
    payload: dict[str, np.ndarray] = dict(
        freq=freq,
        nuts_real_q05=nuts_q05_real,
        nuts_real_q50=nuts_q50_real,
        nuts_real_q95=nuts_q95_real,
        nuts_imag_q05=nuts_q05_imag,
        nuts_imag_q50=nuts_q50_imag,
        nuts_imag_q95=nuts_q95_imag,
        vi_real_q05=vi_q05_real,
        vi_real_q50=vi_q50_real,
        vi_real_q95=vi_q95_real,
        vi_imag_q05=vi_q05_imag,
        vi_imag_q50=vi_q50_imag,
        vi_imag_q95=vi_q95_imag,
        truth_real=truth.real.astype(np.float64),
        truth_imag=truth.imag.astype(np.float64),
    )
    if periodogram_real is not None:
        payload["periodogram_real"] = periodogram_real
        payload["periodogram_imag"] = periodogram_imag
        payload["periodogram_physical_scale"] = np.asarray(True)

    np.savez_compressed(npz_path, **payload)
    print(f"[run_var3] Saved overlay npz → {npz_path}")

    # Compact scalar metrics
    attrs = idata.attrs or {}

    def _fa(key: str) -> float:
        v = attrs.get(key, float("nan"))
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("nan")

    metrics = {
        "seed": seed,
        "mode": mode,
        "N": N,
        "Nb": Nb,
        "Nh": "OFF" if Nh is None else Nh,
        "K": K,
        "nuts": {
            "coverage": _fa("coverage"),
            "riae": _fa("riae_matrix"),
            "lnz": _fa("lnz"),
            "lnz_err": _fa("lnz_err"),
            "ess_median": float(
                np.nanmedian(np.asarray(attrs.get("ess", np.nan), dtype=float))
            ),
            "runtime": _fa("runtime"),
        },
        "vi": {
            "coverage": _fa("vi_coverage"),
            "riae": _fa("vi_riae_matrix"),
        },
    }
    metrics_path = out_path / "metrics_summary.json"
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"[run_var3] Saved metrics    → {metrics_path}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run VAR(2) VI+NUTS inference and save overlay npz for the manuscript."
    )
    p.add_argument("--mode", choices=list(MODE_CONFIG), default="large",
                   help="Data/sampler preset. 'short' for a quick test, 'large' for the paper (default: large).")
    p.add_argument("--seed", type=int, default=0, help="Random seed (default: 0).")
    p.add_argument("--K", type=int, default=10, help="P-spline knots (default: 10).")
    p.add_argument("--n-samples", type=int, default=4000, dest="n_samples",
                   help="Posterior samples per chain (default: 4000).")
    p.add_argument("--n-warmup", type=int, default=4000, dest="n_warmup",
                   help="Warmup samples per chain (default: 4000).")
    p.add_argument("--num-chains", type=int, default=4, dest="num_chains",
                   help="Number of MCMC chains (default: 4).")
    p.add_argument("--vi-steps", type=int, default=100_000, dest="vi_steps",
                   help="VI optimisation steps (default: 100 000).")
    p.add_argument("--chain-method", choices=("parallel", "vectorized", "sequential"),
                   default="parallel", dest="chain_method",
                   help="NumPyro chain execution mode (default: parallel).")
    p.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR,
                   help=f"Output directory (default: {DEFAULT_OUTDIR}).")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        seed=args.seed,
        mode=args.mode,
        K=args.K,
        n_samples=args.n_samples,
        n_warmup=args.n_warmup,
        num_chains=args.num_chains,
        vi_steps=args.vi_steps,
        outdir=args.outdir,
        chain_method=args.chain_method,
    )
