#!/usr/bin/env bash
# Regenerate all data-driven manuscript figures from checked-in intermediate data.
# No inference is run.
#
# Figures produced:
#   ../figures/vi_vs_nuts_var3.pdf         (3D VAR(2), Fig. 2)
#   ../figures/triangle_noise4a_eta0p5.pdf (LISA, Fig. 3 left)
#   ../figures/triangle_noise5a_eta0p5.pdf (LISA, Fig. 3 right)
#   ../figures/lisa_eta_sweep.pdf          (LISA eta sweep, Fig. 4)
#
# Fig. 1 (blocked-likelihood schematic) is inline TikZ in main.tex
# and is rendered by build.sh / pdflatex; nothing to do here.
#
# To regenerate the VAR(2) inference data (VI + NUTS, takes ~2 h on OzStar):
#   python scripts/run_var3.py --mode large --seed 0
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="${HERE}/../data"
FIGURES="${HERE}/../figures"
PYTHON="${PYTHON:-python3}"

echo "[1/3] 3D VAR(2) -> figures/vi_vs_nuts_var3.pdf"
# Prefer the VI-vs-NUTS overlay if it has been generated; fall back to the
# coarse-grain comparison that ships with the repo.
OVERLAY="${DATA}/var3_vi_vs_nuts/posterior_vi_overlay_summary.npz"
if [ -f "${OVERLAY}" ]; then
    echo "      using VI-vs-NUTS overlay: ${OVERLAY}"
    "${PYTHON}" "${HERE}/plot_var3.py" \
        --idata "${OVERLAY}" \
        --output "${FIGURES}/vi_vs_nuts_var3.pdf"
else
    echo "      VI-vs-NUTS data not found; using coarse-grain comparison."
    echo "      Run 'python scripts/run_var3.py --mode large' to generate it."
    "${PYTHON}" "${HERE}/plot_var3.py" \
        --idata \
            "${DATA}/var3_K10_cgOFF/posterior_ci_summary.npz" \
            "${DATA}/var3_K10_cgNH4/posterior_ci_summary.npz" \
        --labels "Coarse OFF" "Coarse ON" \
        --output "${FIGURES}/vi_vs_nuts_var3.pdf"
fi

echo "[2/3] LISA triangle plots -> figures/triangle_noise{4a,5a}_eta0p5.pdf"
"${PYTHON}" "${HERE}/plot_lisa_triangle.py" \
    --input "${DATA}/lisa_triangle_eta0p5.h5" \
    --outdir "${FIGURES}"

echo "[3/3] LISA eta sweep -> figures/lisa_eta_sweep.pdf"
"${PYTHON}" "${HERE}/plot_lisa_eta.py"

echo "Done."
