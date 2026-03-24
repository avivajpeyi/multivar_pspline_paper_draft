python 3d_analysis.py 0 large --coarse-grain both --coarse-nh 4

python 3d_plot.py \
  --idata \
  out_var3/seed_0_large_N16384_K10_cgOFF/posterior_ci_summary.npz \
  out_var3/seed_0_large_N16384_K10_cgNH4/posterior_ci_summary.npz \
  --labels "Coarse OFF" "Coarse ON"
