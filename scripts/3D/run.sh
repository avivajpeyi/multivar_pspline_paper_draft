# python 3d_analysis.py 0 large --coarse-grain both
python 3d_plot.py \
  --idata \
    /Users/avi/Documents/projects/LogPSplinePSD/docs/manuscript/scripts/3D/out_var3/seed_0_large_N16384_K10_cgOFF/inference_data.nc \
    /Users/avi/Documents/projects/LogPSplinePSD/docs/manuscript/scripts/3D/out_var3/seed_0_large_N16384_K10_cgNH4/inference_data.nc \
  --labels "No coarse" "Coarse Nh=4" \
  --with-true