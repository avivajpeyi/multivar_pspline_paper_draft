import pandas as pd
import io
  

df = pd.read_csv("var3d_sim_results.csv")

# 1. Clean and Filter
# We filter out the specific 'divergent' row where RIAE mean is massive (> 1.0)
# and remove any rows that might be incomplete.
subset = df[df['riae_matrix_mean'] < 1.0].copy()

# 2. Preparation for sorting and calculation
subset['Nb_sort'] = subset['Nb'].astype(int)
subset['Nh_val'] = subset['Nh'].apply(lambda x: 1 if x == 'OFF' else int(x))

# Sort logically by Nb, then by the Nh value
subset = subset.sort_values(['Nb_sort', 'Nh_val'])

# Calculate Eta: min(1, 2 / (Nb * Nh))
# subset['eta'] = subset.apply(lambda row: min(1.0, 2.0 / (row['Nb_sort'] * row['Nh_val'])), axis=1)

# Formatting Helper
def fmt_pm(mean, std, precision=3, comma=False):
    if comma:
        return f"{mean:,.0f} \\pm {std:.0f}"
    return f"{mean:.{precision}f} \\pm {std:.{precision}f}"

# 3. Build LaTeX
latex_lines = []
latex_lines.append(r"\begin{table*}[htbp]")
latex_lines.append(r"    \centering")
latex_lines.append(r"    \caption{Full grid of 3D VAR(2) simulation results. $N_{\ell}$ indicates the number of frequency points per partition.}")
latex_lines.append(r"    \label{tab:full_grid_no_outliers}")
# Column layout: Nb, Nh, Nell, eta, Coverage, RIAE, ESS, Runtime
latex_lines.append(r"    \begin{NiceTabular}{cccccccc}[vlines-outer, hlines-outer]")
latex_lines.append(r"        \toprule")
latex_lines.append(r"        $N_{b}$ & $N_{h}$ & $N_{\ell}$ & Coverage & RIAE & ESS & Runtime (s) \\")
latex_lines.append(r"        \midrule")

last_nb = None

for _, row in subset.iterrows():
    # Add a midrule between different Nb groups for better scannability
    if last_nb is not None and row['Nb_sort'] != last_nb:
        latex_lines.append(r"        \midrule")
    last_nb = row['Nb_sort']
    
    line = [
        str(int(row['Nb'])),
        str(row['Nh']).lower(),
        str(int(row['Nc'])),
        f"${fmt_pm(row['coverage_mean']-1, row['coverage_std'])}$",
        f"${fmt_pm(row['riae_matrix_mean'], row['riae_matrix_std'])}$",
        f"${fmt_pm(row['ess_median_mean'], row['ess_median_std'], comma=True)}$",
        f"${fmt_pm(row['runtime_mean'], row['runtime_std'], precision=2)}$"
    ]
    latex_lines.append("        " + " & ".join(line) + r" \\")

latex_lines.append(r"        \bottomrule")
latex_lines.append(r"    \end{NiceTabular}")
latex_lines.append(r"\end{table*}")

# Output the LaTeX code as a a var2_3d_sim_results.tex file
latex_code = "\n".join(latex_lines)
with open("var2_3d_sim_results.tex", "w") as f:
    f.write(latex_code) 