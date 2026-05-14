# LogPSplinePSD manuscript

This folder is self-contained: it holds the LaTeX source, all checked-in
intermediate data, and the plotting scripts needed to regenerate every figure
in `main.tex`. No external data and no inference run is required.

## Layout

```
docs/manuscript/
├── main.tex, custom_commands.tex, definitions_table.tex,
│   blocked_likelihood_tikz.tex, biblio.bib    # LaTeX source
├── build.sh                                   # tectonic main.tex -> build/main.pdf
├── requirements.txt                           # Python deps for the plotting scripts
├── figures/                                   # PDF files loaded by main.tex
└── scripts/
    ├── make_figures.sh                        # one-shot driver for all figures
    ├── 3D/
    │   ├── 3d_plot.py                         # Fig. 2 renderer
    │   ├── run.sh                             # wraps 3d_plot.py with the right args
    │   └── data/                              # checked-in posterior CI summaries (npz)
    └── lisa/
        ├── plot_lisa_triangle_from_h5.py      # Fig. 3 renderer
        ├── plot_eta_sweep.py                  # Fig. 4 renderer
        ├── triangle_plot_data_eta0p5.h5       # Fig. 3 input
        └── eta_sweep_noise4a.csv              # Fig. 4 input
```

## Regenerating the figures

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/make_figures.sh
```

This writes:

| Figure | Output |
| --- | --- |
| Fig. 2 (`vi_vs_nuts_var3`) | `figures/vi_vs_nuts_var3.pdf` |
| Fig. 3 (`triangle_noise{4a,5a}_eta0p5`) | `figures/triangle_noise4a_eta0p5.pdf`, `figures/triangle_noise5a_eta0p5.pdf` |
| Fig. 4 (`lisa_eta_sweep`) | `figures/lisa_eta_sweep.pdf` |

Fig. 1 is inline TikZ (`blocked_likelihood_tikz.tex`) and is rendered by
`build.sh`.

To override the Python interpreter (e.g. point at a project venv):

```bash
PYTHON=/path/to/python bash scripts/make_figures.sh
```

## Building the PDF

```bash
bash build.sh           # tectonic build -> build/main.pdf
bash build.sh clean     # remove LaTeX aux files
```

`build.sh` requires [`tectonic`](https://tectonic-typesetting.github.io/).
