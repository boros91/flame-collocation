# flame-collocation

Code and data for the paper

> I. Boros, *Front-pinned spectral collocation for the flame problem and related ignition problems* (2026).

Archived version of this repository (v1.0.0): [doi:10.5281/zenodo.23479185](https://doi.org/10.5281/zenodo.23479185)

The paper studies global Chebyshev collocation for the ignition problems
y' = y^q (1 - y), y(0) = δ, q ≥ 2 (the flame problem for q = 2), rescaled to [-1, 1].
Two ingredients make it work for very small δ: a sinh map built on the exactly known complex
singularities of the solution, and the replacement of the initial condition by an equivalent
interior ("pin") condition at the front. An Arrhenius reaction rate, for which all quantities
are computed numerically, is also treated.

## Using the method

Python (from the repository root):

```
python example.py
```

```python
import sys; sys.path.insert(0, 'method')
from collocation_q import solve, error
sol = solve(1e-10, 352, 2)              # delta, N (polynomial degree), q; pinned form
sol = solve(1e-10, 352, 2, pin=False)   # the same scheme with the initial condition
print(sol['its'], sol['converged'], error(sol))
```

MATLAB (R2016b or later, no toolboxes): `matlab/method/flame_pinned.m`, with
`example_flame.m` and `verify_method.m` (comparison with the Python implementation);
see `matlab/method/README_method.txt`.

## Repository layout

Note the two figure folders: `figures/` contains the plotting scripts; `figs/` contains the reference figure PDFs distributed with the paper.

| Folder | Contents |
|---|---|
| `method/` | The method: `collocation_q.py` (sinh-mapped collocation for general q, pinned and IVP forms, damped Newton), `flame_q.py` (time function Θ_q and master profile Y_q = Θ_q⁻¹), `collocation.py` (Chebyshev differentiation matrix, barycentric interpolation, q = 2 map setup), `flame.py` (exact q = 2 solution via the Wright ω function), `arrhenius_model.py` (Arrhenius rate: time function, singularity, pinned collocation) |
| `example.py` | Minimal example |
| `experiments/` | One script per experiment of the paper, and `gen_tables.py` (tables and the numbers quoted in the text) |
| `figures/` | Plotting scripts (read `results/`, write `figs/`) |
| `matlab/method/` | MATLAB implementation of the method |
| `matlab/chebfun_comparison/` | MATLAB/Chebfun script for the comparison with Chebfun |
| `results/` | Raw numerical output of the experiments (CSV/TXT), `results/matlab/` the Chebfun output |
| `tables/` | LaTeX table bodies of the paper (written by `experiments/gen_tables.py`) |
| `figs/` | The figures of the paper |

## Requirements

Python 3 with the packages in `requirements.txt` (the versions used for the paper):

```
pip install -r requirements.txt
```

`mpmath` and `python-flint` are needed only for the 256-bit computations (`experiments/exp_highprec*.py`).
The Chebfun comparison needs MATLAB and Chebfun 5.7.0 (the paper used MATLAB R2023b).

## Reproducing the paper

The scripts can be run from any folder, e.g. `python experiments/exp_q2.py`; they read and write
`results/`, `tables/` and `figs/` of the repository.

**The scripts overwrite the reference outputs in `results/`, `tables/` and `figs/`.** To compare a
reproduction with the published files, run it in a fresh clone or a copy of the repository
(or compare with `git diff` afterwards). The version used for the paper is the release v1.0.0,
archived at [doi:10.5281/zenodo.23479185](https://doi.org/10.5281/zenodo.23479185); later
corrections are published as new versions, not by replacing that release.

| Paper | Script | Output (`results/` unless stated) |
|---|---|---|
| Fig. 1 (phase line) | `figures/fig1_phaseline.py` | `figs/fig1_phaseline.pdf` |
| Sect. 3, map centred at the midpoint | `experiments/exp_map_centre.py` | `map_centre.csv` |
| Table 1, Figs. 2–3 (q = 2) | `experiments/exp_q2.py`, `experiments/exp_q2_unmapped.py`, `figures/figs2to4_collocation.py` | `colloc_q2.csv`, `colloc_q2_unmapped.csv`, `figs/fig2_grid_cost.pdf`, `figs/fig3_collocation.pdf` |
| Sect. 5.1, exact convergence exponent, row-scaled condition numbers, reconstruction of y, off-grid defect | `experiments/exp_rate_conditioning.py` | `rate_exact.csv`, `conditioning_scaled.csv`, `reconstruction_defect.csv` |
| Sect. 5.1, sensitivity to rounding (1 vs 2 BLAS threads) | `experiments/exp_rounding_sensitivity.py` | `rounding_sensitivity_q2.csv` |
| Sect. 5, final residuals of all double-precision runs | `experiments/exp_residuals.py` | `residuals_all.csv`, `residuals_summary.txt` |
| Table 2 (choice of the unknown) | `experiments/exp_unknowns.py` | `unknowns_q2.csv` |
| Sect. 5.2, Table 3 (ablation: map × condition × differentiation/integral form; robustness; timing) | `experiments/exp_ablation.py` | `ablation.csv`, `robustness.csv`, `timing.txt` |
| Table 4, Fig. 4 (q = 3) | `experiments/exp_q3.py`, `figures/figs2to4_collocation.py` | `colloc_q3.csv`, `figs/fig4_q3.pdf` |
| Sect. 5.4, Table 5 (Chebfun) | `matlab/chebfun_comparison/run_chebfun_flame.m` | `matlab/chebfun_flame.csv` |
| Sect. 5.5, Fig. 5 (256-bit arithmetic) | `experiments/exp_highprec.py 2`, `experiments/exp_highprec.py 3`, `experiments/exp_highprec_warm.py`, `experiments/exp_highprec_refcheck.py`, `figures/fig5_highprec.py` | `highprec_colloc_q2.csv`, `highprec_colloc_q3.csv`, `highprec_warm_q3.csv`, `hp_reference_check.csv`, `highprec_rates.csv`, `figs/fig5_highprec.pdf` |
| Sect. 5.6, Table 6, Fig. 6 (Arrhenius rate) | `experiments/exp_arrhenius.py`, `figures/fig6_arrhenius.py` | `arrhenius_runs.csv`, `arrhenius_params.csv`, `arrhenius_checks.txt`, `figs/fig6_arrhenius.pdf` |
| Sect. 5.7, Table 7 (AAA) | `experiments/exp_aaa.py`, `experiments/exp_aaa_arrhenius.py` | `aaa_poles.csv`, `aaa_poles_arrhenius.csv` |
| Remark 2.4 (movable singularities) | `experiments/exp_movable.py` | `movable_singularities.csv` |
| Lemma 4.3 (translation mode of the Newton matrix) | `experiments/exp_translation_mode.py` | `translation_mode.csv` |
| All tables and the numbers quoted in the text | `experiments/gen_tables.py` | `tables/table*.tex`, `text_numbers.txt` |

Suggested order: `exp_q2.py` and `exp_q3.py` first (other scripts read their output), then the
remaining experiments, `exp_residuals.py` last among them (it reruns all double-precision solves
and checks them against the saved files), then the figure scripts and `gen_tables.py`.
The double-precision experiments take minutes; the 256-bit computations take about
10 minutes (q = 2) and 30 minutes (q = 3).

Notes on reproducibility:

* Newton's method is stopped when the full Newton correction is below 10⁻¹² and, after it, the
  residual is below 10⁻¹⁰ (step 6 of the algorithm in the paper; in 256-bit arithmetic 2⁻²³⁶ and 2⁻²²⁶).
* The pinned form is insensitive to rounding: rerunning with a different number of BLAS threads
  changes its errors only at the level of rounding errors and its iteration counts not at all.
  The IVP form is not: its iteration counts, convergence and errors can change when only the
  order of the floating-point operations changes (this is one of the findings of the paper).
  The results in `results/` were computed with two OpenBLAS threads (see `results/timing.txt`).
* Regenerated figures can differ from the files in `figs/` in font embedding, not in content.

## License

MIT (see `LICENSE`), for the code and the data.

## Citation

Please cite the paper and this repository (see `CITATION.cff`).
