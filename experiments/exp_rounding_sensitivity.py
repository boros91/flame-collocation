"""Sensitivity of the q = 2 experiment (exp_q2.py setting) to rounding errors:
the same computation is run with 1 and with 2 OpenBLAS threads, which changes only the
order of floating-point operations in the matrix products and LU factorizations.
Writes results/rounding_sensitivity_q2.csv (per run) and prints a summary
(also appended to results/text_numbers.txt by gen_tables.py)."""
import os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(ROOT, 'method'))
import os, sys, subprocess, tempfile


def run(threads, out):
    import numpy as np, pandas as pd
    from collocation_q import solve, error
    rows = []
    for d in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10, 1e-12]:
        for N in range(32, 513, 32):
            sp = solve(d, N, 2, pin=True); si = solve(d, N, 2, pin=False)
            rows.append(dict(delta=d, N=N, pin=error(sp), pin_its=sp['its'],
                             ivp=error(si), ivp_its=si['its'], ivp_conv=int(si['converged'])))
    pd.DataFrame(rows).to_csv(out, index=False)


if __name__ == '__main__':
    if len(sys.argv) == 3:
        run(int(sys.argv[1]), sys.argv[2]); sys.exit()
    import pandas as pd
    files = {}
    for t in (1, 2):
        env = dict(os.environ, OPENBLAS_NUM_THREADS=str(t), OMP_NUM_THREADS=str(t), MKL_NUM_THREADS=str(t))
        files[t] = os.path.join(tempfile.gettempdir(), f'_rs_{t}.csv')
        subprocess.run([sys.executable, __file__, str(t), files[t]], env=env, check=True)
    a, b = pd.read_csv(files[1]), pd.read_csv(files[2])
    df = a.merge(b, on=['delta', 'N'], suffixes=('_1thread', '_2threads'))
    df.to_csv(ROOT + '/results/rounding_sensitivity_q2.csv', index=False)
    m = df.pin_1thread > 1e-12
    print(f'runs {len(df)}; pinned: max rel. error change {(abs(df.pin_1thread - df.pin_2threads) / df.pin_1thread)[m].max():.1e}, '
          f'iteration counts changed {(df.pin_its_1thread != df.pin_its_2threads).sum()}; IVP: iteration counts changed '
          f'{(df.ivp_its_1thread != df.ivp_its_2threads).sum()}, convergence outcome changed '
          f'{(df.ivp_conv_1thread != df.ivp_conv_2threads).sum()}, error changed by more than 0.1% '
          f'{((abs(df.ivp_1thread - df.ivp_2threads) / df.ivp_1thread) > 1e-3).sum()}')
