# Caballero–Farhi–Gourinchas (2008): Julia figure replication

Independent, source-faithful **Julia reconstruction of Figures 4–6** of:

> Caballero, Ricardo J., Emmanuel Farhi, and Pierre-Olivier Gourinchas (2008). “An Equilibrium Model of ‘Global Imbalances’ and Low Interest Rates.” *American Economic Review* 98(1): 358–393. https://doi.org/10.1257/aer.98.1.358

The accompanying [`REPLICATION_AUDIT.md`](REPLICATION_AUDIT.md) documents model equations, GAUSS/source discrepancies, the sensitivity extension, numerical checks, and the limits of reproducibility certification. **The Julia implementations are not literal GAUSS translations and have not been certified as byte-for-byte equivalent to running GAUSS.**

## Inventory

| Julia source | What it computes | Original source | Output directory |
|---|---|---|---|
| `figure4_asset_supply.jl` | Figure 4: one-time permanent asset-supply collapse, four panels; closed-form wealth dynamics | `prog_UR_final.gau` | `output_fig04/` |
| `figure5_fdi.jl` | Figure 5: investment and FDI, four panels; reverse asset-value shooting and forward wealth integration | `prog_UR_inv_final.gau` | `output_fig05/` |
| `figure6_exchange_rates.jl` | Figure 6: CES goods and real exchange rates, five panels; scalar shooting and RK4 | `prog_UR_exch_final.gau` | `output_fig06/` |
| `figure5_dual_accounting_sensitivity.jl` | Additional four-price comparison for Figure 5; GAUSS-defined current account, net-foreign-asset accumulation, NFA, accounting wedge | Added analysis, **not in official GAUSS package**; includes `figure5_fdi.jl` | `output_fig05_sensitivity/` |

**Official original replication package**: the accompanying AEA-provided `113231-V1.zip` (`CFG_AER_Data.xls`, three `.gau` files, `README_CFG.pdf`, `LICENSE.txt`). This package does **not** supply scripts for all published empirical tables. Keep official source files separate from the independent Julia implementation.

## Dependencies and environment

- Julia; the user has confirmed execution of all four files in **Windows PowerShell** on 2026-11-12. **The exact Julia version, package versions, and OS build were not supplied.**
- `CairoMakie.jl` for plots, including publication-style PDFs and PNGs, with explicit **Okabe–Ito** hex colors (e.g., `#0072B2` and `#D55E00`).
- `SummaryTables.jl` for HTML/LaTeX result tables.
- Julia standard library `Printf`; no GAUSS/MATLAB interpreter is needed to run these Julia reconstructions.

**Repository environment caveat (checked on 2026-11-12):** the repository's `mysite/Project.toml` lists `SummaryTables` but **does not list `CairoMakie`**. Even if an already configured machine can run the figures by finding `CairoMakie` in a shared Julia environment, the project is not yet self-contained. From the `mysite` directory, register both packages in that project:

```powershell
julia --project=. -e 'using Pkg; Pkg.add(string(:CairoMakie)); Pkg.add(string(:SummaryTables))'
```

For stronger portability, commit the resulting `Project.toml` and a generated `Manifest.toml` after verifying its compatibility with the rest of the website project. **No package versions are invented or pinned in this README.** Inspect your actual environment using:

```powershell
julia --project=. --version
julia --project=. -e 'using Pkg; Pkg.status()'
```

## Run (from the `mysite` directory)

Place all four `.jl` files in `notes/Julia/Caballero-Farhi-Gourinchas-2008-GI/`; the supplemental Figure 5 file **must** be beside `figure5_fdi.jl` because it uses `include(joinpath(@__DIR__, "figure5_fdi.jl"))`.

```powershell
julia --project=. .\notes\Julia\Caballero-Farhi-Gourinchas-2008-GI\figure4_asset_supply.jl
julia --project=. .\notes\Julia\Caballero-Farhi-Gourinchas-2008-GI\figure5_fdi.jl
julia --project=. .\notes\Julia\Caballero-Farhi-Gourinchas-2008-GI\figure6_exchange_rates.jl
julia --project=. .\notes\Julia\Caballero-Farhi-Gourinchas-2008-GI\figure5_dual_accounting_sensitivity.jl
```

By default each script creates its own `output_figXX` folder relative to the *script's location*, not necessarily the current terminal directory. Each script may also take an explicit output path (see source header); `figure5_fdi.jl` additionally accepts `--paper-label-values` to calculate the **5%** paper-caption scenario (not an incentive-compatible voluntary-FDI equilibrium under the same calibration).

### Files expected under output folders

- `output_fig04/`: `figure4.png`, `figure4.pdf`, `figure4_summary.html`, `figure4_summary.tex`, `figure4_timeseries.csv`.
- `output_fig05/`: `figure5.png`, `figure5.pdf`, `figure5_summary.html`, `figure5_summary.tex`, `figure5_timeseries.csv`, `figure5_metadata.txt`.
- `output_fig06/`: `figure6.png`, `figure6.pdf`, `figure6_summary.html`, `figure6_summary.tex`, `figure6_timeseries.csv`, `figure6_metadata.txt`.
- `output_fig05_sensitivity/`: `figure5_dual_accounting.png`, `figure5_dual_accounting.pdf`, `figure5_dual_accounting_summary.html`, `figure5_dual_accounting_summary.tex`, `figure5_dual_accounting_timeseries.csv`, `figure5_dual_accounting_notes.txt`.

These are *expected script outputs*, not a claim that all of them are already committed to GitHub. The audit inspected the supplied figures and Figure 4–6 tables/CSVs and the supplemental plot/notes; the **Julia-generated supplemental SummaryTables and CSV were not supplied** for direct independent inspection. A separately computed Python reference CSV is available as `figure5_dual_accounting_reference.csv` from the earlier audit.

## Important economic/accounting distinctions

1. **Figure 5 acquisition price κP:** the original GAUSS code actually uses **8.129032%** and **12%** for its displayed extremes. The paper and original legend say **5%** and **12%**. The 5% scenario is an **imposed-FDI sensitivity**: under unchanged primitives, the seller rejects, so it is **not** a voluntary equilibrium. The **9.5%** added case is strictly inside the incentive-compatible interval `(8.129032%, 12%)`; the two original endpoints are indifference boundary cases.
2. **Figure 5 current account:** the original plotted `CAXUp` is `TB_U + r·NFA_U`. With FDI it is **not** the same as `d(NFA_U)/dt`. The supplemental program displays **both**; the second is labeled `NFA accumulation rate` in the audit, not automatically the *statistical* current account. The gap reflects new R-tree rights, acquisition payments and, for general κ, investment. The unexpected asset-price jump at `t=0` is excluded from the continuous-flow comparison.
3. **Figure 6:** the approximately **−1.56%** CA and **1.6%** exchange-rate depreciation in the paper align closely with its finite-horizon display, whereas the continuous-time **strict** limits are **−1.469625%** and **+1.959788%** (ordinary percentage change). This distinction does not change the paper's substantive mechanism.
4. **World real return in Figure 6C** is the output-weighted average of U- and R-consumption-real rates, **not** the U-good-numeraire return. The Julia graph uses the former.

## Recorded validation and limitations

- User-provided PowerShell logs show all four Julia scripts executed successfully, including structural/accounting self-check messages.
- Independent Python/SciPy calculations and time-step/terminal-horizon sensitivity checks support the numerical paths; see audit for exact coverage and residuals.
- The present audit environment **does not have Julia or GAUSS executables**. It has not conducted a new `Meta.parseall`, Julia full rerun, GAUSS interpreter run, type inference/benchmark, or CI verification.
- The source and the generated plots share an intentional distinction: numerical near-equivalence is **not** exact source-runtime identity. **Do not silently alter original plotted series to force a paper caption or an accounting identity to match.**

## Publishing / version control

Commit this README, `REPLICATION_AUDIT.md`, all four Julia scripts, and representative final figures/tables/CSV output where file sizes permit. Avoid committing temporary diagnostics, caches and enormous intermediate files. Preserve the original PDF, Excel data and original GAUSS code with their source/licensing notices if redistribution is appropriate; otherwise document how to obtain the official package. Before public release, capture the actual Julia and package versions using the commands above.

**Repository status at audit:** the three base `.jl` files exist on GitHub `main`; the supplemental `.jl`, this README and the final audit were **not found** in that directory. They must be copied or committed separately. **No GitHub commit or successful CI run is asserted here.**
