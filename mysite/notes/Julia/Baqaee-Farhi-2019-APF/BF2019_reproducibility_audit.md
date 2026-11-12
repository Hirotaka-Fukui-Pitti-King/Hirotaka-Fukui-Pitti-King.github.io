# Baqaee and Farhi (2019): Source-to-publication reproducibility audit

**Audit date:** 2026-11-12  
**Paper:** *JEEA-FBBVA Lecture 2018: The Microeconomic Foundations of Aggregate Production Functions*, *Journal of the European Economic Association*, 17(5), 1337–1392, DOI 10.1093/jeea/jvz050.  
**Source:** Official author replication ZIP, two underlying Excel workbooks, five MAT files, `results tables1-5.xlsx`, the official MATLAB scripts, previously tested Julia Tasks I–VII, and publication Appendix B.

## Executive conclusion

1. **Source Excel → MAT: EXACT, all 8,386 entries.** The original workbook cell ranges and the 2013 calendar year are independently identified from the archived workbooks. No data fitting, scaling, interpolation, or guessed industry reordering is needed.
2. **Official archived results workbook → independently computed mathematical benchmark: reproduced at appropriate source precision, 236 of 236 compared numerical positions.** Benchmark matrices agree to numerical precision. Derivative matrices agree to within 5.52e-06 (analytic derivative vs archived forward differences).
3. **Published appendix tables ≠ all archived calculation outputs.** Of 66 displayed off-diagonal MES, capital-shock and sensitivity comparisons, 14 are exact, 38 agree at displayed precision, 9 can be reconciled by row/column placement switches, and 5 remain unexplained by the printed precision or verified placement switches.
4. **MAT → Julia runtime:** Tasks V–VII were executed successfully by the user (66/66, 89/89, 119/119 tests respectively). **`Krusell_data_audit.jl` itself has not been run in Julia in this environment and must not yet be certified as Julia-execution-verified.**
5. **MATLAB binary:** not executed. Code/source interpretation and the author-provided numeric workbook are independently inspected; do not confuse this with running original MATLAB.

## 1. Exact source-data provenance

| MAT file / variable | Source XLSX, sheet | Identified cells | Matches | Max absolute error |
|---|---|---|---:|---:|
| `IO2013.mat` / `IO2013` | `IOUse_After_Redefinitions_PRO_1997-2016_Summary.xlsx`, `2013` | `C8:CV90` | 8,134 / 8,134 | 0 |
| `output_new.mat` / `output_new` | `BEA-BLS-industry-level-production-account-1998-2015.xlsx`, `Gross Output` | `Q3:Q65` | 63 / 63 | 0 |
| `VA_new.mat` / `VA_new` | Same, `Value Added` | `Q3:Q65` | 63 / 63 | 0 |
| `labor_no_col.mat` / `labor_no_col` | Same, `Labor_NoCol Compensation` | `Q3:Q65` | 63 / 63 | 0 |
| `labor_col.mat` / `labor_col` | Same, `Labor_Col Compensation` | `Q3:Q65` | 63 / 63 | 0 |

The IO source has 83 original rows × 98 columns. The header on `2013!A6:CV7` defines commodity/industry identifiers and the intermediate/final uses; the numeric block `C8:CV90` has **2,555 literal `...` cells**, all represented as zero in `IO2013.mat`. There are **83 negative entries**, retained without clipping. The BEA data header identifies millions of dollars. The production-account header has years 1998–2015 in `B2:S2`, so **2013 = Q2**. Four production-account data blocks each contain 63 industries and are copied exactly, without transformations.

Source provenance: the archive README identifies the November 2, 2017 release of BEA Use Summary Tables and the 2013 BEA/BLS industry-level production accounts. The present audit establishes exact **numeric extraction**, not authorship/collection of the underlying government statistics.

The archived MATLAB disaggregated code subsequently merges IO industries and removes rows/columns, reducing the original 83-row raw source to 63 industries. This later **model construction** should not be mistaken for a preprocessing step used to make the original `.mat` file. After keeping IO parent codes in original order and comparing descriptions with the 63-row production-account list, the following five renamed/merged descriptions are documented:

| 63-sector index | Retained IO-source label | Production-account label |
|---:|---|---|
| 28 | Motor vehicle and parts dealers | Retail trade |
| 45 | Housing | Real estate |
| 55 | Hospitals | Hospitals and nursing and residential care facilities |
| 62 | Federal general government (defense) | Federal |
| 63 | State and local general government | State and local |

The matching index order and the model's group-merging logic are grounded in the supplied MATLAB script; these five *description* differences must not be interpreted as evidence of a different numeric mapping.

All 8,386 original source-to-MAT comparisons are available individually in `io2013_source_mapping.csv` and `production_account_mapping.csv`. The 5-file summary is `source_mat_comparison.csv`.

## 2. Comparison with `results tables1-5.xlsx`

Numerical comparison against the official result workbook covers **236 numeric positions**, all within source-appropriate tolerances. The overall maximum absolute discrepancy is **8.74589269e-05**; this occurs in a rounded, secondary skill-premium value on Sheet4, not in a benchmark MES. The largest deviation among Sheet5 derivative matrices is **5.51893311e-06** (archived forward-difference values versus a separately evaluated analytic derivative).

- **Sheet1:** 3×3 Aggregate Production and Cost MES, original MATLAB orientation.
- **Sheet2:** 3×3 Disaggregated Production and Cost MES, original MATLAB orientation.
- **Sheet3:** high-skilled–capital benchmark and five microelasticity derivative rows; derivative values have truncated display/storage precision relative to Sheet5.
- **Sheet4:** Production and Cost responses for both models. Auxiliary `G4:G5` skill-premium figures are at approximately four-place, intermediate-rounded precision.
- **Sheet5:** original MATLAB orientation of the full 3×3 MES matrices and 3×3 derivative matrices for each parameter, for both `With IO` and `Without IO`. The crucial `Without IO` benchmark is **`B43:D45` and `G43:I45`**, while derivative first-row positions are `C50`, `H50` (theta2) and `C56`, `H56` (theta1). `With IO` corresponding cells are `C29`, `H29` and `C35`, `H35`.

**Orientation:** a paper row–column cell `(i,j)` must be compared to the original MATLAB raw element `(j,i)`. Source workbook Sheets1–2 and Sheet5 store the untransposed MATLAB matrices; Table B.1–B.2 display the transpose. `mes_orientation_audit.csv` records all 36 raw-to-paper baseline cell correspondences.

## 3. Publication discrepancies

The independent numerical audit classifies **66** published comparisons as follows:

| Classification | Interpretation | Count |
|---|---|---:|
| A | Exact numeric match | 14 |
| B | Equal at the number of decimals actually printed | 38 |
| E | Value reconciled by *independently observed* row/column placement switch | 9 |
| F | Numerical discrepancy not resolved by printed precision or placement switch | 5 |

These are **evidence classifications**. Class E does not prove which person or production step caused a row/column switch.

### Table B.1 — Aggregate MES

The original MATLAB-oriented Cost MES cell `Sheet1!H7` is **1.06485615287267**, while the corresponding printed paper cell (Noncollege row, College column) is **1.07**. Normal two-decimal rounding gives **1.06**. The official Excel and independently computed matrix agree; **classification F**.

### Table B.2 — Disaggregated MES

Original production MES `Sheet2!D5` is **0.663104420643937**, appearing in the paper's College row / Capital column as **0.67**. Normal rounding gives **0.66**. The official Excel and independent calculation agree; **classification F**.

### Table B.3 — High-skilled vs Capital

For `theta2` in the Disaggregated model, the archived full derivative matrix `Sheet5!I29` reports approximately **0.0171981374** (Cost), and `Sheet5!D29` reports **0.0448807399** (Production). Printed values are **0.01** and **0.05**. Neither follows ordinary two-decimal rounding; **two classification-F cells**. The article's `theta3` Cost entry is **`0.1`**, *one decimal place*, so the computed **0.0949506** agrees at the actual printed precision; it is **not an additional unexplained discrepancy**.

### Table B.4 — Low-skilled vs Capital

This is the most consequential layout mismatch. The official workbook explicitly labels `theta2` and `theta1` on Sheet5, and the values agree with the previously user-executed Julia Task VII and the independent analytical reference:

| Model | Parameter | Cost derivative | Production derivative | Official Sheet5 first-row cells |
|---|---|---:|---:|---|
| Aggregate / Without IO | `theta2` | 0.605143847 | 0.239565750 | H50 / C50 |
| Aggregate / Without IO | `theta1` | 0.394856153 | 0.971152639 | H56 / C56 |
| Disaggregated / With IO | `theta2` | 0.459618286 | 0.220585000 | H29 / C29 |
| Disaggregated / With IO | `theta1` | 0.263537718 | 0.569561950 | H35 / C35 |

All eight derivative values appear in Table B.4 under the *other* parameter's row, once original column precision is taken into account. This supports a **row-placement/label-swap explanation**, not a sign of the Julia numerical implementation choosing the wrong derivatives. The archived MATLAB code and journal labels use the same economic nomenclature; a separately verified alternate calibration that reproduces the printed row ordering has not been established.

For the Disaggregated `theta4` Production derivative, `Sheet5!C23` is approximately **0.0959972**, whereas the journal prints **0.07**. Its other-column Cost derivative `Sheet5!H23` is **0.0722536**, which rounds to **0.07**. This supports a **Cost/Production column-placement explanation** for that cell. The journal's Cost value `0.1` has only one decimal, so 0.0722536 is within the rounding interval for `0.1`; the precision alone does not reject the Cost column as printed.

For the Disaggregated Production benchmark, the Julia and official Excel value **0.9355196384** normally rounds to **0.94**, consistent with Table B.2. Table B.4 prints **0.93**, an additional **classification-F** difference.

### Table B.5 — Capital shock

Both models' three **Production-side** factor-income-share responses match the journal's printed precision.

| Factor | Aggregate computed | Disaggregated computed |
|---|---:|---:|
| Capital | +0.05007078 | +0.06186436 |
| Noncollege | +0.06795125 | +0.03634919 |
| College | −0.13226130 | −0.12621334 |

The implied college-minus-noncollege income-share response is **-0.20021254** in the Aggregate model and **-0.16256253** in the Disaggregated model. Their magnitudes differ by about **18.81%**, consistent with the paper's description of roughly 20%. The Cost-side shock is a distinct computed object and should not be substituted for Table B.5.

## 4. Numerical and economic checks

The prior user Julia execution provides:

- Task V: 66/66 passed; Aggregate Leontief residual 0 and Hessian symmetry residual ~1.39e−17.
- Task VI: 89/89 passed; Disaggregated Leontief residual ~6.66e−16 and Hessian symmetry residual ~2.78e−17.
- Task VII: 119/119 passed; largest forward/analytic derivative difference ~3.70e−6; Cost MES homogeneity sums equal 1 to reported precision for all four studied cases.

The current independent NumPy/SciPy audit confirms the mathematical benchmarks and full result-workbook derivatives. **No MATLAB executable has been run.** The new Julia Task VIII source file is provided separately but **Julia 1.12 syntax and execution of this new file are not yet verified** in the current environment.

## 5. Reproducibility certification

| Chain | Evidence-based status |
|---|---|
| Source Excel → five MAT datasets | **EXACT:** 8,386/8,386 entries |
| Official MAT → Aggregate MES and capital shock | **Verified in prior user's Julia execution** |
| Official MAT → Disaggregated MES and capital shock | **Verified in prior user's Julia execution** |
| Official MAT → finite-difference sensitivities | **Verified in prior user's Julia execution** |
| Official result workbook → independent numerical implementation | **All 236 comparisons within appropriate precision** |
| Official result workbook → printed Tables B.1–B.5 | **52 as printed, 9 placement-sensitive, 5 unresolved numerical discrepancies** |
| Figures 3–4 | **Verified in earlier tasks**, not rerun in this audit |
| Original MATLAB binary execution | **Not performed** |
| New `Krusell_data_audit.jl` Julia syntax/tests/output | **Pending user's Julia runtime verification** |

**Bottom line:** The five original input MAT datasets have now been reconstructed *exactly* from the official Excel source workbooks. The main remaining source-versus-journal differences arise downstream, between the fully documented computational results and the printed article tables. We have strong cell-level evidence for the Table B.4 row and column placements, but cannot establish the authors' internal publication workflow from the provided files alone.

## 6. Files and traceability

`source_workbook_inventory.csv`, `io2013_source_mapping.csv`, `production_account_mapping.csv`, `source_mat_comparison.csv`, `official_results_workbook_comparison.csv`, `paper_table_discrepancies.csv`, `mes_orientation_audit.csv`, `sensitivity_label_audit.csv` are independent Python-generated audit evidence, not falsely labeled as Julia-generated output.

The supplied `Krusell_data_audit.jl` creates corresponding CSV, Markdown and SummaryTables LaTeX outputs directly from original XLSX and MAT files when executed in Julia 1.12 with the verified companion modules.

