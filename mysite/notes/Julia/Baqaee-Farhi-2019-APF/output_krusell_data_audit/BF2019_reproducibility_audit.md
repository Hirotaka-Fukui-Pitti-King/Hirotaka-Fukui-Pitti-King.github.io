# Baqaee and Farhi (2019): Source-to-publication reproducibility audit

This report distinguishes source-file provenance, official MATLAB-equivalent numerical results, the officially supplied `results tables1-5.xlsx`, and printed journal tables. It does not claim the MATLAB executable was run.

## 1. Source Excel → official MAT files: **EXACT**

| MATLAB variable | Elements | Exact cells | Mismatches | Max abs. error |
|---|---:|---:|---:|---:|
| `IO2013` | 8134 | 8134 | 0 | 0.0 |
| `output_new` | 63 | 63 | 0 | 0.0 |
| `VA_new` | 63 | 63 | 0 | 0.0 |
| `labor_no_col` | 63 | 63 | 0 | 0.0 |
| `labor_col` | 63 | 63 | 0 | 0.0 |

Total 8386 cells; 0 mismatches. Original IO2013 is copied from `2013!C8:CV90` in the BEA Use workbook (83×98); literal `...` entries are converted to numerical zeros. Values, zeros, negatives and ordering are retained. The four production-account vectors each come from their named sheet's 2013 column `Q3:Q65` (63×1). Source files indicate millions of US dollars. These ranges are grounded in sheet headings, row/column identifiers and the 2013 year header, not in a numerically optimized matching search.

## 2. Official archived result workbook

Compared 236 accessible numerical cells/positions in Sheets 1–5 with the verified Julia numerical routines.
Cells exceeding their documented comparison tolerance: 0.

**Precision:** Sheets 1, 2 and 5 save substantially more digits than the journal tables. Sheet3 saves derivatives to about four decimal places; Sheet4 saves capital-shock components to about four decimal places. Comparisons respect those differences.

## 3. Published tables: source-supported differences

| Journal location | MATLAB-equivalent / official Excel | Printed value | Classification |
|---|---:|---:|---|
| B.1 / aggregate / Noncollege / College / cost MES | 1.064856153 | 1.07 | F |
| B.2 / disaggregated / College / Capital / production MES | 0.663104421 | 0.67 | F |
| B.3 / disaggregated / theta2 / HK / production sensitivity | 0.044880758 | 0.05 | F |
| B.3 / disaggregated / theta2 / HK / cost sensitivity | 0.017198137 | 0.01 | F |
| B.4 / aggregate / theta2 / LK / production sensitivity | 0.239565154 | 0.97 | E |
| B.4 / aggregate / theta1 / LK / production sensitivity | 0.971154461 | 0.24 | E |
| B.4 / aggregate / theta2 / LK / cost sensitivity | 0.605143847 | 0.4 | E |
| B.4 / aggregate / theta1 / LK / cost sensitivity | 0.394856153 | 0.6 | E |
| B.4 / disaggregated / Benchmark / LK / MES | 0.935519638 | 0.93 | F |
| B.4 / disaggregated / theta4 / LK / production sensitivity | 0.095997321 | 0.07 | E |
| B.4 / disaggregated / theta2 / LK / production sensitivity | 0.220584668 | 0.57 | E |
| B.4 / disaggregated / theta1 / LK / production sensitivity | 0.569562648 | 0.22 | E |
| B.4 / disaggregated / theta2 / LK / cost sensitivity | 0.459618286 | 0.26 | E |
| B.4 / disaggregated / theta1 / LK / cost sensitivity | 0.263537718 | 0.46 | E |

Classification A: exact; B: agrees at published precision; E: a verified source row/column label switch would reconcile; F: remaining numerical publication discrepancy. Class E describes a plausible *layout* explanation, not proven author intent.

### High-priority interpretation

- Tables B.1–B.2: official Excel matrices are in *MATLAB raw* orientation. The paper displays their transpose. The 1.064856… vs 1.07 (B.1) and 0.663104… vs 0.67 (B.2) cells remain different at two decimals.
- Table B.3: source workbook Sheet3 gives rounded sensitivity values, and Sheet5 records much greater precision. The journal's theta2 values do not both follow ordinary rounding of the source values.
- Table B.4: parameter-label comparison against Sheet5 supports a theta1/theta2 row-placement discrepancy, across both models and both Cost and Production. The disaggregated theta4 Cost/Production publication columns also appear reversed relative to Sheet5. These hypotheses are not evidence of different structural parameters.
- Table B.5: the Production side reproduces all three printed capital-shock effects for both models. The cost side is a distinct object.

## 4. Method and scope

- Archive README identifies the original 2013 BEA Use table (release November 2, 2017) and BEA/BLS production accounts.
- `Krusell_sens.jl` reuses `Krusell_disagg.jl`, `Krusell_agg.jl`, `Relabel_mod.jl`, `WeightedCovMatrix.jl` without modifications.
- All MATLAB-equivalent calculations use the source supplied five MAT arrays and exactly the effective calibrations of the archive.
- Numerical properties audited include Leontief inverse residual, Hessian symmetry, factor shares, and sensitivity orientation.
- MATLAB binary execution was NOT performed by this script. Source-code equivalence is supported by previously independently checked NumPy computations and the original results Excel workbook.

## 5. Certification matrix

| Link | Status |
|---|---|
| Source Excel → MAT | EXACT |
| Official MAT → aggregate & disaggregated numerical models | Validated previously; rerun in this audit |
| Official MAT → capital-shock simulation | Validated previously; rerun in this audit |
| Official MAT → MES sensitivities | Validated previously; rerun in this audit |
| Official results workbook ↔ Julia | Match at source precision |
| Official results workbook ↔ printed journal | Discrepancies classified above |
| Figures 3–4 | Verified in Tasks I–II; not recomputed here |
| Julia 1.12 runtime | Determined by the user's actual execution log, not inferred by script creation |

**Conclusion:** The five numerical input MAT files have an exact, auditable mapping to the two official source Excel workbooks. Reproducibility of the computational model does NOT imply perfect identity with every printed journal table. Explicit exceptions are recorded in `paper_table_discrepancies.csv` and `sensitivity_label_audit.csv`.
