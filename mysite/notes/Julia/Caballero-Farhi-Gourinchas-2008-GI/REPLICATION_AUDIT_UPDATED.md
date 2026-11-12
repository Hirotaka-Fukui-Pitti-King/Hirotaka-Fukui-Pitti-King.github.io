# Caballero, Farhi, and Gourinchas (2008) — Julia Replication Audit

**Audit date:** 2026-11-12. **Paper:** Caballero, Ricardo J., Emmanuel Farhi, and Pierre-Olivier Gourinchas (2008), “An Equilibrium Model of ‘Global Imbalances’ and Low Interest Rates,” *American Economic Review* 98(1), 358–393, DOI 10.1257/aer.98.1.358.

**Verdict:** Figures 4–6 are validated as *continuous-time, source-faithful model reconstructions*, with the three qualifications below. They are **not certified as exact, interpreter-to-interpreter GAUSS replications**. The previously supplied Windows Julia execution logs are valid evidence of running the Julia files, but there was no fresh Julia or GAUSS interpreter run in this audit environment. Do not present the Figure 5 source-defined current account as the derivative of NFA in the FDI case.

## 1. Sources and exact inventory

Original paper (36 pages), particularly Sections I, III, IV, Appendix D–G, and Figures 4–6. The official ZIP `113231-V1.zip` contains `CFG_AER_Data.xls`, `LICENSE.txt`, `README_CFG.pdf`, and the following original scripts:

| GAUSS source | Lines | Julia reimplementation | Git blob SHA (same in GitHub and local audit copy) |
|---|---:|---|---|
| `prog_UR_final.gau` | 229 | `figure4_asset_supply.jl` (375 lines) | `75d5ddf55e64450f79fcb31bf07eaf942b1af7b9` |
| `prog_UR_inv_final.gau` | 324 | `figure5_fdi.jl` (517 lines) | `b5e7899a92e4bb525ef99897ae7450c663dae6d1` |
| `prog_UR_exch_final.gau` | 429 | `figure6_exchange_rates.jl` (679 lines) | `1db717458832111ecb20d2ee5591f8bebd93fb15` |

The three extracted GAUSS scripts were byte-compared with the contents of the supplied ZIP. SHA-256 respectively: `28583037c9d002c97e1c9a573a4a5adc29cabe5e891bf0a286a9f7aa585ad500`, `261e95e06962ee2155cb9fc9d87758536373c3bc8743238acc5d5f8a79b75359`, `275674b1c97ae0c42a2896505167968a89260195dd115761c05254c6cfdbce1c`.

User-supplied artifacts were examined: three plotted figures (PNG/PDF), associated HTML/LaTeX tables generated with SummaryTables.jl, CSVs, Figure 5–6 metadata, and PowerShell execution logs. The CSVs have 1,080 (Figure 4) and 1,081 (Figures 5, 6) rows, with no missing values or duplicate timestamps. This audit does **not** cover Figures 1–3 or Tables 2–3 of the paper.

## 2. Figure 4: permanent asset supply shock

**Original:** Section I.B, Proposition 1, Figure 4 (p.370), `prog_UR_final.gau`. **Result: analytic source-matched reconstruction.**

Calibration: θ=.25, g=.03, pre-shock r=.06, output share xU=.7, δU=.12, initial U NFA=0, R financial wealth initially invested in U=.05, R asset-value crash −25%. Let z=.75. The post-shock capitalization rate for R is

$$\delta_R=\frac{zx_U\delta_U}{1-z(1-x_U)}=.08129032258064517,\quad \bar\delta=x_U\delta_U+(1-x_U)\delta_R,\quad r=g+\theta\bar\delta=.05709677419355.$$

World asset market clears at V=W=X/θ at impact, despite country-level valuation changes. The source ownership notation is AUR = U-held share of R trees and ARU = R-held share of U trees. The Julia asset-holdings and R portfolio reconstruction follow this convention. For U financial wealth relative to output,

$$ w_U(t)=w_U^\infty+[w_U(0^+)-w_U^\infty]e^{-(g+\theta-r)t},\quad w_U^\infty=\frac{1-\delta_U}{g+\theta-r}. $$

U net foreign assets/output nU=wU−vU, trade balance/output bU=1−θwU, and current account/output aU=bU+rnU. Hence along the BGP, aU=gnU.

| Quantity | Julia output | Independent analytic benchmark / paper figure |
|---|---:|---|
| Post-shock world rate | 5.709677% | 5.709677% |
| Initial U CA/output | −10.123766% | Exact analytic evaluation |
| Initial U NFA/output | −3.061224% | Initial valuation effect |
| Long-run U CA/output | −1.442010% | Paper ~−1.4% |
| Long-run U NFA/output | −48.066984% | Paper ~−48% |
| Long-run share of R wealth in U assets | 30.851475% | Paper ~31% |

The code checks world clearing, the change in R valuation, and the two-country international accounting identities. Independent central differences applied to the exported CSV give max interior residual ~5.1e−7 for **d nU/dt + g nU − aU = 0**; this is finite-difference error. The GAUSS code itself states that its wealth equation is solved exactly. **Result:** mathematical and quantitative replication PASS; separate GAUSS runtime comparison NOT VERIFIED.

## 3. Figure 5: investment and FDI

**Original:** Section III, Assumption 6, Proposition 5, Figure 5 (p.380), footnote 30 and Appendix E–F; `prog_UR_inv_final.gau`. **Result: state and valuation dynamics matched, with source parameter and accounting qualifications.**

### 3.1 Parameters, algorithm and equivalence

θ=.25, g=gn=.03, gz=0, κ=0, xU=.7, initial NFA=0, initial R valuation shock −25%. These give δU=.24, δR=.162580645161 and post-shock no-FDI rate .054193548387. Source employs reverse-time integration from old-R share approaching zero, followed by forward integration of U wealth. The Julia version uses backward RK4 for old-R and U-tree valuations and forward RK4 for wealth with Hermite midpoint interpolation; its no-FDI path uses the analytical solution. Define xRo(t)=(1−xU)exp(−gn t); normalized old-R and U-tree valuations vRo and vU. The source equations are

$$r(t)=g^z+\frac{\theta}{1-\kappa}\left[\delta_U(1-x_{Ro})+\delta_R x_{Ro}-g^n v_{Ro}x_{Ro}(\delta_U/\delta_R-1)\right],$$
$$\dot v_{Ro}=(r-g^z)v_{Ro}-\delta_R,\qquad\dot v_U=(r-g^z)v_U-\delta_U.$$

Their terminal values are vRo=(1−κ)δR/(θδU), vU=(1−κ)/θ. These are the GAUSS and Appendix F equations. The Julia code preserves pre-shock cross-holdings and κ=0 as the supported model specialization.

### 3.2 Material source discrepancy: κP

In `prog_UR_inv_final.gau` **lines 191–197**, the four κP scenarios are generated by a sequence between

$$\kappa_P^{min}=g^n\frac{\delta_R}{\delta_U}\frac{1-\kappa}{\theta}=.08129032258064516,\qquad\kappa_P^{max}=g^n\frac{1-\kappa}{\theta}=.12.$$

The two middle computed values are .094193548 and .107096774. The figure chooses the first and last scenarios, but the GAUSS hardcoded legend and the published Figure 5 and footnote 30 say **5%** instead of **8.129032%**. The early GAUSS `kappap=.02` assignment is overwritten before the relevant simulation. This is a real inconsistency, not rounding.

Assumption 6 is **strict:** .081290323 < κP < .12. Hence 5% violates the R private-gain lower bound; 8.129% is the R-indifference endpoint and 12% the U-indifference endpoint. Neither plotted endpoint strictly satisfies the proposition. The authors acknowledge equality benchmarks in the text following Figure 5. **Decision:** the default Julia source-numeric reproduction (8.129% and 12%) is correct; `--paper-label-values` is an explicitly separate sensitivity, not the official source-calculation replication.

### 3.3 Numerical comparison and convergence

| Year 25 | No FDI | FDI κP=8.129% | FDI κP=12% |
|---|---:|---:|---:|
| Source-defined U CA/output | −1.454190% | −3.312574% | −2.659600% |
| U NFA/output | −47.292088% | −7.609913% | −10.972641% |
| U goods TB/output | +1.108736% | −2.887788% | −2.047106% |
| World rate | 5.419355% | 5.582015% | 5.582015% |

Independent SciPy DOP853 at relative tolerance 1e−12 reproduces the low-price CA/output **−3.3125743703%**, NFA/output **−7.6099128919%**, rate **5.5820146626%**. A separate Python backward+forward RK4 implementation with h=.1, .05, .025 and .0125 gave low-price year-25 CA/output between −3.312574370510% and −3.312574370263%, with full expected convergence. DOP853 terminal-time T=250 to T=350 changed low-price year-25 CA by about 2e−9 percentage points. These are **independent calculations**, not new Julia interpreter runs.

### 3.4 Crucial accounting caveat in original GAUSS and replicated Julia

The source sets `TBXUp=1−θ WXUp−κ` and `CAXUp=TBXUp+r*(WXUp−VXUp)`. Define aU_code=bU+r nU, bU=1−θwU−κ and nU=wU−vU. Subtract the asset valuation equation from Appendix F's U wealth dynamics: for the Figure 5 specialization κ=0, with xRn=(1−xU)−xRo,

$$\boxed{\dot n_U+g n_U-a_U^{code} =g^n\left[\frac{1}{\theta x_U}-\frac{v_{Ro}x_{Ro}}{x_U}\left(1-\frac{\delta_U}{\delta_R}\right)-v_U\right]-\kappa_P\frac{x_{Rn}}{x_U}.}$$

This intermediation/option-acquisition wedge is **not zero** in the FDI case. Evaluating independently for κP=8.129032%:

| Date | d nU/dt + g nU − aU_code, pp of U output |
|---|---:|
| Impact | +5.693878 |
| Year 5 | +5.124875 |
| Year 25 | +3.551007 |
| Long-run | +1.659128 |

Finite differencing the submitted FDI CSV independently recovers the same nonzero wedge. It is tied to the adjusted intermediation trade flow in the paper's eqs. (42)–(43), not a computational stability problem. The label “Current account” in Figure 5 therefore refers to the **source-calculated goods-TB-plus-interest series**, which does **not** equal d(NFA)/dt once FDI is introduced. Do not silently redefine Figure 5's plotted line to a new accounting measure in a purported direct replication. The narrower conclusion that FDI rents can support a persistent *goods* trade deficit remains sound, but equation (43) must be used for adjusted external accounting.

At the κP=12% upper endpoint, asymptotic U goods TB is zero, not a strict deficit. The strict Proposition 5 requires an interior point. At the κP≈8.129% lower endpoint, the limiting goods TB is about −1.8852% of U output. No long-run quantitative result is inferred from a statement about strict private gains at an equality boundary.

## 4. Figure 6: CES goods, exchange rates and scalar shooting

**Original:** Section IV, Proposition 6, Figure 6 (p.385), the p.384 discussion, Appendix G eqs. (60)–(63), `prog_UR_exch_final.gau`. **Result: continuous-time equilibrium equivalence and independent stable numerical replication for the benchmark; published long-run prose differs.**

### 4.1 Derivation of Julia's differentiated-clearing reduction

The calibrated environment uses θ=.25, gU=gR=.03, γ=.9, σ=4, xU_pre=.7, δU=.12, δR=.08129032258, initial NFA=0, and R wealth initial U investment=.05. The initial pre-shock q=1.17043358978, physical R output 0.36616466950, and λ=P_R/P_U=1.13321457468.

Write s=XU/(XU+q XR), w=WU/XU, and

$$ P_U=[\gamma+(1-\gamma)q^{1-\sigma}]^{1/(1-\sigma)},\quad P_R=[(1-\gamma)+\gamma q^{1-\sigma}]^{1/(1-\sigma)}.$$

Original Appendix G's U-good clearing condition (61):

$$1=\theta\left[\gamma w P_U^{\sigma-1}+(1-\gamma)\left(\frac1{\theta s}-w\right)P_R^{\sigma-1}\right].$$

Appendix G eqs. (60), (62), (63), with ζ=d(log q)/dt, imply

$$\dot w=(r-\theta-g)w+(1-\delta_U),\quad\dot s=-s(1-s)\zeta,\quad r=g+(1-s)\zeta+\theta\bar\delta,\quad\bar\delta=s\delta_U+(1-s)\delta_R.$$

Let A=P_U^(σ−1), B=P_R^(σ−1), H=1/(θs)−w, and e_U=d(log P_U)/d(log q), e_R likewise. These are

$$e_U=\frac{(1-\gamma)q^{1-\sigma}}{\gamma+(1-\gamma)q^{1-\sigma}},\quad e_R=\frac{\gamma q^{1-\sigma}}{(1-\gamma)+\gamma q^{1-\sigma}}.$$

Differentiate the goods clearing equation and substitute the other equations. Algebraically,

$$\boxed{\zeta=-\frac{[\gamma A-(1-\gamma)B]\,[1-\delta_U-\theta(1-\bar\delta)w]}{\gamma Aw[(1-s)+(\sigma-1)e_U]+(1-\gamma)BH[(1-s)+(\sigma-1)e_R]}.}$$

This is the continuous-time limit of GAUSS `solvederiv`, which otherwise solves for a one-step Euler increment using `nlsys`. It applies when its denominator is nonzero, as verified along the benchmark trajectory; it is **not** a general global equilibrium uniqueness theorem.

At impact, world clearing gives s vU+(1−s)vR=1/θ; with a=ARU and b=AUR, U wealth satisfies w=(1−a)vU+b(1−s)vR/s. Hence

$$v_U=\frac{w-b/(\theta s)}{1-a-b},\qquad v_R=\frac{1/\theta-sv_U}{1-s}.$$

Goods-market clearing determines w from a conjectured q0. Both initial asset valuations are therefore functions of **one scalar** q0; no second independent initial valuation unknown is needed. For each candidate q0, integrate the continuous-time system forward, with discount factor D(t)=exp[−θ∫0^t δbar(z) dz], and evaluate

$$v_U(0)=\frac{\delta_U}{s(0)}\int_0^\infty s(t)D(t)dt,\quad v_R(0)=\frac{\delta_R}{1-s(0)}\int_0^\infty [1-s(t)]D(t)dt.$$

Their initial-share-weighted sum is ∫δbar D dt=1/θ if D(∞)=0, proving that either pricing condition implies the other **after** global clearing and the transversality condition are imposed. Julia's scalar bisection plus independent second PV residual check is therefore mathematically justified. Its RK4 integration and 350-year terminal tail are a different, higher-accuracy discretization than GAUSS Euler with a 250-year PV horizon.

### 4.2 Correct world real-interest series

GAUSS calculates U and R real rates by subtracting corresponding CES price inflation from the common *U-good numeraire* rate and plots an output-share weighted mean (original lines 257–266). Julia correctly implements

$$r_U^{real}=r-e_U\zeta,\quad r_R^{real}=r-e_R\zeta,\quad r_{world}^{real}=s r_U^{real}+(1-s)r_R^{real}.$$

At impact the U-good-numeraire rate is **6.302001%**, whereas the **plotted** world real interest rate is **5.660401%**. These must not be confused.

### 4.3 Paper / Julia / independently reconstructed GAUSS-style Euler

The independent Euler reconstruction uses the GAUSS formulas, h=.025, T=250, per-step nonlinear market-clearing roots, and more stringent convergence than the original stopping criterion. It is **not a direct GAUSS run**.

| Statistic | Julia continuous-time RK4 | Independent source-style Euler | Original paper |
|---|---:|---:|---|
| Impact q | 1.036310 | 1.037489 | Figure 6 visually similar |
| Impact λ=P_R/P_U | 1.028934 | 1.029870 | about 9.5% appreciation |
| Impact world real rate | 5.660401% | 5.661598% | about 5.65% |
| Year-25 CA/U output | −1.627294% | −1.588035% | curve near −1.6% |
| Year-26 CA/U output | −1.606635% | −1.566853% | prose “asymptotic” −1.56% |
| Strict long-run CA/U output | −1.469625% | −1.469625% (source limiting formula) | prose −1.56% |
| Strict long-run NFA/U output | −48.987501% | −48.987501% | about −49% |
| Year-25 λ | 1.151604 | 1.151657 | Figure 6 near 1.15 |
| Strict long-run λ | 1.155423 | 1.155423 | Figure 6 near 1.16 |

The *source GAUSS code's own* `caul=g*(wul−vul)` limiting formula evaluates to **−1.469625%**, not −1.56%; this directly establishes the paper-prose/limiting-formula discrepancy. The approximately −1.56% value is reproduced by Euler at the graph’s 26-year endpoint. The exact asymptotic λ change from pre-shock is **+1.959788%** (ordinary percent), whereas the year-25 change is **+1.622771%**. The paper’s “another 1.6 percent” is closely approximated by the finite-horizon value. The history of the paper’s final plotting run is unavailable, so the source of prose wording cannot be proven conclusively.

At impact, λ falls 9.202197% by the *ordinary proportional change*, or 9.653509 log-percentage points. The paper's 9.5% is closer to the log version. Year 25 λ rises 1.622771% from the pre-shock λ, and the strict long-run value rises 1.959788%. Use a consistent definition in any revised presentation.

### 4.4 Numerical precision, convergence and accounting checks

Independent Python implementation of the Julia-equivalent scalar shooting and RK4 path, with T=350:

| RK4 h | Initial U price reconstruction minus PV-required price | Year-25 CA/U output |
|---:|---:|---:|
| .0500 | +9.20e−8 | −1.6272940011% |
| .0250 | +2.30e−8 | −1.6272939887% |
| .0125 | +5.75e−9 | −1.6272939856% |

The roughly 4× reduction when halving h in the *backward price reconstruction* error reflects the log-geometric midpoint approximation; the forward equilibrium solver is RK4. T=200 and T=350 agree on year-25 CA to the reported ten decimal places, after the stationary-tail correction. Julia metadata report initial U/R PV residuals −1.21e−13/+4.62e−13 and maximum goods-market residual 2.68e−14. Independent backward valuation reconstruction at h=.025 gives ~+2.3e−8 U and −6.06e−8 R initial price discrepancies, economically immaterial. The exported CSV satisfies the continuous-time accounting identity dnU/dt + g nU − aU=0 with maximum interior centered-difference residual about 1.1e−7. The Julia code checks world asset clearing and the exact source definition of CA.

## 5. Reproducibility and Julia quality review

The three files use named parameter structs and small functions, explicit Okabe–Ito HEX colors for CairoMakie graphs and SummaryTables.jl for HTML/LaTeX. User Windows logs confirm real package loading, graph/table generation and passed internal assertions. The checked docstrings immediately precede declared types/functions; a duplicated prose sentence in a Figure 6 docstring is cosmetic only. Figure 4 correctly avoids any ODE solver in favor of its exact wealth solution. Figure 5 uses backward/forward RK4 and Figure 6 uses safeguarded scalar bisection plus RK4, removing unnecessary per-step nonlinear solves. The production code is not modified merely for style.

**Current environment limitations:** no Julia executable here, so no new Julia `Meta.parseall`, package loading, profiling, `@code_warntype`, allocation benchmarks, local Julia h-sweep or end-to-end re-run was performed. Likewise, no GAUSS interpreter run or literal interpreter-to-interpreter output comparison was possible. Package lockfile/UUID/version compatibility and CI on the current GitHub branch were not confirmed. The user’s earlier Windows logs *are* evidence of successful Julia runs, not of GAUSS-exact matching.

## 6. Certification matrix and discrepancy register

| Standard | Figure 4 | Figure 5 | Figure 6 |
|---|---|---|---|
| Continuous-time equation equivalence | PASS (closed form) | PASS for state/price equations; source CA caveat | PASS for benchmark scalar reduction |
| Julia execution | PASS by supplied Windows log | PASS by supplied Windows log | PASS by supplied Windows log |
| Independent numerical precision | PASS by analytic solution | PASS, DOP853 and h/T sweeps | PASS, RK4 h/T sweeps and PV residuals |
| Qualitative figures | PASS | PASS | PASS |
| Quantitative paper/GAUSS algebra | PASS | PASS using **computed** κP, not printed low-price label | PASS at continuous-time precision; prose limit differs |
| Exact GAUSS-interpreter match | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED |
| Remaining concern | GAUSS runtime absent | κP legend and boundary assumptions; CA accounting | Asymptotic prose/finite-horizon distinction; differing numerical integrators |

**D1—Figure 5 price:** GAUSS lines 191–192 compute 8.129%, hardcoded legend and paper footnote 30 show 5%. Difference 3.129 pp. Keep source-computed default; label properly.

**D2—Figure 5 strict assumption:** Both graphed κP endpoints satisfy equality rather than the strict Assumption 6. Keep as explicit boundary comparisons.

**D3—Figure 5 accounting:** Source-defined CA=goods TB+r NFA differs from d(NFA)/dt by +5.694 pp of U output on impact, +3.551 pp at year 25 under the low endpoint. Report the adjusted-accounting wedge; retain source figure series.

**D4—Figure 6 CA limit:** Paper prose −1.56% versus source limiting formula −1.469625%; independent Euler year 26 yields −1.566853%. Do not silently substitute a finite horizon for infinity.

**D5—Figure 6 exchange-rate limit:** Paper ~1.6% is close to year-25 +1.622771%, not strict long-run +1.959788%. Disclose the horizon and percentage convention.

**D6—Figure 6 solver:** GAUSS first-order Euler, 250-year truncated PV; Julia RK4, 350-year tail-corrected PV. Quantitative transient discrepancy is documented; this is not a model-identity failure.

**D7—Execution and delivery:** direct GAUSS interpreter execution and new Julia run not available. Attempts to write this audit file to GitHub through the connected integration returned **HTTP 403 `Resource not accessible by integration`**. The report is supplied as a local Markdown artifact instead; the live GitHub repo was not changed.

## 7. Final decision

Keep `figure4_asset_supply.jl`, `figure5_fdi.jl` and `figure6_exchange_rates.jl` unchanged. They provide a defensible, economically faithful **continuous-time** reimplementation. Disclose the original-source Figure 5 parameter/FDI accounting differences and the Figure 6 finite-time/long-run mismatch whenever presenting the replication. Do not claim complete GAUSS-exact certification without an actual source-runtime comparison. No numerical figure lines were adjusted to force agreement with inconsistent publication prose.

## 10. Addendum — Published κP=5% sensitivity and dual-current-account output (2026-11-12)

**Purpose.** Report both published legend values and the numbers actually used in GAUSS, without relabeling the source-generated curves. Keep the original `figure5_fdi.jl` and its outputs unchanged. The supplementary Julia program `figure5_dual_accounting_sensitivity.jl` (placed alongside `figure5_fdi.jl`) computes κP=5%, 8.129032%, 9.5% and 12% and separates GAUSS's plotted CA from the Appendix-F-consistent derivative of net foreign assets. The independently computed numerical reference CSV is `figure5_dual_accounting_reference.csv` (Python/SciPy DOP853, not Julia execution).

### Distinct economic status of the two ranges

At the original δU=0.24, δR=0.162580645, gn=0.03, θ=0.25, κ=0, the private-gains condition is **0.081290323 < κP < 0.12**. The alternative **0.05 < κP < 0.12** is not a valid private-gains region without changing an underlying primitive. At κP=5%, the required R-seller reservation condition fails, so the 5% result is a *mechanical imposed-FDI sensitivity*, not a voluntary equilibrium. To move the lower threshold to 5% by changing δR alone, one needs δR=0.10 (boundary), corresponding to a 49.495% R-asset crash rather than the original 25%. The GAUSS 8.129% and 12% endpoints themselves are indifference boundaries; κP=9.5% is strictly interior.

### Accounting identity

For the published Figure 5 specialization (κ=0), with old-R per-tree-value-to-output ratio vRo, R's initial world output share xR, U's share xU, and xRn the share of output from U-owned *new* R trees,

$$
CA^{code}_U / X_U=TB_U/X_U+r (W_U-V_U)/X_U,
$$

$$
CA^{adjusted}_U / X_U = [\dot W_U-\dot V_U]/X_U
= CA^{code}_U/X_U+
\underbrace{g^n\frac{\delta_U}{\delta_R}v_{Ro}\frac{x_R}{x_U}
-\kappa_P\frac{x_{Rn}}{x_U}}_{\text{FDI option/transaction adjustment}}.
$$

The supplementary program checks this formula against derivatives obtained independently from the wealth and U-asset ODEs. These are **post-shock continuous flows**, excluding the unanticipated discrete valuation at t=0. In steady state, the adjustment reduces to the intermediation-service net export defined in the original paper's equations (42)–(43).

### New quantitative comparison (percent of U output, except wedge in pp)

| κP | GAUSS CA at year 25 | Adjusted CA at year 25 | Adjustment | NFA at year 25 | Goods TB at year 25 | Assumption 6 |
|---|---:|---:|---:|---:|---:|---|
| 5% | −3.840395 | +0.418176 | +4.258571 | −4.891708 | −3.567339 | Invalid, imposed FDI |
| 8.129032% | −3.312574 | +0.238432 | +3.551007 | −7.609913 | −2.887788 | Lower boundary |
| 9.5% | −3.081313 | +0.159679 | +3.240992 | −8.800879 | −2.590046 | Strict interior |
| 12% | −2.659600 | +0.016070 | +2.675670 | −10.972641 | −2.047106 | Upper boundary |

At impact, all four imposed-FDI scenarios have the same CA_code=−10.105039%, corrected CA=−4.411161% and NFA=−3.061224%. The initial acquisition-price payment is zero because there are no new R trees yet. Under an *actual voluntary-investment equilibrium* at κP=5%, one must endogenize the seller's rejection rather than assume FDI occurs.

### Figure 6 practical interpretation

The differences between the paper's −1.56% and the independently computed GAUSS-style year-26 −1.566853%, and between 1.6% and year-25 +1.622771% real depreciation, are quantitatively tiny. They should be recorded as a **finite-horizon versus strict-limit reporting distinction**, rather than used as grounds to change the accepted core Julia solver or delay replication. The mathematical strict limit remains a distinct definition, not a competing calibrated result.

**Execution caveat:** The supplementary Julia program was *not* syntax-parsed or run in the present environment because Julia is unavailable; numerical reference values and the CA identity were independently evaluated with SciPy. Previous runs of the original three Julia programs remain evidenced by user-supplied Windows logs. No changes to original Figure 4/5/6 code are required.
