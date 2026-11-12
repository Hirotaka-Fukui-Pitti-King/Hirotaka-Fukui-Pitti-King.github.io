# Caballero, Farhi, and Gourinchas (2008) — Julia Replication: Final Mathematical, Accounting, and Reproducibility Audit

**Audit date:** 2026-11-12. **Paper:** Ricardo J. Caballero, Emmanuel Farhi, Pierre-Olivier Gourinchas (2008), “An Equilibrium Model of ‘Global Imbalances’ and Low Interest Rates,” *American Economic Review* 98(1): 358–393, [DOI 10.1257/aer.98.1.358](https://doi.org/10.1257/aer.98.1.358).

**Executive determination.** Adopt the existing four Julia simulations as **continuous-time, source-equation-faithful reconstructions with explicitly disclosed source discrepancies**. Figure 4 is analytic; Figures 5 and 6 independently pass numerical checks. All four were executed successfully according to *user-supplied Windows PowerShell logs*. The auditing environment has **no Julia or GAUSS executable**, so it has neither rerun the code nor certified interpreter-to-interpreter identity. Figure 5's plotted `CA` must not be conflated with the model's NFA accumulation rate, much less with a statistically measured balance-of-payments current account.

**Scope:** Original Figure 4–6, the supplementary Figure 5 four-price/accounting comparison, source GAUSS/Julia equations and output checks, build/documentation. Not an empirical reproduction of Figures 1–3 or Tables 2–3. The first three Julia files are present in GitHub `main`; the fourth is currently supplied as an independent attachment and was *not* on GitHub at the date of review. The versions of the source Python verification helpers used previously are not intended as part of the published Julia project.

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

### 3.4 Exact derivation of the FDI accounting wedge (levels)

**Original source locations:** Section I Definition 1 (p.364), Section III.B equations (41)–(43) (pp.379–380), Appendix E (pp.389–390), Appendix F (pp.390–391), original GAUSS lines 201–216. This distinction was already present in GAUSS; it is **not** a Julia numerical error.

Let \(V_t^U\) be the value of **U-origin trees**, \(W_t^U\) the wealth of U residents including new R trees owned by U, and \(NA_t^U=W_t^U-V_t^U\). Write \(I_t^U\) and \(I_t^R\) for planting investment in U and new R trees undertaken by U, respectively, and \(P_t=\kappa_P X_t^{Rn}\) for the acquisition payment to R sellers. Appendix F gives

$$
\dot W_t^U=(r_t-\theta)W_t^U+(1-\delta^U)X_t^U
 +g^n V_t^U+g^n N_t^R v_t^{Rn}-P_t-I_t^U-I_t^R.
$$

The asset valuation equation for U trees is

$$r_tV_t^U=\delta^U X_t^U+\dot V_t^U-g^n V_t^U,$$

or \(\dot V_t^U=r_t V_t^U-\delta^U X_t^U+g^n V_t^U\). Subtracting **exactly**, without a numerical approximation, gives

$$
\boxed{\dot{NA}_t^U
= \underbrace{(X_t^U-\theta W_t^U-I_t^U)}_{TB_t^U}
 +r_t NA_t^U
 +\underbrace{\left(g^n N_t^R v_t^{Rn}-P_t-I_t^R\right)}_{F_t^{\mathrm{FDI}}}.}
\tag{A1}
$$

With \(\kappa=0\) in Figure 5, \(I_t^U=I_t^R=0\); thus \(F_t^{\mathrm{FDI}}=g^nN_t^R v_t^{Rn}-\kappa_P X_t^{Rn}\). Original GAUSS constructs a *plotted* series `CAXUp=TBXUp+rp*(WXUp−VXUp)` and therefore omits \(F_t^{\mathrm{FDI}}\). The supplementary Julia script reports separately:

$$
CA_t^{\mathrm{GAUSS,model}}=TB_t^U+r_tNA_t^U,\qquad
\underbrace{A_t^{NFA}}_{\text{NFA accumulation rate}}=\dot{NA}_t^U
=CA_t^{\mathrm{GAUSS,model}}+F_t^{\mathrm{FDI}}.
\tag{A2}
$$

In the code's normalization, \(n_t=(W_t^U-V_t^U)/X_t^U\), \(x_{Ro}(t)=(1-x^U)e^{-g^nt}\), \(x_{Rn}(t)=(1-x^U)-x_{Ro}(t)\), and \(v_{Ro}(t)=V_t^{Ro}/X_t^{Ro}\). Appendix F implies \(v_t^{Rn}=(\delta^U/\delta^R)v_t^{Ro}\) per unit tree output. Consequently

$$
\boxed{
\frac{A_t^{NFA}-CA_t^{\mathrm{GAUSS,model}}}{X_t^U}
=g^n\frac{\delta^U}{\delta^R}\frac{x^R}{x^U}v_{Ro}(t)
 -\kappa_P\frac{x_{Rn}(t)}{x^U}
=\dot n_t+g n_t-\frac{CA_t^{\mathrm{GAUSS,model}}}{X_t^U}.}
\tag{A3}
$$

**Long run:** old R trees disappear as an output share, so \(N^R v^{Rn}\to V^R\) and \(X^{Rn}\to X^R\). With general \(\kappa\), the additional investment \(I^R\to\kappa X^R\); hence (A1) becomes exactly the original equation (42): \(\widehat{TB}^U=TB^U+g^nV^R-(\kappa+\kappa_P)X^R\). Equation (43) states that the discounted future path of this **adjusted** balance backs the U net foreign asset position. This derivation establishes why the difference exists; it does not rely on reading slopes from a graph.

**Statistical accounting — interpret with care.** Section I Definition 1 calls \(\dot W-\dot V\) “current account” in this theoretical environment. The GAUSS plot's `TB+r*NFA` uses total asset returns and does not separately identify investment *income* from market *holding gains*. Under international balance-of-payments conventions, the actual statistical current account is goods/services/primary/secondary income; net financial transactions, the capital account, valuation changes and other changes also enter the decomposition of the net international investment position (IIP). In general,

$$\Delta \mathrm{NIIP} = \mathrm{CA} + \mathrm{capital\ account} + \mathrm{valuation\ gains}+\mathrm{other\ volume\ changes},$$

up to sign/statistical conventions and errors. The Bank of Japan's [BOP/IIP accounting FAQ](https://www.boj.or.jp/en/statistics/outline/exp/faqbpsm6.htm) and [ECB statistical explanation](https://www.ecb.europa.eu/stats/balance_of_payments_and_external/balance_of_payments/html/index.en.html) document these distinctions. These standards are **external accounting references**, not claims made by CFG (2008).

The paper explicitly *suggests* regarding the FDI intermediation term as “nontraditional” net service exports, but it does not specify complete statistical treatment for the creation of new assets, transfer of investment options, or valuation versus transaction changes. The sale of an option/right could imply a financial-asset acquisition or nonproduced nonfinancial asset transfer depending on its legal and economic nature; the paper does **not** provide enough detail to make a unique BPM6 code assignment. Therefore **do not label** \(\dot{NA}^U\) unconditionally as the measured statistical current account. Preserve the old CSV column name `ca_adjusted` for backwards compatibility, but in README and audit call it **“model NFA accumulation rate (previously CA_adjusted)”**. The unanticipated asset-price jump at \(t=0\) is excluded from the post-shock flow equations and must not be classified as a current-period export.

**Finding:** The original source figure can coexist with the Appendix-F-consistent NFA-accumulation series without modifying any model. The FDI rent financing a long-run **goods trade deficit** is the robust claim of Proposition 5, not the assumption that source `CAXUp` must equal the statistical current account or \(\dot{NA}\).

### 3.5 Four acquisition prices, participant incentives and quantitative impact

GAUSS (lines 191–192) uses \(\kappa_P\in[0.081290323,0.12]\) with four equally spaced evaluations, but the published legend/footnote says 5% instead of the computed 8.129032%. Assumption 6 is **strict**:

$$
\boxed{0.081290323<\kappa_P<0.12}
$$

for the fixed calibration \(\delta^U=.24,\ \delta^R=.162580645,\ g^n=.03,\ \theta=.25,\ g^z=\kappa=0\). This is *not* interchangeable with \(0.05<\kappa_P<0.12\). The 5% seller rejects a voluntary sale; the experiment is explicitly **imposed FDI**. The 8.129% and 12% price extremes are **indifference boundaries** (R and U, respectively), not strict Proposition 5 cases. The 9.5% case is strictly admissible.

**Year 25: percentages of U output, except the wedge in percentage points.** Comparison compiled from submitted user logs and an independently computed Python/SciPy reference CSV:

| Acquisition price | `CA_GAUSS` | NFA accumulation \(\dot{NA}\) | FDI wedge | NFA stock / output | Goods TB | Economic status |
|---|---:|---:|---:|---:|---:|---|
| 5% | −3.840395% | +0.418176% | +4.258571 pp | −4.891708% | −3.567339% | Imposed FDI, R seller rejects |
| 8.129032% | −3.312574% | +0.238432% | +3.551007 pp | −7.609913% | −2.887788% | R indifferent, source endpoint |
| 9.5% | −3.081313% | +0.159679% | +3.240992 pp | −8.800879% | −2.590046% | Strict bilateral gains |
| 12% | −2.659600% | +0.016070% | +2.675670 pp | −10.972641% | −2.047106% | U indifferent, source endpoint |

At impact (continuous flows at \(0^+\), not the asset-price jump) all four imposed-FDI paths have source CA **−10.105039%**, NFA-accumulation rate **−4.411161%**, NFA stock **−3.061224%**, and wedge **+5.693878 pp**. The wedge initially does not depend on \(\kappa_P\) because the price payment for the *new R trees* is initially zero.

The submitted supplementary execution log says `Appendix-F accounting checks passed`; its notes show a maximum internal algebraic identity residual about \(2.1\times10^{-15}\). A newly executed **independent Python CSV audit** verified the equality `CA_GAUSS + wedge = NFA_accumulation` to \(2.7\times10^{-15}\) **percentage points**, and the four-case derivative identity on interior time points to about \(3.1\times10^{-5}\) pp using discrete centered differences. On overlapping time grids, the GAUSS-computed 8.129% and 12% columns from the base Figure 5 Julia CSV and the Python independent sensitivity reference agree to at most \(5.9\times10^{-9}\) pp for NFA and \(3.4\times10^{-10}\) pp for source CA. This is excellent independent agreement, **not** a new run of the supplementary Julia CSV (the Julia-generated supplementary CSV was not supplied for inspection).

The supplementary plot matches these qualitative paths and uses four distinct Okabe–Ito colors and separate line types. No economic conclusion should be drawn from the 5% case as a *voluntary* FDI market equilibrium unless the contract/bargaining model is changed.

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

## 5. Reproducibility, source/file review and Julia QA

**Executed evidence (not newly rerun here).** The user-provided PowerShell logs show all four Julia programs completed successfully:

| Script | Reported self-test / output |
|---|---|
| `figure4_asset_supply.jl` | `Structural checks passed`, original four plotted panels and SummaryTables |
| `figure5_fdi.jl` | `Structural/numerical checks passed`, original four panels and SummaryTables |
| `figure6_exchange_rates.jl` | `Structural/numerical checks passed`, original five panels and SummaryTables |
| `figure5_dual_accounting_sensitivity.jl` | `Appendix-F accounting checks passed`, added four-panel figure and output directory |

**Repository inventory checked live via GitHub.** `main` contains the three base `.jl` files (blob SHAs listed in §1), but **not** the fourth supplementary `.jl`, `README.md`, or `REPLICATION_AUDIT.md` at the specified folder. `mysite/Project.toml` contains `SummaryTables` but **omits `CairoMakie`**. Therefore the root Julia project is *not yet fully self-contained*. The README provides a PowerShell `Pkg.add(string(:CairoMakie))` command and explains that a checked `Manifest.toml`/version capture remains necessary for durable cross-machine reproducibility. Do not pretend exact Julia/package versions or standard-library UUIDs were verified.

**Available user artifacts examined:** Figure 4/5/6 PNG and PDF, base Figure 4/5/6 SummaryTables HTML/LaTeX files, base Figure 4/5/6 CSVs and Figure 5/6 metadata, the supplementary Figure 5 PNG/PDF and its `notes.txt`, plus an independently computed supplementary reference CSV. The user-generated supplementary HTML, LaTeX and CSV were **not supplied**; those expected outputs have been inferred from program source but not inspected. The four corresponding time-series/reference CSV files contain respectively 1,080, 1,081, 1,081 and 1,041 rows, and no missing data or repeated timestamps. The independent CSV audit found maximum goods-clearing residual in Figure 6 **2.68×10⁻¹⁴**; the model-equation finite-difference residual is **5.05×10⁻⁵ pp** in Figure 4 and **1.09×10⁻⁵ pp** in Figure 6 on an interior grid (differences of already rounded CSV rows and finite-difference truncation, not exact Julia source tests).

**Numerical stability:** Figure 4 uses the exact solution; Figure 5 uses backward/forward RK4, with independent DOP853 and time-step/horizon variation checks documented in §3; Figure 6 reduces GAUSS per-step nonlinear market clearing to a differentiated analytic law and one-dimensional shooting plus RK4, with independent PV/goods residual checks in §4. The plotted output agrees with the qualitative profile of published figures, but the original GAUSS interpreter was **not** executed. The source code's nontrivial structure is isolated in named functions and typed parameter structs; a static inspection found no orphan or consecutive docstrings and no obvious misuse of reserved Julia identifiers. Runtime type stability/allocations, `@code_warntype`, and compilation via `Meta.parseall` **have not been tested in the current environment** (Julia executable absent). The previous successful user executions establish practical executability of the *specific versions they ran*, not current CI success.

**Plot / table presentation:** Base Figure 4 has four panels (CA, NFA, rate, portfolio); base Figure 5 has four (source CA, NFA, rate, goods TB); Figure 6 has five (CA, NFA, world real rate, portfolio, RER); supplemental Figure 5 has four (source CA, model NFA-accumulation rate, NFA, wedge). All provided plots were visually reviewed and use explicit Okabe–Ito colors and legible units; Figure 5 sensitivities correctly label 5%, 8.129%, 9.5%, 12%. Base SummaryTables outputs were cross-read against CSV and reported precision: displayed numbers agree to the last reported rounding digit. Original finite-horizon and strict long-run dashed/reference lines are distinctly presented, and the Figure 6 world real rate is correctly not the U-good numeraire interest rate. **One justified presentation improvement** in future regenerated supplemental figures is to rename panel B from `Adjusted CA = d(NFA)/dt` to `NFA accumulation rate = d(NFA)/dt`; this is a clarity change only, not a numerical correction. Keep old column names for compatibility or provide both aliases.

**Remaining work outside this run:** install/pin dependencies in the repository environment, place the fourth script and final documentation into GitHub, independently rerun all four in a clean Julia environment, optionally run the original GAUSS executable for literal source-run numerical differences, and confirm any CI result. Do not block publication of the working reproductions on a merely cosmetic Figure 6 asymptotic-versus-year-25 difference.

## 6. Final certification

PASS means the specific mathematical or numerical fact has been checked; `USER LOG` means user-supplied evidence of an execution; `PARTIAL` means additional work remains.

| Criterion | Figure 4 | Figure 5 | Figure 6 | Figure 5 supplement |
|---|---|---|---|---|
| Mathematical-equation equivalence | PASS (analytic) | PASS for continuous-time state/asset equations | PASS for calibrated continuous-time scalar reduction | PASS for derivative identity and fixed-primitive sensitivity |
| Source accounting equivalence | PASS | **SOURCE DIFFERENCE:** `CA_GAUSS` is not \(\dot{NA}\) with FDI | PASS for model-defined CA | PASS for the two separately named series, **not** BPM-current-account certification |
| Julia execution | USER LOG PASS | USER LOG PASS | USER LOG PASS | USER LOG PASS |
| Independent numeric verification | PASS | PASS DOP853/RK4 variations | PASS h/T/PV/clearing | PASS Python CSV checks and derivative algebra |
| Qualitative and numerical figure replication | PASS | PASS with source **8.129%** and **12%** | PASS up to integrator/horizon; source prose differs | PASS, supplemental—not part of published figure |
| Direct exact GAUSS interpreter comparison | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | N/A (new analysis) |
| Clean-environment dependency lock / CI | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED | NOT VERIFIED |

### Discrepancy register

| ID | Location | Cause, magnitude, effect, and disposition |
|---|---|---|
| D1 | Figure 5 caption/footnote 30 vs GAUSS 191–192 | Printed 5%, computed **8.129032%**. Keep source-numeric default; retain 5% separately as imposed-FDI experiment. |
| D2 | Assumption 6 / Figure 5 endpoints | Only \(8.129\%<\kappa_P<12\%\) strictly satisfies bilateral gains; endpoints are indifference. Added 9.5% interior result. |
| D3 | Figure 5 `CAXUp` and Appendix F | Source CA = goods TB + r NFA; **model NFA accumulation** differs by +3.551 pp at year 25 for 8.129%. Keep two series; no blanket BPM6 classification. |
| D4 | Figure 6 prose p.384 / source `caul` | “Long run” −1.56% vs analytic **−1.469625%**; source-style Euler year 26 **−1.566853%**. Finite-horizon explanation strongly supported. |
| D5 | Figure 6 prose p.384 / RER | ~1.6% depreciation close to year-25 **+1.622771%**, strict limit **+1.959788%**. No substantive sign/mechanism change. |
| D6 | Figure 6 numerical algorithm | GAUSS Euler/250-year PV vs Julia RK4/350-year plus tail; insignificant qualitative difference but not exact interpreter identity. |
| D7 | GitHub contents/dependencies | Main has only three base Julia scripts; supplemental and docs absent; `CairoMakie` missing from root `Project.toml`. Repo integration required. |
| D8 | Execution/outputs | Local auditor has no Julia or GAUSS executable; supplemental generated tables/CSV not attached; CI/version lock not verified. |

### Adoption decision

**Keep the four existing Julia numerical programs unchanged.** Report both GAUSS and Appendix-F NFA accumulation series, retain the 5% imposed-FDI and 9.5% interior sensitivity separately, and disclose that `CA_adjusted` is a *model accumulation* label rather than a statistical CA. The Figure 6 numerical deviation from printed “long-run” values is economically immaterial for the central mechanism but must remain explicitly traceable. The audit **does not certify exact GAUSS runtime equivalence or a fully locked clean-environment reproduction**.

**GitHub integration:** At document creation the connected GitHub integration had not established a successful write; do not infer a commit SHA. Files are provided individually for manual placement when write access is unavailable. No Julia source refactor is required on present evidence.
