module GoldilocksFiscalSpace

using LinearAlgebra
using Printf
using SummaryTables
using ..GoldilocksCore
using ..GoldilocksHouseholds
using ..GoldilocksCalibration
using ..GoldilocksDeficitDebt

export FIGURE9_SCENARIOS, FIGURE9_LABELS, FiscalScenario,
       official_figure9_reference, figure9_scenarios,
       figure9_curve, run_figure9_curves, scenario_maxima,
       write_figure9_curve, write_figure9_calibration,
       make_figure9_tables, compare_figure9_python,
       local_zlb_debt_response

# Source: code/interest_rate_comparison_JP_US.py, Figure 9, AER p. 4284.
const FIGURE9_SCENARIOS = (:us, :us_high_beta, :us_high_beta_low_phi,
                           :us_high_beta_low_phi_zlb, :jp)
const FIGURE9_LABELS = (
    "U.S.",
    "U.S., low ρ",
    "U.S., low ρ, low φ̃",
    "U.S., low ρ, low φ̃, ZLB",
    "Japan",
)

# These are INDEPENDENTLY computed by running the original author's Python
# calibration, not read from the Figure 9 image and not used as Julia inputs.
const OFFICIAL_FIGURE9_BENCHMARK = Dict{Symbol,NamedTuple}(
    :us => (beta=0.8557697856259636, slope=0.2861891207062559,
            Y=1.0, rho=0.15575388103753046),
    :us_high_beta => (beta=0.8791623020412067, slope=0.2861891207062559,
                      Y=0.9421451518540027, rho=0.12878575438114084),
    :us_high_beta_low_phi => (beta=0.9679354896752341, slope=0.08592936559253667,
                              Y=0.9421451518540009, rho=0.032589836823782),
    :us_high_beta_low_phi_zlb => (beta=0.9632338254121301, slope=0.10878973142632442,
                                  Y=0.9122204838893797, rho=0.03745908729904236),
    :jp => (beta=0.9591746054283945, slope=0.07987045057765649,
            Y=1.0, rho=0.04168215036033463),
)

"""Author-Python verification values; never substitute these for solved counterfactuals."""
official_figure9_reference(key::Symbol) = OFFICIAL_FIGURE9_BENCHMARK[key]

"""One fully re-solved Figure 9 counterfactual and its economic diagnostics."""
struct FiscalScenario{S,H,C,E}
    key::Symbol
    mode::Symbol
    setup::S
    household::H
    core::C
    anchor::E
    phi_reference::Float64
    phi_local::Float64
    inner_residual::Float64
    outer_residual::Float64
end

"""Rebuild a setup with a new state, keeping the ORIGINAL US household targets."""
function _replace_setup(s::CalibrationSetup, core::CoreParams, state::NamedTuple)
    return CalibrationSetup(s.country, s.periods, core, state,
                            s.phi_target, s.top_10_share,
                            s.saver_assets, s.saver_consumption,
                            s.lambda_bu, s.beta_initial, s.ha_rho)
end

"""Replace only the five Figure 9 Scenario-4 source fields: r, B, piw, growth, kappa.

`r_ante`, `B` and `piw` live in the state. The production and monetary
real-growth inputs must BOTH change because they use the same Python key
`real_growth`. U.S. technology, public spending, wealth target, labor shifter,
and HTM share remain U.S. quantities.
"""
function _scenario4_macrocore(us::CoreParams, jp::CoreParams)
    p = us.production
    p_new = ProductionParams(; alpha=p.alpha, tfp=p.tfp,
             fixed_cost=p.fixed_cost, markup=p.markup,
             delta_k=p.delta_k, eps_I=p.eps_I,
             real_growth=jp.monetary.real_growth,
             risk_premium=p.risk_premium)
    m = us.monetary
    m_new = MonetaryParams(; pi_trend=m.pi_trend,
             real_growth=jp.monetary.real_growth,
             phi_pi=m.phi_pi, real_rate=m.real_rate,
             i_shock=m.i_shock, zlb=true, i_zlb=m.i_zlb)
    w = us.wage
    w_new = WageParams(; kappa=jp.wage.kappa,
             beta=w.beta, survival=w.survival,
             labor_shifter=w.labor_shifter, lag=w.lag,
             forward=w.forward, frisch=w.frisch)
    return CoreParams(; production=p_new, wage=w_new,
                       monetary=m_new, fiscal=us.fiscal,
                       mu=us.mu, delta_b=us.delta_b)
end

"""BU-log household with the baseline U.S. A_ss and lambda_bu normalization."""
function _counterfactual_household(s_us::CalibrationSetup,
                                   beta::Real, slope::Real)
    return BULog(; beta, eis=1.0, vp_slope=slope,
                  A_ss=s_us.saver_assets, lambda_bu=s_us.lambda_bu)
end

"""Evaluate one source-style nested counterfactual, without solving a root.

This is deliberately separate from the solver so the residuals and equations
can be audited directly, including the nonzero `labor_foc` in Scenario 4.
"""
function _trial(us::CalibrationSetup, macro_core::CoreParams,
                state_anchor::NamedTuple,
                beta::Real, Y::Real, slope::Real)
    state = (; Y=Float64(Y), B=Float64(state_anchor.B),
              r_ante=Float64(state_anchor.r_ante),
              piw=Float64(state_anchor.piw))
    hh = _counterfactual_household(us, beta, slope)
    setup0 = _replace_setup(us, macro_core, state)
    core = GoldilocksDeficitDebt._household_core(setup0, hh)
    setup = _replace_setup(us, core, state)
    eq = evaluate_steady_state(state, core, hh)
    return (; setup, household=hh, core, eq)
end

"""Inner 2-D source root in (beta,Y), with explicit asset and labor/wage targets."""
function _solve_beta_output(us::CalibrationSetup, macro_core::CoreParams,
                            state_anchor::NamedTuple,
                            slope::Real, beta_guess::Real, Y_guess::Real;
                            binding::Bool=false, tol::Real=3e-10)
    slope > 0 || throw(DomainError(slope, "positive convenience-yield slope required"))
    inner_residual = z -> begin
        trial = _trial(us, macro_core, state_anchor, z[1], z[2], slope)
        eq = trial.eq
        [eq.asset_mkt, binding ? eq.piw_cond : eq.labor_foc]
    end
    result = GoldilocksDeficitDebt._solve_residual(
        inner_residual, [Float64(beta_guess), Float64(Y_guess)];
        tol=tol, maxiter=80)
    solution = _trial(us, macro_core, state_anchor,
                      result.x[1], result.x[2], slope)
    resid = norm(inner_residual(result.x), Inf)
    resid <= max(1e-8, 5tol) || error("counterfactual inner root residual=$resid")
    return merge(solution, (; inner_residual=resid))
end

"""Evaluate the source GE sensitivity at any solved counterfactual.

The original `auxfs.GE_elasticity` sets `zlb=0` for this experiment and
holds the anchor labor FOC at its initial *level*, not at zero. This matters
for Scenario 4, where the baseline labor FOC is nonzero.
"""
function _phi_at_solution(solution)
    return ge_elasticity_finite(solution.setup, solution.household;
                                h=-1e-5, tol=2e-8).phi
end

"""Safeguarded bracketed scalar root; no alternative economic calibration rule."""
function _bracket_phi(f, a::Real, b::Real; ftol=1e-8, xtol=5e-12,
                      maxiter::Int=70)
    lo, hi = min(Float64(a),Float64(b)), max(Float64(a),Float64(b))
    lo > 0 && lo < hi || throw(ArgumentError("invalid positive slope bracket"))
    fl, fh = f(lo), f(hi)
    all(isfinite,(fl,fh)) || error("nonfinite phi residual at slope endpoints")
    abs(fl)<ftol && return (;x=lo,residual=fl)
    abs(fh)<ftol && return (;x=hi,residual=fh)
    signbit(fl)!=signbit(fh) ||
        error("source slope bracket does not straddle GE-phi target: ($fl,$fh)")
    bestx,bestf = abs(fl)<abs(fh) ? (lo,fl) : (hi,fh)
    for _ in 1:maxiter
        # Illinois/Brent-like safeguarded interpolation, with a bisection
        # fallback if the secant guess is too close to an endpoint.
        mid = (lo+hi)/2
        trial = (lo*fh-hi*fl)/(fh-fl)
        if !isfinite(trial) || trial<=lo+0.05*(hi-lo) ||
                               trial>=hi-0.05*(hi-lo)
            trial=mid
        end
        ftrial=f(trial)
        isfinite(ftrial) || error("nonfinite phi residual in outer root")
        if abs(ftrial)<abs(bestf)
            bestx,bestf=trial,ftrial
        end
        abs(ftrial)<ftol && return (;x=trial,residual=ftrial)
        if signbit(ftrial)==signbit(fl)
            lo,fl=trial,ftrial
        else
            hi,fh=trial,ftrial
        end
        (hi-lo)<xtol && break
    end
    abs(bestf)<1e-6 || error("counterfactual outer slope root failed; residual=$bestf")
    return (;x=bestx,residual=bestf)
end

"""Construct the five original Figure 9 counterfactuals.

`:official` uses the author's saved U.S./Japan *household parameters* as
starting economic inputs, then independently re-solves Scenarios 2–4. No
saved counterfactual parameters or deficit-debt points are fed into the model.
`:native` instead uses the Julia-calibrated U.S./Japan BU-log baselines.
"""
function figure9_scenarios(; mode::Symbol=:official,
                            inner_tol::Real=3e-10, phi_tol::Real=1e-8)
    mode in (:official,:native) || throw(ArgumentError("Figure 9 mode must be :official or :native"))
    us = figure8_calibration(:us,:bu_log;mode)
    jp = figure8_calibration(:jp,:bu_log;mode)
    us_setup, jp_setup = us.setup, jp.setup
    phi_ref = ge_elasticity_finite(us_setup,us.household;
                                   h=-1e-5,tol=2e-8).phi
    result = Dict{Symbol,FiscalScenario}()
    # Original U.S. and Japan, with their own distinct setup/core/calibration.
    result[:us] = FiscalScenario(:us,mode,us_setup,us.household,us.core,
                       us.anchor,phi_ref,phi_ref,
                       abs(us.anchor.asset_mkt),0.0)
    result[:jp] = FiscalScenario(:jp,mode,jp_setup,jp.household,jp.core,
                       jp.anchor,phi_ref,NaN,
                       abs(jp.anchor.asset_mkt),0.0)

    # Source stage 1: US steady state at JAPAN's B and r_ante, zero US piw.
    anchor = (; Y=us_setup.state.Y,
                B=jp_setup.state.B, r_ante=jp_setup.state.r_ante,
                piw=us_setup.state.piw)
    step1 = _solve_beta_output(us_setup,us_setup.core,anchor,
              us.household.vp_slope,jp.household.beta,us_setup.state.Y;
              binding=false,tol=inner_tol)
    phi1 = _phi_at_solution(step1)
    result[:us_high_beta] = FiscalScenario(:us_high_beta,mode,
                       step1.setup,step1.household,step1.core,step1.eq,
                       phi_ref,phi1,step1.inner_residual,0.0)

    # Source stage 2: jointly select slope and (beta,Y) so GE_phi=US phi.
    function candidate2(slope)
        trial=_solve_beta_output(us_setup,us_setup.core,anchor,
                 slope,jp.household.beta,us_setup.state.Y;
                 binding=false,tol=inner_tol)
        return (;trial, phi=_phi_at_solution(trial))
    end
    f2=slope->candidate2(slope).phi-phi_ref
    root2=_bracket_phi(f2,jp.household.vp_slope,us.household.vp_slope;
                       ftol=phi_tol)
    step2=candidate2(root2.x)
    result[:us_high_beta_low_phi] = FiscalScenario(
                       :us_high_beta_low_phi,mode,step2.trial.setup,
                       step2.trial.household,step2.trial.core,step2.trial.eq,
                       phi_ref,step2.phi,step2.trial.inner_residual,
                       step2.phi-phi_ref)

    # Source stage 3: Japanese growth, kappa, piw and binding ZLB, but
    # still U.S. production parameters, public spending and saver structure.
    zcore=_scenario4_macrocore(us_setup.core,jp_setup.core)
    anchor_z=(;Y=us_setup.state.Y, B=jp_setup.state.B,
               r_ante=jp_setup.state.r_ante, piw=jp_setup.state.piw)
    function candidate3(slope)
        trial=_solve_beta_output(us_setup,zcore,anchor_z,
                  slope,jp.household.beta,us_setup.state.Y;
                  binding=true,tol=inner_tol)
        return (;trial, phi=_phi_at_solution(trial))
    end
    f3=slope->candidate3(slope).phi-phi_ref
    root3=_bracket_phi(f3,jp.household.vp_slope,us.household.vp_slope;
                       ftol=phi_tol)
    step3=candidate3(root3.x)
    result[:us_high_beta_low_phi_zlb] = FiscalScenario(
                       :us_high_beta_low_phi_zlb,mode,
                       step3.trial.setup,step3.trial.household,
                       step3.trial.core,step3.trial.eq,
                       phi_ref,step3.phi,step3.trial.inner_residual,
                       step3.phi-phi_ref)
    return result
end

"""Compute one Figure 9 locus from re-solved steady-state equilibria."""
function figure9_curve(scenario::FiscalScenario;
                       n_r::Int=120,n_Y::Int=300,
                       tol::Real=2e-5,root_tol::Real=1e-7)
    return deficit_debt_curve(scenario.setup, scenario.household,
               scenario.core;model=:bu_log,mode=scenario.mode,
               n_r,n_Y,tol,root_tol)
end

"""Solve all Figure 9 loci in the exact original paper order."""
function run_figure9_curves(scenarios::AbstractDict;
                            n_r::Int=120,n_Y::Int=300,
                            strict::Bool=true)
    result=Dict{Symbol,DeficitDebtCurve}()
    for key in FIGURE9_SCENARIOS
        s=scenarios[key]
        curve=figure9_curve(s;n_r,n_Y)
        valid_flex=count(p->!ismissing(p)&&p.branch===:flex,curve.records)
        valid_zlb=count(p->!ismissing(p)&&p.branch===:zlb,curve.records)
        if strict && (valid_flex!=n_r || valid_zlb<2)
            error("incomplete Figure 9 $key: flex=$valid_flex/$n_r ZLB=$valid_zlb")
        end
        result[key]=curve
    end
    return result
end

"""The maximum is explicitly a GRID maximum, not a continuous optimum."""
function scenario_maxima(curves::AbstractDict)
    return Dict(k=>grid_maximum(curves[k]) for k in FIGURE9_SCENARIOS)
end

"""Estimate the local ZLB slope with *actual neighboring solved roots*.

The returned ratio ΔB/Δ(deficit) is only a finite-difference diagnostic,
not an equilibrium generated by interpolating missing points.
"""
function local_zlb_debt_response(curve::DeficitDebtCurve, Y_anchor::Real)
    pts=DebtPoint{Float64}[p for p in curve.records if !ismissing(p) && p.branch===:zlb]
    length(pts)>=3 || error("insufficient solved ZLB points")
    i=argmin([abs(p.Y-Y_anchor) for p in pts])
    i=clamp(i,2,length(pts)-1)
    first_pt,last_pt=pts[i-1],pts[i+1]
    dz=last_pt.deficit-first_pt.deficit
    abs(dz)>1e-12 || error("ZLB deficit response indeterminate")
    return (; dB_ddeficit=(last_pt.B-first_pt.B)/dz,
             anchor_Y=pts[i].Y,
             grid_indices=(first_pt.index,last_pt.index))
end

"""Write every numerical equilibrium and unsolved slot, without touching Figure 8."""
function write_figure9_curve(key::Symbol, curve::DeficitDebtCurve;
                             directory::AbstractString=@__DIR__)
    key in FIGURE9_SCENARIOS || throw(ArgumentError("invalid Figure 9 scenario"))
    path=joinpath(directory,"figure9_$(key)_$(curve.mode).tsv")
    issues=Dict(i.index=>i.reason for i in curve.issues)
    ny=length(curve.Y_grid)
    open(path,"w") do io
        println(io,"scenario\tindex\tbranch\tgrid\tstatus\tB\tdeficit\tY\tr_ante\tpiw\ti\tr\tpi\tasset_market_residual\tregime_valid\tnote")
        for idx in eachindex(curve.records)
            branch=idx<=ny ? :zlb : :flex
            grid=idx<=ny ? curve.Y_grid[idx] : curve.r_grid[idx-ny]
            p=curve.records[idx]
            if ismissing(p)
                fields=(key,idx,branch,repr(grid),"UNSOLVED",
                        fill("",10)...,get(issues,idx,"not attempted"))
            else
                fields=(key,idx,branch,repr(grid),"SOLVED",repr(p.B),
                        repr(p.deficit),repr(p.Y),repr(p.r_ante),
                        repr(p.piw),repr(p.i),repr(p.r),repr(p.pi),
                        repr(p.residual),p.regime_valid,
                        get(issues,idx,""))
            end
            println(io,join(string.(fields),'\t'))
        end
    end
    return path
end

"""Write independent re-solved scenario calibration data at full precision."""
function write_figure9_calibration(scenarios::AbstractDict;
                                  directory::AbstractString=@__DIR__)
    mode=scenarios[:us].mode
    path=joinpath(directory,"figure9_calibrations_$(mode).tsv")
    open(path,"w") do io
        println(io,"scenario\tmode\tbeta\trho\tphi_tilde\tkappa\tB\tr_ante\tY\tpiw\tphi_reference\tphi_local\tasset_mkt\tlabor_foc\tpiw_cond\tinner_residual\touter_residual")
        for key in FIGURE9_SCENARIOS
            s=scenarios[key]
            vals=(key,mode,s.household.beta,-log(s.household.beta),
                  s.household.vp_slope,s.core.wage.kappa,
                  s.setup.state.B,s.setup.state.r_ante,s.setup.state.Y,
                  s.setup.state.piw,s.phi_reference,s.phi_local,
                  s.anchor.asset_mkt,s.anchor.labor_foc,s.anchor.piw_cond,
                  s.inner_residual,s.outer_residual)
            println(io,join(string.(vals),'\t'))
        end
    end
    return path
end

"""Original-Python full-curve parity with explicit matching of grid arguments.

The reference file must be obtained by independently executing the author's
`get_deficit_debt` using the *official* model inputs. ZLB grid arguments
are scenario-specific because their solved boundary Y may differ. Non-matching
grid arguments are reported as such; no fabricated cross-grid comparison.
"""
function compare_figure9_python(curves::AbstractDict, path::AbstractString;
                                directory::AbstractString=@__DIR__,
                                filename::AbstractString="figure9_python_julia_parity.tsv",
                                grid_tol::Real=3e-5)
    isfile(path) || throw(ArgumentError("source Python Figure 9 TSV not found: $path"))
    ref=Dict{Tuple{Symbol,Int},NamedTuple}()
    parse_number = v -> lowercase(v)=="nan" ? NaN : parse(Float64,v)
    for (lineno,line) in enumerate(eachline(path))
        lineno==1 && continue
        cols=split(line,'\t')
        length(cols)==7 || error("bad Python benchmark TSV at line $lineno")
        ref[(Symbol(cols[1]),parse(Int,cols[2]))]=(
          branch=Symbol(cols[3]),grid=parse_number(cols[4]),
          B=parse_number(cols[5]),deficit=parse_number(cols[6]),
          r_ante=parse_number(cols[7]))
    end
    rows=NamedTuple[]
    for key in FIGURE9_SCENARIOS
        curve=curves[key]; ny=length(curve.Y_grid)
        for i in eachindex(curve.records)
            src=get(ref,(key,i),nothing)
            src===nothing && error("missing original Python reference $key index $i")
            branch=i<=ny ? :zlb : :flex
            grid=i<=ny ? curve.Y_grid[i] : curve.r_grid[i-ny]
            branch===src.branch || error("original Python branch disagrees at $key index $i")
            matched=abs(grid-src.grid)<=grid_tol
            p=curve.records[i]
            good=matched && !ismissing(p) && isfinite(src.B) && isfinite(src.deficit)
            status=good ? :compared : !matched ? :grid_mismatch :
                   !ismissing(p) ? :python_unsolved :
                   isfinite(src.B) ? :julia_unsolved : :both_unsolved
            bd=good ? p.B-src.B : NaN
            dz=good ? p.deficit-src.deficit : NaN
            push!(rows,(;scenario=key,index=i,branch,grid_julia=grid,
                         grid_python=src.grid,status,
                         julia_B=ismissing(p) ? NaN : p.B,
                         python_B=src.B,
                         julia_deficit=ismissing(p) ? NaN : p.deficit,
                         python_deficit=src.deficit,debt_error=bd,deficit_error=dz))
        end
    end
    output=joinpath(directory,filename)
    open(output,"w") do io
        println(io,"scenario\tindex\tbranch\tgrid_julia\tgrid_python\tstatus\tjulia_B\tpython_B\tjulia_deficit\tpython_deficit\tdebt_error\tdeficit_error")
        for r in rows
            println(io,join(string.((r.scenario,r.index,r.branch,r.grid_julia,
                   r.grid_python,r.status,r.julia_B,r.python_B,
                   r.julia_deficit,r.python_deficit,r.debt_error,
                   r.deficit_error)),'\t'))
        end
    end
    return (;rows,path=output)
end

"""SummaryTables.jl output: counterfactual calibration, fiscal space, parity.

All arithmetic uses unrounded floats; rounding happens only when rendering
cells. The final parity table is produced only if a source audit exists.
"""
function make_figure9_tables(scenarios::AbstractDict,curves::AbstractDict;
                             directory::AbstractString=@__DIR__,parity=nothing)
    mode=scenarios[:us].mode
    values=Matrix{String}(undef,6,7)
    values[1,:]=["Scenario","Beta","Rho","Phi-tilde","Kappa","ZLB","Max residual"]
    fiscal=Matrix{String}(undef,6,6)
    fiscal[1,:]=["Scenario","Max deficit (% GDP)","Debt at max (% GDP)",
                  "ZLB boundary (% GDP)","Debt near JP deficit (% GDP)","Max residual"]
    jp_deficit=scenarios[:jp].anchor.deficit
    for (i,key) in enumerate(FIGURE9_SCENARIOS)
        s=scenarios[key]; c=curves[key]; mx=grid_maximum(c)
        r=s.anchor
        data=(string(key),@sprintf("%.10f",s.household.beta),
              @sprintf("%.8f",-log(s.household.beta)),
              @sprintf("%.8f",s.household.vp_slope),
              @sprintf("%.8f",s.core.wage.kappa),
              string(s.core.monetary.zlb),
              @sprintf("%.3g",max(s.inner_residual,abs(s.outer_residual))))
        for j in 1:7
            values[i+1,j]=data[j]
        end
        allpts=[p for p in c.records if !ismissing(p)]
        near=argmin(p->abs(p.deficit-jp_deficit),allpts)
        vals=(string(key),@sprintf("%.5f",100 * mx.maximum_deficit),
              @sprintf("%.5f",100 * mx.debt_at_maximum/c.periods),
              @sprintf("%.5f",100 * c.boundary.B/c.periods),
              @sprintf("%.5f",100 * near.B/c.periods),
              @sprintf("%.3g",mx.max_equilibrium_residual))
        for j in 1:6
            fiscal[i+1,j]=vals[j]
        end
    end
    paths=Dict{Symbol,String}()
    for (key,mat) in ((:calibration,values),(:fiscal,fiscal))
        table=Table(Cell.(mat);header=1)
        path=joinpath(directory,"figure9_$(key)_$(mode).tex")
        open(path,"w") do io
            show(io,MIME"text/latex"(),table)
        end
        paths[key]=path
    end
    if parity!==nothing
        mat=Matrix{String}(undef,6,7)
        mat[1,:]=["Scenario","Compared","Max abs B error","RMS B error",
                  "Max abs deficit error","RMS deficit error","Uncompared"]
        for (i,key) in enumerate(FIGURE9_SCENARIOS)
            sub=[r for r in parity.rows if r.scenario===key]
            ok=[r for r in sub if r.status===:compared]
            maxb=isempty(ok) ? NaN : maximum(r->abs(r.debt_error),ok)
            maxd=isempty(ok) ? NaN : maximum(r->abs(r.deficit_error),ok)
            rmsb=isempty(ok) ? NaN : sqrt(sum(r.debt_error^2 for r in ok)/length(ok))
            rmsd=isempty(ok) ? NaN : sqrt(sum(r.deficit_error^2 for r in ok)/length(ok))
            vals=(string(key),string(length(ok)),@sprintf("%.5g",maxb),
                  @sprintf("%.5g",rmsb),@sprintf("%.5g",maxd),
                  @sprintf("%.5g",rmsd),string(length(sub)-length(ok)))
            for j in 1:7
                mat[i+1,j]=vals[j]
            end
        end
        table=Table(Cell.(mat);header=1)
        path=joinpath(directory,"figure9_python_parity_$(mode).tex")
        open(path,"w") do io
            show(io,MIME"text/latex"(),table)
        end
        paths[:parity]=path
    end
    return paths
end

end # module GoldilocksFiscalSpace
