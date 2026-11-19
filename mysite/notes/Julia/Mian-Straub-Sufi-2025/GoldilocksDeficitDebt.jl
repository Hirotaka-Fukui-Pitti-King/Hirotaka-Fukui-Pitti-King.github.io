module GoldilocksDeficitDebt

using LinearAlgebra
using Printf
using SummaryTables
using ..GoldilocksCore
using ..GoldilocksHouseholds
using ..GoldilocksCalibration

export FIGURE8_MODELS, DebtPoint, BranchIssue, DeficitDebtCurve,
       original_debt_grids, official_household, figure8_calibration,
       zlb_boundary, solve_fixed_debt, deficit_debt_curve,
       grid_maximum, compare_curves, compare_python_reference,
       write_curve_data, make_deficit_debt_table, validate_point

# Figure 8 (AER p. 4283): OLG-alt is deliberately NOT displayed.
const FIGURE8_MODELS = (:bu, :bu_log, :olg, :ha)
const RESIDUAL_KEYS = (:asset_mkt, :goods_mkt, :fisher, :piw_cond,
                       :invcond, :valcond, :qres, :jvaluation, :Jvaluation)

"""Fully evaluated, not interpolated, stationary equilibrium at one Figure 8 grid point."""
struct DebtPoint{T<:Real}
    index::Int
    branch::Symbol
    Y::T
    B::T
    deficit::T
    r_ante::T
    r::T
    i::T
    i_rule::T
    pi::T
    piw::T
    tax::T
    C::T
    A::T
    residual::T
    regime_valid::Bool
end

"""An unsuccessful solve is retained explicitly rather than hidden in a plotted curve."""
struct BranchIssue
    index::Int
    branch::Symbol
    grid_value::Float64
    reason::String
end

"""Figure 8 run, including unsuccessful or economically inadmissible grid points."""
struct DeficitDebtCurve{S,H,C}
    country::Symbol
    periods::Int
    model::Symbol
    mode::Symbol
    setup::S
    household::H
    core::C
    r_grid::Vector{Float64}
    Y_grid::Vector{Float64}
    records::Vector{Union{Missing,DebtPoint{Float64}}}
    issues::Vector{BranchIssue}
    boundary::NamedTuple
end

# Parameters below are transcribed from the *official* us_ss.pkl and jp_ss.pkl.
# They are fixed reference INPUTS only. No official debt-deficit output is used.
const OFFICIAL_PARAMETERS = Dict{Tuple{Symbol,Symbol},NamedTuple}(
    (:us,:bu) => (beta=0.8557304395059222, slope=0.2862183664715453,
                   v0=0.2862575580728031, survival=1.0, zeta=0.0, eis=1.0),
    (:us,:bu_log) => (beta=0.8557697856259636, slope=0.2861891207062559,
                       v0=0.0, survival=1.0, zeta=0.0, eis=1.0),
    (:us,:olg) => (beta=1.216489717472069, slope=0.0, v0=0.0,
                    survival=0.7829346909379932, zeta=0.0, eis=1.0),
    (:us,:ha) => (beta=0.9423105702425766, slope=0.0, v0=0.0,
                   survival=1.0, zeta=0.0, eis=0.21904952574407402),
    (:jp,:bu) => (beta=0.9591541576555624, slope=0.07989320834710602,
                   v0=0.07990530830725283, survival=1.0, zeta=0.0, eis=1.0),
    (:jp,:bu_log) => (beta=0.9591746054283945, slope=0.07987045057765649,
                       v0=0.0, survival=1.0, zeta=0.0, eis=1.0),
    (:jp,:olg) => (beta=1.055178873502947, slope=0.0, v0=0.0,
                    survival=0.9225826420270378, zeta=0.0, eis=1.0),
    (:jp,:ha) => (beta=0.9817625691279767, slope=0.0, v0=0.0,
                   survival=1.0, zeta=0.0, eis=0.2991173034969982),
)

"""Build a saver with official annual US/Japan calibration INPUT parameters."""
function official_household(s::CalibrationSetup, model::Symbol)
    s.periods == 1 || throw(ArgumentError("Figure 8 uses annual calibrations"))
    model in FIGURE8_MODELS || throw(ArgumentError("not a Figure 8 model: $model"))
    p = OFFICIAL_PARAMETERS[(s.country,model)]
    if model === :bu
        return BUQuadratic(; beta=p.beta, eis=p.eis, vp_0=p.v0,
                           vp_slope=p.slope, A_ss=s.saver_assets,
                           lambda_bu=s.lambda_bu)
    elseif model === :bu_log
        return BULog(; beta=p.beta, eis=p.eis, vp_slope=p.slope,
                     A_ss=s.saver_assets, lambda_bu=s.lambda_bu)
    elseif model === :olg
        return OLGHousehold(; beta=p.beta, survival=p.survival, zeta=p.zeta)
    else
        return HAHousehold(; beta=p.beta, eis=p.eis, lumpsum=0.0,
                           rho_e=s.ha_rho, sd_e=0.92, n_e=7, n_a=300,
                           min_a=0.0, max_a=500.0,
                           source_lottery_extrapolation=true)
    end
end

"""Build parameter-specific core without changing the calibrated economic equations."""
function _household_core(s::CalibrationSetup, household)
    original = s.core
    wage = original.wage
    survival = household isa OLGHousehold ? household.survival : 1.0
    # For the source mode, the paper's wage adjustment is centered at its
    # original steady state, with the household-specific beta/survival.
    w = WageParams(; kappa=wage.kappa, beta=household.beta,
                    survival, labor_shifter=wage.labor_shifter,
                    lag=wage.lag, forward=wage.forward, frisch=wage.frisch)
    return CoreParams(; production=original.production, wage=w,
                      monetary=original.monetary, fiscal=original.fiscal,
                      mu=original.mu, delta_b=original.delta_b)
end

"""
    figure8_calibration(country, model; mode=:native, method=:auto)

For `:native`, recalibrate the household in Julia using the target moments.
The default matches Task V's validated algorithm: HA uses the implicit GE
elasticity for the outer root, while analytic households use the original
finite-difference GE elasticity. `calibrate_model` independently verifies the
final HA elasticity by re-solving the finite-difference general equilibrium.
An explicit `method=:finite` or `:implicit` overrides this selection.
For `:official`, use only the official saved *parameters* as benchmark inputs.
No computed curve or debt-deficit coordinates are loaded from the replication.
"""
function figure8_calibration(country::Symbol, model::Symbol;
                             mode::Symbol=:native, method::Symbol=:auto)
    model in FIGURE8_MODELS || throw(ArgumentError("Figure 8 excludes model: $model"))
    mode in (:native,:official) || throw(ArgumentError("mode must be :native or :official"))
    method in (:auto,:finite,:implicit) || throw(ArgumentError("unsupported GE method: $method"))
    s = country_setup(country,1)
    if mode === :native
        selected_method = method === :auto ? (model === :ha ? :implicit : :finite) : method
        result = calibrate_model(s,model;method=selected_method)
        return (; setup=s, household=result.household, core=result.core,
                 anchor=result.eq, mode, source=result)
    end
    household = official_household(s,model)
    core = _household_core(s,household)
    anchor = evaluate_steady_state(s.state,core,household)
    return (; setup=s, household, core, anchor, mode, source=nothing)
end

"""Reconstruct the *exact* grids of original `auxfs.get_deficit_debt`."""
function original_debt_grids(pi_trend::Real, real_growth::Real,
                             Y_zlb::Real; n_r::Int=40, n_Y::Int=70)
    n_r >= 2 && n_Y >= 2 || throw(ArgumentError("need at least two grid points"))
    Y_zlb > 0 || throw(DomainError(Y_zlb,"positive ZLB-boundary output required"))
    1+pi_trend > 0 || throw(DomainError(pi_trend,"invalid gross trend inflation"))
    r_zlb = 1/(1+pi_trend)-1-real_growth
    return (; r_zlb,
            r_grid=collect(range(r_zlb,stop=1e-4,length=n_r)),
            Y_grid=collect(range(0.0,stop=Float64(Y_zlb),length=n_Y)))
end

"""Use the source's exogenous regime flag, never an undocumented max(i,0)."""
function _regime_core(core::CoreParams, binding::Bool)
    m = core.monetary
    monetary = MonetaryParams(; pi_trend=m.pi_trend,
        real_growth=m.real_growth, phi_pi=m.phi_pi,
        real_rate=m.real_rate, i_shock=m.i_shock,
        zlb=binding, i_zlb=m.i_zlb)
    return CoreParams(; production=core.production, wage=core.wage,
                      monetary, fiscal=core.fiscal, mu=core.mu,
                      delta_b=core.delta_b)
end

@inline function _state(B,Y,r_ante,piw)
    return (; B,Y,r_ante,piw)
end

@inline _candidate(B,Y,r,piw,core,hh) =
    evaluate_steady_state(_state(B,Y,r,piw),core,hh)

# A small Jacobian-based solver avoids differentiating through HA's EGM fixed
# point. Its line search rejects invalid consumption/capital/grid evaluations.
# Both unknowns and residuals are always recomputed at the accepted solution.
function _fd_jacobian(f, x, fx; step=1e-5)
    n=length(x)
    J=Matrix{Float64}(undef,n,n)
    for k in 1:n
        h=step*max(1.0,abs(x[k]))
        vplus=copy(x);vplus[k]+=h
        vminus=copy(x);vminus[k]-=h
        fp=try f(vplus) catch; nothing end
        fm=try f(vminus) catch; nothing end
        if fp !== nothing && fm !== nothing && all(isfinite,fp) && all(isfinite,fm)
            J[:,k]=(fp-fm)/(2h)
        elseif fp !== nothing && all(isfinite,fp)
            J[:,k]=(fp-fx)/h
        elseif fm !== nothing && all(isfinite,fm)
            J[:,k]=(fx-fm)/h
        else
            error("No finite-difference direction for unknown $k")
        end
    end
    return J
end

function _solve_residual(f, x0; tol=1e-7, maxiter=55)
    x=Float64[x0...]
    length(x)>0 && all(isfinite,x) || throw(ArgumentError("finite initial guess required"))
    fx=f(x)
    length(fx)==length(x) || throw(DimensionMismatch("non-square equilibrium system"))
    all(isfinite,fx) || error("nonfinite initial equilibrium residual")
    for iter in 1:maxiter
        err=norm(fx,Inf)
        err<=tol && return (;x, residual=fx, residual_norm=err, iterations=iter-1)
        J=_fd_jacobian(f,x,fx)
        step=J\fx
        all(isfinite,step) || error("nonfinite Newton direction")
        accepted=false
        for j in 0:28
            trial=x .- step/(2.0^j)
            ft=try f(trial) catch; nothing end
            if ft!==nothing && length(ft)==length(fx) && all(isfinite,ft) &&
               norm(ft,Inf)<err
                x=trial;fx=ft;accepted=true;break
            end
        end
        accepted || error("Newton line search stalled (residual=$(err))")
    end
    error("nonlinear stationary solve exceeded $maxiter iterations; residual $(norm(fx,Inf))")
end

"""
    zlb_boundary(core, household, state; tol=2e-7)

Find the joint B,Y equilibrium at `piw=0` and the zero-nominal-rate real
return. The original uses two scalar preliminary solves then a joint root.
"""
function zlb_boundary(core::CoreParams, household, state::NamedTuple;tol=2e-7)
    flex=_regime_core(core,false)
    r=1/(1+flex.monetary.pi_trend)-1-flex.monetary.real_growth
    B0=state.B; Y0=state.Y
    yfun(x)=[_candidate(B0,x[1],r,0.0,flex,household).labor_foc]
    Yguess=_solve_residual(yfun,[Y0];tol).x[1]
    bfun(x)=[_candidate(x[1],Yguess,r,0.0,flex,household).asset_mkt]
    Bguess=_solve_residual(bfun,[B0];tol).x[1]
    F(x)=begin
        eq=_candidate(x[1],x[2],r,0.0,flex,household)
        [eq.asset_mkt,eq.labor_foc]
    end
    sol=_solve_residual(F,[Bguess,Yguess];tol)
    eq=_candidate(sol.x[1],sol.x[2],r,0.0,flex,household)
    abs(eq.i)<1e-6 || error("nonzero nominal rate at ZLB boundary")
    return (; B=sol.x[1],Y=sol.x[2],r_ante=r,piw=0.0,eq,
             residual_norm=sol.residual_norm)
end

"""
    solve_fixed_debt(core, household, B; binding=false, guess, tol=1e-7)

Solve the GE at a *given debt stock*, useful as an additional validation of
curve points. Away from the ZLB unknowns are (r_ante,Y); at the ZLB they are
(r_ante,Y,piw). The original figure instead parametrizes by r_ante or Y.
"""
function solve_fixed_debt(core::CoreParams, household, B::Real;
                          binding::Bool=false, guess=nothing, tol=1e-7)
    regime=_regime_core(core,binding)
    isnothing(guess) && throw(ArgumentError("provide r_ante,Y,piw initial guess"))
    if !binding
        # Distinct anonymous closures: defining f(x) in both branches creates
        # two methods in the same Julia local scope.  The second overwrites
        # the first even when the !binding branch returns, causing a 2-D
        # Newton iterate to be indexed at x[3].
        flex_residual = x -> begin
            e=_candidate(B,x[2],x[1],0.0,regime,household)
            [e.asset_mkt,e.labor_foc]
        end
        sol=_solve_residual(flex_residual,[guess.r_ante,guess.Y];tol)
        return _candidate(B,sol.x[2],sol.x[1],0.0,regime,household)
    end
    zlb_residual = x -> begin
        e=_candidate(B,x[2],x[1],x[3],regime,household)
        [e.asset_mkt,e.fisher,e.piw_cond]
    end
    sol=_solve_residual(zlb_residual,[guess.r_ante,guess.Y,guess.piw];tol)
    return _candidate(B,sol.x[2],sol.x[1],sol.x[3],regime,household)
end

"""Economic and accounting audit; no root may be accepted merely by solver status."""
function validate_point(eq, core::CoreParams;tol=2e-5)
    vals=Float64[getproperty(eq,k) for k in RESIDUAL_KEYS]
    # HA Euler inequalities are state-dependent and checked in the existing
    # EGM tests; scalar household FOCs and budgets can be audited directly.
    for key in (:budget,:H_recursion,:euler)
        if hasproperty(eq,key) && getproperty(eq,key) isa Real
            push!(vals,Float64(getproperty(eq,key)))
        end
    end
    err=norm(vals,Inf)
    abs(eq.deficit+eq.rpost_bonds*eq.B)<=tol ||
        error("stationary government debt identity does not hold")
    eq.Y>0 && eq.N>0 && eq.C>0 && isfinite(eq.B) ||
        error("nonpositive or nonfinite stationary output, employment or consumption")
    isfinite(err) && err<=tol || error("stationary equilibrium residual=$err")
    # This checks only compatibility with the mechanical nominal-rate floor.
    # With `real_rate=1`, Fisher and the source monetary rule can imply
    # i_rule==i_zlb by construction at the ZLB: this is NOT a proof of
    # global equilibrium selection or of a distinct shadow Taylor rule.
    if core.monetary.zlb
        return (;residual=err,regime_valid=eq.i_rule<=core.monetary.i_zlb+tol)
    else
        return (;residual=err,regime_valid=eq.i>=core.monetary.i_zlb-tol)
    end
end

function _point(eq, core, index, branch;tol=2e-5)
    diagnostic=validate_point(eq,core;tol)
    return DebtPoint{Float64}(index,branch,eq.Y,eq.B,eq.deficit,
                             eq.r_ante,eq.r,eq.i,eq.i_rule,eq.pi,
                             eq.piw,eq.tax,eq.C,eq.A,
                             diagnostic.residual,diagnostic.regime_valid)
end

# Use secant-predicted warm starts; unlike the original CubicSpline after its
# first five roots this never alters the equations or extrapolates the curve.
function _predict(points::Vector{Tuple{Float64,Vector{Float64}}}, t::Real)
    isempty(points) && error("need at least one continuation anchor")
    length(points)==1 && return copy(points[end][2])
    (t0,x0),(t1,x1)=points[end-1],points[end]
    iszero(t1-t0) && return copy(x1)
    return x1 .+ ((t-t1)/(t1-t0)) .* (x1.-x0)
end

"""
    deficit_debt_curve(setup, household, core; model, mode=:native,
                       n_r=40, n_Y=70, ...)

Reproduce the two branches and their source ordering. Failed points remain
`missing` with an indexed `BranchIssue`. Source stopping after a positive then
negative deficit is available and enabled by default. Equilibrium solutions
are NEVER inferred by interpolation.
"""
function deficit_debt_curve(s::CalibrationSetup,hh,core::CoreParams;
                            model::Symbol,mode::Symbol=:native,
                            n_r::Int=40,n_Y::Int=70,
                            tol::Real=2e-5, root_tol::Real=1e-7,
                            stop_after_negative::Bool=true)
    s.periods==1 || throw(ArgumentError("Figure 8 uses annual calibrations"))
    boundary=zlb_boundary(core,hh,s.state;tol=root_tol)
    grids=original_debt_grids(core.monetary.pi_trend,
                              core.monetary.real_growth,boundary.Y;n_r,n_Y)
    records=fill!(Vector{Union{Missing,DebtPoint{Float64}}}(undef,n_Y+n_r),missing)
    issues=BranchIssue[]
    zcore=_regime_core(core,true)
    fcore=_regime_core(core,false)
    # In the Python arrays the ZLB portion occupies indices 1:n_Y, ordered
    # from low to high output, although the solver walks high to low.
    z_history=Tuple{Float64,Vector{Float64}}[]
    push!(z_history,(boundary.Y,[boundary.r_ante,boundary.B,0.0]))
    seen_positive=false
    for i in n_Y:-1:1
        y=grids.Y_grid[i]
        if y<=0
            push!(issues,BranchIssue(i,:zlb,Float64(y),"Y=0 excluded: production requires Y>0"))
            break
        end
        guess=_predict(z_history,y)
        zlb_residual = z -> begin
            e=_candidate(z[2],y,z[1],z[3],zcore,hh)
            [e.asset_mkt,e.fisher,e.piw_cond]
        end
        try
            sol=try _solve_residual(zlb_residual,guess;tol=root_tol)
                catch
                    _solve_residual(zlb_residual,last(z_history)[2];tol=root_tol)
                end
            e=_candidate(sol.x[2],y,sol.x[1],sol.x[3],zcore,hh)
            point=_point(e,zcore,i,:zlb;tol)
            records[i]=point
            push!(z_history,(y,copy(sol.x)))
            seen_positive |= point.deficit>0
            if stop_after_negative && seen_positive && point.deficit<0
                push!(issues,BranchIssue(i,:zlb,Float64(y),"source stopping rule: positive deficit followed by negative"))
                break
            end
        catch err
            push!(issues,BranchIssue(i,:zlb,Float64(y),sprint(showerror,err)))
            break
        end
    end
    # The source restarts the non-ZLB branch at the common boundary and
    # increments r_ante on the ascending grid.
    r_history=Tuple{Float64,Vector{Float64}}[]
    push!(r_history,(boundary.r_ante,[boundary.B,boundary.Y]))
    for k in 1:n_r
        r=grids.r_grid[k]
        i=n_Y+k
        guess=_predict(r_history,r)
        flex_residual = z -> begin
            e=_candidate(z[1],z[2],r,0.0,fcore,hh)
            [e.asset_mkt,e.labor_foc]
        end
        try
            sol=try _solve_residual(flex_residual,guess;tol=root_tol)
                catch
                    _solve_residual(flex_residual,last(r_history)[2];tol=root_tol)
                end
            e=_candidate(sol.x[1],sol.x[2],r,0.0,fcore,hh)
            point=_point(e,fcore,i,:flex;tol)
            records[i]=point
            push!(r_history,(r,copy(sol.x)))
        catch err
            push!(issues,BranchIssue(i,:flex,Float64(r),sprint(showerror,err)))
            # Unlike the source (which raises), keep the unsolved tail
            # explicit as missing, not as fabricated equilibrium points.
            break
        end
    end
    return DeficitDebtCurve(s.country,s.periods,model,mode,s,hh,core,
                            grids.r_grid,grids.Y_grid,records,issues,boundary)
end

"""Grid maximum among computed, economically nonnegative-debt equilibria."""
function grid_maximum(curve::DeficitDebtCurve;require_regime::Bool=false)
    pts=DebtPoint{Float64}[p for p in curve.records if !ismissing(p) &&
                            p.B>=0 && (!require_regime || p.regime_valid)]
    isempty(pts) && error("no valid equilibrium for a grid maximum")
    p=argmax(x->x.deficit,pts)
    return (; maximum_deficit=p.deficit,debt_at_maximum=p.B,
            index=p.index,branch=p.branch,
            max_equilibrium_residual=maximum(x->x.residual,pts),
            count=length(pts),grid_based=true)
end

"""Compare exact common grid arguments only; never match different ZLB Y levels by index."""
function compare_curves(native::DeficitDebtCurve,official::DeficitDebtCurve)
    (native.country,native.model,native.periods)==
    (official.country,official.model,official.periods) ||
        throw(ArgumentError("curves must have same country/model/frequency"))
    length(native.records)==length(official.records) ||
        throw(DimensionMismatch("different grid sizes"))
    nY=length(native.Y_grid)
    rows=NamedTuple[]
    for i in eachindex(native.records)
        n=native.records[i];o=official.records[i]
        grid_native=i<=nY ? native.Y_grid[i] : native.r_grid[i-nY]
        grid_official=i<=nY ? official.Y_grid[i] : official.r_grid[i-nY]
        exact_grid=isapprox(grid_native,grid_official;atol=1e-12,rtol=0)
        if !ismissing(n) && !ismissing(o) && exact_grid
            push!(rows,(;index=i,branch=n.branch,grid_native,grid_official,
                         matched=true,native_B=n.B,official_B=o.B,
                         native_deficit=n.deficit,official_deficit=o.deficit,
                         debt_gap=n.B-o.B,deficit_gap=n.deficit-o.deficit))
        else
            # Disparate Y_zlb implies disparate entire ZLB Y grids; an
            # index-by-index difference there would NOT be a valid comparison.
            push!(rows,(;index=i,branch=i<=nY ? :zlb : :flex,
                         grid_native,grid_official,matched=false,
                         native_B=missing,official_B=missing,
                         native_deficit=missing,official_deficit=missing,
                         debt_gap=missing,deficit_gap=missing))
        end
    end
    return rows
end


"""
    compare_python_reference(curves, reference_file; output_file=nothing)

Compare individually re-solved Julia equilibria to the 40×70 grid reference
obtained by running the author's unmodified Python `get_deficit_debt` on the
official saved parameters. Missing solutions are reported explicitly. The TSV
is an AUDIT INPUT only, not a source of Julia-generated equilibrium values.
"""
function compare_python_reference(curves::AbstractVector{<:DeficitDebtCurve},
                                  reference_file::AbstractString;
                                  output_file=nothing)
    isfile(reference_file) || throw(ArgumentError("Python audit reference TSV not found"))
    originals=Dict{Tuple{Symbol,Symbol,Int},Tuple{Float64,Float64}}()
    for (line_no,line) in enumerate(eachline(reference_file))
        line_no==1 && continue
        cols=split(line,'\t')
        length(cols)==5 || error("source audit TSV has wrong column count on line $line_no")
        b=lowercase(cols[4])=="nan" ? NaN : parse(Float64,cols[4])
        z=lowercase(cols[5])=="nan" ? NaN : parse(Float64,cols[5])
        originals[(Symbol(cols[1]),Symbol(cols[2]),parse(Int,cols[3]))] = (b,z)
    end
    rows=NamedTuple[]
    for curve in curves
        for i in eachindex(curve.records)
            key=(curve.country,curve.model,i)
            haskey(originals,key) || error("missing source Python grid row: $key")
            b,z=originals[key]
            point=curve.records[i]
            valid_source=isfinite(b)&&isfinite(z)
            valid_julia=!ismissing(point)
            status=valid_source && valid_julia ? :compared :
                   valid_source ? :julia_unsolved :
                   valid_julia ? :python_unsolved : :both_unsolved
            Bd=status===:compared ? point.B-b : missing
            Zd=status===:compared ? point.deficit-z : missing
            push!(rows,(;country=curve.country,model=curve.model,
                         mode=curve.mode,index=i,status,
                         python_B=b,python_deficit=z,
                         julia_B=valid_julia ? point.B : missing,
                         julia_deficit=valid_julia ? point.deficit : missing,
                         debt_difference=Bd,deficit_difference=Zd))
        end
    end
    if output_file!==nothing
        open(output_file,"w") do io
            println(io,"country\tmodel\tmode\tindex\tstatus\tpython_B\tpython_deficit\tjulia_B\tjulia_deficit\tdebt_difference\tdeficit_difference")
            for row in rows
                println(io,join(string.((row.country,row.model,row.mode,row.index,
                    row.status,row.python_B,row.python_deficit,row.julia_B,
                    row.julia_deficit,row.debt_difference,row.deficit_difference)),"\t"))
            end
        end
    end
    return rows
end

"""Write every grid slot and diagnostic (including missing roots) to a flat TSV."""
function write_curve_data(curve::DeficitDebtCurve;directory::AbstractString=@__DIR__)
    filename=joinpath(directory,"figure8_$(curve.country)_$(curve.model)_$(curve.mode).tsv")
    issue_by_index=Dict(i.index=>i.reason for i in curve.issues)
    open(filename,"w") do io
        println(io,"index\tbranch\tgrid_input\tstatus\tB\tB_over_GDP\tdeficit\tY\tr_ante\tr\ti\ti_rule\tpi\tpiw\ttax\tC\tA\tresidual\tregime_valid\tnote")
        n_Y=length(curve.Y_grid)
        for i in eachindex(curve.records)
            branch=i<=n_Y ? :zlb : :flex
            grid_input=i<=n_Y ? curve.Y_grid[i] : curve.r_grid[i-n_Y]
            p=curve.records[i]
            if ismissing(p)
                println(io,join((i,branch,repr(grid_input),"UNSOLVED",
                                 fill("",15)...,get(issue_by_index,i,"not attempted")),"\t"))
            else
                values=(i,branch,repr(grid_input),"SOLVED",repr(p.B),
                        repr(p.B/curve.periods),repr(p.deficit),repr(p.Y),
                        repr(p.r_ante),repr(p.r),repr(p.i),repr(p.i_rule),
                        repr(p.pi),repr(p.piw),repr(p.tax),repr(p.C),
                        repr(p.A),repr(p.residual),p.regime_valid,
                        get(issue_by_index,i,""))
                println(io,join(values,"\t"))
            end
        end
    end
    return filename
end

"""Publication-style SummaryTables output for *computed* grid maxima."""
function make_deficit_debt_table(curves::AbstractVector{<:DeficitDebtCurve};
                                  directory::AbstractString=@__DIR__,
                                  filename::AbstractString="figure8_maxima.tex")
    mat=Matrix{String}(undef,length(curves)+1,7)
    mat[1,:]=["Country","Model","Mode","Max deficit (% GDP)",
              "Debt at max (% GDP)","ZLB boundary (% GDP)","Max residual"]
    for (i,curve) in enumerate(curves)
        mx=grid_maximum(curve)
        vals=(string(curve.country),string(curve.model),string(curve.mode),
              @sprintf("%.3f",100 * mx.maximum_deficit),
              @sprintf("%.2f",100 * mx.debt_at_maximum/curve.periods),
              @sprintf("%.2f",100 * curve.boundary.B/curve.periods),
              @sprintf("%.3g",mx.max_equilibrium_residual))
        for j in 1:7
            mat[i+1,j]=vals[j]
        end
    end
    tbl=Table(Cell.(mat);header=1)
    path=joinpath(directory,filename)
    open(path,"w") do io
        show(io,MIME"text/latex"(),tbl)
    end
    return (; path,table=tbl)
end

end # module GoldilocksDeficitDebt
