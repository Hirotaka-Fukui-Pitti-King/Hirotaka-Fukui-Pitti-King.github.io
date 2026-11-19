module GoldilocksCalibration

using LinearAlgebra
using Printf
using SummaryTables
using ..GoldilocksCore
using ..GoldilocksHouseholds

export CalibrationSetup, country_setup, calibrated_household, baseline_equilibrium,
       household_derivatives, household_inverse_asset_semi_elasticity,
       ge_elasticity_finite, ge_elasticity_implicit,
       calibration_moments, calibrate_model, calibrate_country, check_calibration,
       calibration_values, write_calibration_values, make_calibration_tables

const MODELS = (:bu, :bu_log, :olg, :olg_alt, :ha)
const JAPAN_KAPPA_ANNUAL = 0.14710702  # official data/clean/jp_kappa.pkl

"""Country/period base point and numerical targets, independent of the household model."""
struct CalibrationSetup{P,T}
    country::Symbol
    periods::Int
    core::P
    state::T
    phi_target::Float64
    top_10_share::Float64
    saver_assets::Float64
    saver_consumption::Float64
    lambda_bu::Float64
    beta_initial::Float64
    ha_rho::Float64
end

"""
    country_setup(country::Symbol, periods::Int=1)

Reproduce the original common macro calibration from primitive targets. The
Japanese wage kappa is the independently estimated **official** input; its IRF
estimation is not performed here. Units are per model period. No pickle loading.
"""
function country_setup(country::Symbol, periods::Int=1)
    (country === :us && periods == 1) || (country === :jp && periods in (1,4)) ||
        throw(ArgumentError("supported (country, periods): (:us,1), (:jp,1/4)"))
    us = country === :us
    Y = 1.0
    G = us ? 0.14 : 0.20
    B = (us ? 1.0 : 2.38) * periods
    growth = (us ? 0.015 : 0.003) / periods
    pi_trend = 0.02 / periods
    pi_actual = (us ? 0.02 : 0.003) / periods
    nominal = us ? 0.015 / periods : 0.0
    r_ante = (1 + nominal)/(1 + pi_actual) - 1 - growth
    piw = pi_actual - pi_trend
    delta_k = 0.08 / periods
    markup = 1.3
    fixed_cost = 0.3
    capital = 2.0 * periods
    eps_I = 4.0
    risk_premium = 0.0
    phi_target = 0.017 / periods
    top_10_share = us ? 0.457 : 0.442
    beta_initial = (us ? 0.95 : 0.97)^(1/periods)
    kappa = (us ? 0.06880706233915822 : JAPAN_KAPPA_ANNUAL)/periods
    firm_disc_rate = (1 + r_ante + risk_premium) * (1 + growth) - (1 - delta_k)
    alpha = markup * firm_disc_rate * capital / (Y + fixed_cost)
    tfp = (Y + fixed_cost) * capital^(-alpha)
    prodp = ProductionParams(; alpha, tfp, fixed_cost, markup, delta_k,
                              eps_I, real_growth=growth, risk_premium)
    prod = production_steady_state(Y, r_ante, prodp)
    isapprox(prod.k, capital; atol=1e-10) || error("capital calibration failed")
    isapprox(prod.N, 1.0; atol=1e-10) || error("labor normalization failed")
    mu = (1 - top_10_share)/(prod.w * prod.N)
    0 < mu < 1 || throw(DomainError(mu, "invalid hand-to-mouth share"))
    saver_assets = (B + (1+growth)*prod.k)/(1-mu)
    C = Y-G-prod.I
    tax = G + r_ante*B
    Z = prod.w*prod.N-tax
    saver_consumption = (C - mu*Z)/(1-mu)
    isapprox(saver_consumption, r_ante*saver_assets + Z +
             prod.pure_profit/(1-mu); atol=1e-9) ||
        error("stationary household budget and resource constraint disagree")
    wtmp = WageParams(; kappa, beta=beta_initial, survival=1.0,
                       frisch=1.0, labor_shifter=1.0)
    shifter = labor_shifter_for_phillips(piw, prod.N, Z, C, wtmp)
    wage = WageParams(; kappa, beta=beta_initial, survival=1.0,
                       frisch=1.0, labor_shifter=shifter)
    monetary = MonetaryParams(; pi_trend, real_growth=growth,
                                zlb=!us, i_zlb=0.0)
    fiscal = FiscalParams(; G, periods)
    core = CoreParams(; production=prodp, wage, monetary, fiscal, mu, delta_b=0.0)
    state = (; Y, B, r_ante, piw)
    lambda_bu = find_lambda_bu(beta_initial, r_ante, phi_target, 1.0, C)
    return CalibrationSetup(country, periods, core, state, phi_target,
                            top_10_share, saver_assets, saver_consumption,
                            lambda_bu, beta_initial, 0.9^(1/periods))
end

# Explicit primitive and steady-state restrictions; do not clip inadmissible parameters.
@inline function _value(x, name::Symbol, default)
    hasproperty(x, name) ? getproperty(x, name) : default
end

"""Solve a bracketed scalar root, stopping on the *function residual*.

`ftol` controls the economically relevant root accuracy; `xtol` controls the
floating-point resolution of the unknown. In the HA asset-market problem,
`dA/dβ` can be large, so stopping just because the β bracket is 1e-10 wide
can leave asset-market residuals above 1e-7. Keep both criteria separate.
"""
function _bisect(f, a::Real, b::Real; ftol=1e-10, xtol=1e-14, maxiter=100)
    ftol > 0 && xtol > 0 && maxiter > 0 ||
        throw(ArgumentError("positive ftol, xtol and maxiter required"))
    lo, hi = Float64(a), Float64(b)
    lo < hi || throw(ArgumentError("invalid root bracket"))
    flo, fhi = f(lo), f(hi)
    isfinite(flo) && isfinite(fhi) ||
        throw(DomainError((flo, fhi), "nonfinite root bracket endpoints"))
    abs(flo) <= ftol && return lo
    abs(fhi) <= ftol && return hi
    signbit(flo) != signbit(fhi) ||
        throw(DomainError((flo, fhi), "calibration root not bracketed"))

    # Preserve the point with the smallest *economic* residual, including
    # when bisection reaches machine precision before the residual target.
    best_x, best_f = abs(flo) <= abs(fhi) ? (lo, flo) : (hi, fhi)
    for _ in 1:maxiter
        mid = lo + (hi-lo)/2
        if mid == lo || mid == hi
            break
        end
        fm = f(mid)
        isfinite(fm) || throw(DomainError(fm, "nonfinite calibration moment"))
        if abs(fm) < abs(best_f)
            best_x, best_f = mid, fm
        end
        abs(fm) <= ftol && return mid
        if signbit(fm) == signbit(flo)
            lo, flo = mid, fm
        else
            hi, fhi = mid, fm
        end
        # `xtol` only ends the bracket search, not the residual requirement.
        # The outer caller makes an independent asset-market check.
        if hi-lo <= xtol * max(1.0, abs(mid))
            break
        end
    end
    return best_x
end

"""Numerically differentiate a scalar/vector function with a symmetric difference."""
function _derivative(f, x::Real, h::Real)
    h > 0 || throw(ArgumentError("positive derivative step required"))
    return (f(x+h) - f(x-h))/(2h)
end

"""
    calibrated_household(setup, model, parameter; beta=nothing)

For each proposed *outer* parameter, solve the *inner* asset-market clearing
restriction from household FOCs. For four analytic households, this is closed
form. HA finds its own discount factor via EGM and the invariant distribution.
"""
function calibrated_household(s::CalibrationSetup, model::Symbol, parameter::Real;
                              beta=nothing)
    model in MODELS || throw(ArgumentError("unknown household model: $model"))
    r = s.state.r_ante
    Z = production_steady_state(s.state.Y, r, s.core.production).w -
        (s.core.fiscal.G + r*s.state.B)
    pp = production_steady_state(s.state.Y, r, s.core.production).pure_profit
    Ztotal = savers_income(Z, pp, s.core.mu)
    A = s.saver_assets
    C = s.saver_consumption
    mu = s.core.mu
    if model === :bu
        b = isnothing(beta) ? s.beta_initial : beta
        v0 = (1-b*(1+r))*C^(-1.0)
        return BUQuadratic(; beta=b, eis=1.0, vp_0=v0,
                           vp_slope=parameter, A_ss=A, lambda_bu=s.lambda_bu)
    elseif model === :bu_log
        # v′(A_ss) = vp_slope: A_ss cancels in the log marginal utility.
        b = (1 - parameter*C)/(1+r)
        return BULog(; beta=b, eis=1.0, vp_slope=parameter,
                      A_ss=A, lambda_bu=s.lambda_bu)
    elseif model === :olg
        survival = parameter
        H = Ztotal/(1-(1-0.0)*survival/(1+r))
        mpc = C/((1+r)*A+H)
        b = (1-mpc)/survival
        return OLGHousehold(; beta=b, survival, zeta=0.0)
    elseif model === :olg_alt
        zeta = parameter
        b = s.beta_initial
        # Solve the same stationary Blanchard-Yaari restriction for survival;
        # the author's reduced-form OLG-alt survival can exceed one.
        f(survival) = begin
            denom = 1 - (1-zeta)*survival/(1+r)
            if denom <= 0 || 1 - survival*b <= 0
                return NaN
            end
            H = Ztotal/denom
            return (1-survival*b)*((1+r)*A+H) - C
        end
        # Local secant/Newton is preferable to a probability-constrained bracket.
        initial = (s.country === :us ? 0.983135682233173 :
                   0.997770183804933^(4/s.periods))
        survival = _newton_scalar(f, initial; lower=0.01,
                                   upper=min((1+r)/(1-zeta),1/b)*0.999999)
        return OLGHousehold(; beta=b, survival, zeta)
    else
        eis = parameter
        eis > 0 || throw(DomainError(eis,"HA EIS must be positive"))
        lower = 0.9^(1/s.periods)
        upper = 1/(1+r) - 0.003
        function demand(b)
            h = HAHousehold(; beta=b, eis, lumpsum=0.0,
                  rho_e=s.ha_rho, sd_e=0.92, n_e=7, n_a=300,
                  min_a=0.0, max_a=500.0,
                  source_lottery_extrapolation=true)
            return h((; Z, r, r_ante=r, pure_profit=pp, mu)).A_ha - A
        end
        # The official package uses unbounded interpolation weights on the
        # [0, 500] grid. Japan quarterly's saved steady state *requires* this
        # source behavior: some policies exceed 500 by about 1.6. Keep it
        # explicit, without changing the grid or silently clipping policies.
        # The GE finite perturbation is only 1e-5 in r. A loose beta root
        # can leave a 1e-6-level baseline asset-market residual, which then
        # contaminates Δlog(B) by an amount comparable to the true response.
        # Stop on the *asset-clearing residual*, not on a 1e-10-wide beta
        # interval. Near the root, such an interval can still imply a demand
        # error of 1e-7 or more. This applies both to final and trial EIS
        # values, so outer line searches can evaluate valid candidates.
        bh = _bisect(demand, lower, upper;
                     ftol=1e-9, xtol=1e-14, maxiter=100)
        demand_error = demand(bh)
        abs(demand_error) <= 5e-8 ||
            error("HA inner asset-market calibration residual $(demand_error) exceeds 5e-8; stationary demand may be numerically discontinuous")
        return HAHousehold(; beta=bh, eis, lumpsum=0.0,
                 rho_e=s.ha_rho, sd_e=0.92, n_e=7, n_a=300,
                 min_a=0.0, max_a=500.0,
                 source_lottery_extrapolation=true)
    end
end

"""Damped finite-difference scalar Newton with strict economic-domain validation."""
function _newton_scalar(f, x0; lower=-Inf, upper=Inf, tol=1e-11, maxiter=60)
    x = Float64(x0)
    lower < x < upper || throw(DomainError(x,"initial guess out of domain"))
    for _ in 1:maxiter
        fx = f(x)
        isfinite(fx) || throw(DomainError(fx,"invalid scalar calibration point"))
        abs(fx) < tol && return x
        h = 1e-5 * max(1,abs(x))
        # one-sided fallback at finite boundaries
        xp = min(x+h, (x+upper)/2)
        xm = max(x-h, (x+lower)/2)
        d = (f(xp)-f(xm))/(xp-xm)
        isfinite(d) && d != 0 || error("ill-conditioned scalar calibration")
        step = fx/d
        accepted = false
        for j in 0:25
            trial = x - step/(2.0^j)
            if lower < trial < upper
                ft = try f(trial) catch; NaN end
                if isfinite(ft) && abs(ft) < abs(fx)
                    x = trial
                    accepted = true
                    break
                end
            end
        end
        accepted || error("scalar calibration line search failed")
    end
    error("scalar calibration did not converge")
end

"""Update the wage forward coefficient without altering the economic regime."""
function _model_core(s::CalibrationSetup, hh)
    core = s.core
    w0 = core.wage
    survival = hh isa OLGHousehold ? hh.survival : one(w0.survival)
    w1 = WageParams(; kappa=w0.kappa, beta=hh.beta,
            survival, labor_shifter=w0.labor_shifter,
            lag=w0.lag, forward=w0.forward, frisch=w0.frisch)
    return CoreParams(; production=core.production, wage=w1,
            monetary=core.monetary, fiscal=core.fiscal,
            mu=core.mu, delta_b=core.delta_b)
end

"""Compute a baseline stationary GE from *calibrated* household FOCs."""
function baseline_equilibrium(s::CalibrationSetup, hh)
    core = _model_core(s,hh)
    eq = evaluate_steady_state(s.state,core,hh)
    return (; eq, core, household=hh)
end

"""Partial-equilibrium derivatives A_Z, C_Z and C_r, original auxfs.py."""
function household_derivatives(eq, hh, core; h=1e-4)
    input = (; Z=eq.Z, r=eq.r, r_ante=eq.r_ante,
              pure_profit=eq.pure_profit, mu=core.mu,
              Y=eq.Y, B=eq.B)
    A_Z = (hh(merge(input,(;Z=input.Z+h))).A - eq.A)/h
    C_Z = (hh(merge(input,(;Z=input.Z+h))).C - eq.C)/h
    C_r = (hh(merge(input,(;r=input.r+h))).C - eq.C)/h
    return (; A_Z, C_Z, C_r, ratio=A_Z/C_Z)
end

"""
    household_inverse_asset_semi_elasticity(s, hh; h=1e-4)

The source auxfs.semi_elasticity_r_B_hh. BU-quad holds SAVER C_ha
fixed and solves Euler + budget for (A_ha,Z). Other models shift
both r and r_ante and choose Z to hold aggregate C constant.
This household partial-equilibrium object differs from GE phi.
"""
function household_inverse_asset_semi_elasticity(s::CalibrationSetup,hh;h=1e-4)
    base=baseline_equilibrium(s,hh)
    eq=base.eq
    Z1 = eq.Z
    if hh isa BUQuadratic
        hh.vp_slope != 0 || throw(DomainError(hh.vp_slope,"zero BU slope"))
        vp1=(1-hh.beta*(1+eq.r_ante+h))*eq.C_ha^(-1/hh.eis)
        A1_ha=hh.A_ss+(hh.vp_0-vp1)*hh.A_ss/hh.vp_slope
        Z1=eq.C_ha - (eq.r+h)*A1_ha - eq.pure_profit/(1-s.core.mu)
        A1=(1-s.core.mu)*A1_ha
    else
        f(z)=begin
            inp=(;Z=z,r=eq.r+h,r_ante=eq.r_ante+h,
                  pure_profit=eq.pure_profit,mu=s.core.mu,
                  Y=eq.Y,B=eq.B)
            hh(inp).C-eq.C
        end
        Z1=_newton_scalar(f,Z1;tol=1e-8)
        inp=(;Z=Z1,r=eq.r+h,r_ante=eq.r_ante+h,
              pure_profit=eq.pure_profit,mu=s.core.mu,
              Y=eq.Y,B=eq.B)
        A1=hh(inp).A
    end
    abs(A1-eq.A)>1e-14 || error("household asset demand insensitive to interest")
    return (; phi=h*eq.A/(A1-eq.A), A_shift=A1, Z_shift=Z1)
end

# Source auxfs.GE_elasticity: r_ante shifted, B and Y endogenous;
# labor_foc target held equal to the baseline (not necessarily zero in Japan).
function _ge_residuals(z, s, hh, core, r, labor_target)
    st = merge(s.state,(; B=z[1], Y=z[2], r_ante=r))
    eq = evaluate_steady_state(st,core,hh)
    return [eq.asset_mkt, eq.labor_foc-labor_target]
end

"""
    ge_elasticity_finite(s, hh; h=-1e-5)

Faithful auxfs.GE_elasticity method: fix piw, perturb r_ante, solve B and Y
using asset clearing and baseline labor_foc. Its `zlb=false` counterfactual
for Japan matches jp_calibration.py. Computes Δr/Δlog B, NOT ∂r/∂log A|C.
"""
function ge_elasticity_finite(s::CalibrationSetup, hh; h=-1e-5, tol=2e-8)
    baseline = baseline_equilibrium(s,hh)
    core = _flex_core(baseline.core)
    r0 = s.state.r_ante
    f(z) = _ge_residuals(z,s,hh,core,r0+h,baseline.eq.labor_foc)
    # An explicit finite-difference Newton step is used for ALL models; the
    # SciML default AD would attempt to differentiate through the HA EGM.
    shifted = _newton_2d(f, [s.state.B, s.state.Y]; tol=tol, maxiter=30)
    residual_norm = norm(f(shifted), Inf)
    B1 = shifted[1]
    B1>0 || throw(DomainError(B1,"GE perturbation produced negative debt"))
    dlogB = log(B1)-log(s.state.B)
    abs(dlogB)>1e-14 || error("GE debt response numerically indistinguishable from zero")
    return (; phi=h/dlogB, B_shift=B1, Y_shift=shifted[2],
             residual_norm, h)
end

function _flex_core(core)
    m = core.monetary
    flex_m = MonetaryParams(; pi_trend=m.pi_trend,
        real_growth=m.real_growth, phi_pi=m.phi_pi,
        real_rate=m.real_rate, i_shock=m.i_shock, zlb=false,
        i_zlb=m.i_zlb)
    return CoreParams(; production=core.production, wage=core.wage,
                      monetary=flex_m, fiscal=core.fiscal,
                      mu=core.mu, delta_b=core.delta_b)
end

"""
    ge_elasticity_implicit(s, hh; steps=(1e-3,1e-3,1e-4))

Implicit function theorem with finite-differenced GE Jacobian:
∂(B,Y)/∂r = -F_{(B,Y)}^{-1}F_r, phi = B/(∂B/∂r).
Useful for comparing a local derivative to the paper's finite-h method.
"""
function ge_elasticity_implicit(s::CalibrationSetup, hh;
                                steps=(1e-3,1e-3,1e-4))
    baseline = baseline_equilibrium(s,hh)
    core = _flex_core(baseline.core)
    v0 = [s.state.B,s.state.Y]
    r0 = s.state.r_ante
    target = baseline.eq.labor_foc
    F(z,r) = _ge_residuals(z,s,hh,core,r,target)
    jac = hcat(
       _derivative(x->F([x,v0[2]],r0),v0[1],steps[1]),
       _derivative(x->F([v0[1],x],r0),v0[2],steps[2]))
    rhs = _derivative(r->F(v0,r),r0,steps[3])
    slopes = -(jac\rhs)
    isfinite(slopes[1]) && abs(slopes[1])>1e-10 ||
        error("near-singular GE bond-supply derivative")
    return (; phi=v0[1]/slopes[1], dB_dr=slopes[1],
             dY_dr=slopes[2], jacobian=jac, rhs)
end

"""Evaluate BOTH BU moments (or just phi) for a proposed outer parameter."""
function calibration_moments(s::CalibrationSetup, model::Symbol, parameter::Real;
                             beta=nothing, method=:finite)
    hh = calibrated_household(s,model,parameter;beta)
    b = baseline_equilibrium(s,hh)
    phi = method === :finite ? ge_elasticity_finite(s,hh).phi :
          method === :implicit ? ge_elasticity_implicit(s,hh).phi :
          throw(ArgumentError("unknown GE derivative method"))
    derivative = household_derivatives(b.eq,hh,b.core)
    return (; phi, phi_error=phi-s.phi_target,
             income_ratio=derivative.ratio,
             ratio_error=derivative.ratio-b.eq.A/b.eq.C,
             eq=b.eq, household=hh, core=b.core,
             household_derivatives=derivative)
end

"""Solve a two-dimensional calibration root with an explicit FD Jacobian."""
function _newton_2d(f, x0; tol=2e-7, maxiter=35)
    x = Float64[x0...]
    length(x)==2 || throw(ArgumentError("two-parameter calibration required"))
    for _ in 1:maxiter
        fx = f(x)
        all(isfinite,fx) || error("nonfinite two-moment calibration")
        norm(fx,Inf)<tol && return x
        dx1,dx2 = 3e-4*max(1,abs(x[1])), 3e-4*max(1,abs(x[2]))
        h1 = (f(x .+ [dx1,0.0])-f(x .- [dx1,0.0]))/(2dx1)
        h2 = (f(x .+ [0.0,dx2])-f(x .- [0.0,dx2]))/(2dx2)
        step = hcat(h1,h2)\fx
        accepted = false
        for j in 0:22
            trial = x .- step/(2.0^j)
            if all(isfinite,trial) && trial[1]>0 && 0<trial[2]<2
                ftrial = try f(trial) catch; [Inf,Inf] end
                if all(isfinite,ftrial) && norm(ftrial,Inf)<norm(fx,Inf)
                    x=trial
                    accepted=true
                    break
                end
            end
        end
        accepted || error("two-moment calibration line search failed")
    end
    error("two-moment calibration did not converge")
end

function _initial_parameter(s::CalibrationSetup, model::Symbol)
    p=s.periods
    if s.country === :us
        return model === :bu ? 0.11694171560510933/p :
               model === :bu_log ? 0.11693451528481619/p :
               model === :olg_alt ? 0.12185450303880266/p :
               model === :olg ? 0.8633361722564332^(1/p) :
               0.212421110417254
    else
        return model === :bu ? 0.012164577243125632*4/p :
               model === :bu_log ? 0.012153196978804324*4/p :
               model === :olg_alt ? 0.014356527688342516*4/p :
               model === :olg ? 0.9834456686973616^(4/p) :
               0.3011275210023721
    end
end

"""
    calibrate_model(setup, model; method=:finite)

Full *moment-based* calibration. Saved parameters are never loaded as output.
For HA, `method=:implicit` avoids repeatedly solving the full GE in every
outer iteration; validate against `ge_elasticity_finite` on the final result.
"""
function calibrate_model(s::CalibrationSetup, model::Symbol;
                         method=:finite, tol=2e-6)
    x0 = _initial_parameter(s,model)
    if model === :bu
        moment_vector = x -> begin
            m = calibration_moments(s,model,x[1];beta=x[2],method)
            return [m.phi_error,m.ratio_error]
        end
        solution = _newton_2d(moment_vector,[x0,s.beta_initial];tol)
        param, beta = solution
        result = calibration_moments(s,model,param;beta,method)
    else
        moment_scalar = x -> calibration_moments(s,model,x;method).phi_error
        # Explicit domain. OLG survival must be below 1+r for finite human wealth.
        lower = model in (:bu_log,:ha,:olg_alt) ? 1e-5 : 0.01
        upper = model === :olg ? (1+s.state.r_ante)*0.999 :
                model === :olg_alt ? 0.85 : model === :ha ? 3.0 : 3.0
        x0 = clamp(x0,lower+1e-5,upper-1e-5)
        param = _newton_scalar(moment_scalar,x0;lower,upper,tol)
        result = calibration_moments(s,model,param;method)
    end
    check_calibration(s,result; tol=max(5e-5,10tol))
    # Check source finite perturbation even when IFT was used during optimization.
    check_phi = ge_elasticity_finite(s,result.household)
    return merge((;country=s.country, periods=s.periods, model,
                    calibrated_parameter=param, phi_finite=check_phi.phi,
                    phi_finite_error=check_phi.phi-s.phi_target),result)
end

"""Economic steady-state and moment checks, including the Japanese ZLB case."""
function check_calibration(s::CalibrationSetup, result; tol=1e-4)
    eq=result.eq
    for name in (:asset_mkt,:goods_mkt,:fisher,:piw_cond,
                 :invcond,:valcond,:qres,:jvaluation,:Jvaluation)
        v=getproperty(eq,name)
        isfinite(v) && abs(v)<tol || error("calibration equilibrium residual $name=$v")
    end
    abs(result.phi_error)<tol || error("GE elasticity calibration missed target")
    if result.household isa BUQuadratic
        abs(result.ratio_error)<tol || error("BU secondary moment missed target")
    end
    isapprox(eq.C,s.state.Y-s.core.fiscal.G-eq.I;atol=tol) ||
        error("goods accounting identity failed")
    if s.country === :jp
        abs(eq.i)<tol || error("Japan binding-ZLB nominal rate is nonzero")
        eq.i_rule < eq.i + tol || error("specified ZLB inconsistent with shadow rate")
    end
    return true
end

"""Calibrate all five models; method=:finite uses original GE perturbations for HA too."""
function calibrate_country(s::CalibrationSetup; method=:finite)
    results = Dict{Symbol,Any}()
    for model in MODELS
        results[model]=calibrate_model(s,model;method)
    end
    return results
end

"""Unrounded economic and household parameter values for quantitative tables."""
function calibration_values(result)
    eq=result.eq
    h=result.household
    return (; country=result.country, periods=result.periods, model=result.model,
        Y=eq.Y, N=eq.N, C=eq.C, A=eq.A, B=eq.B,
        C_ha=eq.C_ha, A_ha=eq.A_ha, A_ss=eq.A_ha,
        MPC_ha=eq.MPC_ha, r_ante=eq.r_ante,
        r=eq.r, i=eq.i, piw=eq.piw, pi=eq.pi, G=result.core.fiscal.G,
        tau=eq.tau, mu=result.core.mu, K=eq.K, Q=eq.Q, I=eq.I,
        alpha=result.core.production.alpha, TFP=result.core.production.tfp,
        beta=h.beta, eis=_value(h,:eis,1.0),
        vp_0=_value(h,:vp_0,missing),
        vp_slope=_value(h,:vp_slope,missing),
        survival=_value(h,:survival,missing),
        zeta=_value(h,:zeta,missing),
        MPC=eq.MPC, phi=result.phi, phi_finite=result.phi_finite,
        kappa=result.core.wage.kappa, kappa_HHNS=eq.kappa_HHNS,
        rho_e=h isa HAHousehold ? 0.9^(1/result.periods) : missing,
        sd_e=h isa HAHousehold ? 0.92 : missing,
        actual_inflation=eq.pi+result.core.monetary.pi_trend,
        labor_shifter=result.core.wage.labor_shifter,
        rho=-log(h.beta), rho_eff=h isa OLGHousehold ? -log(h.beta*h.survival) : missing,
        asset_mkt=eq.asset_mkt, goods_mkt=eq.goods_mkt,
        phi_error=result.phi_error, phi_finite_error=result.phi_finite_error)
end

"""
    write_calibration_values(setup, results; directory=pwd())

Export the *computed* calibration results in lossless Float64 decimal form.
The TSV is a numeric audit artifact, distinct from the three-decimal
publication tables. It never reads or writes original Python pickle data.
"""
function write_calibration_values(s::CalibrationSetup, results;
                                  directory::AbstractString=pwd())
    suffix = s.periods == 4 ? "_quarterly" : "_annual"
    file=joinpath(directory,"calibrated_$(s.country)$(suffix).tsv")
    open(file,"w") do io
        println(io,"model\tvariable\tvalue")
        for model in MODELS
            haskey(results,model) || continue
            vals=calibration_values(results[model])
            for key in propertynames(vals)
                value=getproperty(vals,key)
                if value === missing || value isa Real
                    println(io,model,"\t",key,"\t",
                            value === missing ? "NA" : repr(value))
                end
            end
        end
    end
    return file
end

const COMMON_TABLE = (
    ("Gov. spending to GDP", :G),
    ("Gov. debt to GDP", :B),
    ("Labor tax rate", :tau),
    ("Nominal rate", :i),
    ("Inflation target", :pi_trend),
    ("Actual inflation", :actual_inflation),
    ("Phillips curve slope", :kappa_HHNS),
    ("Frisch elasticity", :frisch),
    ("Effective disutility weight", :labor_shifter),
    ("TFP growth", :real_growth),
    ("Capital adjustment cost", :eps_I),
    ("Capital to GDP", :k),
    ("Capital share", :alpha),
    ("Rate of depreciation", :delta_k),
    ("Markup", :markup),
    ("Fixed cost", :fixed_cost),
    ("Labor income share of savers", :top_10_share))

function _common_value(s,result,k::Symbol)
    eq=result.eq
    p=s.core.production
    return k === :G ? s.core.fiscal.G :
           k === :pi_trend ? s.core.monetary.pi_trend :
           k === :actual_inflation ? eq.piw+s.core.monetary.pi_trend :
           k === :kappa_HHNS ? eq.kappa_HHNS :
           k === :frisch ? s.core.wage.frisch :
           k === :labor_shifter ? eq.kappa_HHNS/(s.core.wage.kappa*(1+1/s.core.wage.frisch)) :
           k === :real_growth ? p.real_growth :
           k === :eps_I ? p.eps_I :
           k === :k ? eq.k :
           k === :alpha ? p.alpha :
           k === :delta_k ? p.delta_k :
           k === :markup ? p.markup :
           k === :fixed_cost ? p.fixed_cost :
           k === :top_10_share ? s.top_10_share : getproperty(eq,k)
end

"""
    make_calibration_tables(setup, models; directory=pwd())

Use SummaryTables.jl's documented low-level `Table`/`Cell` API and LaTeX
show method. This is presentation only: no numerical values are rounded
until conversion to strings. Four source-equivalent country tables can be
built by calling once for each country.
"""
function make_calibration_tables(s::CalibrationSetup, models;
                                 directory::AbstractString=pwd())
    haskey(models,:bu) || throw(ArgumentError("BU reference required for common table"))
    name = s.country === :us ? "us" : "japan"
    suffix = s.periods == 4 ? "_quarterly" : ""
    common = Matrix{String}(undef,length(COMMON_TABLE)+1,2)
    common[1,:]=["Variable","Value"]
    for (i,(label,key)) in enumerate(COMMON_TABLE)
        common[i+1,1]=label
        common[i+1,2]=@sprintf("%.3f",_common_value(s,models[:bu],key))
    end
    ctable = Table(Cell.(common);header=1)
    model_keys = (:bu,:bu_log,:olg,:ha)
    fields = (("Discount rate",:rho), ("Effective discount rate",:rho_eff),
              ("Intertemporal elasticity of substitution",:eis),
              ("Conv. yield level",:vp_0), ("Conv. yield slope",:vp_slope),
              ("Survival probability",:survival),
              ("Persistence of income shock",:rho_e),
              ("Standard deviation of income distribution",:sd_e))
    mat=Matrix{String}(undef,length(fields)+1,length(model_keys)+1)
    mat[1,1] = "Variable"
    for (j, model) in enumerate(model_keys)
        mat[1,j+1] = string(model)
    end
    for (i,(label,key)) in enumerate(fields)
        mat[i+1,1]=label
        for (j,mod) in enumerate(model_keys)
            r=models[mod]
            h=r.household
            v=key===:rho ? -log(h.beta) :
              key===:rho_eff && h isa OLGHousehold ? -log(h.beta*h.survival) :
              key===:eis ? _value(h,:eis,1.0) :
              key===:rho_e && h isa HAHousehold ? s.ha_rho :
              key===:sd_e && h isa HAHousehold ? 0.92 :
              _value(h,key,missing)
            mat[i+1,j+1]=v===missing ? "---" : @sprintf("%.3f",v)
        end
    end
    mtable=Table(Cell.(mat);header=1)
    # Path names correspond to source spreadsheets, with explicit frequency.
    common_file=joinpath(directory,"tab2_$(name)_common_calibration$(suffix).tex")
    model_file=joinpath(directory,"tab3+A3_$(name)_calibration$(suffix).tex")
    open(common_file,"w") do io
        show(io,MIME"text/latex"(),ctable)
    end
    open(model_file,"w") do io
        show(io,MIME"text/latex"(),mtable)
    end
    return (; common_file,model_file,common_table=ctable,model_table=mtable)
end

end # module GoldilocksCalibration
