module GoldilocksCore

using LinearAlgebra
using ForwardDiff
using NonlinearSolve

export ProductionParams, WageParams, MonetaryParams, FiscalParams, CoreParams,
       production_block, production_steady_state,
       equity_block, equity_steady_state,
       long_bond_block, long_bond_steady_state, combined_return,
       fiscal_block, monetary_block, wage_phillips_block,
       labor_shifter_for_phillips, inflation_block, market_clearing, tax_rule_residuals,
       evaluate_steady_state, equilibrium_residuals,
       solve_nonlinear, solve_core_steady_state, equilibrium_jacobian,
       check_equilibrium

# All interest rates and growth rates below are per model period, NOT percentages.
# The interface follows the detrended variables in Appendix G.2 and blocks.py.

"""
    ProductionParams(; alpha, tfp, fixed_cost, markup, delta_k,
                      eps_I, real_growth, risk_premium=0.0)

Parameters for the Cobb-Douglas producer and the capital investment block.
`tfp` is the coefficient named `TFP` in the original Python code.
"""
struct ProductionParams{T<:Real}
    alpha::T
    tfp::T
    fixed_cost::T
    markup::T
    delta_k::T
    eps_I::T
    real_growth::T
    risk_premium::T
end

function ProductionParams(; alpha, tfp, fixed_cost, markup, delta_k,
                            eps_I, real_growth, risk_premium=0.0)
    vals = promote(alpha, tfp, fixed_cost, markup, delta_k,
                   eps_I, real_growth, risk_premium)
    return ProductionParams(vals...)
end

"""
    WageParams(; kappa, beta, labor_shifter, lag=0.0,
                forward=1.0, frisch=1.0, survival=1.0)

Parameters of the linearized wage Phillips curve in `blocks.py`.
The paper's nonlinear Rotemberg equation is not silently substituted here.
"""
struct WageParams{T<:Real}
    kappa::T
    lag::T
    forward::T
    beta::T
    frisch::T
    survival::T
    labor_shifter::T
end

function WageParams(; kappa, beta, labor_shifter, lag=0.0,
                      forward=1.0, frisch=1.0, survival=1.0)
    vals = promote(kappa, lag, forward, beta, frisch, survival,
                   labor_shifter)
    return WageParams(vals...)
end

"""
    MonetaryParams(; phi_pi=1.5, real_rate=1.0, pi_trend,
                    real_growth, i_shock=0.0, zlb=false, i_zlb=0.0)

`zlb=true` fixes the nominal rate at `i_zlb` as the original code does;
it does not replace the monetary condition by `max(i, 0)`.
"""
struct MonetaryParams{T<:Real}
    phi_pi::T
    real_rate::T
    pi_trend::T
    real_growth::T
    i_shock::T
    i_zlb::T
    zlb::Bool
end

function MonetaryParams(; pi_trend, real_growth, phi_pi=1.5,
                          real_rate=1.0, i_shock=0.0,
                          zlb::Bool=false, i_zlb=0.0)
    vals = promote(phi_pi, real_rate, pi_trend, real_growth,
                   i_shock, i_zlb)
    return MonetaryParams(vals..., zlb)
end

"""
    FiscalParams(; G, periods=1, adjust_tax_rate=false,
                  psi=0.0, deficit_ss=0.0, B_ss=0.0,
                  tax_shock=0.0, tau_slope=0.0)

`G` is GOVERNMENT SPENDING, not the nominal GDP growth rate.
`B` is debt in units of output per model period; `B/periods` is
reported as an annual-GDP debt ratio in the original replication.
"""
struct FiscalParams{T<:Real}
    G::T
    psi::T
    deficit_ss::T
    B_ss::T
    tax_shock::T
    tau_slope::T
    periods::Int
    adjust_tax_rate::Bool
end

function FiscalParams(; G, periods::Integer=1, adjust_tax_rate::Bool=false,
                        psi=0.0, deficit_ss=0.0, B_ss=0.0,
                        tax_shock=0.0, tau_slope=0.0)
    periods > 0 || throw(ArgumentError("periods must be positive"))
    vals = promote(G, psi, deficit_ss, B_ss, tax_shock, tau_slope)
    return FiscalParams(vals..., Int(periods), adjust_tax_rate)
end

"""
    CoreParams(; production, wage, monetary, fiscal, mu, delta_b=0.0)

Common non-household parameters. `mu` is the population share of
hand-to-mouth households; `delta_b` is the decay of long nominal bonds.
"""
struct CoreParams{P,W,M,F,T<:Real}
    production::P
    wage::W
    monetary::M
    fiscal::F
    mu::T
    delta_b::T
end

function CoreParams(; production::ProductionParams, wage::WageParams,
                      monetary::MonetaryParams, fiscal::FiscalParams,
                      mu, delta_b=0.0)
    mu1, d1 = promote(mu, delta_b)
    return CoreParams(production, wage, monetary, fiscal, mu1, d1)
end

"""
    production_block(Y, k_lag, k, Q, Q_next, r_ante, p)

One-period capital-production block from `production_CD_raw`.
`k_lag` is capital installed at t-1, `k` capital chosen at t,
`Q_next` the value of Q at t+1. `r_ante` is the detrended
one-period REAL net return, not the nominal government bond yield.
Returns accounting quantities and investment/valuation residuals.
"""
function production_block(Y, k_lag, k, Q, Q_next, r_ante,
                          p::ProductionParams)
    Y_total = Y + p.fixed_cost
    N = (Y_total / (p.tfp * k_lag^p.alpha))^(1 / (1 - p.alpha))
    mpk = p.alpha * Y_total / k_lag
    mpn = (1 - p.alpha) * Y_total / N
    w = mpn / p.markup

    kratio = k / k_lag
    xi = (1 + p.real_growth) * kratio - (1 - p.delta_k) +
         (kratio - 1)^2 / (2 * p.delta_k * p.eps_I)
    xiprime = 1 + p.real_growth +
              (kratio - 1) / (p.delta_k * p.eps_I)

    gross_required_return = 1 + r_ante + p.risk_premium
    invcond = Q_next / gross_required_return - xiprime
    valcond = mpk / p.markup - xi +
              kratio * Q_next / gross_required_return - Q

    I = (1 + p.real_growth) * k - k_lag + p.delta_k * k_lag
    adj_cost = (kratio - 1)^2 * k_lag / (2 * p.delta_k * p.eps_I)
    pure_profit = (1 - 1 / p.markup) * Y_total - p.fixed_cost
    dividend = Y_total - w * N - p.fixed_cost - I - adj_cost - pure_profit

    return (; N, w, I, adj_cost, dividend, mpk, mpn, pure_profit,
            K=k, k, Q, kratio, xi, xiprime, invcond, valcond)
end

"""
    production_steady_state(Y, r_ante, p)

Analytical steady-state version of `production_CD_raw_ss`.
Solves the investor's two first-order conditions for capital and Q
conditional on aggregate output and the required return.
"""
function production_steady_state(Y, r_ante, p::ProductionParams)
    gross_required_return = 1 + r_ante + p.risk_premium
    kratio = one(gross_required_return)
    xi = (1 + p.real_growth) * kratio - (1 - p.delta_k)
    xiprime = 1 + p.real_growth
    Q = gross_required_return * xiprime
    denominator = xi - kratio * Q / gross_required_return + Q
    K = p.alpha * (Y + p.fixed_cost) / (p.markup * denominator)
    result = production_block(Y, K, K, Q, Q, r_ante, p)
    dyn_ineff_cond = result.mpk - p.delta_k - p.real_growth
    return merge(result, (; dyn_ineff_cond))
end

"""
    equity_block(J, j_lag, j, J_next, dividend, r_ante, risk_premium)

The `equity` block of the original Python replication. `J` is
ex-dividend cum-return value today and `j` is its ex-dividend price.
"""
function equity_block(J, j_lag, j, J_next, dividend,
                      r_ante, risk_premium)
    gross_required_return = 1 + r_ante + risk_premium
    rpost_equity = J / j_lag - 1
    jvaluation = J_next / gross_required_return - j
    Jvaluation = dividend + J_next / gross_required_return - J
    return (; rpost_equity, jvaluation, Jvaluation)
end

"""
    equity_steady_state(k, dividend, r_ante, p)

Steady-state capitalization consistent with the original investor block.
In equilibrium `j = (1 + real_growth) * k`. The returned valuation
residuals expose any inconsistency in the supplied dividend.
"""
function equity_steady_state(k, dividend, r_ante,
                             p::ProductionParams)
    j = (1 + p.real_growth) * k
    J = (1 + r_ante + p.risk_premium) * j
    values = equity_block(J, j, j, J, dividend,
                          r_ante, p.risk_premium)
    return merge((; j, J), values)
end

"""
    long_bond_block(q_lag, q, q_next, r_ante, delta_b,
                    pi, pi_e_lag, pi_e, pi_e_next)

Price equation and realized return for nominal geometrically decaying
long bonds from `longbonds_price`. Net inflation inputs are deviations
from trend, exactly as in the original Python implementation.
"""
function long_bond_block(q_lag, q, q_next, r_ante, delta_b,
                         pi, pi_e_lag, pi_e, pi_e_next)
    qres = q - (1 + delta_b * q_next / (1 + pi_e_next)) /
                (1 + r_ante)
    rpost_bonds = ((1 + delta_b * q / (1 + pi_e)) / q_lag) *
                  ((1 + pi_e_lag) / (1 + pi)) - 1
    return (; qres, rpost_bonds)
end

"""
    long_bond_steady_state(r_ante, pi, delta_b)

Closed-form solution of the stationary long-bond pricing equation.
A positive bond price requires `1 + r_ante > delta_b/(1 + pi)`.
"""
function long_bond_steady_state(r_ante, pi, delta_b)
    denominator = 1 + r_ante - delta_b / (1 + pi)
    denominator > zero(denominator) ||
        throw(DomainError(denominator, "stationary bond price must be positive"))
    q = inv(denominator)
    result = long_bond_block(q, q, q, r_ante, delta_b,
                             pi, pi, pi, pi)
    return merge((; q), result)
end

"""
    combined_return(B_lag, j_lag, rpost_bonds, rpost_equity)

Ex-post net return on total saver assets, using START-of-period
bond and equity market values as weights (`combine_r` in blocks.py).
"""
function combined_return(B_lag, j_lag, rpost_bonds, rpost_equity)
    gross = (B_lag * (1 + rpost_bonds) +
             j_lag * (1 + rpost_equity)) / (B_lag + j_lag)
    return gross - 1
end

"""
    fiscal_block(N, w, B_lag, B, rpost_bonds, p)

Endogenous taxes given the debt path; `G` in `p` is government spending.
The debt law is `B = G + (1+rpost_bonds)*B_lag - tax`.
Returns the primary deficit and fiscal-rule residual `deficit_diff`.
"""
function fiscal_block(N, w, B_lag, B, rpost_bonds,
                      p::FiscalParams)
    tax = p.G + (1 + rpost_bonds) * B_lag - B
    base = p.adjust_tax_rate ? w * N : one(w * N)
    tau = tax / base
    Z = w * N - tax
    deficit = p.G - tax
    deficit_diff = p.deficit_ss - p.psi * (B_lag - p.B_ss) - deficit
    B_annual = B / p.periods
    return (; tax, Z, deficit, tau, deficit_diff, B_annual)
end

"""
    monetary_block(pi, pi_next, r_ante, p)

The original `monetary` block, including its exact Fisher residual.
`p.zlb` is an exogenously selected binding regime. Validity of that
regime (inflation relative to its target) must be checked separately.
"""
function monetary_block(pi, pi_next, r_ante, p::MonetaryParams)
    target_factor = p.real_rate * (1 + pi_next + p.pi_trend) +
                    (1 - p.real_rate) * (1 + pi)^p.phi_pi *
                    (1 + p.pi_trend)
    i_rule = (1 + r_ante + p.real_growth) * target_factor - 1 +
             p.i_shock
    i = p.zlb ? p.i_zlb : i_rule
    fisher = (1 + i) / (1 + pi_next + p.pi_trend) - 1 -
             p.real_growth - r_ante
    return (; i, i_rule, fisher)
end

"""
    wage_phillips_block(piw_lag, piw, piw_next, N, Z, C, p)

Linearized wage Phillips curve used in `blocks.py`, distinct from
the nonlinear Rotemberg equation (42) in the printed paper.
The source code fixes the EIS exponent in `labor_foc` at one.
"""
function wage_phillips_block(piw_lag, piw, piw_next,
                            N, Z, C, p::WageParams)
    labor_foc = p.labor_shifter * N^(1 + 1 / p.frisch) - Z / C
    piw_cond = (1 - p.lag) * p.kappa *
               (1 - p.survival * p.beta * p.forward) * labor_foc +
               p.lag * piw_lag +
               p.forward * p.survival * p.beta * piw_next - piw
    kappa_HHNS = p.kappa * p.labor_shifter * (1 + 1 / p.frisch)
    return (; piw_cond, labor_foc, kappa_HHNS)
end

"""
    labor_shifter_for_phillips(piw, N, Z, C, p)

Solve analytically for the labor-disutility weight making the stationary
wage Phillips-curve residual zero, as done numerically in both original
country-calibration scripts. The other wage parameters remain fixed.
"""
function labor_shifter_for_phillips(piw, N, Z, C, p::WageParams)
    slope = (1 - p.lag) * p.kappa *
            (1 - p.survival * p.beta * p.forward)
    iszero(slope) && throw(DomainError(slope,
        "labor shifter is not uniquely identified with zero Phillips slope"))
    offset = (p.lag + p.forward * p.survival * p.beta - 1) * piw
    return (Z / C - offset / slope) / N^(1 + 1 / p.frisch)
end

"""
    inflation_block(piw, piw_next, w_lag, w, w_next)

Price inflation net of trend and the next-period inflation object
`pi_e`, matching `phillips_curve_extra` in the original code.
"""
function inflation_block(piw, piw_next, w_lag, w, w_next)
    pi = (1 + piw) * w_lag / w - 1
    pi_e = (1 + piw_next) * w / w_next - 1
    return (; pi, pi_e)
end

"""
    market_clearing(A, B, C, G, Y, j, adj_cost, I)

Asset and goods market residuals. `G` denotes government spending.
`A` denotes the aggregate household asset position, after multiplying
saver and spender outcomes by their population weights.
"""
function market_clearing(A, B, C, G, Y, j, adj_cost, I)
    asset_mkt = A - B - j
    goods_mkt = C + G + adj_cost + I - Y
    return (; asset_mkt, goods_mkt)
end

"""
    tax_rule_residuals(tax, w, N, Y, reference, fiscal)

The two optional tax-policy targets of `market_clearing` in blocks.py.
`reference` must provide `tax`, `w`, `N`, `Y` for the initial steady state.
"""
function tax_rule_residuals(tax, w, N, Y,
                            reference::NamedTuple, fiscal::FiscalParams)
    tax_diff = tax - reference.tax - fiscal.tax_shock
    tax_rate_diff = tax / (w * N) -
                    reference.tax / (reference.w * reference.N) -
                    fiscal.tau_slope * (Y - reference.Y) / reference.Y
    return (; tax_diff, tax_rate_diff)
end

"""
    evaluate_steady_state(state, params, household; reference=nothing)

Evaluate all non-household blocks in a candidate steady state.
Required `state` fields: `Y`, `B`, `r_ante`, `piw`.

`household` is a callable receiving a NamedTuple with fields
`Z`, `r`, `r_ante`, `pure_profit`, `mu`, `Y`, `B`. It must return a
NamedTuple with at least aggregate `C` and `A` fields. This is a real
extension interface, NOT a substitute for solving BU/OLG/HA households.
Other household fields (e.g. `MPC`) are preserved in the result.

The terminal stationary simplifications are `k_lag=k`, `Q_next=Q`,
`B_lag=B`, `j_lag=j`, `piw_lag=piw_next=piw`, `w_lag=w_next=w`.
If `reference` is supplied, tax-policy targets are also evaluated.
"""
function evaluate_steady_state(state::NamedTuple, p::CoreParams,
                               household; reference=nothing)
    prod = production_steady_state(state.Y, state.r_ante, p.production)
    equity = equity_steady_state(prod.k, prod.dividend,
                                 state.r_ante, p.production)
    inflation = inflation_block(state.piw, state.piw,
                                prod.w, prod.w, prod.w)
    bonds = long_bond_steady_state(state.r_ante, inflation.pi,
                                   p.delta_b)
    r = combined_return(state.B, equity.j, bonds.rpost_bonds,
                        equity.rpost_equity)
    fiscal = fiscal_block(prod.N, prod.w, state.B, state.B,
                          bonds.rpost_bonds, p.fiscal)
    hh_inputs = (; Z=fiscal.Z, r, r_ante=state.r_ante,
                 pure_profit=prod.pure_profit, mu=p.mu,
                 Y=state.Y, B=state.B)
    hh = household(hh_inputs)
    hh isa NamedTuple || throw(ArgumentError("household must return a NamedTuple"))
    hasproperty(hh, :C) && hasproperty(hh, :A) ||
        throw(ArgumentError("household must return C and A"))

    wage = wage_phillips_block(state.piw, state.piw, state.piw,
                               prod.N, fiscal.Z, hh.C, p.wage)
    monetary = monetary_block(inflation.pi, inflation.pi,
                              state.r_ante, p.monetary)
    markets = market_clearing(hh.A, state.B, hh.C, p.fiscal.G,
                              state.Y, equity.j, prod.adj_cost, prod.I)

    result = merge((; Y=state.Y, B=state.B, r_ante=state.r_ante,
                    piw=state.piw), prod, equity, inflation, bonds,
                   (; r), fiscal, hh, wage, monetary, markets)
    if reference !== nothing
        tax_res = tax_rule_residuals(fiscal.tax, prod.w, prod.N,
                                     state.Y, reference, p.fiscal)
        return merge(result, tax_res)
    end
    return result
end

"""
    equilibrium_residuals(equilibrium, targets)

Return a residual vector in the exact order requested by the solver.
`targets` is a tuple of field names such as
`(:asset_mkt, :labor_foc, :deficit_diff)`.
"""
function equilibrium_residuals(eq::NamedTuple,
                               targets::Tuple{Vararg{Symbol}})
    return [getproperty(eq, key) for key in targets]
end

"""
    solve_nonlinear(f, u0; algorithm=TrustRegion(),
                    abstol=1e-10, reltol=1e-10, maxiters=500)

Solve the SQUARE nonlinear residual system `f(u)=0` using a trust-region
method. The final residual must satisfy the ABSOLUTE `abstol` threshold,
regardless of the magnitude of the unknowns. `reltol` is used by the
underlying solver only. Reject nonfinite starting points and residuals.
"""
function solve_nonlinear(f, u0::AbstractVector;
                         algorithm=TrustRegion(),
                         abstol::Real=1e-10, reltol::Real=1e-10,
                         maxiters::Integer=500)
    length(u0) > 0 || throw(ArgumentError("u0 must be nonempty"))
    abstol > 0 || throw(ArgumentError("abstol must be positive"))
    reltol >= 0 || throw(ArgumentError("reltol must be nonnegative"))
    maxiters > 0 || throw(ArgumentError("maxiters must be positive"))
    initial = collect(float.(u0))
    all(isfinite, initial) ||
        throw(ArgumentError("initial unknowns must be finite"))
    initial_residual = f(initial)
    length(initial_residual) == length(initial) ||
        throw(DimensionMismatch("number of targets must equal number of unknowns"))
    all(isfinite, initial_residual) ||
        throw(ArgumentError("initial residual must be finite"))
    problem = NonlinearProblem((u, _) -> f(u), initial)
    sol = solve(problem, algorithm; abstol, reltol, maxiters)
    residual = f(sol.u)
    residual_norm = norm(residual, Inf)
    # The independent economic-equation audit uses an ABSOLUTE residual
    # threshold. In particular, large debt/asset unknowns must not relax it.
    if !(all(isfinite, sol.u) && isfinite(residual_norm) &&
         residual_norm <= abstol)
        error("Nonlinear solver did not meet the absolute residual tolerance; " *
              "retcode=$(sol.retcode), max residual=$(residual_norm)")
    end
    return (; u=sol.u, residual, residual_norm, retcode=sol.retcode)
end

"""
    solve_core_steady_state(initial, params, household, unknowns, targets;
                            reference=nothing, kwargs...)

Solve any selected square subset of common steady-state residual equations.
The user supplies the model-specific household evaluator and the names of
unknowns and targets. Examples of targets in `spending_shocks.py` are
`(:asset_mkt, :labor_foc, :deficit_diff)` off the ZLB and
`(:asset_mkt, :fisher, :deficit_diff, :piw_cond)` at the ZLB.
"""
function solve_core_steady_state(initial::NamedTuple, p::CoreParams,
                                 household,
                                 unknowns::NTuple{N,Symbol},
                                 targets::NTuple{N,Symbol};
                                 reference=nothing, kwargs...) where {N}
    N > 0 || throw(ArgumentError("at least one unknown is required"))
    u0 = [getproperty(initial, key) for key in unknowns]
    make_state(u) = merge(initial, NamedTuple{unknowns}(Tuple(u)))
    residual(u) = equilibrium_residuals(
        evaluate_steady_state(make_state(u), p, household;
                              reference), targets)
    solution = solve_nonlinear(residual, u0; kwargs...)
    final_state = make_state(solution.u)
    eq = evaluate_steady_state(final_state, p, household; reference)
    return (; equilibrium=eq, solution)
end

"""
    equilibrium_jacobian(f, u)

Compute an exact (up to floating-point arithmetic) forward-mode Jacobian
of a vector-valued equilibrium residual mapping, for small/medium systems.
"""
function equilibrium_jacobian(f, u::AbstractVector)
    return ForwardDiff.jacobian(f, u)
end

"""
    check_equilibrium(eq; keys=(...), atol=1e-8)

Check only the specified equilibrium equations. Tax-rule targets and the
labor-supply FOC are not automatically imposed because the selection
depends on the policy regime and calibration experiment.
"""
function check_equilibrium(eq::NamedTuple;
                           keys=(:asset_mkt, :goods_mkt, :invcond,
                                 :valcond, :qres, :jvaluation,
                                 :Jvaluation, :fisher, :piw_cond),
                           atol::Real=1e-8)
    atol >= 0 || throw(ArgumentError("atol must be nonnegative"))
    for key in keys
        value = getproperty(eq, key)
        if !isfinite(value) || abs(value) > atol
            error("Equilibrium residual $(key) = $(value) exceeds $(atol)")
        end
    end
    return true
end

end # module GoldilocksCore
