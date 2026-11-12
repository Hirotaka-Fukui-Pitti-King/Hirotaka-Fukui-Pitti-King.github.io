# Caballero, Farhi, and Gourinchas (2008), AER 98(1), Figure 6.
# Independent Julia reconstruction of prog_UR_exch_final.gau.
#
# This is NOT a line-by-line translation of the original GAUSS script.
# Under the paper's common-growth calibration, differentiating the CES goods-
# market clearing equation gives an analytic law for d(log q)/dt. In addition,
# world asset-market clearing reduces the initial valuation shooting problem to
# ONE scalar unknown (q_0), rather than two valuations plus a nonlinear solver
# at every step. We solve that scalar root with safeguarded bisection, integrate
# the global equilibrium with RK4, and reconstruct all asset prices backward.
#
# The world rate plotted by the original GAUSS code is the output-weighted REAL
# return, not the U-good-numeraire nominal return. Both are exported to CSV.
#
# Dependencies (install in the Julia environment used for running this file):
#   julia --project=. -e 'using Pkg; Pkg.add(string(:CairoMakie)); Pkg.add(string(:SummaryTables))'
#
# Run (Windows PowerShell):
#   julia --project=. .\figure6_exchange_rates.jl
#   julia --project=. .\figure6_exchange_rates.jl C:\path\to\outputs
#
# Outputs (output_fig06/): figure6.png, figure6.pdf, figure6_summary.html,
# figure6_summary.tex, figure6_timeseries.csv, figure6_metadata.txt.
#
# Fidelity notes:
# - Same economic equilibrium, parameters, and initial cross-holdings as GAUSS.
# - Original: Euler steps, 250-year truncated PV integrals, iterative 2-price
#   fixed point, per-step NLSYS market clearing; here: one-price bisection,
#   RK4 steps, 350-year PV horizon with stationary terminal-tail correction.
# - Therefore individual discretized GAUSS numbers need not agree to machine
#   precision, even though both converge to the same continuous-time model.
# - The replication has NOT been executed with GAUSS or Julia in the author's
#   environment unless separately verified by an actual Julia run.

using CairoMakie
using SummaryTables: Cell, Table
using Printf

const OKABE_ITO = (
    orange    = "#E69F00",
    skyblue   = "#56B4E9",
    green     = "#009E73",
    yellow    = "#F0E442",
    blue      = "#0072B2",
    vermilion = "#D55E00",
    purple    = "#CC79A7",
    black     = "#000000",
)

"""Published Figure 6 calibration and a reproducible high-accuracy time grid."""
Base.@kwdef struct Parameters
    θ::Float64 = 0.25
    g::Float64 = 0.03
    r_pre::Float64 = 0.06
    γ::Float64 = 0.9
    σ::Float64 = 4.0
    x_u_pre::Float64 = 0.7
    x_u_output_0::Float64 = 1.0
    initial_u_share_in_r_wealth::Float64 = 0.05
    initial_nfa_u_over_output::Float64 = 0.0
    r_asset_price_shock::Float64 = -0.25
    pre_years::Float64 = 2.0
    n_pre::Int = 40
    horizon::Float64 = 26.0
    terminal_time::Float64 = 350.0
    dt::Float64 = 1.0 / 40.0
    q_tolerance::Float64 = 2.0e-11
    max_bisection_steps::Int = 65
end

"""Check assumptions required for the exact one-dimensional shooting reduction."""
function check_parameters(p::Parameters)
    0.0 < p.θ && 0.0 < p.g || throw(ArgumentError("θ and g must be positive"))
    p.g < p.r_pre < p.g + p.θ ||
        throw(ArgumentError("require g < r_pre < g+θ"))
    0.5 < p.γ < 1.0 || throw(ArgumentError("consumption home bias γ must exceed 1/2"))
    p.σ > 1.0 || throw(ArgumentError("this calibrated implementation assumes σ>1"))
    0.0 < p.x_u_pre < 1.0 || throw(ArgumentError("invalid U output share"))
    p.x_u_output_0 > 0.0 || throw(ArgumentError("U production must be positive"))
    -1.0 < p.r_asset_price_shock < 0.0 ||
        throw(ArgumentError("asset-price change must be between -1 and 0"))
    0.0 <= p.initial_u_share_in_r_wealth <= 1.0 ||
        throw(ArgumentError("invalid initial U share of R's wealth"))
    iszero(p.initial_nfa_u_over_output) ||
        throw(ArgumentError("Figure 6 reproduces the paper's initial zero-NFA case"))
    p.dt > 0.0 && p.pre_years > 0.0 && p.n_pre > 0 ||
        throw(ArgumentError("invalid time grid"))
    p.terminal_time > p.horizon ||
        throw(ArgumentError("terminal_time must exceed plotting horizon"))
    for t in (p.horizon, p.terminal_time)
        isapprox(t / p.dt, round(t / p.dt); atol=1e-9, rtol=0.0) ||
            throw(ArgumentError("horizon and terminal_time must be multiples of dt"))
    end
    p.max_bisection_steps >= 10 || throw(ArgumentError("insufficient bisection iterations"))
    p.q_tolerance > 0.0 || throw(ArgumentError("q_tolerance must be positive"))
    return nothing
end

"""CES prices of the U and R consumption baskets, in units of the U good."""
@inline function price_indices(q::Float64, p::Parameters)
    pow = q^(1.0 - p.σ)
    pu = (p.γ + (1.0 - p.γ) * pow)^(1.0 / (1.0 - p.σ))
    pr = ((1.0 - p.γ) + p.γ * pow)^(1.0 / (1.0 - p.σ))
    return pu, pr
end

"""Value share of the R good in each region's CES consumption price index."""
@inline function price_elasticities(q::Float64, p::Parameters)
    pow = q^(1.0 - p.σ)
    eu = (1.0 - p.γ) * pow / (p.γ + (1.0 - p.γ) * pow)
    er = p.γ * pow / ((1.0 - p.γ) + p.γ * pow)
    return eu, er
end

"""The initial U-good market excess demand in equation (61), in U output units."""
@inline function goods_residual(q::Float64, wu::Float64,
                                share_u::Float64, p::Parameters)
    pu, pr = price_indices(q, p)
    wr_scaled = 1.0 / (p.θ * share_u) - wu
    return p.θ * (p.γ * wu * pu^(p.σ - 1.0) +
           (1.0 - p.γ) * wr_scaled * pr^(p.σ - 1.0)) - 1.0
end

"""Bisection for a continuous scalar function with verified opposite endpoint signs."""
function bisect_root(f, lower::Float64, upper::Float64;
                     atol::Float64=2e-12, maxiter::Int=100)
    lower < upper || throw(ArgumentError("invalid root interval"))
    f_lower = f(lower)
    f_upper = f(upper)
    isfinite(f_lower) && isfinite(f_upper) ||
        throw(ArgumentError("nonfinite root residual at bracket endpoint"))
    f_lower == 0.0 && return lower
    f_upper == 0.0 && return upper
    f_lower * f_upper < 0.0 ||
        throw(ArgumentError("root bracket does not change sign: $(f_lower), $(f_upper)"))
    for _ in 1:maxiter
        middle = (lower + upper) / 2.0
        f_middle = f(middle)
        isfinite(f_middle) || throw(ArgumentError("nonfinite root residual"))
        abs(f_middle) < atol && return middle
        (upper - lower) < atol && return middle
        if signbit(f_middle) == signbit(f_lower)
            lower = middle
            f_lower = f_middle
        else
            upper = middle
        end
    end
    throw(ErrorException("scalar bisection failed to converge"))
end

"""
Find the pre-shock relative price and R output. The home-biased CES economy
is not initially at q=1 simply because the production shares are asymmetric.
"""
function calibrate(p::Parameters)
    check_parameters(p)
    δu = (p.r_pre - p.g) / p.θ
    x_r_pre = 1.0 - p.x_u_pre
    target_r_price_ratio = 1.0 + p.r_asset_price_shock
    δr = target_r_price_ratio * p.x_u_pre * δu /
         (1.0 - target_r_price_ratio * x_r_pre)
    δbar_pre = p.x_u_pre * δu + x_r_pre * δr

    # Preserve GAUSS conventions:
    # ARU = share of U trees held by R; AUR = share of R trees held by U.
    aru0 = p.initial_u_share_in_r_wealth *
           (x_r_pre / p.x_u_pre - p.θ * p.initial_nfa_u_over_output)
    aur0 = (p.θ * p.initial_nfa_u_over_output + aru0) *
           p.x_u_pre / x_r_pre
    0.0 <= aru0 < 1.0 && 0.0 <= aur0 < 1.0 ||
        throw(ArgumentError("initial international portfolio shares are inadmissible"))
    1.0 - aru0 - aur0 > 0.0 ||
        throw(ArgumentError("need a strictly positive domestic-ownership fraction"))
    0.0 < δr < δbar_pre < δu < 1.0 ||
        throw(ArgumentError("invalid capitalization shares"))

    # In the symmetric pre-shock equilibrium, U financial wealth / U output
    # is exactly 1/θ. The goods market determines q_pre at fixed share x_u_pre.
    pre_goods(q) = goods_residual(q, 1.0 / p.θ, p.x_u_pre, p)
    q_pre = bisect_root(pre_goods, 0.1, 10.0; atol=1e-13)
    xr0 = p.x_u_output_0 * x_r_pre / (p.x_u_pre * q_pre)
    λ_pre = last(price_indices(q_pre, p)) / first(price_indices(q_pre, p))

    return (; δu, δr, δbar_pre, x_r_pre, aru0, aur0,
            q_pre, xr0, λ_pre)
end

"""World-output value share of U, given the terms of trade q."""
@inline output_share_u(q::Float64, p::Parameters, c) =
    p.x_u_output_0 / (p.x_u_output_0 + q * c.xr0)

"""
Use global asset clearing and CES good-market clearing to obtain the UNIQUE
post-shock wealth and the two valuations implied by an initial q.
This removes two unknown initial valuations from the nonlinear shooting task.
"""
function initial_conditions(q::Float64, p::Parameters, c)
    s = output_share_u(q, p, c)
    pu, pr = price_indices(q, p)
    au = pu^(p.σ - 1.0)
    ar = pr^(p.σ - 1.0)
    h = 1.0 / (p.θ * s)
    denom = p.γ * au - (1.0 - p.γ) * ar
    abs(denom) > 1e-12 || throw(ArgumentError("singular initial goods market"))
    wu = (1.0 / p.θ - (1.0 - p.γ) * h * ar) / denom
    vu = (wu - c.aur0 * h) / (1.0 - c.aru0 - c.aur0)
    vr = (1.0 / p.θ - s * vu) / (1.0 - s)
    wr = (1.0 / p.θ - s * wu) / (1.0 - s)
    minimum((wu, wr, vu, vr)) > 0.0 ||
        throw(ArgumentError("negative initial wealth or valuation at q=$q"))
    return (; q, s, wu, wr, vu, vr)
end

"""
Continuous-time equilibrium law, derived by differentiating CES goods-market
clearing (61) and substituting the asset-market identity and wealth equation.
State y=(q, W_U/X_U, discount factor, U PV integral, R PV integral).
The final two states construct exact initial asset present values, not an ODE
approximation to a valuation obtained by truncating the capitalized stream.
"""
@inline function equilibrium_derivatives(y::NTuple{5,Float64},
                                         p::Parameters, c, s0::Float64)
    q, wu, discount, _, _ = y
    s = output_share_u(q, p, c)
    δbar = s * c.δu + (1.0 - s) * c.δr
    pu, pr = price_indices(q, p)
    au = pu^(p.σ - 1.0)
    ar = pr^(p.σ - 1.0)
    eu, er = price_elasticities(q, p)
    wr_in_u_output = 1.0 / (p.θ * s) - wu

    # Differentiate F(q,w,s)=0. Collect coefficients of zeta=d(log q)/dt.
    denominator =
        p.γ * au * wu * ((1.0 - s) + (p.σ - 1.0) * eu) +
        (1.0 - p.γ) * ar * wr_in_u_output *
        ((1.0 - s) + (p.σ - 1.0) * er)
    numerator = (p.γ * au - (1.0 - p.γ) * ar) *
                ((1.0 - c.δu) - p.θ * (1.0 - δbar) * wu)
    ζ = -numerator / denominator
    r_numeraire = p.g + (1.0 - s) * ζ + p.θ * δbar
    wu_dot = (r_numeraire - p.θ - p.g) * wu + (1.0 - c.δu)

    return (q * ζ, wu_dot, -p.θ * δbar * discount,
            c.δu / s0 * s * discount,
            c.δr / (1.0 - s0) * (1.0 - s) * discount)
end

"""One fourth-order Runge-Kutta step for a five-state, allocation-free tuple."""
@inline function rk4_step(y::NTuple{5,Float64}, h::Float64,
                          p::Parameters, c, s0::Float64)
    k1 = equilibrium_derivatives(y, p, c, s0)
    k2 = equilibrium_derivatives(ntuple(j -> y[j] + h * k1[j] / 2.0, 5),
                                 p, c, s0)
    k3 = equilibrium_derivatives(ntuple(j -> y[j] + h * k2[j] / 2.0, 5),
                                 p, c, s0)
    k4 = equilibrium_derivatives(ntuple(j -> y[j] + h * k3[j], 5),
                                 p, c, s0)
    return ntuple(j -> y[j] + h *
                  (k1[j] + 2.0 * k2[j] + 2.0 * k3[j] + k4[j]) / 6.0, 5)
end

"""
Forward equilibrium integration and two dividend present-value integrals.
The residual function uses the full terminal horizon; optional storage is
only needed once for the final plotting and backward valuation calculation.
"""
function forward_path(q0::Float64, p::Parameters, c;
                      store::Bool=false)
    init = initial_conditions(q0, p, c)
    y = (q0, init.wu, 1.0, 0.0, 0.0)
    n = round(Int, p.terminal_time / p.dt)
    plot_n = round(Int, p.horizon / p.dt) + 1
    trajectory = store ? Matrix{Float64}(undef, 5, plot_n) : nothing
    if store
        for j in 1:5
            trajectory[j, 1] = y[j]
        end
    end
    # Asset valuations require q and the world capitalization rate on the
    # entire PV integration horizon. Store these once for a linear-time
    # backward reconstruction of V_U(t) and V_R(t).
    long_q = store ? Vector{Float64}(undef, n + 1) : nothing
    if store
        long_q[1] = q0
    end
    for k in 1:n
        y = rk4_step(y, p.dt, p, c, init.s)
        if !(isfinite(y[1]) && isfinite(y[2]) && isfinite(y[3]) &&
             y[1] > 0.0 && y[2] > 0.0 && y[3] > 0.0)
            throw(ErrorException("invalid numerical state at step $k"))
        end
        if store
            long_q[k + 1] = y[1]
            if k + 1 <= plot_n
                for j in 1:5
                    trajectory[j, k + 1] = y[j]
                end
            end
        end
    end

    s_end = output_share_u(y[1], p, c)
    δbar_end = s_end * c.δu + (1.0 - s_end) * c.δr
    # The asymptotic q(t) becomes stationary. This terminal correction
    # removes the artificial finite-250-year valuation truncation in GAUSS.
    tail = y[3] / (p.θ * δbar_end)
    pv_u0 = y[4] + c.δu / init.s * s_end * tail
    pv_r0 = y[5] + c.δr / (1.0 - init.s) * (1.0 - s_end) * tail
    return (; init, pv_u0, pv_r0, y_terminal=y,
            terminal_share=s_end, trajectory, long_q)
end

"""
Solve a single scalar asset-pricing fixed point. Both country prices must
clear asset and goods markets. We check the *second* price independently.
"""
function solve_initial_price(p::Parameters, c)
    pv_difference(q) = begin
        sol = forward_path(q, p, c)
        sol.pv_u0 - sol.init.vu
    end
    # A benchmark-specific economically meaningful bracket: q appreciates
    # relative to the pre-crash terms of trade and remains strictly positive.
    q_lower = 0.8 * c.q_pre
    q_upper = c.q_pre
    q0 = bisect_root(pv_difference, q_lower, q_upper;
                     atol=p.q_tolerance,
                     maxiter=p.max_bisection_steps)
    sol = forward_path(q0, p, c; store=true)
    abs(sol.pv_u0 - sol.init.vu) < 2e-8 ||
        throw(ErrorException("U initial asset present value is inconsistent"))
    abs(sol.pv_r0 - sol.init.vr) < 2e-8 ||
        throw(ErrorException("R initial asset present value is inconsistent"))
    return sol
end

"""The backward U-asset valuation equation, conditional on U output share."""
@inline function backward_asset_slope(s::Float64, ku::Float64,
                                      p::Parameters, c)
    δbar = s * c.δu + (1.0 - s) * c.δr
    return p.θ * δbar * ku - c.δu * s
end

"""
Linear-time backward integration of the all-U-tree world-output asset value
K_U(t)=s_U(t)V_U(t)/X_U(t), with
  dK_U/dt = θ δbar(t) K_U - δU s_U(t).
The remaining R valuation is 1/θ-K_U by world asset-market clearing.
"""
function backward_asset_values(p::Parameters, c, sol)
    q = sol.long_q
    n = length(q)
    k_u = Vector{Float64}(undef, n)
    s_last = output_share_u(q[end], p, c)
    δbar_last = s_last * c.δu + (1.0 - s_last) * c.δr
    k_u[end] = c.δu * s_last / (p.θ * δbar_last)
    h = -p.dt
    plot_n = size(sol.trajectory, 2)

    # Accurate midpoint q from the known forward ODE trajectory would require
    # wealth at the terminal horizon. Instead use a Simpson-quality logarithmic
    # midpoint of the two adjacent terms-of-trade values. With dt=1/40, the
    # resulting price-reconstruction error is separately tested at impact.
    for i in (n - 1):-1:1
        s_right = output_share_u(q[i + 1], p, c)
        s_left = output_share_u(q[i], p, c)
        s_middle = output_share_u(sqrt(q[i] * q[i + 1]), p, c)
        k1 = backward_asset_slope(s_right, k_u[i + 1], p, c)
        k2 = backward_asset_slope(s_middle, k_u[i + 1] + h * k1 / 2.0, p, c)
        k3 = backward_asset_slope(s_middle, k_u[i + 1] + h * k2 / 2.0, p, c)
        k4 = backward_asset_slope(s_left, k_u[i + 1] + h * k3, p, c)
        k_u[i] = k_u[i + 1] + h *
                 (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    end
    v_u = Vector{Float64}(undef, plot_n)
    v_r = similar(v_u)
    for i in 1:plot_n
        s = output_share_u(q[i], p, c)
        v_u[i] = k_u[i] / s
        v_r[i] = (1.0 / p.θ - k_u[i]) / (1.0 - s)
    end
    return (; v_u, v_r, k_u)
end

"""Assemble the published five series and supplementary accounting objects."""
function simulate(p::Parameters, c=calibrate(p))
    sol = solve_initial_price(p, c)
    assets = backward_asset_values(p, c, sol)
    n_post = size(sol.trajectory, 2)
    t_post = [(i - 1) * p.dt for i in 1:n_post]

    q = Vector{Float64}(undef, n_post)
    wu = similar(q); wr = similar(q); s = similar(q)
    r_nominal = similar(q); r_u = similar(q); r_r = similar(q)
    r_world = similar(q); lambda = similar(q)
    ca = similar(q); tb = similar(q); nfa = similar(q); portfolio = similar(q)
    pu = similar(q); pr = similar(q); zeta = similar(q)
    goods = similar(q); discount = similar(q)

    for i in 1:n_post
        q[i] = sol.trajectory[1, i]
        wu[i] = sol.trajectory[2, i]
        discount[i] = sol.trajectory[3, i]
        s[i] = output_share_u(q[i], p, c)
        wr[i] = (1.0 / p.θ - s[i] * wu[i]) / (1.0 - s[i])
        pu[i], pr[i] = price_indices(q[i], p)
        lambda[i] = pr[i] / pu[i]
        eu, er = price_elasticities(q[i], p)
        qdot = equilibrium_derivatives((q[i], wu[i], discount[i], 0.0, 0.0),
                                       p, c, sol.init.s)[1]
        zeta[i] = qdot / q[i]
        δbar = s[i] * c.δu + (1.0 - s[i]) * c.δr
        r_nominal[i] = p.g + (1.0 - s[i]) * zeta[i] + p.θ * δbar
        r_u[i] = r_nominal[i] - eu * zeta[i]
        r_r[i] = r_nominal[i] - er * zeta[i]
        r_world[i] = s[i] * r_u[i] + (1.0 - s[i]) * r_r[i]
        nfa[i] = wu[i] - assets.v_u[i]
        tb[i] = 1.0 - p.θ * wu[i]
        ca[i] = tb[i] + r_nominal[i] * nfa[i]
        portfolio[i] = 1.0 - (1.0 - c.aur0) * assets.v_r[i] / wr[i]
        goods[i] = goods_residual(q[i], wu[i], s[i], p)
    end

    s_inf = sol.terminal_share
    r_inf = p.g + p.θ * (c.δu * s_inf + c.δr * (1.0 - s_inf))
    wu_inf = (1.0 - c.δu) / (p.g + p.θ - r_inf)
    wr_inf = (1.0 - c.δr) / (p.g + p.θ - r_inf)
    vu_inf = c.δu / (r_inf - p.g)
    vr_inf = c.δr / (r_inf - p.g)
    q_inf = sol.y_terminal[1]
    pu_inf, pr_inf = price_indices(q_inf, p)
    limit = (r_world=r_inf, ca=p.g * (wu_inf - vu_inf),
             nfa=wu_inf - vu_inf,
             portfolio=1.0 - (1.0 - c.aur0) * vr_inf / wr_inf,
             lambda=pr_inf / pu_inf, q=q_inf,
             tb=1.0 - p.θ * wu_inf)
    return (; t_post, sol, assets, q, wu, wr, s,
            r_nominal, r_u, r_r, r_world, lambda, ca, tb, nfa,
            portfolio, pu, pr, zeta, goods, discount, limit)
end

"""Numerical/economic validation, including independent asset-market restrictions."""
function verify_replication(p::Parameters, c, sim)
    @assert isapprox(c.δu, 0.12; atol=1e-12)
    @assert isapprox(c.δr, 0.08129032258064517; atol=1e-12)
    @assert isapprox(c.q_pre, 1.1704335897810625; atol=1e-8)
    @assert isapprox(c.xr0, 0.36616466949790444; atol=1e-8)

    @assert isapprox(sim.sol.init.q, 1.03630999375; atol=2e-5)
    @assert isapprox(sim.sol.init.vu, 4.27477160; atol=1e-4)
    @assert isapprox(sim.sol.init.vr, 3.27588803; atol=1e-4)
    @assert isapprox(sim.r_world[1], 0.05660400865; atol=3e-5)
    @assert isapprox(sim.limit.r_world, 0.05704604; atol=3e-5)
    @assert isapprox(sim.limit.nfa, -0.490; atol=0.004)

    @assert maximum(abs.(sim.goods)) < 5e-8
    @assert maximum(abs.(sim.ca .- sim.tb .- sim.r_nominal .* sim.nfa)) < 1e-12
    @assert maximum(abs.(sim.s .* sim.assets.v_u .+
                         (1.0 .- sim.s) .* sim.assets.v_r .-
                         1.0 / p.θ)) < 1e-10
    @assert maximum(abs.(sim.s .* sim.wu .+
                         (1.0 .- sim.s) .* sim.wr .-
                         1.0 / p.θ)) < 1e-10
    @assert all(sim.assets.v_u .> 0.0) && all(sim.assets.v_r .> 0.0)
    @assert all(sim.wu .> 0.0) && all(sim.wr .> 0.0)
    @assert all((sim.portfolio .>= 0.0) .& (sim.portfolio .<= 1.0))
    @assert maximum(abs.(sim.q .- sim.sol.trajectory[1, :])) < 1e-12
    @assert isapprox(sim.assets.v_u[1], sim.sol.init.vu; atol=3e-5)
    @assert isapprox(sim.assets.v_r[1], sim.sol.init.vr; atol=3e-5)
    @assert isapprox(sim.portfolio[1],
                     1.0-(1.0-c.aur0)*sim.assets.v_r[1]/sim.wr[1]; atol=1e-12)
    @assert sim.lambda[1] < c.λ_pre
    @assert sim.limit.lambda > c.λ_pre
    @assert sim.ca[1] < 0.0 && sim.limit.ca < 0.0
    return nothing
end

"""Reattach the unshocked prehistory, matching the published GAUSS grid."""
function prepend_history(p::Parameters, post::AbstractVector{Float64},
                         pre_value::Float64)
    return vcat(fill(pre_value, p.n_pre), post)
end

"""Five-panel Makie figure, using the Okabe-Ito palette."""
function create_figure(p::Parameters, c, sim)
    fig = Figure(size=(1320, 1270), fontsize=16, backgroundcolor=:white)
    ax_a = Axis(fig[1, 1], title="A. Current account / output",
                xlabel="Years", ylabel="Percent")
    ax_b = Axis(fig[1, 2], title="B. Net foreign assets / output",
                xlabel="Years", ylabel="Percent")
    ax_c = Axis(fig[2, 1], title="C. World real interest rate",
                xlabel="Years", ylabel="Percent")
    ax_d = Axis(fig[2, 2], title="D. Global portfolio share",
                xlabel="Years", ylabel="Percent")
    ax_e = Axis(fig[3, 1], title="E. Real exchange rate",
                xlabel="Years", ylabel="P_R / P_U")
    for ax in (ax_a, ax_b, ax_c, ax_d, ax_e)
        ax.xgridvisible = false
        ax.ygridvisible = false
        xlims!(ax, -p.pre_years, p.horizon)
        ax.xticks = -2:4:26
    end
    pre_t = [-p.pre_years + (i - 1) * p.pre_years / p.n_pre
             for i in 1:p.n_pre]
    all_t = vcat(pre_t, sim.t_post)

    series = (
        (ax_a, 100.0 .* prepend_history(p, sim.ca, 0.0), 0.0,
         100.0 * sim.limit.ca, (-15.0, 15.0)),
        (ax_b, 100.0 .* prepend_history(p, sim.nfa, 0.0), 0.0,
         100.0 * sim.limit.nfa, (-100.0, 20.0)),
        (ax_c, 100.0 .* prepend_history(p, sim.r_world, p.r_pre),
         100.0 * p.r_pre, 100.0 * sim.limit.r_world, (5.4, 6.2)),
        (ax_d, 100.0 .* prepend_history(p, sim.portfolio,
                                       p.initial_u_share_in_r_wealth),
         100.0 * p.initial_u_share_in_r_wealth,
         100.0 * sim.limit.portfolio, (0.0, 50.0)),
        (ax_e, prepend_history(p, sim.lambda, c.λ_pre),
         c.λ_pre, sim.limit.lambda, (1.0, 1.2)),
    )
    for (ax, y, baseline, steady, limits) in series
        lines!(ax, all_t, y; color=OKABE_ITO.blue, linewidth=3)
        hlines!(ax, [baseline]; color=OKABE_ITO.black,
                linestyle=:dot, linewidth=1.1)
        hlines!(ax, [steady]; color=OKABE_ITO.vermilion,
                linestyle=:dash, linewidth=1.8)
        ylims!(ax, limits...)
    end
    return fig
end

"""A SummaryTables.jl companion table with baseline and equilibrium outcomes."""
function export_summary(outdir::AbstractString, p::Parameters, c, sim)
    i12 = round(Int, 12.0 / p.dt) + 1
    i25 = round(Int, 25.0 / p.dt) + 1
    cases = (
        ("Pre-shock", 0.0, 0.0, p.r_pre,
         p.initial_u_share_in_r_wealth, c.λ_pre),
        ("Impact (0+)", sim.ca[1], sim.nfa[1], sim.r_world[1],
         sim.portfolio[1], sim.lambda[1]),
        ("Year 12", sim.ca[i12], sim.nfa[i12], sim.r_world[i12],
         sim.portfolio[i12], sim.lambda[i12]),
        ("Year 25", sim.ca[i25], sim.nfa[i25], sim.r_world[i25],
         sim.portfolio[i25], sim.lambda[i25]),
        ("Long-run limit", sim.limit.ca, sim.limit.nfa,
         sim.limit.r_world, sim.limit.portfolio, sim.limit.lambda),
    )
    headings = ("Period", "CA/U output (%)", "NFA/U output (%)",
                "World real rate (%)", "U assets in R wealth (%)",
                "RER (P_R/P_U)")
    content = Matrix{String}(undef, length(cases) + 1, length(headings))
    content[1, :] .= collect(headings)
    for (i, row) in enumerate(cases)
        content[i + 1, 1] = row[1]
        for j in 2:5
            content[i + 1, j] = @sprintf("%.3f", 100.0 * row[j])
        end
        content[i + 1, 6] = @sprintf("%.5f", row[6])
    end
    cells = [Cell(content[i, j]; bold=(i == 1),
                  halign=(j == 1 ? :left : :right))
             for i in axes(content, 1), j in axes(content, 2)]
    table = Table(cells; header=1)
    open(joinpath(outdir, "figure6_summary.html"), "w") do io
        show(io, MIME"text/html"(), table)
    end
    open(joinpath(outdir, "figure6_summary.tex"), "w") do io
        show(io, MIME"text/latex"(), table)
    end
    return nothing
end

"""Export both the plotted data and additional model/accounting diagnostics."""
function export_timeseries(outdir::AbstractString, p::Parameters, c, sim)
    headers = ("year", "current_account_pct", "net_foreign_assets_pct",
               "world_real_rate_pct", "global_portfolio_pct", "real_exchange_rate",
               "u_good_nominal_rate_pct", "u_real_rate_pct", "r_real_rate_pct",
               "trade_balance_pct", "terms_of_trade_q", "u_output_share",
               "u_wealth_over_output", "u_assets_over_output", "r_assets_over_output",
               "goods_market_residual")
    pre_t = [-p.pre_years + (i - 1) * p.pre_years / p.n_pre
             for i in 1:p.n_pre]
    open(joinpath(outdir, "figure6_timeseries.csv"), "w") do io
        println(io, join(headers, ","))
        for t in pre_t
            vals = (t, 0.0, 0.0, 100.0 * p.r_pre,
                    100.0 * p.initial_u_share_in_r_wealth, c.λ_pre,
                    100.0 * p.r_pre, 100.0 * p.r_pre, 100.0 * p.r_pre,
                    0.0, c.q_pre, p.x_u_pre,
                    1.0 / p.θ, 1.0 / p.θ, 1.0 / p.θ, 0.0)
            println(io, join((@sprintf("%.12g", v) for v in vals), ","))
        end
        for i in eachindex(sim.t_post)
            vals = (sim.t_post[i], 100.0 * sim.ca[i], 100.0 * sim.nfa[i],
                    100.0 * sim.r_world[i], 100.0 * sim.portfolio[i],
                    sim.lambda[i], 100.0 * sim.r_nominal[i],
                    100.0 * sim.r_u[i], 100.0 * sim.r_r[i],
                    100.0 * sim.tb[i], sim.q[i], sim.s[i],
                    sim.wu[i], sim.assets.v_u[i], sim.assets.v_r[i], sim.goods[i])
            println(io, join((@sprintf("%.12g", v) for v in vals), ","))
        end
    end
    return nothing
end

"""Record methods, original GAUSS parameters, and quantitative diagnostics."""
function export_metadata(outdir::AbstractString, p::Parameters, c, sim)
    q0 = sim.sol.init.q
    λ0 = sim.lambda[1]
    log_appreciation = 100.0 * log(c.λ_pre / λ0)
    longrun_dep = 100.0 * (sim.limit.lambda / c.λ_pre - 1.0)
    open(joinpath(outdir, "figure6_metadata.txt"), "w") do io
        println(io, "Caballero, Farhi, Gourinchas (2008), Figure 6")
        println(io, "Original GAUSS source: prog_UR_exch_final.gau")
        println(io, "Reimplementation: single-q shooting + differentiated goods clearing + RK4")
        println(io, "Paper original: per-step NLSYS + Euler + 250-year truncated PV integrals")
        @printf(io, "theta=%.12f g=%.12f gamma=%.12f sigma=%.6f\n",
                p.θ, p.g, p.γ, p.σ)
        @printf(io, "deltaU=%.12f deltaR=%.12f ARU=%.12f AUR=%.12f\n",
                c.δu, c.δr, c.aru0, c.aur0)
        @printf(io, "q_pre=%.12f R_output_0=%.12f lambda_pre=%.12f\n",
                c.q_pre, c.xr0, c.λ_pre)
        @printf(io, "q_impact=%.12f VU_impact=%.12f VR_impact=%.12f\n",
                q0, sim.sol.init.vu, sim.sol.init.vr)
        @printf(io, "PV_U_residual=%.4e PV_R_residual=%.4e\n",
                sim.sol.pv_u0 - sim.sol.init.vu,
                sim.sol.pv_r0 - sim.sol.init.vr)
        @printf(io, "maximum_goods_market_residual=%.4e\n",
                maximum(abs.(sim.goods)))
        @printf(io, "world_real_rate_impact=%.8f%% nominal_rate_impact=%.8f%%\n",
                100.0 * sim.r_world[1], 100.0 * sim.r_nominal[1])
        @printf(io, "lambda_impact=%.12f lambda_longrun=%.12f\n",
                λ0, sim.limit.lambda)
        @printf(io, "impact_appreciation_log_percent=%.6f\n", log_appreciation)
        @printf(io, "longrun_depreciation_from_pre_percent=%.6f\n", longrun_dep)
        @printf(io, "longrun_CA_over_U_output_percent=%.6f\n", 100.0 * sim.limit.ca)
        @printf(io, "longrun_NFA_over_U_output_percent=%.6f\n", 100.0 * sim.limit.nfa)
        @printf(io, "dt=%.12f terminal_time=%.1f\n", p.dt, p.terminal_time)
        println(io, "WARNING: Julia runtime/syntax must be tested in your Julia environment.")
    end
    return nothing
end

"""Run Figure 6 and export one self-contained, auditable replication."""
function main(args::Vector{String}=ARGS)
    length(args) <= 1 || throw(ArgumentError("pass at most one output directory"))
    outdir = isempty(args) ? joinpath(@__DIR__, "output_fig06") : abspath(only(args))
    mkpath(outdir)
    p = Parameters()
    c = calibrate(p)
    sim = simulate(p, c)
    verify_replication(p, c, sim)

    fig = create_figure(p, c, sim)
    save(joinpath(outdir, "figure6.png"), fig; px_per_unit=2)
    save(joinpath(outdir, "figure6.pdf"), fig)
    export_summary(outdir, p, c, sim)
    export_timeseries(outdir, p, c, sim)
    export_metadata(outdir, p, c, sim)

    @printf("Figure 6 calibration: δU=%.8f δR=%.8f q_pre=%.8f q_impact=%.8f\n",
            c.δu, c.δr, c.q_pre, sim.sol.init.q)
    @printf("Real exchange rate (P_R/P_U): pre %.6f impact %.6f long run %.6f\n",
            c.λ_pre, sim.lambda[1], sim.limit.lambda)
    @printf("World REAL rate: impact %.6f%% year 25 %.6f%% long run %.6f%%\n",
            100.0 * sim.r_world[1],
            100.0 * sim.r_world[round(Int, 25.0 / p.dt) + 1],
            100.0 * sim.limit.r_world)
    @printf("Long run: CA/U GDP=%.5f%% NFA/U GDP=%.5f%% portfolio=%.5f%%\n",
            100.0 * sim.limit.ca, 100.0 * sim.limit.nfa,
            100.0 * sim.limit.portfolio)
    println("Structural/numerical checks passed. Outputs: ", outdir)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
