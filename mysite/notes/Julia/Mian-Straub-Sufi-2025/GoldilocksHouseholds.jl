module GoldilocksHouseholds

using LinearAlgebra

export BUQuadratic, BULog, OLGHousehold, HAHousehold, HAGrids,
       savers_income, hand_to_mouth, aggregate_households,
       bu_quadratic_steady_state, bu_log_steady_state, olg_steady_state,
       bu_quadratic_period, bu_log_period, olg_period,
       find_lambda_bu, rouwenhorst, asset_grid, ha_grids,
       ha_income, egm_step!, ha_policy_steady_state,
       ha_stationary_distribution, ha_weighted_mpc, ha_steady_state,
       ha_euler_residuals

# One model period is the period used by the original Python replication.
# These routines do not import GoldilocksCore; callable structs accept its
# evaluate_steady_state(household_inputs::NamedTuple) interface directly.

"""Saver-specific disposable income. Total profits accrue to savers."""
function savers_income(Z, pure_profit, mu)
    0 <= mu < 1 || throw(DomainError(mu, "require 0 <= mu < 1"))
    return Z + pure_profit / (1 - mu)
end

"""The hand-to-mouth household in the original blocks.py."""
hand_to_mouth(Z) = (; C_htm=Z, A_htm=zero(Z), MPC_htm=one(Z))

"""Aggregate the saver and hand-to-mouth household quantities."""
function aggregate_households(Z, mu, C_ha, A_ha, MPC_ha)
    0 <= mu < 1 || throw(DomainError(mu, "require 0 <= mu < 1"))
    htm = hand_to_mouth(Z)
    weight = 1 - mu
    return (; C=mu * Z + weight * C_ha,
            A=weight * A_ha,
            MPC=mu + weight * MPC_ha,
            C_ha, A_ha, MPC_ha,
            htm..., one_minus_mu=weight)
end

"""Compute the author's smaller root of the quadratic for lambda_bu in [0,1]."""
function find_lambda_bu(beta, r, vp_slope, eis, C_ss)
    beta > 0 || throw(DomainError(beta, "beta must be positive"))
    eis > 0 || throw(DomainError(eis, "eis must be positive"))
    C_ss > 0 || throw(DomainError(C_ss, "C_ss must be positive"))
    1 + r > 0 || throw(DomainError(r, "gross return must be positive"))
    vpp = -vp_slope  # exactly as in original find_lambda_bu (not /A_ss)
    upp = -C_ss^(-1 / eis - 1) / eis
    coefficient = (1 + vpp / upp) / (beta * (1 + r)) + (1 + r)
    constant = 1 / beta
    discriminant = coefficient^2 - 4 * constant
    discriminant >= 0 || throw(DomainError(discriminant, "no real lambda"))
    root = 2 * constant / (coefficient + sqrt(discriminant))
    0 <= root <= 1 || throw(DomainError(root, "no calibrated root in [0,1]"))
    return root
end

"""Household parameters for quadratic bonds-in-utility, preserving v'(A)."""
struct BUQuadratic{T<:Real}
    beta::T
    eis::T
    vp_0::T
    vp_slope::T
    A_ss::T
    lambda_bu::T
end

function BUQuadratic(; beta, eis, vp_0, vp_slope, A_ss, lambda_bu)
    all(isfinite, (beta, eis, vp_0, vp_slope, A_ss, lambda_bu)) ||
        throw(ArgumentError("BU parameters must be finite"))
    beta > 0 && eis > 0 && A_ss > 0 ||
        throw(DomainError((beta, eis, A_ss), "beta, eis, A_ss must be positive"))
    return BUQuadratic(promote(beta, eis, vp_0, vp_slope, A_ss, lambda_bu)...)
end

"""Household parameters for log bonds-in-utility."""
struct BULog{T<:Real}
    beta::T
    eis::T
    vp_slope::T
    A_ss::T
    lambda_bu::T
end

function BULog(; beta, eis, vp_slope, A_ss, lambda_bu)
    beta > 0 && eis > 0 && A_ss > 0 && vp_slope > 0 ||
        throw(DomainError((beta, eis, vp_slope, A_ss), "positive beta, eis, vp_slope, A_ss required"))
    return BULog(promote(beta, eis, vp_slope, A_ss, lambda_bu)...)
end

"""Blanchard–Yaari OLG; OLG-alt uses the SAME equations with a reduced-form
`survival` coefficient that can exceed one in the author's saved calibrations.
It must not then be interpreted literally as a survival probability."""
struct OLGHousehold{T<:Real}
    beta::T
    survival::T
    zeta::T
end

function OLGHousehold(; beta, survival, zeta)
    beta > 0 && survival > 0 && 0 <= zeta < 1 ||
        throw(DomainError((beta, survival, zeta), "require beta>0, effective survival>0, zeta in [0,1)"))
    return OLGHousehold(promote(beta, survival, zeta)...)
end

"""Residuals at one date; lagged assets and next consumption are explicitly supplied."""
function bu_quadratic_period(p::BUQuadratic; C, C_next, A, A_lag, Z,
                             r, r_ante, pure_profit=0, mu=0, r_ss=r)
    C > 0 && C_next > 0 && 1 + r_ss > 0 ||
        throw(DomainError((C, C_next, r_ss), "positive consumption and gross steady return required"))
    Z_total = savers_income(Z, pure_profit, mu)
    vp = p.vp_0 - p.vp_slope * (A - p.A_ss) / p.A_ss
    return (; euler=C^(-1 / p.eis) - p.beta * (1 + r_ante) * C_next^(-1 / p.eis) - vp,
            budget=(1 + r) * A_lag + Z_total - A - C,
            MPC_ha=1 - p.lambda_bu / (1 + r_ss), vp)
end

"""Log-BU single-date Euler and budget residuals."""
function bu_log_period(p::BULog; C, C_next, A, A_lag, Z,
                        r, r_ante, pure_profit=0, mu=0, r_ss=r)
    C > 0 && C_next > 0 && A > 0 && 1 + r_ss > 0 ||
        throw(DomainError((C, C_next, A), "positive consumption and assets required"))
    Z_total = savers_income(Z, pure_profit, mu)
    vp = p.vp_slope * p.A_ss / A
    return (; euler=C^(-1 / p.eis) - p.beta * (1 + r_ante) * C_next^(-1 / p.eis) - vp,
            budget=(1 + r) * A_lag + Z_total - A - C,
            MPC_ha=1 - p.lambda_bu / (1 + r_ss), vp)
end

"""OLG single-date human-wealth recursion, consumption, and budget condition."""
function olg_period(p::OLGHousehold; A, A_lag, H, H_next, Z,
                     r, r_ante, pure_profit=0, mu=0)
    Z_total = savers_income(Z, pure_profit, mu)
    1 + r_ante > 0 || throw(DomainError(r_ante, "gross expected return must be positive"))
    C_ha = (1 - p.survival * p.beta) * ((1 + r) * A_lag + H)
    return (; H_recursion=-H + Z_total + (1 - p.zeta) * p.survival / (1 + r_ante) * H_next,
            budget=(1 + r) * A_lag + Z_total - C_ha - A,
            C_ha, MPC_ha=1 - p.survival * p.beta)
end

# Robust scalar bracket search, followed by bisection on a real economic root.
# BU unconstrained asset demand may be negative; only C>0 is imposed.
function _bracketed_root(f, center; lower=-Inf, upper=Inf,
                         atol=1e-12, maxiter=180)
    isfinite(center) || throw(DomainError(center, "finite initial asset guess required"))
    lower < upper || throw(ArgumentError("invalid asset domain"))
    center = max(lower + 1e-12, min(upper - 1e-12, center))
    f0 = f(center)
    isfinite(f0) || throw(DomainError(f0, "nonfinite root function at initial guess"))
    abs(f0) <= atol && return center
    step = max(abs(center) * 0.05, 1e-4)
    for _ in 1:80
        for x in (max(lower + 1e-12, center - step), min(upper - 1e-12, center + step))
            if x != center
                fx = f(x)
                if isfinite(fx) && signbit(fx) != signbit(f0)
                    lo, hi = min(x, center), max(x, center)
                    flo = f(lo)
                    for _ in 1:maxiter
                        mid = lo + (hi - lo) / 2
                        fm = f(mid)
                        if abs(fm) <= atol || abs(hi - lo) <= atol * max(1, abs(mid))
                            return mid
                        end
                        if signbit(fm) == signbit(flo)
                            lo, flo = mid, fm
                        else
                            hi = mid
                        end
                    end
                    throw(ErrorException("bisection failed to converge"))
                end
            end
        end
        step *= 1.7
    end
    throw(DomainError(center, "no admissible root found near A_ss"))
end

"""Author's BU-quad steady state, preserving its special EIS=1 root convention."""
function bu_quadratic_steady_state(p::BUQuadratic, Z, r, r_ante, pure_profit, mu)
    Z_total = savers_income(Z, pure_profit, mu)
    delta = 1 - p.beta * (1 + r_ante)
    A = if p.eis == 1
        q = p.vp_slope / p.A_ss * r
        lin = p.vp_slope / p.A_ss * Z_total - (p.vp_0 + p.vp_slope) * r
        cst = delta - (p.vp_0 + p.vp_slope) * Z_total
        disc = lin^2 - 4q * cst
        disc >= 0 || throw(DomainError(disc, "no real quadratic BU asset demand"))
        if abs(q) <= 1e-14
            abs(lin) > 1e-14 || throw(DomainError(lin, "degenerate BU equation"))
            -cst / lin
        else
            (-lin + sqrt(disc)) / (2q) # exact original plus-root branch
        end
    else
        # Original hh_bu_ss uses max(vp,0) inside the nonlinear root,
        # but recomputes the untruncated vp in the returned Euler residual.
        function eq(a)
            c = r * a + Z_total
            c > 0 || return NaN
            vp = max(zero(a), p.vp_0 - p.vp_slope * (a - p.A_ss) / p.A_ss)
            return delta - vp * c^(1 / p.eis)
        end
        _bracketed_root(eq, p.A_ss)
    end
    C_ha = r * A + Z_total
    C_ha > 0 || throw(DomainError(C_ha, "nonpositive steady-state consumption"))
    vp = p.vp_0 - p.vp_slope * (A - p.A_ss) / p.A_ss
    euler = delta - vp * C_ha^(1 / p.eis)
    budget = r * A + Z_total - C_ha
    MPC_ha = 1 - p.lambda_bu / (1 + r)
    return (; C_ha, A_ha=A, MPC_ha, euler, budget, vp)
end

"""Author's BU-log steady state with positive-asset domain."""
function bu_log_steady_state(p::BULog, Z, r, r_ante, pure_profit, mu)
    Z_total = savers_income(Z, pure_profit, mu)
    delta = 1 - p.beta * (1 + r_ante)
    delta > 0 || throw(DomainError(delta, "log BU requires positive convenience Euler wedge"))
    f(a) = begin
        c = r * a + Z_total
        c > 0 && a > 0 ? delta - p.vp_slope * p.A_ss / a * c^(1 / p.eis) : NaN
    end
    A = if p.eis == 1
        denominator = delta - p.vp_slope * p.A_ss * r
        denominator != 0 || throw(DomainError(denominator, "degenerate log BU asset demand"))
        p.vp_slope * p.A_ss * Z_total / denominator
    else
        _bracketed_root(f, max(p.A_ss, 1e-7); lower=0.0)
    end
    A > 0 || throw(DomainError(A, "log BU demands positive assets"))
    C_ha = r * A + Z_total
    C_ha > 0 || throw(DomainError(C_ha, "nonpositive log BU consumption"))
    vp = p.vp_slope * p.A_ss / A
    return (; C_ha, A_ha=A, MPC_ha=1 - p.lambda_bu / (1 + r),
            euler=delta - vp * C_ha^(1 / p.eis),
            budget=r * A + Z_total - C_ha, vp)
end

"""Stationary Blanchard–Yaari human wealth and household asset demand."""
function olg_steady_state(p::OLGHousehold, Z, r, r_ante, pure_profit, mu)
    Z_total = savers_income(Z, pure_profit, mu)
    1 + r_ante > 0 || throw(DomainError(r_ante, "gross expected return must be positive"))
    numerator = (1 - p.zeta) * p.survival
    H_denom = 1 - numerator / (1 + r_ante)
    H_denom > 0 || throw(DomainError(H_denom, "human wealth does not converge"))
    H = Z_total / H_denom
    mpc = 1 - p.survival * p.beta
    mpc > 0 || throw(DomainError(mpc, "OLG consumption propensity nonpositive"))
    denom = mpc * (1 + r) - r
    abs(denom) > 1e-13 || throw(DomainError(denom, "OLG asset demand indeterminate"))
    A = (Z_total - mpc * H) / denom
    C_ha = mpc * ((1 + r) * A + H)
    C_ha > 0 || throw(DomainError(C_ha, "nonpositive OLG consumption"))
    return (; C_ha, A_ha=A, MPC_ha=mpc, H,
            H_recursion=-H + Z_total + numerator / (1 + r_ante) * H,
            budget=(1 + r) * A + Z_total - C_ha - A)
end

# Callable interfaces also expose household residuals and saver-specific quantities.
function _attach_aggregate(result::NamedTuple, x::NamedTuple)
    return merge(aggregate_households(x.Z, x.mu, result.C_ha, result.A_ha, result.MPC_ha), result)
end

function (p::BUQuadratic)(x::NamedTuple)
    return _attach_aggregate(bu_quadratic_steady_state(p, x.Z, x.r, x.r_ante,
                                                      x.pure_profit, x.mu), x)
end
function (p::BULog)(x::NamedTuple)
    return _attach_aggregate(bu_log_steady_state(p, x.Z, x.r, x.r_ante,
                                                x.pure_profit, x.mu), x)
end
function (p::OLGHousehold)(x::NamedTuple)
    return _attach_aggregate(olg_steady_state(p, x.Z, x.r, x.r_ante,
                                             x.pure_profit, x.mu), x)
end

"""Rouwenhorst discretization with unit-mean lognormal productivity states."""
function rouwenhorst(rho_e::Real, sd_e::Real, n_e::Integer)
    -1 < rho_e < 1 || throw(DomainError(rho_e, "Rouwenhorst persistence must lie in (-1,1)"))
    sd_e >= 0 || throw(DomainError(sd_e, "standard deviation must be nonnegative"))
    n_e >= 2 || throw(ArgumentError("n_e must be at least two"))
    p = (1 + Float64(rho_e)) / 2
    Pi = [p 1-p; 1-p p]
    for n in 3:n_e
        P = zeros(Float64, n, n)
        @views begin
            P[1:end-1, 1:end-1] .+= p .* Pi
            P[1:end-1, 2:end] .+= (1-p) .* Pi
            P[2:end, 1:end-1] .+= (1-p) .* Pi
            P[2:end, 2:end] .+= p .* Pi
            P[2:end-1, :] ./= 2
        end
        Pi = P
    end
    pi = fill(1.0 / n_e, n_e)
    for _ in 1:10_000
        newpi = vec(transpose(Pi) * pi)
        if maximum(abs.(newpi .- pi)) < 1e-11
            pi = newpi
            break
        end
        pi = newpi
    end
    s = collect(range(-1.0, 1.0; length=n_e))
    variance = sum(pi .* (s .- sum(pi .* s)).^2)
    s .*= Float64(sd_e) / sqrt(variance)
    y = exp.(s)
    y ./= sum(pi .* y)
    return (; e_grid=y, pi, Pi)
end

"""The exact double-exponential asset grid from sequence_jacobian/utilities/discretize.py."""
function asset_grid(min_a::Real, max_a::Real, n_a::Integer)
    n_a >= 2 || throw(ArgumentError("n_a must be at least two"))
    max_a > min_a || throw(DomainError((min_a, max_a), "max_a must exceed min_a"))
    ub = log1p(log1p(Float64(max_a - min_a)))
    u = range(0.0, ub; length=n_a)
    grid = [Float64(min_a) + expm1(exp(z) - 1) for z in u]
    grid[1] = Float64(min_a)
    grid[end] = Float64(max_a)
    return grid
end

"""Reusable HA Markov and asset grids; row i of Pi is current income state i."""
struct HAGrids{T<:AbstractFloat}
    e_grid::Vector{T}
    pi::Vector{T}
    Pi::Matrix{T}
    a_grid::Vector{T}
end

function ha_grids(; rho_e, sd_e, n_e, min_a, max_a, n_a)
    re = rouwenhorst(rho_e, sd_e, n_e)
    a = asset_grid(min_a, max_a, n_a)
    return HAGrids(re.e_grid, re.pi, re.Pi, a)
end

"""CRRA HA household with EGM and a stationary distribution iterator.

`source_lottery_extrapolation=false` (default) rejects policies outside the
asset grid and enforces nonnegative lottery weights. Set to `true` only when
reproducing the source Sequence-Jacobian interpolation, which extrapolates
linearly beyond the grid and can create tiny signed distribution masses.
"""
struct HAHousehold{T<:Real,G<:HAGrids}
    beta::T
    eis::T
    lumpsum::T
    grids::G
    backward_tol::Float64
    forward_tol::Float64
    backward_maxit::Int
    forward_maxit::Int
    source_lottery_extrapolation::Bool
end

function HAHousehold(; beta, eis, lumpsum=0.0,
                       rho_e=0.9, sd_e=0.92, n_e=7,
                       min_a=0.0, max_a=500.0, n_a=300,
                       backward_tol=1e-8, forward_tol=1e-10,
                       backward_maxit=5000, forward_maxit=100_000,
                       source_lottery_extrapolation::Bool=false)
    beta > 0 && eis > 0 || throw(DomainError((beta, eis), "beta and eis must be positive"))
    backward_tol > 0 && forward_tol > 0 && backward_maxit > 0 && forward_maxit > 0 ||
        throw(ArgumentError("positive tolerances and iteration limits required"))
    g = ha_grids(; rho_e, sd_e, n_e, min_a, max_a, n_a)
    b, e, l = promote(beta, eis, lumpsum)
    return HAHousehold(b, e, l, g, Float64(backward_tol), Float64(forward_tol),
                       Int(backward_maxit), Int(forward_maxit),
                       source_lottery_extrapolation)
end

"""After-tax productivity income plus lumpsum, matching blocks.income exactly."""
function ha_income(Z, e_grid::AbstractVector, lumpsum, pure_profit, mu)
    0 <= mu < 1 || throw(DomainError(mu, "require 0 <= mu < 1"))
    wage_income = Z - lumpsum + pure_profit / (1 - mu)
    return wage_income .* e_grid .+ lumpsum
end

# Robust linear interpolation in EGM; extrapolates as the source Python does.
@inline function _interp_increasing(x::AbstractVector, y::AbstractVector, xq)
    n = length(x)
    j = clamp(searchsortedlast(x, xq), 1, n - 1)
    return y[j] + (xq - x[j]) * (y[j + 1] - y[j]) / (x[j + 1] - x[j])
end

"""One EGM backward step; Va_p = Pi * Va_next; only minimum assets are clipped."""
function egm_step!(Va::AbstractMatrix, a::AbstractMatrix, c::AbstractMatrix,
                   Va_p::AbstractMatrix, grid::AbstractVector, y::AbstractVector,
                   r::Real, beta::Real, eis::Real)
    ne, na = size(Va_p)
    size(Va) == size(a) == size(c) == (ne, na) ||
        throw(DimensionMismatch("EGM matrices have incompatible dimensions"))
    length(grid) == na && length(y) == ne ||
        throw(DimensionMismatch("EGM grids have incompatible dimensions"))
    1 + r > 0 || throw(DomainError(r, "gross return must be positive"))
    endog = Vector{eltype(Va)}(undef, na)
    @inbounds for i in 1:ne
        for j in 1:na
            uc = beta * Va_p[i,j]
            uc > 0 || throw(DomainError(uc, "nonpositive expected marginal utility"))
            endog[j] = uc^(-eis) + grid[j]
            if j > 1 && endog[j] <= endog[j-1]
                throw(DomainError(endog[j], "nonmonotone EGM endogenous grid"))
            end
        end
        for j in 1:na
            coh = (1 + r) * grid[j] + y[i]
            ap = max(grid[1], _interp_increasing(endog, grid, coh))
            cp = coh - ap
            cp > 0 || throw(DomainError(cp, "nonpositive EGM consumption"))
            a[i,j], c[i,j] = ap, cp
            Va[i,j] = (1 + r) * cp^(-1 / eis)
        end
    end
    return nothing
end

"""Solve the stationary HA savings policy using backward EGM iteration."""
function ha_policy_steady_state(p::HAHousehold, y::AbstractVector, r::Real)
    g = p.grids
    ne, na = length(g.e_grid), length(g.a_grid)
    length(y) == ne || throw(DimensionMismatch("income state count mismatch"))
    1 + r > 0 || throw(DomainError(r, "gross return must be positive"))
    min_coh = minimum((1 + r) * g.a_grid[1] .+ y)
    min_coh > 0 || throw(DomainError(min_coh, "negative cash-on-hand at borrowing limit"))
    Va = Matrix{Float64}(undef, ne, na)
    a = similar(Va)
    c = similar(Va)
    @inbounds for j in 1:na, i in 1:ne
        coh = (1 + r) * g.a_grid[j] + y[i]
        Va[i,j] = (1 + r) * (0.1 * coh)^(-1 / p.eis)
    end
    nextVa = similar(Va)
    Va_p = similar(Va)
    olda = similar(a)
    err = Inf
    for it in 1:p.backward_maxit
        mul!(Va_p, g.Pi, Va)
        egm_step!(nextVa, a, c, Va_p, g.a_grid, y, r, p.beta, p.eis)
        if it > 1
            err = maximum(abs.(a .- olda))
            if err < p.backward_tol
                return (; Va=nextVa, a, c, backward_iterations=it,
                        backward_error=err)
            end
        end
        copyto!(olda, a)
        Va, nextVa = nextVa, Va
    end
    throw(ErrorException("HA backward iteration failed to converge; max policy change=$(err)"))
end

# Two explicit modes: strict probability interpolation (default), and
# source-exact signed linear extrapolation (used ONLY for original parity).
# The original SSJ interpolate_coord_robust does not clamp its weights at the
# grid boundary; with a' > max_a its top-neighbor weight becomes negative.
function _lottery_coordinates(a::AbstractMatrix, grid::AbstractVector;
                              source_extrapolation::Bool=false)
    ne, na = size(a)
    idx = Matrix{Int}(undef, ne, na)
    weight = Matrix{Float64}(undef, ne, na)
    @inbounds for j in 1:na, i in 1:ne
        ap = a[i,j]
        if !source_extrapolation
            ap >= grid[1] - 1e-10 && ap <= grid[end] + 1e-10 ||
                throw(DomainError(ap, "policy outside asset grid; enlarge max_a; no silent upper clipping"))
        end
        lo = clamp(searchsortedlast(grid, ap), 1, na - 1)
        w = (grid[lo+1] - ap) / (grid[lo+1] - grid[lo])
        idx[i,j] = lo
        weight[i,j] = source_extrapolation ? w : clamp(w, 0.0, 1.0)
    end
    return idx, weight
end

"""Forward asset lottery: D is after current productivity realization."""
function _policy_forward!(dest, D, idx, weight)
    ne, na = size(D)
    fill!(dest, 0)
    @inbounds for j in 1:na, i in 1:ne
        lo = idx[i,j]
        v = D[i,j]
        w = weight[i,j]
        dest[i,lo] += v * w
        dest[i,lo+1] += v * (1 - w)
    end
    return dest
end

"""Iterate Dbeg -> D=Pi' Dbeg -> Dbeg'=lottery(D,a) as source HetBlock."""
function ha_stationary_distribution(p::HAHousehold, a::AbstractMatrix)
    g = p.grids
    ne, na = size(a)
    (ne, na) == (length(g.e_grid), length(g.a_grid)) ||
        throw(DimensionMismatch("policy and grids have different dimensions"))
    idx, weight = _lottery_coordinates(a, g.a_grid;
        source_extrapolation=p.source_lottery_extrapolation)
    Dbeg = repeat(reshape(g.pi ./ na, ne, 1), 1, na)
    D = similar(Dbeg)
    nextDbeg = similar(Dbeg)
    nextD = similar(Dbeg)
    err = Inf
    for it in 1:p.forward_maxit
        mul!(D, transpose(g.Pi), Dbeg)
        _policy_forward!(nextDbeg, D, idx, weight)
        err = maximum(abs.(nextDbeg .- Dbeg))
        if err < p.forward_tol
            mul!(nextD, transpose(g.Pi), nextDbeg)
            # Negative weights are impossible in strict mode. In source
            # mode they can occur in tiny, very-low-probability upper tails:
            # this reproduces the published numerical algorithm but is NOT
            # a mathematically nonnegative probability distribution.
            minmass = minimum(nextD)
            if !p.source_lottery_extrapolation
                minmass >= -1e-13 || error("negative stationary mass")
            end
            abs(sum(nextD) - 1) < 1e-9 || error("stationary distribution does not sum to one")
            negative_mass = sum(v -> min(v, zero(v)), nextD)
            return (; Dbeg=copy(nextDbeg), D=copy(nextD),
                    distribution_iterations=it, distribution_error=err,
                    minimum_distribution_mass=minmass,
                    negative_distribution_mass=negative_mass,
                    lottery_indices=idx, lottery_weights=weight)
        end
        Dbeg, nextDbeg = nextDbeg, Dbeg
    end
    throw(ErrorException("HA stationary distribution did not converge; max change=$(err)"))
end

"""Author's wealth-difference MPC, constrained-agent override, and e_grid weighting."""
function ha_weighted_mpc(c::AbstractMatrix, a::AbstractMatrix,
                         grid::AbstractVector, r::Real, e_grid::AbstractVector)
    ne, na = size(c)
    size(a) == (ne, na) && length(grid) == na && length(e_grid) == ne ||
        throw(DimensionMismatch("MPC dimensions mismatch"))
    na >= 2 || throw(ArgumentError("at least two asset gridpoints required"))
    mpc = similar(c)
    post = (1 + r) .* grid
    @inbounds for i in 1:ne, j in 1:na
        left, right = j == 1 ? (1, 2) : j == na ? (na-1, na) : (j-1, j+1)
        slope = (c[i,right] - c[i,left]) / (post[right] - post[left])
        if a[i,j] == grid[1] # EXACT comparison, as in original Python
            slope = one(slope)
        end
        mpc[i,j] = slope * e_grid[i]
    end
    return mpc
end

"""Euler inequality: interior residuals should approach zero; bound residuals >=0."""
function ha_euler_residuals(c::AbstractMatrix, a::AbstractMatrix,
                            g::HAGrids, r::Real, beta::Real, eis::Real)
    ne, na = size(c)
    residual = similar(c)
    @inbounds for j in 1:na, i in 1:ne
        marginal_future = zero(eltype(c))
        for k in 1:ne
            cnext = _interp_increasing(g.a_grid, view(c, k, :), a[i,j])
            marginal_future += g.Pi[i,k] * cnext^(-1 / eis)
        end
        residual[i,j] = c[i,j]^(-1 / eis) - beta * (1 + r) * marginal_future
    end
    return residual
end

"""Complete stationary HA solution: EGM, invariant distribution, and weighted MPC."""
function ha_steady_state(p::HAHousehold, Z, r, r_ante, pure_profit, mu)
    isfinite(r) && r > -1 || throw(DomainError(r, "invalid HA net return"))
    # HA pays realized combined return r in the steady state (not r_ante).
    y = ha_income(Z, p.grids.e_grid, p.lumpsum, pure_profit, mu)
    policy = ha_policy_steady_state(p, y, r)
    dist = ha_stationary_distribution(p, policy.a)
    mpc = ha_weighted_mpc(policy.c, policy.a, p.grids.a_grid, r, p.grids.e_grid)
    C_ha = sum(dist.D .* policy.c)
    A_ha = sum(dist.D .* policy.a)
    MPC_ha = sum(dist.D .* mpc)
    euler_residual = ha_euler_residuals(policy.c, policy.a, p.grids, r, p.beta, p.eis)
    avg_income = sum(dist.D .* reshape(y, :, 1))
    budget = r * A_ha + avg_income - C_ha
    return merge((; C_ha, A_ha, MPC_ha, euler=euler_residual,
                    budget, income=y, weighted_mpc=mpc), policy, dist)
end

function (p::HAHousehold)(x::NamedTuple)
    # EGM and discrete distribution transitions are intentionally implemented
    # for floating-point steady states. Do not claim AD through this fixed point.
    result = ha_steady_state(p, x.Z, x.r, x.r_ante, x.pure_profit, x.mu)
    return _attach_aggregate(result, x)
end

end # module GoldilocksHouseholds
