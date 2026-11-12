"""
    SteadyState_SamuelsonSmooth.jl

Replication of Baqaee and Farhi (2019), Section 5.3, Figure 4, based on
`Reswitching - replication/SteadyState_SamuelsonSmooth.m` (2018-11-14).

Two deliberately distinct specifications:
  * `mode=:matlab` (default) reproduces the original MATLAB equations,
    including its beta-weighted CPI and its labor denominator.
  * `mode=:consistent` uses the demand CES weight b in its CPI and corrects
    the labor requirement of input 1 by its unit labor cost 1/A1. It also
    scales the reported capital/labor ratio by L when L differs from 1.

The deviations are documented, reported, and tested. No silent correction is
made to the replicated source.

Required Julia packages: CairoMakie.jl, SummaryTables.jl.
Run: `julia --project=. SteadyState_SamuelsonSmooth.jl`
"""
module SamuelsonSmooth

using CairoMakie
using Printf
using SummaryTables
using Test

export SmoothParams, smooth_equilibrium, simulate_smooth, make_figure,
       export_csv, export_tables, run_tests, run_replication, OKABE_ITO

"""The Okabe--Ito color-blind-friendly palette, in fixed reproducible order."""
const OKABE_ITO = (
    "#000000", "#E69F00", "#56B4E9", "#009E73",
    "#F0E442", "#0072B2", "#D55E00", "#CC79A7",
)

"""
    SmoothParams(; A1=1/4, A2=1/7, A3=1, z3=1/12,
                   consumption_weight=1/2, theta=0.01,
                   sigma=400, omega=1/2, labor=1)

Parameters of the smooth Samuelson economy. `consumption_weight` is MATLAB's
`b`, not the discount factor `beta`.
"""
Base.@kwdef struct SmoothParams
    A1::Float64 = 1 / 4
    A2::Float64 = 1 / 7
    A3::Float64 = 1.0
    z3::Float64 = 1 / 12
    consumption_weight::Float64 = 0.5
    theta::Float64 = 0.01
    sigma::Float64 = 400.0
    omega::Float64 = 0.5
    labor::Float64 = 1.0
end

"""Check that model parameters define positive CES production and final demand."""
function validate(params::SmoothParams)
    p = params
    all(isfinite, (p.A1, p.A2, p.A3, p.z3, p.consumption_weight,
                   p.theta, p.sigma, p.omega, p.labor)) ||
        throw(ArgumentError("all model parameters must be finite"))
    all(x -> x > 0.0, (p.A1, p.A2, p.A3, p.z3, p.theta, p.sigma, p.labor)) ||
        throw(ArgumentError("technologies, elasticities, and labor must be positive"))
    0.0 < p.consumption_weight < 1.0 ||
        throw(ArgumentError("consumption_weight must be strictly between zero and one"))
    0.0 < p.omega < 1.0 ||
        throw(ArgumentError("omega must be strictly between zero and one"))
    return params
end

"""
    log_power_mean(x, y, weight, exponent)

Return `log((weight*x^exponent + (1-weight)*y^exponent)^(1/exponent))`.
The `exponent=0` limit is the weighted geometric mean. The log1p/expm1
formula avoids overflow and cancellation for very high CES elasticities.
"""
function log_power_mean(x::Real, y::Real, weight::Real, exponent::Real)
    x > 0 && y > 0 || throw(DomainError((x, y), "CES prices must be positive"))
    0 < weight < 1 || throw(DomainError(weight, "CES weight must be in (0,1)"))
    lx = log(Float64(x))
    ly = log(Float64(y))
    e = Float64(exponent)
    e == 0.0 && return weight * lx + (1 - weight) * ly
    z = e * (lx - ly)
    if z >= 0.0
        return lx + log1p((1 - weight) * expm1(-z)) / e
    else
        return ly + log1p(weight * expm1(z)) / e
    end
end

"""Stable log(exp(a) + exp(b) + exp(c)) for real log weights."""
@inline function logsumexp3(a::Float64, b::Float64, c::Float64)
    m = max(a, b, c)
    return m + log(exp(a - m) + exp(b - m) + exp(c - m))
end

"""
    smooth_equilibrium(beta, params=SmoothParams(); mode=:matlab)

Compute one steady state, including all six plotted MATLAB series.

`mode=:matlab` reproduces the source's (i) CPI weight beta instead of b
and (ii) labor normalization counting y1 rather than y1/A1.
For non-unit total labor, the source also omits division by labor in K/L.

`mode=:consistent` corrects these source conventions and is *not* the
paper's figure (including division by labor in K/L for non-unit labor).
It can be used to check the CES model's resource constraint and its kinked
limit. The state exposes both `labor_source` and `labor_physical` diagnostics.
"""
function smooth_equilibrium(beta::Real,
                            params::SmoothParams = SmoothParams();
                            mode::Symbol = :matlab)
    validate(params)
    0 < beta < 1 && isfinite(beta) ||
        throw(ArgumentError("beta must be finite and in (0,1)"))
    mode in (:matlab, :consistent) ||
        throw(ArgumentError("mode must be :matlab or :consistent"))

    (; A1, A2, A3, z3, consumption_weight, theta, sigma, omega, labor) = params
    b = consumption_weight
    discount = Float64(beta)
    log_beta = log(discount)
    log_A3 = log(A3)

    # Dual unit-cost equations. They coincide with the active MATLAB formulas.
    p1 = inv(A1)
    p2 = discount / A2
    log_p3 = log_power_mean(p1, discount^2 / z3, omega, 1 - theta) - log_A3
    p3 = exp(log_p3)
    log_p1 = log(p1)
    log_p2 = log(p2)

    # Unnormalised contributions to the labor resource constraint.
    # log_t1 is the CES demand for intermediate good 1 per unit y3.
    log_t1 = log(omega) - theta * (log_p1 - log_p3) + (theta - 1) * log_A3
    log_t2 = log(b) - log1p(-b) - sigma * (log_p2 - log_p3) - log(A2)
    log_t3 = log1p(-omega) - theta * (2 * log_beta - log_p3) +
             (theta - 1) * (log_A3 + log(z3))

    # The source uses log_t1 without its factor 1/A1; this must be preserved
    # in :matlab mode, even though the physical labor requirement is y1/A1.
    log_labor1 = mode === :matlab ? log_t1 : log_t1 - log(A1)
    log_denom = logsumexp3(log_labor1, log_t2, log_t3)
    log_labor = log(labor)

    # Evaluate all flows in log form to avoid y3 == 0 times an infinite
    # CES demand ratio at very high sigma.
    y1 = exp(log_labor + log_t1 - log_denom)
    y2 = exp(log_labor + log_t2 + log(A2) - log_denom)
    y3 = exp(log_labor - log_denom)
    labor_in_tech3 = exp(log_labor + log_t3 - log_denom)
    labor_in_tech2 = exp(log_labor + log_t2 - log_denom)
    labor_source = y1 + labor_in_tech2 + labor_in_tech3
    labor_physical = y1 / A1 + labor_in_tech2 + labor_in_tech3

    # CRITICAL SOURCE INCONSISTENCY: original MATLAB has discount as CPI
    # weight, not b. The consistent CES dual index uses b instead.
    cpi_weight = mode === :matlab ? discount : b
    price_index = exp(log_power_mean(p2, p3, cpi_weight, 1 - sigma))
    demand_consistent_price_index = exp(log_power_mean(p2, p3, b, 1 - sigma))

    capital_value = discount * p2 * y2 + discount^2 * p1 * y1
    nominal_gdp = (discount + 1) * p2 * y2 + discount^2 * p1 * y1 + p3 * y3
    investment_denominator = discount^2 + p1 * y1 + p2 * y2

    return (
        beta = discount,
        gross_interest_rate = inv(discount),
        p1 = p1, p2 = p2, p3 = p3,
        y1 = y1, y2 = y2, y3 = y3,
        capital_output_ratio = capital_value / nominal_gdp,
        real_gdp = nominal_gdp / price_index,
        price_difference = p3 - p2,
        capital_labor_ratio = capital_value /
            (mode === :matlab ? price_index : price_index * labor),
        real_wage = discount^2 / price_index,
        investment_rate = capital_value / investment_denominator,
        capital_value = capital_value,
        nominal_gdp = nominal_gdp,
        investment_denominator = investment_denominator,
        price_index = price_index,
        demand_consistent_price_index = demand_consistent_price_index,
        labor_source = labor_source,
        labor_physical = labor_physical,
        labor_resource_error = labor_physical - labor,
        labor_in_tech3 = labor_in_tech3,
        labor_in_tech2 = labor_in_tech2,
    )
end

"""
    simulate_smooth(params=SmoothParams(); n=100, beta_min=0.4,
                    beta_max=0.9, mode=:matlab)

Return a vector of steady-state named tuples on the original MATLAB grid.
"""
function simulate_smooth(params::SmoothParams = SmoothParams();
                         n::Integer = 100, beta_min::Real = 0.4,
                         beta_max::Real = 0.9, mode::Symbol = :matlab)
    n >= 2 || throw(ArgumentError("n must be at least 2"))
    0 < beta_min < beta_max < 1 ||
        throw(ArgumentError("need 0 < beta_min < beta_max < 1"))
    return [smooth_equilibrium(beta, params; mode=mode)
            for beta in range(beta_min, beta_max; length=n)]
end

"""Column names for exporting the six target series and numerical diagnostics."""
const CSV_COLUMNS = (
    :beta, :gross_interest_rate, :capital_output_ratio, :real_gdp,
    :price_difference, :capital_labor_ratio, :real_wage, :investment_rate,
    :p1, :p2, :p3, :y1, :y2, :y3, :price_index,
    :demand_consistent_price_index, :labor_source, :labor_physical,
    :labor_resource_error,
)

"""Export all replicated series and diagnostics as a headered numerical CSV."""
function export_csv(states::AbstractVector, filepath::AbstractString)
    open(filepath, "w") do io
        println(io, join(string.(CSV_COLUMNS), ","))
        for state in states
            println(io, join((repr(getproperty(state, key)) for key in CSV_COLUMNS), ","))
        end
    end
    return filepath
end

"""
    make_figure(states; axes=:corrected)

Create six panels in the paper's order, using Okabe--Ito blue throughout.
`axes=:corrected` plots gross interest on all six panels.
`axes=:paper` *exactly* imitates MATLAB's third-panel beta x data and its
incorrect "Interest rate" label, for fidelity comparisons only.
"""
function make_figure(states::AbstractVector; axes::Symbol = :corrected)
    axes in (:corrected, :paper) ||
        throw(ArgumentError("axes must be :corrected or :paper"))
    isempty(states) && throw(ArgumentError("states cannot be empty"))

    fig = Figure(size=(1500, 900), fontsize=16)
    panels = (
        (:capital_output_ratio, "Capital/Output Ratio"),
        (:real_gdp, "Real GDP (CPI Deflator)"),
        (:price_difference, "Price Difference of Techniques"),
        (:capital_labor_ratio, "Capital/Labor Ratio"),
        (:real_wage, "Real Wage"),
        (:investment_rate, "Investment Rate"),
    )
    gross_rates = getproperty.(states, Ref(:gross_interest_rate))
    betas = getproperty.(states, Ref(:beta))
    for (i, (series, label)) in enumerate(panels)
        row = fld(i - 1, 3) + 1
        col = mod(i - 1, 3) + 1
        legacy_axis = axes === :paper && i == 3
        x = legacy_axis ? betas : gross_rates
        axis = Axis(fig[row, col];
                    xlabel=(legacy_axis ? "Interest rate" : "Interest rate (gross R = 1/β)"),
                    ylabel=label,
                    xgridvisible=false,
                    ygridvisible=false)
        y = getproperty.(states, Ref(series))
        lines!(axis, x, y; color=OKABE_ITO[6], linewidth=2.8)
    end
    return fig
end

"""Build and save a custom SummaryTables.jl Table to a LaTeX file."""
function latex_table(data::AbstractMatrix{<:AbstractString}, filepath::AbstractString)
    table = Table(Cell.(data); header=1)
    open(filepath, "w") do io
        show(io, MIME"text/latex"(), table)
    end
    return filepath
end

"""Build parameter and numerical-diagnostic tables with SummaryTables.jl."""
function export_tables(params::SmoothParams, states::AbstractVector,
                       output_dir::AbstractString; mode::Symbol = :matlab)
    mkpath(output_dir)
    parameter_rows = (
        ("A1", params.A1), ("A2", params.A2), ("A3", params.A3),
        ("z3", params.z3), ("b", params.consumption_weight),
        ("theta", params.theta), ("sigma", params.sigma),
        ("omega", params.omega), ("L", params.labor),
    )
    par_data = Matrix{String}(undef, length(parameter_rows) + 2, 2)
    par_data[1, :] = ["Parameter", "Value"]
    for (i, (name, value)) in enumerate(parameter_rows)
        par_data[i + 1, :] = [name, @sprintf("%.12g", value)]
    end
    par_data[end, :] = ["mode", string(mode)]
    par_path = latex_table(par_data, joinpath(output_dir, "samuelson_smooth_parameters.tex"))

    worst_physical = maximum(abs(s.labor_resource_error) for s in states)
    worst_source = maximum(abs(s.labor_source - params.labor) for s in states)
    worst_cpi = maximum(abs(s.price_index - s.demand_consistent_price_index)
                        for s in states)
    diagnostic_rows = (
        ("Grid points", Float64(length(states))),
        ("Max source labor-normalization residual", worst_source),
        ("Max physical labor-resource residual", worst_physical),
        ("Max source-vs-CES CPI discrepancy", worst_cpi),
    )
    diagnostic_data = Matrix{String}(undef, length(diagnostic_rows) + 1, 2)
    diagnostic_data[1, :] = ["Diagnostic", "Value"]
    for (i, (name, value)) in enumerate(diagnostic_rows)
        diagnostic_data[i + 1, :] = [name, @sprintf("%.12g", value)]
    end
    diag_path = latex_table(diagnostic_data,
                            joinpath(output_dir, "samuelson_smooth_diagnostics.tex"))
    return (parameters=par_path, diagnostics=diag_path)
end

"""
    run_tests()

Deterministic checks: MATLAB formula anchor values (independently evaluated),
resource accounting, CES demand and production limits. Anchor values are
not the output of an actual MATLAB run; they were computed directly from
`SteadyState_SamuelsonSmooth.m` formulas using independent Python arithmetic.
"""
function run_tests()
    @testset "Samuelson Smooth numerical and economic tests" begin
        p = SmoothParams()
        @test isapprox(log_power_mean(2.0, 8.0, 0.5, 0.0),
                       log(4.0); atol=1e-14)
        @test isapprox(exp(log_power_mean(2.0, 8.0, 0.5, -1.0)),
                       3.2; atol=1e-14)
        @test isapprox(exp(log_power_mean(2.0, 8.0, 0.5, -1e6)),
                       2.0; rtol=1e-5)

        sample = smooth_equilibrium(0.6, p; mode=:matlab)
        # The following anchors are direct, literal evaluations of MATLAB.
        # The stable CES implementation need not reproduce subnormal rounding.
        @test isapprox(sample.p3, 4.15996922306551; rtol=2e-10)
        @test isapprox(sample.y1, 0.07521987318100318; rtol=2e-10)
        @test isapprox(sample.y2, 0.003262294481364213; rtol=2e-10)
        @test isapprox(sample.y3, 0.15038076551557275; rtol=2e-10)
        @test isapprox(sample.capital_output_ratio, 0.1541872621520069; rtol=2e-10)
        @test isapprox(sample.real_gdp, 0.18128642923387095; rtol=2e-7)
        @test isapprox(sample.price_difference, -0.040030776934489865; atol=1e-10)
        @test isapprox(sample.capital_labor_ratio, 0.02795205818888411; rtol=2e-7)
        @test isapprox(sample.real_wage, 0.08634759076422148; rtol=2e-7)
        @test isapprox(sample.investment_rate, 0.1727554987376506; rtol=2e-10)

        @test isapprox(sample.labor_source, p.labor; atol=1e-12)
        @test sample.labor_physical > 1.2 # MATLAB's missing /A1 is detectable
        @test isapprox(sample.labor_resource_error,
                       sample.y1 * (inv(p.A1) - 1); atol=1e-12)
        @test isapprox(log(sample.y2 / sample.y3),
                       log(p.consumption_weight / (1 - p.consumption_weight)) -
                       p.sigma * log(sample.p2 / sample.p3); atol=1e-10)
        @test !isapprox(sample.price_index, sample.demand_consistent_price_index;
                        rtol=1e-8)

        corrected = smooth_equilibrium(0.6, p; mode=:consistent)
        @test isapprox(corrected.labor_physical, p.labor; atol=1e-12)
        @test isapprox(corrected.price_index,
                       corrected.demand_consistent_price_index; atol=1e-12)
        @test corrected.y3 < sample.y3

        records = simulate_smooth(p)
        @test length(records) == 100
        @test all(isfinite(s.real_gdp) && isfinite(s.investment_rate) &&
                  isfinite(s.capital_output_ratio) for s in records)
        @test all(0 <= s.investment_rate <= 1 for s in records)
        @test all(isapprox(s.labor_source, p.labor; atol=1e-10)
                  for s in records)
        @test first(records).beta == 0.4
        @test last(records).beta == 0.9
        @test isapprox(first(records).real_gdp, 0.19954123336786847; rtol=2e-7)
        @test isapprox(last(records).real_gdp, 0.27135688020317145; rtol=2e-7)

        # Any weight b in (0,1) is supported in the consistent CES system.
        alternative = SmoothParams(consumption_weight=0.3, labor=2.0)
        alt = smooth_equilibrium(0.6, alternative; mode=:consistent)
        @test isapprox(alt.labor_physical, alternative.labor; atol=1e-12)
        @test isapprox(alt.price_index, alt.demand_consistent_price_index;
                       atol=1e-12)
        @test isapprox(alt.capital_labor_ratio,
                       alt.capital_value / (alt.price_index * alternative.labor);
                       atol=1e-12)

        # Away from switching thresholds, the consistent model converges
        # to the kinked model as theta -> 0 and sigma -> infinity.
        near_kinked = SmoothParams(theta=1e-8, sigma=10000.0)
        low = smooth_equilibrium(0.6, near_kinked; mode=:consistent)
        high = smooth_equilibrium(0.8, near_kinked; mode=:consistent)
        @test isapprox(low.y3, 1 / 8; atol=2e-7)
        @test isapprox(low.y1, 1 / 16; atol=2e-7)
        @test low.y2 <= 1e-12
        @test high.y3 <= 1e-12
        @test isapprox(high.y2, near_kinked.A2 * near_kinked.labor;
                       atol=2e-7)
    end
    return nothing
end

"""
    run_replication(; output_dir=joinpath(@__DIR__, "output"), mode=:matlab)

Run self-tests, compute the 100 baseline observations, export CSV, PNG/PDF
figures (both corrected and original axes), and SummaryTables LaTeX tables.
"""
function run_replication(; output_dir::AbstractString = joinpath(@__DIR__, "output"),
                         params::SmoothParams = SmoothParams(),
                         mode::Symbol = :matlab)
    run_tests()
    records = simulate_smooth(params; mode=mode)
    mkpath(output_dir)

    basename = mode === :matlab ? "samuelson_smooth" : "samuelson_smooth_consistent"
    export_csv(records, joinpath(output_dir, basename * ".csv"))
    figure = make_figure(records; axes=:corrected)
    save(joinpath(output_dir, basename * ".png"), figure)
    save(joinpath(output_dir, basename * ".pdf"), figure)
    if mode === :matlab
        # Original Figure 4 uses beta rather than 1/beta in the third panel.
        legacy_figure = make_figure(records; axes=:paper)
        save(joinpath(output_dir, basename * "_paper_axes.png"), legacy_figure)
        save(joinpath(output_dir, basename * "_paper_axes.pdf"), legacy_figure)
    end
    export_tables(params, records, output_dir; mode=mode)
    @info "Smooth Samuelson replication files exported" directory=abspath(output_dir) mode=mode
    return records
end

end # module SamuelsonSmooth

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    SamuelsonSmooth.run_replication()
end
