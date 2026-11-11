"""
    SteadyState_SamuelsonKinked.jl

Reproduce the kinked Samuelson (1966) reswitching example used in
Baqaee and Farhi (2019), Section 5.3 and Figure 3.

Source: `Reswitching - replication/SteadyState_SamuelsonKinked.m`.
The numerical definitions in the source are preserved. The plot design is
modernized, the misleading third-panel x-axis label is corrected, and results
are exported to CSV. The figure uses the Okabe--Ito palette; the optional
regime table is produced by SummaryTables.jl.

Usage from the project environment:
    julia --project=. SteadyState_SamuelsonKinked.jl

Install required packages in this environment beforehand:
    using Pkg
    Pkg.add(string(:CairoMakie))
    Pkg.add(string(:SummaryTables))
"""
module SamuelsonKinked

using CairoMakie
using Printf
using SummaryTables
using Test

export KinkedParams, kinked_equilibrium, simulate_kinked,
       switch_rates, save_kinked_figure, save_regime_table, run_replication

"""Structural parameters of the paper's kinked Samuelson economy."""
Base.@kwdef struct KinkedParams
    A1::Float64 = 1 / 4
    A2::Float64 = 1 / 7
    A3::Float64 = 1.0
    z3::Float64 = 1 / 12
    omega::Float64 = 1 / 2
    labor::Float64 = 1.0
end

"""
    kinked_equilibrium(beta, params=KinkedParams())

Calculate a single steady state at discount factor `beta`.
`technique == 2` means the shorter, one-period technique is selected;
`technique == 1` denotes the two-period technique (named `y3` in MATLAB).
At equal costs, the MATLAB source selects technique 1.
"""
function kinked_equilibrium(beta::Real, params::KinkedParams = KinkedParams())
    0 < beta < 1 || throw(ArgumentError("beta must lie in (0, 1)"))
    (; A1, A2, A3, z3, omega, labor) = params
    all(isfinite, (A1, A2, A3, z3, omega, labor)) ||
        throw(ArgumentError("parameters must be finite"))
    all(>(0), (A1, A2, A3, z3, labor)) ||
        throw(ArgumentError("technology coefficients and labor must be positive"))
    0 < omega < 1 || throw(ArgumentError("omega must lie in (0, 1)"))

    b = Float64(beta)
    p1 = inv(A1)
    p3 = (omega * p1 + (1 - omega) * b^2 / z3) / A3
    p2 = b / A2

    if p3 > p2
        y1 = 0.0
        y2 = A2 * labor
        y3 = 0.0
        technique = 2
    else
        y2 = 0.0
        y3 = labor / (omega / (A3 * A1) + (1 - omega) / (A3 * z3))
        y1 = omega * y3 / A3
        technique = 1
    end

    price_index = min(p2, p3)
    capital_value = b * p2 * y2 + b^2 * p1 * y1
    nominal_final_output = (b + 1) * p2 * y2 + b^2 * p1 * y1 + p3 * y3
    nominal_investment_base = b^2 + p1 * y1 + p2 * y2

    return (
        beta = b,
        gross_interest_rate = inv(b),
        technique = technique,
        p1 = p1,
        p2 = p2,
        p3 = p3,
        y1 = y1,
        y2 = y2,
        y3 = y3,
        capital_output_ratio = capital_value / nominal_final_output,
        real_gdp = nominal_final_output / price_index,
        price_difference = p3 - p2,
        capital_labor_ratio = capital_value / price_index,
        real_wage = b^2 / price_index,
        investment_rate = capital_value / nominal_investment_base,
    )
end

"""
    simulate_kinked(; params=KinkedParams(), n=100, beta_min=0.4, beta_max=0.9)

Simulate the same 100-point beta grid as the original MATLAB replication.
"""
function simulate_kinked(; params::KinkedParams = KinkedParams(),
                         n::Integer = 100,
                         beta_min::Real = 0.4,
                         beta_max::Real = 0.9)
    n >= 2 || throw(ArgumentError("n must be at least 2"))
    0 < beta_min < beta_max < 1 ||
        throw(ArgumentError("require 0 < beta_min < beta_max < 1"))
    states = kinked_equilibrium.(range(beta_min, beta_max; length=n), Ref(params))
    keys_out = propertynames(first(states))
    return NamedTuple{keys_out}(map(name -> getproperty.(states, name), keys_out))
end

"""
    switch_rates(params=KinkedParams())

Return the two (real) equilibrium gross-interest switching thresholds, sorted
in increasing order. For the paper's parameters these are 1.5 and 2.0.
"""
function switch_rates(params::KinkedParams = KinkedParams())
    (; A1, A2, A3, z3, omega) = params
    a = (1 - omega) / (A3 * z3)
    b = -inv(A2)
    c = omega / (A3 * A1)
    discriminant = b^2 - 4a * c
    discriminant > 0 || throw(ArgumentError("no two distinct real switch points"))
    roots = ((-b - sqrt(discriminant)) / (2a),
             (-b + sqrt(discriminant)) / (2a))
    all(>(0), roots) || throw(ArgumentError("nonpositive discount-factor switch"))
    return sort(collect(inv.(roots)))
end

# The eight Okabe--Ito colors, specified explicitly to avoid reliance on
# backend/version-dependent categorical palette ordering.
const OKABE_ITO = (
    "#000000", "#E69F00", "#56B4E9", "#009E73",
    "#F0E442", "#0072B2", "#D55E00", "#CC79A7",
)

"""
    save_kinked_figure(result, filepath; legacy_beta_axis=false)

Save a publication-quality 2-by-3 plot. The original MATLAB script mistakenly
plots `beta`, rather than `1/beta`, on the third panel but labels it "Interest
rate". By default every panel uses gross interest R=1/beta. To recreate the
original third-panel x-values, set `legacy_beta_axis=true`; its axis is then
correctly labeled "Discount factor beta".
"""
function save_kinked_figure(result::NamedTuple, filepath::AbstractString;
                            legacy_beta_axis::Bool = false)
    fig = Figure(size=(1500, 920), fontsize=18)
    specifications = (
        (:capital_output_ratio, "Capital / output ratio"),
        (:real_gdp, "Real GDP (CPI deflator)"),
        (:price_difference, "Price difference of techniques"),
        (:capital_labor_ratio, "Capital / labor ratio"),
        (:real_wage, "Real wage"),
        (:investment_rate, "Investment rate"),
    )
    for (i, (variable, y_label)) in enumerate(specifications)
        r = div(i - 1, 3) + 1
        c = mod(i - 1, 3) + 1
        using_beta = i == 3 && legacy_beta_axis
        x = using_beta ? result.beta : result.gross_interest_rate
        xlabel = using_beta ? "Discount factor β" : "Gross interest rate R = 1/β"
        ax = Axis(fig[r, c]; xlabel=xlabel, ylabel=y_label)
        lines!(ax, x, getproperty(result, variable);
               color=OKABE_ITO[mod1(i + 1, length(OKABE_ITO))], linewidth=3)
    end
    save(filepath, fig)
    return filepath
end

"""
    save_regime_table(params, filepath)

Use SummaryTables.jl to export the exact technology-choice regions in LaTeX.
"""
function save_regime_table(params::KinkedParams, filepath::AbstractString)
    lower, upper = switch_rates(params)
    entries = [
        "Gross interest rate" "Cost-minimizing technique";
        @sprintf("R < %.6g", lower) "Technique 2";
        @sprintf("%.6g < R < %.6g", lower, upper) "Technique 1";
        @sprintf("R > %.6g", upper) "Technique 2";
    ]
    cells = Cell.(entries)
    cells[1, 1] = Cell(entries[1, 1]; bold=true)
    cells[1, 2] = Cell(entries[1, 2]; bold=true)
    report = Table(cells; header=1)
    open(filepath, "w") do io
        show(io, MIME"text/latex"(), report)
    end
    return filepath
end

"""
    run_replication(; output_dir=joinpath(@__DIR__, "output"))

Run the simulation, check exact Samuelson switching and economic identities,
then export its numerical series, its figure (PNG and PDF), and the regime table.
"""
function run_replication(; output_dir::AbstractString = joinpath(@__DIR__, "output"))
    params = KinkedParams()
    mkpath(output_dir)
    results = simulate_kinked(; params=params)
    lower, upper = switch_rates(params)

    @test isapprox(lower, 1.5; atol=1e-12)
    @test isapprox(upper, 2.0; atol=1e-12)
    @test kinked_equilibrium(0.4, params).technique == 2
    @test kinked_equilibrium(0.6, params).technique == 1
    @test kinked_equilibrium(0.8, params).technique == 2
    @test length(results.beta) == 100
    @test all(isfinite, results.capital_output_ratio)
    @test all(isfinite, results.real_gdp)
    @test all(x -> 0 <= x <= 1, results.investment_rate)

    columns = (
        :beta, :gross_interest_rate, :technique, :capital_output_ratio,
        :real_gdp, :price_difference, :capital_labor_ratio,
        :real_wage, :investment_rate,
    )
    csvpath = joinpath(output_dir, "samuelson_kinked.csv")
    open(csvpath, "w") do io
        println(io, join(string.(columns), ","))
        for k in eachindex(results.beta)
            println(io, join((getproperty(results, col)[k] for col in columns), ","))
        end
    end

    save_kinked_figure(results, joinpath(output_dir, "samuelson_kinked.png"))
    save_kinked_figure(results, joinpath(output_dir, "samuelson_kinked.pdf"))
    save_regime_table(params, joinpath(output_dir, "samuelson_regimes.tex"))

    @info "Samuelson kinked replication exported" output_dir=abspath(output_dir)
    return results
end

end # module SamuelsonKinked

if abspath(PROGRAM_FILE) == @__FILE__
    SamuelsonKinked.run_replication()
end
