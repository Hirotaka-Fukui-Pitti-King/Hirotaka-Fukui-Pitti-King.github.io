"""
    Krusell_agg.jl

Reproduce the *aggregate* Krusell example of Baqaee and Farhi (2019),
JEEA 17(5), Appendix B, Tables B.1 and B.5, using the original five MAT
files and the existing local Julia companions Relabel_mod.jl and
WeightedCovMatrix.jl. No MATLAB or pre-converted CSV is required.

The published cost MES for Noncollege vs College is 1.07, whereas the supplied
MAT data and the effective MATLAB expressions yield 1.0648561528726668,
which rounds to 1.06 under usual two-decimal rounding. THIS DISCREPANCY IS
NOT SILENTLY ALTERED. Every other off-diagonal cell rounds as published.

Required packages: MAT.jl, SummaryTables.jl. Standard libraries:
LinearAlgebra, SparseArrays, Statistics, Printf, Random, Test.

Run from PowerShell, after installing packages and placing the two companion
files in the same directory:

    julia --depwarn=error --project=. Krusell_agg.jl --data-dir "path/to/Empirical Application"

Use `--test-only` for synthetic tests without any MAT data, `--out-dir PATH`
to change the output location, or `--bench` for a warm-started benchmark.
"""

# Load both companion modules BEFORE defining functions which use them.
# Never dynamically include a module inside run_tests (Julia 1.12 world age).
include(joinpath(@__DIR__, "Relabel_mod.jl"))
include(joinpath(@__DIR__, "WeightedCovMatrix.jl"))

module KrusellAggregate

using ..NestedCESRelabelling: relabel_nested_ces, normalize_input_shares
using ..WeightedCovMatrix: weighted_covariance, aggregate_weighted_covariance
using LinearAlgebra
using SparseArrays
using Statistics
using Printf
using Random
using Test
using MAT: matread
using SummaryTables: Table, Cell

export load_krusell_data, prepare_aggregate_inputs, compute_aggregate_equilibrium,
       compute_aggregate_second_order, compute_aggregate_mes,
       compute_capital_counterfactual, compute_aggregate_model,
       save_aggregate_outputs, run_tests, run_replication, run_benchmark

const N_INDUSTRIES = 63
const N = 65
const K = 4
const M = 2 * N + K
const Q = 5 * N + K
const FACTOR_IDS = (Q - K + 1):(Q - 1)  # 326:328; 329 is an unused terminal factor
const FACTOR_NAMES = ("Capital", "Noncollege", "College")
const MATLAB_FILES = (
    ("IO2013.mat", "IO2013", (83, 98)),
    ("output_new.mat", "output_new", (63, 1)),
    ("VA_new.mat", "VA_new", (63, 1)),
    ("labor_no_col.mat", "labor_no_col", (63, 1)),
    ("labor_col.mat", "labor_col", (63, 1)),
)

"""
    _validated_column(x, label; length_expected=63)

Convert a 63x1 or 63-element real data array to Float64 without changing its
ordering; reject unexpected shapes, missing/non-finite entries, and negatives.
"""
function _validated_column(x::AbstractArray{<:Real}, label::AbstractString;
                           length_expected::Int=N_INDUSTRIES)
    (size(x) == (length_expected, 1) || ndims(x) == 1 && length(x) == length_expected) ||
        throw(DimensionMismatch("$label must be $(length_expected)x1 or a vector of that length; got $(size(x))"))
    values = vec(Float64.(x))
    all(isfinite, values) || throw(ArgumentError("$label contains non-finite values"))
    all(x -> x >= 0.0, values) || throw(ArgumentError("$label contains negative values"))
    return values
end

"""
    validate_data(data)

Validate the five original data arrays. IO2013 and output_new are audited and
loaded for replication fidelity, but neither affects the effective final
aggregate-model calculations in Krusell_agg.m.
"""
function validate_data(data::NamedTuple)
    size(data.io) == (83, 98) || throw(DimensionMismatch("IO2013 must be 83x98"))
    all(isfinite, data.io) || throw(ArgumentError("IO2013 has non-finite entries"))
    for key in (:output, :va, :labor_noncollege, :labor_college)
        _validated_column(getproperty(data, key), string(key))
    end
    sum(data.va) > 0 || throw(ArgumentError("sum(VA_new) must be positive"))
    sum(data.va) > sum(data.labor_noncollege) ||
        throw(ArgumentError("sum(VA_new) must exceed sum(noncollege labor)"))
    sum(data.labor_college) <= sum(data.va) - sum(data.labor_noncollege) ||
        throw(ArgumentError("college compensation exceeds residual value added"))
    return nothing
end

"""
    load_krusell_data(data_dir)

Read and validate the five original MATLAB MAT files using MAT.matread.
Expected MATLAB variable names and shapes are checked. Returns a NamedTuple
with real, finite arrays. The data are NOT pre-processed or silently modified.
"""
function load_krusell_data(data_dir::AbstractString)
    isdir(data_dir) || throw(ArgumentError("Data directory does not exist: $data_dir"))
    values = Dict{String,Matrix{Float64}}()
    for (filename, key, expected) in MATLAB_FILES
        path = joinpath(data_dir, filename)
        isfile(path) || throw(ArgumentError("Required MAT file missing: $path"))
        data = matread(path)
        haskey(data, key) || throw(ArgumentError("Variable '$key' not found in $path"))
        source = data[key]
        source isa AbstractMatrix{<:Real} ||
            throw(ArgumentError("Variable $key in $path must be a real matrix"))
        size(source) == expected || throw(DimensionMismatch(
            "Variable $key must have dimensions $expected; found $(size(source))"))
        values[key] = Matrix{Float64}(source)
    end
    result = (io=values["IO2013"],
              output=vec(values["output_new"]),
              va=vec(values["VA_new"]),
              labor_noncollege=vec(values["labor_no_col"]),
              labor_college=vec(values["labor_col"]))
    validate_data(result)
    return result
end

"""
    prepare_aggregate_inputs(data)

Mirror the *effective* MATLAB `Krusell_agg.m` input construction (lines 64-155).
In particular: original IO data and output_new do not enter the final Omega;
industry noncollege labor shares are all set to the same aggregate ratio;
Fshares is explicitly preallocated as 195x134 rather than MATLAB's dynamic
array expansion. The 1975 skilled/nonskilled calibration variables are also
unused in the surviving capital-shock results.

Return a NamedTuple of original and relabelled inputs. No changes to the
structural calibration or normalizations are introduced.
"""
function prepare_aggregate_inputs(data::NamedTuple)
    validate_data(data)
    share_noncollege = sum(data.labor_noncollege) / sum(data.va)
    share_college_within_residual = sum(data.labor_college) /
                                    (sum(data.va) - sum(data.labor_noncollege))
    0.0 <= share_noncollege <= 1.0 || throw(ArgumentError("invalid noncollege labor share"))
    0.0 <= share_college_within_residual <= 1.0 ||
        throw(ArgumentError("invalid college labor share"))

    labor_share = [0.0, 0.0, fill(share_noncollege, N_INDUSTRIES)...]
    capital_share = [0.0, 0.0, fill(1.0 - share_noncollege, N_INDUSTRIES)...]
    Fshares = zeros(Float64, 3 * N, M)
    # MATLAB Fshares(1:N,1:N)=diag(L_shares), and second block diag(K_shares).
    for i in 1:N
        Fshares[i, i] = labor_share[i]
        Fshares[i, N + i] = capital_share[i]
    end
    # MATLAB: Fshares(1+N:2*N, 2*N+2)=1;
    Fshares[(N + 1):(2 * N), 2 * N + 2] .= 1.0
    # MATLAB: Fshares(1+2*N:3*N, 2*N+1)=1-h; ...2*N+3=h.
    Fshares[(2 * N + 1):(3 * N), 2 * N + 1] .= 1.0 - share_college_within_residual
    Fshares[(2 * N + 1):(3 * N), 2 * N + 3] .= share_college_within_residual
    Fshares = normalize_input_shares(Fshares)  # mirrors bsxfun rdivide and NaN->0
    Fshares[1, 2 * N + 4] = 0.0             # literal no-op preserved as comment-worthy
    Fshares[2, 2 * N + 1] = 0.0             # literal no-op preserved

    # F_cons=[1;zeros(62)]; Omega(1,3:65)=F_cons, all other entries zero.
    Omega = zeros(Float64, N, N)
    Omega[1, 3] = 1.0
    Omega = normalize_input_shares(Omega)

    alpha = vcat(zeros(Float64, 2), ones(Float64, N_INDUSTRIES))
    epsilon = ones(Float64, N)
    theta = vcat(ones(Float64, 2), fill(0.0001, N_INDUSTRIES))
    eta = vcat(ones(Float64, 2), fill(1.67, N_INDUSTRIES),
               fill(0.67, N), fill(0.67, N))
    sigma = 1.0
    final_demand = zeros(Float64, Q)
    final_demand[1] = 1.0

    # Sparse mapping is kept until the separate MATLAB caller normalization.
    relabelled = relabel_nested_ces(Omega, Fshares, alpha, epsilon, theta,
                                    eta, final_demand; K=K, sparse=true)
    return (omega_original=Omega, fshares=Fshares, alpha=alpha,
            epsilon=epsilon, theta=theta, eta=eta, sigma=sigma,
            share_noncollege=share_noncollege,
            share_college_within_residual=share_college_within_residual,
            omega_raw=relabelled.omega,
            elasticities=relabelled.elasticities,
            final_demand=relabelled.final_demand)
end

"""
    compute_aggregate_equilibrium(inputs)

Normalize the relabelled matrix exactly as in the MATLAB caller, solve
`(I-Omega_tr)*Psi = I` using LU rather than an explicit inv, and obtain
Domar weights `lambda = Psi' * b`. The source network is sparse; the caller's
`normalize_input_shares` currently returns a dense 329x329 normalized matrix.
"""
function compute_aggregate_equilibrium(inputs::NamedTuple)
    Omega_tr = normalize_input_shares(inputs.omega_raw)
    q = length(inputs.final_demand)
    size(Omega_tr) == (q, q) || throw(DimensionMismatch("wrong relabelled dimensions"))
    eye_q = Matrix{Float64}(I, q, q)
    A = eye_q - Omega_tr
    factorization = lu(A)
    Psi = factorization \ eye_q
    lambda = transpose(Psi) * inputs.final_demand
    residual = maximum(abs, A * Psi - eye_q)
    all(isfinite, Psi) || throw(ArgumentError("non-finite Leontief inverse"))
    all(isfinite, lambda) || throw(ArgumentError("non-finite Domar weights"))
    return (omega=Omega_tr, psi=Psi, lambda=lambda,
            leontief_residual=residual,
            matrix_condition=cond(A),
            factor_ids=collect(FACTOR_IDS))
end

"""
    compute_aggregate_second_order(omega, psi, lambda, elasticities; sigma=1.0,
                                   final_demand)

Compute the full QxQ log-cost Hessian using the exact matrix identity from
WeightedCovMatrix.jl. This avoids the 329^3 tensor (>285 MB) in the MATLAB
program. The final-demand contribution vanishes exactly when sigma=1.
"""
function compute_aggregate_second_order(omega::AbstractMatrix{<:Real},
                                        psi::AbstractMatrix{<:Real},
                                        lambda::AbstractVector{<:Real},
                                        elasticities::AbstractVector{<:Real};
                                        sigma::Real=1.0,
                                        final_demand::AbstractVector{<:Real})
    q = size(omega, 1)
    size(omega) == (q, q) || throw(DimensionMismatch("omega must be square"))
    size(psi) == (q, q) || throw(DimensionMismatch("psi and omega sizes differ"))
    length(lambda) == q || throw(DimensionMismatch("lambda length must be $q"))
    length(elasticities) == q || throw(DimensionMismatch("elasticities length must be $q"))
    length(final_demand) == q || throw(DimensionMismatch("final_demand length must be $q"))
    isfinite(sigma) || throw(ArgumentError("sigma must be finite"))
    coefficients = lambda .* (elasticities .- 1.0)
    H = aggregate_weighted_covariance(omega, psi, coefficients)
    if sigma != 1.0
        H .+= (sigma - 1.0) .* weighted_covariance(final_demand, psi)
    end
    return (hessian=H, coefficients=coefficients,
            symmetry_residual=maximum(abs, H - transpose(H)))
end

"""
    compute_aggregate_mes(hessian, lambda; factor_ids=326:328)

Reproduce MATLAB `Elast`, `Nonelast`, `rho_1` and `rho_2` EXACTLY in index
orientation. `rho_1` and `rho_2` here are the MATLAB RAW matrices, which are
the transposes of the display matrices in published Table B.1.

Avoid inv by solving `(I+Elast)*Nonelast = Elast`; diagonal scaling remains
on the RIGHT, since factor j's income share normalizes column j.
"""
function compute_aggregate_mes(hessian::AbstractMatrix{<:Real},
                               lambda::AbstractVector{<:Real};
                               factor_ids=FACTOR_IDS)
    n = size(hessian, 1)
    size(hessian) == (n, n) || throw(DimensionMismatch("hessian must be square"))
    length(lambda) == n || throw(DimensionMismatch("wrong lambda size"))
    inds = collect(factor_ids)
    length(inds) == 3 || throw(DimensionMismatch("expected three MES factors"))
    all(i -> 1 <= i <= n, inds) || throw(BoundsError(lambda, inds))
    shares = lambda[inds]
    all(x -> isfinite(x) && x > 0, shares) ||
        throw(ArgumentError("MES factor income shares must be finite and strictly positive"))
    SubMatr = Matrix{Float64}(hessian[inds, inds])
    Elast = SubMatr * Diagonal(1.0 ./ shares)
    Nonelast = (Matrix{Float64}(I, 3, 3) + Elast) \ Elast
    Diffs_1 = reshape(diag(Nonelast), 3, 1) .- Nonelast
    Diffs_2 = -(reshape(diag(Elast), 3, 1) .- Elast)
    rho_1 = -1.0 ./ (Diffs_1 .- 1.0)
    rho_2 = 1.0 .- Diffs_2
    return (mes_production=rho_1, mes_cost=rho_2,
            mes_production_paper=Matrix(transpose(rho_1)),
            mes_cost_paper=Matrix(transpose(rho_2)),
            elast=Elast, nonelast=Nonelast, sub_hessian=SubMatr,
            factor_shares=shares, factor_ids=inds)
end

"""
    compute_capital_counterfactual(nonelast, elast)

Reproduce MATLAB's FINAL surviving Shock_1 and Shock_2 after the capital
shock overwrites the prior skilled-labor shock. Row orientation is unchanged:
Shock_1 = shock_c' * Nonelast, Shock_2 = shock_c' * Elast.
"""
function compute_capital_counterfactual(nonelast::AbstractMatrix{<:Real},
                                        elast::AbstractMatrix{<:Real})
    size(nonelast) == (3, 3) || throw(DimensionMismatch("Nonelast must be 3x3"))
    size(elast) == (3, 3) || throw(DimensionMismatch("Elast must be 3x3"))
    shock = -(1.075 ^ 38 - 1.062 ^ 38) / (1.075 ^ 38)
    shock_vector = [shock, 0.0, 0.0]
    # transpose(shock_vector) * nonelast is 1x3. Return a length-3 vector.
    prod = vec(transpose(shock_vector) * nonelast)
    cost = vec(transpose(shock_vector) * elast)
    return (shock=shock, shock_vector=shock_vector,
            capital_shock_production=prod, capital_shock_cost=cost)
end

"""
    compute_aggregate_model(data)

Calculate the model from already-loaded original MAT data. All downstream
objects are preserved in `equilibrium` and `second_order` for diagnostics.
"""
function compute_aggregate_model(data::NamedTuple)
    inputs = prepare_aggregate_inputs(data)
    equilibrium = compute_aggregate_equilibrium(inputs)
    second_order = compute_aggregate_second_order(
        equilibrium.omega, equilibrium.psi, equilibrium.lambda,
        inputs.elasticities; sigma=inputs.sigma,
        final_demand=inputs.final_demand)
    mes = compute_aggregate_mes(second_order.hessian, equilibrium.lambda)
    counterfactual = compute_capital_counterfactual(mes.nonelast, mes.elast)
    return (inputs=inputs, equilibrium=equilibrium, second_order=second_order,
            mes_production=mes.mes_production, mes_cost=mes.mes_cost,
            mes_production_paper=mes.mes_production_paper,
            mes_cost_paper=mes.mes_cost_paper,
            capital_shock_production=counterfactual.capital_shock_production,
            capital_shock_cost=counterfactual.capital_shock_cost,
            mes=mes, counterfactual=counterfactual)
end

const TABLE_B1_PRODUCTION = [
    NaN 1.67 0.67;
    1.05 NaN 0.88;
    0.67 1.67 NaN
]
const TABLE_B1_COST = [
    NaN 1.67 0.67;
    1.28 NaN 1.07;
    0.67 1.67 NaN
]
const TABLE_B5_AGGREGATE = [0.05, 0.07, -0.13]

"""
    published_discrepancies(result)

List the (row,column) and magnitudes of any off-diagonal B.1 printed cells
that do not match ordinary two-decimal rounding of the computed values.
Expected known exception: Cost, Noncollege row, College column.
"""
function published_discrepancies(result)
    output = NamedTuple[]
    for (name, computed, target) in
        (("Production", result.mes_production_paper, TABLE_B1_PRODUCTION),
         ("Cost", result.mes_cost_paper, TABLE_B1_COST))
        for i in 1:3, j in 1:3
            if i != j && round(computed[i, j]; digits=2) != target[i, j]
                push!(output, (table=name, row=FACTOR_NAMES[i],
                               column=FACTOR_NAMES[j],
                               computed=computed[i, j], published=target[i, j],
                               raw_gap=computed[i, j] - target[i, j]))
            end
        end
    end
    return output
end

function _save_matrix_csv(path::AbstractString, matrix::AbstractMatrix;
                          orientation::String)
    size(matrix) == (3, 3) || throw(DimensionMismatch("MES CSV requires 3x3"))
    open(path, "w") do io
        println(io, "row_orientation_", orientation,
                ",Capital,Noncollege,College")
        for i in 1:3
            print(io, FACTOR_NAMES[i])
            for j in 1:3
                @printf(io, ",%.17g", matrix[i, j])
            end
            println(io)
        end
    end
    return path
end

function _save_diagnostics_csv(path, result)
    e = result.equilibrium
    s = result.second_order
    m = result.mes
    metric = [
        ("industry_count", Float64(N_INDUSTRIES)),
        ("source_nodes", Float64(N)),
        ("relabelled_nodes", Float64(Q)),
        ("Fshares_rows", Float64(size(result.inputs.fshares, 1))),
        ("Fshares_columns", Float64(size(result.inputs.fshares, 2))),
        ("nonzero_omega", Float64(count(x -> !iszero(x), e.omega))),
        ("share_noncollege", result.inputs.share_noncollege),
        ("share_college_within_residual", result.inputs.share_college_within_residual),
        ("sum_final_demand", sum(result.inputs.final_demand)),
        ("sum_terminal_factor_shares", sum(e.lambda[end-K+1:end])),
        ("terminal_4_unused_share", e.lambda[end]),
        ("lambda_capital", m.factor_shares[1]),
        ("lambda_noncollege", m.factor_shares[2]),
        ("lambda_college", m.factor_shares[3]),
        ("leontief_inverse_residual_max", e.leontief_residual),
        ("leontief_matrix_condition_2", e.matrix_condition),
        ("hessian_symmetry_residual_max", s.symmetry_residual),
        ("capital_shock_dlogK", result.counterfactual.shock),
        ("published_B1_mismatch_count", Float64(length(published_discrepancies(result)))),
        ("paper_cost_Noncollege_College", TABLE_B1_COST[2, 3]),
        ("computed_cost_Noncollege_College", result.mes_cost_paper[2, 3]),
    ]
    open(path, "w") do io
        println(io, "metric,value")
        for (label, val) in metric
            @printf(io, "%s,%.17g\n", label, val)
        end
    end
    return path
end

function _mes_table_data(matrix::AbstractMatrix)
    data = Matrix{String}(undef, 4, 4)
    data[1, :] = ["Factor", "Capital", "Noncollege", "College"]
    for i in 1:3
        data[i+1, 1] = FACTOR_NAMES[i]
        for j in 1:3
            data[i+1, j+1] = i == j ? "-" : @sprintf("%.2f", matrix[i, j])
        end
    end
    return data
end

function _summarytable_tex(path::AbstractString, text_matrix::Matrix{String})
    # SummaryTables.jl Table/Cell and show(::MIME"text/latex") as documented.
    tab = Table(Cell.(text_matrix); header=1)
    open(path, "w") do io
        show(io, MIME"text/latex"(), tab)
        println(io)
    end
    return path
end

"""
    save_aggregate_outputs(result, out_dir)

Write the raw MATLAB-oriented MES CSVs, additionally the paper-oriented CSVs,
capital-shock and diagnostic CSVs, and publication tables with SummaryTables.jl.
No figures or fabricated parameter estimates are created. The known B.1
cost-cell mismatch remains visible in the actual table and diagnostic CSV.
"""
function save_aggregate_outputs(result, out_dir::AbstractString)
    mkpath(out_dir)
    outputs = String[]
    push!(outputs, _save_matrix_csv(joinpath(out_dir, "krusell_agg_mes_production.csv"),
                                    result.mes_production; orientation="matlab_raw"))
    push!(outputs, _save_matrix_csv(joinpath(out_dir, "krusell_agg_mes_cost.csv"),
                                    result.mes_cost; orientation="matlab_raw"))
    push!(outputs, _save_matrix_csv(joinpath(out_dir, "krusell_agg_mes_production_paper.csv"),
                                    result.mes_production_paper; orientation="paper_display"))
    push!(outputs, _save_matrix_csv(joinpath(out_dir, "krusell_agg_mes_cost_paper.csv"),
                                    result.mes_cost_paper; orientation="paper_display"))
    shock_path = joinpath(out_dir, "krusell_agg_capital_shock.csv")
    open(shock_path, "w") do io
        println(io, "factor,production_income_share_change,cost_income_share_change")
        for i in 1:3
            @printf(io, "%s,%.17g,%.17g\n", FACTOR_NAMES[i],
                    result.capital_shock_production[i], result.capital_shock_cost[i])
        end
    end
    push!(outputs, shock_path)
    push!(outputs, _save_diagnostics_csv(joinpath(out_dir, "krusell_agg_diagnostics.csv"), result))

    push!(outputs, _summarytable_tex(joinpath(out_dir, "krusell_agg_table_B1_production.tex"),
                                     _mes_table_data(result.mes_production_paper)))
    push!(outputs, _summarytable_tex(joinpath(out_dir, "krusell_agg_table_B1_cost.tex"),
                                     _mes_table_data(result.mes_cost_paper)))
    shock_data = Matrix{String}(undef, 4, 2)
    shock_data[1, :] = ["Factor", "Aggregate model"]
    for i in 1:3
        shock_data[i+1, 1] = FACTOR_NAMES[i]
        shock_data[i+1, 2] = @sprintf("%.2f", result.capital_shock_production[i])
    end
    push!(outputs, _summarytable_tex(joinpath(out_dir, "krusell_agg_table_B5.tex"), shock_data))
    return outputs
end

"""
    _synthetic_data(seed=20261112)

Synthetic data have plausible array shapes but are NEVER substituted for
original MAT data when producing B.1/B.5 replication outputs.
"""
function _synthetic_data(seed::Int=20261112)
    rng = MersenneTwister(seed)
    return (io=zeros(Float64, 83, 98),
            output=rand(rng, N_INDUSTRIES) .* 100 .+ 20,
            va=fill(20.0, N_INDUSTRIES),
            labor_noncollege=fill(5.0, N_INDUSTRIES),
            labor_college=fill(6.0, N_INDUSTRIES))
end

"""
    run_tests(; data_dir=nothing)

Check MATLAB block structure, zero normalization, companion-module integration,
small-network covariance identity, 329-node aggregate equilibrium, exact
analytical MES relationships, data loading and published numerical anchors.
If data_dir is nothing, only synthetic tests run, labelled accordingly.
"""
function run_tests(; data_dir::Union{Nothing,AbstractString}=nothing)
    @testset "Krusell_agg.jl" begin
        @testset "Input audit and economic preparation (synthetic)" begin
            data = _synthetic_data()
            inputs = prepare_aggregate_inputs(data)
            @test size(inputs.omega_original) == (N, N)
            @test size(inputs.fshares) == (3*N, M)
            @test size(inputs.omega_raw) == (Q, Q)
            @test inputs.omega_raw isa SparseMatrixCSC
            @test length(inputs.elasticities) == Q
            @test length(inputs.final_demand) == Q
            @test inputs.final_demand[1] == 1.0
            @test sum(inputs.final_demand) == 1.0
            @test inputs.omega_original[1, 3] == 1.0
            @test count(x -> !iszero(x), inputs.omega_original) == 1
            @test inputs.alpha[1:2] == [0.0, 0.0]
            @test all(==(1.0), inputs.alpha[3:end])
            @test inputs.theta[1:2] == [1.0, 1.0]
            @test all(==(0.0001), inputs.theta[3:end])
            @test inputs.eta[1:2] == [1.0, 1.0]
            @test all(==(1.67), inputs.eta[3:N])
            @test all(==(0.67), inputs.eta[(N+1):end])
            @test all(==(0.0), inputs.fshares[1, :])
            @test all(==(0.0), inputs.fshares[2, :])
            @test all(isapprox(sum(view(inputs.fshares,i,:)), 1.0; atol=1e-14)
                      for i in 3:(3*N))
            @test isapprox(inputs.fshares[3,3],0.25; atol=1e-14)
            @test isapprox(inputs.fshares[3,N+3],0.75; atol=1e-14)
            @test isapprox(inputs.fshares[2*N+3,2*N+1], 0.6; atol=1e-14)
            @test isapprox(inputs.fshares[2*N+3,2*N+3], 0.4; atol=1e-14)
            @test isapprox(inputs.omega_raw[1,N+1],1.0; atol=1e-14)
            @test inputs.omega_raw[N+1,3] == 1.0
            @test_throws ArgumentError load_krusell_data(joinpath(@__DIR__,"nonexistent_data_directory"))
            bad = merge(data,(labor_college=fill(100.0,N_INDUSTRIES),))
            @test_throws ArgumentError prepare_aggregate_inputs(bad)
            bad_shape = merge(data, (va=ones(62),))
            @test_throws DimensionMismatch prepare_aggregate_inputs(bad_shape)
        end
        @testset "Leontief, factor shares and capital shock (synthetic)" begin
            result = compute_aggregate_model(_synthetic_data())
            @test result.equilibrium.leontief_residual < 1e-12
            @test isapprox(sum(result.equilibrium.lambda[end-K+1:end]),1.0; atol=1e-12)
            @test result.equilibrium.lambda[end] == 0.0
            @test result.mes.factor_ids == collect(FACTOR_IDS)
            @test result.mes.factor_shares[1] > 0
            @test result.mes.factor_shares[2] > 0
            @test result.mes.factor_shares[3] > 0
            @test result.second_order.symmetry_residual < 1e-11
            @test all(isapprox(result.mes_production[i,i],1.0; atol=1e-12)
                      for i in 1:3)
            @test all(isapprox(result.mes_cost[i,i],1.0; atol=1e-12)
                      for i in 1:3)
            @test isapprox(result.counterfactual.shock,-0.3701893579024938; atol=1e-14)
            @test result.mes_production_paper == transpose(result.mes_production)
            @test result.mes_cost_paper == transpose(result.mes_cost)
            @test isapprox(sum(result.capital_shock_production),
                           result.counterfactual.shock * sum(result.mes.nonelast[1,:]);
                           atol=1e-12)
            @test compute_aggregate_model(_synthetic_data()).mes_production ==
                  result.mes_production
        end
        @testset "Tensor covariance vs direct matrix identity (small)" begin
            rng = MersenneTwister(7171)
            n = 9
            W = rand(rng,n,n)
            W ./= sum(W;dims=2)
            P = rand(rng,n,n)
            a = randn(rng,n)
            tensor = weighted_covariance(W,P)
            literal = zeros(n,n)
            for k in 1:n
                literal .+= a[k] .* view(tensor,:,:,k)
            end
            efficient = aggregate_weighted_covariance(W,P,a)
            @test isapprox(literal,efficient; atol=2e-12,rtol=2e-12)
            @test maximum(abs,efficient-transpose(efficient)) < 2e-12
            @test iszero(maximum(abs, aggregate_weighted_covariance(W,P,zeros(n))))
        end
        if data_dir !== nothing
            @testset "Official MAT data and Tables B.1/B.5" begin
                data = load_krusell_data(data_dir)
                result = compute_aggregate_model(data)
                @test length(data.va) == N_INDUSTRIES
                @test size(data.io) == (83,98)
                @test isapprox(result.inputs.share_noncollege,
                               0.2439395152250744; atol=1e-12)
                @test isapprox(result.inputs.share_college_within_residual,
                               0.3948561528726668; atol=1e-12)
                @test isapprox(result.equilibrium.lambda[326],
                               0.4575253504176550; atol=1e-10)
                @test isapprox(result.equilibrium.lambda[327],
                               0.2439395152250744; atol=1e-10)
                @test isapprox(result.equilibrium.lambda[328],
                               0.2985351343572706; atol=1e-10)
                expected_prod = [
                    1.0 1.0507522513548324 0.67;
                    1.67 1.0 1.67;
                    0.67 0.8774696302073511 1.0
                ]
                expected_cost = [
                    1.0 1.275143847127333 0.67;
                    1.67 1.0 1.67;
                    0.67 1.0648561528726668 1.0
                ]
                @test isapprox(result.mes_production, expected_prod; atol=1e-8)
                @test isapprox(result.mes_cost, expected_cost; atol=1e-8)
                @test isapprox(result.mes.sub_hessian,
                    [-0.0143654439319198 -0.0747777031633236 0.0891431470952434;
                     -0.0747777031633236 0.1235701288516761 -0.0487924256883526;
                      0.0891431470952434 -0.0487924256883526 -0.0403507214068908];
                    atol=1e-8)
                # Independent per-active-row covariance accumulation WITHOUT
                # building the 329-by-329-by-329 MATLAB tensor.
                active = findall(x -> !iszero(x), result.second_order.coefficients)
                @test active == [133, 198, 263]
                h_reference = zeros(Float64, Q, Q)
                for k in active
                    h_reference .+= result.second_order.coefficients[k] .* 
                        weighted_covariance(view(result.equilibrium.omega, k, :),
                                            result.equilibrium.psi)
                end
                @test isapprox(h_reference, result.second_order.hessian;
                               atol=1e-10, rtol=1e-10)
                @test isapprox(result.capital_shock_production,
                               [0.0500707765474471,0.0679512458178448,
                                -0.1322612952552737]; atol=1e-8)
                @test isapprox(result.capital_shock_cost,
                               [0.0116232564169115,0.1134785805158008,
                                -0.1105392316909114]; atol=1e-8)
                @test all(round(result.capital_shock_production[i];digits=2) ==
                          TABLE_B5_AGGREGATE[i] for i in 1:3)
                issues = published_discrepancies(result)
                @test length(issues) == 1
                @test issues[1].table == "Cost"
                @test issues[1].row == "Noncollege"
                @test issues[1].column == "College"
            end
        end
    end
    return nothing
end

"""
    run_benchmark(data; samples=5)

Warm-started timing for 329-node Leontief solve, optimized Hessian and MES.
Prints median allocations and elapsed times for the current Julia environment.
The reference tensor would use Q^3*8 bytes even before coefficient arrays.
"""
function run_benchmark(data; samples::Int=5)
    samples >= 1 || throw(ArgumentError("samples must be positive"))
    inp = prepare_aggregate_inputs(data)
    eq = compute_aggregate_equilibrium(inp)
    sec = compute_aggregate_second_order(eq.omega,eq.psi,eq.lambda,
                                          inp.elasticities;
                                          sigma=inp.sigma,final_demand=inp.final_demand)
    compute_aggregate_mes(sec.hessian,eq.lambda)
    GC.gc()
    @printf("\nBenchmark (Julia after first compilation; %d samples)\n",samples)
    @printf("%-32s %12s %16s\n","Stage","median ms","median alloc B")
    tasks = (
        ("Leontief + Domar weights", ()->compute_aggregate_equilibrium(inp)),
        ("Optimized full Hessian", ()->compute_aggregate_second_order(
            eq.omega,eq.psi,eq.lambda,inp.elasticities;
            sigma=inp.sigma,final_demand=inp.final_demand)),
        ("MES (3 factors)", ()->compute_aggregate_mes(sec.hessian,eq.lambda)),
        ("Entire aggregate calculation", ()->compute_aggregate_model(data)),
    )
    for (name,f) in tasks
        f()  # warm up this concrete closure before timing or allocation measurement
        elapsed = Float64[]
        allocations = Int[]
        for _ in 1:samples
            GC.gc()
            push!(elapsed,@elapsed f())
            push!(allocations,@allocated f())
        end
        @printf("%-32s %12.3f %16d\n", name,1e3*median(elapsed),Int(median(allocations)))
    end
    @printf("MATLAB tensor theoretical storage (329^3*8): %d bytes (%.1f MiB)\n",
            Q^3*sizeof(Float64),Q^3*sizeof(Float64)/1024^2)
    return nothing
end

"""
    run_replication(data_dir; out_dir=joinpath(@__DIR__,"output_krusell_agg"),
                    tests=true, bench=false)

Execute original-data replication (never synthetic fallback), export all CSV
and LaTeX outputs, print published-table differences, and optionally benchmark.
"""
function run_replication(data_dir::AbstractString;
                         out_dir::AbstractString=joinpath(@__DIR__,"output_krusell_agg"),
                         tests::Bool=true,bench::Bool=false)
    data = load_krusell_data(data_dir)
    if tests
        run_tests(;data_dir=data_dir)
    end
    result = compute_aggregate_model(data)
    files = save_aggregate_outputs(result,out_dir)
    println("\nAggregate Krusell reproduction: raw MATLAB MES (rows/columns as coded)")
    display(result.mes_production)
    display(result.mes_cost)
    println("\nPaper-oriented MES (transpose of raw MATLAB):")
    display(result.mes_production_paper)
    display(result.mes_cost_paper)
    @printf("\nCapital shock dlog K = %.12f\n",result.counterfactual.shock)
    println("Production income-share changes: ",result.capital_shock_production)
    println("Cost income-share changes:       ",result.capital_shock_cost)
    println("Published B.1 ordinary-rounding exceptions: ", published_discrepancies(result))
    @printf("Leontief residual %.3e; Hessian symmetry residual %.3e\n",
            result.equilibrium.leontief_residual,
            result.second_order.symmetry_residual)
    for file in files
        isfile(file) && filesize(file) > 0 || error("Missing/empty output: $file")
        println("Wrote: ",abspath(file))
    end
    bench && run_benchmark(data)
    return result
end

function _cli(args::Vector{String})
    data_dir = nothing
    out_dir = joinpath(@__DIR__,"output_krusell_agg")
    test_only = false
    tests = true
    bench = false
    i = 1
    while i <= length(args)
        flag = args[i]
        if flag == "--data-dir" || flag == "--out-dir"
            i < length(args) || throw(ArgumentError("$flag requires a path"))
            if flag == "--data-dir"
                data_dir = args[i+1]
            else
                out_dir = args[i+1]
            end
            i += 2
        elseif flag == "--test-only"
            test_only = true
            i += 1
        elseif flag == "--no-tests"
            tests = false
            i += 1
        elseif flag == "--bench"
            bench = true
            i += 1
        else
            throw(ArgumentError("Unknown argument: $flag"))
        end
    end
    if test_only
        run_tests()
        return nothing
    end
    data_dir === nothing && throw(ArgumentError(
        "Provide --data-dir PATH containing the five original MAT files; " *
        "or run --test-only for synthetic tests"))
    return run_replication(data_dir;out_dir=out_dir,tests=tests,bench=bench)
end

end # module KrusellAggregate

# The macro invocation must be parenthesized before any trailing operator.
if abspath(PROGRAM_FILE) == (@__FILE__)
    KrusellAggregate._cli(ARGS)
end
