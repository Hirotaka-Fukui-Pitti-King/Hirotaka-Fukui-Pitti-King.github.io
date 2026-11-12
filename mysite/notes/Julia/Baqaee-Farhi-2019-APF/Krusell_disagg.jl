#=
    Krusell_disagg.jl

Faithful disaggregated-industry Krusell replication of Baqaee and Farhi (2019),
JEEA 17(5), Appendix B, Tables B.2 and B.5.

This standalone Julia file reuses the verified MATLAB MAT reader, the
Relabel_mod.jl nested-CES mapping, and WeightedCovMatrix.jl's memory-efficient
second-order operator THROUGH Krusell_agg.jl.  The aggregate script's
model-specific DATA CONSTRUCTION is never used for the disaggregated model.

The supplied MATLAB script and original MAT data give production MES
(College row, Capital column) = 0.66310442064, or 0.66 to two decimals;
published Table B.2 prints 0.67. That discrepancy is reported, never fixed by
altering the calibration. MATLAB raw MES matrices are transposed only for the
paper display, never for the internal shock calculation.

Dependencies: MAT.jl and SummaryTables.jl (from Krusell_agg.jl), plus Julia
standard libraries. Place Krusell_agg.jl, Relabel_mod.jl, and
WeightedCovMatrix.jl in the same directory. Do not include these companions
inside functions: Julia 1.12 world-age restrictions apply.

PowerShell examples, from the website root:

    julia --depwarn=error --project=. notes/Julia/Baqaee-Farhi-2019-APF/Krusell_disagg.jl --test-only
    julia --depwarn=error --project=. notes/Julia/Baqaee-Farhi-2019-APF/Krusell_disagg.jl --data-dir notes/Julia/Baqaee-Farhi-2019-APF/data --bench

`--out-dir PATH`, `--no-tests`, `--bench`, and `--test-only` are supported.
=#

# Include the aggregate file ONCE before defining this module. It in turn
# includes the two smaller companion modules. It does not execute the aggregate
# CLI, because PROGRAM_FILE is this disaggregated script.
include(joinpath(@__DIR__, "Krusell_agg.jl"))

"""
    KrusellDisaggregated

Original-data replication of the disaggregated nested-CES Krusell model in
Baqaee and Farhi (2019), using the verified Task III-V Julia modules.
"""
module KrusellDisaggregated

using ..KrusellAggregate: load_krusell_data, validate_data,
                          compute_aggregate_second_order,
                          compute_aggregate_mes,
                          compute_capital_counterfactual,
                          compute_aggregate_model
using ..NestedCESRelabelling: relabel_nested_ces, normalize_input_shares
using ..WeightedCovMatrix: weighted_covariance, aggregate_weighted_covariance
using LinearAlgebra
using SparseArrays
using Statistics
using Printf
using Random
using Test
using SummaryTables: Table, Cell

export load_disaggregated_data, clean_industry_io,
       build_disaggregated_blocks, prepare_disaggregated_inputs,
       compute_disaggregated_equilibrium, compute_disaggregated_model,
       compare_aggregate_disaggregated, published_discrepancies,
       save_disaggregated_outputs, run_tests, run_replication, run_benchmark

const N_INDUSTRIES = 63
const N = 65
const K = 4
const M = 2 * N + K
const Q = 5 * N + K
const FACTOR_IDS = (Q - K + 1):(Q - 1)  # 326:328, excludes terminal node 329
const FACTOR_NAMES = ("Capital", "Noncollege", "College")

const TABLE_B2_PRODUCTION = [
    NaN 1.43 0.70;
    0.94 NaN 0.94;
    0.67 1.59 NaN
]
const TABLE_B2_COST = [
    NaN 1.47 0.72;
    1.09 NaN 1.09;
    0.64 1.54 NaN
]
const TABLE_B5_DISAGGREGATED = [0.06, 0.04, -0.13]

# These are the independently calculated source-MATLAB references for the
# ORIGINAL five distribution files, not quantities to substitute for a model.
const REFERENCE_MES_PRODUCTION_RAW = [
    1.0 0.935519638417 0.663104420644;
    1.429784083397 1.0 1.593348585004;
    0.696252349208 0.937433338184 1.0
]
const REFERENCE_MES_COST_RAW = [
    1.0 1.090739328629 0.641727378145;
    1.468570139963 1.0 1.541982496411;
    0.715139734593 1.092970545927 1.0
]
const REFERENCE_FACTOR_SHARES = [0.453580444324, 0.251626082980, 0.294793472695, 0.0]
const REFERENCE_SHOCK_PRODUCTION = [0.061864362620, 0.036349191509, -0.126213343273]
const REFERENCE_SHOCK_COST = [0.030645773771, 0.064236507573, -0.101982938067]

"""
    load_disaggregated_data(data_dir)

Load all five original MAT datasets with the same checks as Task V. In this
model all five are used, including IO2013 and gross output. Never synthesize
missing original data for the official replication.
"""
load_disaggregated_data(data_dir::AbstractString) = load_krusell_data(data_dir)

"""
    clean_industry_io(io2013)

Match the complete ORIGINAL MATLAB IO table transformation in Krusell_disagg.m:
merge original-index rows (before columns), merge original-index columns
(including final demand aggregation into original column 75), then remove the
original-index rows and columns. The result is 63 industries x 64 columns,
where the last column is final consumption. Negative source entries are
PRESERVED here; clipping occurs later in the Matlab workflow.
"""
function clean_industry_io(io2013::AbstractMatrix{<:Real})
    size(io2013) == (83, 98) || throw(DimensionMismatch(
        "IO2013 must be 83x98, received $(size(io2013))"))
    all(isfinite, io2013) || throw(ArgumentError("IO2013 contains non-finite entries"))
    io = Matrix{Float64}(io2013)

    # ORIGINAL indices, before any deletion. Preserve MATLAB's operation order.
    io[48, :] .= io[48, :] .+ io[49, :]
    io[28, :] .= vec(sum(view(io, 28:31, :); dims=1))
    io[59, :] .= vec(sum(view(io, 59:60, :); dims=1))
    io[67, :] .= vec(sum(view(io, 67:69, :); dims=1))
    io[70, :] .= vec(sum(view(io, 70:71, :); dims=1))

    io[:, 48] .= io[:, 48] .+ io[:, 49]
    io[:, 28] .= vec(sum(view(io, :, 28:31); dims=2))
    io[:, 59] .= vec(sum(view(io, :, 59:60); dims=2))
    io[:, 67] .= vec(sum(view(io, :, 67:69); dims=2))
    io[:, 70] .= vec(sum(view(io, :, 70:71); dims=2))
    final_sources = vcat(collect(75:80), collect(83:94))
    io[:, 75] .= vec(sum(view(io, :, final_sources); dims=2))

    # Delete by ORIGINAL indices only after all additions. Retained indices
    # remain in original order (unlike repeatedly deleting at moving offsets).
    row_delete = Set(vcat(collect(29:31), [49, 60, 68, 69], collect(71:83)))
    col_delete = Set(vcat(collect(29:31), [49, 60, 68, 69],
                          collect(71:74), collect(76:98)))
    row_keep = [i for i in 1:83 if !(i in row_delete)]
    col_keep = [j for j in 1:98 if !(j in col_delete)]
    length(row_keep) == N_INDUSTRIES || error("IO row deletion error")
    length(col_keep) == N_INDUSTRIES + 1 || error("IO column deletion error")
    return io[row_keep, col_keep]
end

"""
    _matlab_row_normalize_signed(A)

Replicate `bsxfun(@rdivide,A,sum(A,2)); A(isnan(A))=0`, BEFORE the later
`max(A,0)` step. In particular, negative values are NOT clipped here.
All-zero rows remain zero. A nonzero row with an exactly zero algebraic sum
would produce Inf/NaN in MATLAB; reject such pathological input explicitly.
"""
function _matlab_row_normalize_signed(A::AbstractMatrix{<:Real})
    all(isfinite, A) || throw(ArgumentError("normalization input must be finite"))
    B = Matrix{Float64}(A)
    for i in axes(B, 1)
        row = view(B, i, :)
        total = sum(row)
        if iszero(total)
            all(iszero, row) || throw(ArgumentError(
                "nonzero row $i has zero algebraic sum; MATLAB would produce infinities"))
            continue
        end
        row ./= total
    end
    return B
end

"""
    validate_disaggregated_data(data)

Additional economic checks beyond the source MAT dimensions: positive output
and value added, nonnegative capital residual, and valid per-industry ratios.
Inputs are not modified. This makes previously silent MATLAB divide-by-zero
failures explicit without changing results for the official data.
"""
function validate_disaggregated_data(data::NamedTuple)
    validate_data(data)
    all(x -> x > 0.0, data.output) ||
        throw(ArgumentError("gross output must be positive in every industry"))
    all(x -> x > 0.0, data.va) ||
        throw(ArgumentError("value added must be positive in every industry"))
    all(data.va .<= data.output) ||
        throw(ArgumentError("value added must not exceed gross output"))
    all(data.labor_noncollege .+ data.labor_college .<= data.va) ||
        throw(ArgumentError("labor compensation exceeds value added in an industry"))
    return nothing
end

"""
    build_disaggregated_blocks(data, io_clean)

Construct the ORIGINAL 65x65 Omega and 195x134 Fshares (before relabelling),
with industry-specific value-added and labor-income shares. Replicate the
unclipped signed row normalizations in the source MATLAB, including zero rows.
Auxiliary nodes 1:2 have zero value-added/factor shares. The transient
MATLAB padded labor values 1,1 are skipped where they would be discarded
anyway; all entries of the final Fshares are unchanged.
"""
function build_disaggregated_blocks(data::NamedTuple,
                                    io_clean::AbstractMatrix{<:Real})
    validate_disaggregated_data(data)
    size(io_clean) == (N_INDUSTRIES, N_INDUSTRIES + 1) ||
        throw(DimensionMismatch("cleaned IO must be 63x64"))
    all(isfinite, io_clean) || throw(ArgumentError("cleaned IO has non-finite entries"))
    va = data.va
    gross_output = data.output
    labor_nc = data.labor_noncollege
    labor_c = data.labor_college
    residual = va .- labor_nc
    all(x -> x > 0.0, residual) ||
        throw(ArgumentError("VA - noncollege compensation must be positive"))

    alpha = vcat(zeros(2), va ./ gross_output)
    labor_share = vcat(zeros(2), labor_nc ./ va)
    other_share = vcat(zeros(2), 1.0 .- labor_nc ./ va)
    # Inner value-added CES nest: capital conditional on (capital+college).
    capital_within = (va .- labor_nc .- labor_c) ./ residual
    college_within = labor_c ./ residual

    Fshares = zeros(Float64, 3 * N, M)
    for i in 1:N
        Fshares[i, i] = labor_share[i]
        Fshares[i, N + i] = other_share[i]
    end
    # MATLAB Fshares(N+1:2N,2N+2)=1, including first two auxiliary nests.
    Fshares[(N + 1):(2 * N), 2 * N + 2] .= 1.0
    # MATLAB Fshares(2N+1:3N,2N+1) and (2N+3), followed by
    # Fshares(2N+1:2N+2,[2N+1,2N+3])=0.
    Fshares[(2 * N + 3):(3 * N), 2 * N + 1] .= capital_within
    Fshares[(2 * N + 3):(3 * N), 2 * N + 3] .= college_within
    Fshares = _matlab_row_normalize_signed(Fshares)
    # Original MATLAB repeats these two assignments after normalization.
    Fshares[1, 2 * N + 4] = 0.0
    Fshares[2, 2 * N + 1] = 0.0

    final_consumption = max.(0.0, vec(io_clean[:, end]))
    Omega = zeros(Float64, N, N)
    Omega[1, 3:N] .= final_consumption
    Omega[3:N, 3:N] .= transpose(view(io_clean, :, 1:N_INDUSTRIES))
    Omega = _matlab_row_normalize_signed(Omega)

    epsilon = vcat(ones(2), fill(0.5, N_INDUSTRIES))
    theta = vcat([0.9, 1.0], fill(0.0001, N_INDUSTRIES))
    eta = vcat(ones(2), fill(1.67, N_INDUSTRIES),
               fill(0.67, N), fill(0.67, N))
    sigma = 0.9
    final_demand = zeros(Float64, Q)
    final_demand[1] = 1.0
    return (omega_original=Omega, fshares=Fshares, alpha=alpha,
            epsilon=epsilon, theta=theta, eta=eta, sigma=sigma,
            final_demand=final_demand, final_consumption=final_consumption,
            labor_share=labor_share, other_share=other_share,
            capital_within=capital_within, college_within=college_within)
end

"""
    prepare_disaggregated_inputs(data; sparse=true)

Perform IO cleaning and build the original, signed-normalized input blocks,
then call the VERIFIED `relabel_nested_ces` API to expand to 329 nodes.
Preserve source negative entries through these steps; clipping and subsequent
row normalization belong to the caller equilibrium routine.
"""
function prepare_disaggregated_inputs(data::NamedTuple; sparse::Bool=true)
    cleaned = clean_industry_io(data.io)
    blocks = build_disaggregated_blocks(data, cleaned)
    relabelled = relabel_nested_ces(
        sparse ? SparseArrays.sparse(blocks.omega_original) : blocks.omega_original,
        sparse ? SparseArrays.sparse(blocks.fshares) : blocks.fshares,
        blocks.alpha, blocks.epsilon, blocks.theta, blocks.eta,
        blocks.final_demand; K=K, sparse=sparse)
    return merge(blocks,
                 (io_clean=cleaned, omega_raw=relabelled.omega,
                  elasticities=relabelled.elasticities,
                  final_demand=relabelled.final_demand))
end

"""
    compute_disaggregated_equilibrium(inputs)

Match MATLAB's POST-relabelling `max(Omega_tr,0)`, row normalization,
Leontief inverse, and Domar weights. Normalize via Relabel_mod.jl's verified
function, then use LU to solve the inverse linear system. The normalized
Omega is dense as in Krusell_agg.jl; the pre-normalization network is sparse.
"""
function compute_disaggregated_equilibrium(inputs::NamedTuple)
    omega = normalize_input_shares(inputs.omega_raw)
    size(omega) == (Q, Q) || throw(DimensionMismatch("relabelled omega must be 329x329"))
    id = Matrix{Float64}(I, Q, Q)
    A = id - omega
    fac = lu(A)
    psi = fac \ id
    lambda = transpose(psi) * inputs.final_demand
    all(isfinite, psi) || throw(ArgumentError("Leontief inverse is not finite"))
    all(isfinite, lambda) || throw(ArgumentError("Domar weights are not finite"))
    resid = maximum(abs, A * psi - id)
    return (omega=omega, psi=psi, lambda=lambda,
            leontief_residual=resid, matrix_condition=cond(A),
            factor_ids=collect(FACTOR_IDS))
end

"""
    compute_disaggregated_model(data; sparse=true)

Calculate the disaggregated 329-node model using shared aggregate-program
second-order, MES and capital-shock OPERATORS. Only the IO/factor-share
construction is disaggregated-specific. Results retain the MATLAB raw MES
orientation and separately provide their transposes for paper display.
"""
function compute_disaggregated_model(data::NamedTuple; sparse::Bool=true)
    inputs = prepare_disaggregated_inputs(data; sparse=sparse)
    eq = compute_disaggregated_equilibrium(inputs)
    second = compute_aggregate_second_order(
        eq.omega, eq.psi, eq.lambda, inputs.elasticities;
        sigma=inputs.sigma, final_demand=inputs.final_demand)
    mes = compute_aggregate_mes(second.hessian, eq.lambda; factor_ids=FACTOR_IDS)
    cf = compute_capital_counterfactual(mes.nonelast, mes.elast)
    return (inputs=inputs, equilibrium=eq, second_order=second, mes=mes,
            counterfactual=cf, omega=eq.omega, psi=eq.psi,
            lambda=eq.lambda, hessian=second.hessian,
            elasticities=inputs.elasticities,
            mes_production=mes.mes_production, mes_cost=mes.mes_cost,
            mes_production_paper=mes.mes_production_paper,
            mes_cost_paper=mes.mes_cost_paper,
            capital_shock_production=cf.capital_shock_production,
            capital_shock_cost=cf.capital_shock_cost)
end

"""
    compare_aggregate_disaggregated(aggregate, disaggregated)

Compare original calibrated benchmarks, NOT a causal decomposition holding
all other primitives fixed. Return the publication-oriented MES and factor
income-share responses and their disaggregated-minus-aggregate differences.
"""
function compare_aggregate_disaggregated(aggregate, disaggregated)
    return (
        production=(aggregate=aggregate.mes_production_paper,
                    disaggregated=disaggregated.mes_production_paper,
                    delta=disaggregated.mes_production_paper .- aggregate.mes_production_paper),
        cost=(aggregate=aggregate.mes_cost_paper,
              disaggregated=disaggregated.mes_cost_paper,
              delta=disaggregated.mes_cost_paper .- aggregate.mes_cost_paper),
        shock_production=(aggregate=aggregate.capital_shock_production,
                          disaggregated=disaggregated.capital_shock_production,
                          delta=disaggregated.capital_shock_production .-
                                aggregate.capital_shock_production),
        shock_cost=(aggregate=aggregate.capital_shock_cost,
                    disaggregated=disaggregated.capital_shock_cost,
                    delta=disaggregated.capital_shock_cost .-
                          aggregate.capital_shock_cost),
    )
end

"""
    published_discrepancies(result)

Report every Table B.2 off-diagonal entry whose ordinary two-decimal
rounding differs from the JEEA printed cell. The expected one-cell exception
is Production MES, College row / Capital column: 0.6631044 vs printed 0.67.
"""
function published_discrepancies(result)
    issues = NamedTuple[]
    for (name, computed, target) in
        (("Production", result.mes_production_paper, TABLE_B2_PRODUCTION),
         ("Cost", result.mes_cost_paper, TABLE_B2_COST))
        for i in 1:3, j in 1:3
            if i != j && round(computed[i, j]; digits=2) != target[i, j]
                push!(issues, (table=name, row=FACTOR_NAMES[i],
                               column=FACTOR_NAMES[j], computed=computed[i, j],
                               published=target[i, j],
                               raw_gap=computed[i, j] - target[i, j]))
            end
        end
    end
    return issues
end

function _matrix_csv(path::AbstractString, values::AbstractMatrix;
                     orientation::AbstractString)
    size(values) == (3, 3) || throw(DimensionMismatch("MES matrix must be 3x3"))
    open(path, "w") do io
        println(io, "row_orientation_", orientation, ",Capital,Noncollege,College")
        for i in 1:3
            print(io, FACTOR_NAMES[i])
            for j in 1:3
                @printf(io, ",%.17g", values[i, j])
            end
            println(io)
        end
    end
    return path
end

function _diagnostics_csv(path::AbstractString, result, comparison)
    inp = result.inputs
    eq = result.equilibrium
    second = result.second_order
    issues = published_discrepancies(result)
    metric = [
        ("source_io_rows", 83.0),
        ("source_io_columns", 98.0),
        ("cleaned_io_rows", Float64(size(inp.io_clean, 1))),
        ("cleaned_io_columns", Float64(size(inp.io_clean, 2))),
        ("original_nodes", Float64(N)),
        ("relabelled_nodes", Float64(Q)),
        ("fshares_rows", Float64(size(inp.fshares, 1))),
        ("fshares_columns", Float64(size(inp.fshares, 2))),
        ("raw_omega_nnz", Float64(nnz(inp.omega_raw))),
        ("normalized_omega_nnz", Float64(count(x -> !iszero(x), eq.omega))),
        ("normalized_omega_density", count(x -> !iszero(x), eq.omega) / length(eq.omega)),
        ("cleaned_io_negative_entries", Float64(count(x -> x < 0.0, inp.io_clean))),
        ("original_omega_negative_entries", Float64(count(x -> x < 0.0, inp.omega_original))),
        ("final_consumption_total", sum(inp.final_consumption)),
        ("sum_final_demand_shares", sum(inp.final_demand)),
        ("factor_capital_domars", eq.lambda[326]),
        ("factor_noncollege_domars", eq.lambda[327]),
        ("factor_college_domars", eq.lambda[328]),
        ("terminal_factor_329_domars", eq.lambda[329]),
        ("sum_terminal_factor_shares", sum(eq.lambda[326:329])),
        ("leontief_residual_max", eq.leontief_residual),
        ("leontief_condition_2", eq.matrix_condition),
        ("hessian_symmetry_residual", second.symmetry_residual),
        ("final_demand_covariance_max", maximum(abs, weighted_covariance(inp.final_demand, eq.psi))),
        ("capital_shock_dlogK", result.counterfactual.shock),
        ("published_B2_mismatch_count", Float64(length(issues))),
        ("paper_B2_production_College_Capital", TABLE_B2_PRODUCTION[3, 1]),
        ("computed_B2_production_College_Capital", result.mes_production_paper[3, 1]),
        ("aggregate_minus_disaggregate_noncollege_shock",
            -comparison.shock_production.delta[2]),
        ("agg_college_minus_noncollege_shock",
            comparison.shock_production.aggregate[3] -
            comparison.shock_production.aggregate[2]),
        ("disagg_college_minus_noncollege_shock",
            comparison.shock_production.disaggregated[3] -
            comparison.shock_production.disaggregated[2]),
        ("absolute_relative_labor_response_reduction_fraction",
            1.0 - abs((comparison.shock_production.disaggregated[3] -
                       comparison.shock_production.disaggregated[2]) /
                      (comparison.shock_production.aggregate[3] -
                       comparison.shock_production.aggregate[2]))),
    ]
    open(path, "w") do io
        println(io, "metric,value")
        for (label, val) in metric
            @printf(io, "%s,%.17g\n", label, val)
        end
    end
    return path
end

function _shock_csv(path::AbstractString, result)
    open(path, "w") do io
        println(io, "factor,production_income_share_change,cost_income_share_change")
        for i in 1:3
            @printf(io, "%s,%.17g,%.17g\n", FACTOR_NAMES[i],
                    result.capital_shock_production[i], result.capital_shock_cost[i])
        end
    end
    return path
end

function _comparison_csv(path::AbstractString, comp)
    open(path, "w") do io
        println(io, "object,row_factor,column_factor,aggregate,disaggregated,disaggregated_minus_aggregate")
        for (name, sub) in (("production_mes_paper", comp.production),
                            ("cost_mes_paper", comp.cost))
            for i in 1:3, j in 1:3
                i == j && continue
                @printf(io, "%s,%s,%s,%.17g,%.17g,%.17g\n",
                        name, FACTOR_NAMES[i], FACTOR_NAMES[j],
                        sub.aggregate[i, j], sub.disaggregated[i, j], sub.delta[i, j])
            end
        end
        for (name, sub) in (("capital_shock_production", comp.shock_production),
                            ("capital_shock_cost", comp.shock_cost))
            for i in 1:3
                @printf(io, "%s,%s,,%.17g,%.17g,%.17g\n", name, FACTOR_NAMES[i],
                        sub.aggregate[i], sub.disaggregated[i], sub.delta[i])
            end
        end
        # With both labor quantities held fixed in the capital counterfactual,
        # college-minus-noncollege log share responses equal the first-order
        # log skill-premium response (not a mechanism decomposition).
        ag = comp.shock_production.aggregate[3] - comp.shock_production.aggregate[2]
        dis = comp.shock_production.disaggregated[3] -
              comp.shock_production.disaggregated[2]
        @printf(io, "%s,%s,,%.17g,%.17g,%.17g\n",
                "college_minus_noncollege_log_share_response", "College/Noncollege",
                ag, dis, dis - ag)
    end
    return path
end

function _mes_table_data(values::AbstractMatrix)
    data = Matrix{String}(undef, 4, 4)
    data[1, :] = ["Factor", "Capital", "Noncollege", "College"]
    for i in 1:3
        data[i + 1, 1] = FACTOR_NAMES[i]
        for j in 1:3
            data[i + 1, j + 1] = i == j ? "-" : @sprintf("%.2f", values[i, j])
        end
    end
    return data
end

function _summarytable_tex(path::AbstractString, values::Matrix{String})
    tab = Table(Cell.(values); header=1)
    open(path, "w") do io
        show(io, MIME"text/latex"(), tab)
        println(io)
    end
    return path
end

"""
    save_disaggregated_outputs(result, aggregate, out_dir)

Write four MES CSVs (two raw MATLAB, two transposed paper display), capital
shocks, all diagnostics, Aggregate-vs-Disaggregated numerical comparisons,
three Table B.2/B.5 LaTeX tables and one comparison table. The LaTeX is
produced via SummaryTables.jl's Table/Cell API, not handwritten formatting.
"""
function save_disaggregated_outputs(result, aggregate, out_dir::AbstractString)
    mkpath(out_dir)
    outputs = String[]
    push!(outputs, _matrix_csv(joinpath(out_dir, "krusell_disagg_mes_production.csv"),
                               result.mes_production; orientation="raw_matlab"))
    push!(outputs, _matrix_csv(joinpath(out_dir, "krusell_disagg_mes_cost.csv"),
                               result.mes_cost; orientation="raw_matlab"))
    push!(outputs, _matrix_csv(joinpath(out_dir, "krusell_disagg_mes_production_paper.csv"),
                               result.mes_production_paper; orientation="paper_display"))
    push!(outputs, _matrix_csv(joinpath(out_dir, "krusell_disagg_mes_cost_paper.csv"),
                               result.mes_cost_paper; orientation="paper_display"))
    comp = compare_aggregate_disaggregated(aggregate, result)
    push!(outputs, _shock_csv(joinpath(out_dir, "krusell_disagg_capital_shock.csv"), result))
    push!(outputs, _diagnostics_csv(joinpath(out_dir, "krusell_disagg_diagnostics.csv"), result, comp))
    push!(outputs, _comparison_csv(joinpath(out_dir, "krusell_agg_vs_disagg.csv"), comp))
    push!(outputs, _summarytable_tex(joinpath(out_dir, "krusell_disagg_table_B2_production.tex"),
                                     _mes_table_data(result.mes_production_paper)))
    push!(outputs, _summarytable_tex(joinpath(out_dir, "krusell_disagg_table_B2_cost.tex"),
                                     _mes_table_data(result.mes_cost_paper)))
    b5 = Matrix{String}(undef, 4, 2)
    b5[1, :] = ["Factor", "Disaggregated model"]
    for i in 1:3
        b5[i + 1, 1] = FACTOR_NAMES[i]
        b5[i + 1, 2] = @sprintf("%.2f", result.capital_shock_production[i])
    end
    push!(outputs, _summarytable_tex(joinpath(out_dir, "krusell_disagg_table_B5.tex"), b5))

    compare_rows = Matrix{String}(undef, 7, 5)
    compare_rows[1, :] = ["Object", "Factor / pair", "Aggregate", "Disaggregated", "Difference"]
    lines = (
        ("Production MES", "Capital / Noncollege", comp.production, 1, 2),
        ("Production MES", "Capital / College", comp.production, 1, 3),
        ("Production MES", "Noncollege / College", comp.production, 2, 3),
    )
    for (i, (label, pair, sub, r, c)) in enumerate(lines)
        compare_rows[i + 1, :] = [label, pair,
                                  @sprintf("%.3f", sub.aggregate[r, c]),
                                  @sprintf("%.3f", sub.disaggregated[r, c]),
                                  @sprintf("%+.3f", sub.delta[r, c])]
    end
    for i in 1:3
        sub = comp.shock_production
        compare_rows[i + 4, :] = ["Capital shock", FACTOR_NAMES[i],
                                  @sprintf("%+.3f", sub.aggregate[i]),
                                  @sprintf("%+.3f", sub.disaggregated[i]),
                                  @sprintf("%+.3f", sub.delta[i])]
    end
    push!(outputs, _summarytable_tex(joinpath(out_dir, "krusell_agg_vs_disagg.tex"),
                                     compare_rows))
    return outputs
end

# Independent elementwise group-summation reference for IO merging, including
# row-column group intersections. This differs algorithmically from the actual
# in-place cleaning, and is used solely in tests.
function _reference_io_by_groups(raw::AbstractMatrix)
    row_delete = Set(vcat(collect(29:31), [49, 60, 68, 69], collect(71:83)))
    col_delete = Set(vcat(collect(29:31), [49, 60, 68, 69],
                          collect(71:74), collect(76:98)))
    keep_r = [i for i in 1:83 if !(i in row_delete)]
    keep_c = [j for j in 1:98 if !(j in col_delete)]
    function row_group(i)
        if i == 48
            return [48, 49]
        elseif i == 28
            return collect(28:31)
        elseif i == 59
            return [59, 60]
        elseif i == 67
            return [67, 68, 69]
        elseif i == 70
            return [70, 71]
        end
        return [i]
    end
    function col_group(j)
        if j == 48
            return [48, 49]
        elseif j == 28
            return collect(28:31)
        elseif j == 59
            return [59, 60]
        elseif j == 67
            return [67, 68, 69]
        elseif j == 70
            return [70, 71]
        elseif j == 75
            return vcat(collect(75:80), collect(83:94))
        end
        return [j]
    end
    reference = zeros(Float64, N_INDUSTRIES, N_INDUSTRIES + 1)
    for (ii, i) in enumerate(keep_r), (jj, j) in enumerate(keep_c)
        reference[ii, jj] = sum(raw[r, c] for r in row_group(i) for c in col_group(j))
    end
    return reference
end

function _synthetic_data(seed::Int=20261112)
    rng = MersenneTwister(seed)
    io = zeros(Float64, 83, 98)
    for j in 1:70, i in 1:70
        io[i, j] = rand(rng) * 20.0
    end
    for i in 1:83
        io[i, 75] = 40.0 + rand(rng)
    end
    # Introduce one signed entry that survives to the clean IO table;
    # do not clip it at load or pre-relabelling stages.
    io[5, 10] = -0.25
    return (io=io, output=fill(100.0, N_INDUSTRIES),
            va=fill(50.0, N_INDUSTRIES),
            labor_noncollege=fill(10.0, N_INDUSTRIES),
            labor_college=fill(8.0, N_INDUSTRIES))
end

"""
    run_tests(; data_dir=nothing)

Synthetic tests require no data. When data_dir is supplied, also read the
original five MAT files and check the reference MES, Domar weights, shocks,
published discrepancy and aggregate/disaggregated comparison. Test the
3-factor Hessian using independent weighted moment loops, not a 329^3 tensor.
"""
function run_tests(; data_dir::Union{Nothing,AbstractString}=nothing)
    @testset "Krusell_disagg.jl" begin
        @testset "IO cleaning: source indices and group-sum reference" begin
            X = reshape(collect(1.0:(83 * 98)), 83, 98)
            cleaned = clean_industry_io(X)
            @test size(cleaned) == (63, 64)
            @test cleaned == _reference_io_by_groups(X)
            @test size(clean_industry_io(zeros(83, 98))) == (63, 64)
            @test all(iszero, clean_industry_io(zeros(83, 98)))
            @test_throws DimensionMismatch clean_industry_io(zeros(83, 97))
            X[1, 1] = NaN
            @test_throws ArgumentError clean_industry_io(X)
        end
        @testset "Signed row normalization and negative-entry ordering" begin
            A = [2.0 -1.0 0.0; 0.0 0.0 0.0; 0.5 1.0 0.5]
            W = _matlab_row_normalize_signed(A)
            @test W[1, :] == [2.0, -1.0, 0.0]
            @test all(iszero, W[2, :])
            @test W[3, :] == [0.25, 0.5, 0.25]
            post = normalize_input_shares(W)
            @test post[1, :] == [1.0, 0.0, 0.0]
            @test all(iszero, post[2, :])
            @test_throws ArgumentError _matlab_row_normalize_signed([1.0 -1.0])
            @test_throws ArgumentError _matlab_row_normalize_signed([NaN 1.0])
        end
        @testset "Synthetic industry shares and 329-node integration" begin
            data = _synthetic_data()
            clean = clean_industry_io(data.io)
            inp = build_disaggregated_blocks(data, clean)
            @test size(inp.omega_original) == (N, N)
            @test size(inp.fshares) == (3 * N, M)
            @test inp.alpha[1:2] == [0.0, 0.0]
            @test all(isapprox.(inp.alpha[3:end], 0.5; atol=1e-14))
            @test all(isapprox.(inp.labor_share[3:end], 0.2; atol=1e-14))
            @test all(isapprox.(inp.other_share[3:end], 0.8; atol=1e-14))
            @test all(isapprox.(inp.capital_within, 0.8; atol=1e-14))
            @test all(isapprox.(inp.college_within, 0.2; atol=1e-14))
            @test all(iszero, inp.fshares[1:2, :])
            @test all(iszero, inp.fshares[(2*N+1):(2*N+2), :])
            @test all(iszero, inp.omega_original[2, :])
            @test isapprox(sum(inp.omega_original[1, :]), 1.0; atol=1e-12)
            @test inp.omega_original[1, 3] > 0.0
            @test inp.omega_original[3, 3 + 1] == clean[2, 1] / sum(clean[:, 1])
            @test inp.epsilon == vcat(ones(2), fill(0.5, N_INDUSTRIES))
            @test inp.theta == vcat([0.9, 1.0], fill(0.0001, N_INDUSTRIES))
            @test length(inp.eta) == 3 * N
            @test inp.sigma == 0.9
            @test inp.final_demand[1] == 1.0
            @test sum(inp.final_demand) == 1.0
            inp_sparse = prepare_disaggregated_inputs(data; sparse=true)
            inp_dense = prepare_disaggregated_inputs(data; sparse=false)
            @test size(inp_sparse.omega_raw) == (Q, Q)
            @test size(inp_dense.omega_raw) == (Q, Q)
            @test Matrix(inp_sparse.omega_raw) == inp_dense.omega_raw
            @test inp_sparse.elasticities == inp_dense.elasticities
            @test isa(inp_sparse.omega_raw, SparseMatrixCSC)
            eq = compute_disaggregated_equilibrium(inp_sparse)
            @test size(eq.psi) == (Q, Q)
            @test eq.leontief_residual < 1e-10
            @test eq.lambda[329] == 0.0
            @test isapprox(sum(eq.lambda[326:329]), 1.0; atol=1e-10)
            @test all(x -> x > 0.0, eq.lambda[326:328])
            @test maximum(abs, eq.omega - normalize_input_shares(inp_dense.omega_raw)) == 0.0
            @test_throws DimensionMismatch build_disaggregated_blocks(data, zeros(63, 63))
            bad = merge(data, (output=fill(0.0, N_INDUSTRIES),))
            @test_throws ArgumentError build_disaggregated_blocks(bad, clean)
        end
        @testset "Covariance, Hessian, MES and shocks" begin
            rng = MersenneTwister(712)
            n = 8
            W = rand(rng, n, n)
            W ./= sum(W; dims=2)
            P = rand(rng, n, n)
            a = randn(rng, n)
            tensor = weighted_covariance(W, P)
            ref = zeros(n, n)
            for k in 1:n
                ref .+= a[k] .* view(tensor, :, :, k)
            end
            @test isapprox(ref, aggregate_weighted_covariance(W, P, a);
                           atol=1e-12, rtol=1e-12)
            @test maximum(abs, weighted_covariance([1.0; zeros(n-1)], P)) < 1e-12
            data = _synthetic_data()
            result = compute_disaggregated_model(data)
            @test length(result.lambda) == Q
            @test length(result.elasticities) == Q
            @test size(result.hessian) == (Q, Q)
            @test result.second_order.symmetry_residual < 1e-10
            @test maximum(abs, weighted_covariance(result.inputs.final_demand,
                                                   result.psi)) < 1e-12
            @test result.mes_production_paper == Matrix(transpose(result.mes_production))
            @test result.mes_cost_paper == Matrix(transpose(result.mes_cost))
            @test size(result.mes_production) == (3, 3)
            @test all(isapprox.(diag(result.mes_production), 1.0; atol=1e-12))
            @test all(isapprox.(diag(result.mes_cost), 1.0; atol=1e-12))
            @test isapprox(result.counterfactual.shock, -0.3701893579024938; atol=1e-13)
            @test isapprox(result.capital_shock_production,
                           result.counterfactual.shock .* vec(result.mes.nonelast[1, :]);
                           atol=1e-12)
            @test compute_disaggregated_model(data).mes_production == result.mes_production
        end
        if data_dir !== nothing
            @testset "Official MAT data: Tables B.2/B.5 and factor shares" begin
                data = load_disaggregated_data(data_dir)
                res = compute_disaggregated_model(data)
                agg = compute_aggregate_model(data)
                comp = compare_aggregate_disaggregated(agg, res)
                clean = res.inputs.io_clean
                @test size(data.io) == (83, 98)
                @test size(clean) == (63, 64)
                @test clean == _reference_io_by_groups(data.io)
                @test count(x -> x < 0.0, clean) == 2
                @test count(x -> x < 0.0, res.inputs.omega_original) == 1
                @test size(res.inputs.fshares) == (195, 134)
                @test size(res.omega) == (329, 329)
                @test nnz(res.inputs.omega_raw) == 3785
                @test count(x -> !iszero(x), res.omega) == 3784
                @test res.equilibrium.leontief_residual < 1e-10
                @test res.second_order.symmetry_residual < 1e-10
                @test res.lambda[329] == 0.0
                @test isapprox(res.lambda[326:329], REFERENCE_FACTOR_SHARES; atol=1e-8)
                @test isapprox(sum(res.lambda[326:329]), 1.0; atol=1e-10)
                @test isapprox(res.mes_production, REFERENCE_MES_PRODUCTION_RAW; atol=1e-8)
                @test isapprox(res.mes_cost, REFERENCE_MES_COST_RAW; atol=1e-8)
                @test isapprox(res.capital_shock_production,
                               REFERENCE_SHOCK_PRODUCTION; atol=1e-8)
                @test isapprox(res.capital_shock_cost,
                               REFERENCE_SHOCK_COST; atol=1e-8)
                @test all(round(res.capital_shock_production[i]; digits=2) ==
                          TABLE_B5_DISAGGREGATED[i] for i in 1:3)
                issues = published_discrepancies(res)
                @test length(issues) == 1
                @test issues[1].table == "Production"
                @test issues[1].row == "College"
                @test issues[1].column == "Capital"
                @test isapprox(res.mes_production_paper[3, 1],
                               0.663104420644; atol=1e-8)
                @test isapprox(comp.shock_production.delta[2],
                               -0.031602054309; atol=1e-8)
                skill_agg = comp.shock_production.aggregate[3] -
                            comp.shock_production.aggregate[2]
                skill_dis = comp.shock_production.disaggregated[3] -
                            comp.shock_production.disaggregated[2]
                @test isapprox(1.0 - abs(skill_dis / skill_agg),
                               0.18805019; atol=1e-7)
                # Independent full-economy sub-Hessian via 3x3 moment sums:
                # avoids allocating the original ~272 MiB covariance tensor.
                P = view(res.psi, :, FACTOR_IDS)
                Href = zeros(3, 3)
                for k in eachindex(res.second_order.coefficients)
                    a_k = res.second_order.coefficients[k]
                    iszero(a_k) && continue
                    weights = view(res.omega, k, :)
                    means = vec(transpose(weights) * P)
                    for j in 1:3, i in 1:3
                        moment = sum(weights[l] * P[l, i] * P[l, j] for l in 1:Q)
                        Href[i, j] += a_k * (moment - means[i] * means[j])
                    end
                end
                @test isapprox(Href, res.mes.sub_hessian; atol=1e-9, rtol=1e-9)
                @test maximum(abs, weighted_covariance(res.inputs.final_demand,
                                                       res.psi)) < 1e-12
            end
        end
    end
    return nothing
end

"""
    run_benchmark(data, data_dir; samples=5)

Warm-start each stage and report time/allocations for MAT loading, cleaning,
input construction, sparse relabelling, Leontief solve, Hessian, MES, and
complete run. Timings vary with hardware and GC; no extrapolated Julia
performance claims are made.
"""
function run_benchmark(data::NamedTuple, data_dir::AbstractString; samples::Int=5)
    samples >= 1 || throw(ArgumentError("samples must be positive"))
    cleaned = clean_industry_io(data.io)
    blocks = build_disaggregated_blocks(data, cleaned)
    inputs = prepare_disaggregated_inputs(data)
    eq = compute_disaggregated_equilibrium(inputs)
    second = compute_aggregate_second_order(eq.omega, eq.psi, eq.lambda,
                                            inputs.elasticities;
                                            sigma=inputs.sigma,
                                            final_demand=inputs.final_demand)
    compute_aggregate_mes(second.hessian, eq.lambda)
    GC.gc()
    @printf("\nBenchmark (Julia, compiled, %d samples)\n", samples)
    @printf("%-34s %12s %15s\n", "Stage", "median ms", "median alloc B")
    tasks = (
        ("MAT loading (5 files)", () -> load_disaggregated_data(data_dir)),
        ("IO cleaning only", () -> clean_industry_io(data.io)),
        ("Factor shares + IO blocks", () -> build_disaggregated_blocks(data, cleaned)),
        ("Relabelling (sparse)", () -> relabel_nested_ces(
            SparseArrays.sparse(blocks.omega_original),
            SparseArrays.sparse(blocks.fshares), blocks.alpha, blocks.epsilon,
            blocks.theta, blocks.eta, blocks.final_demand; K=K, sparse=true)),
        ("Leontief + Domar weights", () -> compute_disaggregated_equilibrium(inputs)),
        ("Full Hessian (matrix operator)", () -> compute_aggregate_second_order(
            eq.omega, eq.psi, eq.lambda, inputs.elasticities;
            sigma=inputs.sigma, final_demand=inputs.final_demand)),
        ("MES (3 factors)", () -> compute_aggregate_mes(second.hessian, eq.lambda)),
        ("Full model (without MAT loading)", () -> compute_disaggregated_model(data)),
    )
    for (name, f) in tasks
        f()  # warm-up concrete closure, including any new generic instantiation
        times = Float64[]
        allocations = Int[]
        for _ in 1:samples
            GC.gc()
            push!(times, @elapsed f())
            push!(allocations, @allocated f())
        end
        @printf("%-34s %12.3f %15d\n", name, 1000 * median(times), Int(median(allocations)))
    end
    @printf("Network: %d x %d, input nnz=%d, normalized nnz=%d (density %.4f)\n",
            Q, Q, nnz(inputs.omega_raw), count(x -> !iszero(x), eq.omega),
            count(x -> !iszero(x), eq.omega) / (Q * Q))
    @printf("Dense original covariance tensor storage: %d bytes (%.1f MiB)\n",
            Q^3 * sizeof(Float64), Q^3 * sizeof(Float64) / 1024^2)
    return nothing
end

"""
    run_replication(data_dir; out_dir=..., tests=true, bench=false)

Strict original-data replication with diagnostic and publication outputs.
Aggregate comparison is computed from the same original MAT files using the
already verified aggregate module, without altering either source module.
"""
function run_replication(data_dir::AbstractString;
                         out_dir::AbstractString=joinpath(@__DIR__, "output_krusell_disagg"),
                         tests::Bool=true, bench::Bool=false)
    data = load_disaggregated_data(data_dir)
    tests && run_tests(; data_dir=data_dir)
    result = compute_disaggregated_model(data)
    aggregate = compute_aggregate_model(data)
    comp = compare_aggregate_disaggregated(aggregate, result)
    files = save_disaggregated_outputs(result, aggregate, out_dir)

    println("\nDisaggregated Krusell replication: raw MATLAB MES")
    display(result.mes_production)
    display(result.mes_cost)
    println("\nPaper display (transposed MES):")
    display(result.mes_production_paper)
    display(result.mes_cost_paper)
    @printf("\nCapital shock dlog K = %.12f\n", result.counterfactual.shock)
    println("Production income-share changes: ", result.capital_shock_production)
    println("Cost income-share changes:       ", result.capital_shock_cost)
    println("Table B.2 printed-cell discrepancies: ", published_discrepancies(result))
    @printf("Leontief residual %.3e; Hessian symmetry residual %.3e\n",
            result.equilibrium.leontief_residual,
            result.second_order.symmetry_residual)
    @printf("Disaggregated - Aggregate Noncollege shock: %.12f\n",
            comp.shock_production.delta[2])
    for path in files
        isfile(path) && filesize(path) > 0 || error("Missing/empty result file: $path")
        println("Wrote: ", abspath(path))
    end
    bench && run_benchmark(data, data_dir)
    return result
end

function _cli(args::Vector{String})
    data_dir = nothing
    out_dir = joinpath(@__DIR__, "output_krusell_disagg")
    test_only = false
    tests = true
    bench = false
    i = 1
    while i <= length(args)
        flag = args[i]
        if flag == "--data-dir" || flag == "--out-dir"
            i < length(args) || throw(ArgumentError("$flag requires a path"))
            if flag == "--data-dir"
                data_dir = args[i + 1]
            else
                out_dir = args[i + 1]
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
            throw(ArgumentError("Unknown CLI argument: $flag"))
        end
    end
    if test_only
        run_tests()
        return nothing
    end
    data_dir === nothing && throw(ArgumentError(
        "Pass --data-dir PATH containing the five original MAT files, or --test-only"))
    return run_replication(data_dir; out_dir=out_dir, tests=tests, bench=bench)
end

end # module KrusellDisaggregated

if abspath(PROGRAM_FILE) == (@__FILE__)
    KrusellDisaggregated._cli(ARGS)
end
