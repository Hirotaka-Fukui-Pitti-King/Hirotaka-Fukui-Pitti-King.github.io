"""
    NestedCESRelabelling

Exact translation of the network *mapping* in `Relabel_mod.m` from the official
replication files for Baqaee and Farhi (2019). The public `relabel_nested_ces`
function uses a typed NamedTuple result, while `relabel_mod` preserves the
original MATLAB argument and tuple-return conventions.

This file uses only Julia standard libraries. Run it directly for verification
and timing measurements; include it from other code for the two public APIs.
"""
module NestedCESRelabelling

using LinearAlgebra
using SparseArrays
using Random
using Statistics
using Printf
using Test

export relabel_nested_ces, relabel_mod, node_ranges,
       validate_economic_inputs, normalize_input_shares,
       run_tests, run_benchmarks

# Load the optional test companion at MODULE DEFINITION TIME, before run_tests
# is defined. Deferring Base.include(...) until after the module definition can
# trigger Julia 1.12 world-age errors when run_tests calls methods defined by
# that include. Ordinary users who include this file as a library do not load
# the optional companion; they can load it independently in their application.
const _COVARIANCE_COMPANION = joinpath(@__DIR__, "WeightedCovMatrix.jl")
# Parenthesize @__FILE__ explicitly: on Julia 1.12 a following && can
# otherwise be parsed as part of the zero-argument macro invocation.
const _IS_STANDALONE = abspath(PROGRAM_FILE) == (@__FILE__)
if _IS_STANDALONE && isfile(_COVARIANCE_COMPANION)
    include("WeightedCovMatrix.jl")
end

"""
    node_ranges(N, K)

Return ranges of the six *disjoint* node groups, indexed from 1 in Julia:
original industries, intermediate-input CES nodes, value-added CES nodes,
two further N-node CES groups, and K terminal factor nodes. The original
MATLAB code calls the last three N-node groups collectively `3N` CES nests.
"""
function node_ranges(N::Integer, K::Integer)
    N >= 1 || throw(ArgumentError("N must be positive"))
    K >= 1 || throw(ArgumentError("K must be positive"))
    return (
        industries = 1:N,
        intermediate = (N + 1):(2 * N),
        value_added = (2 * N + 1):(3 * N),
        ces_first = (3 * N + 1):(4 * N),
        ces_second = (4 * N + 1):(5 * N),
        factors = (5 * N + 1):(5 * N + K),
    )
end

function _column(v::AbstractVector{<:Real}, expected::Integer, name::AbstractString)
    length(v) == expected || throw(DimensionMismatch(
        "$name must have length $expected; received $(length(v))"))
    return Vector{Float64}(v)
end

function _column(v::AbstractMatrix{<:Real}, expected::Integer, name::AbstractString)
    size(v) == (expected, 1) || throw(DimensionMismatch(
        "$name must have dimensions ($expected, 1); received $(size(v))"))
    return Vector{Float64}(view(v, :, 1))
end

function _check_sizes(Omega::AbstractMatrix, Fshares::AbstractMatrix,
                      N::Integer, M::Integer, K::Integer)
    N >= 1 || throw(ArgumentError("N must be positive"))
    K >= 1 || throw(ArgumentError("K must be positive"))
    M == 2 * N + K || throw(DimensionMismatch(
        "M must equal 2N+K = $(2 * N + K); received $M"))
    size(Omega) == (N, N) || throw(DimensionMismatch(
        "Omega must be $N-by-$N; received $(size(Omega))"))
    size(Fshares) == (3 * N, M) || throw(DimensionMismatch(
        "Fshares must be $(3 * N)-by-$M; received $(size(Fshares))"))
    return 5 * N + K
end

function _require_finite(label::AbstractString, values)
    all(isfinite, values) ||
        throw(ArgumentError("$label contains non-finite values"))
    return nothing
end

"""
    validate_economic_inputs(Omega, Fshares, alpha, beta;
                             check_row_sums=false, atol=1e-10)

OPTIONAL economic validation; this is deliberately not a normalization step.
Require nonnegative weights, `alpha` in [0,1], and optionally require every
NONZERO row of `Omega` and `Fshares`, and the final demand vector, to sum to
one. Zero rows are legitimate in the original model. The default relabelling
API does not enforce these economic restrictions, since MATLAB does not.
"""
function validate_economic_inputs(Omega::AbstractMatrix{<:Real},
                                  Fshares::AbstractMatrix{<:Real},
                                  alpha::AbstractVector{<:Real},
                                  beta::AbstractVector{<:Real};
                                  check_row_sums::Bool=false,
                                  atol::Real=1e-10)
    atol >= 0 || throw(ArgumentError("atol must be nonnegative"))
    for (name, x) in (("Omega", Omega), ("Fshares", Fshares),
                      ("alpha", alpha), ("beta", beta))
        _require_finite(name, x)
    end
    all(x -> x >= -atol, Omega) || throw(ArgumentError("Omega has negative shares"))
    all(x -> x >= -atol, Fshares) || throw(ArgumentError("Fshares has negative shares"))
    all(x -> x >= -atol, beta) || throw(ArgumentError("beta has negative shares"))
    all(x -> -atol <= x <= 1 + atol, alpha) ||
        throw(ArgumentError("alpha contains values outside [0,1]"))
    if check_row_sums
        for (name, matrix) in (("Omega", Omega), ("Fshares", Fshares))
            for i in axes(matrix, 1)
                s = sum(view(matrix, i, :))
                if abs(s) > atol && abs(s - 1.0) > atol
                    throw(ArgumentError("$name row $i sums to $s, not 0 or 1"))
                end
            end
        end
        abs(sum(beta) - 1.0) <= atol ||
            throw(ArgumentError("final demand shares do not sum to 1"))
    end
    return nothing
end

function _relabel_dense(Omega::AbstractMatrix{<:Real},
                        Fshares::AbstractMatrix{<:Real},
                        alpha::AbstractVector{Float64}, N::Int, K::Int)
    Q = 5 * N + K
    W = zeros(Float64, Q, Q)
    copyto!(view(W, (N + 1):(2 * N), 1:N), Omega)
    copyto!(view(W, (2 * N + 1):(5 * N), (3 * N + 1):Q), Fshares)
    for i in 1:N
        W[i, N + i] = 1.0 - alpha[i]
        W[i, 2 * N + i] = alpha[i]
    end
    return W
end

function _append_block!(rows::Vector{Int}, cols::Vector{Int},
                        values::Vector{Float64},
                        matrix::AbstractMatrix{<:Real},
                        row_offset::Int, col_offset::Int)
    for j in axes(matrix, 2), i in axes(matrix, 1)
        value = Float64(matrix[i, j])
        if !iszero(value)
            push!(rows, row_offset + i)
            push!(cols, col_offset + j)
            push!(values, value)
        end
    end
    return nothing
end

# Preserve sparsity when input blocks are already SparseMatrixCSC.
function _append_block!(rows::Vector{Int}, cols::Vector{Int},
                        values::Vector{Float64},
                        matrix::SparseMatrixCSC{<:Real,<:Integer},
                        row_offset::Int, col_offset::Int)
    ii, jj, vv = findnz(matrix)
    for k in eachindex(vv)
        value = Float64(vv[k])
        if !iszero(value)
            push!(rows, row_offset + ii[k])
            push!(cols, col_offset + jj[k])
            push!(values, value)
        end
    end
    return nothing
end

function _relabel_sparse(Omega::AbstractMatrix{<:Real},
                         Fshares::AbstractMatrix{<:Real},
                         alpha::AbstractVector{Float64}, N::Int, K::Int)
    Q = 5 * N + K
    rows = Int[]
    cols = Int[]
    values = Float64[]
    _append_block!(rows, cols, values, Omega, N, 0)
    _append_block!(rows, cols, values, Fshares, 2 * N, 3 * N)
    for i in 1:N
        interm = 1.0 - alpha[i]
        va = alpha[i]
        if !iszero(interm)
            push!(rows, i)
            push!(cols, N + i)
            push!(values, interm)
        end
        if !iszero(va)
            push!(rows, i)
            push!(cols, 2 * N + i)
            push!(values, va)
        end
    end
    return SparseArrays.sparse(rows, cols, values, Q, Q)
end

"""
    relabel_nested_ces(Omega, Fshares, alpha, epsilon, theta, eta, beta;
                       K, sparse=false, check_economic=false,
                       check_row_sums=false)

Return `(omega, elasticities, final_demand)` as a NamedTuple. `Omega` is N×N,
`Fshares` is 3N×(2N+K), and the other inputs have lengths N, N, N, 3N,
and 5N+K, respectively. N×1 column matrices are also accepted for these
vectors (as in MATLAB). All output values are Float64.

`omega` is a (5N+K)×(5N+K) dense matrix by default; `sparse=true` returns
SparseMatrixCSC. Entries are copied exactly. No clipping, row normalization,
or padding of final demand is performed. Set `check_economic=true` for an
optional economic feasibility check of input shares. `check_row_sums=true`
additionally checks that nonzero input rows sum to one.
"""
function relabel_nested_ces(Omega::AbstractMatrix{<:Real},
                            Fshares::AbstractMatrix{<:Real},
                            alpha::Union{AbstractVector{<:Real},AbstractMatrix{<:Real}},
                            epsilon::Union{AbstractVector{<:Real},AbstractMatrix{<:Real}},
                            theta::Union{AbstractVector{<:Real},AbstractMatrix{<:Real}},
                            eta::Union{AbstractVector{<:Real},AbstractMatrix{<:Real}},
                            beta::Union{AbstractVector{<:Real},AbstractMatrix{<:Real}};
                            K::Integer, sparse::Bool=false,
                            check_economic::Bool=false,
                            check_row_sums::Bool=false)
    N = size(Omega, 1)
    M = size(Fshares, 2)
    Q = _check_sizes(Omega, Fshares, N, M, K)
    alpha_vec = _column(alpha, N, "alpha")
    eps_vec = _column(epsilon, N, "epsilon")
    theta_vec = _column(theta, N, "theta")
    eta_vec = _column(eta, 3 * N, "eta")
    beta_vec = _column(beta, Q, "beta")
    for (name, arr) in (("Omega", Omega), ("Fshares", Fshares),
                        ("alpha", alpha_vec), ("epsilon", eps_vec),
                        ("theta", theta_vec), ("eta", eta_vec),
                        ("beta", beta_vec))
        _require_finite(name, arr)
    end
    if check_economic || check_row_sums
        validate_economic_inputs(Omega, Fshares, alpha_vec, beta_vec;
                                 check_row_sums=check_row_sums)
    end
    omega = sparse ? _relabel_sparse(Omega, Fshares, alpha_vec, N, Int(K)) :
                     _relabel_dense(Omega, Fshares, alpha_vec, N, Int(K))
    elasticities = vcat(eps_vec, theta_vec, eta_vec, ones(Float64, K))
    return (omega=omega, elasticities=elasticities, final_demand=beta_vec)
end

"""
    relabel_mod(Omega, Fshares, alpha, epsilon, theta, eta, beta, N, M, K;
                sparse=false, check_economic=false)

Positional MATLAB-compatible interface. Return the ordinary tuple
`(NewOmega, Elasticities, FinalDemand)` in exactly that order. Dimension
arguments are checked, not silently ignored.
"""
function relabel_mod(Omega::AbstractMatrix{<:Real},
                     Fshares::AbstractMatrix{<:Real},
                     alpha, epsilon, theta, eta, beta,
                     N::Integer, M::Integer, K::Integer; kwargs...)
    _check_sizes(Omega, Fshares, N, M, K)
    result = relabel_nested_ces(Omega, Fshares, alpha, epsilon, theta,
                                eta, beta; K=K, kwargs...)
    return result.omega, result.elasticities, result.final_demand
end

"""
    normalize_input_shares(Omega)

OPTIONAL MATLAB-compatible preprocessing from the Krusell caller:
`max(Omega, 0)`, followed by row normalization, leaving zero rows at zero.
Does not change its argument, and returns a dense Float64 matrix. It is
never called implicitly by `relabel_nested_ces`.
"""
function normalize_input_shares(Omega::AbstractMatrix{<:Real})
    _require_finite("Omega", Omega)
    W = Matrix{Float64}(Omega)
    for i in axes(W, 1)
        for j in axes(W, 2)
            W[i, j] = max(W[i, j], 0.0)
        end
        total = sum(view(W, i, :))
        if !iszero(total)
            for j in axes(W, 2)
                W[i, j] /= total
            end
        end
    end
    return W
end

# Literal loop-by-loop reference independently mirroring the MATLAB body.
function _matlab_reference(Omega::AbstractMatrix{<:Real},
                           Fshares::AbstractMatrix{<:Real},
                           alpha::AbstractVector{<:Real},
                           epsilon::AbstractVector{<:Real},
                           theta::AbstractVector{<:Real},
                           eta::AbstractVector{<:Real},
                           beta::AbstractVector{<:Real},
                           N::Int, M::Int, K::Int)
    Q = 5 * N + K
    NewOmega = zeros(Float64, Q, Q)
    for i in 1:N
        NewOmega[i, N + i] = 1 - alpha[i]
        NewOmega[i, 2 * N + i] = alpha[i]
        for j in 1:N
            NewOmega[N + i, j] = Omega[i, j]
        end
    end
    for i in 1:(3 * N)
        for j in 1:M
            NewOmega[2 * N + i, 3 * N + j] = Fshares[i, j]
        end
    end
    Elasticities = vcat(epsilon, theta, eta, ones(K))
    FinalDemand = copy(beta)
    return NewOmega, Elasticities, FinalDemand
end

function _fixture(N::Int, K::Int, rng::AbstractRNG; dense::Bool=false)
    M = 2 * N + K
    Omega = zeros(Float64, N, N)
    Fshares = zeros(Float64, 3 * N, M)
    alpha = rand(rng, N) .* 0.6 .+ 0.2
    for i in 1:N
        if dense
            Omega[i, :] .= rand(rng, N)
            Omega[i, :] ./= sum(view(Omega, i, :))
            Fshares[i, :] .= rand(rng, M)
            Fshares[i, :] ./= sum(view(Fshares, i, :))
        else
            Omega[i, mod1(i + 1, N)] = 1.0
            labor_share = rand(rng)
            Fshares[i, i] = labor_share
            Fshares[i, N + i] = 1.0 - labor_share
        end
        Fshares[N + i, 2 * N + min(2, K)] = 1.0
        skilled = rand(rng)
        Fshares[2 * N + i, 2 * N + 1] = 1.0 - skilled
        Fshares[2 * N + i, 2 * N + min(3, K)] += skilled
    end
    epsilon = rand(rng, N) .+ 0.3
    theta = rand(rng, N) .+ 0.5
    eta = rand(rng, 3 * N) .+ 0.2
    beta = zeros(Float64, 5 * N + K)
    beta[1] = 1.0
    return Omega, Fshares, alpha, epsilon, theta, eta, beta
end

function _max_difference(A::AbstractArray, B::AbstractArray)
    size(A) == size(B) || throw(DimensionMismatch("arrays have different shapes"))
    abs_error = maximum(abs.(A .- B))
    scale = maximum(abs, A)
    relative_error = iszero(scale) ? abs_error : abs_error / scale
    return abs_error, relative_error
end

function _check_block_structure(W::AbstractMatrix, Omega, Fshares, alpha, N, K)
    reference = zeros(Float64, size(W))
    reference[(N + 1):(2 * N), 1:N] .= Omega
    reference[(2 * N + 1):(5 * N), (3 * N + 1):(5 * N + K)] .= Fshares
    for i in 1:N
        reference[i, N + i] = 1.0 - alpha[i]
        reference[i, 2 * N + i] = alpha[i]
    end
    return W == reference
end

"""
    run_tests(; integration=true)

Run self-contained deterministic tests plus a small Leontief-inverse test.
When run directly as a script, an adjacent `WeightedCovMatrix.jl` is
included during module definition (before this function is defined). The
integration test then also checks its optimized routine, without any dynamic
`include` or Julia world-age workaround inside a running function.
"""
function run_tests(; integration::Bool=true)
    @testset "Relabel_mod.jl" begin
        rng = MersenneTwister(2019)
        @testset "N=1 and explicit block layout" begin
            N, K = 1, 2
            M, Q = 2 * N + K, 5 * N + K
            Omega = reshape([0.25], 1, 1)
            Fshares = zeros(3 * N, M)
            Fshares[1, 1] = 0.3
            Fshares[1, 2] = 0.7
            Fshares[2, 3] = 1.0
            Fshares[3, 4] = 1.0
            alpha = [0.4]
            eps, theta, eta = [0.6], [0.8], [1.1, 1.2, 1.3]
            beta = zeros(Q)
            beta[1] = 1
            result = relabel_nested_ces(Omega, Fshares, alpha, eps, theta,
                                        eta, beta; K=K, check_economic=true,
                                        check_row_sums=false)
            W, E, b = _matlab_reference(Omega, Fshares, alpha, eps, theta,
                                       eta, beta, N, M, K)
            @test size(result.omega) == (Q, Q)
            @test result.omega == W
            @test result.elasticities == E
            @test result.final_demand == b
            @test result.omega[1, N + 1] == 0.6
            @test result.omega[1, 2 * N + 1] == 0.4
            @test result.omega[N + 1, 1] == 0.25
            @test result.omega[2 * N + 1, 3 * N + 1] == 0.3
            @test result.omega[2 * N + 1, 3 * N + 2] == 0.7
            @test result.elasticities[end-K+1:end] == ones(K)
            @test node_ranges(N, K).factors == (5 * N + 1):Q
            compat = relabel_mod(Omega, Fshares, alpha, eps, theta, eta,
                                 beta, N, M, K)
            @test compat == (W, E, b)
            @test relabel_nested_ces(Omega, Fshares, reshape(alpha, N, 1),
                                     reshape(eps, N, 1), reshape(theta, N, 1),
                                     reshape(eta, 3 * N, 1),
                                     reshape(beta, Q, 1); K=K).omega == W
        end

        @testset "N=2, block-zero conditions and boundary shares" begin
            N, K = 2, 4
            M = 2 * N + K
            Omega = [0.1 0.2; 0.3 0.4]
            Fshares = reshape(collect(1.0:(3 * N * M)), 3 * N, M) ./ 100
            alpha = [0.0, 1.0]
            eps, theta, eta = [0.6, 0.7], [0.4, 0.8], collect(1.0:6.0)
            beta = collect(1.0:(5 * N + K)) ./ 100
            W, E, b = _matlab_reference(Omega, Fshares, alpha, eps, theta,
                                       eta, beta, N, M, K)
            res = relabel_nested_ces(Omega, Fshares, alpha, eps, theta,
                                     eta, beta; K=K)
            @test res.omega == W
            @test res.elasticities == E
            @test res.final_demand == b
            @test _check_block_structure(res.omega, Omega, Fshares, alpha, N, K)
            @test res.omega[1, N + 1] == 1.0
            @test res.omega[1, 2 * N + 1] == 0.0
            @test res.omega[2, N + 2] == 0.0
            @test res.omega[2, 2 * N + 2] == 1.0
            @test all(iszero, view(res.omega, (5 * N + 1):(5 * N + K), :))
            @test all(iszero, view(res.omega, 1:N, 1:N))
            @test all(iszero, view(res.omega, N+1:2*N, N+1:5*N+K))
            @test Matrix(relabel_nested_ces(sparse(Omega), sparse(Fshares),
                         alpha, eps, theta, eta, beta; K=K, sparse=true).omega) == W
        end

        @testset "Random input and sparse/dense equivalence" begin
            for N in (3, 10, 30, 65)
                K = 4
                Omega, Fshares, alpha, eps, theta, eta, beta =
                    _fixture(N, K, rng)
                M = 2 * N + K
                W, E, b = _matlab_reference(Omega, Fshares, alpha, eps,
                                           theta, eta, beta, N, M, K)
                dense = relabel_nested_ces(Omega, Fshares, alpha, eps,
                                           theta, eta, beta; K=K)
                sparse_result = relabel_nested_ces(sparse(Omega), sparse(Fshares),
                                                   alpha, eps, theta, eta,
                                                   beta; K=K, sparse=true)
                @test dense.omega == W
                @test Matrix(sparse_result.omega) == W
                @test relabel_nested_ces(sparse(Omega), sparse(Fshares),
                                        alpha, eps, theta, eta, beta;
                                        K=K).omega == W
                @test sparse_result.elasticities == E
                @test sparse_result.final_demand == b
                @test nnz(sparse_result.omega) == count(x -> !iszero(x), W)
                @test _check_block_structure(W, Omega, Fshares, alpha, N, K)
                if N == 65
                    @test size(W) == (329, 329)
                    @test length(E) == 329
                    @test length(b) == 329
                    @test size(Fshares) == (195, 134)
                end
            end
        end

        @testset "Zero, non-normalized and invalid inputs" begin
            N, K = 3, 4
            Omega, Fshares, alpha, eps, theta, eta, beta = _fixture(N, K, rng)
            Omega[2, :] .= 0.0
            Fshares[1, :] .= 0.0
            Omega[1, 1] = -0.5
            alpha[1] = 1.25
            result = relabel_nested_ces(Omega, Fshares, alpha, eps, theta,
                                         eta, beta; K=K)
            @test result.omega[N + 1, 1] == -0.5
            @test result.omega[1, 2 * N + 1] == 1.25
            @test result.omega[1, N + 1] == -0.25
            @test all(iszero, view(result.omega, N + 2, :))
            @test_throws ArgumentError relabel_nested_ces(
                Omega, Fshares, alpha, eps, theta, eta, beta;
                K=K, check_economic=true)
            Omega[1, 1] = 0.5
            alpha[1] = 0.5
            clipped = normalize_input_shares(result.omega)
            @test all(x -> x >= 0, clipped)
            @test all(iszero, view(clipped, N + 2, :))
            @test all(i -> (isapprox(sum(view(clipped, i, :)), 1.0; atol=1e-12) ||
                            all(iszero, view(clipped, i, :))), axes(clipped, 1))
            @test_throws DimensionMismatch relabel_mod(
                Omega, Fshares, alpha, eps, theta, eta, beta, N, 2 * N + K - 1, K)
            @test_throws DimensionMismatch relabel_nested_ces(
                Omega[:, 1:(N - 1)], Fshares, alpha, eps, theta, eta, beta; K=K)
            @test_throws DimensionMismatch relabel_nested_ces(
                Omega, Fshares, alpha, eps, theta, eta[1:end-1], beta; K=K)
            @test_throws DimensionMismatch relabel_nested_ces(
                Omega, Fshares, alpha, eps, theta, eta, beta[1:end-1]; K=K)
            broken = copy(Omega)
            broken[1, 1] = NaN
            @test_throws ArgumentError relabel_nested_ces(
                broken, Fshares, alpha, eps, theta, eta, beta; K=K)
        end

        @testset "Independent normalization and economic checks" begin
            A = [0.0 0.0; -2.0 1.0; 2.0 2.0]
            normed = normalize_input_shares(A)
            @test normed == [0.0 0.0; 0.0 1.0; 0.5 0.5]
            @test A[2, 1] == -2.0
            N, K = 3, 4
            Omega, Fshares, alpha, eps, theta, eta, beta = _fixture(N, K, rng)
            @test validate_economic_inputs(Omega, Fshares, alpha, beta;
                                           check_row_sums=true) === nothing
            @test relabel_nested_ces(Omega, Fshares, alpha, eps, theta,
                                     eta, beta; K=K, check_economic=true,
                                     check_row_sums=true).omega isa Matrix{Float64}
            Fshares[2, :] .*= 0.4
            @test_throws ArgumentError validate_economic_inputs(
                Omega, Fshares, alpha, beta; check_row_sums=true)
        end

        @testset "Minimal downstream integration" begin
            if integration
                N, K = 3, 4
                Omega, Fshares, alpha, eps, theta, eta, beta = _fixture(N, K, rng)
                output = relabel_nested_ces(Omega, Fshares, alpha,
                                            eps, theta, eta, beta; K=K)
                W = normalize_input_shares(output.omega)
                Q = size(W, 1)
                identity = Matrix{Float64}(I, Q, Q)
                # Solve the linear system, instead of forming inv(I-W).
                Psi = (identity - W) \ identity
                @test isapprox((identity - W) * Psi, identity; atol=1e-10)
                lambda = transpose(Psi) * output.final_demand
                coefficients = lambda .* (output.elasticities .- 1.0)
                @test length(coefficients) == Q
                @test all(isfinite, coefficients)
                if _IS_STANDALONE && isfile(_COVARIANCE_COMPANION)
                    # Regression check: when run as a script with its companion
                    # present, the companion must already be defined.
                    @test isdefined(@__MODULE__, :WeightedCovMatrix)
                end
                if isdefined(@__MODULE__, :WeightedCovMatrix)
                    H = WeightedCovMatrix.aggregate_weighted_covariance(
                        W, Psi, coefficients)
                    tensor = WeightedCovMatrix.weighted_covariance(W, Psi)
                    Href = zeros(Float64, Q, Q)
                    for k in 1:Q
                        Href .+= coefficients[k] .* view(tensor, :, :, k)
                    end
                    @test isapprox(H, Href; atol=1e-10, rtol=1e-10)
                else
                    @info "WeightedCovMatrix.jl not adjacent; integration verified through Leontief system only"
                end
            end
        end
    end
    println("Relabel_mod.jl: all Julia testsets completed (see the Test Summary above).")
    return nothing
end

function _timing(f, args...; samples::Int=20)
    f(args...)                         # warm-up / precompile
    GC.gc()
    bytes = @allocated f(args...)
    samples_time = Vector{Float64}(undef, samples)
    for i in 1:samples
        tic = time_ns()
        f(args...)
        samples_time[i] = (time_ns() - tic) / 1e9
    end
    return median(samples_time), bytes
end

function _optimized_dense(Omega, Fshares, alpha, epsilon, theta, eta, beta, K)
    return relabel_nested_ces(Omega, Fshares, alpha, epsilon, theta, eta, beta;
                              K=K).omega
end
function _optimized_sparse(Omega, Fshares, alpha, epsilon, theta, eta, beta, K)
    return relabel_nested_ces(Omega, Fshares, alpha, epsilon, theta, eta, beta;
                              K=K, sparse=true).omega
end
function _reference_only(Omega, Fshares, alpha, epsilon, theta, eta,
                         beta, N, M, K)
    return first(_matlab_reference(Omega, Fshares, alpha, epsilon, theta,
                                   eta, beta, N, M, K))
end

"""
    run_benchmarks(; Ns=(10,30,65), K=4, samples=20)

Warm-started, reproducible dense/sparse/reference relabelling benchmarks.
Reports median wall time, allocated bytes, nonzeros, output storage, and errors.
The synthetic fixtures reproduce the model's sparse *shape* but are NOT BEA data.
"""
function run_benchmarks(; Ns=(10, 30, 65), K::Int=4, samples::Int=20)
    @printf("\nRelabelling benchmarks: reference vs dense vs sparse\n")
    @printf("%5s  %10s  %10s  %10s  %11s  %11s  %11s  %8s  %10s\n",
            "N", "ref ms", "dense ms", "sparse ms",
            "ref alloc", "dense alloc", "sparse alloc", "nnz", "max |diff|")
    for N in Ns
        rng = MersenneTwister(2000 + N)
        Omega, Fshares, alpha, eps, theta, eta, beta = _fixture(N, K, rng)
        args = (Omega, Fshares, alpha, eps, theta, eta, beta, N, 2 * N + K, K)
        # Explicit argument forms for the public implementation:
        public_args = (Omega, Fshares, alpha, eps, theta, eta, beta, K)
        rt, rb = _timing(_reference_only, args...; samples=samples)
        dt, db = _timing(_optimized_dense, public_args...; samples=samples)
        st, sb = _timing(_optimized_sparse, public_args...; samples=samples)
        Wref = _reference_only(args...)
        Wdense = _optimized_dense(public_args...)
        Wsparse = _optimized_sparse(public_args...)
        err = max(maximum(abs.(Wref .- Wdense)),
                  maximum(abs.(Wref .- Matrix(Wsparse))))
        @printf("%5d  %10.4f  %10.4f  %10.4f  %11d  %11d  %11d  %8d  %10.3e\n",
                N, 1e3 * rt, 1e3 * dt, 1e3 * st, rb, db, sb, nnz(Wsparse), err)
        @printf("        output storage dense=%d B, sparse=%d B; scale-free relative error %.3e\n",
                Base.summarysize(Wdense), Base.summarysize(Wsparse),
                iszero(maximum(abs, Wref)) ? err : err / maximum(abs, Wref))
    end
    return nothing
end

end # module NestedCESRelabelling

if NestedCESRelabelling._IS_STANDALONE
    NestedCESRelabelling.run_tests()
    NestedCESRelabelling.run_benchmarks()
end
