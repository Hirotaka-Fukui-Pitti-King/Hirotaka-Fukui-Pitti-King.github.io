"""
    WeightedCovMatrix

Weighted input-output covariances for Baqaee and Farhi (2019), equation (7).

This file has no external package dependencies. Include it once with
`include("WeightedCovMatrix.jl")`, then call `WeightedCovMatrix.weighted_covariance`
or `WeightedCovMatrix.aggregate_weighted_covariance`.

Running this file directly executes self-tests and a small, warm-started benchmark.
"""
module WeightedCovMatrix

using LinearAlgebra
using Printf
using Random
using Statistics
using Test

export weighted_covariance, aggregate_weighted_covariance,
       validate_probability_weights, run_tests, run_benchmarks

# Core routines use Float64 BLAS inputs, matching the original MATLAB double
# computations. Input arrays are not modified; dense Float64 matrices are reused.
_as_matrix(A::Matrix{Float64}) = A
_as_matrix(A::AbstractMatrix{<:Real}) = Matrix{Float64}(A)
_as_vector(v::Vector{Float64}) = v
_as_vector(v::AbstractVector{<:Real}) = Vector{Float64}(v)

function _check_dimensions(W::AbstractMatrix, P::AbstractMatrix)
    n, ncols = size(P)
    n == ncols || throw(DimensionMismatch("Psi must be square; got $(size(P))"))
    n > 0 || throw(ArgumentError("Psi must have at least one row"))
    size(W, 2) == n || throw(DimensionMismatch(
        "weight must have $n columns; got $(size(W, 2))"))
    size(W, 1) > 0 || throw(ArgumentError("weight must have at least one row"))
    return n, size(W, 1)
end

function _check_finite(W, P)
    all(isfinite, W) || throw(ArgumentError("weight contains non-finite values"))
    all(isfinite, P) || throw(ArgumentError("Psi contains non-finite values"))
    return nothing
end

"""
    validate_probability_weights(weight; atol=1e-12, allow_zero_rows=true)

Validate nonnegative, finite row weights that sum to one, optionally allowing
all-zero rows. Accept a vector or an m-by-N matrix. No normalization is done.
This check is OPTIONAL: the original MATLAB operator is defined algebraically
for negative, zero, and non-normalized weights as well.
"""
function validate_probability_weights(W::AbstractMatrix{<:Real};
                                      atol::Real=1e-12,
                                      allow_zero_rows::Bool=true)
    atol >= 0 || throw(ArgumentError("atol must be nonnegative"))
    all(isfinite, W) || throw(ArgumentError("weights must be finite"))
    for k in axes(W, 1)
        row = view(W, k, :)
        all(x -> x >= 0, row) ||
            throw(ArgumentError("row $k contains negative weights"))
        total = sum(row)
        if allow_zero_rows && all(iszero, row)
            continue
        end
        abs(total - 1) <= atol ||
            throw(ArgumentError("weights in row $k sum to $total, not 1"))
    end
    return nothing
end

function validate_probability_weights(w::AbstractVector{<:Real}; kwargs...)
    return validate_probability_weights(reshape(w, 1, length(w)); kwargs...)
end

# Implements the literal MATLAB expression for one row using preallocated
# storage. Here P is real, so MATLAB's conjugate transpose equals transpose(P).
function _covariance_row!(out::AbstractMatrix{Float64},
                          w::AbstractVector{Float64},
                          P::Matrix{Float64},
                          scaled::Matrix{Float64},
                          mu::Vector{Float64})
    n = size(P, 1)
    for j in 1:n, i in 1:n
        scaled[i, j] = w[i] * P[i, j]
    end
    mul!(out, transpose(P), scaled)
    mul!(mu, transpose(P), w)
    for j in 1:n, i in 1:n
        out[i, j] -= mu[i] * mu[j]
    end
    return out
end

"""
    weighted_covariance(weight::AbstractMatrix, Psi; check_finite=true)

MATLAB-compatible `WeightedCovMatrix(weight, matr)`.

`weight` has size (m, N), `Psi` has size (N, N); output has size (N, N, m),
INCLUDING when m == 1. Slice k equals

    Psi' * Diagonal(weight[k, :]) * Psi -
      (weight[k, :]' * Psi)' * (weight[k, :]' * Psi)

for real-valued inputs. Non-normalized and negative weights are permitted, and
zero rows produce zero slices. Computation is in Float64; no automatic
normalization, recentering, or symmetrization is performed.
"""
function weighted_covariance(weight::AbstractMatrix{<:Real},
                             Psi::AbstractMatrix{<:Real};
                             check_finite::Bool=true)
    n, m = _check_dimensions(weight, Psi)
    check_finite && _check_finite(weight, Psi)
    W = _as_matrix(weight)
    P = _as_matrix(Psi)
    result = Array{Float64}(undef, n, n, m)
    scaled = Matrix{Float64}(undef, n, n)
    mu = Vector{Float64}(undef, n)
    for k in 1:m
        _covariance_row!(view(result, :, :, k), view(W, k, :), P, scaled, mu)
    end
    return result
end

"""
    weighted_covariance(weight::AbstractVector, Psi; check_finite=true)

Convenience API for one row: returns an N-by-N MATRIX, not N-by-N-by-1.
For MATLAB's 3D shape, pass `reshape(weight, 1, length(weight))` instead.
"""
function weighted_covariance(weight::AbstractVector{<:Real},
                             Psi::AbstractMatrix{<:Real};
                             check_finite::Bool=true)
    n = size(Psi, 1)
    size(Psi, 2) == n ||
        throw(DimensionMismatch("Psi must be square; got $(size(Psi))"))
    n > 0 || throw(ArgumentError("Psi must have at least one row"))
    length(weight) == n ||
        throw(DimensionMismatch("weight must have $n entries; got $(length(weight))"))
    check_finite && _check_finite(weight, Psi)
    P = _as_matrix(Psi)
    w = _as_vector(weight)
    result = Matrix{Float64}(undef, n, n)
    scaled = Matrix{Float64}(undef, n, n)
    mu = Vector{Float64}(undef, n)
    return _covariance_row!(result, w, P, scaled, mu)
end

"""
    aggregate_weighted_covariance(Omega, Psi, coefficients; check_finite=true)

Compute H = sum(coefficients[k] * C_k for k=1:m) with NO N-by-N-by-m
covariance tensor and NO diagonal matrix materializations.

- `Omega`: m-by-N weights (rows need not sum to one).
- `Psi`: N-by-N real exposure matrix.
- `coefficients`: vector of length m; in Baqaee-Farhi, a_k = lambda_k*(theta_k-1).

Identity:
    H = Psi' * Diagonal(Omega' * a) * Psi -
        (Omega * Psi)' * Diagonal(a) * (Omega * Psi)

The algorithm computes three dense matrix products and uses 5-argument `mul!`
to subtract the second term in place. Floating-point results may differ in
last digits from a sum of separate covariances (different summation order).
"""
function aggregate_weighted_covariance(Omega::AbstractMatrix{<:Real},
                                       Psi::AbstractMatrix{<:Real},
                                       coefficients::AbstractVector{<:Real};
                                       check_finite::Bool=true)
    n, m = _check_dimensions(Omega, Psi)
    length(coefficients) == m || throw(DimensionMismatch(
        "coefficients must have length $m; got $(length(coefficients))"))
    if check_finite
        _check_finite(Omega, Psi)
        all(isfinite, coefficients) ||
            throw(ArgumentError("coefficients contain non-finite values"))
    end
    W = _as_matrix(Omega)
    P = _as_matrix(Psi)
    a = _as_vector(coefficients)

    # First term: P' * Diagonal(W' * a) * P.
    weighted_inputs = transpose(W) * a
    scaled_P = Matrix{Float64}(undef, n, n)
    for j in 1:n, i in 1:n
        scaled_P[i, j] = weighted_inputs[i] * P[i, j]
    end
    H = Matrix{Float64}(undef, n, n)
    mul!(H, transpose(P), scaled_P)

    # Second term: (W * P)' * Diagonal(a) * (W * P).
    exposure = W * P
    scaled_exposure = similar(exposure)
    for j in 1:n, k in 1:m
        scaled_exposure[k, j] = a[k] * exposure[k, j]
    end
    mul!(H, transpose(exposure), scaled_exposure, -1.0, 1.0)
    return H
end

"""
    aggregate_weighted_covariance(weight::AbstractVector, Psi, coefficient::Real)

Single-row shortcut, useful for the final-demand term:
`(sigma - 1) * weighted_covariance(b, Psi)`.
No three-dimensional allocation is made.
"""
function aggregate_weighted_covariance(weight::AbstractVector{<:Real},
                                       Psi::AbstractMatrix{<:Real},
                                       coefficient::Real;
                                       check_finite::Bool=true)
    if check_finite && !isfinite(coefficient)
        throw(ArgumentError("coefficient must be finite"))
    end
    return Float64(coefficient) .* weighted_covariance(weight, Psi;
                                                      check_finite=check_finite)
end

# Independent scalar-loop reference: the original equation, not the optimized
# matrix identity. Used only for tests on small matrices.
function _scalar_reference(w::AbstractVector, P::AbstractMatrix)
    n = size(P, 1)
    C = zeros(Float64, n, n)
    for j in 1:n, i in 1:n
        first_moment = 0.0
        mean_i = 0.0
        mean_j = 0.0
        for l in 1:n
            first_moment += w[l] * P[l, i] * P[l, j]
            mean_i += w[l] * P[l, i]
            mean_j += w[l] * P[l, j]
        end
        C[i, j] = first_moment - mean_i * mean_j
    end
    return C
end

# Deliberately allocates the MATLAB-style 3D covariance tensor.
function _tensor_then_sum(W::AbstractMatrix, P::AbstractMatrix, a::AbstractVector)
    tensor = weighted_covariance(W, P)
    n = size(P, 1)
    H = zeros(Float64, n, n)
    for k in 1:length(a)
        H .+= a[k] .* view(tensor, :, :, k)
    end
    return H
end

"""
    run_tests(; verbose=true)

Reproducible tests of analytic cases, shape conventions, zero/invalid rows,
PSD for probability weights, indefinite coefficients, and matrix aggregation.
"""
function run_tests(; verbose::Bool=true)
    rng = MersenneTwister(20261112)
    set = @testset "WeightedCovMatrix.jl" begin
        @testset "Two factors: analytic covariance" begin
            P = [1.0 2.0; 3.0 4.0]
            w = [0.25, 0.75]
            expected = fill(0.75, 2, 2)  # p(1-p) * [(-2)^2] = 0.75
            C = weighted_covariance(w, P)
            @test size(C) == (2, 2)
            @test isapprox(C, expected; rtol=1e-13, atol=1e-13)
            W = reshape(w, 1, 2)
            C3 = weighted_covariance(W, P)
            @test size(C3) == (2, 2, 1)
            @test isapprox(C3[:, :, 1], expected; atol=1e-13)
        end

        @testset "Three factors and nonuniform weights" begin
            P = [1.0 0.0 2.0; 0.0 1.0 1.0; 2.0 3.0 1.0]
            w = [0.2, 0.3, 0.5]
            @test isapprox(weighted_covariance(w, P),
                           _scalar_reference(w, P); rtol=1e-13, atol=1e-13)
            @test validate_probability_weights(w) === nothing
            @test eigmin(Symmetric(weighted_covariance(w, P))) >= -1e-12
        end

        @testset "Zero rows, raw weights, and shapes" begin
            P = Matrix{Float64}(I, 3, 3)
            W = [0.0 0.0 0.0;
                 1/3 1/3 1/3;
                 0.8 0.1 0.1]
            C = weighted_covariance(W, P)
            @test size(C) == (3, 3, 3)
            @test all(iszero, C[:, :, 1])
            @test validate_probability_weights(W) === nothing
            @test_throws ArgumentError validate_probability_weights(W;
                                                                  allow_zero_rows=false)
            @test isapprox(C[:, :, 3], _scalar_reference(vec(W[3, :]), P);
                           atol=1e-13)
            raw = [2.0, 0.0, 0.0]
            @test isapprox(weighted_covariance(raw, P)[1, 1], -2.0; atol=1e-14)
            @test_throws ArgumentError validate_probability_weights(raw)
            signed = [-1.0, 2.0, 0.0]
            @test isapprox(weighted_covariance(signed, P),
                           _scalar_reference(signed, P); atol=1e-13)
            @test_throws ArgumentError validate_probability_weights(signed)
        end

        @testset "Aggregate identity and mixed signs" begin
            for n in (2, 3, 10, 50)
                W = rand(rng, n, n)
                W ./= sum(W; dims=2)
                P = randn(rng, n, n)
                a = randn(rng, n)
                reference = _tensor_then_sum(W, P, a)
                optimized = aggregate_weighted_covariance(W, P, a)
                @test isapprox(optimized, reference; rtol=5e-11, atol=5e-11)
                @test isapprox(optimized, transpose(optimized);
                               rtol=5e-11, atol=5e-11)
                for k in 1:min(n, 3)
                    @test isapprox(weighted_covariance(view(W, k, :), P),
                                   _scalar_reference(view(W, k, :), P);
                                   rtol=5e-12, atol=5e-12)
                end
            end
            P = [2.0 1.0; 1.0 3.0]
            W = [0.5 0.5; 0.3 0.7]
            @test all(iszero, aggregate_weighted_covariance(W, P, zeros(2)))
            @test isapprox(aggregate_weighted_covariance(W, P, [0.0, 1.0]),
                           weighted_covariance(view(W, 2, :), P); atol=1e-13)
            # Rectangular m-by-N weights (m != N), including an unnormalized row.
            rectangular_W = [0.2 0.3 0.5;
                             2.0 0.0 0.0]
            rectangular_P = [1.0 2.0 0.0;
                             0.0 1.0 3.0;
                             4.0 0.0 1.0]
            rectangular_a = [1.5, -0.4]
            @test size(weighted_covariance(rectangular_W, rectangular_P)) == (3, 3, 2)
            @test isapprox(aggregate_weighted_covariance(
                               rectangular_W, rectangular_P, rectangular_a),
                           _tensor_then_sum(rectangular_W, rectangular_P, rectangular_a);
                           rtol=1e-13, atol=1e-13)
            # Cobb-Douglas nests have theta = 1, so a = lambda .* (theta .- 1) = 0.
            lambda = [0.6, 0.4]
            theta = ones(2)
            @test all(iszero, aggregate_weighted_covariance(
                W, P, lambda .* (theta .- 1)))
        end

        @testset "Validation" begin
            P = Matrix{Float64}(I, 2, 2)
            @test_throws DimensionMismatch weighted_covariance([0.2, 0.3, 0.5], P)
            @test_throws DimensionMismatch weighted_covariance(ones(3, 3), P)
            @test_throws DimensionMismatch weighted_covariance(ones(2, 2), ones(2, 3))
            @test_throws DimensionMismatch aggregate_weighted_covariance(
                ones(2, 2), P, ones(3))
            @test_throws ArgumentError weighted_covariance([NaN, 1.0], P)
            @test_throws ArgumentError aggregate_weighted_covariance(
                ones(2, 2), P, [Inf, 1.0])
            @test_throws ArgumentError validate_probability_weights([0.5, 0.4])
        end
    end
    verbose && println("All Julia testsets completed; check the Test summary above.")
    return set
end

"""
    run_benchmarks(; sizes=(10, 50, 100), repetitions=5, seed=20261112)

Compare (A) tensor-then-sum with (B) direct matrix aggregation. Warm up both
methods first so the timing samples do not include JIT compilation. Prints
median elapsed seconds and median allocated bytes over independent repetitions.
No benchmark values are precomputed or guessed.
"""
function run_benchmarks(; sizes=(10, 50, 100),
                        repetitions::Integer=5,
                        seed::Integer=20261112)
    repetitions > 0 || throw(ArgumentError("repetitions must be positive"))
    rng = MersenneTwister(seed)
    println("\nBenchmark: tensor then sum (A) vs matrix aggregation (B)")
    @printf("%5s %12s %12s %13s %13s %11s\n",
            "N", "A time (s)", "B time (s)", "A alloc (B)", "B alloc (B)", "max |diff|")
    for n in sizes
        n > 0 || throw(ArgumentError("matrix sizes must be positive"))
        W = rand(rng, n, n)
        W ./= sum(W; dims=2)
        P = randn(rng, n, n)
        a = randn(rng, n)
        ref = _tensor_then_sum(W, P, a)
        opt = aggregate_weighted_covariance(W, P, a)
        err = maximum(abs, ref .- opt)
        @assert isapprox(ref, opt; atol=5e-10, rtol=5e-10)

        t_a = Float64[]
        t_b = Float64[]
        mem_a = Int[]
        mem_b = Int[]
        # Both methods have already run once on these concrete argument types.
        for _ in 1:repetitions
            GC.gc()
            push!(t_a, @elapsed _tensor_then_sum(W, P, a))
            push!(mem_a, @allocated _tensor_then_sum(W, P, a))
            GC.gc()
            push!(t_b, @elapsed aggregate_weighted_covariance(W, P, a))
            push!(mem_b, @allocated aggregate_weighted_covariance(W, P, a))
        end
        @printf("%5d %12.6f %12.6f %13d %13d %11.3e\n",
                n, median(t_a), median(t_b),
                Int(median(mem_a)), Int(median(mem_b)), err)
    end
    return nothing
end

end # module WeightedCovMatrix

if abspath(PROGRAM_FILE) == @__FILE__
    WeightedCovMatrix.run_tests()
    WeightedCovMatrix.run_benchmarks()
end
