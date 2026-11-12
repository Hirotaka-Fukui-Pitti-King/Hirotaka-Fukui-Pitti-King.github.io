#     Krusell_sens.jl
#
# Baqaee and Farhi (2019), JEEA 17(5), Appendix B, Tables B.3 and B.4.
# Replicate the effective MATLAB sensitivity calculations in `Krusell_agg_sens.m`
# and `Krusell_disagg_sens.m`, using the previously validated Julia benchmark
# implementations and the original five MAT files.
#
# The sensitivity is d(MES)/d(theta_k), not d(MES)/d(shock multiplier).
# The MATLAB raw MES orientation is retained. Table B.3 is raw [1,3],
# Table B.4 is raw [1,2]. The paper tables have several discrepancies with
# its public MATLAB scripts; these are reported rather than corrected silently.
#
# Prerequisites in the SAME directory:
#     Krusell_disagg.jl, Krusell_agg.jl, Relabel_mod.jl, WeightedCovMatrix.jl
# Packages: MAT.jl, SummaryTables.jl; all other dependencies are Julia stdlib.
#
# PowerShell usage:
#     julia --depwarn=error --project=. Krusell_sens.jl --test-only
#     julia --depwarn=error --project=. Krusell_sens.jl --data-dir PATH --bench
#     julia --depwarn=error --project=. Krusell_sens.jl --data-dir PATH --step 1e-5
#
# No module is loaded dynamically within a test function (Julia 1.12 world age).

# Krusell_disagg.jl loads Krusell_agg.jl, which in turn loads Relabel_mod.jl
# and WeightedCovMatrix.jl. Do not include these modules again here.
include(joinpath(@__DIR__, "Krusell_disagg.jl"))

module KrusellSensitivities

import ..KrusellAggregate
import ..KrusellDisaggregated
using ..KrusellAggregate: load_krusell_data, prepare_aggregate_inputs,
                          compute_aggregate_equilibrium,
                          compute_aggregate_second_order, compute_aggregate_mes
using ..KrusellDisaggregated: prepare_disaggregated_inputs,
                              compute_disaggregated_equilibrium
using ..WeightedCovMatrix: aggregate_weighted_covariance, weighted_covariance
using LinearAlgebra
using Statistics
using Printf
using Test
using Random
using SummaryTables: Table, Cell

export load_sensitivity_data, baseline_calibration, prepare_sensitivity_context,
       perturb_microelasticity, compute_sensitivity, analytic_sensitivity,
       compute_all_sensitivities, verify_sensitivity_results,
       save_sensitivity_results, run_tests, run_replication, run_benchmark

const N_ORIGINAL = 65
const Q = 329
const FACTORS = 326:328
const THETA_BENCH = (theta0=0.9, theta1=0.67, theta2=1.67,
                     theta3=0.5, theta4=0.0001)
const PARAMETERS = (:theta0, :theta3, :theta4, :theta2, :theta1)
const AGG_PARAMETERS = (:theta2, :theta1)
const PAIRS = ((:HK, 1, 3), (:LK, 1, 2))
const PAIR_LABELS = (HK="High-skilled labor / Capital",
                     LK="Low-skilled labor / Capital")
const PARAM_LABELS = (theta0="Consumption: theta0",
                      theta3="VA versus INT: theta3",
                      theta4="INT: theta4",
                      theta2="VA outer nest: theta2",
                      theta1="VA inner nest: theta1")

# The paper's printed strings are retained, including unequal precisions
# such as "0.1" vs "0.10". A printed number's match is assessed using
# the precision actually printed, not always 2 decimal places.
const PUBLISHED = (
    HK=(benchmark=(aggregate_cost="0.67", aggregate_production="0.67",
                   disaggregated_cost="0.64", disaggregated_production="0.66"),
        aggregate_cost=(theta2="0", theta1="1"),
        aggregate_production=(theta2="0", theta1="1"),
        disaggregated_cost=(theta0="0.13", theta3="0.1", theta4="0.09",
                            theta2="0.01", theta1="0.67"),
        disaggregated_production=(theta0="0.12", theta3="0.09", theta4="0.08",
                                  theta2="0.05", theta1="0.65")),
    LK=(benchmark=(aggregate_cost="1.28", aggregate_production="1.05",
                   disaggregated_cost="1.09", disaggregated_production="0.93"),
        aggregate_cost=(theta2="0.4", theta1="0.6"),
        aggregate_production=(theta2="0.97", theta1="0.24"),
        disaggregated_cost=(theta0="0.11", theta3="0.09", theta4="0.1",
                            theta2="0.26", theta1="0.46"),
        disaggregated_production=(theta0="0.14", theta3="0.11", theta4="0.07",
                                  theta2="0.57", theta1="0.22")),
)

"""
    load_sensitivity_data(path)

Use the same validated five-MAT-file reader as Tasks V and VI.
"""
load_sensitivity_data(path::AbstractString) = load_krusell_data(path)

"""
    baseline_calibration(model)

Return the independently chosen microelasticities. The aggregate model only
varies theta1 and theta2; the other three are not defined as active aggregate
sensitivity parameters in the effective MATLAB script.
"""
function baseline_calibration(model::Symbol)
    if model === :aggregate
        return (theta1=THETA_BENCH.theta1, theta2=THETA_BENCH.theta2)
    elseif model === :disaggregated
        return THETA_BENCH
    end
    throw(ArgumentError("model must be :aggregate or :disaggregated; got $model"))
end

function _parameter_value(model::Symbol, parameter::Symbol)
    calib = baseline_calibration(model)
    hasproperty(calib, parameter) || throw(ArgumentError(
        "$parameter is not an active microelasticity for $model"))
    return Float64(getproperty(calib, parameter))
end

function _parameter_indices(parameter::Symbol)
    if parameter === :theta0
        # theta(1) is at transformed index N+1.
        return (N_ORIGINAL + 1):(N_ORIGINAL + 1)
    elseif parameter === :theta1
        # eta(N+1:3N): last TWO nested-CES blocks, including inactive nodes.
        return (3 * N_ORIGINAL + 1):(5 * N_ORIGINAL)
    elseif parameter === :theta2
        # eta(3:N): 63 outer value-added nests.
        return (2 * N_ORIGINAL + 3):(3 * N_ORIGINAL)
    elseif parameter === :theta3
        # epsilon(3:N): industry-level value added vs intermediates.
        return 3:N_ORIGINAL
    elseif parameter === :theta4
        # theta(3:N): industry intermediate-input nests.
        return (N_ORIGINAL + 3):(2 * N_ORIGINAL)
    end
    throw(ArgumentError("unrecognized microelasticity $parameter"))
end

"""
    perturb_microelasticity(inputs, model, parameter; relative_step=1e-5)

Mirror `theta_k_shock = 1 + relative_step` in the original MATLAB scripts.
Only the appropriate coordinates of the relabelled elasticity vector change.
For theta0, the scalar final-demand sigma changes too, exactly as the
Disaggregated MATLAB script specifies. The original `inputs` are unmodified.
"""
function perturb_microelasticity(inputs::NamedTuple, model::Symbol,
                                 parameter::Symbol; relative_step::Real=1e-5)
    base = _parameter_value(model, parameter)
    h = Float64(relative_step)
    isfinite(h) && h > -1 && h != 0.0 ||
        throw(ArgumentError("relative_step must be finite, nonzero, and > -1"))
    length(inputs.elasticities) == Q || throw(DimensionMismatch(
        "expected Q=$Q transformed elasticities"))
    multiplier = 1.0 + h
    multiplier != 1.0 || throw(ArgumentError("step is too small for Float64"))
    modified = Float64.(copy(inputs.elasticities))
    modified[_parameter_indices(parameter)] .*= multiplier
    sigma = parameter === :theta0 ? Float64(inputs.sigma) * multiplier : Float64(inputs.sigma)
    return (elasticities=modified, sigma=sigma, parameter=parameter,
            base_value=base, multiplier=multiplier,
            perturbed_value=base * multiplier,
            delta_micro=base * (multiplier - 1.0))
end

function _mes_at(equilibrium::NamedTuple, final_demand::AbstractVector,
                 elasticities::AbstractVector, sigma::Real)
    second = compute_aggregate_second_order(
        equilibrium.omega, equilibrium.psi, equilibrium.lambda, elasticities;
        sigma=sigma, final_demand=final_demand)
    mes = compute_aggregate_mes(second.hessian, equilibrium.lambda;
                                factor_ids=FACTORS)
    return (mes=mes, second=second)
end

"""
    prepare_sensitivity_context(model, data)

Construct the original benchmark ONCE, using the verified aggregate or
industry-disaggregated data preparation and equilibrium solvers.  For all
subsequent microelasticity perturbations, Omega, Psi, lambda and factor shares
are fixed (local comparative statics at the benchmark equilibrium).
"""
function prepare_sensitivity_context(model::Symbol, data::NamedTuple)
    baseline_calibration(model) # early validation
    inputs = if model === :aggregate
        prepare_aggregate_inputs(data)
    else
        prepare_disaggregated_inputs(data; sparse=true)
    end
    equilibrium = if model === :aggregate
        compute_aggregate_equilibrium(inputs)
    else
        compute_disaggregated_equilibrium(inputs)
    end
    evaluated = _mes_at(equilibrium, inputs.final_demand,
                        inputs.elasticities, inputs.sigma)
    return (model=model, inputs=inputs, equilibrium=equilibrium,
            baseline=evaluated.mes,
            baseline_hessian=evaluated.second.hessian)
end

"""
    compute_sensitivity(context, parameter; step=1e-5, method=:forward)

Return MATLAB-oriented 3x3 production/cost MES derivatives with respect to
**the level of theta_k**, not to its multiplicative shock.  `method=:forward`
reproduces MATLAB `(rho(1+h)-rho(1)) / (theta_k*((1+h)-1))`.
Central differences are only an independent numerical diagnostic.
"""
function compute_sensitivity(context::NamedTuple, parameter::Symbol;
                             step::Real=1e-5, method::Symbol=:forward)
    h = Float64(step)
    isfinite(h) && 0 < h < 1 || throw(ArgumentError("step must be in (0,1)"))
    inp = context.inputs
    plus = perturb_microelasticity(inp, context.model, parameter;
                                    relative_step=h)
    plus_mes = _mes_at(context.equilibrium, inp.final_demand,
                       plus.elasticities, plus.sigma).mes
    if method === :forward
        prod = (plus_mes.mes_production .- context.baseline.mes_production) ./ plus.delta_micro
        cost = (plus_mes.mes_cost .- context.baseline.mes_cost) ./ plus.delta_micro
    elseif method === :central
        minus = perturb_microelasticity(inp, context.model, parameter;
                                         relative_step=-h)
        minus_mes = _mes_at(context.equilibrium, inp.final_demand,
                            minus.elasticities, minus.sigma).mes
        denom = plus.perturbed_value - minus.perturbed_value
        prod = (plus_mes.mes_production .- minus_mes.mes_production) ./ denom
        cost = (plus_mes.mes_cost .- minus_mes.mes_cost) ./ denom
    else
        throw(ArgumentError("method must be :forward or :central"))
    end
    all(isfinite, prod) && all(isfinite, cost) ||
        throw(ArgumentError("non-finite MES derivative for $parameter"))
    return (model=context.model, parameter=parameter, method=method,
            base_value=plus.base_value, perturbed_value=plus.perturbed_value,
            step=h, delta_micro=plus.delta_micro,
            mes_production=prod, mes_cost=cost,
            baseline_production=context.baseline.mes_production,
            baseline_cost=context.baseline.mes_cost,
            perturbed_production=plus_mes.mes_production,
            perturbed_cost=plus_mes.mes_cost)
end

"""
    analytic_sensitivity(context, parameter)

Analytic derivative at fixed benchmark Omega, Psi, lambda:

  dH = d(sigma)*Cov_b(Psi,Psi)
       + sum_k lambda_k*d(elasticity_k)*Cov_{Omega(k)}(Psi,Psi),
  dE = dH_FF*diag(1/lambda_F),
  dN = (I+E)^(-1)*dE*(I+E)^(-1).

The dH calculation REUSES the efficient weighted-covariance operator, and
never constructs a 329x329x329 tensor.
"""
function analytic_sensitivity(context::NamedTuple, parameter::Symbol)
    _parameter_value(context.model, parameter) # reject inactive parameters
    eq = context.equilibrium
    deriv_elasticity = zeros(Float64, Q)
    deriv_elasticity[_parameter_indices(parameter)] .= 1.0
    deriv_sigma = parameter === :theta0 ? 1.0 : 0.0
    coeff = eq.lambda .* deriv_elasticity
    dH = aggregate_weighted_covariance(eq.omega, eq.psi, coeff)
    if deriv_sigma != 0.0
        dH .+= deriv_sigma .* weighted_covariance(context.inputs.final_demand, eq.psi)
    end
    shares = context.baseline.factor_shares
    dE = dH[FACTORS, FACTORS] * Diagonal(1.0 ./ shares)
    E = context.baseline.elast
    A = Matrix{Float64}(I, 3, 3) + E
    dN = (A \ dE) / A
    mes_prod = context.baseline.mes_production
    dProd = (mes_prod .^ 2) .* (reshape(diag(dN), 3, 1) .- dN)
    dCost = reshape(diag(dE), 3, 1) .- dE
    return (model=context.model, parameter=parameter,
            mes_production=dProd, mes_cost=dCost,
            derivative_E=dE, derivative_N=dN,
            derivative_hessian=dH)
end

"""
    compute_all_sensitivities(data; step=1e-5)

Compute both model contexts ONCE; run independent one-at-a-time forward
perturbations, with analytic sensitivities used for verification.
"""
function compute_all_sensitivities(data::NamedTuple; step::Real=1e-5)
    contexts = (aggregate=prepare_sensitivity_context(:aggregate, data),
                disaggregated=prepare_sensitivity_context(:disaggregated, data))
    results = Dict{Symbol,Dict{Symbol,NamedTuple}}()
    analytics = Dict{Symbol,Dict{Symbol,NamedTuple}}()
    for mode in (:aggregate, :disaggregated)
        context = getproperty(contexts, mode)
        sub = Dict{Symbol,NamedTuple}()
        sub_analytic = Dict{Symbol,NamedTuple}()
        active = mode === :aggregate ? AGG_PARAMETERS : PARAMETERS
        for parameter in active
            sub[parameter] = compute_sensitivity(context, parameter; step=step)
            sub_analytic[parameter] = analytic_sensitivity(context, parameter)
        end
        results[mode] = sub
        analytics[mode] = sub_analytic
    end
    return (contexts=contexts, sensitivities=results,
            analytic=analytics, step=Float64(step))
end

function _paper_string(pair::Symbol, mode::Symbol, mes_type::Symbol,
                       parameter::Symbol)
    lookup_key = Symbol(string(mode), "_", string(mes_type))
    lookup = getproperty(getproperty(PUBLISHED, pair), lookup_key)
    return hasproperty(lookup, parameter) ? getproperty(lookup, parameter) : "–"
end

function _paper_precision_tolerance(value::AbstractString)
    value == "–" && return NaN
    parts = split(value, ".")
    decimals = length(parts) == 2 ? length(parts[2]) : 0
    # Integer entries 0 and 1 denote exact theoretical benchmark limits.
    return decimals == 0 ? 1e-7 : 0.5 * 10.0^(-decimals) + 1e-12
end

function _printed_match(value::Real, printed::AbstractString)
    printed == "–" && return false
    return abs(Float64(value) - parse(Float64, printed)) <
           _paper_precision_tolerance(printed)
end

function _entry(run, pair::Symbol, mode::Symbol, mes_type::Symbol,
                parameter::Symbol)
    rr = run.sensitivities[mode][parameter]
    row, col = pair === :HK ? (1, 3) : (1, 2)
    matrix = mes_type === :cost ? rr.mes_cost : rr.mes_production
    return matrix[row, col]
end

function _baseline_entry(run, pair::Symbol, mode::Symbol, mes_type::Symbol)
    row, col = pair === :HK ? (1, 3) : (1, 2)
    mes = getproperty(run.contexts, mode).baseline
    matrix = mes_type === :cost ? mes.mes_cost : mes.mes_production
    return matrix[row, col]
end

"""
    verify_sensitivity_results(run)

Report discrepancies from the *published* tables, which need not equal the
original source-code results. Check actual printed precision (e.g. "0.1"
is one decimal place).  Also report comparison of forward differences with
analytic derivatives and homogeneity sums for cost MES.
"""
function verify_sensitivity_results(run)
    printed_discrepancies = NamedTuple[]
    maximum_analytic_error = 0.0
    for mode in (:aggregate, :disaggregated)
        active = mode === :aggregate ? AGG_PARAMETERS : PARAMETERS
        for p in active
            forward = run.sensitivities[mode][p]
            analytic = run.analytic[mode][p]
            err = max(maximum(abs, forward.mes_cost .- analytic.mes_cost),
                      maximum(abs, forward.mes_production .- analytic.mes_production))
            maximum_analytic_error = max(maximum_analytic_error, err)
        end
        for (pair, _, _) in PAIRS
            for typ in (:cost, :production)
                benchmark = _baseline_entry(run, pair, mode, typ)
                printed_bench = getproperty(getproperty(PUBLISHED, pair).benchmark,
                                             Symbol(string(mode), "_", string(typ)))
                if !_printed_match(benchmark, printed_bench)
                    push!(printed_discrepancies,
                          (pair=pair, model=mode, type=typ, parameter=:benchmark,
                           computed=benchmark, published=printed_bench,
                           gap=benchmark - parse(Float64, printed_bench)))
                end
                for p in active
                    observed = _entry(run, pair, mode, typ, p)
                    printed = _paper_string(pair, mode, typ, p)
                    if !_printed_match(observed, printed)
                        push!(printed_discrepancies,
                              (pair=pair, model=mode, type=typ, parameter=p,
                               computed=observed, published=printed,
                               gap=observed - parse(Float64, printed)))
                    end
                end
            end
        end
    end
    cost_sums = Dict{Tuple{Symbol,Symbol},Float64}()
    for mode in (:aggregate, :disaggregated), (pair, row, col) in PAIRS
        active = mode === :aggregate ? AGG_PARAMETERS : PARAMETERS
        cost_sums[(mode, pair)] = sum(
            run.analytic[mode][p].mes_cost[row, col] for p in active)
    end
    return (printed_discrepancies=printed_discrepancies,
            max_forward_analytic_error=maximum_analytic_error,
            cost_derivative_sums=cost_sums)
end

function _csv_real(x::Real)
    return isfinite(x) ? @sprintf("%.17g", x) : ""
end

function _write_all_rows(path::AbstractString, run, mode::Symbol)
    context = getproperty(run.contexts, mode)
    active = mode === :aggregate ? AGG_PARAMETERS : PARAMETERS
    open(path, "w") do io
        println(io, "Model,Microelasticity,BenchmarkValue,PerturbedValue,FiniteDifferenceStep,MESType,FactorPair,BaselineMES,PerturbedMES,ComputedDerivative,AnalyticDerivative,PublishedDerivative,AbsoluteDiscrepancy,PrintedPrecisionMatch")
        for p in active
            fwd = run.sensitivities[mode][p]
            exact = run.analytic[mode][p]
            for (pair, row, col) in PAIRS, typ in (:cost, :production)
                baseline = typ === :cost ? context.baseline.mes_cost : context.baseline.mes_production
                perturbed = typ === :cost ? fwd.perturbed_cost : fwd.perturbed_production
                calc = typ === :cost ? fwd.mes_cost : fwd.mes_production
                deriv = typ === :cost ? exact.mes_cost : exact.mes_production
                published = _paper_string(pair, mode, typ, p)
                paper_value = published == "–" ? NaN : parse(Float64, published)
                gap = abs(calc[row, col] - paper_value)
                match_printed = _printed_match(calc[row, col], published)
                println(io, join((string(mode), string(p), _csv_real(fwd.base_value),
                                   _csv_real(fwd.perturbed_value), _csv_real(fwd.step),
                                   string(typ), string(pair), _csv_real(baseline[row, col]),
                                   _csv_real(perturbed[row, col]), _csv_real(calc[row, col]),
                                   _csv_real(deriv[row, col]), published,
                                   _csv_real(gap), string(match_printed)), ","))
            end
        end
    end
    return path
end

function _table_data(run, pair::Symbol; published::Bool=false)
    values = Matrix{String}(undef, 7, 5)
    values[1, :] = ["Microelasticity", "Aggregate Cost", "Aggregate Production",
                    "Disaggregated Cost", "Disaggregated Production"]
    for (j, (mode, typ)) in enumerate((( :aggregate, :cost),
                                     ( :aggregate, :production),
                                     ( :disaggregated, :cost),
                                     ( :disaggregated, :production)))
        if published
            values[2, j + 1] = getproperty(getproperty(PUBLISHED, pair).benchmark,
                                           Symbol(string(mode), "_", string(typ)))
        else
            values[2, j + 1] = @sprintf("%.2f", _baseline_entry(run, pair, mode, typ))
        end
    end
    values[2, 1] = "Benchmark MES"
    for (i, parameter) in enumerate(PARAMETERS)
        values[i + 2, 1] = getproperty(PARAM_LABELS, parameter)
        for (j, (mode, typ)) in enumerate((( :aggregate, :cost),
                                          ( :aggregate, :production),
                                          ( :disaggregated, :cost),
                                          ( :disaggregated, :production)))
            if mode === :aggregate && !(parameter in AGG_PARAMETERS)
                values[i + 2, j + 1] = "-"
            elseif published
                values[i + 2, j + 1] = _paper_string(pair, mode, typ, parameter)
            else
                values[i + 2, j + 1] = @sprintf("%.2f", _entry(run, pair, mode, typ, parameter))
            end
        end
    end
    return values
end

function _save_table_comparison(path::AbstractString, run, pair::Symbol)
    open(path, "w") do io
        println(io, "FactorPair,Microelasticity,Model,MESType,Computed,Published,RawGap,PrintedPrecisionMatch")
        for param in (:benchmark, PARAMETERS...)
            for mode in (:aggregate, :disaggregated), typ in (:cost, :production)
                if mode === :aggregate && param !== :benchmark && !(param in AGG_PARAMETERS)
                    continue
                end
                value = param === :benchmark ? _baseline_entry(run, pair, mode, typ) :
                        _entry(run, pair, mode, typ, param)
                label = if param === :benchmark
                    getproperty(getproperty(PUBLISHED, pair).benchmark,
                                Symbol(string(mode), "_", string(typ)))
                else
                    _paper_string(pair, mode, typ, param)
                end
                diff = value - parse(Float64, label)
                println(io, join((string(pair), string(param), string(mode), string(typ),
                                  _csv_real(value), label, _csv_real(diff),
                                  string(_printed_match(value, label))), ","))
            end
        end
    end
    return path
end

function _save_convergence(path::AbstractString, run;
                           steps=(1e-3, 1e-4, 1e-5, 1e-6))
    open(path, "w") do io
        println(io, "Model,Microelasticity,Step,Method,FactorPair,MESType,Derivative,AnalyticDerivative,AbsoluteError")
        for mode in (:aggregate, :disaggregated)
            context = getproperty(run.contexts, mode)
            active = mode === :aggregate ? AGG_PARAMETERS : PARAMETERS
            for p in active
                exact = run.analytic[mode][p]
                for step in steps, method in (:forward, :central)
                    numerical = compute_sensitivity(context, p;
                                                    step=step, method=method)
                    for (pair, row, col) in PAIRS, typ in (:cost, :production)
                        calc = typ === :cost ? numerical.mes_cost[row, col] :
                               numerical.mes_production[row, col]
                        target = typ === :cost ? exact.mes_cost[row, col] :
                                 exact.mes_production[row, col]
                        println(io, join((string(mode), string(p), _csv_real(step),
                                          string(method), string(pair), string(typ),
                                          _csv_real(calc), _csv_real(target),
                                          _csv_real(abs(calc - target))), ","))
                    end
                end
            end
        end
    end
    return path
end

function _save_diagnostics(path::AbstractString, run, report)
    open(path, "w") do io
        println(io, "Metric,Model,FactorPair,Value")
        println(io, "forward_analytic_max_abs_error,all,all,",
                _csv_real(report.max_forward_analytic_error))
        for ((mode, pair), value) in sort(collect(report.cost_derivative_sums);
                                          by=pair -> string(first(pair)))
            println(io, "sum_analytic_cost_sensitivities,$mode,$pair,", _csv_real(value))
        end
        for mode in (:aggregate, :disaggregated)
            ctx = getproperty(run.contexts, mode)
            println(io, "leontief_residual,$mode,all,",
                    _csv_real(ctx.equilibrium.leontief_residual))
            println(io, "hessian_symmetry_residual,$mode,all,",
                    _csv_real(maximum(abs, ctx.baseline_hessian .- transpose(ctx.baseline_hessian))))
        end
        for discrepancy in report.printed_discrepancies
            key = "published_gap_$(discrepancy.type)_$(discrepancy.parameter)"
            println(io, "$key,$(discrepancy.model),$(discrepancy.pair),",
                    _csv_real(discrepancy.gap))
        end
    end
    return path
end

function _summarytable_tex(path::AbstractString, text_matrix::Matrix{String})
    table = Table(Cell.(text_matrix); header=1)
    open(path, "w") do io
        show(io, MIME"text/latex"(), table)
        println(io)
    end
    return path
end

"""
    save_sensitivity_results(run, out_dir)

Save the source-MATLAB-based computed values, the printed-paper comparison,
finite-difference step convergence, diagnostics and SummaryTables LaTeX.
The publication tables use the COMPUTED values, not paper values pasted in.
"""
function save_sensitivity_results(run, out_dir::AbstractString)
    mkpath(out_dir)
    report = verify_sensitivity_results(run)
    files = String[]
    push!(files, _write_all_rows(joinpath(out_dir, "krusell_sens_aggregate.csv"),
                                 run, :aggregate))
    push!(files, _write_all_rows(joinpath(out_dir, "krusell_sens_disaggregated.csv"),
                                 run, :disaggregated))
    push!(files, _save_table_comparison(joinpath(out_dir, "krusell_sens_table_B3.csv"),
                                        run, :HK))
    push!(files, _save_table_comparison(joinpath(out_dir, "krusell_sens_table_B4.csv"),
                                        run, :LK))
    push!(files, _save_convergence(joinpath(out_dir, "krusell_sens_step_convergence.csv"), run))
    push!(files, _save_diagnostics(joinpath(out_dir, "krusell_sens_diagnostics.csv"),
                                   run, report))
    push!(files, _summarytable_tex(joinpath(out_dir, "krusell_sens_table_B3.tex"),
                                   _table_data(run, :HK)))
    push!(files, _summarytable_tex(joinpath(out_dir, "krusell_sens_table_B4.tex"),
                                   _table_data(run, :LK)))
    return files
end

"""
    run_tests(; data_dir=nothing)

Synthetic tests run without MAT files; data-backed tests also check the
published numerical anchors (including explicit source-vs-paper differences)
and the analytic homogeneity identity. No Julia code is imported dynamically.
"""
function run_tests(; data_dir::Union{Nothing,AbstractString}=nothing)
    @testset "Krusell_sens.jl" begin
        @testset "Input validation and exact perturbation mapping" begin
            @test baseline_calibration(:aggregate) == (theta1=0.67, theta2=1.67)
            @test baseline_calibration(:disaggregated) == THETA_BENCH
            @test_throws ArgumentError baseline_calibration(:other)
            @test_throws ArgumentError _parameter_value(:aggregate, :theta0)
            @test_throws ArgumentError _parameter_value(:disaggregated, :wrong)
            @test _parameter_indices(:theta0) == 66:66
            @test _parameter_indices(:theta1) == 196:325
            @test _parameter_indices(:theta2) == 133:195
            @test _parameter_indices(:theta3) == 3:65
            @test _parameter_indices(:theta4) == 68:130
            dummy = (elasticities=ones(Q), sigma=0.9)
            for p in PARAMETERS
                before = copy(dummy.elasticities)
                out = perturb_microelasticity(dummy, :disaggregated, p)
                ids = _parameter_indices(p)
                @test all(out.elasticities[ids] .== 1.00001)
                untouched = setdiff(1:Q, ids)
                @test all(out.elasticities[untouched] .== 1.0)
                @test dummy.elasticities == before
                @test out.sigma == (p === :theta0 ? 0.9 * 1.00001 : 0.9)
                @test isapprox(out.delta_micro, getproperty(THETA_BENCH, p) *
                                               (1.00001 - 1.0); rtol=0, atol=0)
            end
            @test_throws ArgumentError perturb_microelasticity(dummy, :aggregate, :theta0)
            @test_throws ArgumentError perturb_microelasticity(dummy, :disaggregated,
                                                                :theta1; relative_step=0.0)
            @test_throws ArgumentError perturb_microelasticity(dummy, :disaggregated,
                                                                :theta1; relative_step=-1.0)
        end
        @testset "Synthetic aggregate and disaggregated contexts" begin
            aggdata = KrusellAggregate._synthetic_data(20261112)
            disdata = KrusellDisaggregated._synthetic_data(20261112)
            ca = prepare_sensitivity_context(:aggregate, aggdata)
            cd = prepare_sensitivity_context(:disaggregated, disdata)
            for ctx in (ca, cd)
                @test size(ctx.equilibrium.omega) == (Q, Q)
                @test size(ctx.equilibrium.psi) == (Q, Q)
                @test length(ctx.equilibrium.lambda) == Q
                @test ctx.equilibrium.leontief_residual < 1e-10
                @test size(ctx.baseline.mes_production) == (3, 3)
                @test size(ctx.baseline_hessian) == (Q, Q)
                @test ctx.equilibrium.factor_ids == collect(FACTORS)
                @test all(isfinite, ctx.baseline.mes_cost)
            end
            for (ctx, p) in ((ca, :theta1), (ca, :theta2),
                             (cd, :theta0), (cd, :theta1), (cd, :theta2),
                             (cd, :theta3), (cd, :theta4))
                fd = compute_sensitivity(ctx, p; step=1e-5)
                ad = analytic_sensitivity(ctx, p)
                @test maximum(abs, fd.mes_cost .- ad.mes_cost) < 3e-5
                @test maximum(abs, fd.mes_production .- ad.mes_production) < 3e-5
                @test all(isfinite, fd.mes_production)
                @test all(isfinite, ad.mes_cost)
                cc = compute_sensitivity(ctx, p; step=1e-4, method=:central)
                @test maximum(abs, cc.mes_cost .- ad.mes_cost) < 3e-5
            end
            @test isapprox(analytic_sensitivity(ca, :theta1).mes_cost[1,3], 1.0;
                           atol=1e-10)
            @test isapprox(analytic_sensitivity(ca, :theta2).mes_cost[1,3], 0.0;
                           atol=1e-10)
        end
        @testset "Published precision matching and important discrepancies" begin
            @test _printed_match(0.094950644, "0.1")
            @test !_printed_match(0.017198137, "0.01")
            @test !_printed_match(0.66310442, "0.67")
            @test _printed_match(0.66310442, "0.66")
            @test _paper_string(:HK, :aggregate, :cost, :theta1) == "1"
            @test _paper_string(:LK, :disaggregated, :production, :theta4) == "0.07"
        end
        if data_dir !== nothing
            @testset "Official five-MAT-file sensitivity replication" begin
                data = load_sensitivity_data(data_dir)
                run = compute_all_sensitivities(data)
                ca = run.contexts.aggregate
                cd = run.contexts.disaggregated
                @test isapprox(ca.baseline.mes_production[1,2], 1.05075225135; atol=1e-8)
                @test isapprox(ca.baseline.mes_cost[1,2], 1.27514384713; atol=1e-8)
                @test isapprox(cd.baseline.mes_production[1,2], 0.93551963842; atol=1e-8)
                @test isapprox(cd.baseline.mes_cost[1,3], 0.64172737815; atol=1e-8)
                @test ca.equilibrium.leontief_residual < 1e-10
                @test cd.equilibrium.leontief_residual < 1e-10
                @test ca.equilibrium.factor_ids == collect(FACTORS)
                @test cd.equilibrium.factor_ids == collect(FACTORS)
                report = verify_sensitivity_results(run)
                @test report.max_forward_analytic_error < 3e-5
                for mode in (:aggregate, :disaggregated), pair in (:HK, :LK)
                    @test abs(report.cost_derivative_sums[(mode, pair)] - 1.0) < 1e-9
                end
                # Independent Python/SciPy references from original MATLAB expressions.
                @test isapprox(run.analytic[:aggregate][:theta1].mes_cost[1,2],
                               0.394856152873; atol=2e-7)
                @test isapprox(run.analytic[:aggregate][:theta2].mes_cost[1,2],
                               0.605143847127; atol=2e-7)
                @test isapprox(run.analytic[:disaggregated][:theta0].mes_cost[1,3],
                               0.130304790; atol=2e-6)
                @test isapprox(run.analytic[:disaggregated][:theta4].mes_cost[1,3],
                               0.088519557; atol=2e-6)
                @test isapprox(run.analytic[:disaggregated][:theta2].mes_cost[1,2],
                               0.459618286; atol=2e-6)
                @test isapprox(run.analytic[:disaggregated][:theta1].mes_production[1,2],
                               0.569564063; atol=2e-6)
                @test any(x -> x.pair === :LK && x.model === :aggregate &&
                               x.parameter === :theta2 && x.type === :cost,
                          report.printed_discrepancies)
                @test any(x -> x.pair === :LK && x.model === :disaggregated &&
                               x.parameter === :theta1 && x.type === :production,
                          report.printed_discrepancies)
                @test any(x -> x.pair === :HK && x.model === :disaggregated &&
                               x.parameter === :theta2 && x.type === :cost,
                          report.printed_discrepancies)
            end
        end
    end
    return nothing
end

function _timed_medians(f::Function; samples::Int=5)
    samples > 0 || throw(ArgumentError("samples must be positive"))
    f() # JIT warm-up outside measurement
    times = Float64[]
    allocations = Int[]
    for _ in 1:samples
        trial = @timed f()
        push!(times, 1000.0 * trial.time)
        push!(allocations, trial.bytes)
    end
    return (ms=median(times), bytes=round(Int, median(allocations)))
end

"""
    run_benchmark(data; samples=5, step=1e-5)

Compare one-time benchmark context builds with reuse of Omega/Psi/lambda
for all parameter perturbations, and analytic (no finite differences) checks.
Times exclude JIT compilation and report median of `samples` measurements.
"""
function run_benchmark(data::NamedTuple; samples::Int=5, step::Real=1e-5)
    @printf("\nSensitivity benchmark (Julia compiled; %d samples)\n", samples)
    @printf("%-44s %12s %17s\n", "Stage", "median ms", "median alloc B")
    for mode in (:aggregate, :disaggregated)
        build = () -> prepare_sensitivity_context(mode, data)
        one = _timed_medians(build; samples=samples)
        context = build()
        active = mode === :aggregate ? AGG_PARAMETERS : PARAMETERS
        forward = () -> foreach(p -> compute_sensitivity(context, p; step=step), active)
        exact = () -> foreach(p -> analytic_sensitivity(context, p), active)
        f = _timed_medians(forward; samples=samples)
        a = _timed_medians(exact; samples=samples)
        @printf("%-44s %12.3f %17d\n", "$mode context build", one.ms, one.bytes)
        @printf("%-44s %12.3f %17d\n", "$mode all forward differences", f.ms, f.bytes)
        @printf("%-44s %12.3f %17d\n", "$mode all analytic derivatives", a.ms, a.bytes)
    end
    print("No 329^3 covariance tensor is generated.\n")
    return nothing
end

"""
    run_replication(data_dir; out_dir=..., step=1e-5, bench=false)

Read official MAT data, run all tests including independent reference anchors,
calculate Tables B.3/B.4, export CSV and SummaryTables LaTeX, and report
source-code-vs-paper discrepancies without changing the calibration.
"""
function run_replication(data_dir::AbstractString;
                         out_dir::AbstractString=joinpath(@__DIR__, "output_krusell_sens"),
                         step::Real=1e-5, bench::Bool=false)
    run_tests(; data_dir=data_dir)
    data = load_sensitivity_data(data_dir)
    run = compute_all_sensitivities(data; step=step)
    report = verify_sensitivity_results(run)
    println("\nKrusell MES sensitivity replication (MATLAB forward difference)")
    @printf("step h=%.9g; denominator = theta_k*((1+h)-1)\n", step)
    for (pair, _, _) in PAIRS
        println("\nTable ", pair === :HK ? "B.3 (High-skilled / Capital)" : "B.4 (Low-skilled / Capital)")
        display(_table_data(run, pair))
    end
    @printf("\nLargest |forward - analytic| (all 3x3 cells): %.4e\n",
            report.max_forward_analytic_error)
    for mode in (:aggregate, :disaggregated), (pair, _, _) in PAIRS
        @printf("Cost homogeneity sum, %-14s %s: %.12f\n", string(mode),
                string(pair), report.cost_derivative_sums[(mode, pair)])
    end
    println("\nPublished-cell discrepancies (evaluated at printed precision):")
    if isempty(report.printed_discrepancies)
        println("  None")
    else
        for entry in report.printed_discrepancies
            @printf("  %-2s %-14s %-10s %-10s computed=% .9f published=%s\n",
                    string(entry.pair), string(entry.model), string(entry.type),
                    string(entry.parameter), entry.computed, entry.published)
        end
    end
    files = save_sensitivity_results(run, out_dir)
    for file in files
        isfile(file) || throw(ErrorException("Output was not generated: $file"))
        println("Wrote: ", abspath(file))
    end
    if bench
        run_benchmark(data; step=step)
    end
    return run
end

function _cli(args::Vector{String})
    data_dir = nothing
    out_dir = joinpath(@__DIR__, "output_krusell_sens")
    step = 1e-5
    bench = false
    test_only = false
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--test-only"
            test_only = true
        elseif arg == "--bench"
            bench = true
        elseif arg in ("--data-dir", "--out-dir", "--step")
            i < length(args) || throw(ArgumentError("Missing value following $arg"))
            value = args[i + 1]
            if arg == "--data-dir"
                data_dir = value
            elseif arg == "--out-dir"
                out_dir = value
            else
                step = parse(Float64, value)
            end
            i += 1
        else
            throw(ArgumentError("Unknown argument $arg"))
        end
        i += 1
    end
    if test_only
        run_tests()
        return nothing
    end
    data_dir === nothing && throw(ArgumentError(
        "Pass --data-dir PATH with the original five MAT files, or use --test-only"))
    run_replication(data_dir; out_dir=out_dir, step=step, bench=bench)
    return nothing
end

end # module KrusellSensitivities

if abspath(PROGRAM_FILE) == (@__FILE__)
    KrusellSensitivities._cli(ARGS)
end
