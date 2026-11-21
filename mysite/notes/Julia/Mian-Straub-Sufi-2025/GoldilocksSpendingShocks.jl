module GoldilocksSpendingShocks

using LinearAlgebra
using Printf
using SummaryTables
using ..GoldilocksCore
using ..GoldilocksHouseholds
using ..GoldilocksCalibration
using ..GoldilocksDeficitDebt
using ..GoldilocksTransition

export FIGURE10_PANELS, FIGURE10_MODELS, SpendingExperiment,
       experiment_config, spending_experiment,
       build_transition_problem, solve_spending_experiment,
       run_figure10_experiments, write_transition_tsv,
       make_figure10_tables, long_run_multiplier,
       compare_figure10_python

const FIGURE10_PANELS=(:A,:B,:C,:D)
const FIGURE10_MODELS=Dict(:A=>(:bu,:bu_log,:olg,:ha),
                           :B=>(:bu,:bu_log,:olg,:ha),
                           :C=>(:bu,:bu_log,:olg,:ha),
                           :D=>(:bu,:bu_log))

"""Exact four experiments of the original Python `spending_shocks.py`."""
function experiment_config(panel::Symbol;T::Int=1000)
    T>0 || throw(ArgumentError("positive time horizon required"))
    panel in FIGURE10_PANELS || throw(ArgumentError("unknown Figure 10 panel $panel"))
    if panel===:A
        return (;panel,country=:us,kind=:deficit,permanent=0.0001,
                transitory=0.005,rho=0.9,psi=0.0,psi_terminal=0.0,
                models=FIGURE10_MODELS[panel],T,years=100)
    elseif panel===:B
        return (;panel,country=:jp,kind=:deficit,permanent=0.001,
                transitory=0.005,rho=0.9,psi=0.0,psi_terminal=0.0,
                models=FIGURE10_MODELS[panel],T,years=100)
    elseif panel===:C
        return (;panel,country=:us,kind=:deficit,permanent=0.005,
                transitory=0.03,rho=0.9,psi=0.018,psi_terminal=0.018,
                models=FIGURE10_MODELS[panel],T,years=100)
    else
        return (;panel,country=:us,kind=:psi,permanent=NaN,
                transitory=NaN,rho=NaN,psi=NaN,psi_terminal=-0.023,
                models=FIGURE10_MODELS[panel],T,years=40)
    end
end

function _replace_core(core::CoreParams; fiscal=core.fiscal,
                       wage=core.wage,monetary=core.monetary)
    return CoreParams(;production=core.production,wage,monetary,fiscal,
                       mu=core.mu,delta_b=core.delta_b)
end

function _with_fiscal(core::CoreParams;psi,deficit_ss,B_ss)
    f=core.fiscal
    fiscal=FiscalParams(;G=f.G,periods=f.periods,
              adjust_tax_rate=f.adjust_tax_rate,psi,deficit_ss,B_ss,
              tax_shock=f.tax_shock,tau_slope=f.tau_slope)
    return _replace_core(core;fiscal)
end

function _with_miu(core::CoreParams, i_initial)
    m=core.monetary
    monetary=MonetaryParams(;phi_pi=m.phi_pi,real_rate=m.real_rate,
        pi_trend=m.pi_trend,real_growth=m.real_growth,
        i_shock=m.i_shock,zlb=true,i_zlb=i_initial)
    w=core.wage
    wage=WageParams(;kappa=5*w.kappa,beta=w.beta,
        survival=w.survival,labor_shifter=w.labor_shifter,
        lag=w.lag,forward=w.forward,frisch=w.frisch)
    return _replace_core(core;wage,monetary)
end

struct SpendingExperiment{C,H,S,I,E}
    panel::Symbol
    model::Symbol
    mode::Symbol
    core_initial::C
    core_terminal::C
    household::H
    state_initial::S
    initial::I
    terminal::E
    deficit_path::Vector{Float64}
    psi_path::Vector{Float64}
    shock_path::Vector{Float64}
    equilibrium_error::Float64
end

"""Solve the terminal GE exactly, with the author's ZLB-specific targets."""
function _terminal_eq(core::CoreParams,hh,initial::NamedTuple;
                      tol::Real=2e-8)
    zlb=core.monetary.zlb
    fields=zlb ? (:B,:Y,:r_ante,:piw) : (:B,:Y,:r_ante)
    targets=zlb ? (:asset_mkt,:fisher,:deficit_diff,:piw_cond) :
                  (:asset_mkt,:labor_foc,:deficit_diff)
    x0=[Float64(getproperty(initial,k)) for k in fields]
    term_residual = v -> begin
        state=merge((;Y=initial.Y,B=initial.B,
                      r_ante=initial.r_ante,piw=initial.piw),
                    NamedTuple{fields}(Tuple(v)))
        eq=evaluate_steady_state(state,core,hh)
        [getproperty(eq,k) for k in targets]
    end
    sol=GoldilocksDeficitDebt._solve_residual(term_residual,x0;
                                               tol=tol,maxiter=85)
    state=merge((;Y=initial.Y,B=initial.B,
                  r_ante=initial.r_ante,piw=initial.piw),
                NamedTuple{fields}(Tuple(sol.x)))
    eq=evaluate_steady_state(state,core,hh)
    err=maximum(abs(getproperty(eq,k)) for k in targets)
    err<=tol*2 || error("terminal steady state residual $err for $(targets)")
    return (;eq,state,error=err,targets)
end

"""Recreate author Figure 10 initial and terminal economies without reading
any saved transition results; `mode` controls only initial household calibration.
"""
function spending_experiment(panel::Symbol,model::Symbol;
                             mode::Symbol=:official,T::Int=1000,
                             terminal_tol::Real=2e-8)
    cfg=experiment_config(panel;T)
    model in cfg.models || throw(ArgumentError("model $model excluded from Figure 10 panel $panel"))
    mode in (:official,:native) || throw(ArgumentError("mode must be official or native"))
    input=figure8_calibration(cfg.country,model;mode)
    hh=input.household
    core=input.core
    if panel===:D
        # As the original script: fix nominal i at the PRE-EXPERIMENT US
        # steady-state value, turn on zlb, multiply original kappa by five.
        core=_with_miu(core,input.anchor.i)
        # The Python source uses the BU-QUADRATIC anchor for BOTH Panel-D
        # households, not each household's independently saved baseline.
        bu_anchor=model===:bu ? input.anchor :
                  figure8_calibration(:us,:bu;mode).anchor
        initial_psi=-bu_anchor.deficit/bu_anchor.B
        B_ss=0.0
        deficit_ss=0.0
        shock=fill(-0.023-initial_psi,T)
        psi_path=fill(-0.023,T)
        deficit_path=fill(0.0,T)
    else
        initial_psi=cfg.psi
        B_ss=input.anchor.B
        deficit_ss=input.anchor.deficit
        shock=transition_shocks(T;permanent=cfg.permanent,
                    transitory=cfg.transitory,rho=cfg.rho)
        psi_path=fill(initial_psi,T)
        deficit_path=deficit_ss .+ shock
    end
    core_ini=_with_fiscal(core;psi=initial_psi,deficit_ss,B_ss)
    initial=evaluate_steady_state(input.setup.state,core_ini,hh)
    initial_err=maximum(abs.((initial.asset_mkt,initial.goods_mkt,
                                      initial.fisher,initial.piw_cond)))
    initial_err<1e-4 || error("invalid Figure 10 initial steady-state residual $initial_err")
    # Python uses shock[-1] for the terminal CALIBRATION, even if there is a
    # tiny transitory tail, and subtracts that same value from transition inputs.
    terminal_psi=panel===:D ? initial_psi+shock[end] : initial_psi
    terminal_def=panel===:D ? deficit_ss : deficit_ss+shock[end]
    core_term=_with_fiscal(core;psi=terminal_psi,
                           deficit_ss=terminal_def,B_ss)
    term=_terminal_eq(core_term,hh,initial;tol=terminal_tol)
    return SpendingExperiment(panel,model,mode,core_ini,core_term,hh,
         input.setup.state,initial,term.eq,deficit_path,psi_path,shock,term.error)
end

"""One terminal-anchored full-horizon transition problem."""
function build_transition_problem(e::SpendingExperiment)
    return TransitionProblem(e.core_terminal,e.household,e.initial,e.terminal,
             e.deficit_path,e.psi_path,e.model)
end

"""The Figure 10 permanent primary-deficit long-run output multiplier."""
function long_run_multiplier(e::SpendingExperiment)
    den=e.terminal.deficit-e.initial.deficit
    abs(den)>1e-15 || throw(DomainError(den,"zero long-run fiscal adjustment"))
    return (e.terminal.Y-e.initial.Y)/den
end

"""Solve the COMPLETE nonlinear model rather than interpolating steady states.

This is compute-intensive at T=1000, especially the HA EGM distribution
transition. Nonconvergence is propagated as an error (not silently omitted).
"""
function solve_spending_experiment(e::SpendingExperiment;
                                   tol::Real=1e-5,maxit::Int=50,
                                   verbose::Bool=true)
    problem=build_transition_problem(e)
    solution=solve_transition(problem;tol,maxit,verbose)
    validation=check_transition(solution;tol,goods_tol=tol)
    return (;experiment=e,solution,validation,
             multiplier=(e.panel in (:A,:B) ? long_run_multiplier(e) : NaN))
end

"""Export an actually solved transition, including source units and residuals."""
function write_transition_tsv(result;directory::AbstractString=@__DIR__)
    e=result.experiment
    path=joinpath(directory,"figure10_$(e.panel)_$(e.model)_$(e.mode).tsv")
    open(path,"w") do io
        println(io,"t0\tdeficit\tB\tB_annual\tY\tpi\tpiw\tr\tr_ante\ti\tC\tA\tgoods_mkt\tasset_mkt\tdeficit_diff")
        for (t,z) in enumerate(result.solution.periods)
            vals=(t-1,z.deficit,z.B,z.B_annual,z.Y,z.pi,z.piw,
                  z.r,z.r_ante,z.i,z.C,z.A,z.goods_mkt,z.asset_mkt,z.deficit_diff)
            println(io,join(string.(vals),'\t'))
        end
    end
    return path
end

"""Run original 14 model-experiment paths; preserve exact case-specific errors."""
function run_figure10_experiments(;mode::Symbol=:official,T::Int=1000,
                panels=FIGURE10_PANELS,models=nothing,
                directory::AbstractString=@__DIR__,
                tol::Real=1e-5,maxit::Int=50,verbose::Bool=true)
    results=Dict{Tuple{Symbol,Symbol},Any}()
    for panel in panels
        cfg=experiment_config(panel;T)
        for model in (models===nothing ? cfg.models : models)
            model in cfg.models || continue
            @info "Figure 10 model experiment" panel model mode T
            exp=spending_experiment(panel,model;mode,T)
            out=solve_spending_experiment(exp;tol,maxit,verbose)
            write_transition_tsv(out;directory)
            results[(panel,model)]=out
        end
    end
    return results
end

"""Parse an original Python transition value, preserving source missingness.

The author's non-ZLB panels A and C do not report a time-varying `piw`:
those TSV cells are intentionally blank. They are *not* zero and must not
be imputed from Julia's steady-state value. Any other blank is an error.
"""
function _figure10_python_value(raw::AbstractString, panel::Symbol,
                                model::Symbol, t::Int, field::Symbol)
    cell=strip(raw)
    if isempty(cell)
        field===:piw && panel in (:A,:C) && return missing
        throw(ArgumentError("unexpected missing original Python Figure 10 value: " *
                            "panel=$panel model=$model t=$t field=$field"))
    end
    value=tryparse(Float64,cell)
    value===nothing && throw(ArgumentError(
        "unparseable original Python Figure 10 value: " *
        "panel=$panel model=$model t=$t field=$field value=$(repr(cell))"))
    isfinite(value) || throw(ArgumentError(
        "nonfinite original Python Figure 10 value: " *
        "panel=$panel model=$model t=$t field=$field value=$value"))
    return value
end

"""Compare solved model dates to original Python numerical transition paths.

For non-ZLB panels A/C the original Python TSV omits `piw` values; these
comparisons are explicitly marked `missing` in the output and excluded from
per-variable error metrics. All other values must be present and finite.
`selected` counts compared dates; `diagnostics` counts only observed values.
"""
function compare_figure10_python(results::AbstractDict;
                                 source::AbstractString=joinpath(@__DIR__,
                                     "figure10_original_python_reference.tsv"),
                                 directory::AbstractString=@__DIR__,
                                 mode::Symbol=:official)
    isfile(source) || throw(ArgumentError("original Figure 10 Python reference missing: $source"))
    selected=Dict{Tuple{Symbol,Symbol},Int}()
    fields=(:deficit,:B,:B_annual,:Y,:C,:A,:r,:r_ante,:pi,:piw,:i,
            :goods_mkt,:asset_mkt,:deficit_diff)
    maxima=Dict{Tuple{Symbol,Symbol,Symbol},Float64}()
    squares=Dict{Tuple{Symbol,Symbol,Symbol},Float64}()
    count=Dict{Tuple{Symbol,Symbol,Symbol},Int}()
    missing_count=Dict{Tuple{Symbol,Symbol,Symbol},Int}()
    path=joinpath(directory,"figure10_python_julia_parity_$(mode).tsv")
    open(path,"w") do dest
        println(dest,"panel\tmodel\tt0\tvariable\tpython\tjulia\tabsolute_error")
        open(source,"r") do src
            labels=Symbol.(split(readline(src),'\t';keepempty=true))
            column=Dict(s=>i for (i,s) in enumerate(labels))
            for name in (:panel,:model,:t0,fields...)
                haskey(column,name) || error("source TSV is missing $name")
            end
            for row in eachline(src)
                parts=split(row,'\t';keepempty=true)
                length(parts)==length(labels) || throw(ArgumentError(
                    "malformed original Python Figure 10 TSV row: expected " *
                    "$(length(labels)) columns, got $(length(parts))"))
                panel=Symbol(parts[column[:panel]])
                model=Symbol(parts[column[:model]])
                key=(panel,model)
                haskey(results,key) || continue
                t=parse(Int,parts[column[:t0]])
                result=results[key]
                if t+1>length(result.solution.periods) || t<0
                    throw(DimensionMismatch("Python time index not in Julia solution: $key, $t"))
                end
                selected[key]=get(selected,key,0)+1
                zz=result.solution.periods[t+1]
                for field in fields
                    v=Float64(getproperty(zz,field))
                    isfinite(v) || throw(ArgumentError(
                        "nonfinite Julia Figure 10 value: $key t=$t field=$field"))
                    expected=_figure10_python_value(
                        parts[column[field]],panel,model,t,field)
                    kk=(panel,model,field)
                    if ismissing(expected)
                        missing_count[kk]=get(missing_count,kk,0)+1
                        println(dest,join(string.((panel,model,t,field,
                                                   "missing",v,"missing")),'\t'))
                        continue
                    end
                    err=abs(v-expected)
                    maxima[kk]=max(get(maxima,kk,0.0),err)
                    squares[kk]=get(squares,kk,0.0)+err^2
                    count[kk]=get(count,kk,0)+1
                    println(dest,join(string.((panel,model,t,field,expected,v,err)),'\t'))
                end
            end
        end
    end
    for (key,out) in results
        get(selected,key,0)==length(out.solution.periods) ||
            error("Python time series for $key incomplete; matched $(get(selected,key,0))")
    end
    diagnostics=Dict{Tuple{Symbol,Symbol,Symbol},NamedTuple}()
    for (key,mx) in maxima
        diagnostics[key]=(;max_error=mx,rmse=sqrt(squares[key]/count[key]),
                           matched=count[key])
    end
    return (;path,diagnostics,selected,missing_count)
end

"""LaTeX tables using the existing SummaryTables.jl API (no raw TeX faking)."""
function make_figure10_tables(results::AbstractDict;
                              directory::AbstractString=@__DIR__,
                              mode::Symbol=:official,parity=nothing)
    n=length(results)
    policies=Matrix{String}(undef,n+1,7)
    policies[1,:]=["Panel","Country","Model","Type","Initial ψ","Terminal ψ","Horizon"]
    outcomes=Matrix{String}(undef,n+1,8)
    outcomes[1,:]=["Panel","Model","B initial","B terminal","Def initial","Def terminal","Y terminal","Max GE residual"]
    mul=Matrix{String}(undef,9,3)
    mul[1,:]=["Country","Model","Long-run dY/dz"]
    j=0
    for (k,out) in sort(collect(results);by=x->(string(x.first[1]),string(x.first[2])))
        j+=1
        e=out.experiment
        cfg=experiment_config(e.panel;T=length(e.shock_path))
        vals=(string(e.panel),string(cfg.country),string(e.model),string(cfg.kind),
              @sprintf("%.8g",e.core_initial.fiscal.psi),
              @sprintf("%.8g",e.core_terminal.fiscal.psi),string(cfg.T))
        for c in 1:7
            policies[j+1,c]=vals[c]
        end
        dat=(string(e.panel),string(e.model),@sprintf("%.8f",e.initial.B),
             @sprintf("%.8f",e.terminal.B),@sprintf("%.8f",e.initial.deficit),
             @sprintf("%.8f",e.terminal.deficit),@sprintf("%.8f",e.terminal.Y),
             @sprintf("%.5g",out.validation.max_target_residual))
        for c in 1:8
            outcomes[j+1,c]=dat[c]
        end
    end
    for (idx,(country,model)) in enumerate(Iterators.product((:us,:jp),(:bu,:bu_log,:olg,:ha)))
        key=(country===:us ? :A : :B,model)
        if haskey(results,key)
            mul[idx+1,:]=[string(country),string(model),
                          @sprintf("%.8f",results[key].multiplier)]
        else
            mul[idx+1,:]=[string(country),string(model),"Not computed"]
        end
    end
    accuracy=Matrix{String}(undef,n+1,6)
    accuracy[1,:]=["Panel","Model","Max target","Max goods","Python max B","Python max deficit"]
    for (j,(key,out)) in enumerate(sort(collect(results);by=x->(string(x.first[1]),string(x.first[2]))))
        e=out.experiment
        maxb=parity===nothing ? NaN :
             haskey(parity.diagnostics,(e.panel,e.model,:B)) ?
             parity.diagnostics[(e.panel,e.model,:B)].max_error : NaN
        maxd=parity===nothing ? NaN :
             haskey(parity.diagnostics,(e.panel,e.model,:deficit)) ?
             parity.diagnostics[(e.panel,e.model,:deficit)].max_error : NaN
        vals=(string(e.panel),string(e.model),
              @sprintf("%.5g",out.validation.max_target_residual),
              @sprintf("%.5g",out.validation.max_goods_residual),
              @sprintf("%.5g",maxb),@sprintf("%.5g",maxd))
        for c in 1:6
            accuracy[j+1,c]=vals[c]
        end
    end
    tables=Dict(:policy=>policies,:outcomes=>outcomes,:multipliers=>mul,
                :accuracy=>accuracy)
    paths=Dict{Symbol,String}()
    for (name,mat) in tables
        path=joinpath(directory,"figure10_$(name)_$(mode).tex")
        open(path,"w") do io
            show(io,MIME"text/latex"(),Table(Cell.(mat);header=1))
        end
        paths[name]=path
    end
    return paths
end

end # module GoldilocksSpendingShocks
