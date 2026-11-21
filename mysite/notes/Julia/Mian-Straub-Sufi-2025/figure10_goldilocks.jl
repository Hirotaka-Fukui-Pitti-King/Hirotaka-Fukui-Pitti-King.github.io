# Figure 10, Mian, Straub and Sufi (2025), AER pp. 4284-4286.
# Full non-linear 1000-date transition; not interpolated steady states.
# Project-root PowerShell:
#   $env:GOLDILOCKS_FIG10_MODE = "official"  # or native, both
#   julia --project=. .\figure10_goldilocks.jl
# No GitHub Actions are used or triggered by this script.

include(joinpath(@__DIR__,"GoldilocksCore.jl"))
include(joinpath(@__DIR__,"GoldilocksHouseholds.jl"))
include(joinpath(@__DIR__,"GoldilocksCalibration.jl"))
include(joinpath(@__DIR__,"GoldilocksDeficitDebt.jl"))
include(joinpath(@__DIR__,"GoldilocksTransition.jl"))
include(joinpath(@__DIR__,"GoldilocksSpendingShocks.jl"))
include(joinpath(@__DIR__,"GoldilocksStyle.jl"))

using CairoMakie
using Printf
using .GoldilocksTransition
using .GoldilocksSpendingShocks
using .GoldilocksStyle

const FIG10_TITLES = (
    :A=>"A. United States: small deficit shock",
    :B=>"B. Japan: small deficit shock",
    :C=>"C. United States: large deficit shock",
    :D=>"D. United States: nominal-debt-growth rule",
)
const FIG10_VARS=(:deficit,:B_annual,:Y)
const FIG10_YLABELS=("Primary deficit (% initial GDP)",
                     "Government debt (% initial GDP)",
                     "GDP (% initial GDP)")

"""Render exactly 4 rows x 3 columns, with original models by panel."""
function plot_figure10(results::AbstractDict,mode::Symbol;
                       directory::AbstractString=@__DIR__)
    fig=with_theme(publication_theme(;fontsize=12)) do
        f=Figure(size=(1620,1530),fontsize=13)
        for (row,panel) in enumerate(FIGURE10_PANELS)
            cfg=experiment_config(panel)
            for (col,var) in enumerate(FIG10_VARS)
                ax=Axis(f[row,col],title=(col==1 ? FIG10_TITLES[row].second : ""),
                        xlabel="Time (years)",ylabel=FIG10_YLABELS[col])
                models=cfg.models
                base=results[(panel,first(models))]
                first_eq=base.experiment.initial
                hlines!(ax,[100*getproperty(first_eq,var)];
                        color=:black,linewidth=1.5)
                for model in models
                    result=results[(panel,model)]
                    style=model_style(model)
                    periods=result.solution.periods
                    years=min(cfg.years,length(periods))
                    x=collect(0:years-1)
                    y=[100*getproperty(periods[t],var) for t in 1:years]
                    lines!(ax,x,y;color=style.color,
                           linestyle=style.linestyle,linewidth=2.1,
                           label=style.label)
                    terminal=100*getproperty(result.experiment.terminal,var)
                    hlines!(ax,[terminal];color=style.color,
                            linestyle=:dot,linewidth=0.6)
                end
                col==1 && axislegend(ax;position=:rb,labelsize=10,
                                      framevisible=false)
            end
        end
        f
    end
    stem="figure10_free_lunch_$(mode)"
    pdf=joinpath(directory,stem*".pdf")
    png=joinpath(directory,stem*".png")
    save(pdf,fig)
    save(png,fig;px_per_unit=2)
    return (;pdf,png)
end

"""Independently solve all 14 transition paths, output TSV and SummaryTables."""
function run_figure10(;mode::Symbol=:official,T::Int=1000,
                      directory::AbstractString=@__DIR__,
                      tol::Real=1e-5,maxit::Int=50)
    mode in (:official,:native) || throw(ArgumentError("invalid Figure 10 mode"))
    if T!=1000
        @warn "Non-original transition horizon: NOT a full Figure 10 replication" T
    end
    results=run_figure10_experiments(;mode,T,directory,tol,maxit)
    source=joinpath(directory,"figure10_original_python_reference.tsv")
    parity= if isfile(source) && T==1000
        compare_figure10_python(results;source,directory,mode)
    else
        @warn "Python/Julia Figure 10 parity not established: source absent or non-original horizon" source T
        nothing
    end
    if parity!==nothing
        for panel in FIGURE10_PANELS, model in FIGURE10_MODELS[panel]
            v=parity.diagnostics[(panel,model,:B)]
            d=parity.diagnostics[(panel,model,:deficit)]
            @info "Original Python / Julia Figure 10 parity" panel model mode matched=v.matched max_debt_error=v.max_error max_deficit_error=d.max_error
        end
    end
    tables=make_figure10_tables(results;mode,directory,parity)
    plot=plot_figure10(results,mode;directory)
    return (;results,tables,plot)
end

if abspath(PROGRAM_FILE)==@__FILE__
    mode=Symbol(get(ENV,"GOLDILOCKS_FIG10_MODE","official"))
    T=parse(Int,get(ENV,"GOLDILOCKS_FIG10_T","1000"))
    if mode===:both
        official=run_figure10(;mode=:official,T)
        native=run_figure10(;mode=:native,T)
        println("Figure 10 official PDF: ",official.plot.pdf)
        println("Figure 10 native PDF: ",native.plot.pdf)
    else
        result=run_figure10(;mode,T)
        println("Figure 10 PDF: ",result.plot.pdf)
        println("Figure 10 PNG: ",result.plot.png)
        for (k,v) in result.tables
            println("Figure 10 table ",k,": ",v)
        end
    end
end
