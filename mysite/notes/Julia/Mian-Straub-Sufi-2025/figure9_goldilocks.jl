# Figure 9, Mian–Straub–Sufi (2025), AER pp. 4283–4284.
# Five nested BU-log COUNTERFACTUALS, not five household models.
# From the project root (Windows PowerShell):
#   $env:GOLDILOCKS_FIG9_MODE = "official" # or native / both
#   julia --project=. .\figure9_goldilocks.jl
# No GitHub Actions, no new folders.

include(joinpath(@__DIR__,"GoldilocksCore.jl"))
include(joinpath(@__DIR__,"GoldilocksHouseholds.jl"))
include(joinpath(@__DIR__,"GoldilocksCalibration.jl"))
include(joinpath(@__DIR__,"GoldilocksDeficitDebt.jl"))
include(joinpath(@__DIR__,"GoldilocksFiscalSpace.jl"))
include(joinpath(@__DIR__,"GoldilocksStyle.jl"))

using CairoMakie
using Printf
using .GoldilocksCore
using .GoldilocksCalibration
using .GoldilocksDeficitDebt
using .GoldilocksFiscalSpace
using .GoldilocksStyle

const FIGURE9_STYLES = (
    (color=OKABE_ITO.blue,         linestyle=:solid,   width=3.2),
    (color=OKABE_ITO.vermilion,    linestyle=:dash,    width=2.5),
    (color=OKABE_ITO.bluish_green, linestyle=:dashdot, width=2.5),
    (color=OKABE_ITO.orange,       linestyle=:dot,     width=2.5),
    (color=OKABE_ITO.reddish_purple,linestyle=:solid, width=3.2),
)

"""Draw one branch, splitting on missing roots and never interpolating failures."""
function _plot_figure9_branch!(ax, curve::DeficitDebtCurve,
                                branch::Symbol, style)
    X=Float64[]; Z=Float64[]
    function flush_segment!()
        if length(X)>1
            lines!(ax,X,Z;color=style.color,linestyle=style.linestyle,
                   linewidth=style.width)
        end
        empty!(X);empty!(Z)
    end
    for p in curve.records
        if ismissing(p) || p.branch!==branch
            flush_segment!()
        else
            push!(X,100 * p.B/curve.periods)
            push!(Z,100 * p.deficit)
        end
    end
    flush_segment!()
    return nothing
end

"""Render the original one-panel, five-scenario Figure 9 in CairoMakie."""
function plot_figure9(curves::AbstractDict,mode::Symbol;
                      directory::AbstractString=@__DIR__)
    fig=with_theme(publication_theme(;fontsize=17)) do
        f=Figure(size=(1030,650),fontsize=17)
        ax=Axis(f[1,1],xlabel="Debt to GDP (percent)",
                ylabel="Primary deficit to GDP (percent)",
                xticks=0:100:400,yticks=0:1:5)
        handles=Any[]
        for (j,key) in enumerate(FIGURE9_SCENARIOS)
            curve=curves[key]; style=FIGURE9_STYLES[j]
            _plot_figure9_branch!(ax,curve,:zlb,style)
            _plot_figure9_branch!(ax,curve,:flex,style)
            push!(handles,lines!(ax,[NaN,NaN],[NaN,NaN];
                    color=style.color,linestyle=style.linestyle,
                    linewidth=style.width))
        end
        # The author's two black reference markers are BU-QUADRATIC steady
        # states even though the five curves are BU-log. Preserve this detail.
        for country in (:us,:jp)
            input=figure8_calibration(country,:bu;mode)
            eq=input.anchor
            scatter!(ax,[100*eq.B/input.setup.periods],
                         [100*eq.deficit];color=OKABE_ITO.black,
                         markersize=10)
        end
        hlines!(ax,[0.0];color=:gray60,linestyle=:dot,linewidth=1)
        xlims!(ax,0,430);ylims!(ax,0,5.2)
        Legend(f[1,1],handles,collect(FIGURE9_LABELS);
               framevisible=false,tellwidth=false,tellheight=false,
               halign=:right,valign=:top,
               margin=(10,12,12,3),labelsize=14)
        f
    end
    stem="figure9_japanese_fiscal_space_$(mode)"
    pdf=joinpath(directory,stem*".pdf")
    png=joinpath(directory,stem*".png")
    save(pdf,fig)
    save(png,fig;px_per_unit=3)
    return (;pdf,png,figure=fig)
end

"""Create the Native or Official Figure 9, numerics, tables and parity audit."""
function run_figure9(;mode::Symbol=:official,n_r::Int=120,n_Y::Int=300,
                      directory::AbstractString=@__DIR__,strict::Bool=true)
    mode in (:official,:native) || throw(ArgumentError("invalid Figure 9 mode"))
    @info "Re-solving Figure 9 sequential counterfactual calibration" mode
    scenarios=figure9_scenarios(;mode)
    calfile=write_figure9_calibration(scenarios;directory)
    for key in FIGURE9_SCENARIOS
        s=scenarios[key]
        @info "Figure 9 calibrated counterfactual" key mode beta=s.household.beta rho=(-log(s.household.beta)) slope=s.household.vp_slope Y=s.setup.state.Y inner_residual=s.inner_residual outer_residual=s.outer_residual
    end
    curves=Dict{Symbol,DeficitDebtCurve}()
    for key in FIGURE9_SCENARIOS
        s=scenarios[key]
        @info "Solving Figure 9 locus" key mode n_r n_Y
        curve=figure9_curve(s;n_r,n_Y)
        for issue in curve.issues
            @warn "Figure 9 grid stopped/unsolved" key mode branch=issue.branch index=issue.index input=issue.grid_value reason=issue.reason
        end
        nf=count(p->!ismissing(p)&&p.branch===:flex,curve.records)
        nz=count(p->!ismissing(p)&&p.branch===:zlb,curve.records)
        if strict && (nf!=n_r || nz<2)
            error("Figure 9 $key incomplete: flex $nf/$n_r, ZLB $nz")
        end
        path=write_figure9_curve(key,curve;directory)
        mx=grid_maximum(curve)
        @info "Figure 9 grid maximum" key mode deficit_percent=(100*mx.maximum_deficit) debt_percent=(100*mx.debt_at_maximum/curve.periods) count_flex=nf count_zlb=nz file=path
        curves[key]=curve
    end
    source=joinpath(directory,"figure9_original_python_reference.tsv")
    parity=nothing
    if isfile(source) && n_r==120 && n_Y==300
        parity=compare_figure9_python(curves,source;directory,
                    filename="figure9_python_julia_parity_$(mode).tsv")
        for key in FIGURE9_SCENARIOS
            rows=[r for r in parity.rows if r.scenario===key &&
                                                   r.status===:compared]
            if !isempty(rows)
                @info "Python/Julia Figure 9 parity (matched grid only)" key mode count=length(rows) max_debt_error=maximum(r->abs(r.debt_error),rows) max_deficit_error=maximum(r->abs(r.deficit_error),rows)
            end
        end
    elseif mode===:official
        @warn "Original Python Figure 9 grid reference not available; parity unverified" source
    end
    tables=make_figure9_tables(scenarios,curves;directory,parity)
    figure=plot_figure9(curves,mode;directory)
    return (;scenarios,curves,figure,tables,parity,calfile)
end

"""Produce both calibration modes and a common-grid audit without interpolation."""
function run_both_figure9(;directory::AbstractString=@__DIR__,
                           n_r::Int=120,n_Y::Int=300,strict::Bool=true)
    official=run_figure9(;mode=:official,directory,n_r,n_Y,strict)
    native=run_figure9(;mode=:native,directory,n_r,n_Y,strict)
    path=joinpath(directory,"figure9_native_vs_official.tsv")
    open(path,"w") do io
        println(io,"scenario\tindex\tbranch\tmatched\tnative_B\tofficial_B\tnative_deficit\tofficial_deficit\tdebt_gap\tdeficit_gap")
        for key in FIGURE9_SCENARIOS
            for r in compare_curves(native.curves[key],official.curves[key])
                vals=(key,r.index,r.branch,r.matched,r.native_B,
                      r.official_B,r.native_deficit,r.official_deficit,
                      r.debt_gap,r.deficit_gap)
                println(io,join(string.(vals),'\t'))
            end
        end
    end
    return (;native,official,comparison_file=path)
end

if abspath(PROGRAM_FILE)==@__FILE__
    mode=Symbol(get(ENV,"GOLDILOCKS_FIG9_MODE","official"))
    if mode===:both
        result=run_both_figure9()
        println("Figure 9 native: ",result.native.figure.pdf)
        println("Figure 9 official: ",result.official.figure.pdf)
        println("Cross-mode comparison: ",result.comparison_file)
    else
        result=run_figure9(;mode)
        println("Figure 9 PDF: ",result.figure.pdf)
        println("Figure 9 PNG: ",result.figure.png)
        println("Figure 9 counterfactual calibration: ",result.calfile)
        for (name,path) in result.tables
            println("Figure 9 ",name," table: ",path)
        end
    end
end
