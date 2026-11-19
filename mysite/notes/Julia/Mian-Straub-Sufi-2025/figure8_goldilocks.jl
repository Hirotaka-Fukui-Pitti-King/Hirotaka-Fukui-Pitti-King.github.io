# Figure 8, Mian–Straub–Sufi (2025), annual US/Japan only.
# Run in project root: julia --project=. .\figure8_goldilocks.jl
# GOLDILOCKS_FIG8_MODE=official (or native) selects calibrated inputs.

include(joinpath(@__DIR__, "GoldilocksCore.jl"))
include(joinpath(@__DIR__, "GoldilocksHouseholds.jl"))
include(joinpath(@__DIR__, "GoldilocksCalibration.jl"))
include(joinpath(@__DIR__, "GoldilocksStyle.jl"))
include(joinpath(@__DIR__, "GoldilocksDeficitDebt.jl"))

using CairoMakie
using Printf
using .GoldilocksCalibration
using .GoldilocksStyle
using .GoldilocksDeficitDebt

# This is a re-solved equilibrium Figure, never a trace of the saved image.
function _draw_branch!(ax, curve::DeficitDebtCurve, branch::Symbol, style)
    # Missing roots split polylines. Do not connect over a failure gap or
    # join ZLB/flex through extrapolated/missing equilibrium points.
    xs=Float64[];ys=Float64[]
    function draw_segment!()
        if length(xs)>=2
            lines!(ax,xs,ys;color=style.color,linestyle=style.linestyle,
                   linewidth=2.6)
        end
        empty!(xs);empty!(ys)
    end
    for p in curve.records
        if ismissing(p) || p.branch!==branch 
            draw_segment!()
        else
            push!(xs,100 * p.B/curve.periods)
            push!(ys,100 * p.deficit)
        end
    end
    draw_segment!()
end

function _plot_figure8(curves::AbstractVector{<:DeficitDebtCurve}, mode::Symbol;
                       directory::AbstractString=@__DIR__)
    by_key=Dict((c.country,c.model)=>c for c in curves)
    fig=with_theme(publication_theme(;fontsize=17)) do
        f=Figure(size=(1240,550),fontsize=17)
        axs=(Axis(f[1,1],title="Panel A. United States",
                  xlabel="Debt to GDP (percent)",
                  ylabel="Deficit to GDP (percent)",
                  xticks=0:50:250,yticks=0:0.5:3),
             Axis(f[1,2],title="Panel B. Japan",
                  xlabel="Debt to GDP (percent)",
                  ylabel="Deficit to GDP (percent)",
                  xticks=0:100:400,yticks=0:1:5))
        for (panel,country) in enumerate((:us,:jp))
            ax=axs[panel]
            for model in FIGURE8_MODELS
                curve=by_key[(country,model)]
                style=model_style(model)
                _draw_branch!(ax,curve,:zlb,style)
                _draw_branch!(ax,curve,:flex,style)
            end
            ref=by_key[(country,:bu)]
            ref_eq=GoldilocksCore.evaluate_steady_state(ref.setup.state,
                                                        ref.core,ref.household)
            scatter!(ax,[100 * ref_eq.B/ref.periods],[100 * ref_eq.deficit];
                     color=OKABE_ITO.black,markersize=10)
            hlines!(ax,[0.0];color=:gray65,linestyle=:dot,linewidth=1)
        end
        xlims!(axs[1],0,250);ylims!(axs[1],0,3.2)
        xlims!(axs[2],0,430);ylims!(axs[2],0,5.2)
        # Original legend is printed only in the Japanese panel.
        handles=[lines!(axs[2],[NaN,NaN],[NaN,NaN];
                         color=model_style(m).color,
                         linestyle=model_style(m).linestyle,
                         linewidth=2.6) for m in FIGURE8_MODELS]
        Legend(f[1,2],handles,[model_style(m).label for m in FIGURE8_MODELS];
               tellheight=false,tellwidth=false,
               halign=:right,valign=:top,margin=(5,8,10,2),
               framevisible=false,labelsize=13)
        f
    end
    stem="figure8_deficit_debt_$(mode)"
    pdf=joinpath(directory,stem*".pdf")
    png=joinpath(directory,stem*".png")
    save(pdf,fig)
    save(png,fig;px_per_unit=3)
    return (;pdf,png,figure=fig)
end

"""
    run_figure8(;mode=:native,n_r=40,n_Y=70)

Recalibrate all 4 plotted savers in Julia (default), or use author-saved
parameters strictly as inputs (`mode=:official`). Data and outputs stay in the
current project root. Do not claim parity until the source curves are checked.
"""
function run_figure8(;mode::Symbol=:native,n_r::Int=40,n_Y::Int=70,
                      directory::AbstractString=@__DIR__,strict::Bool=true)
    mode in (:native,:official) || throw(ArgumentError("unknown calibration mode"))
    curves=DeficitDebtCurve[]
    for country in (:us,:jp)
        for model in FIGURE8_MODELS
            @info "Solving Figure 8" country model mode n_r n_Y
            input=figure8_calibration(country,model;mode)
            curve=deficit_debt_curve(input.setup,input.household,input.core;
                                     model,mode,n_r,n_Y)
            for issue in curve.issues
                @warn "Unsolved/terminated Figure 8 grid point" country=country model=model index=issue.index branch=issue.branch grid_value=issue.grid_value reason=issue.reason
            end
            count_flex=count(p->!ismissing(p)&&p.branch===:flex,curve.records)
            count_zlb=count(p->!ismissing(p)&&p.branch===:zlb,curve.records)
            if strict && (count_flex!=n_r || count_zlb<2)
                error("Incomplete Figure 8 $country / $model: $count_flex/$n_r flex and $count_zlb ZLB roots")
            end
            path=write_curve_data(curve;directory)
            mx=grid_maximum(curve)
            @info "Computed deficit-debt maximum (grid approximation)" country=country model=model deficit_percent=(100*mx.maximum_deficit) debt_percent=(100*mx.debt_at_maximum) count_flex=count_flex count_zlb=count_zlb data=path
            push!(curves,curve)
        end
    end
    table=make_deficit_debt_table(curves;directory,
                                  filename="figure8_maxima_$(mode).tex")
    reference=joinpath(directory,"figure8_original_python_reference.tsv")
    if isfile(reference) && n_r==40 && n_Y==70
        report=joinpath(directory,"figure8_python_parity_$(mode).tsv")
        rows=compare_python_reference(curves,reference;output_file=report)
        diffs=[abs(row.deficit_difference) for row in rows if row.status===:compared]
        @info "Original Python vs Julia pointwise audit" mode=mode compared=length(diffs) max_deficit_error=(isempty(diffs) ? missing : maximum(diffs)) report=report
    end
    figure=_plot_figure8(curves,mode;directory)
    return (;curves,table,figure)
end


"""Compute both parameter modes without inventing missing common ZLB grid points."""
function run_both_figures(;n_r::Int=40,n_Y::Int=70,
                           directory::AbstractString=@__DIR__,strict::Bool=true)
    native=run_figure8(;mode=:native,n_r,n_Y,directory,strict)
    official=run_figure8(;mode=:official,n_r,n_Y,directory,strict)
    path=joinpath(directory,"figure8_native_vs_official.tsv")
    official_map=Dict((c.country,c.model)=>c for c in official.curves)
    open(path,"w") do io
        println(io,"country\tmodel\tbranch\tindex\tgrid_native\tgrid_official\tmatched\tnative_debt\tofficial_debt\tnative_deficit\tofficial_deficit\tdebt_difference\tdeficit_difference")
        for c in native.curves
            benchmark=official_map[(c.country,c.model)]
            for row in compare_curves(c,benchmark)
                values=(c.country,c.model,row.branch,row.index,
                        row.grid_native,row.grid_official,row.matched,
                        row.native_B,row.official_B,row.native_deficit,
                        row.official_deficit,row.debt_gap,row.deficit_gap)
                println(io,join(string.(values),"\t"))
            end
        end
    end
    return (;native,official,comparison_file=path)
end

if abspath(PROGRAM_FILE)==@__FILE__
    mode=Symbol(get(ENV,"GOLDILOCKS_FIG8_MODE","native"))
    if mode===:both
        result=run_both_figures()
        println("Figures (native/official): ",result.native.figure.pdf," | ",
                result.official.figure.pdf)
        println("Parameter-mode comparison: ",result.comparison_file)
    else
        result=run_figure8(;mode)
        println("Figure saved: ",result.figure.pdf)
        println("Figure saved: ",result.figure.png)
        println("Table saved: ",result.table.path)
    end
end
