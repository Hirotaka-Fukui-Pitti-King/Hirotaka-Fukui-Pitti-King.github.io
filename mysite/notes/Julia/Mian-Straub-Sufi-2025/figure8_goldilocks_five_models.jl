# Five-model EXTENSION of Mian–Straub–Sufi (2025), Figure 8.
# The original four-model figure8_goldilocks.jl is not modified.
# Run from the project root:
#   julia --project=. .\figure8_goldilocks_five_models.jl
# Modes: set GOLDILOCKS_FIG8_MODE to native, official, or both.

include(joinpath(@__DIR__, "figure8_goldilocks.jl"))

using CairoMakie
using Printf
using .GoldilocksCore
using .GoldilocksHouseholds
using .GoldilocksCalibration
using .GoldilocksDeficitDebt
using .GoldilocksStyle

const FIGURE8_FIVE_MODELS = (:bu, :bu_log, :olg, :olg_alt, :ha)

# Taken from official annual us_ss.pkl and jp_ss.pkl. They are inputs to the
# independently solved stationary GE, NEVER coordinates of a deficit-debt curve.
const FIVE_MODEL_OLG_ALT_OFFICIAL = Dict(
    :us => (beta=0.95, survival=1.0025600010536202,
            zeta=0.21906450444646552),
    :jp => (beta=0.97, survival=1.0035976421982629,
            zeta=0.08072458197737008),
)

"""
    five_model_calibration(country, model; mode=:native)

Return the same `(setup, household, core, anchor, mode, source)` contract as
`figure8_calibration`. The four original models delegate unchanged; OLG-alt
uses the *same* OLG household equations and source-specific calibration.
"""
function five_model_calibration(country::Symbol, model::Symbol;
                                mode::Symbol=:native)
    model in FIGURE8_FIVE_MODELS ||
        throw(ArgumentError("not among the five Figure 8 extension models: $model"))
    mode in (:native, :official) ||
        throw(ArgumentError("mode must be :native or :official"))
    model !== :olg_alt && return figure8_calibration(country, model; mode)
    s = country_setup(country, 1)
    if mode === :native
        result = calibrate_model(s, :olg_alt; method=:finite)
        return (; setup=s, household=result.household, core=result.core,
                anchor=result.eq, mode, source=result)
    end
    p = FIVE_MODEL_OLG_ALT_OFFICIAL[country]
    household = OLGHousehold(; beta=p.beta, survival=p.survival, zeta=p.zeta)
    # Reuse the original model's household-specific Phillips-curve parameters.
    core = GoldilocksDeficitDebt._household_core(s, household)
    anchor = evaluate_steady_state(s.state, core, household)
    return (; setup=s, household, core, anchor, mode, source=nothing)
end

"""Save each point and each failure without overwriting four-model TSV files."""
function _write_five_model_data(curve::DeficitDebtCurve;
                                directory::AbstractString=@__DIR__)
    path = joinpath(directory, "figure8_five_models_$(curve.country)_$(curve.model)_$(curve.mode).tsv")
    issues = Dict(issue.index => issue.reason for issue in curve.issues)
    n_Y = length(curve.Y_grid)
    open(path, "w") do io
        println(io, "index\tbranch\tgrid_input\tstatus\tB\tB_over_GDP\tdeficit\tY\tr_ante\tr\ti\ti_rule\tpi\tpiw\ttax\tC\tA\tresidual\tregime_valid\tnote")
        for i in eachindex(curve.records)
            branch = i <= n_Y ? :zlb : :flex
            grid_value = i <= n_Y ? curve.Y_grid[i] : curve.r_grid[i-n_Y]
            p = curve.records[i]
            if ismissing(p)
                fields = (i, branch, repr(grid_value), "UNSOLVED",
                          fill("", 15)..., get(issues, i, "not attempted"))
            else
                fields = (i, branch, repr(grid_value), "SOLVED", repr(p.B),
                          repr(p.B/curve.periods), repr(p.deficit), repr(p.Y),
                          repr(p.r_ante), repr(p.r), repr(p.i), repr(p.i_rule),
                          repr(p.pi), repr(p.piw), repr(p.tax), repr(p.C),
                          repr(p.A), repr(p.residual), p.regime_valid,
                          get(issues, i, ""))
            end
            println(io, join(fields, '\t'))
        end
    end
    return path
end

# The original source puts the legend inside Japan; a shared bottom legend
# gives five long descriptions room to breathe. The axes retain source units.
function _plot_five_models(curves::AbstractVector{<:DeficitDebtCurve},
                           mode::Symbol; directory::AbstractString=@__DIR__)
    by_key = Dict((c.country, c.model) => c for c in curves)
    fig = with_theme(publication_theme(; fontsize=17)) do
        f = Figure(size=(1320, 660), fontsize=17)
        axs = (
            Axis(f[1,1], title="Panel A. United States",
                 xlabel="Debt to GDP (percent)",
                 ylabel="Primary deficit to GDP (percent)",
                 xticks=0:50:250, yticks=0:0.5:3),
            Axis(f[1,2], title="Panel B. Japan",
                 xlabel="Debt to GDP (percent)",
                 ylabel="Primary deficit to GDP (percent)",
                 xticks=0:100:400, yticks=0:1:5),
        )
        handles = Any[]
        for (panel, country) in enumerate((:us, :jp))
            ax = axs[panel]
            for model in FIGURE8_FIVE_MODELS
                curve = by_key[(country, model)]
                style = model_style(model)
                _draw_branch!(ax, curve, :zlb, style)
                _draw_branch!(ax, curve, :flex, style)
                if model === :olg_alt
                    # Source calibrations make OLG and OLG-alt nearly exactly
                    # observationally equivalent on this locus. Mark ORIGINAL
                    # computed OLG-alt grid points, without offset/jitter.
                    for branch in (:zlb, :flex)
                        points = [p for p in curve.records if !ismissing(p) &&
                                  p.branch === branch && p.B >= 0 &&
                                  p.deficit >= 0]
                        if !isempty(points)
                            stride = max(1, cld(length(points), 8))
                            selected = points[1:stride:end]
                            scatter!(ax,
                                [100*p.B/curve.periods for p in selected],
                                [100*p.deficit for p in selected];
                                marker=:diamond, markersize=7,
                                color=style.color)
                        end
                    end
                end
            end
            ref = by_key[(country, :bu)]
            eq = evaluate_steady_state(ref.setup.state, ref.core, ref.household)
            scatter!(ax, [100*eq.B/ref.periods], [100*eq.deficit];
                     color=OKABE_ITO.black, markersize=10)
            hlines!(ax, [0.0]; color=:gray65, linestyle=:dot, linewidth=1)
        end
        xlims!(axs[1], 0, 250); ylims!(axs[1], 0, 3.2)
        xlims!(axs[2], 0, 430); ylims!(axs[2], 0, 5.2)
        for model in FIGURE8_FIVE_MODELS
            style = model_style(model)
            push!(handles, lines!(axs[2], [NaN, NaN], [NaN, NaN];
                                  color=style.color,
                                  linestyle=style.linestyle,
                                  linewidth=2.8))
        end
        labels = [model_style(model).label for model in FIGURE8_FIVE_MODELS]
        Legend(f[2,1:2], handles, labels;
               orientation=:horizontal, nbanks=3,
               tellwidth=false, framevisible=false, labelsize=13)
        Label(f[3,1:2],
              "OLG-alt diamonds mark computed points; OLG and OLG-alt may overlap.";
              fontsize=12, color=:gray35, tellwidth=false)
        rowgap!(f.layout, 10)
        f
    end
    stem = "figure8_five_models_deficit_debt_$(mode)"
    pdf = joinpath(directory, stem * ".pdf")
    png = joinpath(directory, stem * ".png")
    save(pdf, fig)
    save(png, fig; px_per_unit=3)
    return (; pdf, png, figure=fig)
end

"""
    run_figure8_five_models(; mode=:native, n_r=40, n_Y=70, strict=true)

Re-solve US and Japan five-model equilibrium loci and render a two-panel
Figure 8 extension. All names start with `figure8_five_models_` to preserve
original four-model results. Original Python parity is checked only if its
independently computed 1,100-point reference TSV is present.
"""
function run_figure8_five_models(; mode::Symbol=:native,
                                  n_r::Int=40, n_Y::Int=70,
                                  directory::AbstractString=@__DIR__,
                                  strict::Bool=true)
    mode in (:native, :official) || throw(ArgumentError("invalid mode: $mode"))
    curves = DeficitDebtCurve[]
    for country in (:us, :jp), model in FIGURE8_FIVE_MODELS
        @info "Solving five-model Figure 8 extension" country model mode n_r n_Y
        input = five_model_calibration(country, model; mode)
        curve = deficit_debt_curve(input.setup, input.household, input.core;
                                   model, mode, n_r, n_Y)
        count_flex = count(p -> !ismissing(p) && p.branch === :flex,
                           curve.records)
        count_zlb = count(p -> !ismissing(p) && p.branch === :zlb,
                          curve.records)
        if strict && (count_flex != n_r || count_zlb < 2)
            error("Incomplete locus for $country/$model ($mode): " *
                  "$count_flex/$n_r flex and $count_zlb ZLB equilibria")
        end
        for issue in curve.issues
            @warn "Unsolved/terminated five-model grid point" country model mode index=issue.index branch=issue.branch reason=issue.reason
        end
        path = _write_five_model_data(curve; directory)
        mx = grid_maximum(curve)
        @info "Five-model computed maximum" country model mode deficit_percent=(100*mx.maximum_deficit) debt_percent=(100*mx.debt_at_maximum/curve.periods) data=path
        push!(curves, curve)
    end
    table = make_deficit_debt_table(curves; directory,
                  filename="figure8_five_models_maxima_$(mode).tex")
    reference = joinpath(directory, "figure8_five_models_original_python_reference.tsv")
    if n_r == 40 && n_Y == 70 && isfile(reference)
        report = joinpath(directory, "figure8_five_models_python_parity_$(mode).tsv")
        rows = compare_python_reference(curves, reference; output_file=report)
        errs = [abs(row.deficit_difference) for row in rows
                if row.status === :compared]
        @info "Original Python five-model parity" mode compared=length(errs) max_absolute_deficit_error=(isempty(errs) ? missing : maximum(errs)) report=report
    end
    figure = _plot_five_models(curves, mode; directory)
    return (; curves, table, figure)
end

"""Generate both calibration versions and save only exact common-grid comparisons."""
function run_both_five_model_figures(; n_r::Int=40, n_Y::Int=70,
                                      directory::AbstractString=@__DIR__,
                                      strict::Bool=true)
    native = run_figure8_five_models(; mode=:native, n_r, n_Y,
                                      directory, strict)
    official = run_figure8_five_models(; mode=:official, n_r, n_Y,
                                        directory, strict)
    official_map = Dict((c.country,c.model) => c for c in official.curves)
    path = joinpath(directory, "figure8_five_models_native_vs_official.tsv")
    open(path, "w") do io
        println(io,"country\tmodel\tbranch\tindex\tgrid_native\tgrid_official\tmatched\tnative_debt\tofficial_debt\tnative_deficit\tofficial_deficit\tdebt_difference\tdeficit_difference")
        for c in native.curves
            benchmark = official_map[(c.country,c.model)]
            for row in compare_curves(c, benchmark)
                fields = (c.country, c.model, row.branch, row.index,
                          row.grid_native, row.grid_official, row.matched,
                          row.native_B, row.official_B,
                          row.native_deficit, row.official_deficit,
                          row.debt_gap, row.deficit_gap)
                println(io, join(string.(fields), '\t'))
            end
        end
    end
    return (; native, official, comparison_file=path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    mode = Symbol(get(ENV, "GOLDILOCKS_FIG8_MODE", "native"))
    if mode === :both
        result = run_both_five_model_figures()
        println("Five-model figures: ", result.native.figure.pdf,
                " | ", result.official.figure.pdf)
        println("Cross-calibration comparison: ", result.comparison_file)
    else
        result = run_figure8_five_models(; mode)
        println("Five-model Figure 8: ", result.figure.pdf)
        println("Five-model Figure 8: ", result.figure.png)
        println("Five-model maxima table: ", result.table.path)
    end
end
