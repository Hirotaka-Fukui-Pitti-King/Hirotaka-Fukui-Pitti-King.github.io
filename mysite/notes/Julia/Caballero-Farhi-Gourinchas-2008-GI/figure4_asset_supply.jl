# Caballero, Farhi, and Gourinchas (2008), AER 98(1), Figure 4.
# Julia reconstruction of prog_UR_final.gau (not a line-by-line translation).
#
# First-time installation in the Julia REPL:
#   using Pkg
#   Pkg.add(string(:CairoMakie))
#   Pkg.add(string(:SummaryTables))
#
# Run from PowerShell:
#   julia figure4_asset_supply.jl
#   julia figure4_asset_supply.jl C:\path\to\output
#
# Output: Figure 4 (.pdf, .png), a SummaryTables table (.html, .tex),
#         and a time-series CSV. All calculations use exact analytic solutions.

using CairoMakie
using SummaryTables: Cell, Table
using Printf

# Exact Okabe-Ito (Wong) categorical palette; color is never used to encode
# quantities. Hexadecimal specifications avoid palette-name/version ambiguity.
const OKABE_ITO = (
    orange    = "#E69F00",
    skyblue   = "#56B4E9",
    green     = "#009E73",
    yellow    = "#F0E442",
    blue      = "#0072B2",
    vermilion = "#D55E00",
    purple    = "#CC79A7",
    black     = "#000000",
)

"""Published Figure 4 defaults, including the GAUSS sampling grid."""
Base.@kwdef struct Parameters
    θ::Float64 = 0.25
    g::Float64 = 0.03
    r_pre::Float64 = 0.06
    x_u::Float64 = 0.70
    world_output_0::Float64 = 1.0
    initial_nfa_u_over_output::Float64 = 0.0
    initial_u_share_in_r_wealth::Float64 = 0.05
    asset_price_change_r::Float64 = -0.25
    n_pre::Int = 40
    n_post::Int = 26 * 40
    pre_years::Float64 = 2.0
    post_step::Float64 = 1.0 / 40.0
end

"""Check economic admissibility before evaluating the analytic solution."""
function check_parameters(p::Parameters)
    0.0 < p.θ || throw(ArgumentError("θ must be positive"))
    0.0 < p.g || throw(ArgumentError("g must be positive"))
    0.0 < p.x_u < 1.0 || throw(ArgumentError("x_u must lie in (0,1)"))
    p.world_output_0 > 0.0 || throw(ArgumentError("initial output must be positive"))
    p.g < p.r_pre < p.g + p.θ ||
        throw(ArgumentError("require g < r_pre < g + θ"))
    -1.0 < p.asset_price_change_r < 0.0 ||
        throw(ArgumentError("expected a negative R-asset valuation shock smaller than 100%"))
    0.0 <= p.initial_u_share_in_r_wealth <= 1.0 ||
        throw(ArgumentError("initial portfolio fraction must be in [0,1]"))
    p.n_pre > 0 && p.n_post > 0 ||
        throw(ArgumentError("numbers of observations must be positive"))
    p.pre_years > 0.0 && p.post_step > 0.0 ||
        throw(ArgumentError("time steps must be positive"))
    return nothing
end

"""
Calibrate delta_R from the *endogenous* post-crash relative asset value.
Maintain the original GAUSS definitions of AUR and ARU:
AUR = share of R trees held by U, ARU = share of U trees held by R.
"""
function calibrate(p::Parameters)
    check_parameters(p)

    δu = (p.r_pre - p.g) / p.θ
    x_r = 1.0 - p.x_u
    z = 1.0 + p.asset_price_change_r  # V_R(0+)/V_R(0-)
    δr = z * p.x_u * δu / (1.0 - z * x_r)
    δbar = p.x_u * δu + x_r * δr
    r_post = p.g + p.θ * δbar

    # Pre-shock gross cross-holdings; exactly the GAUSS initial conditions.
    aur0 = p.initial_u_share_in_r_wealth +
           p.θ * p.initial_nfa_u_over_output *
           (1.0 - p.initial_u_share_in_r_wealth) * p.x_u / x_r
    aru0 = aur0 * x_r / p.x_u -
           p.θ * p.initial_nfa_u_over_output
    (0.0 <= aur0 <= 1.0 && 0.0 <= aru0 <= 1.0) ||
        throw(ArgumentError("initial asset-holding fractions are inadmissible"))
    0.0 < δr < δbar < δu < 1.0 ||
        throw(ArgumentError("the calibrated financial development parameters are inadmissible"))

    xu0 = p.world_output_0 * p.x_u
    xr0 = p.world_output_0 * x_r
    vu0 = δu / (r_post - p.g) * xu0
    vr0 = δr / (r_post - p.g) * xr0
    wu0 = (1.0 - aru0) * vu0 + aur0 * vr0
    wr0 = aru0 * vu0 + (1.0 - aur0) * vr0

    return (; δu, δr, δbar, r_post, xu0, xr0, vu0, vr0, wu0, wr0,
            aur0, aru0)
end

"""
Evaluate the full time path using the exact closed-form wealth solution.
The time grid matches GAUSS: -2:0.05:-0.05 followed by 0:0.025:25.975.
U portfolio acquisition of R trees (AUR) is held at its pre-shock value;
R acquisition of U trees (ARU) is inferred from R's wealth identity.
"""
function simulate(p::Parameters, c=calibrate(p))
    pre_t = [-p.pre_years + (k - 1) * p.pre_years / p.n_pre
             for k in 1:p.n_pre]
    post_t = [(k - 1) * p.post_step for k in 1:p.n_post]
    time = vcat(pre_t, post_t)
    n = length(time)

    r = Vector{Float64}(undef, n)
    wu_y = Vector{Float64}(undef, n)
    vu_y = Vector{Float64}(undef, n)

    # Pre-shock holdings at the chosen portfolio allocation.
    wu_y_pre = ((1.0 - c.aru0) + c.aur0 * (1.0 - p.x_u) / p.x_u) / p.θ
    # Post-shock analytic solution of d(W_U/X_U)/dt =
    # (r_post - θ - g)*(W_U/X_U) + (1 - δu).
    speed = p.g + p.θ - c.r_post
    wu_y_limit = (1.0 - c.δu) / speed
    wu_y_impact = c.wu0 / c.xu0
    vu_y_post = c.δu / (c.r_post - p.g)

    for i in eachindex(time)
        t = time[i]
        if t < 0.0
            r[i] = p.r_pre
            wu_y[i] = wu_y_pre
            vu_y[i] = 1.0 / p.θ
        else
            r[i] = c.r_post
            wu_y[i] = wu_y_limit +
                      (wu_y_impact - wu_y_limit) * exp(-speed * t)
            vu_y[i] = vu_y_post
        end
    end

    # Global clearing is exact: W_U+W_R = V_U+V_R = X_world/θ.
    # With common growth rates, output shares stay fixed.
    xr_y = 1.0 - p.x_u
    wr_y = (1.0 / p.θ .- p.x_u .* wu_y) ./ xr_y
    vr_y = (1.0 / p.θ .- p.x_u .* vu_y) ./ xr_y
    na_u = wu_y .- vu_y
    na_r = wr_y .- vr_y
    tb_u = 1.0 .- p.θ .* wu_y
    tb_r = 1.0 .- p.θ .* wr_y
    ca_u = tb_u .+ r .* na_u
    ca_r = tb_r .+ r .* na_r

    # Original GAUSS portfolio-accounting convention (AUR is constant).
    aru = (wr_y .- (1.0 - c.aur0) .* vr_y) .* xr_y ./
          (vu_y .* p.x_u)
    share_u_in_r = aru .* vu_y .* p.x_u ./ (wr_y .* xr_y)

    return (; time, r, wu_y, vu_y, wr_y, vr_y, na_u, na_r,
            tb_u, tb_r, ca_u, ca_r, aru, share_u_in_r,
            speed, wu_y_limit)
end

"""Post-shock balanced-growth limits for comparison with published Figure 4."""
function asymptotes(p::Parameters, c=calibrate(p))
    wu_y = (1.0 - c.δu) / (p.g + p.θ - c.r_post)
    vu_y = c.δu / (c.r_post - p.g)
    wr_y = (1.0 - c.δr) / (p.g + p.θ - c.r_post)
    vr_y = c.δr / (c.r_post - p.g)
    nfa_u = wu_y - vu_y
    ca_u = p.g * nfa_u
    share_u_in_r = 1.0 - (1.0 - c.aur0) * vr_y / wr_y
    return (; wu_y, vu_y, wr_y, vr_y, nfa_u, ca_u, share_u_in_r,
            r=c.r_post)
end

"""
Structural and quantitative smoke tests, independent of the plotting library.
Numerical reference values are independently computed from the printed equations;
they are NOT claimed to be output from running the original GAUSS interpreter.
"""
function verify_replication(p::Parameters, c, sim, lim)
    atol = 5e-11
    @assert isapprox(c.δu, 0.12; atol)
    @assert isapprox(c.δr, 0.08129032258064517; atol)
    @assert isapprox(c.r_post, 0.05709677419354839; atol)

    # R valuation shock and global asset-market clearing at impact.
    old_vr0 = c.xr0 / p.θ
    @assert isapprox(c.vr0 / old_vr0,
                     1.0 + p.asset_price_change_r; atol)
    @assert isapprox(c.vu0 + c.vr0, p.world_output_0 / p.θ; atol)
    @assert isapprox(c.wu0 + c.wr0, p.world_output_0 / p.θ; atol)

    # Accounting identities hold at every pre- and post-shock date.
    @assert maximum(abs.(p.x_u .* sim.na_u .+
                         (1.0 - p.x_u) .* sim.na_r)) < atol
    @assert maximum(abs.(p.x_u .* sim.ca_u .+
                         (1.0 - p.x_u) .* sim.ca_r)) < atol
    @assert maximum(abs.(sim.ca_u .- sim.tb_u .-
                         sim.r .* sim.na_u)) < atol
    @assert maximum(abs.(sim.ca_r .- sim.tb_r .-
                         sim.r .* sim.na_r)) < atol
    @assert all((0.0 .<= sim.aru) .& (sim.aru .<= 1.0))
    @assert all((0.0 .<= sim.share_u_in_r) .&
                (sim.share_u_in_r .<= 1.0))

    # Long-run benchmarks (the source figure reports rounded values).
    @assert isapprox(100.0 * lim.ca_u, -1.442009510026881; atol=1e-8)
    @assert isapprox(100.0 * lim.nfa_u, -48.06698366756255; atol=1e-8)
    @assert isapprox(100.0 * lim.share_u_in_r, 30.85147471910112;
                     atol=1e-8)

    # Impact benchmark, and annual variation from the wealth accumulation ODE.
    i0 = p.n_pre + 1
    @assert isapprox(sim.time[i0], 0.0; atol)
    @assert isapprox(100.0 * sim.ca_u[i0], -10.123765635286377;
                     atol=1e-8)
    @assert isapprox(100.0 * sim.na_u[i0], -3.061224489795933;
                     atol=1e-8)
    @assert isapprox(100.0 * sim.share_u_in_r[i0], 7.209302325581413;
                     atol=1e-8)
    @assert isapprox(sim.wu_y[i0], c.wu0 / c.xu0; atol)
    return nothing
end

"""Draw the four original panels with an explicit Okabe-Ito color system."""
function create_figure(p::Parameters, sim, lim)
    fig = Figure(size=(1180, 850), fontsize=16,
                 backgroundcolor=:white)

    ax_a = Axis(fig[1, 1], title="A. Current account / output",
                xlabel="Years", ylabel="Percent", xticks=-2:4:26)
    ax_b = Axis(fig[1, 2], title="B. Net foreign assets / output",
                xlabel="Years", ylabel="Percent", xticks=-2:4:26)
    ax_c = Axis(fig[2, 1], title="C. World real interest rate",
                xlabel="Years", ylabel="Percent", xticks=-2:4:26)
    ax_d = Axis(fig[2, 2], title="D. U assets in R's portfolio",
                xlabel="Years", ylabel="Percent", xticks=-2:4:26)

    for ax in (ax_a, ax_b, ax_c, ax_d)
        ax.xgridvisible = false
        ax.ygridvisible = false
        xlims!(ax, -p.pre_years, 26.0)
    end

    # Main paths (blue), asymptotes (vermilion), zero / pre-shock reference
    # lines (gray/black). These reflect original variables, not new scenarios.
    lines!(ax_a, sim.time, 100.0 .* sim.ca_u,
           color=OKABE_ITO.blue, linewidth=3)
    hlines!(ax_a, [100.0 * lim.ca_u],
            color=OKABE_ITO.vermilion, linewidth=2, linestyle=:dash)
    hlines!(ax_a, [0.0], color=OKABE_ITO.black,
            linewidth=1, linestyle=:dot)
    ylims!(ax_a, -15.0, 15.0)

    lines!(ax_b, sim.time, 100.0 .* sim.na_u,
           color=OKABE_ITO.blue, linewidth=3)
    hlines!(ax_b, [100.0 * lim.nfa_u],
            color=OKABE_ITO.vermilion, linewidth=2, linestyle=:dash)
    hlines!(ax_b, [0.0], color=OKABE_ITO.black,
            linewidth=1, linestyle=:dot)
    ylims!(ax_b, -100.0, 20.0)

    lines!(ax_c, sim.time, 100.0 .* sim.r,
           color=OKABE_ITO.blue, linewidth=3)
    hlines!(ax_c, [100.0 * p.r_pre],
            color=OKABE_ITO.black, linewidth=1.5, linestyle=:dash)
    ylims!(ax_c, 5.5, 6.15)

    lines!(ax_d, sim.time, 100.0 .* sim.share_u_in_r,
           color=OKABE_ITO.blue, linewidth=3)
    hlines!(ax_d, [100.0 * lim.share_u_in_r],
            color=OKABE_ITO.vermilion, linewidth=2, linestyle=:dash)
    hlines!(ax_d, [100.0 * p.initial_u_share_in_r_wealth],
            color=OKABE_ITO.black, linewidth=1, linestyle=:dot)
    ylims!(ax_d, 0.0, 50.0)
    return fig
end

"""Snapshot values for a publication table; this table was not in the GAUSS file."""
function snapshot_rows(p::Parameters, sim, lim)
    i_pre = p.n_pre
    i_impact = p.n_pre + 1
    i_year25 = findmin(abs.(sim.time .- 25.0))[2]
    return [
        ("Current account / U output (%)",
         100.0 * sim.ca_u[i_pre], 100.0 * sim.ca_u[i_impact],
         100.0 * sim.ca_u[i_year25], 100.0 * lim.ca_u),
        ("Net foreign assets / U output (%)",
         100.0 * sim.na_u[i_pre], 100.0 * sim.na_u[i_impact],
         100.0 * sim.na_u[i_year25], 100.0 * lim.nfa_u),
        ("World real interest rate (%)",
         100.0 * sim.r[i_pre], 100.0 * sim.r[i_impact],
         100.0 * sim.r[i_year25], 100.0 * lim.r),
        ("U assets in R's portfolio (%)",
         100.0 * sim.share_u_in_r[i_pre],
         100.0 * sim.share_u_in_r[i_impact],
         100.0 * sim.share_u_in_r[i_year25],
         100.0 * lim.share_u_in_r),
    ]
end

"""Export a SummaryTables.jl table to both HTML and LaTeX."""
function export_table(outdir::AbstractString, p::Parameters, sim, lim)
    rows = snapshot_rows(p, sim, lim)
    headings = ["Variable", "Pre-shock", "Impact (0+)",
                "Year 25", "Long-run limit"]
    strings = Matrix{String}(undef, length(rows) + 1, length(headings))
    strings[1, :] .= headings
    for (i, values) in enumerate(rows)
        strings[i + 1, 1] = values[1]
        for j in 2:length(headings)
            strings[i + 1, j] = @sprintf("%.3f", values[j])
        end
    end

    cells = [Cell(strings[i, j]; bold=(i == 1),
                  halign=(j == 1 ? :left : :right))
             for i in axes(strings, 1), j in axes(strings, 2)]
    tbl = Table(cells; header=1)
    open(joinpath(outdir, "figure4_summary.html"), "w") do io
        show(io, MIME"text/html"(), tbl)
    end
    open(joinpath(outdir, "figure4_summary.tex"), "w") do io
        show(io, MIME"text/latex"(), tbl)
    end
    return tbl
end

"""Save all original Figure 4 series for numerical comparisons and future work."""
function export_csv(outdir::AbstractString, sim)
    open(joinpath(outdir, "figure4_timeseries.csv"), "w") do io
        println(io, "year,ca_u_pct,tb_u_pct,nfa_u_pct,world_rate_pct,u_portfolio_in_r_pct")
        for i in eachindex(sim.time)
            @printf(io, "%.6f,%.10f,%.10f,%.10f,%.10f,%.10f\n",
                    sim.time[i], 100.0 * sim.ca_u[i],
                    100.0 * sim.tb_u[i], 100.0 * sim.na_u[i],
                    100.0 * sim.r[i], 100.0 * sim.share_u_in_r[i])
        end
    end
    return nothing
end

"""Calibrate, solve, check invariants, and produce reproducible artifacts."""
function main()
    outdir = isempty(ARGS) ? joinpath(@__DIR__, "output_fig04") : abspath(ARGS[1])
    mkpath(outdir)
    p = Parameters()
    c = calibrate(p)
    sim = simulate(p, c)
    lim = asymptotes(p, c)
    verify_replication(p, c, sim, lim)

    fig = create_figure(p, sim, lim)
    save(joinpath(outdir, "figure4.png"), fig; px_per_unit=2)
    save(joinpath(outdir, "figure4.pdf"), fig)
    export_table(outdir, p, sim, lim)
    export_csv(outdir, sim)

    @printf("Figure 4 calibration: δU=%.8f δR=%.8f r_post=%.8f\n",
            c.δu, c.δr, c.r_post)
    @printf("Long-run: CA/U output=%.4f%% NFA/U output=%.4f%% Portfolio=%.4f%%\n",
            100.0 * lim.ca_u, 100.0 * lim.nfa_u,
            100.0 * lim.share_u_in_r)
    println("Structural checks passed. Outputs: ", outdir)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
