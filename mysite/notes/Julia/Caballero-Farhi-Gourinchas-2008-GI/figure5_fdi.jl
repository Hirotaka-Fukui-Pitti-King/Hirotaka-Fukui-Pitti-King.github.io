# Caballero, Farhi, and Gourinchas (2008), AER 98(1), Figure 5.
# Reimplementation of prog_UR_inv_final.gau in Julia.
#
# Defaults reproduce the *source-code* parameter values. A real discrepancy
# exists: the original GAUSS code simulates κP = 8.129...% and 12%, while
# its legend and the published paper say 5% and 12%. Use
# --paper-label-values to simulate 5% and 12% instead. At κP=5%, however,
# the bilateral-private-gains Assumption 6 is not satisfied in this calibration.
#
# Install (once) from Julia / Windows PowerShell:
#   julia -e 'using Pkg; Pkg.add(string(:CairoMakie)); Pkg.add(string(:SummaryTables))'
#
# Execute:
#   julia --project=. figure5_fdi.jl
#   julia --project=. figure5_fdi.jl --paper-label-values
#   julia --project=. figure5_fdi.jl C:\path\to\output
#
# Output: figure5.png, figure5.pdf, figure5_summary.html/.tex,
#         figure5_timeseries.csv, figure5_metadata.txt.
#
# The old-tree prices are solved via a 2-state backward RK4 system;
# the wealth path is integrated forward by RK4. The no-FDI case uses an
# exact analytic solution. No numerical nonlinear fixed-point solver is needed.

using CairoMakie
using SummaryTables: Cell, Table
using Printf

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

"""Parameters for published Figure 5 (not a general investment model)."""
Base.@kwdef struct Parameters
    θ::Float64 = 0.25
    g::Float64 = 0.03
    gn::Float64 = 0.03
    gz::Float64 = 0.0
    κ::Float64 = 0.0
    r_pre::Float64 = 0.06
    x_u::Float64 = 0.7
    initial_u_share_in_r_wealth::Float64 = 0.05
    initial_nfa_u_over_output::Float64 = 0.0
    asset_price_change_r::Float64 = -0.25
    pre_years::Float64 = 2.0
    n_pre::Int = 40
    horizon::Float64 = 26.0
    terminal_time::Float64 = 350.0
    dt::Float64 = 1.0 / 40.0
end

"""Reject parameterizations not supported by these Figure 5 formulas."""
function check_parameters(p::Parameters)
    p.θ > 0.0 || throw(ArgumentError("θ must be positive"))
    0.0 < p.x_u < 1.0 || throw(ArgumentError("x_u must be in (0,1)"))
    p.g > 0.0 && p.gn > 0.0 && p.gz >= 0.0 ||
        throw(ArgumentError("growth parameters must be nonnegative and gn positive"))
    isapprox(p.g, p.gn + p.gz; atol=1e-12, rtol=0.0) ||
        throw(ArgumentError("the Figure 5 calibration requires g = gn + gz"))
    p.κ == 0.0 ||
        throw(ArgumentError("this verified Figure 5 specialization requires κ=0, as in the GAUSS file"))
    p.gz < p.r_pre < p.g + p.θ ||
        throw(ArgumentError("require gz < r_pre < g + θ"))
    -1.0 < p.asset_price_change_r < 0.0 ||
        throw(ArgumentError("the valuation shock must be in (-1,0)"))
    0.0 <= p.initial_u_share_in_r_wealth <= 1.0 ||
        throw(ArgumentError("invalid initial portfolio share"))
    p.initial_nfa_u_over_output == 0.0 ||
        throw(ArgumentError("the Figure 5 benchmark assumes initial NFA=0"))
    p.n_pre > 0 && p.dt > 0.0 && p.pre_years > 0.0 ||
        throw(ArgumentError("invalid time discretization"))
    p.terminal_time > p.horizon ||
        throw(ArgumentError("terminal_time must exceed the plotting horizon"))
    for v in (p.horizon, p.terminal_time)
        isapprox(v / p.dt, round(v / p.dt); atol=1e-9, rtol=0.0) ||
            throw(ArgumentError("horizon and terminal time must be multiples of dt"))
    end
    return nothing
end

"""
Calibrate the crash, source-code asset-holding convention, and FDI-price bounds.
AUR is the U-held share of R trees, ARU the R-held share of U trees.
"""
function calibrate(p::Parameters)
    check_parameters(p)
    δu = (p.r_pre - p.gz) * (1.0 - p.κ) / p.θ
    xr = 1.0 - p.x_u
    z = 1.0 + p.asset_price_change_r
    δr = z * p.x_u * δu / (1.0 - z * xr)
    δbar = p.x_u * δu + xr * δr
    r_no = p.gz + p.θ * δbar / (1.0 - p.κ)
    speed_no = p.g + p.θ - r_no

    aur0 = p.initial_u_share_in_r_wealth +
           p.θ * p.initial_nfa_u_over_output *
           (1.0 - p.initial_u_share_in_r_wealth) * p.x_u / xr
    aru0 = aur0 * xr / p.x_u -
           p.θ * p.initial_nfa_u_over_output

    0.0 < δr < δbar < δu < 1.0 ||
        throw(ArgumentError("inconsistent financial-development calibration"))
    0.0 <= aur0 <= 1.0 && 0.0 <= aru0 <= 1.0 ||
        throw(ArgumentError("inadmissible initial cross-holdings"))
    speed_no > 0.0 || throw(ArgumentError("unstable no-FDI wealth dynamics"))

    # GAUSS line 192: lower bound = gn * δR/δU * (1-κ)/θ.
    # Its value is about 8.129%, NOT the 5% written in the original legend.
    κp_min = p.gn * δr / δu * (1.0 - p.κ) / p.θ
    κp_max = p.gn * (1.0 - p.κ) / p.θ
    return (; δu, δr, δbar, xr, r_no, speed_no,
            aur0, aru0, κp_min, κp_max)
end

"""Exact no-FDI solution: common-growth investment model, eqs. (31)–(36)."""
function simulate_no_fdi(p::Parameters, c, post_t::Vector{Float64})
    vu_y = c.δu / c.δbar * (1.0 - p.κ) / p.θ
    vr_y = c.δr / c.δbar * (1.0 - p.κ) / p.θ
    wu0_y = (1.0 - c.aru0) * vu_y + c.aur0 * c.xr / p.x_u * vr_y
    # GAUSS lines 75 and 95–105. New investment options accrue to savers.
    flow = 1.0 - c.δu - p.κ +
           p.gn * (1.0 - p.κ) / p.θ * c.δu / c.δbar
    wu_limit = flow / c.speed_no
    wu = [wu_limit + (wu0_y - wu_limit) * exp(-c.speed_no * t)
          for t in post_t]
    vu = fill(vu_y, length(post_t))
    rr = fill(c.r_no, length(post_t))
    nfa = wu .- vu
    tb = 1.0 .- p.θ .* wu .- p.κ
    ca = tb .+ rr .* nfa
    return (; r=rr, wu, vu, nfa, tb, ca, wu_limit,
            nfa_limit=wu_limit - vu_y)
end

"""Share of world output attributable to old R trees after the crash."""
@inline old_r_share(t::Float64, p::Parameters, c) =
    c.xr * exp(-p.gn * t)

"""Equation (44)/Appendix F: equilibrium rate from old-R tree valuation."""
@inline function fdi_rate(t::Float64, v_old::Float64,
                          p::Parameters, c)
    xold = old_r_share(t, p, c)
    asset_income = c.δu * (1.0 - xold) + c.δr * xold
    option_spread = p.gn * v_old * xold * (c.δu / c.δr - 1.0)
    return p.gz + p.θ / (1.0 - p.κ) * (asset_income - option_spread)
end

"""Asset-price ODEs for pre-existing R and U trees, Appendix F."""
@inline function asset_derivatives(t::Float64, v_old::Float64,
                                   v_u::Float64, p::Parameters, c)
    rr = fdi_rate(t, v_old, p, c)
    return ((rr - p.gz) * v_old - c.δr,
            (rr - p.gz) * v_u - c.δu)
end

"""
Backward fourth-order Runge–Kutta, with exact exponential old-tree share.
Terminal conditions at t=350 approximate the infinite-horizon boundary.
Retain the full state path for forward integration; the original GAUSS
implementation used backward Euler with a 1/40-year step.
"""
function solve_backward_assets(p::Parameters, c)
    n = round(Int, p.terminal_time / p.dt) + 1
    t = [(i - 1) * p.dt for i in 1:n]
    v_old = Vector{Float64}(undef, n)
    v_u = Vector{Float64}(undef, n)
    r = Vector{Float64}(undef, n)

    # Appendix F terminal values, not the pre-shock valuation of old R trees.
    v_old[end] = (1.0 - p.κ) * c.δr / c.δu / p.θ
    v_u[end] = (1.0 - p.κ) / p.θ
    h = -p.dt

    for i in (n - 1):-1:1
        t_right = t[i + 1]
        y1 = v_old[i + 1]
        y2 = v_u[i + 1]
        k1a, k1b = asset_derivatives(t_right, y1, y2, p, c)
        k2a, k2b = asset_derivatives(t_right + h / 2.0,
                                     y1 + h * k1a / 2.0,
                                     y2 + h * k1b / 2.0, p, c)
        k3a, k3b = asset_derivatives(t_right + h / 2.0,
                                     y1 + h * k2a / 2.0,
                                     y2 + h * k2b / 2.0, p, c)
        k4a, k4b = asset_derivatives(t_right + h,
                                     y1 + h * k3a,
                                     y2 + h * k3b, p, c)
        v_old[i] = y1 + h * (k1a + 2.0 * k2a + 2.0 * k3a + k4a) / 6.0
        v_u[i] = y2 + h * (k1b + 2.0 * k2b + 2.0 * k3b + k4b) / 6.0
    end
    for i in eachindex(t)
        r[i] = fdi_rate(t[i], v_old[i], p, c)
    end
    return (; t, v_old, v_u, r)
end

"""Forward equation for US financial wealth per unit of US output."""
@inline function wealth_derivative(t::Float64, w::Float64,
                                   v_old::Float64, rr::Float64,
                                   κp::Float64, p::Parameters, c)
    xold = old_r_share(t, p, c)
    xnew = c.xr - xold
    flow = (1.0 - c.δu - p.κ / p.x_u) +
           p.gn * ((1.0 - p.κ) / (p.θ * p.x_u) -
                   v_old * xold / p.x_u * (1.0 - c.δu / c.δr)) -
           κp * xnew / p.x_u
    return (rr - p.θ - p.g) * w + flow
end

"""
Forward RK4 wealth solver. For midpoint values of the backward state, use
cubic Hermite interpolation; this preserves fourth-order local accuracy
without another ODE package or noisy linear interpolation of asset values.
"""
function solve_forward_wealth(p::Parameters, c, assets,
                              post_n::Int, κp::Float64)
    post_t = assets.t[1:post_n]
    vo = assets.v_old[1:post_n]
    vu = assets.v_u[1:post_n]
    rr = assets.r[1:post_n]
    wu = Vector{Float64}(undef, post_n)
    wu[1] = (1.0 - c.aru0) * vu[1] +
            c.aur0 * c.xr / p.x_u * vo[1]

    for i in 1:(post_n - 1)
        t0, t1 = post_t[i], post_t[i + 1]
        h = t1 - t0
        tm = t0 + h / 2.0
        # Hermite midpoint of old-R tree price; source prices are RK4 smooth.
        slope0 = (rr[i] - p.gz) * vo[i] - c.δr
        slope1 = (rr[i + 1] - p.gz) * vo[i + 1] - c.δr
        vo_mid = (vo[i] + vo[i + 1]) / 2.0 +
                 h * (slope0 - slope1) / 8.0
        r_mid = fdi_rate(tm, vo_mid, p, c)
        w = wu[i]
        k1 = wealth_derivative(t0, w, vo[i], rr[i], κp, p, c)
        k2 = wealth_derivative(tm, w + h * k1 / 2.0,
                               vo_mid, r_mid, κp, p, c)
        k3 = wealth_derivative(tm, w + h * k2 / 2.0,
                               vo_mid, r_mid, κp, p, c)
        k4 = wealth_derivative(t1, w + h * k3,
                               vo[i + 1], rr[i + 1], κp, p, c)
        wu[i + 1] = w + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    end
    nfa = wu .- vu
    tb = 1.0 .- p.θ .* wu .- p.κ
    ca = tb .+ rr .* nfa
    return (; κp, wu, vu, r=rr, nfa, tb, ca)
end

"""Match GAUSS pre-shock grid and expose both recorded and published κP choices."""
function simulate(p::Parameters, c; paper_label_values::Bool=false)
    np = round(Int, p.horizon / p.dt) + 1
    pre = [-p.pre_years + (i - 1) * p.pre_years / p.n_pre
           for i in 1:p.n_pre]
    assets = solve_backward_assets(p, c)
    post = assets.t[1:np]
    times = vcat(pre, post)
    no = simulate_no_fdi(p, c, post)

    κps = paper_label_values ? [0.05, 0.12] :
          collect(range(c.κp_min, c.κp_max; length=4))
    scenarios = [solve_forward_wealth(p, c, assets, np, κp) for κp in κps]
    return (; times, post, pre, no, assets, scenarios, paper_label_values)
end

"""Numerical checks against equations and independent high-accuracy calculations."""
function verify_replication(p::Parameters, c, sim)
    # Reference values are obtained independently using SciPy DOP853 on the
    # equations transcribed from the GAUSS source; not GAUSS run outputs.
    @assert isapprox(c.δu, 0.24; atol=1e-12)
    @assert isapprox(c.δr, 0.1625806451612903; atol=1e-12)
    @assert isapprox(c.r_no, 0.054193548387096765; atol=1e-12)
    @assert isapprox(c.κp_min, 0.08129032258064516; atol=1e-12)
    @assert isapprox(c.κp_max, 0.12; atol=1e-12)

    pre_vr = (1.0 - p.κ) / p.θ
    post_vr = c.δr / c.δbar * pre_vr
    @assert isapprox(post_vr / pre_vr,
                     1.0 + p.asset_price_change_r; atol=1e-12)
    @assert all(sim.assets.v_old .> 0.0)
    @assert all(sim.assets.v_u .> 0.0)
    @assert isapprox(sim.assets.r[end], p.r_pre; atol=1e-5)
    @assert isapprox(sim.assets.r[1], 0.050979262672801326; atol=2e-6)
    @assert isapprox(sim.assets.v_old[1], 3.0; atol=2e-6)
    @assert isapprox(sim.assets.v_u[1], 4.428571428584832; atol=2e-6)

    i25 = round(Int, 25.0 / p.dt) + 1
    @assert isapprox(sim.no.ca[i25], -0.014541898524409624; atol=1e-8)
    @assert isapprox(sim.no.nfa[i25], -0.4729208759755972; atol=1e-8)
    @assert isapprox(sim.assets.r[i25], 0.05582014662602605; atol=2e-6)

    for sc in sim.scenarios
        @assert minimum(sc.wu) > 0.0
        @assert maximum(abs.(sc.ca .- sc.tb .- sc.r .* sc.nfa)) < 1e-12
        @assert isapprox(sc.nfa[1], sim.no.nfa[1]; atol=2e-6)
    end
    @assert maximum(abs.(sim.no.ca .- sim.no.tb .-
                         sim.no.r .* sim.no.nfa)) < 1e-12

    if !sim.paper_label_values
        low, high = first(sim.scenarios), last(sim.scenarios)
        @assert isapprox(low.ca[i25], -0.03312574369604109; atol=2e-5)
        @assert isapprox(low.nfa[i25], -0.07609912898508675; atol=2e-5)
        @assert isapprox(high.ca[i25], -0.02659600408009459; atol=2e-5)
        @assert isapprox(high.nfa[i25], -0.10972640546351098; atol=2e-5)
    end
    return nothing
end

"""Prepend the unshocked period to post-crash series (GAUSS plotting convention)."""
function complete_series(p::Parameters, post::AbstractVector,
                         pre_value::Real)
    return vcat(fill(Float64(pre_value), p.n_pre), post)
end

"""Draw exactly the four panels in the original Figure 5."""
function create_figure(p::Parameters, c, sim)
    fig = Figure(size=(1320, 920), fontsize=16, backgroundcolor=:white)
    ax_a = Axis(fig[1, 1], title="A. Current account / output",
                xlabel="Years", ylabel="Percent")
    ax_b = Axis(fig[1, 2], title="B. Net foreign assets / output",
                xlabel="Years", ylabel="Percent")
    ax_c = Axis(fig[2, 1], title="C. World real interest rate",
                xlabel="Years", ylabel="Percent")
    ax_d = Axis(fig[2, 2], title="D. Trade balance / output",
                xlabel="Years", ylabel="Percent")
    for ax in (ax_a, ax_b, ax_c, ax_d)
        ax.xgridvisible = false
        ax.ygridvisible = false
        xlims!(ax, -p.pre_years, p.horizon)
        ax.xticks = -2:4:26
    end

    colors = (OKABE_ITO.vermilion, OKABE_ITO.green)
    choices = (first(sim.scenarios), last(sim.scenarios))
    base = sim.no

    for (ax, field) in ((ax_a, :ca), (ax_b, :nfa), (ax_d, :tb))
        lines!(ax, sim.times,
               100.0 .* complete_series(p, getproperty(base, field), 0.0),
               color=OKABE_ITO.blue, linewidth=3, label="No FDI")
        for (j, sc) in enumerate(choices)
            lbl = @sprintf("FDI: κP = %.3f%%", 100.0 * sc.κp)
            lines!(ax, sim.times,
                   100.0 .* complete_series(p, getproperty(sc, field), 0.0),
                   color=colors[j], linewidth=2.8,
                   linestyle=(j == 1 ? :solid : :dash), label=lbl)
        end
        hlines!(ax, [0.0]; color=OKABE_ITO.black,
                linestyle=:dot, linewidth=1.2)
        axislegend(ax, position=:rt, framevisible=false, labelsize=12)
    end

    lines!(ax_c, sim.times,
           100.0 .* complete_series(p, base.r, p.r_pre),
           color=OKABE_ITO.blue, linewidth=3, label="No FDI")
    lines!(ax_c, sim.times,
           100.0 .* complete_series(p, sim.assets.r[1:length(sim.post)],
                                    p.r_pre),
           color=OKABE_ITO.vermilion, linewidth=3, label="FDI")
    hlines!(ax_c, [100.0 * p.r_pre]; color=OKABE_ITO.black,
            linestyle=:dot, linewidth=1.2)
    axislegend(ax_c, position=:rb, framevisible=false, labelsize=12)

    ylims!(ax_a, -15.0, 5.0)
    ylims!(ax_b, -60.0, 10.0)
    ylims!(ax_c, 5.0, 6.2)
    ylims!(ax_d, -15.0, 5.0)
    return fig
end

"""Export a supplementary SummaryTables.jl comparison table."""
function export_table(outdir::AbstractString, p::Parameters, c, sim)
    idx0 = 1
    idx25 = round(Int, 25.0 / p.dt) + 1
    low, high = first(sim.scenarios), last(sim.scenarios)
    cases = (("No FDI", sim.no),
             (@sprintf("FDI κP=%.3f%%", 100.0 * low.κp), low),
             (@sprintf("FDI κP=%.3f%%", 100.0 * high.κp), high))
    headings = ("Scenario", "CA 0+ (%)", "CA year 25 (%)",
                "NFA year 25 (%)", "TB year 25 (%)", "r year 25 (%)")
    content = Matrix{String}(undef, length(cases) + 1, length(headings))
    content[1, :] .= collect(headings)
    for (i, (label, sc)) in enumerate(cases)
        content[i + 1, 1] = label
        metrics = (sc.ca[idx0], sc.ca[idx25], sc.nfa[idx25],
                   sc.tb[idx25], sc.r[idx25])
        for j in 1:length(metrics)
            content[i + 1, j + 1] = @sprintf("%.3f", 100.0 * metrics[j])
        end
    end
    cells = [Cell(content[i, j]; bold=(i == 1),
                  halign=(j == 1 ? :left : :right))
             for i in axes(content, 1), j in axes(content, 2)]
    tbl = Table(cells; header=1)
    open(joinpath(outdir, "figure5_summary.html"), "w") do io
        show(io, MIME"text/html"(), tbl)
    end
    open(joinpath(outdir, "figure5_summary.tex"), "w") do io
        show(io, MIME"text/latex"(), tbl)
    end
    return nothing
end

"""Export all four source-code κP scenarios (or two paper-label scenarios)."""
function export_csv(outdir::AbstractString, p::Parameters, sim)
    open(joinpath(outdir, "figure5_timeseries.csv"), "w") do io
        headers = ["year", "no_fdi_ca_pct", "no_fdi_nfa_pct",
                   "no_fdi_tb_pct", "no_fdi_rate_pct", "fdi_rate_pct"]
        for (j, _) in enumerate(sim.scenarios)
            append!(headers, ["fdi$(j)_ca_pct", "fdi$(j)_nfa_pct",
                              "fdi$(j)_tb_pct"])
        end
        println(io, join(headers, ","))
        t = sim.times
        r_fdi = sim.assets.r[1:length(sim.post)]
        for i in eachindex(t)
            is_pre = i <= p.n_pre
            k = i - p.n_pre
            values = Float64[t[i]]
            if is_pre
                append!(values, [0.0, 0.0, 0.0, 100.0 * p.r_pre,
                                 100.0 * p.r_pre])
                for _ in sim.scenarios
                    append!(values, [0.0, 0.0, 0.0])
                end
            else
                append!(values, [100.0 * sim.no.ca[k],
                                 100.0 * sim.no.nfa[k],
                                 100.0 * sim.no.tb[k],
                                 100.0 * sim.no.r[k],
                                 100.0 * r_fdi[k]])
                for sc in sim.scenarios
                    append!(values, [100.0 * sc.ca[k],
                                     100.0 * sc.nfa[k],
                                     100.0 * sc.tb[k]])
                end
            end
            println(io, join((@sprintf("%.10f", v) for v in values), ","))
        end
    end
    return nothing
end

"""Record original-source versus printed-legend discrepancy for auditability."""
function export_metadata(outdir::AbstractString, p::Parameters, c, sim)
    open(joinpath(outdir, "figure5_metadata.txt"), "w") do io
        println(io, "Caballero, Farhi, Gourinchas (2008): Figure 5")
        println(io, "Source: prog_UR_inv_final.gau, lines 191-192 and 256 / 314")
        @printf(io, "deltaU=%.12f\ndeltaR=%.12f\nr_no_fdi=%.12f\n",
                c.δu, c.δr, c.r_no)
        @printf(io, "kappaP_lower_bound=%.12f (%.6f%%)\n",
                c.κp_min, 100.0 * c.κp_min)
        @printf(io, "kappaP_upper_bound=%.12f (%.6f%%)\n",
                c.κp_max, 100.0 * c.κp_max)
        println(io, "Mode: ", sim.paper_label_values ? "paper legend values" : "GAUSS numerical values")
        println(io, "Simulated kappaP values: ",
                join([@sprintf("%.9f", sc.κp) for sc in sim.scenarios], ", "))
        println(io, "WARNING: the source GAUSS code computes kappaP=8.129032...%,")
        println(io, "         but its legend and the paper say 5%. This script does not")
        println(io, "         relabel 8.129...% as 5%.")
        println(io, "         With kappaP=5%, the lower private-gains condition")
        println(io, "         (Assumption 6) fails in the calibrated model.")
        println(io, "         The GAUSS endpoints are boundary cases of the strict inequalities.")
    end
    return nothing
end

"""Single-file replication workflow with numerical diagnostics and exports."""
function main(args::Vector{String}=ARGS)
    paper_label_values = "--paper-label-values" in args
    destinations = filter(x -> x != "--paper-label-values", args)
    length(destinations) <= 1 ||
        throw(ArgumentError("use at most one output directory"))
    outdir = isempty(destinations) ? joinpath(@__DIR__, "output_fig05") :
             abspath(only(destinations))
    mkpath(outdir)

    p = Parameters()
    c = calibrate(p)
    sim = simulate(p, c; paper_label_values)
    verify_replication(p, c, sim)

    fig = create_figure(p, c, sim)
    save(joinpath(outdir, "figure5.png"), fig; px_per_unit=2)
    save(joinpath(outdir, "figure5.pdf"), fig)
    export_table(outdir, p, c, sim)
    export_csv(outdir, p, sim)
    export_metadata(outdir, p, c, sim)

    @printf("Figure 5 calibration: δU=%.8f δR=%.8f r_no_fdi=%.8f\n",
            c.δu, c.δr, c.r_no)
    @printf("FDI rate: impact %.6f%%, year 25 %.6f%%\n",
            100.0 * sim.assets.r[1],
            100.0 * sim.assets.r[round(Int, 25 / p.dt) + 1])
    @printf("GAUSS κP lower/upper: %.6f%% / %.6f%%\n",
            100.0 * c.κp_min, 100.0 * c.κp_max)
    if !paper_label_values
        println("SOURCE DISCREPANCY: GAUSS legend says 5% but evaluates 8.129...% (details in metadata).")
    else
        println("PAPER-LEGEND MODE: κP=5% violates the calibrated bilateral-gains condition.")
    end
    println("Structural/numerical checks passed. Outputs: ", outdir)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
