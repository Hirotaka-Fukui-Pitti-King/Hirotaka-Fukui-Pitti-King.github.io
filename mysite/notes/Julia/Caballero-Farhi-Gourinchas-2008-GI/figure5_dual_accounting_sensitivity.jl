# Caballero, Farhi, and Gourinchas (2008), Figure 5:
# published 5% label versus computed GAUSS 8.129% and accounting audit.
#
# Place THIS FILE next to figure5_fdi.jl. The original source is not modified.
# Run from PowerShell:
#   julia --project=. .\figure5_dual_accounting_sensitivity.jl
#   julia --project=. .\figure5_dual_accounting_sensitivity.jl C:\output\directory
#
# Dependencies are precisely those already used by figure5_fdi.jl:
# CairoMakie.jl, SummaryTables.jl, plus the Julia standard library Printf.
#
# The 5% scenario imposes FDI mechanically but is NOT a voluntary bilateral
# equilibrium under the published Assumption 6 and unchanged calibration.
# The 8.129% and 12% scenarios are indifference boundary cases; 9.5% is
# provided as a strictly admissible interior comparison.
#
# CA_GAUSS = goods TB + r*NFA (the original GAUSS plotted series).
# CA_adjusted = d(NFA)/dt, including FDI option transactions in Appendix F.
# The discrete unanticipated valuation jump at t=0 is excluded from both flows.

include(joinpath(@__DIR__, "figure5_fdi.jl"))

"""Compute Appendix-F-consistent capital-flow accounting, without editing GAUSS source series."""
function fdi_accounting(p::Parameters, c, sc, assets, post_t)
    n = length(post_t)
    wedge = Vector{Float64}(undef, n)
    ca_adjusted = Vector{Float64}(undef, n)
    derivative_ca = Vector{Float64}(undef, n)
    for i in eachindex(post_t)
        t = post_t[i]
        vo = assets.v_old[i]
        vu = assets.v_u[i]
        rr = assets.r[i]
        xnew = c.xr - old_r_share(t, p, c)

        # Appendix F: g^n N_R v_Rn / X_U - κP X_Rn / X_U,
        # where v_Rn=(δU/δR)*v_Ro, and κ=0 in the published Figure 5.
        wedge[i] = p.gn * (c.δu / c.δr) * vo * c.xr / p.x_u -
                   sc.κp * xnew / p.x_u
        ca_adjusted[i] = sc.ca[i] + wedge[i]

        # Independent check: d(W_U - V_U)/dt / X_U,
        # where w_U=W_U/X_U, v_U=V_U/X_U and NFA_U/X_U=w_U-v_U.
        wdot = wealth_derivative(t, sc.wu[i], vo, rr, sc.κp, p, c)
        vdot = (rr - p.gz) * vu - c.δu
        derivative_ca[i] = wdot - vdot + p.g * sc.nfa[i]
    end
    residual = maximum(abs.(ca_adjusted .- derivative_ca))
    residual < 5e-10 || error("FDI current-account identity failed: residual=$residual")
    return (; wedge, ca_adjusted, derivative_ca, residual)
end

"""Compute strict balanced-growth limits for κ=0; distinguish CA and TB."""
function fdi_limit(p::Parameters, c, κp::Float64)
    r = p.r_pre
    wealth_flow = 1.0 - c.δu + p.gn / (p.θ * p.x_u) - κp * c.xr / p.x_u
    wu = wealth_flow / (p.θ + p.g - r)
    vu = 1.0 / p.θ
    nfa = wu - vu
    tb = 1.0 - p.θ * wu
    ca_gauss = tb + r * nfa
    ca_adjusted = p.g * nfa
    wedge = ca_adjusted - ca_gauss
    return (; ca_gauss, ca_adjusted, nfa, tb, wedge, r)
end

"""Classify the voluntary private-gains condition without treating endpoints as strict."""
function price_status(κp::Float64, c)
    if κp < c.κp_min - 1e-10
        return "Imposed FDI; seller rejects"
    elseif isapprox(κp, c.κp_min; atol=1e-10, rtol=0.0)
        return "R indifference boundary"
    elseif isapprox(κp, c.κp_max; atol=1e-10, rtol=0.0)
        return "U indifference boundary"
    elseif c.κp_min < κp < c.κp_max
        return "Strict bilateral gains"
    else
        return "Outside Assumption 6"
    end
end

"""Construct one common asset path and four wealth/accounting scenarios."""
function comparison_cases(p::Parameters)
    c = calibrate(p)
    assets = solve_backward_assets(p, c)
    n = round(Int, p.horizon / p.dt) + 1
    post_t = assets.t[1:n]
    no = simulate_no_fdi(p, c, post_t)
    specifications = (
        ("Published label: 5%", 0.05),
        ("GAUSS lower: 8.129%", c.κp_min),
        ("Interior: 9.5%", 0.095),
        ("GAUSS upper: 12%", c.κp_max),
    )
    @assert c.κp_min < 0.095 < c.κp_max
    cases = map(specifications) do (label, price)
        sc = solve_forward_wealth(p, c, assets, n, price)
        acct = fdi_accounting(p, c, sc, assets, post_t)
        lim = fdi_limit(p, c, price)
        (; label, price, status=price_status(price, c), sc, acct, lim)
    end
    return (; c, assets, post_t, no, cases)
end

"""Save case-by-case comparisons using SummaryTables.jl (HTML and LaTeX)."""
function write_comparison_table(outdir::AbstractString, p::Parameters, comp)
    headings = ("κP", "Time", "CA GAUSS (%)", "CA adjusted (%)",
                "FDI wedge (pp)", "NFA (%)", "TB (%)")
    rows = Vector{NTuple{7,String}}()
    push!(rows, headings)
    for case in comp.cases
        for (period, idx) in (("Impact 0+", 1),
                              ("Year 5", round(Int, 5.0 / p.dt) + 1),
                              ("Year 25", round(Int, 25.0 / p.dt) + 1))
            push!(rows, (case.label, period,
                         @sprintf("%.3f", 100.0 * case.sc.ca[idx]),
                         @sprintf("%.3f", 100.0 * case.acct.ca_adjusted[idx]),
                         @sprintf("%.3f", 100.0 * case.acct.wedge[idx]),
                         @sprintf("%.3f", 100.0 * case.sc.nfa[idx]),
                         @sprintf("%.3f", 100.0 * case.sc.tb[idx])))
        end
        lim = case.lim
        push!(rows, (case.label, "Strict limit",
                     @sprintf("%.3f", 100.0 * lim.ca_gauss),
                     @sprintf("%.3f", 100.0 * lim.ca_adjusted),
                     @sprintf("%.3f", 100.0 * lim.wedge),
                     @sprintf("%.3f", 100.0 * lim.nfa),
                     @sprintf("%.3f", 100.0 * lim.tb)))
    end
    mat = [Cell(rows[i][j]; bold=(i == 1),
                halign=(j <= 2 ? :left : :right))
           for i in eachindex(rows), j in eachindex(headings)]
    tbl = Table(mat; header=1)
    open(joinpath(outdir, "figure5_dual_accounting_summary.html"), "w") do io
        show(io, MIME"text/html"(), tbl)
    end
    open(joinpath(outdir, "figure5_dual_accounting_summary.tex"), "w") do io
        show(io, MIME"text/latex"(), tbl)
    end
    return nothing
end

"""Save a post-shock time-series CSV, including both current-account measures."""
function write_comparison_csv(outdir::AbstractString, comp)
    labels = ("paper5", "gauss8129", "interior95", "gauss12")
    open(joinpath(outdir, "figure5_dual_accounting_timeseries.csv"), "w") do io
        fields = ["year", "no_fdi_ca_pct", "no_fdi_nfa_pct"]
        for label in labels
            append!(fields, ("$(label)_ca_gauss_pct", "$(label)_ca_adjusted_pct",
                             "$(label)_fdi_wedge_pp", "$(label)_nfa_pct",
                             "$(label)_tb_pct"))
        end
        println(io, join(fields, ","))
        for i in eachindex(comp.post_t)
            nums = Float64[comp.post_t[i], 100 * comp.no.ca[i],
                           100 * comp.no.nfa[i]]
            for case in comp.cases
                append!(nums, (100 * case.sc.ca[i],
                               100 * case.acct.ca_adjusted[i],
                               100 * case.acct.wedge[i],
                               100 * case.sc.nfa[i], 100 * case.sc.tb[i]))
            end
            println(io, join((@sprintf("%.10f", a) for a in nums), ","))
        end
    end
    return nothing
end

"""Plot the GAUSS and Appendix-F accounts separately with Okabe–Ito colors."""
function plot_comparison(p::Parameters, comp)
    fig = Figure(size=(1400, 950), fontsize=16, backgroundcolor=:white)
    ax1 = Axis(fig[1, 1], title="A. GAUSS current account / U output",
               xlabel="Years", ylabel="Percent")
    ax2 = Axis(fig[1, 2], title="B. Adjusted CA = d(NFA)/dt / U output",
               xlabel="Years", ylabel="Percent")
    ax3 = Axis(fig[2, 1], title="C. Net foreign assets / U output",
               xlabel="Years", ylabel="Percent")
    ax4 = Axis(fig[2, 2], title="D. FDI-accounting wedge / U output",
               xlabel="Years", ylabel="Percentage points")
    axlist = (ax1, ax2, ax3, ax4)
    for ax in axlist
        xlims!(ax, 0.0, p.horizon)
        ax.xgridvisible = false
        ax.ygridvisible = false
        hlines!(ax, [0.0]; color=OKABE_ITO.black, linestyle=:dot,
                linewidth=1.0)
    end
    colors = (OKABE_ITO.vermilion, OKABE_ITO.blue,
              OKABE_ITO.green, OKABE_ITO.purple)
    patterns = (:solid, :dash, :solid, :dot)
    for (j, case) in enumerate(comp.cases)
        for (ax, v) in ((ax1, case.sc.ca),
                        (ax2, case.acct.ca_adjusted),
                        (ax3, case.sc.nfa),
                        (ax4, case.acct.wedge))
            lines!(ax, comp.post_t, 100.0 .* v;
                   color=colors[j], linewidth=2.8,
                   linestyle=patterns[j], label=case.label)
        end
    end
    for ax in (ax1, ax2)
        lines!(ax, comp.post_t, 100.0 .* comp.no.ca;
               color=OKABE_ITO.black, linewidth=1.8,
               linestyle=:dash, label="No FDI")
    end
    axislegend(ax1, position=:rb, framevisible=false, labelsize=11)
    axislegend(ax3, position=:rb, framevisible=false, labelsize=11)
    return fig
end

"""Run the four-case comparison without overwriting the official Figure 5 files."""
function main_comparison(args::Vector{String}=ARGS)
    length(args) <= 1 || throw(ArgumentError("Provide at most one output directory."))
    outdir = isempty(args) ? joinpath(@__DIR__, "output_fig05_sensitivity") :
             abspath(only(args))
    mkpath(outdir)
    p = Parameters()
    comp = comparison_cases(p)
    write_comparison_table(outdir, p, comp)
    write_comparison_csv(outdir, comp)
    fig = plot_comparison(p, comp)
    save(joinpath(outdir, "figure5_dual_accounting.png"), fig; px_per_unit=2)
    save(joinpath(outdir, "figure5_dual_accounting.pdf"), fig)
    open(joinpath(outdir, "figure5_dual_accounting_notes.txt"), "w") do io
        @printf(io, "GAUSS computed lower price: %.9f%%\n", 100 * comp.c.κp_min)
        println(io, "Published figure labels: 5% and 12% (5% outside Assumption 6).")
        println(io, "Default Figure 5 script remains unmodified.")
        println(io, "CA_GAUSS = goods TB + r*NFA; CA_adjusted = d(NFA)/dt, excluding jump valuation.")
        for case in comp.cases
            @printf(io, "%s: κP=%.9f%%; %s; max accounting residual %.3e\n",
                    case.label, 100 * case.price, case.status, case.acct.residual)
        end
    end
    for case in comp.cases
        idx = round(Int, 25.0 / p.dt) + 1
        @printf("%s: 25y CA_GAUSS=%+.4f%%, CA_adjusted=%+.4f%%, NFA=%+.4f%%; %s\n",
                case.label, 100 * case.sc.ca[idx],
                100 * case.acct.ca_adjusted[idx],
                100 * case.sc.nfa[idx], case.status)
    end
    println("Appendix-F accounting checks passed. Outputs: ", outdir)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_comparison()
end
