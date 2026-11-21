# LOCAL DIAGNOSTIC ONLY. This is NOT the original Sequence-Space Jacobian
# construction: the Julia GE Jacobian is obtained by reducing its old,
# augmented finite-difference system with a Schur complement.
#
# Compares the FIRST, undamped GE Newton correction at zero outer deviation.
# It does NOT modify any economic block, update any solver, or use the Python
# path as a starting value.

include(joinpath(@__DIR__, "diagnose_goldilocks_panel_d.jl"))
using LinearAlgebra
using Printf

function diagnose_first_update(; model::Symbol=:bu, root::AbstractString=@__DIR__)
    model in (:bu, :bu_log) || throw(ArgumentError("Panel D supports BU and BU-log"))
    d = diagnostic_panel_d(; model=model, T=10, root=root)
    targets = (:asset_mkt, :fisher, :deficit_diff, :piw_cond)
    unknowns = (:B, :Y, :r_ante, :piw)
    T = 10
    r_python = vcat((d.source_residual[k] for k in targets)...)
    r_julia = vcat((d.julia_residual[k] for k in targets)...)
    H_python = d.source_H
    H_julia = d.reduced_H

    # Python SSJ initial update and the diagnostic Julia Schur-reduced update.
    delta_python = -(lu(H_python) \ r_python)
    delta_julia = -(lu(H_julia) \ r_julia)
    @printf("%s: GE Jacobian max absolute error: %.10e\n",
            string(model), maximum(abs.(H_julia .- H_python)))
    @printf("%s: outer initial residual max absolute error: %.10e\n",
            string(model), maximum(abs.(r_julia .- r_python)))
    @printf("%s: FIRST GE correction max absolute error: %.10e\n",
            string(model), maximum(abs.(delta_julia .- delta_python)))
    @printf("%s: source linear solve residual: %.10e\n",
            string(model), norm(H_python * delta_python + r_python, Inf))
    @printf("%s: reduced Julia linear solve residual: %.10e\n",
            string(model), norm(H_julia * delta_julia + r_julia, Inf))
    @printf("%s: source Jacobian condition number (2-norm): %.10e\n",
            string(model), cond(H_python))

    # Additional independent check using the original Python-recorded update,
    # when supplied. It is not required for BU-log because the Python H and
    # residual already determine its correction uniquely.
    ref_updates = joinpath(root, "figure10_ssj_D_$(model)_T10_updates.tsv")
    if isfile(ref_updates)
        delta_saved = fill(NaN, length(delta_python))
        byname = Dict(k => i for (i, k) in enumerate(unknowns))
        for (line_no, line) in enumerate(eachline(ref_updates))
            line_no == 1 && continue
            fields = split(line, '\t')
            if fields[1] == "0" && fields[2] == "delta"
                name = Symbol(fields[3])
                index = (byname[name] - 1) * T + parse(Int, fields[4]) + 1
                delta_saved[index] = parse(Float64, fields[5])
            end
        end
        all(isfinite, delta_saved) || error("Incomplete original first-update benchmark")
        @printf("%s: Python LU vs saved Python first update: %.10e\n",
                string(model), maximum(abs.(delta_python .- delta_saved)))
    end

    output = joinpath(root, "figure10_ssj_D_$(model)_T10_first_update_comparison.tsv")
    open(output, "w") do io
        println(io, "unknown\ttime0\tpython_correction\tjulia_diagnostic_correction\tabs_error")
        for (k, name) in enumerate(unknowns), t in 1:T
            index = (k - 1) * T + t
            println(io, join((name, t-1, delta_python[index],
                        delta_julia[index], abs(delta_python[index]-delta_julia[index])), '\t'))
        end
    end
    println("Saved: ", output)
    return (; delta_python, delta_julia, source_H=H_python, reduced_H=H_julia,
             source_residual=r_python, julia_residual=r_julia, output)
end

if abspath(PROGRAM_FILE) == @__FILE__
    model = Symbol(get(ENV, "GOLDILOCKS_SOURCE_MODEL", "bu"))
    diagnose_first_update(; model=model)
end
