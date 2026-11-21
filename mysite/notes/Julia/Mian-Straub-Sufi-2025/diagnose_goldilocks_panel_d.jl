# LOCAL DIAGNOSTIC ONLY: source GE Jacobian vs reduced *old Julia* augmented Jacobian.
# This script does NOT implement the source SSJ Jacobian algorithm. The Schur
# complement is used only to localize existing economic/timing discrepancies.
# Do NOT use its approximated Jacobian as the production Figure 10 solver.
using LinearAlgebra
using SparseArrays
using Printf

include(joinpath(@__DIR__,"GoldilocksCore.jl"))
include(joinpath(@__DIR__,"GoldilocksHouseholds.jl"))
include(joinpath(@__DIR__,"GoldilocksCalibration.jl"))
include(joinpath(@__DIR__,"GoldilocksDeficitDebt.jl"))
include(joinpath(@__DIR__,"GoldilocksTransition.jl"))
include(joinpath(@__DIR__,"GoldilocksSpendingShocks.jl"))
using .GoldilocksCore
using .GoldilocksHouseholds
using .GoldilocksCalibration
using .GoldilocksDeficitDebt
using .GoldilocksTransition
using .GoldilocksSpendingShocks

function diagnostic_panel_d(;model::Symbol=:bu,T::Int=10,
                              root::AbstractString=@__DIR__)
    model in (:bu,:bu_log) || throw(ArgumentError("Panel D requires BU or BU-log"))
    T==10 || throw(ArgumentError("benchmark supplied for T=10 only"))
    e=spending_experiment(:D,model;mode=:official,T)
    p=build_transition_problem(e)
    p_linear=TransitionProblem(e.core_terminal,e.household,e.terminal,e.terminal,
                    fill(e.core_terminal.fiscal.deficit_ss,T),
                    fill(e.core_terminal.fiscal.psi,T),model)
    outer=e.core_terminal.monetary.zlb ? [:B,:Y,:r_ante,:piw] : [:B,:Y,:r_ante]
    inner=filter(s -> !(s in outer),p.names)
    n=length(p.names)
    pos=p.pos
    ix=(t,j)->GoldilocksTransition._ix(p,t,j)
    row_outer=[ix(t,j) for j in 1:length(outer) for t in 1:T]
    row_inner=[ix(t,j) for j in (length(outer)+1):n for t in 1:T]
    col_outer=[ix(t,pos[v]) for v in outer for t in 1:T]
    col_inner=[ix(t,pos[v]) for v in inner for t in 1:T]
    length(row_outer)==length(col_outer) || error("GE system not square")
    length(row_inner)==length(col_inner) || error("internal system not square")

    x_steady=GoldilocksTransition._starting_vector(p_linear)
    println("Computing old Julia FULL augmented Jacobian at TERMINAL steady state...")
    J=transition_jacobian(p_linear,x_steady)
    block_FU=Matrix(J[row_outer,col_outer])
    block_FW=Matrix(J[row_outer,col_inner])
    block_GU=Matrix(J[row_inner,col_outer])
    block_GW=Matrix(J[row_inner,col_inner])
    fac=lu(block_GW)
    # Mathematical Schur complement; this is diagnostic, NOT source SSJ.
    H_reduced=block_FU-block_FW*(fac\block_GU)
    python_jac=joinpath(root,"figure10_ssj_D_$(model)_T10_ge_jacobian.tsv")
    isfile(python_jac) || error("Missing original Python diagnostic: $python_jac")
    H_py=zeros(Float64,length(row_outer),length(col_outer))
    for (ln,line) in enumerate(eachline(python_jac))
        ln==1 && continue
        f=split(line,'\t')
        H_py[parse(Int,f[1])+1,parse(Int,f[4])+1]=parse(Float64,f[7])
    end
    difference=H_reduced-H_py
    largest=argmax(abs.(difference))
    i,j=Tuple(largest)
    @printf("Original GE H: (%d,%d); max |HJulia-Hsource| = %.9e\n",
            size(H_py,1),size(H_py,2),maximum(abs,difference))
    @printf("Element max absolute error: row %d, col %d, Julia=%.12g, Python=%.12g, diff=%.4g\n",
            i-1,j-1,H_reduced[i,j],H_py[i,j],difference[i,j])

    # Evaluate source-style OUTER targets after solving only internal unknowns,
    # holding U=0. The original SolvedBlocks instead solve each internal
    # subsystem in DAG order. This joint solve is a diagnostic of economics.
    x=GoldilocksTransition._starting_vector(p)
    r_before=transition_residual(p,x)
    println("Old Julia augmented max residual with ALL inners at terminal anchor: ",norm(r_before,Inf))
    err=Inf
    for it in 1:50
        r=transition_residual(p,x)
        err=norm(r[row_inner],Inf)
        err<1e-9 && break
        x[col_inner] .-= fac\r[row_inner]
        all(isfinite,x) || error("nonfinite internal diagnostic path")
    end
    err<1e-8 || error("internal frozen-J diagnostic did not converge: $err")
    r=transition_residual(p,x)
    target_names=e.core_terminal.monetary.zlb ?
        (:asset_mkt,:fisher,:deficit_diff,:piw_cond) :
        (:asset_mkt,:labor_foc,:deficit_diff)
    vals=Dict{Symbol,Vector{Float64}}()
    for (j,k) in enumerate(target_names)
        vals[k]=[r[ix(t,j)] for t in 1:T]
    end
    refpath=joinpath(root,"figure10_ssj_D_$(model)_T10_residuals.tsv")
    isfile(refpath) || error("Missing original Python diagnostics: $refpath")
    original=Dict{Symbol,Vector{Float64}}(k=>fill(NaN,T) for k in target_names)
    for (ln,line) in enumerate(eachline(refpath))
        ln==1 && continue
        f=split(line,'\t')
        f[1]=="0" || continue
        k=Symbol(f[2]);k in target_names || continue
        original[k][parse(Int,f[3])+1]=parse(Float64,f[4])
    end
    output=joinpath(root,"figure10_ssj_D_$(model)_T10_julia_diagnostic.tsv")
    open(output,"w") do io
        println(io,"target\ttime0\tpython_initial_residual\tjulia_reduced_residual\tabs_error")
        for k in target_names
            m=maximum(abs.(vals[k].-original[k]))
            @printf("Target %-16s source max %.8g Julia max %.8g max-point-error %.8g\n",
                string(k),maximum(abs.(original[k])),maximum(abs.(vals[k])),m)
            for t in 1:T
                py=original[k][t];jl=vals[k][t]
                println(io,join((k,t-1,py,jl,abs(py-jl)),'\t'))
            end
        end
    end
    println("Saved diagnostic comparison: $output")
    return (;source_H=H_py,reduced_H=H_reduced,
            source_residual=original,julia_residual=vals,
            output)
end

if abspath(PROGRAM_FILE)==@__FILE__
    model=Symbol(get(ENV,"GOLDILOCKS_SOURCE_MODEL","bu"))
    diagnostic_panel_d(;model)
end
