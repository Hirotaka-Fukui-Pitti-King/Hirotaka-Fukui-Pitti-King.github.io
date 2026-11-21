module GoldilocksSourceSSJ

# Fidelity-oriented translation of the embedded sequence-jacobian package's
# JacobianDict, FactoredJacobianDict and Block.solve_impulse_nonlinear algebra.
#
# IMPORTANT SCOPE: The source-specific economic block Jacobians, their
# nested SolvedBlock nonlinear evaluators, and the full HA backward fake-news
# preparation must still be supplied by the model port. This module does NOT
# itself implement the complete Figure 10 solver. Do not claim replication
# merely because a function in this module converges on a toy problem.

using LinearAlgebra

export SourceJacobian, SourceFactoredJacobian, pack_source_jacobian,
       unpack_source_paths, pack_source_paths, source_linear_apply,
       source_compose, source_solvedblock_jacobian,
       source_fixed_jacobian_nonlinear, source_fake_news_jacobian,
       SourceNonlinearResult

"""A JacobianDict with ORIGINAL ordered variable-block storage (one T×T block per pair)."""
struct SourceJacobian
    outputs::Vector{Symbol}
    inputs::Vector{Symbol}
    T::Int
    blocks::Dict{Tuple{Symbol,Symbol},Matrix{Float64}}

    function SourceJacobian(outputs::AbstractVector{Symbol},
                            inputs::AbstractVector{Symbol}, T::Integer,
                            blocks::AbstractDict)
        T > 0 || throw(ArgumentError("positive time horizon required"))
        os, ins = collect(outputs), collect(inputs)
        length(unique(os)) == length(os) || throw(ArgumentError("duplicate outputs"))
        length(unique(ins)) == length(ins) || throw(ArgumentError("duplicate inputs"))
        b = Dict{Tuple{Symbol,Symbol},Matrix{Float64}}()
        for ((o,i),block) in blocks
            o in os && i in ins || throw(ArgumentError("unrecognized Jacobian key $((o,i))"))
            size(block) == (T,T) || throw(DimensionMismatch("Jacobian $((o,i)) must be T×T"))
            all(isfinite, block) || throw(ArgumentError("nonfinite Jacobian block $((o,i))"))
            b[(o,i)] = Matrix{Float64}(block)
        end
        return new(os, ins, Int(T), b)
    end
end

SourceJacobian(outputs::AbstractVector{Symbol}, inputs::AbstractVector{Symbol}, T::Integer) =
    SourceJacobian(outputs,inputs,T,Dict{Tuple{Symbol,Symbol},Matrix{Float64}}())

"""Pack in source Python's `JacobianDict.pack`: variable-major, then time-major."""
function pack_source_jacobian(j::SourceJacobian)
    T=j.T
    H=zeros(Float64,T*length(j.outputs),T*length(j.inputs))
    for (o_idx,o) in enumerate(j.outputs), (i_idx,i) in enumerate(j.inputs)
        if haskey(j.blocks,(o,i))
            @views H[((o_idx-1)*T+1):(o_idx*T),
                     ((i_idx-1)*T+1):(i_idx*T)] .= j.blocks[(o,i)]
        end
    end
    return H
end

"""Pack ImpulseDict-like paths in the same ordered layout as original Python."""
function pack_source_paths(paths::AbstractDict, names::AbstractVector{Symbol}, T::Integer)
    T > 0 || throw(ArgumentError("positive time horizon required"))
    v=Vector{Float64}(undef,length(names)*T)
    for (k,name) in enumerate(names)
        haskey(paths,name) || throw(KeyError(name))
        path=paths[name]
        length(path)==T || throw(DimensionMismatch("path $name must have length $T"))
        @views v[((k-1)*T+1):(k*T)] .= path
    end
    all(isfinite,v) || throw(ArgumentError("nonfinite packed impulse"))
    return v
end

function unpack_source_paths(x::AbstractVector, names::AbstractVector{Symbol}, T::Integer)
    length(x)==length(names)*T || throw(DimensionMismatch("unpack dimensions"))
    return Dict{Symbol,Vector{Float64}}(
        name=>Float64.(x[((k-1)*T+1):(k*T)]) for (k,name) in enumerate(names))
end

"""Apply a source Jacobian to input paths without changing variable ordering."""
function source_linear_apply(j::SourceJacobian, paths::AbstractDict)
    return unpack_source_paths(pack_source_jacobian(j) *
             pack_source_paths(paths,j.inputs,j.T),j.outputs,j.T)
end

"""Chain-rule composition, equivalent to Python `JacobianDict.compose`."""
function source_compose(left::SourceJacobian, right::SourceJacobian)
    left.T==right.T || throw(DimensionMismatch("composition horizons differ"))
    T=left.T
    out=Dict{Tuple{Symbol,Symbol},Matrix{Float64}}()
    for o in left.outputs, i in right.inputs
        acc=zeros(Float64,T,T)
        found=false
        for intermediate in left.inputs
            if haskey(left.blocks,(o,intermediate)) &&
               haskey(right.blocks,(intermediate,i))
                acc .+= left.blocks[(o,intermediate)] * right.blocks[(intermediate,i)]
                found=true
            end
        end
        if found
            out[(o,i)] = acc
        end
    end
    return SourceJacobian(left.outputs,right.inputs,T,out)
end

"""A source-compatible FactoredJacobianDict; factorize ONCE and reuse."""
struct SourceFactoredJacobian{F}
    unknowns::Vector{Symbol}
    targets::Vector{Symbol}
    T::Int
    factorization::F
end

function SourceFactoredJacobian(j::SourceJacobian)
    length(j.outputs)==length(j.inputs) ||
        throw(DimensionMismatch("source requires equal unknowns and targets"))
    H=pack_source_jacobian(j)
    factorization=lu(H;check=true)
    return SourceFactoredJacobian(copy(j.inputs),copy(j.outputs),j.T,factorization)
end

"""Translate FactoredJacobianDict.apply: returns -H_U^{-1} times targets."""
function source_linear_apply(f::SourceFactoredJacobian, targets::AbstractDict)
    b=pack_source_paths(targets,f.targets,f.T)
    x=-(f.factorization\b)
    all(isfinite,x) || error("nonfinite source LU correction")
    return unpack_source_paths(x,f.unknowns,f.T)
end

"""Implicit-function elimination as used by Python SolvedBlock Jacobians.

`raw` holds derivatives of the *unsolved* internal economic block.
This is the exact Jacobian chain rule: O_Z - O_U (G_U\\G_Z).
No nonlinear dynamic equation is replaced or approximated here.
"""
function source_solvedblock_jacobian(raw::SourceJacobian,
                inner_unknowns::AbstractVector{Symbol},
                inner_targets::AbstractVector{Symbol},
                external_inputs::AbstractVector{Symbol},
                requested_outputs::AbstractVector{Symbol})
    T=raw.T
    length(inner_unknowns)==length(inner_targets) ||
        throw(DimensionMismatch("inner system must be square"))
    all(x -> x in raw.inputs, inner_unknowns) &&
    all(x -> x in raw.inputs, external_inputs) &&
    all(x -> x in raw.outputs, inner_targets) &&
    all(x -> x in raw.outputs, requested_outputs) ||
        throw(ArgumentError("missing source SolvedBlock inputs or outputs"))
    # Build subsets without changing their original row/column order.
    function subset(os,ins)
        pairs=Dict{Tuple{Symbol,Symbol},Matrix{Float64}}(
            (o,i)=>raw.blocks[(o,i)] for o in os for i in ins
            if haskey(raw.blocks,(o,i)))
        return SourceJacobian(collect(os),collect(ins),T,pairs)
    end
    H_U=pack_source_jacobian(subset(inner_targets,inner_unknowns))
    H_Z=pack_source_jacobian(subset(inner_targets,external_inputs))
    O_U=pack_source_jacobian(subset(requested_outputs,inner_unknowns))
    O_Z=pack_source_jacobian(subset(requested_outputs,external_inputs))
    implied=O_Z-O_U*(lu(H_U)\H_Z)
    blocks=Dict{Tuple{Symbol,Symbol},Matrix{Float64}}()
    for (oi,o) in enumerate(requested_outputs), (ii,i) in enumerate(external_inputs)
        blocks[(o,i)]=Matrix(implied[((oi-1)*T+1):(oi*T),
                                   ((ii-1)*T+1):(ii*T)])
    end
    return SourceJacobian(collect(requested_outputs),collect(external_inputs),T,blocks)
end

"""Construct the HA Jacobian from original HetBlock.build_F and J_from_F.

Supply the CURLY objects generated by the household backward fake-news pass;
this function implements steps 3 and 4 of the exact embedded algorithm.
`curlyY` has length T; `curlyD` is T×ne×na; `curlyE` is (T-1)×ne×na.
"""
function source_fake_news_jacobian(curlyY::AbstractVector,
                                  curlyD::AbstractArray{<:Real,3},
                                  curlyE::AbstractArray{<:Real,3})
    T=length(curlyY)
    size(curlyD,1)==T || throw(DimensionMismatch("curlyD time axis"))
    size(curlyE,1)==T-1 || throw(DimensionMismatch("curlyE time axis"))
    size(curlyD)[2:3]==size(curlyE)[2:3] ||
        throw(DimensionMismatch("curlyD and curlyE household grid"))
    F=zeros(Float64,T,T)
    @views F[1,:] .= curlyY
    # Same contraction as Python: curlyE.reshape((T-1,-1)) @
    # curlyD.reshape((T,-1)).T, with Julia's column-major reshaping.
    ngrid=size(curlyD,2)*size(curlyD,3)
    Eflat=reshape(permutedims(curlyE,(2,3,1)),ngrid,T-1)
    Dflat=reshape(permutedims(curlyD,(2,3,1)),ngrid,T)
    @views F[2:end,:] .= transpose(Eflat)*Dflat
    J=copy(F)
    for col in 2:T
        @views J[2:end,col] .+= J[1:end-1,col-1]
    end
    return (;F,J)
end

struct SourceNonlinearResult
    unknowns::Dict{Symbol,Vector{Float64}}
    residuals::Dict{Symbol,Vector{Float64}}
    history::Vector{Dict{Symbol,Float64}}
    iterations::Int
    T::Int
end

"""Exact outer iteration in the embedded `Block.solve_impulse_nonlinear`.

`evaluate(U)` must itself call the model's ORIGINAL nested SolvedBlock
nonlinear algorithms, respecting distinct initial and terminal steady states.
`H_U` must be assembled at TERMINAL steady state via source block Jacobians.
This routine deliberately does not provide line search, continuation or GMRES.
"""
function source_fixed_jacobian_nonlinear(evaluate::Function,
                                         H_U::SourceJacobian;
                                         tol::Real=1e-5,maxit::Int=50,
                                         verbose::Bool=true)
    isfinite(tol) && tol>0 || throw(ArgumentError("positive finite tolerance required"))
    maxit>0 || throw(ArgumentError("positive maxit required"))
    factored=SourceFactoredJacobian(H_U)
    T=H_U.T
    unknowns=Dict{Symbol,Vector{Float64}}(k=>zeros(T) for k in H_U.inputs)
    history=Dict{Symbol,Float64}[]
    for iter in 0:maxit-1
        raw=evaluate(unknowns)
        residuals=Dict{Symbol,Vector{Float64}}(
             k=>Float64.(raw[k]) for k in H_U.outputs)
        errors=Dict{Symbol,Float64}(k=>maximum(abs,residuals[k]) for k in H_U.outputs)
        push!(history,errors)
        all(isfinite,values(errors)) || error("nonfinite source nonlinear GE target")
        if verbose
            @info "Source SSJ fixed-J iteration" iteration=iter errors
        end
        if all(v<tol for v in values(errors))
            return SourceNonlinearResult(unknowns,residuals,history,iter,T)
        end
        correction=source_linear_apply(factored,residuals)
        for k in H_U.inputs
            unknowns[k] .+= correction[k]
        end
    end
    error("Source fixed-J nonlinear system did not converge after $maxit iterations: $(last(history))")
end

end # module GoldilocksSourceSSJ
