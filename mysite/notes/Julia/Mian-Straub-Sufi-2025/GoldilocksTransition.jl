module GoldilocksTransition

using LinearAlgebra
using SparseArrays
using Printf
using ..GoldilocksCore
using ..GoldilocksHouseholds

export TransitionProblem, TransitionSolution, transition_unknowns,
       transition_residual, transition_outcomes, transition_jacobian,
       transition_ha_households, solve_transition, transition_accuracy,
       transition_shocks, check_transition

"""Exact author-style exogenous paths, indexed from Julia t=1 (=Python t=0)."""
function transition_shocks(T::Integer; permanent=0.0, transitory=0.0, rho=0.9)
    T > 0 || throw(ArgumentError("positive horizon required"))
    return [permanent + transitory * rho^(t-1) for t in 1:T]
end

"""One perfect-foresight experiment, with explicit initial and terminal equilibria.

The terminal anchor is not the same as the initial economy. Both are required
for lead/lag boundary conditions, nominal bond capital gains and the HA
initial distribution / terminal policy function.
"""
struct TransitionProblem{C,H,I,E}
    core::C
    household::H
    initial::I
    terminal::E
    deficit_ss::Vector{Float64}
    psi::Vector{Float64}
    horizon::Int
    model::Symbol
    names::Vector{Symbol}
    pos::Dict{Symbol,Int}
end

function transition_unknowns(model::Symbol, binding::Bool)
    model in (:bu, :bu_log, :olg, :ha) ||
        throw(ArgumentError("unsupported transition household $model"))
    variables = Symbol[:B, :Y, :r_ante]
    binding && push!(variables, :piw)
    append!(variables, (:k, :Q, :j, :J, :q))
    if model in (:bu, :bu_log)
        append!(variables, (:A_ha, :C_ha))
    elseif model === :olg
        append!(variables, (:A_ha, :H))
    end
    return variables
end

function TransitionProblem(core::CoreParams, household, initial::NamedTuple,
                           terminal::NamedTuple, deficit_ss::AbstractVector,
                           psi::AbstractVector, model::Symbol)
    T = length(deficit_ss)
    T > 0 && length(psi) == T || throw(DimensionMismatch("fiscal paths must have equal positive length"))
    names = transition_unknowns(model, core.monetary.zlb)
    pos = Dict{Symbol,Int}(s => j for (j,s) in enumerate(names))
    all(isfinite, deficit_ss) && all(isfinite, psi) ||
        throw(ArgumentError("nonfinite exogenous policy path"))
    return TransitionProblem(core,household,initial,terminal,
            Float64.(deficit_ss),Float64.(psi),T,model,names,pos)
end

@inline _nt(p::TransitionProblem) = length(p.names)
@inline _ix(p::TransitionProblem, t::Int, j::Int) = (t-1)*_nt(p)+j

function _get(p::TransitionProblem, X::AbstractMatrix, t::Int, var::Symbol)
    if t < 1
        return getproperty(p.initial, var)
    elseif t > p.horizon
        return getproperty(p.terminal, var)
    elseif haskey(p.pos,var)
        return X[p.pos[var],t]
    elseif var === :piw
        return p.initial.piw  # fixed outside ZLB as in original unknowns_td
    else
        throw(KeyError(var))
    end
end

function _production(p::TransitionProblem, X::AbstractMatrix, t::Int)
    if t < 1
        return (; w=p.initial.w, N=p.initial.N)
    elseif t > p.horizon
        return (; w=p.terminal.w, N=p.terminal.N)
    end
    c=p.core
    return production_block(_get(p,X,t,:Y),_get(p,X,t-1,:k),
            _get(p,X,t,:k),_get(p,X,t,:Q),_get(p,X,t+1,:Q),
            _get(p,X,t,:r_ante),c.production)
end

@inline function _pi(p::TransitionProblem,X::AbstractMatrix,t::Int)
    if t < 1
        return p.initial.pi
    elseif t > p.horizon
        return p.terminal.pi
    end
    return (1+_get(p,X,t,:piw))*_production(p,X,t-1).w/_production(p,X,t).w - 1
end

# The heterogenous-agent block is globally time-coupled, not a steady-state
# asset-demand evaluation. It performs backward policy EGM and forward
# distribution propagation at each nonlinear residual evaluation.
"""Evaluate the full HA transition, given economy-wide Z and realized r paths."""
function transition_ha_households(hh::HAHousehold, initial::NamedTuple,
                                  terminal::NamedTuple,
                                  Z::AbstractVector, r::AbstractVector,
                                  profit::AbstractVector, mu::Real)
    T=length(Z)
    T==length(r)==length(profit) || throw(DimensionMismatch("HA input path"))
    g=hh.grids
    ne,na=length(g.e_grid),length(g.a_grid)
    Va_next=copy(terminal.Va)
    Va_p=zeros(Float64,ne,na)
    Va=zeros(Float64,ne,na)
    a=zeros(Float64,ne,na)
    c=zeros(Float64,ne,na)
    policies_a=Vector{Matrix{Float64}}(undef,T)
    policies_c=Vector{Matrix{Float64}}(undef,T)
    for t in T:-1:1
        y=ha_income(Z[t],g.e_grid,hh.lumpsum,profit[t],mu)
        mul!(Va_p,g.Pi,Va_next)
        egm_step!(Va,a,c,Va_p,g.a_grid,y,r[t],hh.beta,hh.eis)
        policies_a[t]=copy(a)
        policies_c[t]=copy(c)
        Va_next,Va=Va,Va_next
    end
    Dbeg=copy(initial.Dbeg)
    D=zeros(Float64,ne,na)
    nextDbeg=similar(Dbeg)
    C=zeros(Float64,T);A=zeros(Float64,T)
    mass_error=0.0
    for t in 1:T
        mul!(D,transpose(g.Pi),Dbeg)
        A[t]=sum(D.*policies_a[t])
        C[t]=sum(D.*policies_c[t])
        mass_error=max(mass_error,abs(sum(D)-1))
        if t < T
            ind,wgt=GoldilocksHouseholds._lottery_coordinates(
                  policies_a[t],g.a_grid;
                  source_extrapolation=hh.source_lottery_extrapolation)
            GoldilocksHouseholds._policy_forward!(nextDbeg,D,ind,wgt)
            Dbeg,nextDbeg=nextDbeg,Dbeg
        end
    end
    mass_error < 1e-7 || error("HA transition mass conservation error=$mass_error")
    return (;C_ha=C,A_ha=A,mass_error)
end

# First non-household pass: production, bond capital gains and fiscal incomes.
# The realized long-bond return at t=1 must use last period's ACTUAL expected
# inflation from the initial steady state, not a perfect-foresight expectation
# made after the shock. This is why pi_e_lag is not replaced by pi_t at t=1.
function _economy_input(p::TransitionProblem, X::AbstractMatrix, t::Int)
    c=p.core
    prod=_production(p,X,t)
    pi=_pi(p,X,t)
    pi_next=_pi(p,X,t+1)
    pi_e=pi_next
    pi_e_lag=t==1 ? p.initial.pi_e : pi
    pi_e_next=_pi(p,X,t+2)
    q=_get(p,X,t,:q)
    bond=long_bond_block(_get(p,X,t-1,:q),q,_get(p,X,t+1,:q),
          _get(p,X,t,:r_ante),c.delta_b,pi,pi_e_lag,pi_e,pi_e_next)
    capital=equity_block(_get(p,X,t,:J),_get(p,X,t-1,:j),
                         _get(p,X,t,:j),_get(p,X,t+1,:J),
                         prod.dividend,_get(p,X,t,:r_ante),
                         c.production.risk_premium)
    B=_get(p,X,t,:B); B_lag=_get(p,X,t-1,:B)
    r=combined_return(B_lag,_get(p,X,t-1,:j),
                      bond.rpost_bonds,capital.rpost_equity)
    tax=c.fiscal.G+(1+bond.rpost_bonds)*B_lag-B
    Z=prod.w*prod.N-tax
    deficit=c.fiscal.G-tax
    deficit_diff=p.deficit_ss[t]-p.psi[t]*(B_lag-c.fiscal.B_ss)-deficit
    return merge(prod,bond,capital,(;pi,pi_e,pi_e_lag,pi_e_next,r,
           tax,Z,deficit,deficit_diff,B_annual=B/c.fiscal.periods,
           q,j=_get(p,X,t,:j),J=_get(p,X,t,:J),
           B,Y=_get(p,X,t,:Y),r_ante=_get(p,X,t,:r_ante)))
end

function _ha_aggregates(p::TransitionProblem,X::AbstractMatrix)
    T=p.horizon
    econ=[_economy_input(p,X,t) for t in 1:T]
    if p.model !== :ha
        return econ,nothing
    end
    hh=transition_ha_households(p.household,p.initial,p.terminal,
          [z.Z for z in econ],[z.r for z in econ],
          [z.pure_profit for z in econ],p.core.mu)
    return econ,hh
end

function _period(p::TransitionProblem,X::AbstractMatrix,t::Int,
                 econ_t::NamedTuple,hhpath)
    c=p.core; mu=c.mu
    Z=econ_t.Z; r=econ_t.r; r_ante=econ_t.r_ante
    Aha=zero(r); Cha=zero(r); euler=zero(r); budget=zero(r); Hrec=zero(r)
    if p.model===:bu || p.model===:bu_log
        Aha=_get(p,X,t,:A_ha); Cha=_get(p,X,t,:C_ha)
        Cnext=_get(p,X,t+1,:C_ha)
        hp=p.model===:bu ? bu_quadratic_period : bu_log_period
        h=hp(p.household;C=Cha,C_next=Cnext,A=Aha,
             A_lag=_get(p,X,t-1,:A_ha),Z,r,r_ante,
             pure_profit=econ_t.pure_profit,mu,r_ss=p.terminal.r)
        euler,budget=h.euler,h.budget
    elseif p.model===:olg
        Aha=_get(p,X,t,:A_ha)
        h=olg_period(p.household;A=Aha,A_lag=_get(p,X,t-1,:A_ha),
              H=_get(p,X,t,:H),H_next=_get(p,X,t+1,:H),Z,r,r_ante,
              pure_profit=econ_t.pure_profit,mu)
        Cha=h.C_ha;Hrec=h.H_recursion;budget=h.budget
    elseif p.model===:ha
        Aha=hhpath.A_ha[t];Cha=hhpath.C_ha[t]
    end
    C=mu*Z+(1-mu)*Cha
    A=(1-mu)*Aha
    wage=wage_phillips_block(_get(p,X,t-1,:piw),_get(p,X,t,:piw),
                             _get(p,X,t+1,:piw),econ_t.N,Z,C,c.wage)
    mon=monetary_block(econ_t.pi,_pi(p,X,t+1),r_ante,c.monetary)
    markets=market_clearing(A,econ_t.B,C,c.fiscal.G,econ_t.Y,
                             _get(p,X,t,:j),econ_t.adj_cost,econ_t.I)
    return merge(econ_t,(;A,C,A_ha=Aha,C_ha=Cha,euler,budget,H_recursion=Hrec,
                          piw=_get(p,X,t,:piw)),wage,mon,markets)
end

function _target_vector(p::TransitionProblem,z::NamedTuple)
    out=Float64[z.asset_mkt]
    if p.core.monetary.zlb
        append!(out,(z.fisher,z.deficit_diff,z.piw_cond))
    else
        append!(out,(z.labor_foc,z.deficit_diff))
    end
    append!(out,(z.invcond,z.valcond,z.jvaluation,z.Jvaluation,z.qres))
    if p.model===:bu || p.model===:bu_log
        append!(out,(z.euler,z.budget))
    elseif p.model===:olg
        append!(out,(z.H_recursion,z.budget))
    end
    length(out)==_nt(p) || error("non-square transition system")
    return out
end

"""Evaluate the complete non-linear GE transition residual; no static household shortcut."""
function transition_residual(p::TransitionProblem,x::AbstractVector)
    length(x)==_nt(p)*p.horizon || throw(DimensionMismatch("transition state length"))
    X=reshape(x,_nt(p),p.horizon)
    econ,hhpath=_ha_aggregates(p,X)
    R=Vector{Float64}(undef,length(x))
    for t in 1:p.horizon
        z=_period(p,X,t,econ[t],hhpath)
        R[((t-1)*_nt(p)+1):(t*_nt(p))]=_target_vector(p,z)
    end
    return R
end

"""Return all time paths at economic levels, not deviations from terminal steady state."""
function transition_outcomes(p::TransitionProblem,x::AbstractVector)
    X=reshape(x,_nt(p),p.horizon)
    econ,hhpath=_ha_aggregates(p,X)
    z=[_period(p,X,t,econ[t],hhpath) for t in 1:p.horizon]
    return z
end

"""Evaluate HA transition equations with *fixed* aggregate household paths.

This is exclusively a local preconditioner building block. The actual HA
nonlinear residual and all Krylov Jacobian-vector products still recompute
backward EGM policies and the forward asset distribution across the full path.
"""
function _ha_frozen_residual(p::TransitionProblem,x::AbstractVector,hhpath)
    p.model===:ha || throw(ArgumentError("frozen household residual is HA-only"))
    length(x)==_nt(p)*p.horizon || throw(DimensionMismatch("HA transition state length"))
    X=reshape(x,_nt(p),p.horizon)
    R=Vector{Float64}(undef,length(x))
    for t in 1:p.horizon
        econ=_economy_input(p,X,t)
        z=_period(p,X,t,econ,hhpath)
        R[((t-1)*_nt(p)+1):(t*_nt(p))]=_target_vector(p,z)
    end
    return R
end

"""Sparse local HA Jacobian with household aggregates held fixed.

This intentionally omits the global household derivative and MUST NOT replace
`transition_residual` or the full HA Jacobian-vector product. Its sole purpose
is to right-precondition the true, nonlocal HA Newton system. A five-date
coloring is valid because the remaining non-household blocks have at most
2-period lead/lag dependence.
"""
function _ha_frozen_jacobian(p::TransitionProblem,x::AbstractVector;
                             step::Float64=1e-6)
    p.model===:ha || throw(ArgumentError("HA local preconditioner is HA-only"))
    step>0 || throw(ArgumentError("positive finite-difference step required"))
    T=p.horizon
    n=_nt(p)
    _,hhpath=_ha_aggregates(p,reshape(x,n,T))
    base=_ha_frozen_residual(p,x,hhpath)
    rows=Int[]
    cols=Int[]
    vals=Float64[]
    for var in 1:n, color in 0:4
        trial=copy(x)
        for t in (color+1):5:T
            j=_ix(p,t,var)
            trial[j]+=step*max(1.0,abs(x[j]))
        end
        changed=_ha_frozen_residual(p,trial,hhpath)
        for t in (color+1):5:T
            j=_ix(p,t,var)
            h=trial[j]-x[j]
            for tt in max(1,t-2):min(T,t+2), eq in 1:n
                k=_ix(p,tt,eq)
                dv=(changed[k]-base[k])/h
                if dv!=0.0
                    push!(rows,k)
                    push!(cols,j)
                    push!(vals,dv)
                end
            end
        end
    end
    J=sparse(rows,cols,vals,length(x),length(x))
    all(isfinite,nonzeros(J)) || error("nonfinite HA frozen-block preconditioner")
    return J
end

"""Build the exact sparse local Jacobian for BU or OLG models with 5-date coloring.

The source's price/wage inflation terms require up to two leads/lags. All
local unknowns (including consumption and capital values) are represented as
columns; no static household approximation is made. The HA distribution
changes nonlocally, and is deliberately EXCLUDED from this local Jacobian.
"""
function transition_jacobian(p::TransitionProblem,x::AbstractVector;
                             step::Float64=1e-6)
    p.model!==:ha || throw(ArgumentError("HA requires a nonlocal sequence-space Jacobian; local coloring invalid"))
    T=p.horizon;n=_nt(p)
    base=transition_residual(p,x)
    rows=Int[];cols=Int[];vals=Float64[]
    for var in 1:n, color in 0:4
        trial=copy(x)
        for t in (color+1):5:T
            j=_ix(p,t,var)
            trial[j]+=step*max(1.0,abs(x[j]))
        end
        changed=transition_residual(p,trial)
        for t in (color+1):5:T
            j=_ix(p,t,var)
            h=trial[j]-x[j]
            for s in max(1,t-2):min(T,t+2), eq in 1:n
                k=_ix(p,s,eq)
                dv=(changed[k]-base[k])/h
                if dv!=0
                    push!(rows,k);push!(cols,j);push!(vals,dv)
                end
            end
        end
    end
    return sparse(rows,cols,vals,length(x),length(x))
end

"""Restarted GMRES on an operator, with an explicit true linear residual check.

The operator may be the full nonlocal HA Jv, or the right-preconditioned
composition `Jv(P\\v)`; only the preconditioner freezes household responses.
`tol` is a RELATIVE 2-norm tolerance for this inner *linear* Newton solve,
not the tolerance on the outer nonlinear economic equilibrium.
"""
function _gmres(A,b;tol=1e-3,restart=40,max_cycles=12)
    0<tol<1 || throw(ArgumentError("GMRES tolerance must be between zero and one"))
    restart>0 && max_cycles>0 || throw(ArgumentError("positive GMRES budgets required"))
    n=length(b)
    x=zeros(Float64,n)
    normb=norm(b)
    normb==0 && return x
    best_relative=Inf
    for cycle in 1:max_cycles
        r=b-A(x)
        beta=norm(r)
        best_relative=min(best_relative,beta/normb)
        beta<=tol*normb && return x
        V=zeros(Float64,n,restart+1)
        H=zeros(Float64,restart+1,restart)
        V[:,1]=r/beta
        g=zeros(Float64,restart+1)
        g[1]=beta
        for j in 1:restart
            w=A(view(V,:,j))
            # Modified Gram-Schmidt, then one re-orthogonalization pass.
            # Numerical HA Jv products are noisier than analytic matrix products.
            for pass in 1:2
                for i in 1:j
                    coeff=dot(view(V,:,i),w)
                    H[i,j]+=coeff
                    w .-= coeff .* view(V,:,i)
                end
            end
            H[j+1,j]=norm(w)
            if H[j+1,j]>0
                V[:,j+1]=w/H[j+1,j]
            end
            y=H[1:j+1,1:j]\g[1:j+1]
            candidate=x+V[:,1:j]*y
            relative=norm(b-A(candidate))/normb
            best_relative=min(best_relative,relative)
            isfinite(relative) || error("nonfinite HA GMRES linear residual")
            if relative<=tol
                return candidate
            end
            if H[j+1,j]<1e-13
                x=candidate
                break
            end
            if j==restart
                x=candidate
            end
        end
    end
    error("HA matrix-free GMRES did not converge: target relative tolerance=$tol, best relative residual=$best_relative, restart=$restart, max_cycles=$max_cycles")
end

"""Inexact-Newton forcing parameter for the inner HA Krylov solve.

An early nonlinear residual of 0.05 does NOT require its linear system to be
solved to relative accuracy 2e-5. Tighten the inner tolerance as the outer
nonlinear equilibrium approaches the requested 1e-5 absolute threshold.
"""
@inline _ha_gmres_forcing(err::Real)=clamp(sqrt(err),0.02,0.12)

function _starting_vector(p::TransitionProblem; kind::Symbol=:terminal)
    base=kind===:terminal ? p.terminal : kind===:initial ? p.initial :
         throw(ArgumentError("start must be :initial or :terminal"))
    x=zeros(Float64,_nt(p)*p.horizon)
    for t in 1:p.horizon
        for (i,var) in enumerate(p.names)
            x[_ix(p,t,i)]=Float64(getproperty(base,var))
        end
    end
    return x
end

"""Full-path nonlinear Newton; sparse direct solve for BU/OLG and genuine
matrix-free Jv for HA. Fails explicitly rather than manufacturing a path.

No local sparse Jacobian is ever substituted for the HA sequence-space
Jacobian. `maxit=50`, `tol=1e-5` match the original script. Without running
Julia, convergence of the full 1000-date HA problem cannot be certified.
"""
function solve_transition(p::TransitionProblem;tol::Real=1e-5,maxit::Int=50,
                          guess::Symbol=:terminal, verbose::Bool=true,
                          line_search_max::Int=20)
    x=_starting_vector(p;kind=guess)
    history=Float64[]
    for it in 1:maxit
        R=transition_residual(p,x)
        err=norm(R,Inf)
        push!(history,err)
        if verbose
            @info "Figure 10 transition iteration" model=p.model T=p.horizon iteration=it max_residual=err
        end
        all(isfinite,R) || error("Nonfinite transition equilibrium residual")
        if err<tol
            return TransitionSolution(p,x,transition_outcomes(p,x),history,it-1,err)
        end
        delta=if p.model===:ha
            # The full HA Jv re-solves backward EGM and forward distributions.
            # A frozen-HA local Jacobian is used ONLY as a right preconditioner.
            # It is not a substitute for the true nonlocal HA derivative.
            h=1e-5/max(1.0,norm(x,Inf))
            xnorm=copy(x)
            operator = v -> begin
                hv=h/max(1.0,norm(v,Inf))
                (transition_residual(p,xnorm .+ hv.*v).-R)./hv
            end
            P=lu(_ha_frozen_jacobian(p,x))
            eta=_ha_gmres_forcing(err)
            # Solve (J*P^(-1))*u = -R, and return delta=P^(-1)*u.
            # GMRES tests the residual in the UNPRECONDITIONED equations.
            rhs=-R
            preconditioned = v -> operator(P\v)
            u=_gmres(preconditioned,rhs;tol=eta,restart=40,max_cycles=12)
            P\u
        else
            J=transition_jacobian(p,x)
            -(lu(J)\R)
        end
        all(isfinite,delta) || error("Nonfinite transition Newton direction")
        accepted=false
        for k in 0:line_search_max
            candidate=x .+ delta./2.0^k
            newR=try transition_residual(p,candidate) catch; nothing end
            if newR!==nothing && all(isfinite,newR) && norm(newR,Inf)<err
                x=candidate
                accepted=true
                break
            end
        end
        accepted || error("Nonlinear transition line search failed at iteration $it, residual=$err")
    end
    error("Nonlinear transition not converged after $maxit Newton steps: $(last(history))")
end

struct TransitionSolution{P,E}
    problem::P
    vector::Vector{Float64}
    periods::Vector{E}
    residual_history::Vector{Float64}
    iterations::Int
    residual_norm::Float64
end

function transition_accuracy(s::TransitionSolution)
    p=s.problem
    allres=transition_residual(p,s.vector)
    goods=maximum(abs(z.goods_mkt) for z in s.periods)
    market=maximum(abs(z.asset_mkt) for z in s.periods)
    fiscal=maximum(abs(z.deficit_diff) for z in s.periods)
    hamass=if p.model===:ha
        _,out=_ha_aggregates(p,reshape(s.vector,_nt(p),p.horizon))
        out.mass_error
    else
        0.0
    end
    return (;max_target_residual=norm(allres,Inf),max_goods_residual=goods,
              max_asset_residual=market,max_fiscal_residual=fiscal,
              max_ha_mass_error=hamass,
              max_terminal_debt_distance=abs(last(s.periods).B-p.terminal.B),
              max_terminal_output_distance=abs(last(s.periods).Y-p.terminal.Y))
end

function check_transition(s::TransitionSolution;tol=1e-5,goods_tol=1e-5)
    a=transition_accuracy(s)
    a.max_target_residual<=tol || error("transition targets residual $(a.max_target_residual)")
    a.max_goods_residual<=goods_tol || error("goods market residual $(a.max_goods_residual)")
    return a
end

end # module GoldilocksTransition
