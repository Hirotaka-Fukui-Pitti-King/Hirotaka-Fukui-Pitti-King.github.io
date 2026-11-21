# Pure local tests of the original SSJ algebra. Never triggers GitHub Actions.
using Test
using LinearAlgebra
include(joinpath(@__DIR__, "GoldilocksSourceSSJ.jl"))
using .GoldilocksSourceSSJ

@testset "Source JacobianDict block-major packing and LU" begin
    T=3
    L=[1.0 0.0 0.0; -0.5 1.0 0.0; 0.0 -0.5 1.0]
    I3=Matrix{Float64}(I,T,T)
    J=SourceJacobian([:target_a,:target_b],[:unknown_a,:unknown_b],T,
         Dict((:target_a,:unknown_a)=>2*I3,
              (:target_a,:unknown_b)=>0.1*L,
              (:target_b,:unknown_a)=>L,
              (:target_b,:unknown_b)=>3*I3))
    H=pack_source_jacobian(J)
    @test size(H)==(6,6)
    @test H[1:3,1:3]≈2*I3
    @test H[4:6,1:3]≈L
    @test H[1:3,4:6]≈0.1*L
    @test H[4:6,4:6]≈3*I3
    factored=SourceFactoredJacobian(J)
    target_paths=Dict(:target_a=>[1.0,2.0,3.0],
                      :target_b=>[3.0,2.0,1.0])
    correction=source_linear_apply(factored,target_paths)
    packed=pack_source_paths(correction,J.inputs,T)
    @test norm(H*packed+pack_source_paths(target_paths,J.outputs,T),Inf)<1e-12
end

@testset "Source SolvedBlock implicit-function differentiation" begin
    T=4
    I4=Matrix{Float64}(I,T,T)
    # G = u-z = 0, outcome = 2u+3z; hence d(outcome)/dz=5I.
    inner=SourceJacobian([:G,:outcome],[:u,:z],T,
         Dict((:G,:u)=>I4,(:G,:z)=>-I4,
              (:outcome,:u)=>2*I4,(:outcome,:z)=>3*I4))
    effective=source_solvedblock_jacobian(inner,[:u],[:G],[:z],[:outcome])
    @test effective.blocks[(:outcome,:z)]≈5*I4
    prod=source_compose(effective,SourceJacobian([:z],[:x],T,
                  Dict((:z,:x)=>2*I4)))
    @test prod.blocks[(:outcome,:x)]≈10*I4
end

@testset "Original fake-news F and J recursion" begin
    T=4
    Y=[1.0,2.0,3.0,4.0]
    D=reshape(collect(1.0:8.0),4,1,2)
    E=reshape([0.5,-0.5,0.25,-0.25,0.75,-0.75],3,1,2)
    result=source_fake_news_jacobian(Y,D,E)
    @test size(result.J)==(T,T)
    @test result.F[1,:]≈Y
    for r in 2:T, c in 1:T
        @test result.F[r,c]≈sum(E[r-1,:,:].*D[c,:,:])
    end
    for c in 2:T
        @test result.J[2:end,c]≈result.F[2:end,c].+result.J[1:end-1,c-1]
    end
end

@testset "Source fixed Jacobian nonlinearity (no line search or refresh)" begin
    T=6
    eye=Matrix{Float64}(I,T,T)
    HU=SourceJacobian([:target],[:u],T,Dict((:target,:u)=>eye))
    z=[0.1/(i+1) for i in 1:T]
    calls=Ref(0)
    evaluate = U -> begin
        calls[]+=1
        Dict(:target=>U[:u] .+ 0.05.*U[:u].^2 .- z)
    end
    result=source_fixed_jacobian_nonlinear(evaluate,HU;tol=1e-10,maxit=20,verbose=false)
    @test result.iterations>0
    @test calls[]==result.iterations+1
    @test maximum(abs,result.residuals[:target])<1e-10
    @test all(v -> v>0,result.unknowns[:u])
    @test length(result.history)==calls[]
    @test_throws ErrorException source_fixed_jacobian_nonlinear(
         _ -> Dict(:target=>fill(1.0,T)), HU;tol=1e-10,maxit=2,verbose=false)
end
println("Source SSJ algebra tests complete; this is not a Figure 10 parity test.")
