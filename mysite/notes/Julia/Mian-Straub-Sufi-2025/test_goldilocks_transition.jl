# Local-only tests. These do not use or trigger GitHub Actions.
using Test
using LinearAlgebra
using SparseArrays
include(joinpath(@__DIR__,"GoldilocksCore.jl"))
include(joinpath(@__DIR__,"GoldilocksHouseholds.jl"))
include(joinpath(@__DIR__,"GoldilocksCalibration.jl"))
include(joinpath(@__DIR__,"GoldilocksDeficitDebt.jl"))
include(joinpath(@__DIR__,"GoldilocksTransition.jl"))
using .GoldilocksCore
using .GoldilocksHouseholds
using .GoldilocksCalibration
using .GoldilocksDeficitDebt
using .GoldilocksTransition

@testset "GoldilocksTransition: source timing, equilibrium, colored Jacobian" begin
    @test transition_shocks(4;permanent=0.001,transitory=0.005,rho=0.9) ≈
          [0.006,0.0055,0.00505,0.004645]
    @test length(transition_unknowns(:bu,false))==10
    @test length(transition_unknowns(:bu,true))==11
    @test length(transition_unknowns(:olg,false))==10
    @test length(transition_unknowns(:ha,true))==9
    @test_throws ArgumentError transition_unknowns(:olg_alt,false)

    # With ZERO shock, the terminal-anchored path must equal the original
    # steady state at every date, including t=1 and t=T where initial and
    # terminal conditions enter different dates of the model.
    for (country,model) in ((:us,:bu),(:us,:bu_log),(:us,:olg),
                            (:jp,:bu),(:jp,:bu_log),(:jp,:olg))
        inp=figure8_calibration(country,model;mode=:official)
        core=inp.core;eq=inp.anchor
        F=core.fiscal
        fiscal=FiscalParams(;G=F.G,periods=F.periods,adjust_tax_rate=F.adjust_tax_rate,
                             psi=0.0,deficit_ss=eq.deficit,B_ss=eq.B,
                             tax_shock=F.tax_shock,tau_slope=F.tau_slope)
        c=CoreParams(;production=core.production,wage=core.wage,
                     monetary=core.monetary,fiscal,mu=core.mu,
                     delta_b=core.delta_b)
        T=7
        p=TransitionProblem(c,inp.household,eq,eq,
                   fill(eq.deficit,T),zeros(T),model)
        x=GoldilocksTransition._starting_vector(p)
        r=transition_residual(p,x)
        @test maximum(abs.(r))<1e-5
        z=transition_outcomes(p,x)
        @test maximum(abs(zz.goods_mkt) for zz in z)<1e-5
        @test maximum(abs(zz.asset_mkt) for zz in z)<1e-5
        @test maximum(abs(zz.B-eq.B) for zz in z)<1e-12
        @test maximum(abs(zz.Y-eq.Y) for zz in z)<1e-12
        @test all(isfinite, [zz.r for zz in z])
        @test p.core.monetary.zlb==(country===:jp)
        # Full, independently differentiated directional derivative checks
        J=transition_jacobian(p,x)
        v=[sin(0.1*i) for i in eachindex(x)]
        h=1e-7
        direct=(transition_residual(p,x.+h.*v).-r)./h
        @test norm(J*v-direct,Inf)<2e-3
    end

    # HA policy is obtained with BACKWARD EGM and its distribution evolves
    # FORWARD from the initial distribution (not a static asset demand).
    for country in (:us,:jp)
        inp=figure8_calibration(country,:ha;mode=:official)
        core=inp.core;eq=inp.anchor
        F=core.fiscal
        fiscal=FiscalParams(;G=F.G,periods=F.periods,
                   psi=0.0,deficit_ss=eq.deficit,B_ss=eq.B,
                   adjust_tax_rate=F.adjust_tax_rate,
                   tax_shock=F.tax_shock,tau_slope=F.tau_slope)
        c=CoreParams(;production=core.production,wage=core.wage,
                     monetary=core.monetary,fiscal,mu=core.mu,
                     delta_b=core.delta_b)
        p=TransitionProblem(c,inp.household,eq,eq,
                            fill(eq.deficit,3),zeros(3),:ha)
        x=GoldilocksTransition._starting_vector(p)
        result=transition_residual(p,x)
        @test maximum(abs.(result))<1e-4
        zz=transition_outcomes(p,x)
        @test maximum(abs(z.A-eq.A) for z in zz)<1e-4
        @test maximum(abs(z.C-eq.C) for z in zz)<1e-4
        # The right preconditioner freezes HA aggregates, but the true
        # nonlinear residual still recomputes all EGM and distribution paths.
        _,hhfixed=GoldilocksTransition._ha_aggregates(p,reshape(x,length(p.names),p.horizon))
        frozen=GoldilocksTransition._ha_frozen_residual(p,x,hhfixed)
        @test norm(frozen-result,Inf)<1e-9
        Jf=GoldilocksTransition._ha_frozen_jacobian(p,x)
        @test size(Jf)==(length(x),length(x))
        @test all(isfinite,nonzeros(Jf))
        vv=[sin(0.19*i) for i in eachindex(x)]
        hdiff=1e-7
        fd=(GoldilocksTransition._ha_frozen_residual(p,x.+hdiff.*vv,hhfixed).-frozen)./hdiff
        @test norm(Jf*vv-fd,Inf)<5e-3
    end
end

@testset "HA matrix-free linear-solver controls" begin
    @test GoldilocksTransition._ha_gmres_forcing(0.04829) == 0.12
    @test GoldilocksTransition._ha_gmres_forcing(1e-8) == 0.02
    A=[3.0 0.5 -0.1; -0.25 2.0 0.2; 0.1 -0.2 1.5]
    b=[1.0,-2.0,3.0]
    x=GoldilocksTransition._gmres(v -> A*v,b;
                                    tol=1e-8,restart=3,max_cycles=2)
    @test norm(A*x-b,Inf)<1e-7
end
println("GoldilocksTransition core dynamic/timing tests completed.")
