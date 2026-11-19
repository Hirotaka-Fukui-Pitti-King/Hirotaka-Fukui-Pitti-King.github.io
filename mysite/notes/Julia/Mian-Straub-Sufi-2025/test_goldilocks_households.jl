# Research Task IV — all files at project ROOT, no src/ or test/ folders needed.
# PowerShell (from project directory):
#    julia --project=. .\test_goldilocks_households.jl
# The goldilocks original-Python reference numbers were obtained by executing
# its actual SimpleBlock/HetBlock steady_state methods on specified inputs.

using Test
using LinearAlgebra
using ForwardDiff

include(joinpath(@__DIR__, "GoldilocksCore.jl"))
include(joinpath(@__DIR__, "GoldilocksHouseholds.jl"))
using .GoldilocksCore
using .GoldilocksHouseholds

const BASE_IN = (Z=0.8, r=0.01, r_ante=0.01, pure_profit=0.0, mu=0.2,
                 Y=1.0, B=1.0)

@testset "GoldilocksHouseholds: household models and original Python parity" begin

    @testset "HTM and aggregation" begin
        htm = hand_to_mouth(0.8)
        @test htm.C_htm == 0.8
        @test htm.A_htm == 0
        @test htm.MPC_htm == 1
        agg = aggregate_households(0.8, 0.2, 1.1, 2.5, 0.3)
        @test agg.C ≈ 1.04
        @test agg.A ≈ 2.0
        @test agg.MPC ≈ 0.44
        @test agg.one_minus_mu ≈ 0.8
        @test savers_income(0.8, 0.1, 0.2) ≈ 0.925
        @test_throws DomainError savers_income(1.0, 0.0, 1.0)
        @test_throws DomainError aggregate_households(1.0, -0.1, 1.0, 1.0, 0.2)
    end

    @testset "BU-quadratic: original steady-state reference" begin
        b = 0.95
        r = 0.01
        v0 = (1 - b*(1+r)) / 0.82
        bu = BUQuadratic(beta=b, eis=1.0, vp_0=v0,
                         vp_slope=0.005, A_ss=2.0, lambda_bu=0.9)
        s = bu(BASE_IN)
        # from original blocks.hh_bu_ss.steady_state with identical inputs
        @test isapprox(s.C_ha, 0.8200000000000001; atol=1e-10)
        @test isapprox(s.A_ha, 2.0000000000000053; atol=1e-10)
        @test isapprox(s.MPC_ha, 0.1089108910891089; atol=1e-12)
        @test abs(s.euler) < 1e-10
        @test abs(s.budget) < 1e-10
        @test isapprox(s.C, .2*.8 + .8*s.C_ha; atol=1e-12)
        @test isapprox(s.A, .8*s.A_ha; atol=1e-12)
        @test isapprox(find_lambda_bu(.95,.01,.005,1.,.82),
                       0.9657454333891378; atol=1e-12)
        per = bu_quadratic_period(bu; C=s.C_ha, C_next=s.C_ha,
                                  A=s.A_ha, A_lag=s.A_ha, Z=BASE_IN.Z,
                                  r=r, r_ante=r, mu=.2)
        @test abs(per.euler) < 1e-10
        @test abs(per.budget) < 1e-10
        @test_throws DomainError BUQuadratic(beta=.95, eis=0., vp_0=.1,
                vp_slope=.01, A_ss=2., lambda_bu=.8)
        @test_throws DomainError find_lambda_bu(.95, .01, .005, 0., .82)
    end

    @testset "BU-log: original steady-state reference" begin
        bu = BULog(beta=.95, eis=1.0, vp_slope=.02,
                   A_ss=2.0, lambda_bu=.9)
        s = bu(BASE_IN)
        @test isapprox(s.C_ha, 0.8079800498753118; atol=1e-10)
        @test isapprox(s.A_ha, 0.7980049875311703; atol=1e-10)
        @test isapprox(s.MPC_ha, 0.1089108910891089; atol=1e-12)
        @test abs(s.euler) < 1e-10
        @test abs(s.budget) < 1e-10
        per = bu_log_period(bu; C=s.C_ha, C_next=s.C_ha,
                             A=s.A_ha, A_lag=s.A_ha, Z=.8,
                             r=.01, r_ante=.01, mu=.2)
        @test abs(per.euler) < 1e-10
        @test abs(per.budget) < 1e-10
        @test_throws DomainError BULog(beta=.95,eis=1.,vp_slope=-.01,
                                        A_ss=2.,lambda_bu=.9)
        @test_throws DomainError bu_log_period(bu; C=.5, C_next=.5,
            A=0., A_lag=0., Z=.8, r=.01, r_ante=.01)
    end

    @testset "BU generalized EIS, bounded-vp policy, and AD" begin
        # Set A*=2 as an exactly known equilibrium and do NOT use EIS=1.
        e = 0.7
        beta = .96
        r = .01
        C = .82
        vp0 = (1-beta*(1+r)) / C^(1/e)
        bu = BUQuadratic(beta=beta,eis=e,vp_0=vp0,
                         vp_slope=.005,A_ss=2.,lambda_bu=.9)
        s = bu(BASE_IN)
        @test isapprox(s.A_ha, 2.0; atol=1e-7)
        @test abs(s.euler) < 1e-9
        @test abs(s.budget) < 1e-12
        vpb = BULog(beta=beta,eis=e,
                    vp_slope=(1-beta*(1+r)) * 2 / (2*C^(1/e)),
                    A_ss=2.,lambda_bu=.9)
        sl = vpb(BASE_IN)
        @test isapprox(sl.A_ha, 2.0; atol=1e-7)
        @test abs(sl.euler) < 1e-9
        # ForwardDiff on analytical stationary functions (one-dimensional).
        df = ForwardDiff.derivative(z -> bu_quadratic_steady_state(
            BUQuadratic(beta=.95,eis=1.,vp_0=(1-.95*1.01)/.82,
                        vp_slope=.005,A_ss=2.,lambda_bu=.9),
            z,.01,.01,0.,.2).C_ha, .8)
        @test isfinite(df)
        @test isfinite(ForwardDiff.derivative(z -> bu_log_steady_state(
            BULog(beta=.95,eis=1.,vp_slope=.02,A_ss=2.,lambda_bu=.9),
            z,.01,.01,0.,.2).C_ha,.8))
    end

    @testset "OLG (and identical OLG-alt code path)" begin
        olg = OLGHousehold(beta=.96, survival=.9, zeta=.1)
        x = merge(BASE_IN, (; Z=.9))
        s = olg(x)
        # From the original hh_olg.steady_state under the same calibration.
        @test isapprox(s.C_ha, 0.9221325376884422; atol=1e-10)
        @test isapprox(s.A_ha, 2.2132537688442206; atol=1e-10)
        @test isapprox(s.H, 4.545; atol=1e-10)
        @test isapprox(s.MPC_ha, 0.136; atol=1e-12)
        @test abs(s.H_recursion) < 1e-10
        @test abs(s.budget) < 1e-10
        per = olg_period(olg; A=s.A_ha,A_lag=s.A_ha,
                          H=s.H,H_next=s.H,Z=.9,r=.01,
                          r_ante=.01,mu=.2)
        @test abs(per.H_recursion) < 1e-10
        @test abs(per.budget) < 1e-10
        @test_throws DomainError OLGHousehold(beta=.96,survival=0.,zeta=.1)
        @test isfinite(ForwardDiff.derivative(z -> olg_steady_state(
            olg,z,.01,.01,0.,.2).C_ha,.9))
    end

    @testset "Rouwenhorst, exact asset grid, source discretization" begin
        m = rouwenhorst(.7,.2,3)
        @test size(m.Pi) == (3,3)
        @test all(isapprox.(vec(sum(m.Pi; dims=2)), 1.0; atol=1e-13))
        @test minimum(m.Pi) >= 0
        @test sum(m.pi) ≈ 1.0
        @test isapprox(sum(m.pi .* m.e_grid),1.; atol=1e-12)
        @test isapprox(sqrt(sum(m.pi .* (log.(m.e_grid) .-
                         sum(m.pi .* log.(m.e_grid))).^2)), .2; atol=1e-10)
        @test norm(transpose(m.Pi) * m.pi - m.pi,Inf) < 1e-9
        @test_throws DomainError rouwenhorst(1.0,.2,3)
        @test_throws ArgumentError rouwenhorst(.7,.2,1)
        a = asset_grid(0.,50.,35)
        @test length(a) == 35
        @test a[1] == 0.
        @test a[end] == 50.
        @test issorted(a)
        @test all(diff(a) .> 0)
        @test_throws DomainError asset_grid(1.,1.,35)
        @test_throws ArgumentError asset_grid(0.,50.,1)
    end

    @testset "HA EGM and invariant distribution: original Python reference" begin
        hh = HAHousehold(beta=.96,eis=1.,lumpsum=0.,
                         rho_e=.7,sd_e=.2,n_e=3,min_a=0.,max_a=50.,n_a=35)
        x = (Z=1.,r=.01,r_ante=.01,pure_profit=0.,mu=.2,Y=1.,B=1.)
        s = hh(x)
        # Reference from original blocks.hh_ha.steady_state (source HetBlock)
        # Python, numpy/scipy/numba, identical grids and parameter values:
        @test isapprox(s.C_ha, 1.0012257867529868; atol=1e-5)
        @test isapprox(s.A_ha, 0.12257867415174585; atol=1e-4)
        @test isapprox(s.MPC_ha, 0.5703099683756223; atol=1e-4)
        @test isapprox(s.a[1,3], 0.01167639; atol=2e-5)
        @test isapprox(s.c[1,1], 0.73876427; atol=2e-6)
        @test isapprox(s.D[1,1], 0.20401591; atol=2e-4)
        @test size(s.a) == (3,35)
        @test size(s.c) == (3,35)
        @test size(s.D) == (3,35)
        @test minimum(s.c) > 0
        @test minimum(s.a) >= -1e-12
        @test maximum(s.a) <= 50.0 + 1e-10
        @test abs(sum(s.D) - 1) < 1e-9
        @test minimum(s.D) >= -1e-13
        @test norm(transpose(hh.grids.Pi) * s.Dbeg - s.D,Inf) < 1e-9
        @test s.backward_error < hh.backward_tol
        @test s.distribution_error < hh.forward_tol
        @test abs(s.budget) < 1e-6
        @test isapprox(s.C, .2*1. + .8*s.C_ha; atol=1e-10)
        @test isapprox(s.A, .8*s.A_ha; atol=1e-10)
        @test isapprox(s.MPC,.2 + .8*s.MPC_ha; atol=1e-10)
        @test all(isfinite, s.euler)
        # EGM Euler inequality for agents at the asset constraint;
        # interior conditions have interpolation approximation error.
        unconstrained = s.a .> hh.grids.a_grid[1] + 1e-8
        if any(unconstrained)
            @test maximum(abs.(s.euler[unconstrained])) < 0.1
        end
        @test ha_income(1., [0.8,1.2], .1, .2, .2) ≈
              [(.9 + .2/.8)*.8 + .1, (.9 + .2/.8)*1.2 + .1]
    end

    @testset "HA boundary failures and weighted MPC convention" begin
        hh = HAHousehold(beta=.96,eis=1.,rho_e=.7,sd_e=.2,
                         n_e=3,min_a=0.,max_a=50.,n_a=12)
        @test_throws DomainError ha_policy_steady_state(hh, fill(-2.,3), .01)
        a = fill(60.,3,12)
        @test_throws DomainError ha_stationary_distribution(hh,a)
        grid = hh.grids.a_grid
        cs = repeat(reshape(grid .+ 1, 1, :),3,1)
        pol = repeat(reshape(grid,1,:),3,1)
        m = ha_weighted_mpc(cs,pol,grid,.01,hh.grids.e_grid)
        @test all(isapprox.(m[:,1],hh.grids.e_grid; atol=1e-12))
        @test_throws DomainError HAHousehold(beta=.96,eis=0.,
                                               n_e=3,n_a=12)
    end

    @testset "Official saved US/Japan steady states, all five model variants" begin
        # Ground truth: data/clean/us_ss.pkl and jp_ss.pkl in official ZIP.
        # The values below are the SAVERS' conditional stationary outcomes,
        # NOT a newly solved full general equilibrium in Julia.
        for scenario in (
            (region=:US,
             Z=.73030294117647099, r=-.019901960784313832,
             r_ante=-.019901960784313819, mu=.63852231185057984,
             Ass=8.3822600933187363, lambda=.92800339004230614,
             C=.56347952951532299, A=8.3822600933187399,
             mpc_bu=.053152487903218582,
             q_beta=.8557304395059222, q_vp0=.28625755807280312,
             q_slope=.2862183664715453,
             l_beta=.85576978562596362, l_slope=.28618912070625591,
             o_beta=1.216489717472069, o_surv=.78293469093799317,
             alt_beta=.95, alt_surv=1.0025600010536202,
             alt_zeta=.21906450444646552,
             mpc_olg=.047567999021759011,
             ha_beta=.94231057024257658, ha_eis=.21904952574407402,
             ha_C=.56347964267163109, ha_A=8.3822601287980909,
             ha_mpc=.026271831876216071),
            (region=:JP,
             Z=.66027664406779618, r=-.0059910269192420262,
             r_ante=-.0059910269192421347, mu=.65956043488436422,
             Ass=12.88334391600525, lambda=.9315111194212814,
             C=.58309218385715555, A=12.88334391600533,
             mpc_bu=.062874536701389427,
             q_beta=.95915415765556244, q_vp0=.079905308307252829,
             q_slope=.079893208347106018,
             l_beta=.9591746054283945, l_slope=.079870450577656488,
             o_beta=1.055178873502947, o_surv=.92258264202703777,
             alt_beta=.97, alt_surv=1.0035976421982629,
             alt_zeta=.080724581977370077,
             mpc_olg=.026510287072537664,
             ha_beta=.98176256912797666, ha_eis=.29911730349699822,
             ha_C=.58309242635104774, ha_A=12.88334393137217,
             ha_mpc=.01773143101556595),
        )
            x = (Z=scenario.Z, r=scenario.r,
                 r_ante=scenario.r_ante, pure_profit=0.,
                 mu=scenario.mu,Y=1.,B=1.)
            quad = BUQuadratic(beta=scenario.q_beta,eis=1.,
                               vp_0=scenario.q_vp0,vp_slope=scenario.q_slope,
                               A_ss=scenario.Ass,lambda_bu=scenario.lambda)
            blog = BULog(beta=scenario.l_beta,eis=1.,
                          vp_slope=scenario.l_slope,A_ss=scenario.Ass,
                          lambda_bu=scenario.lambda)
            olg = OLGHousehold(beta=scenario.o_beta,
                               survival=scenario.o_surv,zeta=0.)
            # Authors' OLG-alt calibrations actually have survival > 1.
            alt = OLGHousehold(beta=scenario.alt_beta,
                               survival=scenario.alt_surv,zeta=scenario.alt_zeta)
            for household in (quad,blog,olg,alt)
                result = household(x)
                @test isapprox(result.C_ha,scenario.C; atol=2e-9)
                @test isapprox(result.A_ha,scenario.A; atol=2e-8)
                target_mpc = household isa OLGHousehold ? scenario.mpc_olg : scenario.mpc_bu
                @test isapprox(result.MPC_ha,target_mpc; atol=2e-8)
                @test abs(result.budget) < 1e-9
                if household isa OLGHousehold
                    @test abs(result.H_recursion) < 1e-9
                else
                    @test abs(result.euler) < 1e-9
                end
            end
            ha = HAHousehold(beta=scenario.ha_beta,eis=scenario.ha_eis,
                             rho_e=.9,sd_e=.92,n_e=7,
                             min_a=0.,max_a=500.,n_a=300)
            result = ha(x)
            @test isapprox(result.C_ha,scenario.ha_C; atol=2e-4)
            @test isapprox(result.A_ha,scenario.ha_A; atol=2e-4)
            @test isapprox(result.MPC_ha,scenario.ha_mpc; atol=2e-4)
            @test abs(sum(result.D)-1) < 1e-9
            @test minimum(result.D) >= -1e-13
        end
    end

    @testset "Integration: all households callable from GoldilocksCore" begin
        g = 0.015
        rr = 1.015 / 1.02 - 1 - g
        alpha = 1.3 * ((1 + rr) * (1 + g) - 0.92) * 2 / 1.3
        tfp = 1.3 / 2^alpha
        prod = ProductionParams(alpha=alpha,tfp=tfp,fixed_cost=.3,
                                markup=1.3,delta_k=.08,eps_I=4.,real_growth=g)
        wage = WageParams(kappa=.1,beta=.95,labor_shifter=1.)
        mon = MonetaryParams(pi_trend=.02,real_growth=g)
        fiscal = FiscalParams(G=.14)
        core = CoreParams(production=prod,wage=wage,monetary=mon,
                          fiscal=fiscal,mu=.2,delta_b=0.)
        state = (Y=1., B=1., r_ante=rr, piw=0.)
        households = (
            BUQuadratic(beta=.95,eis=1.,vp_0=.1,vp_slope=.005,
                        A_ss=2.,lambda_bu=.9),
            BULog(beta=.95,eis=1.,vp_slope=.02,
                  A_ss=2.,lambda_bu=.9),
            OLGHousehold(beta=.96,survival=.9,zeta=.1),
            HAHousehold(beta=.96,eis=1.,rho_e=.7,sd_e=.2,
                        n_e=3,min_a=0.,max_a=50.,n_a=35),
        )
        for h in households
            eq = evaluate_steady_state(state,core,h)
            @test isfinite(eq.C)
            @test isfinite(eq.A)
            @test isfinite(eq.MPC)
            @test eq.C > 0
            @test eq.C_ha > 0
            @test isapprox(eq.asset_mkt, eq.A - eq.B - eq.j; atol=1e-12)
            @test isapprox(eq.goods_mkt,
                           eq.C + core.fiscal.G + eq.adj_cost + eq.I - eq.Y;
                           atol=1e-12)
        end
        # This test verifies connectivity, NOT general-equilibrium market clearing.
    end
end

println("GoldilocksHouseholds: test suite finished (see Test Summary).")
