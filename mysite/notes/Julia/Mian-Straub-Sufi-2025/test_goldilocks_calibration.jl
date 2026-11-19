# Research Task V: original steady-state targets vs native Julia calibration.
# Run from the project root: julia --project=. .\test_goldilocks_calibration.jl
# No GitHub Actions, extra directories or Python subprocesses are needed.

using Test
using LinearAlgebra

include(joinpath(@__DIR__, "GoldilocksCore.jl"))
include(joinpath(@__DIR__, "GoldilocksHouseholds.jl"))
include(joinpath(@__DIR__, "GoldilocksCalibration.jl"))

using .GoldilocksCore
using .GoldilocksHouseholds
using .GoldilocksCalibration

# Explicit original references extracted ONCE from the official .pkl files;
# they are test expectations, never inputs to the calibration algorithm.
const OFFICIAL = Dict(
    (:us, 1, :bu) => (beta=0.855730439505922, parameter=0.286218366471545, MPC=0.657735750297225),
    (:us, 1, :bu_log) => (beta=0.855769785625964, parameter=0.286189120706256, MPC=0.657735750297225),
    (:us, 1, :olg) => (beta=1.21648971747207, parameter=0.782934690937993, MPC=0.655717082166859),
    (:us, 1, :olg_alt) => (beta=0.95, parameter=0.219064504446466, MPC=0.655717082158654),
    (:us, 1, :ha) => (beta=0.942310570242577, parameter=0.219049525744074, MPC=0.648018992900645),
    (:jp, 1, :bu) => (beta=0.959154157655562, parameter=0.079893208347106, MPC=0.680965414815832),
    (:jp, 1, :bu_log) => (beta=0.959174605428394, parameter=0.0798704505776565, MPC=0.680965414815832),
    (:jp, 1, :olg) => (beta=1.05517887350295, parameter=0.922582642027038, MPC=0.66858558548643),
    (:jp, 1, :olg_alt) => (beta=0.97, parameter=0.0807245819773701, MPC=0.668585585484778),
    (:jp, 1, :ha) => (beta=0.981762569127977, parameter=0.299117303496998, MPC=0.665596915548181),
    (:jp, 4, :bu) => (beta=0.98948683205027, parameter=0.0205716602447097, MPC=0.671762452359789),
    (:jp, 4, :bu_log) => (beta=0.989502301511836, parameter=0.0205482852523202, MPC=0.671762452359789),
    (:jp, 4, :olg) => (beta=1.01365674254795, parameter=0.979904636364207, MPC=0.661856279721733),
    (:jp, 4, :olg_alt) => (beta=0.992414117281496, parameter=0.0209564287341407, MPC=0.661856279722871),
    (:jp, 4, :ha) => (beta=0.995312737340786, parameter=0.29762697621627, MPC=0.661103584433518),
)

const TARGETS = Dict(
    (:us,1) => (B=1.0, C=0.67, mu=0.63852231185058,
                r=-0.01990196078431382, alpha=0.14959901960784272,
                kappa=0.06880706233915822, labor_shifter=1.0900043898),
    (:jp,1) => (B=2.38, C=0.634, mu=0.659560434884364,
                r=-0.00599102691924213, alpha=0.153982,
                kappa=0.14710702, labor_shifter=0.9258836847),
    (:jp,4) => (B=9.52, C=0.634, mu=0.659570959728938,
                r=-0.0014994379215589, alpha=0.1539955,
                kappa=0.036776755, labor_shifter=0.9258876357),
)

@testset "GoldilocksCalibration: US/Japan native steady-state calibration" begin

    @testset "Common macro targets and reference construction" begin
        @test_throws ArgumentError country_setup(:us,4)
        @test_throws ArgumentError country_setup(:unknown,1)
        @test_throws ArgumentError country_setup(:jp,12)
        for ((country,periods),target) in TARGETS
            s=country_setup(country,periods)
            p=s.core.production
            st=s.state
            @test isapprox(st.B,target.B;atol=1e-13)
            @test isapprox(st.r_ante,target.r;atol=1e-12)
            @test isapprox(s.core.mu,target.mu;atol=1e-11)
            @test isapprox(p.alpha,target.alpha;atol=2e-8)
            @test isapprox(s.core.wage.kappa,target.kappa;atol=1e-12)
            @test isapprox(s.core.wage.labor_shifter,target.labor_shifter;atol=2e-7)
            prod=production_steady_state(st.Y,st.r_ante,p)
            Z=prod.w-(s.core.fiscal.G+st.B*st.r_ante)
            @test isapprox((1-s.core.mu)*s.saver_consumption+s.core.mu*Z,
                           st.Y-s.core.fiscal.G-prod.I;atol=1e-10)
            @test isapprox(prod.N,1;atol=1e-10)
            @test isapprox(prod.K,2periods;atol=1e-9)
            @test isapprox(st.Y-s.core.fiscal.G-prod.I,target.C;atol=1e-10)
        end
    end

    @testset "Steep HA-style asset demand: residual-based root accuracy" begin
        # A 1e-10-wide beta bracket need not imply accurate asset clearing:
        # dA/dbeta can be of order 10^4 in a near-unit-root household model.
        beta_star = 0.981762569127977
        steep_demand(b) = 1e4 * (b - beta_star)
        beta_root = GoldilocksCalibration._bisect(
            steep_demand, 0.90, 1.01;
            ftol=1e-9, xtol=1e-14, maxiter=100)
        @test abs(steep_demand(beta_root)) < 5e-8
        @test abs(beta_root - beta_star) < 5e-12
    end

    @testset "Original SSJ lottery extrapolation is explicit" begin
        policy = reshape([0.1, 1.5, 2.1], 1, 3)
        grid = [0.0, 1.0, 2.0]
        @test_throws DomainError GoldilocksHouseholds._lottery_coordinates(policy, grid)
        idx, weight = GoldilocksHouseholds._lottery_coordinates(policy, grid;
            source_extrapolation=true)
        @test idx[1, 3] == 2
        @test weight[1, 3] < 0  # Source-exact upper linear extrapolation
        @test all(isfinite, weight)
    end

    @testset "Source reference household moments at official parameters" begin
        # Reconstruct all baseline savers using source parameters, rather than
        # the calibrated routine itself, to check steady-state asset clearing.
        for ((country,periods,model),ref) in OFFICIAL
            s=country_setup(country,periods)
            h=calibrated_household(s,model,ref.parameter;
                    beta=model === :bu ? ref.beta : nothing)
            if model === :ha
                @test h.source_lottery_extrapolation
            end
            b=baseline_equilibrium(s,h)
            if model === :ha
                # Regression: the HA inner beta root must clear assets to
                # economic residual accuracy even at official outer EIS.
                # A beta-only bisection tolerance formerly left 1e-7 errors.
                @test abs(b.eq.asset_mkt) <= 5e-8
            end
            @test isapprox(h.beta,ref.beta; atol=model === :ha ? 3e-5 : 2e-8)
            @test isapprox(b.eq.B,s.state.B;atol=1e-10)
            @test isapprox(b.eq.A,s.state.B+b.eq.j;atol=model === :ha ? 1e-4 : 1e-7)
            @test isapprox(b.eq.C,s.state.Y-s.core.fiscal.G-b.eq.I;
                           atol=model === :ha ? 1e-4 : 1e-7)
            @test isapprox(b.eq.MPC,ref.MPC;atol=model === :ha ? 2e-3 : 1e-5)
            if model === :ha
                @test isapprox(sum(b.eq.D), 1.0; atol=1e-9)
                @test b.eq.negative_distribution_mass >= -1e-5
                if country === :jp && periods == 4
                    # At the official quarterly-Japan HA solution, the
                    # source policy is off-grid and creates a tiny signed
                    # tail mass; do not mistake it for a valid probability.
                    @test maximum(b.eq.a) > 500.0
                end
            end
            @test abs(b.eq.fisher)<1e-8
            @test abs(b.eq.piw_cond)<1e-7
        end
    end

    @testset "General equilibrium vs household partial derivatives" begin
        s=country_setup(:us,1)
        ref=OFFICIAL[(:us,1,:bu)]
        h=calibrated_household(s,:bu,ref.parameter;beta=ref.beta)
        b=baseline_equilibrium(s,h)
        derivative=household_derivatives(b.eq,h,b.core)
        @test isfinite(derivative.A_Z)
        @test isfinite(derivative.C_Z)
        @test isfinite(derivative.C_r)
        @test derivative.C_Z>0
        @test isapprox(derivative.ratio,b.eq.A/b.eq.C;atol=8e-4)
        ge=ge_elasticity_finite(s,h)
        implicit=ge_elasticity_implicit(s,h)
        @test isapprox(ge.phi,s.phi_target;atol=3e-4)
        @test isapprox(implicit.phi,ge.phi;atol=1e-3)
        @test ge.residual_norm<2e-8
    end

    @testset "Full re-calibration: five models × three economies/frequencies" begin
        for (country,periods) in ((:us,1),(:jp,1),(:jp,4))
            s=country_setup(country,periods)
            results=Dict{Symbol,Any}()
            for model in (:bu,:bu_log,:olg,:olg_alt,:ha)
                # Finite difference nested GE for analytical models, IFT
                # at HA outer iterations, followed by original finite GE check.
                method=model === :ha ? :implicit : :finite
                result=calibrate_model(s,model;method)
                results[model]=result
                ref=OFFICIAL[(country,periods,model)]
                v=calibration_values(result)
                @test isfinite(result.phi)
                @test abs(result.phi-s.phi_target)<5e-5
                @test abs(result.phi_finite-s.phi_target)<1e-3
                @test abs(v.asset_mkt)<2e-4
                if model === :ha
                    @test abs(v.asset_mkt) <= 5e-8
                end
                @test abs(v.goods_mkt)<2e-4
                @test isapprox(v.beta,ref.beta;atol=model===:ha ? 2e-3 : 8e-3)
                @test isapprox(result.calibrated_parameter,ref.parameter;
                               atol=model===:ha ? 3e-2 : 3e-2)
                @test isapprox(v.MPC,ref.MPC;atol=model===:ha ? 8e-3 : 3e-3)
                @test check_calibration(s,result;tol=1e-3)
                if model === :bu
                    @test abs(result.ratio_error)<1e-4
                end
                if country === :jp
                    @test v.i==0
                    @test s.core.monetary.zlb
                else
                    @test !s.core.monetary.zlb
                end
                println("CALIBRATION ", country,"/",periods,"/",model,
                        " beta=",v.beta," outer=",result.calibrated_parameter,
                        " phi=",result.phi," residual=",v.asset_mkt)
            end
            # LaTeX outputs: these are generated by SummaryTables, not
            # hand-coded LaTeX or pre-filled saved table values.
            reference_file=write_calibration_values(s,results;directory=@__DIR__)
            @test isfile(reference_file)
            @test filesize(reference_file)>100
            names=make_calibration_tables(s,results;directory=@__DIR__)
            @test isfile(names.common_file)
            @test isfile(names.model_file)
            @test filesize(names.common_file)>100
            @test filesize(names.model_file)>100
        end
    end
end

println("GoldilocksCalibration: tests finished; compare the output above with the official references.")
