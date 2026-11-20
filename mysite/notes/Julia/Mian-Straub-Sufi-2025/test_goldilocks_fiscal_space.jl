# Task VII, Figure 9: source-faithful nested BU-log counterfactuals.
# Run locally: julia --project=. .\test_goldilocks_fiscal_space.jl
# Original Python reference TSVs must be copied into this same directory.
# NEVER use GitHub Actions for code testing/auditing.

using Test
using LinearAlgebra

include(joinpath(@__DIR__,"GoldilocksCore.jl"))
include(joinpath(@__DIR__,"GoldilocksHouseholds.jl"))
include(joinpath(@__DIR__,"GoldilocksCalibration.jl"))
include(joinpath(@__DIR__,"GoldilocksDeficitDebt.jl"))
include(joinpath(@__DIR__,"GoldilocksFiscalSpace.jl"))

using .GoldilocksCore
using .GoldilocksHouseholds
using .GoldilocksCalibration
using .GoldilocksDeficitDebt
using .GoldilocksFiscalSpace

const SOURCE_MAXIMA = Dict(
    :us => (deficit=0.02001401937915126, B=1.0779696716090164,
            count=231),
    :us_high_beta => (deficit=0.028723706432618346,
                      B=1.3938106306663003,count=232),
    :us_high_beta_low_phi => (deficit=0.019885126128179942,
                              B=1.4306049637547094,count=229),
    :us_high_beta_low_phi_zlb => (deficit=0.024053754415159223,
                                  B=1.7392420216319697,count=176),
    :jp => (deficit=0.0243284085757261,
            B=1.7837123868315141,count=177),
)

@testset "GoldilocksFiscalSpace: original five Figure 9 counterfactuals" begin
    @testset "Scenario identity, annual units, original Figure 9 grid" begin
        @test FIGURE9_SCENARIOS == (
            :us,:us_high_beta,:us_high_beta_low_phi,
            :us_high_beta_low_phi_zlb,:jp)
        @test length(FIGURE9_LABELS)==5
        @test_throws ArgumentError figure9_scenarios(;mode=:wrong)
        @test_throws ErrorException GoldilocksFiscalSpace._bracket_phi(
            x->1.0,0.1,0.3)
        @test_throws ArgumentError GoldilocksFiscalSpace._bracket_phi(
            x->x,0.0,0.3)
        @test isfile(joinpath(@__DIR__,"figure9_original_python_calibrations.tsv"))
        @test isfile(joinpath(@__DIR__,"figure9_original_python_reference.tsv"))
    end

    @testset "Official US/Japan inputs and independently re-solved nested calibration" begin
        ss=figure9_scenarios(;mode=:official)
        @test length(ss)==5
        for key in FIGURE9_SCENARIOS
            s=ss[key]; p=official_figure9_reference(key)
            @test s.mode===:official
            @test s.household isa BULog
            @test s.setup.periods==1
            @test isapprox(s.household.beta,p.beta;atol=5e-4,rtol=0)
            @test isapprox(s.household.vp_slope,p.slope;atol=2e-3,rtol=0)
            @test isapprox(s.setup.state.Y,p.Y;atol=6e-4,rtol=0)
            @test isapprox(-log(s.household.beta),p.rho;atol=6e-4,rtol=0)
            @test abs(s.anchor.asset_mkt)<2e-7
            @test abs(s.anchor.goods_mkt)<2e-5
            @test abs(s.anchor.fisher)<2e-5
            @test abs(s.anchor.piw_cond)<2e-5
            @test abs(s.anchor.euler)<2e-6
            @test abs(s.anchor.budget)<2e-6
        end
        @test abs(ss[:us].phi_reference-0.017)<4e-5
        @test abs(ss[:jp].anchor.labor_foc)>0.01 # JP binding-ZLB FOC is nonzero.

        for key in (:us_high_beta,:us_high_beta_low_phi)
            @test abs(ss[key].anchor.labor_foc)<2e-7
            @test isapprox(ss[key].setup.state.B,ss[:jp].setup.state.B;atol=1e-12)
            @test isapprox(ss[key].setup.state.r_ante,
                           ss[:jp].setup.state.r_ante;atol=1e-12)
            @test ss[key].setup.state.piw==ss[:us].setup.state.piw
        end
        for key in (:us_high_beta_low_phi,:us_high_beta_low_phi_zlb)
            @test abs(ss[key].outer_residual)<4e-5
            @test abs(ss[key].phi_local-ss[:us].phi_reference)<4e-5
        end
        @test ss[:us_high_beta].phi_local > ss[:us].phi_reference
        @test abs(ss[:us_high_beta_low_phi_zlb].anchor.piw_cond)<2e-7
        @test abs(ss[:us_high_beta_low_phi_zlb].anchor.labor_foc)>0.05
        @test ss[:us_high_beta_low_phi_zlb].core.monetary.zlb
        @test !ss[:us_high_beta_low_phi].core.monetary.zlb
        @test ss[:us_high_beta_low_phi_zlb].setup.state.piw == ss[:jp].setup.state.piw
        @test ss[:us_high_beta_low_phi_zlb].core.wage.kappa == ss[:jp].core.wage.kappa
        @test ss[:us_high_beta_low_phi_zlb].core.monetary.real_growth == ss[:jp].core.monetary.real_growth
        @test ss[:us_high_beta_low_phi_zlb].core.production.real_growth == ss[:jp].core.production.real_growth
        @test ss[:us_high_beta_low_phi_zlb].core.production.tfp == ss[:us].core.production.tfp
        @test ss[:us_high_beta_low_phi_zlb].core.fiscal.G == ss[:us].core.fiscal.G
        @test ss[:us_high_beta_low_phi_zlb].core.mu == ss[:us].core.mu
        @test ss[:us_high_beta_low_phi_zlb].core.wage.labor_shifter == ss[:us].core.wage.labor_shifter

        # Distinct valid initial guesses for the same 2-D nested root.
        usref=ss[:us]; jpref=ss[:jp]
        anchor1=(; Y=usref.setup.state.Y, B=jpref.setup.state.B,
                   r_ante=jpref.setup.state.r_ante,
                   piw=usref.setup.state.piw)
        alt=GoldilocksFiscalSpace._solve_beta_output(
                usref.setup,usref.setup.core,anchor1,
                usref.household.vp_slope,0.90,0.95;binding=false)
        @test isapprox(alt.household.beta,ss[:us_high_beta].household.beta;atol=3e-7)
        @test isapprox(alt.setup.state.Y,ss[:us_high_beta].setup.state.Y;atol=3e-7)
        @test alt.inner_residual<1e-8
        @test_throws DomainError GoldilocksFiscalSpace._counterfactual_household(
                usref.setup,-0.2,usref.household.vp_slope)
    end

    @testset "Original 120 x 300 grid, equilibrium residuals, source curve maxima" begin
        ss=figure9_scenarios(;mode=:official)
        curves=run_figure9_curves(ss;n_r=120,n_Y=300,strict=true)
        for key in FIGURE9_SCENARIOS
            c=curves[key]; benchmark=SOURCE_MAXIMA[key]
            @test length(c.records)==420
            @test length(c.r_grid)==120
            @test length(c.Y_grid)==300
            @test c.r_grid[end]==1e-4
            @test c.Y_grid[1]==0
            @test all(diff(c.r_grid).>0)
            @test all(diff(c.Y_grid).>0)
            @test isapprox(c.r_grid[1],1/(1+c.core.monetary.pi_trend)-1-c.core.monetary.real_growth;atol=1e-12)
            flex=[p for p in c.records if !ismissing(p) && p.branch===:flex]
            zlb=[p for p in c.records if !ismissing(p) && p.branch===:zlb]
            @test length(flex)==120
            @test length(zlb)>=2
            @test abs(length(flex)+length(zlb)-benchmark.count)<=3
            @test all(p->p.residual<2e-5,flex)
            @test all(p->p.residual<2e-5,zlb)
            @test all(p->abs(p.deficit + p.B * GoldilocksDeficitDebt._candidate(
                   p.B,p.Y,p.r_ante,p.piw,
                   GoldilocksDeficitDebt._regime_core(c.core,p.branch===:zlb),
                   c.household).rpost_bonds)<2e-5,flex)
            mx=grid_maximum(c)
            @test isapprox(mx.maximum_deficit,benchmark.deficit;atol=1e-4)
            @test isapprox(mx.debt_at_maximum,benchmark.B;atol=3e-3)
        end
        mx=scenario_maxima(curves)
        @test mx[:us_high_beta].maximum_deficit > mx[:us].maximum_deficit
        @test mx[:us_high_beta_low_phi].maximum_deficit < mx[:us_high_beta].maximum_deficit
        @test mx[:us_high_beta_low_phi_zlb].maximum_deficit > mx[:us_high_beta_low_phi].maximum_deficit
        @test mx[:jp].maximum_deficit > mx[:us].maximum_deficit

        for key in (:us_high_beta_low_phi_zlb,:jp)
            m=local_zlb_debt_response(curves[key],ss[key].setup.state.Y)
            @test m.dB_ddeficit<0 # Fiscal free-lunch region at Japanese B.
        end
        # The full reference is an independently run ORIGINAL PYTHON model,
        # never image digitization or a saved Julia output curve.
        reference=joinpath(@__DIR__,"figure9_original_python_reference.tsv")
        parity=compare_figure9_python(curves,reference;directory=@__DIR__)
        @test length(parity.rows)==5*420
        for key in FIGURE9_SCENARIOS
            rows=[r for r in parity.rows if r.scenario===key]
            comp=[r for r in rows if r.status===:compared]
            @test length(comp)>=SOURCE_MAXIMA[key].count-5
            @test maximum(r->abs(r.debt_error),comp)<3e-3
            @test maximum(r->abs(r.deficit_error),comp)<1e-4
        end
        tables=make_figure9_tables(ss,curves;directory=@__DIR__,parity)
        @test all(isfile,values(tables))
        @test length(tables)==3
        for key in FIGURE9_SCENARIOS
            path=write_figure9_curve(key,curves[key];directory=@__DIR__)
            @test isfile(path)
            @test length(readlines(path))==421
        end
        @test isfile(write_figure9_calibration(ss;directory=@__DIR__))
    end
end

println("GoldilocksFiscalSpace Figure 9 tests finished (see Test Summary).")
