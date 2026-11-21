# Local-only policy experiment tests. No GitHub Actions.
using Test
using LinearAlgebra
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

const SOURCE_MULTIPLIERS = Dict(
    (:A,:bu) => -41.35093396304689,
    (:A,:bu_log) => -38.770238655187555,
    (:A,:olg) => -39.97989758150471,
    (:A,:ha) => -40.36545630293299,
    (:B,:bu) => 5.768396482498939,
    (:B,:bu_log) => 5.756211906097848,
    (:B,:olg) => 5.7787147434789015,
    (:B,:ha) => 5.754180948640302,
)

@testset "GoldilocksSpendingShocks: exact four source experiments" begin
    @test length(FIGURE10_PANELS)==4
    @test sum(length(FIGURE10_MODELS[p]) for p in FIGURE10_PANELS)==14
    @test FIGURE10_MODELS[:D]==(:bu,:bu_log)
    @test isfile(joinpath(@__DIR__,"figure10_original_python_reference.tsv"))
    @test experiment_config(:A).permanent==0.0001
    @test experiment_config(:B).transitory==0.005
    @test experiment_config(:C).psi==0.018
    @test experiment_config(:D).psi_terminal==-0.023
    @test experiment_config(:D).years==40
    @test_throws ArgumentError experiment_config(:E)
    @test_throws ArgumentError spending_experiment(:D,:ha;T=3)
end

@testset "Figure 10 Python TSV: non-ZLB piw is genuinely missing" begin
    parse_source=GoldilocksSpendingShocks._figure10_python_value
    @test ismissing(parse_source("",:A,:bu,0,:piw))
    @test ismissing(parse_source("  ",:A,:ha,999,:piw))
    @test ismissing(parse_source("",:C,:bu_log,12,:piw))
    @test parse_source("0.00125",:B,:ha,3,:piw)==0.00125
    @test parse_source("-0.020",:A,:bu,4,:r)==-0.020
    @test_throws ArgumentError parse_source("",:B,:bu,0,:piw)
    @test_throws ArgumentError parse_source("",:D,:bu_log,0,:piw)
    @test_throws ArgumentError parse_source("",:A,:bu,0,:deficit)
    @test_throws ArgumentError parse_source("NaN",:A,:bu,0,:deficit)
    @test_throws ArgumentError parse_source("not-a-number",:A,:bu,0,:deficit)
end

@testset "Source terminal steady states and fiscal feedback" begin
    for panel in FIGURE10_PANELS
        for model in FIGURE10_MODELS[panel]
            # T=1000 is essential for parity with the original shock[-1].
            exp=spending_experiment(panel,model;mode=:official,T=1000)
            cfg=experiment_config(panel)
            term=exp.terminal
            @test abs(term.asset_mkt)<5e-7
            @test abs(term.deficit_diff)<5e-7
            @test abs(term.goods_mkt)<1e-5
            @test abs(term.fisher)<1e-5
            @test abs(term.piw_cond)<1e-5
            @test exp.core_terminal.monetary.zlb==(panel in (:B,:D))
            @test length(exp.shock_path)==1000
            @test length(exp.psi_path)==1000
            if panel in (:A,:B,:C)
                @test isapprox(exp.shock_path[1],cfg.permanent+cfg.transitory;atol=1e-14)
                @test isapprox(exp.shock_path[end],cfg.permanent+cfg.transitory*0.9^999;atol=1e-14)
                @test exp.core_terminal.fiscal.psi==cfg.psi
                if panel in (:A,:B)
                    @test isapprox(long_run_multiplier(exp),SOURCE_MULTIPLIERS[(panel,model)];atol=0.1)
                end
            else
                @test exp.core_terminal.fiscal.psi==-0.023
                @test exp.core_initial.fiscal.B_ss==0.0
                @test exp.core_initial.fiscal.deficit_ss==0.0
                @test exp.core_initial.wage.kappa==5*figure8_calibration(:us,model;mode=:official).core.wage.kappa
                @test isapprox(exp.core_initial.monetary.i_zlb,exp.initial.i;atol=1e-10)
            end
        end
    end
end

# A short-horizon smoke test of the NONLINEAR DYNAMIC ENGINE, not the
# original permanent-deficit Figure 10 experiment. With T=20 and the source
# rho=0.9, shock[end] is about 0.000775; the implied US BU terminal deficit
# exceeds the maximum feasible deficit on the source Figure 8 locus. The
# stationary Newton solver rightly cannot find such a terminal equilibrium.
# Keep the actual T=1000 source experiments above and below unchanged.
# Instead, apply a small, quickly vanishing TRANSITORY shock around a single
# independently calibrated steady state: this isolates dynamic convergence.
if get(ENV,"GOLDILOCKS_FIG10_SMOKE","0")=="1"
    @testset "Short-horizon BU nonlinear transition (transitory-only diagnostic)" begin
        baseline=spending_experiment(:A,:bu;mode=:official,T=1000)
        Tsmoke=20
        impulse=transition_shocks(Tsmoke;permanent=0.0,
                                   transitory=0.0001,rho=0.25)
        deficit_path=baseline.initial.deficit .+ impulse
        problem=TransitionProblem(baseline.core_initial,baseline.household,
                                  baseline.initial,baseline.initial,
                                  deficit_path,zeros(Tsmoke),:bu)
        start=GoldilocksTransition._starting_vector(problem)
        @test norm(transition_residual(problem,start),Inf)>1e-5
        @test abs(deficit_path[end]-baseline.initial.deficit)<1e-12
        solution=solve_transition(problem;tol=1e-5,maxit=50,verbose=true)
        accuracy=check_transition(solution;tol=1e-5,goods_tol=1e-5)
        @test solution.iterations>0
        @test accuracy.max_target_residual<=1e-5
        @test accuracy.max_goods_residual<=1e-5
        @test length(solution.periods)==Tsmoke
        @test all(z -> isfinite(z.B) && isfinite(z.Y),solution.periods)
    end
end


# HA-specific diagnostic of the fully time-coupled EGM/distribution Newton
# system and its sparse frozen-household RIGHT preconditioner. The original
# Figure 10 shock and all T=1000 tests remain unchanged; this is intentionally
# a small, quickly vanishing transitory impulse around a single steady state.
if get(ENV,"GOLDILOCKS_FIG10_HA_SMOKE","0")=="1"
    @testset "Short-horizon HA matrix-free Newton diagnostic" begin
        baseline=spending_experiment(:A,:ha;mode=:official,T=1000)
        Tsmoke=20
        impulse=transition_shocks(Tsmoke;permanent=0.0,
                                  transitory=0.0001,rho=0.25)
        deficit_path=baseline.initial.deficit .+ impulse
        problem=TransitionProblem(baseline.core_initial,baseline.household,
                                  baseline.initial,baseline.initial,
                                  deficit_path,zeros(Tsmoke),:ha)
        start=GoldilocksTransition._starting_vector(problem)
        @test norm(transition_residual(problem,start),Inf)>1e-5
        solution=solve_transition(problem;tol=1e-5,maxit=50,verbose=true)
        accuracy=check_transition(solution;tol=1e-5,goods_tol=1e-5)
        @test solution.iterations>0
        @test accuracy.max_target_residual<=1e-5
        @test accuracy.max_goods_residual<=1e-5
        @test accuracy.max_ha_mass_error<=1e-7
        @test length(solution.periods)==Tsmoke
        @test all(z -> isfinite(z.B) && isfinite(z.Y),solution.periods)
    end
end

# The nonlinear 1000-period paths are COMPUTATIONAL full-system tests and
# deliberately not claimed by the steady-state tests above. Enable explicitly
# after smoke-testing the native sparse/matrix-free solver locally.
if get(ENV,"GOLDILOCKS_FIG10_FULL_TESTS","0")=="1"
    @testset "Figure 10 full T=1000 nonlinear GE paths" begin
        solved=Dict{Tuple{Symbol,Symbol},Any}()
        ha_only=get(ENV,"GOLDILOCKS_FIG10_HA_ONLY","0")=="1"
        for panel in FIGURE10_PANELS
            for model in FIGURE10_MODELS[panel]
                if ha_only && (panel!==:A || model!==:ha)
                    continue
                end
                exp=spending_experiment(panel,model;mode=:official,T=1000)
                out=solve_spending_experiment(exp;tol=1e-5,maxit=50,verbose=true)
                solved[(panel,model)]=out
                a=out.validation
                @test a.max_target_residual<=1e-5
                @test a.max_goods_residual<=1e-5
                @test length(out.solution.periods)==1000
                @test a.max_ha_mass_error<=1e-7
                @test all(z -> isfinite(z.B) && isfinite(z.Y), out.solution.periods)
                if panel===:A
                    @test exp.terminal.B>exp.initial.B
                elseif panel===:B
                    @test exp.terminal.B<exp.initial.B
                elseif panel===:D
                    @test exp.terminal.deficit<exp.initial.deficit
                end
            end
        end
        parity=compare_figure10_python(solved;
            source=joinpath(@__DIR__,"figure10_original_python_reference.tsv"),
            directory=@__DIR__,mode=:official)
        @test length(parity.selected)==(ha_only ? 1 : 14)
        # Python intentionally leaves piw blank in non-ZLB panels A and C.
        # Record that absence, but require 1,000 observed values for each
        # economically comparable series instead of filling piw with zero.
        expected_missing=Dict{Tuple{Symbol,Symbol,Symbol},Int}(
            (panel,model,:piw)=>1000 for (panel,model) in keys(solved)
            if panel in (:A,:C))
        @test parity.missing_count==expected_missing
        for (key,n) in parity.selected
            @test n==1000
            for field in (:deficit,:B,:B_annual,:Y,:C,:A,:r,:r_ante,
                          :pi,:piw,:i,:goods_mkt,:asset_mkt,:deficit_diff)
                if field===:piw && key[1] in (:A,:C)
                    @test !haskey(parity.diagnostics,(key[1],key[2],field))
                else
                    @test parity.diagnostics[(key[1],key[2],field)].matched==1000
                end
            end
            @test parity.diagnostics[(key[1],key[2],:B)].max_error<1e-3
            @test parity.diagnostics[(key[1],key[2],:deficit)].max_error<1e-4
        end
    end
end
println("GoldilocksSpendingShocks test suite completed. Full nonlinear tests gated by GOLDILOCKS_FIG10_FULL_TESTS=1")
