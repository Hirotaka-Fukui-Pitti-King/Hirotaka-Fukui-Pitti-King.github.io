# GoldilocksCore audit, Research Task III.
# Run from PowerShell (not from the PowerShell prompt as Julia code):
#   julia --project=. test/test_goldilocks_core.jl
#
# The Python reference numbers below were obtained by EXECUTING the original
# function bodies in code/blocks.py (decorators removed, time-shift inputs
# emulated as callable scalars). The full Sequence-Jacobian equilibrium solver
# and OLG/HA household blocks are outside the scope of this test.

using Test
using LinearAlgebra
using ForwardDiff

include(joinpath(@__DIR__, "GoldilocksCore.jl"))
using .GoldilocksCore

# Reference values are from original blocks.py, not retyped equations.
const REF_TRANS = (
    N = 0.9124641604719883,
    w = 1.0162292037783476,
    I = 0.42410000000000003,
    invcond = -0.3682710754707408,
    valcond = 0.09454313468888631,
    adj_cost = 0.047368421052631705,
    dividend = -0.035104784688995166,
    mpk = 0.25263157894736843,
    K = 2.14,
    mpn = 1.1178521241561825,
    pure_profit = -0.16363636363636358,
)
const REF_SS = (
    N = 0.7045096574051698,
    w = 1.316195906650921,
    I = 0.3127346796012634,
    invcond = 0.0,
    valcond = 0.0,
    adj_cost = 0.0,
    dividend = 0.12362895676237295,
    mpk = 0.14581049999999987,
    K = 3.2919439958027743,
    Q = 1.052555,
    k = 3.2919439958027743,
    mpn = 1.4478154973160133,
    pure_profit = -0.16363636363636358,
    dyn_ineff_cond = 0.05081049999999987,
)

function compare_reference(actual::NamedTuple, expected::NamedTuple; atol=5e-12)
    for name in propertynames(expected)
        @test hasproperty(actual, name)
        @test isapprox(getproperty(actual, name), getproperty(expected, name);
                       atol=atol, rtol=1e-11)
    end
end

@testset "GoldilocksCore: original Python parity and economic audit" begin
    pp = ProductionParams(
        alpha=0.32, tfp=1.3, fixed_cost=0.3, markup=1.1,
        delta_k=0.08, eps_I=4.0, real_growth=0.015,
        risk_premium=0.012,
    )

    @testset "A. Production/investment: shifted and steady" begin
        tr = production_block(1.2, 1.90, 2.14, 1.06, 1.08, 0.025, pp)
        compare_reference(tr, REF_TRANS)
        @test tr.k != 1.90
        @test tr.Q != 1.08
        @test isapprox(tr.kratio, 2.14/1.90; atol=1e-12)
        @test isapprox(pp.tfp * 1.90^pp.alpha * tr.N^(1-pp.alpha),
                       1.2+pp.fixed_cost; atol=1e-12)
        @test isapprox(tr.w, tr.mpn / pp.markup; atol=1e-12)
        @test isapprox(tr.I, (1+pp.real_growth)*2.14 - 1.90 + pp.delta_k*1.90;
                       atol=1e-12)
        @test isapprox(tr.dividend + tr.w*tr.N + pp.fixed_cost +
                       tr.I + tr.adj_cost + tr.pure_profit,
                       1.2+pp.fixed_cost; atol=1e-12)
        @test abs(tr.invcond) > 0.01  # FOCs need not hold off equilibrium
        @test abs(tr.valcond) > 0.01

        ss = production_steady_state(1.2, 0.025, pp)
        compare_reference(ss, REF_SS)
        @test isapprox(ss.k, ss.K; atol=1e-14)
        @test abs(ss.invcond) < 1e-12
        @test abs(ss.valcond) < 1e-12
        @test_throws DomainError production_block(-5.0, 1.9, 2.14,
                                                   1.06, 1.08, 0.025, pp)
    end

    @testset "B. Equity: lagged holdings and forward value" begin
        eq = equity_block(2.30, 1.97, 2.02, 2.33, 0.14, 0.03, 0.012)
        compare_reference(eq, (
            rpost_equity=0.1675126903553299,
            jvaluation=0.21608445297504808,
            Jvaluation=0.0760844529750484,
        ))
        ss_prod = production_steady_state(1.2, 0.025, pp)
        eqss = equity_steady_state(ss_prod.k, ss_prod.dividend, 0.025, pp)
        @test isapprox(eqss.j, (1+pp.real_growth)*ss_prod.k; atol=1e-12)
        @test abs(eqss.jvaluation) < 1e-12
        @test abs(eqss.Jvaluation) < 1e-12
        @test isapprox(eqss.rpost_equity, 0.025+pp.risk_premium;
                       atol=1e-12)
    end

    @testset "C. Nominal long bonds: shifted inflation" begin
        q = long_bond_block(21.4, 21.1, 20.8, 0.025, 0.91,
                            0.02, 0.012, 0.01, 0.013)
        compare_reference(q, (
            qres=1.8950304577083266,
            rpost_bonds=-0.07224565777083469,
        ))
        qs = long_bond_steady_state(0.025, 0.01, 0.91)
        @test qs.q > 0
        @test abs(qs.qres) < 1e-12
        @test isapprox(qs.rpost_bonds, 0.025; atol=1e-12)
        @test_throws DomainError long_bond_steady_state(-0.50, 0.0, 0.95)
    end

    @testset "D. Beginning-of-period portfolio weights" begin
        @test isapprox(combined_return(3.0, 2.1, 0.02, 0.08),
                       0.04470588235294137; atol=1e-13)
        @test isapprox(combined_return(1.0, 0.0, 0.03, -0.1),
                       0.03; atol=1e-13)
    end

    @testset "E. Fiscal budget and tax base" begin
        flevel = FiscalParams(G=0.23, periods=4,
                              adjust_tax_rate=false, psi=0.1,
                              deficit_ss=-0.03, B_ss=3.0)
        frate = FiscalParams(G=0.23, periods=4,
                             adjust_tax_rate=true, psi=0.1,
                             deficit_ss=-0.03, B_ss=3.0)
        ss_level = fiscal_block(1.1, 0.95, 3.0, 3.1, 0.025, flevel)
        compare_reference(ss_level, (
            tax=0.20499999999999963,
            Z=0.8400000000000003,
            deficit=0.025000000000000383,
            tau=0.20499999999999963,
            deficit_diff=-0.05500000000000038,
            B_annual=0.775,
        ))
        ss_rate = fiscal_block(1.1, 0.95, 3.0, 3.1, 0.025, frate)
        @test isapprox(ss_rate.tau, 0.19617224880382741; atol=1e-12)
        @test ss_rate.tax == ss_level.tax
        @test isapprox(ss_rate.tax + 3.1,
                       0.23 + (1.025)*3.0; atol=1e-12)
        # Tax changes when the debt level changes or G changes.
        @test fiscal_block(1.1,0.95,3.0,3.2,0.025,flevel).tax < ss_level.tax
        fmore = FiscalParams(G=0.33, periods=4)
        @test isapprox(fiscal_block(1.1,0.95,3.0,3.1,0.025,fmore).tax -
                       ss_level.tax, 0.10; atol=1e-12)
        ref = (; tax=0.7, w=1.03, N=0.96, Y=1.0)
        ftax = FiscalParams(G=0.2, tax_shock=0.04, tau_slope=0.2)
        tr = tax_rule_residuals(0.8, 1.05, 1.02, 1.15, ref, ftax)
        compare_reference(tr, (
            tax_diff=0.06000000000000009,
            tax_rate_diff=0.009036650258808995,
        ))
    end

    @testset "F. Monetary rule and binding regime" begin
        poff = MonetaryParams(pi_trend=0.005, real_growth=0.01,
                              phi_pi=1.5, real_rate=0.8,
                              i_shock=0.005, zlb=false)
        pon = MonetaryParams(pi_trend=0.005, real_growth=0.01,
                             phi_pi=1.5, real_rate=0.8,
                             i_shock=0.005, zlb=true, i_zlb=0.0)
        off = monetary_block(0.012, -0.005, 0.02, poff)
        on = monetary_block(0.012, -0.005, 0.02, pon)
        compare_reference(off, (
            i=0.039767697360777114,
            fisher=0.009767697360777008,
        ))
        compare_reference(on, (i=0.0, fisher=-0.03))
        @test isapprox(off.i, on.i_rule; atol=1e-13)
        @test on.i == pon.i_zlb
        @test on.i_rule > on.i # explicitly selected ZLB can be inconsistent
        # A binding candidate must be checked against the shadow rule.
        pbind = MonetaryParams(pi_trend=0.0, real_growth=0.0,
                               zlb=true, i_zlb=0.0)
        bind = monetary_block(-0.02, -0.02, -0.03, pbind)
        @test bind.i == 0.0
        @test bind.i_rule < bind.i
        @test abs(bind.fisher) > 0  # off-equilibrium: Fisher not imposed here
    end

    @testset "G. Wage Phillips curve and expected inflation" begin
        wp = WageParams(kappa=0.2, beta=0.98, labor_shifter=0.95,
                         lag=0.3, forward=0.7, frisch=1.2, survival=0.9)
        w = wage_phillips_block(0.02, 0.01, 0.015, 1.1, 0.7, 0.8, wp)
        compare_reference(w, (
            piw_cond=0.01899397519234792,
            labor_foc=0.25638442223037716,
            kappa_HHNS=0.3483333333333334,
        ))
        pinf = inflation_block(0.01, 0.015, 1.02, 1.08, 1.10)
        compare_reference(pinf, (
            pi=-0.046111111111111214,
            pi_e=-0.003454545454545488,
        ))
        shifter = labor_shifter_for_phillips(0.01,1.1,0.7,0.8,wp)
        wpc = WageParams(kappa=wp.kappa, beta=wp.beta,
                          labor_shifter=shifter, lag=wp.lag,
                          forward=wp.forward, frisch=wp.frisch,
                          survival=wp.survival)
        @test abs(wage_phillips_block(0.01,0.01,0.01,1.1,0.7,0.8,wpc).piw_cond) < 1e-12
        wpflat = WageParams(kappa=0.0, beta=0.98, labor_shifter=1.0)
        @test_throws DomainError labor_shifter_for_phillips(0.0,1.0,1.0,1.0,wpflat)
    end

    @testset "H. Clearing residuals and target order" begin
        m = market_clearing(4.1, 3.2, 0.7, 0.25, 1.15, 0.4, 0.03, 0.16)
        compare_reference(m, (asset_mkt=0.49999999999999944,
                              goods_mkt=-0.010000000000000009))
        @test equilibrium_residuals(m, (:goods_mkt,:asset_mkt)) ==
              [m.goods_mkt, m.asset_mkt]
        @test check_equilibrium((;asset_mkt=0.0); keys=(:asset_mkt,))
        @test_throws ErrorException check_equilibrium((;asset_mkt=0.02);
                                                      keys=(:asset_mkt,))
    end

    @testset "I. ForwardDiff, analytical and finite differences" begin
        f(x) = [x[1]^2 + 3*x[2], x[1]*x[2]]
        u = [1.2, 0.7]
        J_ad = equilibrium_jacobian(f,u)
        J_an = [2.4 3.0; 0.7 1.2]
        @test isapprox(J_ad, J_an; atol=1e-12)
        h = 1e-5
        J_fd = hcat(((f(u .+ h.*[1.0,0.0]) - f(u .- h.*[1.0,0.0]))/(2h)),
                     ((f(u .+ h.*[0.0,1.0]) - f(u .- h.*[0.0,1.0]))/(2h)))
        @test isapprox(J_ad, J_fd; atol=1e-9)
        # Economic FOCs, not merely an algebra example.
        residual(x) = let v = production_block(1.2,1.90,x[1],x[2],1.08,0.025,pp)
            [v.invcond, v.valcond]
        end
        Je = equilibrium_jacobian(residual, [2.14,1.06])
        @test size(Je) == (2,2)
        @test all(isfinite,Je)
        @test isapprox(Je[1,1], -1/(1.90*pp.delta_k*pp.eps_I); atol=1e-12)
        @test isapprox(Je[1,2], 0.0; atol=1e-12)
        @test isapprox(Je[2,2], -1.0; atol=1e-12)
        @test isapprox(Je, hcat(
            (residual([2.14+h,1.06])-residual([2.14-h,1.06]))/(2h),
            (residual([2.14,1.06+h])-residual([2.14,1.06-h]))/(2h)
        ); atol=1e-9)
    end

    @testset "J. NonlinearSolve v4: solutions, errors, and scaling" begin
        f1(u) = [u[1]^2-4]
        s1 = solve_nonlinear(f1, [2.4])
        @test isapprox(s1.u[1], 2.0; atol=1e-9)
        @test s1.residual_norm <= 1e-10
        f2(u) = [u[1]+u[2]-3, u[1]^2+u[2]-5]
        s2 = solve_nonlinear(f2, [1.8,1.1])
        @test isapprox(s2.u, [2.0,1.0]; atol=1e-8)
        @test s2.residual_norm <= 1e-10
        @test_throws DimensionMismatch solve_nonlinear(u->[u[1]],[1.0,2.0])
        @test_throws ArgumentError solve_nonlinear(u->[NaN],[1.0])
        @test_throws ArgumentError solve_nonlinear(u->[u[1]],[NaN])
        @test_throws ArgumentError solve_nonlinear(f1,[2.4]; abstol=-1e-4)
        @test_throws ArgumentError solve_nonlinear(f1,[2.4]; maxiters=0)
        @test_throws Exception solve_nonlinear(u->[1.0], [3.0]; maxiters=10)
        # Old implementation incorrectly accepted an O(1e-7) residual
        # at O(1e9) initial unknowns; final residual must be absolute.
        @test_throws Exception solve_nonlinear(u->[1e-7], [1e9]; maxiters=10)
    end

    @testset "K. General-equilibrium adapter (artificial household)" begin
        g = 0.015
        ra = 1.015/1.02 - 1 - g
        alpha = ((1+ra)*(1+g)-0.92)*2
        tfp = 1.3/2^alpha
        pr = ProductionParams(alpha=alpha, tfp=tfp, fixed_cost=0.3,
                              markup=1.3, delta_k=0.08, eps_I=4.0,
                              real_growth=g)
        target_Y, target_B = 1.0, 2.1
        prod = production_steady_state(target_Y,ra,pr)
        fr = FiscalParams(G=0.18, periods=4, B_ss=target_B,
                          deficit_ss=-ra*target_B, psi=0.1)
        fisc = fiscal_block(prod.N,prod.w,target_B,target_B,ra,fr)
        Cstar = target_Y - fr.G - prod.I - prod.adj_cost
        w0 = WageParams(kappa=0.2,beta=0.98,labor_shifter=1.0,
                         lag=0.25,forward=0.6,frisch=1.0,survival=0.9)
        hstar = labor_shifter_for_phillips(0.0,prod.N,fisc.Z,Cstar,w0)
        wag = WageParams(kappa=w0.kappa,beta=w0.beta,
                          labor_shifter=hstar, lag=w0.lag,
                          forward=w0.forward,frisch=w0.frisch,
                          survival=w0.survival)
        mon = MonetaryParams(real_growth=g,pi_trend=0.005,real_rate=1.0)
        cp = CoreParams(production=pr,wage=wag,monetary=mon,
                         fiscal=fr,mu=0.2,delta_b=0.0)
        # Explicit artificial demand schedules. No claim of household optimality.
        function test_household(h)
            pp_here = production_steady_state(h.Y,h.r_ante,pr)
            j_here = (1+pr.real_growth)*pp_here.k
            dy,db = h.Y-target_Y,h.B-target_B
            (; C=h.Y-fr.G-pp_here.I-pp_here.adj_cost+dy+db,
               A=h.B+j_here+2dy+3db)
        end
        initial = (;Y=1.08,B=2.25,r_ante=ra,piw=0.0)
        unsolved = evaluate_steady_state(initial,cp,test_household)
        @test abs(unsolved.asset_mkt) > 0.01
        @test abs(unsolved.goods_mkt) > 0.01
        sol = solve_core_steady_state(initial,cp,test_household,
                                      (:Y,:B),(:asset_mkt,:goods_mkt))
        eq = sol.equilibrium
        @test isapprox(eq.Y,target_Y; atol=1e-9)
        @test isapprox(eq.B,target_B; atol=1e-9)
        @test isapprox(eq.C,Cstar; atol=1e-9)
        @test sol.solution.residual_norm < 1e-10
        @test check_equilibrium(eq)
        @test abs(eq.labor_foc) < 1e-10
        @test abs(eq.deficit_diff) < 1e-10
        @test isapprox(eq.A,eq.B+eq.j; atol=1e-9)
        # Make sure selection is explicit; an unselected residual can fail.
        wrong_w = WageParams(kappa=0.2,beta=0.98,labor_shifter=hstar*1.1,
                             lag=0.25,forward=0.6,frisch=1.0,survival=0.9)
        cp_wrong = CoreParams(production=pr,wage=wrong_w,monetary=mon,
                              fiscal=fr,mu=0.2)
        bad_eq = evaluate_steady_state((;Y=target_Y,B=target_B,
                                        r_ante=ra,piw=0.0),cp_wrong,test_household)
        @test abs(bad_eq.labor_foc) > 1e-4
        @test_throws ErrorException check_equilibrium(bad_eq)
        @test_throws ArgumentError evaluate_steady_state(initial,cp,_ -> (;C=0.5))
        @test_throws ArgumentError evaluate_steady_state(initial,cp,_ -> 1.0)
        # Dual numbers through the complete core evaluator.
        ge_res(u) = equilibrium_residuals(
            evaluate_steady_state(merge(initial,(;Y=u[1], B=u[2])),
                                  cp,test_household),
            (:asset_mkt,:goods_mkt)
        )
        Jge = equilibrium_jacobian(ge_res,[1.08,2.25])
        @test isapprox(Jge,[2.0 3.0; 1.0 1.0]; atol=1e-9)
    end
end

println("GoldilocksCore audit tests completed (see Test Summary above).")

