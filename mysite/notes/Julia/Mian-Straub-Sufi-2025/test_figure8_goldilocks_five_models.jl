# Run locally in project root:
# julia --project=. .\test_figure8_goldilocks_five_models.jl
# No GitHub Actions and no new directory.

using Test
include(joinpath(@__DIR__, "figure8_goldilocks_five_models.jl"))

# Independent reference: direct execution of the original Python
# auxfs.get_deficit_debt on official us_ss.pkl/jp_ss.pkl for OLG-alt.
const FIVE_OLG_ALT_NONZLB = Dict(
    :us => (
        B=(0.011053294849377717, 0.37793581320175723,
           0.7077498217545176, 1.0099201790068206,
           1.2914925273657203, 1.5578148552271902,
           1.8129569470062314, 2.0599494175179145),
        deficit=(0.00038253069429711184, 0.011205638064872014,
                 0.017475274870929924, 0.019928808641594853,
                 0.01908153076986918, 0.015292330980794522,
                 0.008807822210667005, -0.0002059949407208217),
    ),
    :jp => (
        B=(0.2913595732553045, 0.8006344305520376,
           1.302217971460644, 1.8014535793811943,
           2.303402173564874, 2.8128471717243824,
           3.334198846582036, 3.8712235697631603),
        deficit=(0.006587011528693509, 0.015503377464883006,
                 0.02099160779380088, 0.023195354813281777,
                 0.022186214886340594, 0.017968341666995624,
                 0.010482646457713074, -0.00038712235503862713),
    ),
)

@testset "Five-model Figure 8: US/Japan, including OLG-alt" begin
    @testset "Source Figure 8 remains four models; extension has five" begin
        @test FIGURE8_MODELS == (:bu,:bu_log,:olg,:ha)
        @test FIGURE8_FIVE_MODELS == (:bu,:bu_log,:olg,:olg_alt,:ha)
        @test length(unique(FIGURE8_FIVE_MODELS)) == 5
        @test_throws ArgumentError figure8_calibration(:us,:olg_alt;mode=:official)
        @test_throws ArgumentError five_model_calibration(:us,:not_a_model)
        @test_throws ArgumentError five_model_calibration(:us,:olg_alt;mode=:invalid)
        @test model_style(:olg_alt).color == OKABE_ITO.orange
        @test model_style(:olg_alt).linestyle == :dashdot
    end

    @testset "Official OLG-alt calibration, stationary GE, and original Python parity" begin
        for country in (:us,:jp)
            p = FIVE_MODEL_OLG_ALT_OFFICIAL[country]
            input = five_model_calibration(country,:olg_alt;mode=:official)
            h = input.household
            @test h isa OLGHousehold
            @test h.beta ≈ p.beta atol=1e-15
            @test h.survival ≈ p.survival atol=1e-14
            @test h.zeta ≈ p.zeta atol=1e-14
            @test h.survival > 1 # source reduced-form continuation parameter
            @test abs(input.anchor.asset_mkt) < 2e-5
            @test abs(input.anchor.goods_mkt) < 2e-5
            @test abs(input.anchor.piw_cond) < 2e-5
            @test input.setup.periods == 1

            curve = deficit_debt_curve(input.setup,input.household,input.core;
                                       model=:olg_alt, mode=:official,
                                       n_r=8, n_Y=12)
            ref = FIVE_OLG_ALT_NONZLB[country]
            @test length(curve.records) == 20
            @test all(i -> !ismissing(curve.records[i]),13:20)
            for k in 1:8
                pt = curve.records[12+k]
                if !ismissing(pt)
                    @test pt.branch === :flex
                    @test isapprox(pt.B,ref.B[k];atol=3e-3)
                    @test isapprox(pt.deficit,ref.deficit[k];atol=2e-4)
                    @test pt.residual < 2e-5
                    @test abs(pt.deficit+pt.r_ante*pt.B) < 2e-5
                end
            end
            @test grid_maximum(curve).maximum_deficit > 0

            # Comparison in the same source r-grid: source OLG and OLG-alt
            # are observationally almost equivalent here. No fabricated
            # vertical offsets are used to separate their plotted curves.
            reference_olg = five_model_calibration(country,:olg;mode=:official)
            olg_curve = deficit_debt_curve(reference_olg.setup,
                         reference_olg.household,reference_olg.core;
                         model=:olg,mode=:official,n_r=8,n_Y=12)
            @test length(olg_curve.r_grid) == length(curve.r_grid)
            for k in 1:8
                a=curve.records[12+k]; b=olg_curve.records[12+k]
                if !ismissing(a) && !ismissing(b)
                    @test abs(a.B-b.B) < 1e-3
                    @test abs(a.deficit-b.deficit) < 1e-5
                end
            end
            # A new per-model TSV filename must not overwrite the original
            # four-model Figure 8 file.
            path = _write_five_model_data(curve;directory=@__DIR__)
            @test basename(path) == "figure8_five_models_$(country)_olg_alt_official.tsv"
            @test isfile(path)
            @test length(readlines(path)) == 21
            rm(path;force=true)
        end
    end

    @testset "Native HA extension delegates to validated IFT calibration" begin
        input = five_model_calibration(:us,:ha;mode=:native)
        @test input.household isa HAHousehold
        @test input.source !== nothing
        @test abs(input.source.phi_finite_error) < 5e-5
        @test abs(input.anchor.asset_mkt) < 5e-5
    end

    @testset "Native OLG-alt calibration and saver interface" begin
        input = five_model_calibration(:us,:olg_alt;mode=:native)
        @test input.household isa OLGHousehold
        @test input.source !== nothing
        @test abs(input.source.phi_finite_error) < 5e-5
        @test abs(input.anchor.asset_mkt) < 5e-5
    end
end

println("Five-model Figure 8 extension tests completed.")
