#=
Krusell_data_audit.jl — Baqaee and Farhi (2019), JEEA 17(5)
Task VIII: inspect source Excel -> released .mat -> exact MATLAB-equivalent
Julia results -> official results workbook -> published Appendix Tables B.1–B.5.

Required same-directory files (unchanged): Krusell_sens.jl, Krusell_disagg.jl,
Krusell_agg.jl, Relabel_mod.jl, WeightedCovMatrix.jl.
Dependencies: XLSX.jl, MAT.jl, SummaryTables.jl; rest Julia stdlib.

Validated source-cell mappings from the official archive:
* IO2013.mat: 2013!C8:CV90. '...' -> 0; original negatives retained.
* output_new.mat: Gross Output!Q3:Q65
* VA_new.mat: Value Added!Q3:Q65
* labor_no_col.mat: Labor_NoCol Compensation!Q3:Q65
* labor_col.mat: Labor_Col Compensation!Q3:Q65
2013 is column Q for the 1998–2015 production-account sheets.
There are 8134 + 4*63 = 8386 original input numbers, all verified in
independent Python comparisons against the supplied MATLAB v5 files.

Windows PowerShell, from the user's website repository root:
  julia --project=. -e 'using Pkg; Pkg.add(string(:XLSX))'
  julia --project=. -e 'Meta.parseall(read(ARGS[1], String))' .\notes\Julia\Baqaee-Farhi-2019-APF\Krusell_data_audit.jl
  julia --depwarn=error --project=. .\notes\Julia\Baqaee-Farhi-2019-APF\Krusell_data_audit.jl --test-only
  julia --depwarn=error --project=. .\notes\Julia\Baqaee-Farhi-2019-APF\Krusell_data_audit.jl --data-dir ".\notes\Julia\Baqaee-Farhi-2019-APF\data" --source-dir ".\notes\Julia\Baqaee-Farhi-2019-APF\data\source" --bench

The five source Excel and MAT files are required for the complete audit.
Run --test-only without those files. No MATLAB executable is used.
If source files are missing, do not claim source-data reconstruction passed.
The file has not been parsed or executed in Julia in the authoring container.
=#

# In the Task VII dependency chain, Krusell_sens.jl includes the disaggregated
# model, which includes aggregate, Relabel_mod and WeightedCovMatrix ONCE.
# Include BEFORE defining this module: avoids Julia 1.12 world-age problems.
include(joinpath(@__DIR__, "Krusell_sens.jl"))

module KrusellDataAudit

using ..KrusellAggregate: load_krusell_data, compute_capital_counterfactual
using ..KrusellSensitivities: compute_all_sensitivities
import ..KrusellSensitivities
using XLSX
using MAT: matread
using SummaryTables: Table, Cell
using Test
using Printf
using LinearAlgebra
using Statistics
using SHA

export load_official_source_workbooks, load_official_mat_data,
       inspect_source_data_layout, reconstruct_io2013,
       reconstruct_production_accounts, compare_source_and_mat,
       read_official_results_workbook, compare_matlab_julia_paper,
       classify_discrepancies, run_reproducibility_audit,
       save_audit_results, run_tests, run_benchmark

const IO_WORKBOOK = "IOUse_After_Redefinitions_PRO_1997-2016_Summary.xlsx"
const PRODUCTION_WORKBOOK = "BEA-BLS-industry-level-production-account-1998-2015.xlsx"
const RESULT_WORKBOOK = "results tables1-5.xlsx"
const PRODUCTION_SHEETS = (
    (matfile="output_new.mat", variable="output_new", sheet="Gross Output"),
    (matfile="VA_new.mat", variable="VA_new", sheet="Value Added"),
    (matfile="labor_no_col.mat", variable="labor_no_col", sheet="Labor_NoCol Compensation"),
    (matfile="labor_col.mat", variable="labor_col", sheet="Labor_Col Compensation")
)
const FACTOR_NAMES = ("Capital", "Noncollege", "College")
const PAIR_NAMES = (:HK, :LK)
const PARAMETERS = (:theta0, :theta3, :theta4, :theta2, :theta1)
const AGG_PARAMETERS = (:theta2, :theta1)
const ACTIVE_MODELS = (:aggregate, :disaggregated)
const TYPES = (:production, :cost)

# Published B.1/B.2 are in PAPER orientation, not original MATLAB orientation.
const B1_PROD = [NaN 1.67 0.67; 1.05 NaN 0.88; 0.67 1.67 NaN]
const B1_COST = [NaN 1.67 0.67; 1.28 NaN 1.07; 0.67 1.67 NaN]
const B2_PROD = [NaN 1.43 0.70; 0.94 NaN 0.94; 0.67 1.59 NaN]
const B2_COST = [NaN 1.47 0.72; 1.09 NaN 1.09; 0.64 1.54 NaN]
const B5_AGG = [0.05, 0.07, -0.13]
const B5_DIS = [0.06, 0.04, -0.13]

"""
    load_official_source_workbooks(source_dir)

Require all three ORIGINAL workbooks and list their actual worksheet names.
Do not fall back silently to downloaded, revised, or synthetic data.
"""
function load_official_source_workbooks(source_dir::AbstractString)
    isdir(source_dir) || throw(ArgumentError("Source directory does not exist: $source_dir"))
    io_path = joinpath(source_dir, IO_WORKBOOK)
    production_path = joinpath(source_dir, PRODUCTION_WORKBOOK)
    result_path = joinpath(source_dir, RESULT_WORKBOOK)
    for path in (io_path, production_path, result_path)
        isfile(path) || throw(ArgumentError("Required official XLSX file missing: $path"))
    end
    io_names = XLSX.sheetnames(XLSX.readxlsx(io_path))
    production_names = XLSX.sheetnames(XLSX.readxlsx(production_path))
    result_names = XLSX.sheetnames(XLSX.readxlsx(result_path))
    "2013" in io_names || throw(ArgumentError("2013 sheet is absent from IO workbook"))
    for name in ("Gross Output", "Value Added", "Labor_NoCol Compensation",
                 "Labor_Col Compensation", "NAICS codes")
        name in production_names || throw(ArgumentError("Missing production sheet: $name"))
    end
    for name in ("Sheet1", "Sheet2", "Sheet3", "Sheet4", "Sheet5")
        name in result_names || throw(ArgumentError("Missing results worksheet $name"))
    end
    return (io_path=io_path, production_path=production_path,
            result_path=result_path, io_sheets=io_names,
            production_sheets=production_names, result_sheets=result_names)
end

"""Read all five original MAT files through the previously verified Task V loader."""
load_official_mat_data(data_dir::AbstractString) = load_krusell_data(data_dir)

# XLSX.readdata(path, sheet, cellrange) is an XLSX.jl documented stable API.
# A single-column range may be represented as a vector on some versions.
function _matrix_range(path::AbstractString, sheet::AbstractString,
                       range::AbstractString, nrow::Int, ncol::Int)
    data = XLSX.readdata(path, sheet, range)
    if nrow == 1 && ncol == 1 && !(data isa AbstractArray)
        return reshape([data], 1, 1)
    elseif data isa AbstractMatrix
        size(data) == (nrow, ncol) || throw(DimensionMismatch(
            "$sheet!$range: expected $(nrow)x$(ncol), got $(size(data))"))
        return data
    elseif data isa AbstractVector && ncol == 1 && length(data) == nrow
        return reshape(collect(data), nrow, 1)
    end
    throw(DimensionMismatch("$sheet!$range: incompatible XLSX data representation"))
end

function _required_numeric(value, ref::AbstractString; allow_dots::Bool=false)
    if allow_dots && value isa AbstractString && strip(value) == "..."
        return 0.0
    end
    value isa Real && !(value isa Bool) || throw(ArgumentError(
        "Expected numeric value at $ref; found $(repr(value))"))
    converted = Float64(value)
    isfinite(converted) || throw(ArgumentError("Non-finite value at $ref"))
    return converted
end

"""
    inspect_source_data_layout(sources)

Check sheet headings and the 2013 columns before trusting the cell mapping.
The layout is documented by source metadata, not selected via numerical fit.
"""
function inspect_source_data_layout(sources::NamedTuple)
    sourceyear = _matrix_range(sources.io_path, "2013", "A4:A4", 1, 1)[1,1]
    year = _matrix_range(sources.production_path, "Gross Output", "Q2:Q2", 1, 1)[1,1]
    sourceyear == "2013" || sourceyear == 2013 || throw(ArgumentError(
        "IO data year is not 2013: $(repr(sourceyear))"))
    year == 2013 || year == "2013" || throw(ArgumentError(
        "Production account year is not 2013: $(repr(year))"))
    for spec in PRODUCTION_SHEETS
        head = _matrix_range(sources.production_path, spec.sheet, "Q2:Q2", 1, 1)[1,1]
        head == 2013 || head == "2013" || throw(ArgumentError(
            "Unexpected year at $(spec.sheet)!Q2"))
    end
    return (io_year=2013, production_year=2013,
            io_range="2013!C8:CV90", production_range="Q3:Q65",
            io_unit="millions of dollars", production_unit="millions of dollars")
end

"""
    reconstruct_io2013(sources)

Take 83 source rows × 98 columns literally from the 2013 BEA Use table.
The string "..." denotes missing/suppressed/non-applicable entries and is
ZERO in the supplied MAT file. Other missing/non-numeric values are errors.
Negative source values are preserved. No sorting, scaling, or aggregation.
"""
function reconstruct_io2013(sources::NamedTuple)
    raw = _matrix_range(sources.io_path, "2013", "C8:CV90", 83, 98)
    io = Matrix{Float64}(undef, 83, 98)
    dots = 0
    for i in 1:83, j in 1:98
        cell = raw[i,j]
        cell isa AbstractString && strip(cell) == "..." && (dots += 1)
        io[i,j] = _required_numeric(cell, "2013!row$(i+7):col$(j+2)";
                                    allow_dots=true)
    end
    return (values=io, dots_as_zero=dots,
            negative_count=count(x -> x < 0.0, io),
            zero_count=count(iszero, io), source_range="2013!C8:CV90")
end

"""
    reconstruct_production_accounts(sources)

2013 is Excel column Q. The original production-account industry order is
A3:A65, with no transformation into the four official MAT vectors.
"""
function reconstruct_production_accounts(sources::NamedTuple)
    values = Dict{String,Vector{Float64}}()
    for spec in PRODUCTION_SHEETS
        raw = _matrix_range(sources.production_path, spec.sheet, "Q3:Q65", 63, 1)
        values[spec.variable] = [ _required_numeric(raw[i,1],
                                "$(spec.sheet)!Q$(i+2)") for i in 1:63 ]
    end
    names = _matrix_range(sources.production_path, "Gross Output", "A3:A65", 63, 1)
    descriptions = [string(names[i,1]) for i in 1:63]
    return (variables=values, descriptions=descriptions, source_range="Q3:Q65")
end

function _max_relative(a::AbstractArray{<:Real}, b::AbstractArray{<:Real})
    size(a) == size(b) || throw(DimensionMismatch("Cannot compare different shapes"))
    return maximum(abs(Float64(a[i]) - Float64(b[i])) /
                   max(abs(Float64(b[i])), 1.0) for i in eachindex(a))
end

"""Compare each reconstructed source array with the corresponding official MAT data."""
function compare_source_and_mat(io, production, data::NamedTuple)
    pairs = (("IO2013", io.values, data.io),
             ("output_new", production.variables["output_new"], data.output),
             ("VA_new", production.variables["VA_new"], data.va),
             ("labor_no_col", production.variables["labor_no_col"], data.labor_noncollege),
             ("labor_col", production.variables["labor_col"], data.labor_college))
    rows = NamedTuple[]
    for (name, source, official) in pairs
        size(source) == size(official) || throw(DimensionMismatch(
            "Source and MAT dimension mismatch for $name"))
        diffs = abs.(source .- official)
        push!(rows, (variable=name, n=length(source), matches=count(iszero, diffs),
                     mismatches=count(x -> !iszero(x), diffs),
                     max_abs=maximum(diffs), max_rel=_max_relative(source, official),
                     status=all(iszero, diffs) ? "Exact reconstruction" : "Discrepancy"))
    end
    return rows
end

"""Read every sheet used in the original archived results workbook."""
function read_official_results_workbook(sources::NamedTuple)
    specs = (("Sheet1", "A1:I7", 7, 9), ("Sheet2", "A1:I7", 7, 9),
             ("Sheet3", "A1:F11", 11, 6), ("Sheet4", "A1:G10", 10, 7),
             ("Sheet5", "A1:I58", 58, 9))
    return Dict(name => _matrix_range(sources.result_path, name, rg, nr, nc)
                for (name, rg, nr, nc) in specs)
end

# These values are evidence classifications, NOT automatic proof of the
# ultimate cause of a publication discrepancy.
const CLASSIFICATIONS = (
    A="Exact numerical match", B="Agreement at printed precision",
    C="Intermediate rounding possible (not independently documented)",
    D="Matrix-orientation discrepancy", E="Parameter/row/column placement",
    F="Numerical discrepancy not explained", G="Insufficient evidence"
)

function _printed_match(actual::Real, published::AbstractString)
    value = parse(Float64, published)
    decimals = length(split(published, ".")) == 2 ? length(split(published,".")[2]) : 0
    tol = decimals == 0 ? 1e-7 : 0.5 * 10.0^(-decimals) + 1e-12
    return abs(Float64(actual) - value) < tol
end

"""
    classify_discrepancies(computed, published; row_swap_match=false, orientation=false)

Return the most conservative, source-supported classification. A proposed
row/column switch requires a *separately verified* matching comparison.
"""
function classify_discrepancies(computed::Real, published::AbstractString;
                                row_swap_match::Bool=false,
                                orientation::Bool=false)
    if _printed_match(computed, published)
        return abs(computed - parse(Float64, published)) < 1e-12 ? "A" : "B"
    end
    orientation && return "D"
    row_swap_match && return "E"
    return "F"
end

function _addcomparison!(rows::Vector{NamedTuple}, section::String, object::String,
                         sheet::String, cell::String, computed::Real,
                         original; tolerance::Real=1e-8)
    if original === missing || original === nothing
        push!(rows, (section=section, object=object, sheet=sheet, cell=cell,
                     julia_value=Float64(computed), excel_value=missing,
                     abs_error=missing, rel_error=missing, within_tolerance=false,
                     status="Missing original cell"))
        return
    end
    original isa Real || throw(ArgumentError("Expected numeric results cell $sheet!$cell"))
    official = Float64(original)
    delta = abs(Float64(computed) - official)
    push!(rows, (section=section, object=object, sheet=sheet, cell=cell,
                 julia_value=Float64(computed), excel_value=official,
                 abs_error=delta, rel_error=delta/max(abs(official),1.0),
                 within_tolerance=delta<=tolerance,
                 status=delta<=tolerance ? "Match within numerical precision" : "Different"))
end

function _excel_col(n::Integer)
    n>=1 || throw(ArgumentError("Invalid column $n"))
    buf = ""
    k = Int(n)
    while k>0
        k, r = divrem(k-1, 26)
        buf = string(Char(Int('A')+r), buf)
    end
    return buf
end

function _source_matrix_comparisons!(rows::Vector{NamedTuple}, book,
                                     mode::Symbol, base)
    name = mode === :aggregate ? "Sheet1" : "Sheet2"
    for (typ, col0) in ((:production, 2), (:cost, 7))
        matrix = typ === :production ? base.mes_production : base.mes_cost
        for i in 1:3, j in 1:3
            row = i+4
            col = col0+j-1
            _addcomparison!(rows, "benchmark", "$(mode)_$(typ)_raw[$i,$j]",
                            name, string(_excel_col(col),row), matrix[i,j],
                            book[name][row,col]; tolerance=2e-10)
        end
    end
end

function _sheet5_comparisons!(rows::Vector{NamedTuple}, book, run)
    s = book["Sheet5"]
    baseline_rows = Dict(:disaggregated=>4, :aggregate=>43)
    sensitivity_rows = Dict(
        :disaggregated => Dict(:theta0=>11,:theta3=>17,:theta4=>23,
                              :theta2=>29,:theta1=>35),
        :aggregate => Dict(:theta2=>50,:theta1=>56))
    for model in ACTIVE_MODELS
        context = getproperty(run.contexts, model)
        for (typ,col0) in ((:production,2),(:cost,7))
            matrix = getproperty(context.baseline, Symbol("mes_",typ))
            row0 = baseline_rows[model]
            for i in 1:3, j in 1:3
                row = row0+i-1; col = col0+j-1
                _addcomparison!(rows,"Sheet5 benchmark","$(model)_$(typ)[$i,$j]",
                                "Sheet5",string(_excel_col(col),row), matrix[i,j],
                                s[row,col]; tolerance=2e-10)
            end
            for (parameter,start_row) in sensitivity_rows[model]
                sens = run.sensitivities[model][parameter]
                derivative = getproperty(sens,Symbol("mes_",typ))
                for i in 1:3, j in 1:3
                    row=start_row+i-1; col=col0+j-1
                    _addcomparison!(rows,"Sheet5 derivative",
                      "$(model)_$(parameter)_$(typ)[$i,$j]", "Sheet5",
                      string(_excel_col(col),row), derivative[i,j],s[row,col];
                      tolerance=2e-5)
                end
            end
        end
    end
end

function _sheet3_comparisons!(rows::Vector{NamedTuple}, book, run)
    s=book["Sheet3"]
    for (model,col0) in ((:aggregate,2),(:disaggregated,5)),
        (typ,offset) in ((:cost,0),(:production,1))
        context=getproperty(run.contexts,model)
        value=getproperty(context.baseline, Symbol("mes_",typ))[1,3]
        _addcomparison!(rows,"Sheet3 benchmark","$(model)_$(typ)_HK",
                        "Sheet3", string(_excel_col(col0+offset),5), value,
                        s[5,col0+offset]; tolerance=5e-10)
    end
    for (parameter,row) in ((:theta0,7),(:theta3,8),(:theta4,9),
                            (:theta2,10),(:theta1,11))
        # The 0s for inactive aggregate microelasticities are explicit in Sheet3.
        for (model,col0) in ((:aggregate,2),(:disaggregated,5)),
            (typ,offset) in ((:cost,0),(:production,1))
            value = model===:aggregate && !(parameter in AGG_PARAMETERS) ? 0.0 :
               getproperty(run.sensitivities[model][parameter], Symbol("mes_",typ))[1,3]
            cell=string(_excel_col(col0+offset),row)
            _addcomparison!(rows,"Sheet3 derivative","$(model)_$(parameter)_$(typ)_HK",
                 "Sheet3",cell,value,s[row,col0+offset]; tolerance=5.0e-5)
        end
    end
end

function _sheet4_comparisons!(rows::Vector{NamedTuple}, book, run)
    s=book["Sheet4"]
    for (model,ii) in ((:aggregate,1),(:disaggregated,2))
        base = getproperty(run.contexts,model).baseline
        cf=compute_capital_counterfactual(base.nonelast,base.elast)
        for (typ,row) in ((:production,3+ii),(:cost,8+ii))
            v = getproperty(cf,Symbol("capital_shock_",typ))
            for j in 1:3
                _addcomparison!(rows,"Sheet4 capital shock",
                  "$(model)_$(typ)_$(FACTOR_NAMES[j])", "Sheet4",
                  string(_excel_col(j+1),row), v[j],s[row,j+1];
                  tolerance=5.1e-5)
            end
        end
        # Sheet4!G4:G5 is computed from rounded factor-share changes,
        # not necessarily the difference of the unrounded internal changes.
        value=cf.capital_shock_production[3]-cf.capital_shock_production[2]
        printed=s[3+ii,7]
        _addcomparison!(rows,"Sheet4 derived skill premium",
            "$(model)_college_minus_noncollege_unrounded", "Sheet4",
            string("G",3+ii),value,printed;tolerance=1.1e-4)
    end
end

"""
    compare_matlab_julia_paper(book, run)

Compare all numerical workbook cells against the actual existing Julia
benchmark/sensitivity functions. The workbook's original MAT-oriented
3×3 matrices are never silently transposed.
"""
function compare_matlab_julia_paper(book, run)
    rows=NamedTuple[]
    for model in ACTIVE_MODELS
        _source_matrix_comparisons!(rows,book,model,
            getproperty(run.contexts,model).baseline)
    end
    _sheet3_comparisons!(rows,book,run)
    _sheet4_comparisons!(rows,book,run)
    _sheet5_comparisons!(rows,book,run)
    return rows
end

function _published_string(pair::Symbol, model::Symbol, typ::Symbol,
                           parameter::Symbol)
    paper=getproperty(KrusellSensitivities.PUBLISHED,pair)
    key=Symbol(string(model),"_",string(typ))
    return parameter===:benchmark ? getproperty(paper.benchmark,key) :
           getproperty(getproperty(paper,key),parameter)
end

function _paper_row(table::String, row::String, col::String, object::String,
                    model::String, parameter::String, calc::Real,
                    published::AbstractString, bookcell::AbstractString,
                    excelval; swap::Bool=false,
                    evidence::String="Original MAT and source-matched Julia")
    classification=classify_discrepancies(calc,published;row_swap_match=swap)
    return (table=table, row=row, column=col, object=object,
            model=model, parameter=parameter, matlab_equivalent=Float64(calc),
            official_excel_value=excelval, published=published,
            raw_gap=Float64(calc)-parse(Float64,published),
            classification=classification,
            evidence=string(evidence,"; official workbook ",bookcell))
end

function _publication_audit(book,run)
    out=NamedTuple[]
    for (model,table,prod,cost) in ((:aggregate,"B.1",B1_PROD,B1_COST),
                                    (:disaggregated,"B.2",B2_PROD,B2_COST))
        base=getproperty(run.contexts,model).baseline
        workbook=model===:aggregate ? "Sheet1" : "Sheet2"
        for (typ,reference,col0) in ((:production,prod,2),(:cost,cost,7))
            calc=getproperty(base,Symbol("mes_",typ))
            for i in 1:3,j in 1:3
                i==j && continue
                published=@sprintf("%.2f",reference[i,j])
                # Paper row i, column j equals MATLAB raw [j,i].
                original=calc[j,i]
                rawcell=string(_excel_col(col0+i-1),4+j)
                ev=book[workbook][4+j,col0+i-1]
                push!(out,_paper_row(table,FACTOR_NAMES[i],FACTOR_NAMES[j],
                    string(typ," MES"),String(model),"benchmark",original,
                    published,string(workbook,"!",rawcell),ev))
            end
        end
    end
    for (model,table,expected,row) in ((:aggregate,"B.5",B5_AGG,4),
                                       (:disaggregated,"B.5",B5_DIS,5))
        base=getproperty(run.contexts,model).baseline
        cf=compute_capital_counterfactual(base.nonelast,base.elast)
        for j in 1:3
            pub=@sprintf("%.2f",expected[j])
            excelcell=string("Sheet4!",_excel_col(j+1),row)
            push!(out,_paper_row(table,String(model),FACTOR_NAMES[j],
                "Production income-share change",String(model),"capital shock",
                cf.capital_shock_production[j],pub,excelcell,
                book["Sheet4"][row,j+1]))
        end
    end
    for (pair,table,idx) in ((:HK,"B.3",3),(:LK,"B.4",2)),
        model in ACTIVE_MODELS, typ in TYPES
        base=getproperty(run.contexts,model).baseline
        calc=getproperty(base,Symbol("mes_",typ))[1,idx]
        pub=_published_string(pair,model,typ,:benchmark)
        col0=typ===:production ? 2 : 7
        baserow=model===:aggregate ? 43 : 4
        rawcell=string("Sheet5!",_excel_col(col0+idx-1),baserow)
        push!(out,_paper_row(table,"Benchmark",String(pair),"MES",
                    String(model),"benchmark",calc,pub,rawcell,
                    book["Sheet5"][baserow,col0+idx-1]))
        for parameter in (model === :aggregate ? AGG_PARAMETERS : PARAMETERS)
            sens=run.sensitivities[model][parameter]
            derivative=getproperty(sens,Symbol("mes_",typ))[1,idx]
            published=_published_string(pair,model,typ,parameter)
            baserow=if model===:aggregate
                parameter===:theta2 ? 50 : 56
            else
                get(Dict(:theta0=>11,:theta3=>17,:theta4=>23,:theta2=>29,:theta1=>35),parameter,0)
            end
            cell=string("Sheet5!",_excel_col(col0+idx-1),baserow)
            excel=book["Sheet5"][baserow,col0+idx-1]
            # Explicitly verify swapped theta1/theta2 row/cell labels in B.4;
            # theta4 B.4 cost/production is printed in opposite columns.
            alternative = if pair===:LK && parameter in (:theta1,:theta2)
                other=parameter===:theta1 ? :theta2 : :theta1
                getproperty(run.sensitivities[model][other],Symbol("mes_",typ))[1,idx]
            elseif pair===:LK && model===:disaggregated && parameter===:theta4
                other=typ===:production ? :cost : :production
                getproperty(sens,Symbol("mes_",other))[1,idx]
            else
                NaN
            end
            plausible_swap=isfinite(alternative) && _printed_match(alternative,published)
            push!(out,_paper_row(table,String(parameter),String(pair),
                  string(typ," sensitivity"),String(model),String(parameter),
                  derivative,published,cell,excel;swap=plausible_swap,
                  evidence=plausible_swap ? "Exact source-computed value agrees under a theta1/theta2 row or cost/production column switch" :
                    "Original MAT finite difference and analytic derivative"))
        end
    end
    return out
end

function _write_csv(path::AbstractString, headers::Vector{String}, rows)
    function escape_cell(value)
        (value===missing || value===nothing) && return ""
        s = if value isa AbstractFloat
            isfinite(value) ? @sprintf("%.16g",value) : string(value)
        else
            string(value)
        end
        if occursin(',',s)||occursin('"',s)||occursin('\n',s)||occursin('\r',s)
            return string('"',replace(s,"\""=>"\"\""),'"')
        end
        return s
    end
    open(path,"w") do io
        println(io,join(escape_cell.(headers),","))
        for row in rows
            println(io,join(escape_cell.(row),","))
        end
    end
    return path
end

function _named_rows(rows::AbstractVector{<:NamedTuple})
    isempty(rows) && return (String[],Vector{Vector{Any}}())
    keys=String.(propertynames(first(rows)))
    data=Vector{Vector{Any}}()
    for row in rows
        push!(data,Any[getproperty(row,Symbol(k)) for k in keys])
    end
    return keys,data
end

function _save_named(path, rows)
    keys,data=_named_rows(rows)
    isempty(keys) && throw(ArgumentError("Cannot export empty comparison table: $path"))
    return _write_csv(path,keys,data)
end

function _matrix_text(rows::Vector{Vector{String}})
    n=length(rows)
    n>0 || throw(ArgumentError("Empty LaTeX table"))
    p=length(rows[1])
    all(x -> length(x)==p,rows) || throw(DimensionMismatch("Ragged LaTeX table"))
    table=Matrix{String}(undef,n,p)
    for i in 1:n, j in 1:p
        table[i,j]=rows[i][j]
    end
    return table
end

function _summary_tex(path, matrix::Matrix{String})
    t=Table(Cell.(matrix);header=1)
    open(path,"w") do io
        show(io,MIME"text/latex"(),t)
        println(io)
    end
    return path
end

function _data_provenance_rows(source,io,production,data)
    rows=NamedTuple[]
    raw_labels=_matrix_range(source.io_path,"2013","A8:B90",83,2)
    for i in 1:83,j in 1:98
        official=data.io[i,j]
        val=io.values[i,j]
        push!(rows,(mat_variable="IO2013", source_workbook=IO_WORKBOOK,
            source_sheet="2013",source_cell=string(_excel_col(j+2),i+7),
            io_code=string(raw_labels[i,1]), industry=string(raw_labels[i,2]),
            mat_row=i, mat_col=j, year=2013, source_value=val,
            mat_value=official, exact=isequal(val,official),
            note="... mapped to zero; source negatives retained"))
    end
    # Industry codes from the 2013 IO original row order. MATLAB merges and
    # deletes sectors before retaining exactly 63; five descriptions differ
    # from the production accounts because they describe merged aggregates.
    drop = Set(vcat(collect(29:31), [49,60,68,69], collect(71:83)))
    keep = [k for k in 1:83 if !(k in drop)]
    length(keep) == 63 || error("Original IO row concordance must retain 63 industries")
    p_rows=NamedTuple[]
    for spec in PRODUCTION_SHEETS
        vals=production.variables[spec.variable]
        official=if spec.variable=="output_new"
            data.output
        elseif spec.variable=="VA_new"
            data.va
        elseif spec.variable=="labor_no_col"
            data.labor_noncollege
        else
            data.labor_college
        end
        for i in 1:63
            oldrow = keep[i]
            io_code = string(raw_labels[oldrow,1])
            io_label = string(raw_labels[oldrow,2])
            same_label = production.descriptions[i] == io_label
            push!(p_rows,(mat_variable=spec.variable,
                 source_workbook=PRODUCTION_WORKBOOK,
                 source_sheet=spec.sheet,source_cell=string("Q",i+2),
                 industry=production.descriptions[i], year=2013,
                 original_row=i, io_original_row=oldrow,
                 io_industry_code=io_code, io_retained_label=io_label,
                 io_label_matches_production=same_label,
                 source_value=vals[i],mat_value=official[i],
                 exact=isequal(vals[i],official[i]),
                 note=same_label ? "Same industry label and row order" :
                   "Merged-sector labels differ; index mapping grounded in MATLAB IO deletions"))
        end
    end
    return rows,p_rows
end

function _orientation_rows(book,run)
    rows=NamedTuple[]
    for model in ACTIVE_MODELS, typ in TYPES
        base=getproperty(run.contexts,model).baseline
        matrix=getproperty(base,Symbol("mes_",typ))
        for i in 1:3,j in 1:3
            push!(rows,(model=String(model), mes_type=String(typ),
                matlab_row=FACTOR_NAMES[i],matlab_col=FACTOR_NAMES[j],
                raw=matrix[i,j],paper_row=FACTOR_NAMES[j],
                paper_col=FACTOR_NAMES[i],paper_oriented=matrix[i,j],
                transformation="paper display = transpose(MATLAB raw MES)"))
        end
    end
    return rows
end

function _label_rows(book,run)
    rows=NamedTuple[]
    for model in ACTIVE_MODELS, typ in TYPES,
        parameter in (:theta2,:theta1,:theta4)
        model===:aggregate && parameter===:theta4 && continue
        sens=run.sensitivities[model][parameter]
        val=getproperty(sens,Symbol("mes_",typ))[1,2]
        altparam=parameter===:theta2 ? :theta1 : parameter===:theta1 ? :theta2 : :theta4
        other=parameter===:theta4 ?
              getproperty(sens,Symbol("mes_",typ===:cost ? :production : :cost))[1,2] :
              getproperty(run.sensitivities[model][altparam],Symbol("mes_",typ))[1,2]
        printed=_published_string(:LK,model,typ,parameter)
        push!(rows,(table="B.4",model=String(model),parameter=String(parameter),
             type=String(typ),source_derivative=val,published=printed,
             alternate_source_derivative=other,
             agrees_under_switch=_printed_match(other,printed),
             switch_kind=parameter===:theta4 ? "cost/production column" : "theta1/theta2 row"))
    end
    return rows
end

function _inventory(sources)
    rows=NamedTuple[]
    for (file,names) in ((IO_WORKBOOK,sources.io_sheets),
                         (PRODUCTION_WORKBOOK,sources.production_sheets),
                         (RESULT_WORKBOOK,sources.result_sheets))
        for s in names
            area = if file == IO_WORKBOOK && s == "2013"
                "C8:CV90"
            elseif file == PRODUCTION_WORKBOOK && s in (x.sheet for x in PRODUCTION_SHEETS)
                "Q3:Q65"
            elseif file == RESULT_WORKBOOK
                "Official result tab"
            else
                "Not used in 2013 benchmark"
            end
            push!(rows,(workbook=file,sheet=s,year=2013,
                       audited_range=area, status=area=="Not used in 2013 benchmark" ?
                       "Inventoried" : "Audited"))
        end
    end
    return rows
end

function _diagnostics(run,src_compare,official_comparison,paper_audit,io)
    rows=NamedTuple[]
    push!(rows,(metric="Excel to MAT exact element count",
               result=string(sum(x.matches for x in src_compare)),unit="cells"))
    push!(rows,(metric="Excel to MAT mismatch count",
               result=string(sum(x.mismatches for x in src_compare)),unit="cells"))
    push!(rows,(metric="IO source dots mapped to zero",result=string(io.dots_as_zero),unit="cells"))
    push!(rows,(metric="IO source negative entries retained",result=string(io.negative_count),unit="cells"))
    push!(rows,(metric="Official results numerical comparison count",
               result=string(length(official_comparison)),unit="cells"))
    push!(rows,(metric="Official results outside respective tolerances",
               result=string(count(x -> x.status=="Different",official_comparison)),unit="cells"))
    push!(rows,(metric="Published table comparisons",result=string(length(paper_audit)),unit="cells"))
    for classification in ("A","B","C","D","E","F","G")
        push!(rows,(metric="Published discrepancy class $(classification)",
             result=string(count(x->x.classification==classification,paper_audit)),unit="cells"))
    end
    for model in ACTIVE_MODELS
        context=getproperty(run.contexts,model)
        eq=context.equilibrium
        r=maximum(abs,(Matrix{Float64}(I,329,329)-eq.omega)*eq.psi-Matrix{Float64}(I,329,329))
        push!(rows,(metric="$(model) Leontief max residual",result=@sprintf("%.16g",r),unit="absolute"))
        push!(rows,(metric="$(model) Hessian symmetry max residual",
          result=@sprintf("%.16g",maximum(abs,context.baseline_hessian-transpose(context.baseline_hessian))),unit="absolute"))
        push!(rows,(metric="$(model) factors 326:328 Domar sum",
          result=@sprintf("%.16g",sum(eq.lambda[326:328])),unit="income share"))
        push!(rows,(metric="$(model) terminal node 329 Domar weight",
          result=@sprintf("%.16g",eq.lambda[329]),unit="weight"))
    end
    return rows
end

function _markdown_report(src_compare,official_comparison,paper_audit,io,run)
    source_total=sum(x.n for x in src_compare)
    source_mismatch=sum(x.mismatches for x in src_compare)
    officially_different=filter(x -> x.status=="Different",official_comparison)
    source_status=source_mismatch==0 ? "EXACT" : "DISCREPANCY"
    io = IOBuffer()
    println(io,"# Baqaee and Farhi (2019): Source-to-publication reproducibility audit")
    println(io)
    println(io,"This report distinguishes source-file provenance, official MATLAB-equivalent numerical results, ",
            "the officially supplied `results tables1-5.xlsx`, and printed journal tables. ",
            "It does not claim the MATLAB executable was run.")
    println(io)
    println(io,"## 1. Source Excel → official MAT files: **$source_status**")
    println(io)
    println(io,"| MATLAB variable | Elements | Exact cells | Mismatches | Max abs. error |")
    println(io,"|---|---:|---:|---:|---:|")
    for r in src_compare
        println(io,"| `$(r.variable)` | $(r.n) | $(r.matches) | $(r.mismatches) | $(r.max_abs) |")
    end
    println(io)
    println(io,"Total $source_total cells; $source_mismatch mismatches. ",
            "Original IO2013 is copied from `2013!C8:CV90` in the BEA Use workbook ",
            "(83×98); literal `...` entries are converted to numerical zeros. ",
            "Values, zeros, negatives and ordering are retained. ",
            "The four production-account vectors each come from their named sheet's ",
            "2013 column `Q3:Q65` (63×1). Source files indicate millions of US dollars. ",
            "These ranges are grounded in sheet headings, row/column identifiers and the ",
            "2013 year header, not in a numerically optimized matching search.")
    println(io)
    println(io,"## 2. Official archived result workbook")
    println(io)
    println(io,"Compared $(length(official_comparison)) accessible numerical cells/positions ",
            "in Sheets 1–5 with the verified Julia numerical routines.")
    println(io,"Cells exceeding their documented comparison tolerance: $(length(officially_different)).")
    for r in officially_different[1:min(length(officially_different),20)]
        println(io,"- $(r.sheet)!$(r.cell), $(r.object): Julia=$(r.julia_value), Excel=$(r.excel_value), ",
                "absolute gap=$(r.abs_error)")
    end
    println(io)
    println(io,"**Precision:** Sheets 1, 2 and 5 save substantially more digits than ",
            "the journal tables. Sheet3 saves derivatives to about four decimal places; ",
            "Sheet4 saves capital-shock components to about four decimal places. ",
            "Comparisons respect those differences.")
    println(io)
    println(io,"## 3. Published tables: source-supported differences")
    println(io)
    println(io,"| Journal location | MATLAB-equivalent / official Excel | Printed value | Classification |")
    println(io,"|---|---:|---:|---|")
    for r in paper_audit
        if r.classification in ("E","F")
            exc=string(r.table," / ",r.model," / ",r.row," / ",r.column," / ",r.object)
            println(io,"| ",replace(exc,"|"=>"/")," | ",@sprintf("%.9f",r.matlab_equivalent),
                    " | ",r.published," | ",r.classification," |")
        end
    end
    println(io)
    println(io,"Classification A: exact; B: agrees at published precision; ",
           "E: a verified source row/column label switch would reconcile; ",
           "F: remaining numerical publication discrepancy. ",
           "Class E describes a plausible *layout* explanation, not proven author intent.")
    println(io)
    println(io,"### High-priority interpretation")
    println(io)
    println(io,"- Tables B.1–B.2: official Excel matrices are in *MATLAB raw* orientation. ",
           "The paper displays their transpose. The 1.064856… vs 1.07 (B.1) and ",
           "0.663104… vs 0.67 (B.2) cells remain different at two decimals.")
    println(io,"- Table B.3: source workbook Sheet3 gives rounded sensitivity values, ",
           "and Sheet5 records much greater precision. The journal's theta2 values ",
           "do not both follow ordinary rounding of the source values.")
    println(io,"- Table B.4: parameter-label comparison against Sheet5 supports ",
           "a theta1/theta2 row-placement discrepancy, across both models and ",
           "both Cost and Production. The disaggregated theta4 Cost/Production ",
           "publication columns also appear reversed relative to Sheet5. ",
           "These hypotheses are not evidence of different structural parameters.")
    println(io,"- Table B.5: the Production side reproduces all three printed ",
           "capital-shock effects for both models. The cost side is a distinct object.")
    println(io)
    println(io,"## 4. Method and scope")
    println(io)
    println(io,"- Archive README identifies the original 2013 BEA Use table (release ",
           "November 2, 2017) and BEA/BLS production accounts.")
    println(io,"- `Krusell_sens.jl` reuses `Krusell_disagg.jl`, `Krusell_agg.jl`, ",
           "`Relabel_mod.jl`, `WeightedCovMatrix.jl` without modifications.")
    println(io,"- All MATLAB-equivalent calculations use the source supplied ",
           "five MAT arrays and exactly the effective calibrations of the archive.")
    println(io,"- Numerical properties audited include Leontief inverse residual, ",
           "Hessian symmetry, factor shares, and sensitivity orientation.")
    println(io,"- MATLAB binary execution was NOT performed by this script. ",
           "Source-code equivalence is supported by previously independently checked ",
           "NumPy computations and the original results Excel workbook.")
    println(io)
    println(io,"## 5. Certification matrix")
    println(io)
    println(io,"| Link | Status |")
    println(io,"|---|---|")
    println(io,"| Source Excel → MAT | ",source_status," |")
    println(io,"| Official MAT → aggregate & disaggregated numerical models | Validated previously; rerun in this audit |")
    println(io,"| Official MAT → capital-shock simulation | Validated previously; rerun in this audit |")
    println(io,"| Official MAT → MES sensitivities | Validated previously; rerun in this audit |")
    println(io,"| Official results workbook ↔ Julia | ",isempty(officially_different) ? "Match at source precision" : "Listed differences remain"," |")
    println(io,"| Official results workbook ↔ printed journal | Discrepancies classified above |")
    println(io,"| Figures 3–4 | Verified in Tasks I–II; not recomputed here |")
    println(io,"| Julia 1.12 runtime | Determined by the user's actual execution log, not inferred by script creation |")
    println(io)
    println(io,"**Conclusion:** The five numerical input MAT files have an exact, ",
           "auditable mapping to the two official source Excel workbooks. ",
           "Reproducibility of the computational model does NOT imply perfect ",
           "identity with every printed journal table. Explicit exceptions are ",
           "recorded in `paper_table_discrepancies.csv` and `sensitivity_label_audit.csv`.")
    return String(take!(io))
end

"""Generate the eight audit CSVs, one Markdown report and two SummaryTables LaTeX tables."""
function save_audit_results(result, out_dir::AbstractString)
    mkpath(out_dir)
    files=String[]
    push!(files,_save_named(joinpath(out_dir,"source_workbook_inventory.csv"),
                 _inventory(result.sources)))
    push!(files,_save_named(joinpath(out_dir,"io2013_source_mapping.csv"),
                 result.io_rows))
    push!(files,_save_named(joinpath(out_dir,"production_account_mapping.csv"),
                 result.production_rows))
    push!(files,_save_named(joinpath(out_dir,"source_mat_comparison.csv"),
                 result.source_comparison))
    push!(files,_save_named(joinpath(out_dir,"official_results_workbook_comparison.csv"),
                 result.official_comparison))
    push!(files,_save_named(joinpath(out_dir,"paper_table_discrepancies.csv"),
                 result.paper_comparison))
    push!(files,_save_named(joinpath(out_dir,"mes_orientation_audit.csv"),
                 _orientation_rows(result.book,result.run)))
    push!(files,_save_named(joinpath(out_dir,"sensitivity_label_audit.csv"),
                 _label_rows(result.book,result.run)))
    push!(files,_save_named(joinpath(out_dir,"BF2019_reproducibility_diagnostics.csv"),
                 _diagnostics(result.run,result.source_comparison,
                     result.official_comparison,result.paper_comparison,result.io)))
    report=_markdown_report(result.source_comparison,result.official_comparison,
                            result.paper_comparison,result.io,result.run)
    report_path=joinpath(out_dir,"BF2019_reproducibility_audit.md")
    write(report_path,report);push!(files,report_path)

    disp=Vector{Vector{String}}()
    push!(disp,["Table", "Model", "Object", "Code result", "Printed", "Class"])
    for r in result.paper_comparison
        if r.classification in ("E","F")
            push!(disp,[r.table,r.model,string(r.row," / ",r.column),
                        @sprintf("%.6f",r.matlab_equivalent),r.published,r.classification])
        end
    end
    push!(files,_summary_tex(joinpath(out_dir,"BF2019_discrepancy_summary.tex"),
                        _matrix_text(disp)))
    status=Vector{Vector{String}}()
    push!(status,["Link", "Status", "Evidence"])
    # Avoid unparenthesized ternaries within array literals: Julia's parser
    # requires whitespace around the colon separating the two alternatives.
    source_state = if all(x -> x.mismatches == 0, result.source_comparison)
        "Exact"
    else
        "Different"
    end
    push!(status, ["Excel to MAT", source_state, "8,386 mapped source cells"])
    push!(status, ["MAT to models", "Previously validated", "Julia Tasks V-VII"])
    workbook_state = if all(x -> x.within_tolerance, result.official_comparison)
        "Match"
    else
        "Exceptions logged"
    end
    push!(status, ["Results workbook to Julia", workbook_state,
                   "Official workbook Sheets 1-5"])
    push!(status,["Workbook to paper","Exceptions logged","Tables B.1-B.5"])
    push!(status,["Figures 3-4","Previously validated","Julia Tasks I-II"])
    push!(status,["MATLAB executable","Not run","Original scripts inspected"])
    push!(files,_summary_tex(joinpath(out_dir,"BF2019_reproducibility_matrix.tex"),
                             _matrix_text(status)))
    return files
end

"""
    run_reproducibility_audit(data_dir, source_dir; out_dir, save=true)

Always load and check the source Excel and MAT files before calculating the
publication results. Never report full provenance from MAT-only calculations.
"""
function run_reproducibility_audit(data_dir::AbstractString,
                                   source_dir::AbstractString;
                                   out_dir::AbstractString=joinpath(@__DIR__,"output_krusell_data_audit"),
                                   save::Bool=true)
    sources=load_official_source_workbooks(source_dir)
    layout=inspect_source_data_layout(sources)
    data=load_official_mat_data(data_dir)
    io=reconstruct_io2013(sources)
    production=reconstruct_production_accounts(sources)
    source_comparison=compare_source_and_mat(io,production,data)
    all(x->x.mismatches==0,source_comparison) || throw(ArgumentError(
        "Official Excel and MAT data do not exactly match; stop certification"))
    book=read_official_results_workbook(sources)
    run=compute_all_sensitivities(data;step=1e-5)
    official_comparison=compare_matlab_julia_paper(book,run)
    paper_comparison=_publication_audit(book,run)
    io_rows,production_rows=_data_provenance_rows(sources,io,production,data)
    result=(sources=sources,layout=layout,data=data,io=io,
            production=production,book=book,run=run,
            source_comparison=source_comparison,
            official_comparison=official_comparison,
            paper_comparison=paper_comparison,
            io_rows=io_rows,production_rows=production_rows)
    if save
        files=save_audit_results(result,out_dir)
        foreach(path->println("Wrote: ",abspath(path)),files)
    end
    return result
end

"""Unit tests use synthetic inputs; official-file integration tests run when both paths supplied."""
function run_tests(;data_dir::Union{Nothing,AbstractString}=nothing,
                    source_dir::Union{Nothing,AbstractString}=nothing)
    @testset "Krusell_data_audit.jl" begin
        @testset "Pure validation" begin
            @test _required_numeric(3,"A1")==3
            @test _required_numeric("...","A1";allow_dots=true)==0
            @test _required_numeric(-5,"A1")==-5
            @test_throws ArgumentError _required_numeric("not recorded","A1";allow_dots=true)
            @test_throws ArgumentError _required_numeric(missing,"A1")
            @test_throws ArgumentError _required_numeric(NaN,"A1")
            @test _excel_col(1)=="A"
            @test _excel_col(26)=="Z"
            @test _excel_col(27)=="AA"
            @test _excel_col(100)=="CV"
            @test_throws ArgumentError _excel_col(0)
            @test _printed_match(0.93552,"0.94")
            @test !_printed_match(0.93552,"0.93")
            @test classify_discrepancies(0.6631044206,"0.67")=="F"
            @test classify_discrepancies(0.239565,"0.97";row_swap_match=true)=="E"
            @test classify_discrepancies(1.0,"1") == "A"
            @test classify_discrepancies(0.13030,"0.13") == "B"
            @test _max_relative([1.0,0.0],[1.0,0.0])==0
            @test_throws DimensionMismatch _max_relative(zeros(2),zeros(3))
            @test_throws ArgumentError load_official_source_workbooks(joinpath(tempdir(),"not_existing_987645"))
            @test_throws ArgumentError load_official_mat_data(joinpath(tempdir(),"not_existing_987645"))
            v=(io=ones(83,98),output=ones(63),va=ones(63),
              labor_noncollege=zeros(63),labor_college=zeros(63))
            x=(values=ones(83,98),)
            p=(variables=Dict("output_new"=>ones(63),"VA_new"=>ones(63),
               "labor_no_col"=>zeros(63),"labor_col"=>zeros(63)),)
            r=compare_source_and_mat(x,p,v)
            @test length(r)==5
            @test all(z->z.mismatches==0,r)
            @test sum(z->z.n,r)==8386
        end
        if data_dir !== nothing || source_dir !== nothing
            data_dir===nothing && throw(ArgumentError("--data-dir is required for source audit tests"))
            source_dir===nothing && throw(ArgumentError("--source-dir is required for source audit tests"))
            @testset "Original files and source-to-model integration" begin
                audit=run_reproducibility_audit(data_dir,source_dir;save=false)
                @test audit.layout.io_year==2013
                @test audit.layout.production_year==2013
                @test size(audit.io.values)==(83,98)
                @test audit.io.dots_as_zero==2555
                @test length(audit.production.descriptions)==63
                @test length(audit.io_rows)==8134
                @test length(audit.production_rows)==252
                @test count(x->x.mat_variable == "output_new" && !x.io_label_matches_production,
                            audit.production_rows)==5
                @test sum(x->x.matches,audit.source_comparison)==8386
                @test all(x->x.mismatches==0,audit.source_comparison)
                @test all(x->x.max_abs==0,audit.source_comparison)
                @test length(audit.book)==5
                @test length(audit.paper_comparison)>50
                @test count(x->x.classification=="E",audit.paper_comparison)>=8
                @test count(x->x.classification=="F",audit.paper_comparison)>=2
                @test maximum(abs,audit.run.contexts.aggregate.equilibrium.lambda[326:328] .-
                        [0.45752535,0.24393952,0.29853513]) < 2e-7
                @test maximum(abs,audit.run.contexts.disaggregated.equilibrium.lambda[326:328] .-
                        [0.453580444324,0.251626082980,0.294793472695]) < 3e-10
                @test iszero(audit.run.contexts.aggregate.equilibrium.lambda[329])
                @test iszero(audit.run.contexts.disaggregated.equilibrium.lambda[329])
                @test all(x->x.status!="Missing original cell",audit.official_comparison)
                @test all(x->x.within_tolerance,audit.official_comparison)
                @test audit.book["Sheet5"][50,3] ≈ 0.239565749593496
                @test audit.book["Sheet5"][56,3] ≈ 0.971152639328565
                @test audit.book["Sheet5"][29,3] ≈ 0.220585000498819
                @test audit.book["Sheet5"][35,3] ≈ 0.569561950214403
                @test audit.book["Sheet4"][4,2] ≈ 0.0501
                # Determinism of source-data reconstruction and paper comparison.
                @test audit.io.values == reconstruct_io2013(audit.sources).values
                @test audit.paper_comparison == _publication_audit(audit.book,audit.run)
            end
        end
    end
    return nothing
end

"""Warm benchmarks, excluding the initial JIT compilation of each stage."""
function run_benchmark(data_dir::AbstractString,source_dir::AbstractString;
                       samples::Int=5)
    samples>=2 || throw(ArgumentError("Need at least 2 benchmark samples"))
    sources=load_official_source_workbooks(source_dir)
    data=load_official_mat_data(data_dir)
    functions=(
      ("Load source XLSX metadata",()->load_official_source_workbooks(source_dir)),
      ("Extract 2013 IO matrix",()->reconstruct_io2013(sources)),
      ("Extract 2013 production accounts",()->reconstruct_production_accounts(sources)),
      ("Load five MAT arrays",()->load_official_mat_data(data_dir)),
      ("Read official results workbook",()->read_official_results_workbook(sources)),
      ("Aggregate/disaggregated sensitivities",()->compute_all_sensitivities(data)),
      ("Complete audit, excluding outputs",()->run_reproducibility_audit(data_dir,source_dir;save=false))
    )
    println("\nTask VIII benchmark (warmed; $(samples) samples)")
    @printf("%-43s %11s %15s\n","Stage","median ms","median alloc B")
    for (name,fun) in functions
        fun() # compile/warm
        times=Float64[];bytes=Int[]
        for _ in 1:samples
            GC.gc()
            push!(bytes,@allocated fun())
            push!(times,@elapsed fun())
        end
        @printf("%-43s %11.3f %15d\n",name,1000 * median(times),round(Int,median(bytes)))
    end
    return nothing
end

function _cli(args::Vector{String})
    data_dir=nothing
    source_dir=nothing
    out_dir=joinpath(@__DIR__,"output_krusell_data_audit")
    bench=false
    test_only=false
    no_tests=false
    i=1
    while i<=length(args)
        arg=args[i]
        if arg=="--bench"
            bench=true
        elseif arg=="--test-only"
            test_only=true
        elseif arg=="--no-tests"
            no_tests=true
        elseif arg in ("--data-dir","--source-dir","--out-dir")
            i<length(args) || throw(ArgumentError("Missing value for $arg"))
            value=args[i+1]
            if arg == "--data-dir"
                data_dir = value
            elseif arg == "--source-dir"
                source_dir = value
            else
                out_dir = value
            end
            i+=1
        else
            throw(ArgumentError("Unknown option: $arg"))
        end
        i+=1
    end
    if test_only
        run_tests()
        return nothing
    end
    data_dir===nothing && throw(ArgumentError("Pass --data-dir PATH"))
    source_dir===nothing && throw(ArgumentError("Pass --source-dir PATH with three original Excel files"))
    !no_tests && run_tests(;data_dir=data_dir,source_dir=source_dir)
    result=run_reproducibility_audit(data_dir,source_dir;out_dir=out_dir)
    println("Source MAT: ",sum(x->x.matches,result.source_comparison)," / 8386 cells exact")
    println("Official workbook: ",count(x->x.within_tolerance,result.official_comparison),
            " / ",length(result.official_comparison)," cells match at stored precision")
    println("Printed discrepancies: ",count(x->x.classification in ("E","F"),result.paper_comparison))
    bench && run_benchmark(data_dir,source_dir)
    return nothing
end

end # module KrusellDataAudit

if abspath(PROGRAM_FILE) == (@__FILE__)
    KrusellDataAudit._cli(ARGS)
end
