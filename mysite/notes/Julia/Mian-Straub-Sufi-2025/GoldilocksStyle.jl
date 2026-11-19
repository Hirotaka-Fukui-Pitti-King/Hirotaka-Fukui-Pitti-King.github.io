module GoldilocksStyle

using CairoMakie

export OKABE_ITO, MODEL_STYLES, SERIES_LABELS, model_style, publication_theme, save_figure

# Exact Okabe–Ito qualitative palette (hexadecimal sRGB).
# Yellow is kept for annotations rather than default line cycling: it has
# limited contrast against a white background.
const OKABE_ITO = (
    orange         = "#E69F00",
    sky_blue       = "#56B4E9",
    bluish_green   = "#009E73",
    yellow         = "#F0E442",
    blue           = "#0072B2",
    vermilion      = "#D55E00",
    reddish_purple = "#CC79A7",
    black          = "#000000",
)

const LINE_PALETTE = [
    OKABE_ITO.blue,
    OKABE_ITO.vermilion,
    OKABE_ITO.bluish_green,
    OKABE_ITO.orange,
    OKABE_ITO.sky_blue,
    OKABE_ITO.reddish_purple,
    OKABE_ITO.black,
]

# Stable mapping, independent of the order in which a figure plots its models.
# The keys follow the original plotting_options.py and model definitions.
const MODEL_STYLES = (
    bu      = (color = OKABE_ITO.blue,
               linestyle = :solid,
               label = "Bonds in utility - quad."),
    bu_log  = (color = OKABE_ITO.vermilion,
               linestyle = :dash,
               label = "Bonds in utility - log"),
    olg     = (color = OKABE_ITO.bluish_green,
               linestyle = :dot,
               label = "Overlapping generations"),
    olg_alt = (color = OKABE_ITO.orange,
               linestyle = :dashdot,
               label = "Overlapping generations - alt."),
    ha      = (color = OKABE_ITO.reddish_purple,
               linestyle = :solid,
               label = "Idiosyncratic risk"),
)

# Original replication variable names; percentages are plain Unicode text,
# not TeX escapes, to work consistently in PDF and PNG output.
const SERIES_LABELS = (
    deficit  = "Primary deficit (% of initial GDP)",
    B_annual = "Debt (% of initial GDP)",
    Y        = "GDP (% of initial steady state)",
)

"""
    model_style(model::Symbol)

Return the consistent `(color, linestyle, label)` for a model code.
Recognized codes: `:bu`, `:bu_log`, `:olg`, `:olg_alt`, `:ha`.
"""
function model_style(model::Symbol)
    hasproperty(MODEL_STYLES, model) ||
        throw(ArgumentError("Unknown model code: $(model)"))
    return getproperty(MODEL_STYLES, model)
end

"""
    publication_theme(; fontsize=16)

Return a reusable, colorblind-accessible CairoMakie theme.  Use with
`with_theme(publication_theme()) do ... end` to avoid mutating the global theme.
The serif/math fonts are provided by Makie's `theme_latexfonts()`; a
system LaTeX installation is not required for this theme.
"""
function publication_theme(; fontsize::Integer = 16)
    fontsize > 0 || throw(ArgumentError("fontsize must be positive"))

    return merge(
        theme_latexfonts(),
        Theme(
            fontsize = fontsize,
            palette = (color = LINE_PALETTE,),
            Axis = (
                xgridvisible = false,
                ygridvisible = false,
                topspinevisible = false,
                rightspinevisible = false,
                xautolimitmargin = (0.0, 0.0),
            ),
            Lines = (linewidth = 2.5,),
            Legend = (framevisible = false,),
        ),
    )
end

"""
    save_figure(fig::Figure, stem; outdir=joinpath(@__DIR__, "..", "output"),
                formats=("pdf", "png"), px_per_unit=3)

Write publication-ready vector PDF and high-resolution PNG figures. Returns
an array of absolute file paths.  No working-directory assumptions are made.
"""
function save_figure(
    fig::Figure,
    stem::AbstractString;
    outdir::AbstractString = joinpath(@__DIR__, "..", "output"),
    formats = ("pdf", "png"),
    px_per_unit::Real = 3,
)
    isempty(stem) && throw(ArgumentError("stem cannot be empty"))
    px_per_unit > 0 || throw(ArgumentError("px_per_unit must be positive"))
    mkpath(outdir)
    saved = String[]

    for fmt in formats
        extension = String(fmt)
        extension in ("pdf", "png", "svg") ||
            throw(ArgumentError("Unsupported graphics format: $(extension)"))
        path = abspath(joinpath(outdir, "$(stem).$(extension)"))
        if extension == "png"
            save(path, fig; px_per_unit = px_per_unit)
        else
            save(path, fig)
        end
        push!(saved, path)
    end

    return saved
end

end # module GoldilocksStyle
