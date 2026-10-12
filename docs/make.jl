using Documenter
using NLPModelsExtensions

makedocs(
    sitename = "NLPModelsExtensions.jl",
    format = Documenter.HTML(
        prettyurls = Base.get(ENV, "CI", nothing) == "true",
        mathengine = Documenter.KaTeX(),
        size_threshold_ignore = ["ParametricNLPModels/reference.md"],
    ),
    modules = [NLPModelsExtensions],
    repo = Documenter.Remotes.GitHub("madsuite-org", "NLPModelsExtensions.jl"),
    checkdocs = :exports,
    clean = true,
    pages = [
        "Home" => "index.md",
        "ParametricNLPModels" => [
            "Overview" => "ParametricNLPModels/index.md",
            "Reference" => "ParametricNLPModels/reference.md",
        ],
    ],
)

deploydocs(
    repo = "github.com/madsuite-org/NLPModelsExtensions.jl.git",
    devbranch = "main",
    push_preview = true,
)
