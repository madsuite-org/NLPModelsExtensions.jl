using Documenter
using NLPModelsExtensions

makedocs(
    sitename = "NLPModelsExtensions.jl",
    format = Documenter.HTML(
        prettyurls = Base.get(ENV, "CI", nothing) == "true",
        mathengine = Documenter.KaTeX(),
        size_threshold_ignore = ["reference.md"],
    ),
    modules = [NLPModelsExtensions],
    repo = Documenter.Remotes.GitHub("madsuite-org", "NLPModelsExtensions.jl"),
    checkdocs = :exports,
    clean = true,
    pages = [
        "Home" => "index.md",
        "Batched models" => "batched.md",
        "Parametric models" => "parametric.md",
        "Reference" => "reference.md",
    ],
)

deploydocs(
    repo = "github.com/madsuite-org/NLPModelsExtensions.jl.git",
    devbranch = "main",
    push_preview = true,
)
