using Documenter, SparseSymmetricTensors

makedocs(
    modules = [SparseSymmetricTensors],
    format = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
    authors = "Charlie Colley",
    sitename = "SparseSymmetricTensors.jl",
    pages = Any["index.md"]
    # strict = true,
    # clean = true,
    # checkdocs = :exports,
)

deploydocs(
    repo = "github.com/chuckcol/SparseSymmetricTensors.jl.git",
    push_preview = true
)
