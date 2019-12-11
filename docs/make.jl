using Documenter, FwiFlow
makedocs(sitename="FwiFlow", modules=[FwiFlow],
pages = Any[
    "index.md",
    "tutorial.md",
    "api.md"
],
authors = "Dongzhuo Li and Kailai Xu")

deploydocs(
    repo = "github.com/lidongzh/FwiFlow.jl.git",
)