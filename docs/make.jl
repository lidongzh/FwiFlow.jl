using Documenter, FwiFlow
makedocs(sitename="FwiFlow", modules=[FwiFlow],
pages = Any[
    "index.md",
    "api.md",
    "tutorials/fwi.md"
],
authors = "Dongzhuo Li and Kailai Xu")

deploydocs(
    repo = "github.com/lidongzh/FwiFlow.jl.git",
)