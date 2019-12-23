using Documenter, FwiFlow
makedocs(sitename="FwiFlow", modules=[FwiFlow],
pages = Any[
    "index.md",
    "api.md",
    "Tutorial" => ["tutorials/fwi.md","tutorials/flow.md", "tutorials/timefrac.md"]

],
authors = "Dongzhuo Li and Kailai Xu")

deploydocs(
    repo = "github.com/lidongzh/FwiFlow.jl.git",
)