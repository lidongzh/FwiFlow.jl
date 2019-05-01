dir = @__DIR__

jobs = ["FWI/Src", "FWI", "Laplacian", "Poisson", "Saturation", "Upwlap", "Upwps"]
for j in jobs
    cd("Ops/$j")
    if !isdir("build")
        mkdir("build")
    end
    cd("build")
    run(`cmake ..`)
    run(`make -j`)
    cd(dir)
end
