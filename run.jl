dir = @__DIR__

jobs = ["FWI/Src", "FWI", "Laplacian", "Poisson", "Saturation", "Upwlap", "Upwps"]
for j in jobs
    cd("Ops/$j")
    run(`rm -rf build`)
    mkdir("build")
    cd("build")
    run(`cmake ..`)
    run(`make -j`)
    cd(dir)
end
