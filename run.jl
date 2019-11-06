dir = @__DIR__

cd("Ops/FWI/Src")
run(`. compile.sh`)
cd("../")
run(`. compile.sh`)
cd(dir)
jobs = ["FWI", "Laplacian", "Poisson", "Saturation", "Upwlap", "Upwps"]
for j in jobs
    cd("Ops/$j")
    run(`rm -rf build`)
    mkdir("build")
    cd("build")
    run(`cmake ..`)
    run(`make -j`)
    cd(dir)
end
