push!(LOAD_PATH, "@stdlib")
import Pkg; Pkg.add("Conda"); 
using Conda
using ADCME

@info "Install Boost"
pkgs = Conda._installed_packages()
if !("boost" in pkgs)
    Conda.add("boost", channel="anaconda")
end

@info "Install AMGCL"
UNZIP = joinpath(ADCME.BINDIR, "unzip")
if !isdir("$(@__DIR__)/amgcl")
    download("https://github.com/ddemidov/amgcl/archive/master.zip", "$(@__DIR__)/amgcl.zip")
    run(`$UNZIP -o $(@__DIR__)/amgcl.zip -d $(@__DIR__)`)
    mv("$(@__DIR__)/amgcl-master","$(@__DIR__)/amgcl", force=true)
    rm("$(@__DIR__)/amgcl.zip")
end

@info "Build Custom Operators"
rm("CustomOps/build", force=true, recursive=true)
mkdir("CustomOps/build")
cd("CustomOps/build")
ADCME.cmake()
ADCME.make()



