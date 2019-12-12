push!(LOAD_PATH, "@stdlib")
import Pkg; Pkg.add("Conda"); using Conda

CC = joinpath(Conda.BINDIR, "gcc")
CXX = joinpath(Conda.BINDIR, "g++")
CMAKE = joinpath(Conda.BINDIR, "cmake")
MAKE = joinpath(Conda.BINDIR, "make")

@info "Install CONDA dependencies..."
pkgs = Conda._installed_packages()
if !("boost" in pkgs)
    Conda.add("boost", channel="anaconda")
end



SRC_DIR = "$(@__DIR__)/../src"
CUR_DIR = @__DIR__
# install dependencies
if !isdir("$SRC_DIR/amgcl")
    download("https://github.com/ddemidov/amgcl/archive/master.zip", "$SRC_DIR/amgcl.zip")
    run(`unzip -o $SRC_DIR/amgcl.zip -d $SRC_DIR`)
    run(`mv $SRC_DIR/amgcl-master $SRC_DIR/amgcl`)
    run(`rm $SRC_DIR/amgcl.zip`)
end

function compile_op(DIR)
    DIR = abspath(DIR)
    cd(DIR)
    if !isdir("build")
        mkdir("build")
    end
    flag = false
    for file in readdir("build")
        if Sys.islinux() && endswith(file, ".so")
            flag = true
            break
        end
        if Sys.isapple() && endswith(file, ".dylib")
            flag = true
            break
        end
    end
    if flag
        @info "Library exists"
        return 
    end
    cd("build")
    run(`$CMAKE ..`)
    run(`$MAKE -j`)
    cd(CUR_DIR)
end


try
    run(`nvcc --version`)
    global NVCC = true
catch
    @warn("`nvcc` is not found. The FWI module of `FwiFlow` only has a GPU kernel.")
    global NVCC = false
end

if NVCC
    try
        DIR = joinpath(SRC_DIR, "CppOps/FWI/Src")
        compile_op(DIR)
    catch
        @warn("CppOps/FWI/Src Failed.")
        global NVCC = false
    end
end

if NVCC
    try
        DIR = joinpath(SRC_DIR, "CppOps/FWI")
        compile_op(DIR)
    catch
        @warn("CppOps/FWI Failed.")
        global NVCC = false
    end
end

for name in ["Poisson", "Laplacian", "Upwlap", "Upwps", "Saturation"]
    DIR = joinpath(SRC_DIR, "CppOps/$name")
    compile_op(DIR)
end


