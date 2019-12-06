using Conda

CC = joinpath(Conda.BINDIR, "gcc")
CXX = joinpath(Conda.BINDIR, "g++")
CMAKE = joinpath(Conda.BINDIR, "cmake")
MAKE = joinpath(Conda.BINDIR, "make")


SRC_DIR = "$(@__DIR__)/../src"
CUR_DIR = @__DIR__
# install dependencies
if !isdir("$SRC_DIR/amgcl")
    download("https://github.com/ddemidov/amgcl/archive/master.zip", "$SRC_DIR/amgcl.zip")
    run(`unzip $SRC_DIR/amgcl.zip -d $SRC_DIR`)
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
        if Sys.apple() && endswith(file, ".dylib")
            flag = true
            break
        end
    end
    if flag
        @info "Library exists"
    end
    cd("build")
    run(`$CMAKE ..`)
    run(`$MAKE -j`)
    cd(CUR_DIR)
end



for name in ["Poisson", "Laplacian", "Upwlap", "Upwps", "Saturation"]
    DIR = joinpath(SRC_DIR, "Ops/$name")
    compile_op(DIR)
end


try
    run(`nvcc`)
    DIR = joinpath(SRC_DIR, "Ops/$name")
    compile_op(DIR)
else
    error("`nvcc` is not found. The core module of `FwiFlow` only has a GPU kernel.")
end