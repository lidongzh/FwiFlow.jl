SRC_DIR = "$__DIR__/../src"
# install dependencies
if !isdir("$SRC_DIR/amgcl")
    download("https://github.com/ddemidov/amgcl/archive/master.zip", "$SRC_DIR/amgcl.zip")
    run(`unzip $SRC_DIR/amgcl.zip -d $SRC_DIR`)
end

