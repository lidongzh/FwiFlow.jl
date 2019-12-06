dir=$(pwd)

cd Ops/FWI/Src
source compile.sh
cd ..
source compile.sh
cd $dir

cd Ops/Poisson
source compile.sh
cd $dir

cd Ops/Laplacian
source compile.sh
cd $dir

cd Ops/Upwlap
source compile.sh
cd $dir

cd Ops/Upwps
source compile.sh
cd $dir

cd Ops/Saturation
source compile.sh
cd $dir
