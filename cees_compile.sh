dir=$(pwd)

cd Ops/FWI/Src
make all -j
cd ..
source compile.sh
cd $dir

module load compilers/gcc74
export CC=/usr/local/gcc-7.4.0/bin/gcc
export CXX=/usr/local/gcc-7.4.0/bin/g++

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

module remove compilers/gcc74
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++