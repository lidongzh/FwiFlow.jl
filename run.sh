set -x
rm -rf src/data/*
rm -rf Ops/AdvectionDiffusion/data/*
rm -rf Ops/FWI/CUFD/Phase*
cd src
julia generate_m.jl
cd ..
cd Ops/FWI/ops/
python main_calc_obs_data.py 
cd ../../../src 
julia learn_m.jl
