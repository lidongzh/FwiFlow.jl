# Tutorial 

## FWI

We consider a standard FWI problem. First of all, we load necessary packages
```julia
using FwiFlow
using PyCall
using LinearAlgebra
using PyPlot
using Random
np = pyimport("numpy")
```

We specify several parameters
```julia
nz = 134
nx = 384
dz = 24. # meters
dx = 24.
nSteps = 2001
dt = 0.0025
f0 = 4.5
filter_para = [0, 0.1, 100.0, 200.0]
isAc = true
nPml = 32
nPad = 32 - mod((nz+2*nPml), 32)
nz_pad = nz + 2*nPml + nPad
nx_pad = nx + 2*nPml
```

We will use the reflection type of problem: sources on one side and receivers on the other side. 
```julia
x_src = collect(4:8:384)
z_src = 2ones(Int64, size(x_src))
x_rec = collect(3:381)
z_rec = 2ones(Int64, size(x_rec))
```

Now we generate the parameter files 
```julia
para_fname = "./para_file.json"
survey_fname = "./survey_file.json"
data_dir_name = "./Data"
paraGen(nz_pad, nx_pad, dz, dx, nSteps, dt, f0, nPml, nPad, filter_para, isAc, para_fname, survey_fname, data_dir_name)
surveyGen(z_src, x_src, z_rec, x_rec, survey_fname)
```

We load a true model from the file `Model_Cp_true.bin` and specify all necessary parameters to run on 2 GPUs
```julia
tf_cp = constant(reshape(reinterpret(Float32,read("Mar_models/Model_Cp_true.bin")),(nz_pad, nx_pad)), dtype=Float64)
cs = zeros(nz_pad, nx_pad)
den = 1000.0 .* ones(nz_pad, nx_pad)
cp_pad_value = 3000.0

tf_cs = constant(cs)
tf_den = constant(den)

src = Matrix{Float64}(undef, 1, 2001)
src[1,:] = Float64.(reinterpret(Float32, read("../Ops/FWI/Src/params/Mar_source_2001.bin")))
tf_stf = constant(repeat(src, outer=length(z_src)))

tf_gpu_id0 = constant(0, dtype=Int32)
tf_gpu_id1 = constant(1, dtype=Int32)
nGpus = 2
tf_gpu_id_array = constant(collect(0:nGpus-1), dtype=Int32)
tf_shot_ids0 = constant(collect(Int32, 0:length(x_src)-1), dtype=Int32)
shot_id_points = Int32.(trunc.(collect(LinRange(0, length(z_src)-1, nGpus+1))))
```

Now it's ready to generate the observation data. 
```julia
res = fwi_obs_op(tf_cp, tf_cs, tf_den, tf_stf, tf_gpu_id0, tf_shot_ids0, para_fname)
sess = Session(); init(sess);
run(sess, res)
```

We can see some files are generated in the folder XXXX. We now consider inversion: `cp` is not known. The idea is to make `cp` as an independent variable to be updated; this can be done by specifying `cp` with `Variable`
```julia
cp_init = reshape(reinterpret(Float32,read("Mar_models/Model_Cp_init_1D.bin")),(nz_pad, nx_pad))
tf_cp_inv = Variable(cp_init, dtype=Float64)

Mask = ones(nz_pad, nx_pad)
Mask[nPml+1:nPml+10,:] .= 0.0
tf_cp_inv_msk = tf_cp_inv .* constant(Mask) + constant(cp_init[1,1] .* (1. .- Mask))
```
We have used the mask trick to make the computation numerically stable. 

Now we can form the loss function with [`fwi_op`](@ref) and optimizes it with ADCME
```julia
loss = fwi_op(tf_cp_inv_msk, tf_cs, tf_den, tf_stf, tf_gpu_id_array[1], tf_shot_ids0, para_fname)
sess = Session(); init(sess)
BFGS!(sess, loss)
```