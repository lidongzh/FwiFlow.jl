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

![](assets/doc_domain.png)

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
paraGen(nz, nx, dz, dx, nSteps, dt, f0, nPml, nPad, para_fname, survey_fname, data_dir_name)
surveyGen(z_src, x_src, z_rec, x_rec, survey_fname)
```
The parameter file `param_file.json` will have the following format:
```json
{"nz":134,"nx":384,"dz":24.0,"dx":24.0,"nSteps":2001,"dt":0.0025,"f0":4.5,"nPoints_pml":32,"nPad":26,"survey_fname":"./survey_file.json","data_dir_name":"./Data"}
```
and the survey file will have the following format:
```json
{"nShots":48,"shot0":{"z_src":2,"x_src":4,"nrec":379,"z_rec":[...]}, "shot1":{"z_src":2,"x_src":12,"nrec":379,"z_rec":[...]}, "shot2":...}
```

We load a true model from the file `Model_Cp_true.bin` and specify all necessary parameters to run on 2 GPUs
```julia
# tf_cp = constant(reshape(reinterpret(Float32,read("Mar_models/Model_Cp_true.bin")),(nz_pad, nx_pad)), dtype=Float64)

cp = 2000 * ones(nz_pad, nx_pad)
cs = zeros(nz_pad, nx_pad)
den = 1000.0 .* ones(nz_pad, nx_pad)
cp_pad_value = 3000.0

src = Matrix{Float64}(undef, 1, 2001)
# src[1,:] = Float64.(reinterpret(Float32, read("../Ops/FWI/Src/params/Mar_source_2001.bin")))
stf = repeat(src, outer=length(z_src))
stf = ones(size(stf)...)
shot_ids = collect(Int32, 0:length(x_src)-1)
```

Now it's ready to generate the observation data. 
```julia
res = fwi_obs_op(cp, cs, den, stf, 0, shot_ids, para_fname)
sess = Session(); init(sess);
run(sess, res)
```

We can see some files are generated in the folder XXXX. We now consider inversion: `cp` is not known. The idea is to make `cp` as an independent variable to be updated; this can be done by specifying `cp` with `Variable`
```julia
# cp_init = reshape(reinterpret(Float32,read("Mar_models/Model_Cp_init_1D.bin")),(nz_pad, nx_pad))
cp_init = cp
cp_inv = Variable(cp_init)

Mask = ones(nz_pad, nx_pad)
Mask[nPml+1:nPml+10,:] .= 0.0
cp_inv_msk = cp_inv .* constant(Mask) + constant(cp_init[1,1] .* (1. .- Mask))
```
We have used the mask trick to make the computation numerically stable. 

Now we can form the loss function with [`fwi_op`](@ref) and optimizes it with ADCME
```julia
loss = fwi_op(cp_inv_msk, cs, den, stf, 0, shot_ids, para_fname)
sess = Session(); init(sess)
BFGS!(sess, loss)
```
