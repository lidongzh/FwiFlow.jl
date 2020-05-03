# Full Waveform Inversion

In this example, we perform full-waveform inversion (FWI) with `FwiFlow`, using high level APIs. The core of the FWI module is wrapped in the [`FWI`](@ref) struct. 

To begin with, let us construct a `FWI` struct using the same parameters in [this article](./fwi_lowlevel.md). 

**Setting the geometry**
```julia
using FwiFlow
using ADCME
using MAT
using PyPlot

oz = 0.0 
ox = 0.0
dz_orig = 24.0 
dx_orig = 24.0 
nz_orig = 134 
nx_orig = 384 
dz = dz_orig/1.0
dx = dx_orig/1.0
nz = Int64(round((dz_orig * nz_orig) / dz));
nx = Int64(round((dx_orig * nx_orig) / dx))
dt = 0.0025
nSteps = 2000

# source and receiver locations
ind_src_x = collect(4:8:384)
ind_src_z = 2ones(Int64, size(ind_src_x))
ind_rec_x = collect(3:381)
ind_rec_z = 2ones(Int64, size(ind_rec_x))

fwi = FWI(nz, nx, dz, dx, nSteps, dt; 
    ind_src_x = ind_src_x, ind_src_z = ind_src_z, ind_rec_x = ind_rec_x, ind_rec_z = ind_rec_z)
```

We can also visualize the geometry setting using [`plot`](@ref)
```julia
plot(fwi)
```
![](./assets/../../assets/fwidomain.png)

**Preparing the data**
Next, we load the source time function and use it to do forward simulation 
```julia
stf = matread("$(DATADIR)/sourceF_4p5_2_high.mat")["sourceF"][:]
cp = Float64.(reshape(reinterpret(Float32,read("$DATADIR/Model_Cp_true.bin")), (fwi.nz_pad, fwi.nx_pad)))|>Array
cs = zeros(fwi.nz_pad, fwi.nx_pad)
ρ = 2500.0 .* ones(fwi.nz_pad, fwi.nx_pad)
shot_ids = [10]
sess = Session()
obs = compute_observation(sess, fwi, cp, cs, ρ, stf, shot_ids, gpu_id=0, is_padded=true)
```

**Run the simulation**
Finally, we perform the computation and plot `obs`
```julia
close("all")
imshow(obs[1,:,:], vmax=2000, vmin=-2000, extent=[0, fwi.nx*fwi.dx, fwi.dt*(fwi.nSteps-1), 0])
xlabel("Receiver Location (m)")
ylabel("Time (s)")
colorbar()
axis("normal")
set_cmap("gray")
```

**Inverse Modeling**
To carry out inverse modeling, i.e., estimating $c_p$, $c_s$ and $\rho$ from observations, we need [`compute_misfit`](@ref)

In this case, we make those quantities `Variables`
```julia
# initial condition
cs_init = zeros(fwi.nz, fwi.nx)
ρ_init = 2500.0 .* ones(fwi.nz, fwi.nx)
cp_init_ = Float64.(reshape(reinterpret(Float32,read("$DATADIR/Model_Cp_init_1D.bin")), (fwi.nz_pad, fwi.nx_pad)))|>Array
cp_init = cp_init_[fwi.nPml+1:fwi.nPml+fwi.nz, fwi.nPml+1:fwi.nPml+fwi.nx]

# make variables
cs_inv = Variable(cs_init)
ρ_inv = Variable(ρ_init)
cp_inv = Variable(cp_init)

# allocate GPUs
loss = constant(0.0)
nGpus = length(use_gpu())
shot_id_points = Int64.(trunc.(collect(LinRange(1, length(ind_src_z), nGpus+1))))

loss = constant(0.0)
for i = 1:nGpus
    global loss
    shot_ids = collect(shot_id_points[i]:shot_id_points[i+1])
    loss += compute_misfit(fwi, cp_inv, cs_inv, ρ_inv, 
            stf , shot_ids; gpu_id = i-1, is_padded=false,
            is_masked = false, cp_ref = cp_init, cs_ref =  cs_init, ρ_ref = ρ_init)
end

sess = Session(); init(sess)
BFGS!(sess, loss)
```
