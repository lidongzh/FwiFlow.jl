# void cufd(double *res, double *grad_Cp, double *grad_Cs, double *grad_Den,
#           double *grad_stf, const double *Cp, const double *Cs,
#           const double *Den, const double *stf, int calc_id, const int gpu_id,
#           int group_size, const int *shot_ids, const string para_fname);

function obscalc(cp,cs,den,stf,shot_ids,para_fname)
    m, n = size(cp)
    res = zeros(1)
    grad_Cp = zeros(m, n)
    grad_Cs = zeros(m, n)
    grad_Den = zeros(m, n)
    grad_stf = zeros(size(stf)...)
    calc_id = Int32(2)
    gpu_id = Int32(0)
    group_size = length(shot_ids)
    ccall((:cufd, "./Src/build/libCUFD.so"), Cvoid, (Ref{Cdouble}, Ref{Cdouble}, Ref{Cdouble}, Ref{Cdouble}, 
            Ref{Cdouble}, Ref{Cdouble}, Ref{Cdouble}, 
            Ref{Cdouble}, Ref{Cdouble}, Cint, Cint, Cint, Ref{Cint}, Cstring),
        res, grad_Cp, grad_Cs, grad_Den, grad_stf, cp, cs, den, stf, calc_id, gpu_id, group_size, shot_ids,
        para_fname)
end
nz = 134
nx = 384
cp = 2500ones(nz, nx)
cs = zeros(nz, nx)
den = 1000ones(nz, nx)
shot_ids = Int32[0 1]
para_fname = "/home/lidongzh/TwoPhaseFlowFWI/Ops/FWI/Src/params/Par_file_obs_data.json"

src = Matrix{Float64}(undef, 1, 2001)
src[1,:] = Float64.(reinterpret(Float32, read("/home/lidongzh/TwoPhaseFlowFWI/Ops/FWI/Src/params/ricker_10Hz.bin")))
stf = repeat(src, outer=30)

obscalc(cp,cs,den,stf,shot_ids,para_fname)