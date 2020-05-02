using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
Random.seed!(233)
include("fwi_util.jl")
include("fwi_util_op.jl")
np = pyimport("numpy")

# argsparse.jl
# ENV["CUDA_VISIBLE_DEVICES"] = 1
# ENV["PARAMDIR"] = "Src/params/"
# config = tf.ConfigProto(device_count = Dict("GPU"=>0))

nz = 400
nx = 400
dz = 20
dx = 20
nSteps = 2001
dt = 0.0025
f0 = 4.5
filter_para = [0, 0.1, 100.0, 200.0]
nPml = 32
isAc = false
nPad = 0
# x_src = collect(5:10:nx-2nPml-5)
# z_src = 2ones(Int64, size(x_src))
# x_rec = collect(5:100-nPml)
# z_rec = 2ones(Int64, size(x_rec))

x_src = [200-nPml]
z_src = [200-nPml]

z = (5:10:nz-2nPml-5)|>collect
x = (5:10:nx-2nPml-5)|>collect
x_rec, z_rec = np.meshgrid(x, z)
x_rec = x_rec[:]
z_rec = z_rec[:]

# x_src = 5
# z_src = [300-nPml]
# z_rec = collect(5:1:nz-2nPml-5)
# x_rec = (nx-2nPml-100) .* ones(Int64, size(z_rec))

para_fname = "./para_file.json"
survey_fname = "./survey_file.json"
data_dir_name = "./Data"
scratch_dir_name="./Scratch"
# paraGen(nz, nx, dz, dx, nSteps, dt, f0, nPml, nPad, filter_para, isAc, para_fname, survey_fname, data_dir_name, scratch_dir_name=scratch_dir_name)
# surveyGen(z_src, x_src, z_rec, x_rec, survey_fname)
paraGen(nz, nx, dz, dx, nSteps, dt, f0, nPml, 
        nPad, para_fname, survey_fname, data_dir_name, 
        scratch_dir_name=scratch_dir_name)
surveyGen(z_src, x_src, z_rec, x_rec, survey_fname)

cp = 3000ones(nz, nx)
# cp = (1. .+ 0.1*rand(nz, nx)) .* 3000.
cs = 3000.0/sqrt(3.0) .* ones(nz,nx)
# cs = zeros(nz,nx)
den = 2000.0 .* ones(nz, nx)
function vel2moduli(cp,cs,den)
    lambda = (cp.^2 - 2.0 .* cs.^2) .* den ./ 1e6
    mu = cs.^2 .* den ./ 1e6
    return lambda, mu
end
lambda, mu = vel2moduli(cp,cs,den)

tf_lambda = constant(lambda)
tf_mu = constant(mu)
tf_den = constant(den)

# # src = Matrix{Float64}(undef, 1, 2001)
# src[1,:] = Float64.(reinterpret(Float32, read("./Src/params/ricker_10Hz.bin")))
src = sourceGene(f0, nSteps, dt)
tf_stf = constant(repeat(src, outer=length(z_src)))
tf_para_fname = tf.strings.join([para_fname])
tf_gpu_id0 = constant(0, dtype=Int32)
tf_gpu_id1 = constant(1, dtype=Int32)
tf_shot_ids0 = constant(collect(Int32, 0:length(x_src)-1), dtype=Int32)
tf_shot_ids1 = constant(collect(Int32, 13:25), dtype=Int32)

res1 = fwi_obs_op(tf_lambda, tf_mu, tf_den, tf_stf, tf_gpu_id0, tf_shot_ids0, tf_para_fname)
# res2 = fwi_obs_op(tf_cp2, tf_cs2, tf_den2, tf_stf, tf_gpu_id1, tf_shot_ids0, tf_para_fname)

sess=Session();init(sess);
@time run(sess, res1)
# error("")

# gradient check -- v
function scalar_function(m)
    # return fwi_op(m, tf_mu, tf_den, tf_stf, tf_gpu_id0, tf_shot_ids0, tf_para_fname)
    return fwi_op(tf_lambda, m, tf_den, tf_stf, tf_gpu_id0, tf_shot_ids0, tf_para_fname)
    # return fwi_op(tf_lambda, tf_mu, m, tf_stf, tf_gpu_id0, tf_shot_ids0, tf_para_fname)
    # return fwi_op(tf_lambda, tf_mu, tf_den, m, tf_gpu_id0, tf_shot_ids0, tf_para_fname)
end


# lambda2,_ = vel2moduli(3200.0, 3200.0/sqrt(3.0), den)
# m_ = constant(lambda2)

_,mu2 = vel2moduli(3200.0, 3200.0/sqrt(3.0), den)
m_ = constant(mu2)

# m_ = constant(2200ones(nz,nx))

# # src2 = circshift(src, (0,30))
# src2 = sourceGene(f0,nSteps,dt) .*1.5
# m_ = constant(repeat(src2, outer=length(z_src)))

# for forward_backward wavefield comparison
# yy = scalar_function(m_)
# gradm = gradients(yy, m_)
# sess=Session();init(sess);
# @time G = run(sess, gradm)
# imshow(G);colorbar();
# A=read("SnapGPU.bin");A=reshape(reinterpret(Float32,A),(200,200));
# B=read("SnapGPU_back.bin");B=reshape(reinterpret(Float32,B),(200,200));
# imshow(A[33:end-32,33:end-32]-B[33:end-32,33:end-32]);colorbar();
# error("")

v0 = zeros(nz, nx)
# PLEASE!!!!!!!!!!!!!! Don't perturb in the CPML region!!!!!!!!!!!!!!!!!!!!!!!
v0[nPml+5:nz-nPml-5, nPml+5:nx-nPml-5] .= 1.0
# # perturb moduli
v_ = constant(Float64.((1. .+ 0.1*rand(nz, nx)) .* 1e3 .* v0))
# # perturb density
# v_ = constant(Float64.((1. .+ 0.1*rand(nz, nx)) .* 500 .* v0))
# perturb sft

# s0 = zeros(1, nSteps)
# s0[1, 20:end-20] .= 1.0
# src_perturb = rand(1,nSteps) .* s0 *0.1
# v_ = src_perturb


y_ = scalar_function(m_)
dy_ = gradients(y_, m_)
ms_ = Array{Any}(undef, 5)
ys_ = Array{Any}(undef, 5)
s_ = Array{Any}(undef, 5)
w_ = Array{Any}(undef, 5)
gs_ =  @. 1 / 10^(1:5)

for i = 1:5
    g_ = gs_[i]
    ms_[i] = m_ + g_ * v_
    ys_[i] = scalar_function(ms_[i])
    s_[i] = ys_[i] - y_
    w_[i] = s_[i] - g_*sum(v_.*dy_)
end

sess = Session()
init(sess)
sval_ = run(sess, s_)
wval_ = run(sess, w_)
# error("")
sval_ = [x[1] for x in sval_]
wval_ = [x[1] for x in wval_]
close("all")
loglog(gs_, abs.(sval_), "*-", label="finite difference")
loglog(gs_, abs.(wval_), "+-", label="FWI gradient")
loglog(gs_, gs_.^2 * 0.5*abs(wval_[1])/gs_[1]^2, "--",label="\$\\mathcal{O}(\\gamma^2)\$")
loglog(gs_, gs_ * 0.5*abs(sval_[1])/gs_[1], "--",label="\$\\mathcal{O}(\\gamma)\$")

plt.gca().invert_xaxis()
legend()
xlabel("\$\\gamma\$")
ylabel("Error")

savefig("Convergence_test_mu.pdf", bbox_inches="tight",pad_inches = 0, dpi = 300);
