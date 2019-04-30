using PyTensorFlow
using PyCall
using LinearAlgebra
using PyPlot
using Random
Random.seed!(233)
include("fwi_util.jl")
np = pyimport("numpy")

# argsparse.jl
# ENV["CUDA_VISIBLE_DEVICES"] = 1
# ENV["PARAMDIR"] = "Src/params/"
# config = tf.ConfigProto(device_count = Dict("GPU"=>0))

nz = 100
nx = 100
dz = 20
dx = 20
nSteps = 2001
dt = 0.0025
f0 = 4.5
filter_para = [0, 0.1, 100.0, 200.0]
nPml = 32
isAc = true
nPad = 0
x_src = collect(5:10:100-nPml)
z_src = 2ones(Int64, size(x_src))
# x_rec = collect(5:250-nPml)
# z_rec = 2ones(Int64, size(x_rec))

z = (1:nz-33)|>collect
x = (1:nx-33)|>collect
x_rec, z_rec = np.meshgrid(x, z)
x_rec = x_rec[:]
z_rec = z_rec[:]

para_fname = "./para_file.json"
survey_fname = "./survey_file.json"
data_dir_name = "./Data"
paraGen(nz, nx, dz, dx, nSteps, dt, f0, nPml, nPad, filter_para, isAc, para_fname, survey_fname, data_dir_name)
surveyGen(z_src, x_src, z_rec, x_rec, survey_fname)

# cp = constant(3000ones(nz, nx))
cp = (1. .+ 0.1*rand(nz, nx)) .* 3000
cs = zeros(nz, nx)
den = 1000.0 .* ones(nz, nx)

tf_cp = constant(cp)
tf_cs = constant(cs)
tf_den = constant(den)

src = Matrix{Float64}(undef, 1, 2001)
src[1,:] = Float64.(reinterpret(Float32, read("./Src/params/ricker_10Hz.bin")))
tf_stf = constant(repeat(src, outer=30))
tf_para_fname = tf.strings.join([para_fname])
tf_gpu_id0 = constant(0, dtype=Int32)
tf_gpu_id1 = constant(1, dtype=Int32)
tf_shot_ids0 = constant(collect(Int32, 1:length(x_src)), dtype=Int32)
# tf_shot_ids1 = constant(collect(Int32, 13:25), dtype=Int32)
# shot_ids = constant(zeros(1,1), dtype=Int32)

res1 = fwi_obs_op(tf_cp, tf_cs, tf_den, tf_stf, tf_gpu_id0, tf_shot_ids0, tf_para_fname)
# res2 = fwi_obs_op(tf_cp, tf_cs, tf_den, tf_stf, tf_gpu_id1, tf_shot_ids1, tf_para_fname)

sess=Session();init(sess);
@time run(sess, res1)
# error("")

# function obj()
#     res = 0.0
#     for i = 1:29
#         gpu_id = mod(i, 2)
#         res += fwi_op(tf_cp, tf_cs, tf_den, tf_stf, constant(gpu_id, dtype=Int32), constant([i], dtype=Int32), tf_para_fname)
#     end
#     return res
# end
# J = obj()

# J1 = fwi_op(tf_cp, tf_cs, tf_den, tf_stf, tf_gpu_id0, tf_shot_ids0, tf_para_fname)
# J2 = fwi_op(tf_cp, tf_cs, tf_den, tf_stf, tf_gpu_id1, tf_shot_ids1, tf_para_fname)
# J = J1 + J2
# # config = tf.ConfigProto()
# # config.allow_growth
# # config.intra_op_parallelism_threads = 2
# # config.inter_op_parallelism_threads = 2
# sess=Session();init(sess);
# # @time run(sess, J)
# gg = gradients(J, cp)
# grad_cp = run(sess, gg)
# imshow(grad_cp);colorbar();

# error("")

# gradient check -- v
function scalar_function(m)
    return fwi_op(m, tf_cs, tf_den, tf_stf, tf_gpu_id0, tf_shot_ids0, tf_para_fname)
end

# m_ = constant(3000ones(nz, nx))
m_ = constant(cp)
# v_ = 100. .* (1. .+ rand(Float32, 384, 134))

v0 = zeros(nz, nx)
# PLEASE!!!!!!!!!!!!!! Don't perturb in the CPML region!!!!!!!!!!!!!!!!!!!!!!!
v0[33:nz-33-1, 33:nx-33-1] .= 1.0
v_ = Float64.((1. .+ 0.1*rand(nz, nx)) .* 300 .* v0)
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
sval_ = [x[1] for x in sval_]
wval_ = [x[1] for x in wval_]
close("all")
loglog(gs_, abs.(sval_), "*-", label="finite difference")
loglog(gs_, abs.(wval_), "+-", label="automatic differentiation")
loglog(gs_, gs_.^2 * 0.5*abs(wval_[1])/gs_[1]^2, "--",label="\$\\mathcal{O}(\\gamma^2)\$")
loglog(gs_, gs_ * 0.5*abs(sval_[1])/gs_[1], "--",label="\$\\mathcal{O}(\\gamma)\$")

plt.gca().invert_xaxis()
legend()
xlabel("\$\\gamma\$")
ylabel("Error")
