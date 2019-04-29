using PyTensorFlow
using PyCall
using LinearAlgebra
using PyPlot
using Random
Random.seed!(233)

if Sys.islinux()
py"""
import tensorflow as tf
libFwiOp = tf.load_op_library('build/libFwiOp.so')
@tf.custom_gradient
def fwi_op(cp,cs,den,stf,gpu_id,shot_ids,para_fname):
    res = libFwiOp.fwi_op(cp,cs,den,stf,gpu_id,shot_ids,para_fname)
    def grad(dy):
        return libFwiOp.fwi_op_grad(dy, tf.constant(1.0,dtype=tf.float64),cp,cs,den,stf,gpu_id,shot_ids,para_fname)
    return res, grad
"""
elseif Sys.isapple()
py"""
import tensorflow as tf
libFwiOp = tf.load_op_library('build/libFwiOp.dylib')
@tf.custom_gradient
def fwi_op(cp,cs,den):
    res = libFwiOp.fwi_op(cp,cs,den,stf,gpu_id,shot_ids,para_fname)
    def grad(dy):
        return libFwiOp.fwi_op_grad(dy,tf.constant(1.0,dtype=tf.float64),cp,cs,den,stf,gpu_id,shot_ids,para_fname)
    return res, grad
"""
elseif Sys.iswindows()
py"""
import tensorflow as tf
libFwiOp = tf.load_op_library('build/libFwiOp.dll')
@tf.custom_gradient
def fwi_op(cp,cs,den):
    res = libFwiOp.fwi_op(cp,cs,den,stf,gpu_id,shot_ids,para_fname)
    def grad(dy):
        return libFwiOp.fwi_op_grad(dy,tf.constant(1.0,dtype=tf.float64),cp,cs,den,stf,gpu_id,shot_ids,para_fname)
    return res, grad
"""
end

fwi_op = py"fwi_op"
# ENV["CUDA_VISIBLE_DEVICES"] = 1
# ENV["PARAMDIR"] = "Src/params/"
config = tf.ConfigProto(device_count = Dict("GPU"=>0))
# TODO: 
nz = 134
nx = 384
cp = constant(2500ones(nz, nx))
cs = constant(zeros(nz, nx))
den = constant(1000ones(nz, nx))

# u = fwi_op(cp,cs,den)
# sess = Session()
# init(sess)
# run(sess, u)

# TODO: 
# error("")
# argsparse.jl

src = Matrix{Float64}(undef, 1, 2001)
src[1,:] = Float64.(reinterpret(Float32, read("Src/params/ricker_10Hz.bin")))
tf_stf = constant(repeat(src, outer=30))
tf_para_fname = tf.strings.join(["./Src/params/Par_file_obs_data.json"])
tf_gpu_id0 = constant(0, dtype=Int32)
tf_gpu_id1 = constant(1, dtype=Int32)
tf_shot_ids0 = constant(collect(Int32, 1:15), dtype=Int32)
tf_shot_ids1 = constant(collect(Int32, 16:29), dtype=Int32)
# shot_ids = constant(zeros(1,1), dtype=Int32)

# function obj()
#     res = 0.0
#     for i = 1:29
#         gpu_id = mod(i, 2)
#         res += fwi_op(cp, cs, den, tf_stf, constant(gpu_id, dtype=Int32), constant([i], dtype=Int32), tf_para_fname)
#     end
#     return res
# end
# J = obj()

J1 = fwi_op(cp, cs, den, tf_stf, tf_gpu_id0, tf_shot_ids0, tf_para_fname)
J2 = fwi_op(cp, cs, den, tf_stf, tf_gpu_id1, tf_shot_ids1, tf_para_fname)
J = J1 + J2
# config = tf.ConfigProto()
# config.allow_growth
# config.intra_op_parallelism_threads = 2
# config.inter_op_parallelism_threads = 2
sess=Session(config=config);init(sess);
# @time run(sess, J)
gg = gradients(J, cp)
grad_cp = run(sess, gg)
imshow(grad_cp);colorbar();

error("")

# gradient check -- v
function scalar_function(m)
    return fwi_op(m,cs,den,src,prj_dir)
end

m_ = constant(3300ones(nz, nx))
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
close("all")
loglog(gs_, abs.(sval_), "*-", label="finite difference")
loglog(gs_, abs.(wval_), "+-", label="automatic differentiation")
loglog(gs_, gs_.^2 * 0.5*abs(wval_[1])/gs_[1]^2, "--",label="\$\\mathcal{O}(\\gamma^2)\$")
loglog(gs_, gs_ * 0.5*abs(sval_[1])/gs_[1], "--",label="\$\\mathcal{O}(\\gamma)\$")

plt.gca().invert_xaxis()
legend()
xlabel("\$\\gamma\$")
ylabel("Error")
