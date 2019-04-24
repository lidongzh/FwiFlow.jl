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
def fwi_op(cp,cs,den,src,prjdir):
    res = libFwiOp.fwi_op(cp,cs,den,src,prjdir)
    def grad(dy):
        return libFwiOp.fwi_op_grad(dy, tf.constant(1.0,dtype=tf.float64), cp,cs,den,src,prjdir)
    return res, grad
"""
elseif Sys.isapple()
py"""
import tensorflow as tf
libFwiOp = tf.load_op_library('build/libFwiOp.dylib')
@tf.custom_gradient
def fwi_op(cp,cs,den):
    res = libFwiOp.fwi_op(cp,cs,den)
    def grad(dy):
        return libFwiOp.fwi_op_grad(dy, res, cp,cs,den)
    return res, grad
"""
elseif Sys.iswindows()
py"""
import tensorflow as tf
libFwiOp = tf.load_op_library('build/libFwiOp.dll')
@tf.custom_gradient
def fwi_op(cp,cs,den):
    res = libFwiOp.fwi_op(cp,cs,den)
    def grad(dy):
        return libFwiOp.fwi_op_grad(dy, res, cp,cs,den)
    return res, grad
"""
end

fwi_op = py"fwi_op"
# ENV["CUDA_VISIBLE_DEVICES"] = 1
ENV["PARAMDIR"] = "Src/params/"
config = tf.ConfigProto(device_count = Dict("GPU"=>0))
# TODO: 
nz = 134
nx = 384
cp = constant(3000ones(nz, nx))
cs = constant(zeros(nz, nx))
den = constant(1000ones(nz, nx))

# u = fwi_op(cp,cs,den)
# sess = Session()
# init(sess)
# run(sess, u)

# TODO: 
# error("")

src = constant(Float64.(reinterpret(Float32, read("Src/params/ricker_10Hz.bin"))))
prj_dir = tf.strings.join(["Src/params/"])

res=fwi_op(cp,cs,den,src,prj_dir)
gg=gradients(res,cp)
sess=Session();init(sess);
grad_cp = run(sess, gg)
imshow(grad_cp)

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
