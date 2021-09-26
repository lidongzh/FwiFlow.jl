using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
Random.seed!(233)

if Sys.islinux()
py"""
import tensorflow as tf
libUpwpsOp = tf.load_op_library('build/libUpwpsOp.so')
@tf.custom_gradient
def upwps_op(permi,mobi,src,funcref,h,rhograv,index):
    pres = libUpwpsOp.upwps_op(permi,mobi,src,funcref,h,rhograv,index)
    def grad(dy):
        return libUpwpsOp.upwps_op_grad(dy, pres, permi,mobi,src,funcref,h,rhograv,index)
    return pres, grad
"""
elseif Sys.isapple()
py"""
import tensorflow as tf
libUpwpsOp = tf.load_op_library('build/libUpwpsOp.dylib')
@tf.custom_gradient
def upwps_op(permi,mobi,src,funcref,h,rhograv,index):
    pres = libUpwpsOp.upwps_op(permi,mobi,src,funcref,h,rhograv,index)
    def grad(dy):
        return libUpwpsOp.upwps_op_grad(dy, pres, permi,mobi,src,funcref,h,rhograv,index)
    return pres, grad
"""
elseif Sys.iswindows()
py"""
import tensorflow as tf
libUpwpsOp = tf.load_op_library('build/libUpwpsOp.dll')
@tf.custom_gradient
def upwps_op(permi,mobi,src,funcref,h,rhograv,index):
    pres = libUpwpsOp.upwps_op(permi,mobi,src,funcref,h,rhograv,index)
    def grad(dy):
        return libUpwpsOp.upwps_op_grad(dy, pres, permi,mobi,src,funcref,h,rhograv,index)
    return pres, grad
"""
end

upwps_op = py"upwps_op"

# gradient check -- v
h = 20.0
rho = 1000.0
G = 9.8
len_z = 16*20
len_x = 32*20
nz = Int(len_z/h + 1)
nx = Int(len_x/h + 1)
tf_h=constant(1.0)
coef = zeros(nz, nx)
ones_cons = ones(nz, nx)
rhs  = zeros(nz, nx)
for i = 1:nz
    for j = 1:nx
        rhs[i,j] = -sin(2*pi/len_z*(i-1)*h) * sin(2*pi/len_x*(j-1)*h)
        coef[i,j] = 1.0 - cos(2*pi/len_z*(i-1)*h) * sin(2*pi/len_x*(j-1)*h) * len_z / (2*pi*rho*G)

        # rhs[i,j] = 2.0*(i-1)*h*exp(-(((i-1)*h)^2) -(((j-1)*h)^2)) * rho * G
        # coef[i,j] = 1.0 + exp(-(((i-1)*h)^2) -(((j-1)*h)^2))
    end
end
# coef = 1.0 .+ rand(nz, nx)
# rhs = rand(nz, nx)

tf_coef = constant(coef)
tf_rhs = constant(rhs)
tf_funcref = constant(rand(nz, nx))
tf_ones = constant(ones_cons)
function scalar_function(m)
    # return sum(tanh(upwps_op(tf_coef, tf_ones, m, tf_funcref, tf_h, constant(rho*G), constant(0))))
    # return sum(tanh(upwps_op(m, tf_ones, tf_rhs, tf_ones, tf_h, constant(rho*G), constant(0))))
    return sum(tanh(upwps_op(tf_ones, m, tf_rhs, tf_ones, tf_h, constant(rho*G), constant(0))))
end

# m_ = tf_rhs
m_  = tf_coef
v_ = 0.01*rand(nz, nx)
y_ = scalar_function(m_)
dy_ = gradients(y_, m_)
ms_ = Array{Any}(undef, 5)
ys_ = Array{Any}(undef, 5)
s_ = Array{Any}(undef, 5)
w_ = Array{Any}(undef, 5)
gs_ =  @. 1 / 10^(1:5)

for i = 1:5
    g_ = gs_[i]
    ms_[i] = m_ + g_*v_
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
