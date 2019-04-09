using PyTensorFlow
using PyCall
using LinearAlgebra
using PyPlot
using Random
Random.seed!(233)

if Sys.islinux()
py"""
import tensorflow as tf
libUpwlapOp = tf.load_op_library('build/libUpwlapOp.so')
@tf.custom_gradient
def upwlap_op(perm,mobi,func,h,rhograv):
    out = libUpwlapOp.upwlap_op(perm,mobi,func,h,rhograv)
    def grad(dy):
        return libUpwlapOp.upwlap_op_grad(dy, out, perm,mobi,func,h,rhograv)
    return out, grad
"""
elseif Sys.isapple()
py"""
import tensorflow as tf
libUpwlapOp = tf.load_op_library('build/libUpwlapOp.dylib')
@tf.custom_gradient
def upwlap_op(perm,mobi,func,h,rhograv):
    out = libUpwlapOp.upwlap_op(perm,mobi,func,h,rhograv)
    def grad(dy):
        return libUpwlapOp.upwlap_op_grad(dy, out, perm,mobi,func,h,rhograv)
    return out, grad
"""
elseif Sys.iswindows()
py"""
import tensorflow as tf
libUpwlapOp = tf.load_op_library('build/libUpwlapOp.dll')
@tf.custom_gradient
def upwlap_op(perm,mobi,func,h,rhograv):
    out = libUpwlapOp.upwlap_op(perm,mobi,func,h,rhograv)
    def grad(dy):
        return libUpwlapOp.upwlap_op_grad(dy, out, perm,mobi,func,h,rhograv)
    return out, grad
"""
end

upwlap_op = py"upwlap_op"

# TODO: 
# u = upwlap_op(perm,mobi,func,h,rhograv)
# sess = Session()
# init(sess)
# run(sess, u)

# TODO: 
h = 1.0
rho = 1000.0
G = 0.0
len_z = 16
len_x = 32
nz = Int(len_z/h + 1)
nx = Int(len_x/h + 1)
tf_h=constant(1.0)


perm = rand(nz, nx)
mobi = rand(nz, nx)
func = rand(nz, nx)

tf_perm = constant(perm)
tf_mobi = constant(mobi)
tf_func = constant(func)



# gradient check -- v
function scalar_function(m)
    # return sum(tanh(upwlap_op(m, tf_mobi, tf_func, tf_h, constant(rho*G))))
    return sum(tanh(upwlap_op(tf_perm, m, tf_func, tf_h, constant(rho*G))))
    # return sum(tanh(upwlap_op(tf_perm, tf_mobi, m, tf_h, constant(rho*G))))
end

# m_ = constant(rand(10,20))
# m_ = tf_perm
m_ = tf_mobi
# m_ = tf_func
v_ = rand(nz, nx)
y_ = scalar_function(m_)
dy_ = gradients(y_, m_)
ms_ = Array{Any}(undef, 5)
ys_ = Array{Any}(undef, 5)
s_ = Array{Any}(undef, 5)
w_ = Array{Any}(undef, 5)
gs_ =  @. 1 / 20^(1:5)

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
