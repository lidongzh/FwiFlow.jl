using PyTensorFlow
using PyCall
using LinearAlgebra
using PyPlot
using Random
Random.seed!(233)

if Sys.islinux()
py"""
import tensorflow as tf
libSatOp = tf.load_op_library('build/libSatOp.so')
@tf.custom_gradient
def sat_op(s0,p,permi,poro,qw,qo,sref,dt,h):
    s = libSatOp.sat_op(s0,p,permi,poro,qw,qo,sref,dt,h)
    def grad(dy):
        return libSatOp.sat_op_grad(dy, s, s0,p,permi,poro,qw,qo,sref,dt,h)
    return s, grad
"""
elseif Sys.isapple()
py"""
import tensorflow as tf
libSatOp = tf.load_op_library('build/libSatOp.dylib')
@tf.custom_gradient
def sat_op(s0,p,permi,poro,qw,qo,sref,dt,h):
    s = libSatOp.sat_op(s0,p,permi,poro,qw,qo,sref,dt,h)
    def grad(dy):
        return libSatOp.sat_op_grad(dy, s, s0,p,permi,poro,qw,qo,sref,dt,h)
    return s, grad
"""
elseif Sys.iswindows()
py"""
import tensorflow as tf
libSatOp = tf.load_op_library('build/libSatOp.dll')
@tf.custom_gradient
def sat_op(s0,p,permi,poro,qw,qo,sref,dt,h):
    s = libSatOp.sat_op(s0,p,permi,poro,qw,qo,sref,dt,h)
    def grad(dy):
        return libSatOp.sat_op_grad(dy, s, s0,p,permi,poro,qw,qo,sref,dt,h)
    return s, grad
"""
end

sat_op = py"sat_op"

py"""
import tensorflow as tf
libUpwpsOp = tf.load_op_library('../Upwps/build/libUpwpsOp.so')
@tf.custom_gradient
def upwps_op(permi,mobi,src,funcref,h,rhograv,index):
    pres = libUpwpsOp.upwps_op(permi,mobi,src,funcref,h,rhograv,index)
    def grad(dy):
        return libUpwpsOp.upwps_op_grad(dy, pres, permi,mobi,src,funcref,h,rhograv,index)
    return pres, grad
"""
upwps_op = py"upwps_op"

py"""
import tensorflow as tf
libUpwlapOp = tf.load_op_library('../Upwlap/build/libUpwlapOp.so')
@tf.custom_gradient
def upwlap_op(perm,mobi,func,h,rhograv):
    out = libUpwlapOp.upwlap_op(perm,mobi,func,h,rhograv)
    def grad(dy):
        return libUpwlapOp.upwlap_op_grad(dy, out, perm,mobi,func,h,rhograv)
    return out, grad
"""
upwlap_op = py"upwlap_op"

function ave_normal(quantity, m, n)
    aa = sum(quantity)
    return aa/(m*n)
end


# TODO: 
nz=20
nx=30
sw = constant(zeros(nz, nx))
swref = constant(zeros(nz,nx) .+ 0.1)
K = constant(ones(nz, nx))
ϕ = constant(0.25 .* ones(nz, nx))
q1 = zeros(nz,nx)
q2 = zeros(nz,nx)
q1[10,5] =1.
q2[10,25] = -1.
qw = constant(q1)
qo = constant(q2)
dt = constant(1.0)
h = constant(1.0)

# function step(sw)
#     λw = sw.*sw
#     λo = (1-sw).*(1-sw)
#     λ = λw + λo
#     f = λw/λ
#     q = qw + qo + λw/(λo+1e-16).*qo

#     # Θ = laplacian_op(K.*λo, potential_c, h, constant(0.0))
#     Θ = upwlap_op(K, λo, constant(zeros(nz,nx)), h, constant(0.0))

#     load_normal = (Θ+q) - ave_normal(Θ+q, nz, nx)

#     p = upwps_op(K, λ, load_normal, constant(zeros(nz,nx)), h, constant(0.0), constant(2))
#     sw = sat_op(sw,p,K,ϕ,qw,qo,dt,h)
# end

# NT=20
# function evolve(sw, NT)
#     # qw_arr = constant(qw) # qw: NT x m x n array
#     # qo_arr = constant(qo)
#     tf_sw = TensorArray(NT+1)
#     function condition(i, ta)
#         tf.less(i, NT+1)
#     end
#     function body(i, tf_sw)
#         sw_local = step(read(tf_sw, i))
#         i+1, write(tf_sw, i+1, sw_local)
#     end
#     tf_sw = write(tf_sw, 1, sw)
#     i = constant(1, dtype=Int32)
#     _, out = while_loop(condition, body, [i;tf_sw])
#     read(out, NT+1)
# end

# u = evolve(sw, NT)

λw = sw.*sw
λo = (1-sw).*(1-sw)
λ = λw + λo
f = λw/λ
q = qw + qo + λw/(λo+1e-16).*qo

# Θ = laplacian_op(K.*λo, potential_c, h, constant(0.0))
Θ = upwlap_op(K, λo, constant(zeros(nz,nx)), h, constant(0.0))

load_normal = (Θ+q) - ave_normal(Θ+q, nz, nx)

p = upwps_op(K, λ, load_normal, constant(zeros(nz,nx)), h, constant(0.0), constant(2))
s = sat_op(sw,p,K,ϕ,qw,qo,swref,dt,h)

sess = Session()
init(sess)
S=run(sess, s)
imshow(S);colorbar();
# J = tf.nn.l2_loss(s)
# tf_grad_K = gradients(J, K)
# grad_K = run(sess, tf_grad_K)
error("")
# TODO: 


# gradient check -- v
function scalar_function(m)
    return sum(tanh(sat_op(s0,p,permi,poro,qw,qo,dt,h)))
end

m_ = constant(rand(10,20))
v_ = rand(10,20)
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
