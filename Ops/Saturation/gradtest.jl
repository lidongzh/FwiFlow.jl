using PyTensorFlow
using PyCall
using LinearAlgebra
using PyPlot
using Random
Random.seed!(233)

if Sys.islinux()
py"""
import tensorflow as tf
libSatOp = tf.load_op_library('./build/libSatOp.so')
@tf.custom_gradient
def sat_op(s0,pt,permi,poro,qw,qo,muw,muo,sref,dt,h):
    sat = libSatOp.sat_op(s0,pt,permi,poro,qw,qo,muw,muo,sref,dt,h)
    def grad(dy):
        return libSatOp.sat_op_grad(dy, sat, s0,pt,permi,poro,qw,qo,muw,muo,sref,dt,h)
    return sat, grad
"""
elseif Sys.isapple()
py"""
import tensorflow as tf
libSatOp = tf.load_op_library('./build/libSatOp.dylib')
@tf.custom_gradient
def sat_op(s0,pt,permi,poro,qw,qo,muw,muo,sref,dt,h):
    sat = libSatOp.sat_op(s0,pt,permi,poro,qw,qo,muw,muo,sref,dt,h)
    def grad(dy):
        return libSatOp.sat_op_grad(dy, sat, s0,pt,permi,poro,qw,qo,muw,muo,sref,dt,h)
    return sat, grad
"""
elseif Sys.iswindows()
py"""
import tensorflow as tf
libSatOp = tf.load_op_library('./build/libSatOp.dll')
@tf.custom_gradient
def sat_op(s0,pt,permi,poro,qw,qo,muw,muo,sref,dt,h):
    sat = libSatOp.sat_op(s0,pt,permi,poro,qw,qo,muw,muo,sref,dt,h)
    def grad(dy):
        return libSatOp.sat_op_grad(dy, sat, s0,pt,permi,poro,qw,qo,muw,muo,sref,dt,h)
    return sat, grad
"""
end

sat_op = py"sat_op"

if Sys.islinux()
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
elseif Sys.isapple()
    py"""
    import tensorflow as tf
    libUpwpsOp = tf.load_op_library('../Upwps/build/libUpwpsOp.dylib')
    @tf.custom_gradient
    def upwps_op(permi,mobi,src,funcref,h,rhograv,index):
        pres = libUpwpsOp.upwps_op(permi,mobi,src,funcref,h,rhograv,index)
        def grad(dy):
            return libUpwpsOp.upwps_op_grad(dy, pres, permi,mobi,src,funcref,h,rhograv,index)
        return pres, grad
    """
end
upwps_op = py"upwps_op"

if Sys.islinux()
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
elseif Sys.isapple()
py"""
import tensorflow as tf
libUpwlapOp = tf.load_op_library('../Upwlap/build/libUpwlapOp.dylib')
@tf.custom_gradient
def upwlap_op(perm,mobi,func,h,rhograv):
    out = libUpwlapOp.upwlap_op(perm,mobi,func,h,rhograv)
    def grad(dy):
        return libUpwlapOp.upwlap_op_grad(dy, out, perm,mobi,func,h,rhograv)
    return out, grad
"""
end    
upwlap_op = py"upwlap_op"

if Sys.islinux()
py"""
import tensorflow as tf
libPoissonOp = tf.load_op_library('../Poisson/build/libPoissonOp.so')
@tf.custom_gradient
def poisson_op(coef,g,h,rhograv,index):
    p = libPoissonOp.poisson_op(coef,g,h,rhograv,index)
    def grad(dy):
        return libPoissonOp.poisson_op_grad(dy, p, coef, g, h, rhograv, index)
    return p, grad
"""
elseif Sys.isapple()
py"""
import tensorflow as tf
libPoissonOp = tf.load_op_library('../Poisson/build/libPoissonOp.so')
@tf.custom_gradient
def poisson_op(coef,g,h,rhograv,index):
    p = libPoissonOp.poisson_op(coef,g,h,rhograv,index)
    def grad(dy):
        return libPoissonOp.poisson_op_grad(dy, p, coef, g, h, rhograv, index)
    return p, grad
"""
end    
poisson_op = py"poisson_op"



function ave_normal(quantity, m, n)
    aa = sum(quantity)
    return aa/(m*n)
end


# TODO: 
const ALPHA = 0.006323996017182
const SRC_CONST = 5.6146
nz=20
nx=30
sw = constant(zeros(nz, nx))
swref = constant(zeros(nz,nx))
μw = constant(1.0)
μo = constant(1.0)
K = constant(100.0 .* ones(nz, nx))
ϕ = constant(0.25 .* ones(nz, nx))
dt = constant(30.0)
h = constant(100.0)
q1 = zeros(nz,nx)
q2 = zeros(nz,nx)
q1[10,5] = 1400.0 / 100.0^3 * SRC_CONST
q2[10,25] = -2200.0 /100.0^3 * SRC_CONST
qw = constant(q1)
qo = constant(q2)

λw = sw.*sw/μw
λo = (1-sw).*(1-sw)/μo
λ = λw + λo
f = λw/λ
q = qw + qo + λw/(λo+1e-16).*qo

# Θ = laplacian_op(K.*λo, potential_c, h, constant(0.0))
Θ = upwlap_op(K, λo, constant(zeros(nz,nx)), h, constant(0.0))

load_normal = (Θ+q/ALPHA) - ave_normal(Θ+q/ALPHA, nz, nx)

tf_comp_p0 = upwps_op(K, λ, load_normal, constant(zeros(nz,nx)), h, constant(0.0), constant(2))
sess = Session()
init(sess)
p0 = run(sess, tf_comp_p0)
tf_p0 = constant(p0)

# s = sat_op(sw,p0,K,ϕ,qw,qo,sw,dt,h)

# function step(sw)
#     λw = sw.*sw
#     λo = (1-sw).*(1-sw)
#     λ = λw + λo
#     f = λw/λ
#     q = qw + qo + λw/(λo+1e-16).*qo

#     # Θ = laplacian_op(K.*λo, constant(zeros(nz,nx)), h, constant(0.0))
#     Θ = upwlap_op(K, λo, constant(zeros(nz,nx)), h, constant(0.0))
#     # Θ = constant(zeros(nz,nx))

#     load_normal = (Θ+q/ALPHA) - ave_normal(Θ+q/ALPHA, nz, nx)

#     p = poisson_op(λ.*K, load_normal, h, constant(0.0), constant(0)) # potential p = pw - ρw*g*h 
#     # p = upwps_op(K, λ, load_normal, constant(zeros(nz,nx)), h, constant(0.0), constant(0))
    # sw = sat_op(sw,p,K,ϕ,qw,qo,μw,μo,sw,dt,h)
#     return sw
# end

# NT=100
# function evolve(sw, NT, qw, qo)
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

# s = evolve(sw, NT, qw, qo)



# J = tf.nn.l2_loss(s)
# tf_grad_K = gradients(J, K)
# sess = Session()
# init(sess)
# # P = run(sess,p0)

# # error("")
# S=run(sess, s)
# imshow(S);colorbar();

# error("")

# grad_K = run(sess, tf_grad_K)
# imshow(grad_K);colorbar();
# error("")
# TODO: 

# gradient check -- v
function scalar_function(m)
    # return sum(tanh(sat_op(m,tf_p0,K,ϕ,qw,qo,μw,μo,constant(zeros(nz,nx)),dt,h)))
    # return sum(tanh(sat_op(sw,m,K,ϕ,qw,qo,μw,μo,constant(zeros(nz,nx)),dt,h)))
    # return sum(tanh(sat_op(sw,tf_p0,m,ϕ,qw,qo,μw,μo,constant(zeros(nz,nx)),dt,h)))
    return sum(tanh(sat_op(sw,tf_p0,K,m,qw,qo,μw,μo,constant(zeros(nz,nx)),dt,h)))
end

# m_ = sw
# m_ = tf_p0
# m_ = K
m_ = ϕ

v_ = rand(nz,nx)
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
