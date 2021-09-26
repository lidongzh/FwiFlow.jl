using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
Random.seed!(233)

py"""
import tensorflow as tf
libEikonal = tf.load_op_library('./build/libEikonal.so')
@tf.custom_gradient
def eikonal_lekj(f, srcx, srcy, m, n, h):
    u = libEikonal.eikonal(f, srcx, srcy, m, n, h)
    def grad(du):
        return libEikonal.eikonal_grad(du, u, f, srcx, srcy, m, n, h)
    return u, grad
"""
eikonal_ = py"eikonal_lekj"

function eikonal(f::Union{Array{Float64}, PyObject},
    srcx::Int64,srcy::Int64,h::Float64)
    n_, m_ = size(f) # m width, n depth 
    n = n_-1
    m = m_-1
    # eikonal_ = load_op_and_grad("$(@__DIR__)/build/libEikonal","eikonal")
    # f,srcx,srcy,m,n,h = convert_to_tensor([f,srcx,srcy,m,n,h], [Float64,Int64,Int64,Int64,Int64,Float64])
    f = tf.cast(f, dtype=tf.float64)
    srcx = tf.cast(srcx, dtype=tf.int64)
    srcy = tf.cast(srcy, dtype=tf.int64)
    m = tf.cast(m, dtype=tf.int64)
    n = tf.cast(n, dtype=tf.int64)
    h = tf.cast(h, dtype=tf.float64)
    f = tf.reshape(f, (-1,))
    u = eikonal_(f,srcx,srcy,m,n,h)
    u.set_shape((length(f),))
    tf.reshape(u, (n_, m_))
end

# TODO: specify your input parameters
m = 60
n = 30
h = 0.1

f = ones(n+1, m+1)
for i = 1:m+1
    f[12:18, i] .= 10.
end
srcx = 30
srcy = 3
u = eikonal(f,srcx,srcy,h)
sess = Session(); init(sess)
@show run(sess, u)
pcolormesh(run(sess, u)|>Array)
axis("scaled")
colorbar()
gca().invert_yaxis()
# uncomment it for testing gradients
# error() 


# TODO: change your test parameter to `m`
#       in the case of `multiple=true`, you also need to specify which component you are testings
# gradient check -- v
function scalar_function(m_)
    return sum(eikonal(m_,srcx,srcy,h)^2)
end

# TODO: change `m_` and `v_` to appropriate values
m_ = constant(rand((m+1),(n+1)))
v_ = rand((m+1),(n+1))
y_ = scalar_function(m_)
dy_ = gradients(y_, m_)

ms_ = Array{Any}(undef, 5)
ys_ = Array{Any}(undef, 5)
s_ = Array{Any}(undef, 5)
w_ = Array{Any}(undef, 5)
gs_ =  @. 0.1 / 10^(1:5)

for i = 1:5
    g_ = gs_[i]
    ms_[i] = m_ + g_*v_
    ys_[i] = scalar_function(ms_[i])
    s_[i] = ys_[i] - y_
    w_[i] = s_[i] - g_*sum(v_.*dy_)
end

sess = Session(); init(sess)
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

