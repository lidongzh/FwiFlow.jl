using PyTensorFlow
using PyCall
using LinearAlgebra
using PyPlot
using Random
Random.seed!(233)

if Sys.islinux()
py"""
import tensorflow as tf
libPoissonOp = tf.load_op_library('build/libPoissonOp.so')
@tf.custom_gradient
def poisson_op(coef,g,h):
    p = libPoissonOp.poisson_op(coef,g,h)
    def grad(dy):
        return libPoissonOp.poisson_op_grad(dy, p, coef,g,h)
    return p, grad
"""
elseif Sys.isapple()
py"""
import tensorflow as tf
libPoissonOp = tf.load_op_library('build/libPoissonOp.dylib')
@tf.custom_gradient
def poisson_op(coef,g,h):
    p = libPoissonOp.poisson_op(coef,g,h)
    def grad(dy):
        return libPoissonOp.poisson_op_grad(dy, p, coef,g,h)
    return p, grad
"""
elseif Sys.iswindows()
py"""
import tensorflow as tf
libPoissonOp = tf.load_op_library('build/libPoissonOp.dll')
@tf.custom_gradient
def poisson_op(coef,g,h):
    p = libPoissonOp.poisson_op(coef,g,h)
    def grad(dy):
        return libPoissonOp.poisson_op_grad(dy, p, coef,g,h)
    return p, grad
"""
end

poisson_op = py"poisson_op"

len_z = 4
len_x = 4

rho = 1000.0
G = 9.8

nScale = 5
tf_g = Array{Any}(undef, nScale)
tf_coef = Array{Any}(undef, nScale)
tf_h = Array{Any}(undef, nScale)
tf_p = Array{Any}(undef, nScale)
p_true_array = Array{Any}(undef, nScale)
p_inv_array = Array{Any}(undef, nScale)

h_array = @. 1 / 2^(1:nScale)

for iScale = 1:nScale
    h = h_array[iScale]
    nz = Int(len_z/h + 1)
    nx = Int(len_x/h + 1)
    g = zeros(nz, nx)
    coef = zeros(nz, nx)
    p_true = zeros(nz, nx)
    for i = 1:nz
        for j = 1:nx
            g[i,j] = -sin(2*pi/len_z*(i-1)*h) * sin(2*pi/len_x*(j-1)*h)
            coef[i,j] = 1.0 - cos(2*pi/len_z*(i-1)*h) * sin(2*pi/len_x*(j-1)*h) * len_z / (2*pi*rho*G)

            # g[i,j] = 2.0*(i-1)*h*exp(-(((i-1)*h)^2) -(((j-1)*h)^2)) * rho * G
            # coef[i,j] = 1.0 + exp(-(((i-1)*h)^2) -(((j-1)*h)^2))

            p_true[i,j] = rho*G*(i-1)*h
        end
    end
    p_true_array[iScale] = p_true .- mean(p_true)
    # p_true_array[iScale] = p_true
    tf_g[iScale] = constant(g)
    tf_coef[iScale] = constant(coef)
    tf_h[iScale] = constant(h)
    tf_p[iScale] = poisson_op(tf_coef[iScale], tf_g[iScale], tf_h[iScale])
end

sess = Session()
init(sess)
p_inv_array = run(sess, tf_p)
for iScale = 1:nScale
    p_inv_array[iScale] = p_inv_array[iScale] .- mean(p_inv_array[iScale])
end

function l2_error(p_true, p_inv, iScale)
    l2_error = 0.0
    l2_norm = 0.0
    h = h_array[iScale]
    nz = size(p_true)[1]
    nx = size(p_true)[2]
    for i = 1:nz
        for j = 1:nx
            l2_error += (p_true[i,j]-p_inv[i,j])^2 * h^2
            l2_norm += (p_true[i,j])^2 * h^2
        end
    end
    return sqrt(l2_error)/sqrt(l2_norm)
end

Error_array = Array{Any}(undef, nScale)
for iScale = 1:nScale
    Error_array[iScale] = l2_error(p_true_array[iScale], p_inv_array[iScale], iScale)
end

loglog(h_array, Error_array, "*-", label="MMS convergence")
loglog(h_array, h_array.^2 * 0.5*Error_array[1]/h_array[1]^2, "--",label="\$\\mathcal{O}(h^2)\$")
loglog(h_array, h_array * 0.5*Error_array[1]/h_array[1], "-",label="\$\\mathcal{O}(h)\$")

plt.gca().invert_xaxis()
legend()
xlabel("\$h\$")
ylabel("Error")

# imshow(p)