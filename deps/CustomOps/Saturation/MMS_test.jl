using PyTensorFlow
using PyCall
using LinearAlgebra
using PyPlot
using Random
Random.seed!(233)

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

sat_op = py"sat_op"

len_z = 1.0 * 2pi
len_x = 1.0 * 2pi

nScale = 5
tf_permi = Array{Any}(undef, nScale)
tf_poro = Array{Any}(undef, nScale)
tf_h = Array{Any}(undef, nScale)
tf_s = Array{Any}(undef, nScale)
s_true_array = Array{Any}(undef, nScale)
s_inv_array = Array{Any}(undef, nScale)

h_array = @. 1 / (2pi)^(1:nScale)

dt = 0.00001

for iScale = 1:nScale
    h = h_array[iScale]
    nz = Int64(trunc(len_z/h + 1))
    nx = Int64(trunc(len_x/h + 1))
    pt = zeros(nz,nx)
    s_true = zeros(nz, nx)
    q = zeros(nz,nx)
    poro = zeros(nz,nx)
    for i = 1:nz           
      for j = 1:nx
        x1 = len_z*(i-1)*h
        x2 = len_x*(j-1)*h
        s_true[i,j] = dt * sin(x1)*sin(x2)
        pt[i,j] = cos(x1)*cos(x2)
        poro[i,j] = sin(x1)^2 * sin(x2)^2
        q[i,j] = sin(x1)*sin(x2) - (-3.0 * sin(x1)*cos(x1)*sin(x2)^2*cos(x2) - 3.0 * sin(x1)^2*cos(x1)*sin(x2)*cos(x2))
      end
    end
    s_true_array[iScale] = s_true
    tf_permi[iScale] = constant(ones(nz,nx))
    tf_poro[iScale] = constant(poro)
    tf_pt = constant(pt)
    tf_q = constant(q)
    tf_s0 = constant(zeros(nz,nx))
    tf_h[iScale] = constant(h)
    tf_dt = constant(dt)
    tf_ones = constant(ones(nz, nx))
    tf_s[iScale] = sat_op(tf_s0, tf_pt, tf_permi[iScale], tf_poro[iScale], tf_q, tf_q,
        constant(1.0),constant(1.0),tf_s0,tf_dt,tf_h[iScale])
end

sess = Session()
init(sess)
s_inv_array = run(sess, tf_s)


function l2_error(s_true, s_inv, iScale)
    l2_error = 0.0
    l2_norm = 0.0
    h = h_array[iScale]
    nz = size(s_true)[1]
    nx = size(s_true)[2]
    for i = 1:nz
        for j = 1:nx
            l2_error += (s_true[i,j]-s_inv[i,j])^2 * h^2
            l2_norm += (s_true[i,j])^2 * h^2
        end
    end
    return sqrt(l2_error)/sqrt(l2_norm)
end

Error_array = Array{Any}(undef, nScale)
for iScale = 1:nScale
    Error_array[iScale] = l2_error(s_true_array[iScale], s_inv_array[iScale], iScale)
end

loglog(h_array, Error_array, "*-", label="MMS convergence")
loglog(h_array, h_array.^2 * 0.5*Error_array[1]/h_array[1]^2, "--",label="\$\\mathcal{O}(h^2)\$")
loglog(h_array, h_array * 0.5*Error_array[1]/h_array[1], "-",label="\$\\mathcal{O}(h)\$")

plt.gca().invert_xaxis()
legend()
xlabel("\$h\$")
ylabel("Error")

# imshow(p)