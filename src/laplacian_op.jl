using PyTensorFlow
using PyCall
using LinearAlgebra
using PyPlot
using Random
Random.seed!(233)

if Sys.islinux()
py"""
import tensorflow as tf
libLaplacian = tf.load_op_library('../Ops/Laplacian/build/libLaplacian.so')
@tf.custom_gradient
def laplacian_op(coef,func,h,rhograv):
    p = libLaplacian.laplacian(coef,func,h,rhograv)
    def grad(dy):
        return libLaplacian.laplacian_grad(dy, coef, func, h, rhograv)
    return p, grad
"""
elseif Sys.isapple()
py"""
import tensorflow as tf
libPoissonOp = tf.load_op_library('../Ops/Laplacian/build/libPoissonOp.dylib')
@tf.custom_gradient
def laplacian_op(coef,func,h,rhograv):
    p = libLaplacian.laplacian(coef,func,h,rhograv)
    def grad(dy):
        return libLaplacian.laplacian_grad(dy, coef, func, h, rhograv)
    return p, grad
"""
elseif Sys.iswindows()
py"""
import tensorflow as tf
libPoissonOp = tf.load_op_library('../Ops/Laplacian/build/libPoissonOp.dll')
@tf.custom_gradient
def laplacian_op(coef,func,h,rhograv):
    p = libLaplacian.laplacian(coef,func,h,rhograv)
    def grad(dy):
        return libLaplacian.laplacian_grad(dy, coef, func, h, rhograv)
    return p, grad
"""
end

laplacian_op = py"laplacian_op"