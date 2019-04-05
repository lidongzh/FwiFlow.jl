using PyTensorFlow
using PyCall
using LinearAlgebra
using PyPlot
using Random
Random.seed!(233)

if Sys.islinux()
py"""
import tensorflow as tf
libPoissonOp = tf.load_op_library('../Ops/Poisson/build/libPoissonOp.so')
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
libPoissonOp = tf.load_op_library('../Ops/Poisson/build/libPoissonOp.dylib')
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
libPoissonOp = tf.load_op_library('../Ops/Poisson/build/libPoissonOp.dll')
@tf.custom_gradient
def poisson_op(coef,g,h):
    p = libPoissonOp.poisson_op(coef,g,h)
    def grad(dy):
        return libPoissonOp.poisson_op_grad(dy, p, coef,g,h)
    return p, grad
"""
end

poisson_op = py"poisson_op"
