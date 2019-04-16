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
def poisson_op(coef,g,h,rhograv,index):
    p = libPoissonOp.poisson_op(coef,g,h,rhograv,index)
    def grad(dy):
        return libPoissonOp.poisson_op_grad(dy, p, coef, g, h, rhograv, index)
    return p, grad
"""
elseif Sys.isapple()
py"""
import tensorflow as tf
libPoissonOp = tf.load_op_library('../Ops/Poisson/build/libPoissonOp.dylib')
@tf.custom_gradient
def poisson_op(coef,g,h,rhograv,index):
    p = libPoissonOp.poisson_op(coef,g,h,rhograv,index)
    def grad(dy):
        return libPoissonOp.poisson_op_grad(dy, p, coef, g, h, rhograv, index)
    return p, grad
"""
elseif Sys.iswindows()
py"""
import tensorflow as tf
libPoissonOp = tf.load_op_library('../Ops/Poisson/build/libPoissonOp.dll')
@tf.custom_gradient
def poisson_op(coef,g,h,rhograv,index):
    p = libPoissonOp.poisson_op(coef,g,h,rhograv,index)
    def grad(dy):
        return libPoissonOp.poisson_op_grad(dy, p, coef, g, h, rhograv, index)
    return p, grad
"""
end

poisson_op = py"poisson_op"





if Sys.islinux()
py"""
import tensorflow as tf
libUpwpsOp = tf.load_op_library('../Ops/Upwps/build/libUpwpsOp.so')
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
libUpwpsOp = tf.load_op_library('../Ops/Upwps/build/libUpwpsOp.dylib')
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
libUpwpsOp = tf.load_op_library('../Ops/Upwps/build/libUpwpsOp.dll')
@tf.custom_gradient
def upwps_op(permi,mobi,src,funcref,h,rhograv,index):
    pres = libUpwpsOp.upwps_op(permi,mobi,src,funcref,h,rhograv,index)
    def grad(dy):
        return libUpwpsOp.upwps_op_grad(dy, pres, permi,mobi,src,funcref,h,rhograv,index)
    return pres, grad
"""
end

upwps_op = py"upwps_op"

