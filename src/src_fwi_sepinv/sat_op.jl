using PyTensorFlow
using PyCall
using LinearAlgebra
using PyPlot
using Random
Random.seed!(233)

if Sys.islinux()
py"""
import tensorflow as tf
libSatOp = tf.load_op_library('../Ops/Saturation/build/libSatOp.so')
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
libSatOp = tf.load_op_library('../Ops/Saturation/build/libSatOp.dylib')
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
libSatOp = tf.load_op_library('../Ops/Saturation/build/libSatOp.dll')
@tf.custom_gradient
def sat_op(s0,pt,permi,poro,qw,qo,muw,muo,sref,dt,h):
    sat = libSatOp.sat_op(s0,pt,permi,poro,qw,qo,muw,muo,sref,dt,h)
    def grad(dy):
        return libSatOp.sat_op_grad(dy, sat, s0,pt,permi,poro,qw,qo,muw,muo,sref,dt,h)
    return sat, grad
"""
end

sat_op = py"sat_op"