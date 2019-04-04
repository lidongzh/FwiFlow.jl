#!/usr/bin/env python3
"""
Tests for the inner product Tensorflow operation.
.. moduleauthor:: David Stutz

"""
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
Poisson_module = tf.load_op_library('build/libPoisson.so')
PoissonGrad_module = tf.load_op_library('build/libPoissonGrad.so')
@ops.RegisterGradient("Poisson")
def poisson_grad_cc(op, grad):
    return PoissonGrad_module.poisson_grad(grad, op.inputs[0], op.inputs[1], op.inputs[2])


sess = tf.Session()
nz = 10
nx = 15
coef = tf.constant(np.random.rand(nz, nx), dtype=tf.float64)
g = tf.constant(np.random.rand(nz, nx), dtype=tf.float64)
h = 10.0
p = Poisson_module.poisson(coef, g, h)
# g = tf.gradients(tf.reduce_sum(p), coef, g, h)
print(sess.run(p))
# print(sess.run(g))
