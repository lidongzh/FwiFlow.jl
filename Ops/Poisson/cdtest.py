#!/usr/bin/env python3
"""
Tests for the inner product Tensorflow operation.
.. moduleauthor:: David Stutz

"""
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from matplotlib import pyplot as plt
Poisson_module = tf.load_op_library('build/libPoissonOp.so')
PoissonGrad_module = tf.load_op_library('build/libPoissonGradOp.so')
@ops.RegisterGradient("PoissonOp")
def poisson_grad_cc(op, grad):
    return PoissonGrad_module.poisson_grad(grad, op.inputs[0], op.inputs[1], op.inputs[2])


# sess = tf.Session()
# len_z = 2
# len_x = 2
# h = 1/(2**1)
# nz = int(len_z/h + 1)
# nx = int(len_x/h + 1)
# rho = 1000.0
# G = 9.8

# g_np = np.zeros((nz, nx))
# coef_np = np.zeros((nz, nx))
# p_true = np.zeros((nz, nx))

# # for i in range(nz):
# # 	for j in range(nx):
# # 		g_np[i,j] = np.sin(2*np.pi/len_z*i*h)*np.sin(2*np.pi/len_x*j*h)
# # 		coef_np[i,j] = 1.0-np.cos(2*np.pi/len_z*i*h)*np.sin(2*np.pi/len_x*j*h)*len_z/(2*np.pi*rho*G)
# # 		p_true[i,j] = rho*G*i*h

# for i in range(nz):
# 	for j in range(nx):
# 		g_np[i,j] = 2*i*h*np.exp(-(i*h)**2-(j*h)**2) * rho * G
# 		coef_np[i,j] = 1 + np.exp(-(i*h)**2-(j*h)**2)
# 		p_true[i,j] = rho*G*i*h

# p_true = p_true - np.mean(p_true)

# # coef_np = np.ones((nz, nx))
# # g_np = np.zeros((nz, nx))
# # g_np[5,5] = -1.0
# # g_np[15,15] = 1.0

# print(np.mean(g_np))

# coef = tf.constant(coef_np, dtype=tf.float64)
# # g_np = g_np - np.mean(g_np)
# g = tf.constant(g_np, dtype=tf.float64)
# p = Poisson_module.poisson_op(coef, g, h, G*rho, 1)
# # g = tf.gradients(tf.reduce_sum(p), coef, g, h)
# # print(sess.run(p))
# # print(sess.run(g))

# p_inv_np = sess.run(p)
# p_inv_np = p_inv_np - np.mean(p_inv_np)

# plt.subplot(1,2,1)
# plt.imshow(p_inv_np)
# plt.colorbar()
# plt.title('inv')

# plt.subplot(1,2,2)
# plt.imshow(p_true)
# plt.colorbar()
# plt.show()

sess = tf.Session()
h = 0.2
nz = 5
nx = 5
rho = 1000.0
G = 0.0

g_np = np.zeros((nz, nx))
g_np[0,0]=25.0
g_np[4,4]=-25.0
coef_np = np.ones((nz, nx))

print(np.mean(g_np))

coef = tf.constant(coef_np, dtype=tf.float64)
# g_np = g_np - np.mean(g_np)
g = tf.constant(g_np, dtype=tf.float64)
p = Poisson_module.poisson_op(coef, g, h, G*rho, 1)
# g = tf.gradients(tf.reduce_sum(p), coef, g, h)
# print(sess.run(p))
# print(sess.run(g))

p_inv_np = sess.run(p)
# p_inv_np = p_inv_np - np.mean(p_inv_np)

plt.imshow(p_inv_np)
plt.colorbar()
plt.title('inv')
plt.show()