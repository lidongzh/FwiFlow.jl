#!/usr/bin/env python3
"""
Tests for the inner product Tensorflow operation.
.. moduleauthor:: David Stutz
"""

import unittest
import numpy as np
import os
import tensorflow as tf
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt
FWI_module = tf.load_op_library('./build/libFWI.so')
FWIGrad_module = tf.load_op_library('./build/libFWIGrad.so')
@ops.RegisterGradient("FWI")
def _FWI_grad_cc(op, grad):
    return FWIGrad_module.fwi_grad(grad, op.inputs[0], op.inputs[1])

# os.system('../CUFD/Src/CUFD ../CUFD/Phase1/Bin/')

# MAT = sio.loadmat(Model_path + 'Vtrue.mat')
# cp = MAT['Vtrue'].astype('float32')
m_init = np.fromfile('./Model_Cp.bin', dtype='float32', count=-1)
m_init = np.reshape(m_init, (224, 448), order='F')
sess = tf.Session()

tf_m_init = tf.constant(m_init, dtype=tf.float64)
res = FWI_module.fwi(tf_m_init, tf.constant(0, dtype=tf.int64))
g = tf.gradients(res, tf_m_init)
misfit = sess.run(res)
print("misfit = ", misfit)
grad = sess.run(g)
print("grad = ", grad)
# print(sess.run(g))
# Img = sess.run(g)

plt.figure(1)
plt.imshow(grad[0][10+32:,:])
plt.gca().axis('off')
plt.set_cmap('seismic')
plt.colorbar()
# plt.clim(-1, 1)
plt.show()
plt.savefig("testfig.png")
