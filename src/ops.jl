using PyTensorFlow
using PyCall
using PyPlot
using DelimitedFiles
using JLD2

global sody
if Sys.isapple()
    sody = "dylib"
elseif Sys.islinux()
    sody = "so"
end
if !(@isdefined initialized)
    py"""
import tensorflow as tf
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt
FWI_module = tf.load_op_library('../Ops/FWI/ops/build/libFWI.so')
FWIGrad_module = tf.load_op_library('../Ops/FWI/ops/build/libFWIGrad.so')
@ops.RegisterGradient("FWI")
def _FWI_grad_cc(op, grad):
    return FWIGrad_module.fwi_grad(grad, op.inputs[0], op.inputs[1])
"""

py"""
ConvectionDiffusion_module = tf.load_op_library('../Ops/AdvectionDiffusion/build/libConvectionDiffusion.so')
ConvectionDiffusionGrad_module = tf.load_op_library('../Ops/AdvectionDiffusion/build/libConvectionDiffusionGrad.so')
@ops.RegisterGradient("ConvectionDiffusion")
def convection_diffusion_grad_cc(op, grad):
    return ConvectionDiffusionGrad_module.convection_diffusion_grad(grad, op.inputs[0], op.inputs[1], op.inputs[2], op.inputs[3])
    """
    global initialized = true
end
fwi = py"FWI_module.fwi"
convection_diffusion = py"ConvectionDiffusion_module.convection_diffusion"

cd("../Ops/FWI/ops")

m = 134
n = 384
NT = 5

u0_ = zeros(m-70, n-70)
for i = 1:m-70
    for j = 1:n-70
        u0_[i,j] = 100 * sin(2π*(i-1)/(m-70-1)) * sin(2π*(j-1)/(n-70-1))
    end
end
#u0_[30:50,50:250] .= 100

u0 = zeros(m, n)
u0[35:end-35-1, 35:end-35-1] = u0_
mask = zeros(m, n)
mask[35:end-35-1, 35:end-35-1] .= 1.0


# while loop 
function evolve(uh, NT)
    ta = TensorArray(NT)
    function condition(i, ta)
        tf.less(i, NT+1)
    end
    function body(i, ta)
        uh = convection_diffusion(read(ta, i-1), a, b1, b2)
        i+1, write(ta, i, uh)
    end
    ta = write(ta, 1, uh)
    i = constant(2, dtype=Int32)
    _, out = while_loop(condition, body, [i;ta])
    read(out, NT)
end

function evolve_explicit(uh, NT, a, b1, b2)
    for i = 2:NT+1
        uh = convection_diffusion(uh, a, b1, b2)
    end 
    uh
end

@pyimport numpy as np
function show_field(u0)
    xx = (0:133)*24
    yy = (0:383)*24
    X, Y = np.meshgrid(yy, xx)
    pcolormesh(X, Y ,u0, cmap="gray")
    gca()[:axes][:xaxis][:set_ticklabels]([])
    ylabel("depth (m)")
    xlabel("distance")
    plt[:gca]()[:invert_yaxis]()
end
