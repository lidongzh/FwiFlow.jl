using PyTensorFlow
using PyCall
using PyPlot
using DelimitedFiles
if Sys.isapple()
    py"""
    import tensorflow as tf
    from tensorflow.python.framework import ops
    ConvectionDiffusion_module = tf.load_op_library('build/libConvectionDiffusion.dylib')
    ConvectionDiffusionGrad_module = tf.load_op_library('build/libConvectionDiffusionGrad.dylib')
    @ops.RegisterGradient("ConvectionDiffusion")
    def convection_diffusion_grad_cc(op, grad):
        return ConvectionDiffusionGrad_module.convection_diffusion_grad(grad, op.inputs[0], op.inputs[1], op.inputs[2], op.inputs[3])
    """
else
    py"""
    import tensorflow as tf
    from tensorflow.python.framework import ops
    Poisson_module = tf.load_op_library('build/libPoisson.so')
    @ops.RegisterGradient("Poisson")
    """
end

poisson = py"Poisson_module.poisson"
nz = 10
nx = 15
coef = 10.0*randn(nz, nx)
g = 10.0*randn(nz, nx)
h = 10.0

p = poisson(coef, g, h)
sess = Session()
run(sess, p)



