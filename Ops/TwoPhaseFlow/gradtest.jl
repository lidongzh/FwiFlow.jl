using PyTensorFlow
using PyCall
using PyPlot
using DelimitedFiles

if !(@isdefined __init)
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
    ConvectionDiffusion_module = tf.load_op_library('build/libConvectionDiffusion.so')
    ConvectionDiffusionGrad_module = tf.load_op_library('build/libConvectionDiffusionGrad.so')
    @ops.RegisterGradient("ConvectionDiffusion")
    def convection_diffusion_grad_cc(op, grad):
        return ConvectionDiffusionGrad_module.convection_diffusion_grad(grad, op.inputs[0], op.inputs[1], op.inputs[2], op.inputs[3])
    """
end
global __init = true
end

convection_diffusion = py"ConvectionDiffusion_module.convection_diffusion"

function scalar_function(m, a, b1, b2)
    v = norm(convection_diffusion(m, a, b1, b2))
    return sum(tanh(convection_diffusion(m, a, b1, b2)/v))
    # return sum(tanh(m))
end

function randzero()
    C = rand(20,20)
    C[1,:] .= 0.0
    C[end,:] .= 0.0
    C[:,1] .= 0.0
    C[:,end] .= 0.0
end

C = rand(20,20)
m = constant(C)
a = constant(C)
b1 = constant(C)
b2 = constant(C)

v = constant(rand(20,20))
y = scalar_function(a, b1,b2, m)
dy = gradients(y, m)
ms = Array{Any}(undef, 5)
ys = Array{Any}(undef, 5)
s = Array{Any}(undef, 5)
w = Array{Any}(undef, 5)
gs =  @. 1 / 10^(1:5)

for i = 1:5
    g = gs[i]
    ms[i] = m + g*v
    ys[i] = scalar_function( a,   b1, b2, ms[i])
    s[i] = ys[i] - y 
    w[i] = s[i] - g*sum(v.*dy)
end

sess = Session()
init(sess)
sval = run(sess, s)
wval = run(sess, w)
close("all")
loglog(gs, abs.(sval), "*-", label="finite difference")
loglog(gs, abs.(wval), "+-", label="automatic differentiation")
loglog(gs, gs.^2 * 0.5*abs(wval[1])/gs[1]^2, "--",label="\$\\mathcal{O}(\\gamma^2)\$")
loglog(gs, gs * 0.5*abs(sval[1])/gs[1], "--",label="\$\\mathcal{O}(\\gamma)\$")

plt[:gca]()[:invert_xaxis]()
legend()
xlabel("\$\\gamma\$")
ylabel("Error")
savefig("gradtest.png")
@pyimport matplotlib2tikz as mpl
mpl.save("./advection_diffusion_grad_check.tex")
