include("ops.jl")

m = 134
n = 384
ms = Array{Any}(undef, 10)
a = Variable(ones(m, n))
b1 = Variable(ones(m, n))
b2 = Variable(ones(m, n))
ms[1] = 3000*Variable(ones(m,n))
for i = 2:10
    ms[i] = convection_diffusion(ms[i-1], a, b1, b2)
end

J = constant(0.0)
for i = 1:1
    global J
    J += fwi(ms[i], constant(i, dtype=Int64))
end

sess = Session(config=tf.ConfigProto(device_count=Dict("GPU"=> 0)))
init(sess)
run(sess, J)

#= Next step :
cd FWI/ops/
python main_calc_obs_data.py
=#