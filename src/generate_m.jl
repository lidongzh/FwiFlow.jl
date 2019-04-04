include("ops.jl")
using Random
Random.seed!(233)

u = Variable(u0)
n1 = 0.05*randn(m,n)
n2 = 0.05*randn(m,n)
n3 = 0.05*randn(m,n)
a = 100*Variable(ones(m,n)+n1)
b1 = 1.0*Variable(ones(m, n)+n2)
b2 = -10.0*Variable(ones(m,n)+n3)
u = u.*mask

nsrc = 5
us = Array{Any}(undef, nsrc)
us[1] = u
for i = 2:nsrc
    @show i
    us[i] = evolve(us[i-1], 5)
end

sess = Session()
init(sess)
U = run(sess, us)
for i = 1:nsrc
    @show i
    writedlm("../../AdvectionDiffusion/data/U$i.txt", U[i] .+ 3500.0)
end    
