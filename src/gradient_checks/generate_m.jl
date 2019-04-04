include("ops.jl")

m = 134
n = 384
A = zeros(m, n)
B1 = zeros(m, n)
B2 = zeros(m, n)
u0 = zeros(m, n)

# for i = 1:m 
#     for j = 1:n 
#         A[i,j] = 1.0 + 0.1*exp(-0.001*((i-60)^2))
#         B1[i,j] = 1.0 + 0.1*sin(2π/m*i)
#         B2[i,j] = 1.0 + 0.1*cos(2π/n*j)
#         u0[i,j] = exp(-0.001*((j-200)^2+(i-60)^2))
#     end
# end

for i = 1:m 
    for j = 1:n 
        # A[i,j] = 1.0 + 0.1*exp(-0.001*((i-60)^2))
        # B1[i,j] = 1.0 + 0.1*sin(2π/m*i)
        # B2[i,j] = 1.0 + 0.1*cos(2π/n*j)
        # u0[i,j] = exp(-0.001*((j-200)^2+(i-60)^2))
        A[i,j] = ((i/m)^2+(j/n)^2)*0.2+1.0
        B1[i,j] = (2.0-(i/m)^2)/2.0
        B2[i,j] = (2.0-(j/n)^2)/2.0
    end
end
# u0[50:80, 50:80] .= 1.0
# u0[50:80, 200:230] .= 1.0
for i = 1:m
    for j = 1:n
        u0[i,j] = 100 * sin(2π*(i-1)/(m-1)) * sin(2π*(j-1)/(n-1))
    end
end


u = Variable(u0)
a = 100*Variable(ones(m,n))
b1 = 1.0*Variable(ones(m, n))
b2 = -10.0*Variable(ones(m,n))

# # DL 02/25/2019 testing random media
# u = constant(500*(0.1.+rand(134,384)))

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
