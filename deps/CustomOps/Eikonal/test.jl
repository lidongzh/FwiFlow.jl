using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
using ADCMEKit
Random.seed!(233)

reset_default_graph()
include("eikonal_op.jl")

m = 60
n = 30
h = 0.1

f = ones(n+1, m+1)
f[12:18, :] .= 10.
srcx = 5
srcy = 15

u = PyObject[]
for (k,(x,y)) in enumerate(zip(5*ones(Int64, length(1:n)), 1:n))
    push!(u,eikonal(f,x,y,h))
end

for (k,(x,y)) in enumerate( zip(1:m,5*ones(Int64, length(1:m))))
    push!(u,eikonal(f,x,y,h))
end

for (k,(x,y)) in enumerate( zip(1:m,25*ones(Int64, length(1:m))))
    push!(u,eikonal(f,x,y,h))
end


sess = Session()
uobs = run(sess, u)

F = Variable(ones(n+1, m+1))
# pl = placeholder(F0'[:])
# F = reshape(pl, n+1, m+1)

# F = Variable(ones(n+1, m+1))
u = PyObject[]
for (k,(x,y)) in enumerate(zip(5*ones(Int64, length(1:n)), 1:n))
    push!(u,eikonal(F,x,y,h))
end

for (k,(x,y)) in enumerate(zip(1:m,5*ones(Int64, length(1:m))))
    push!(u,eikonal(F,x,y,h))
end

for (k,(x,y)) in enumerate(zip(1:m,25*ones(Int64, length(1:m))))
    push!(u,eikonal(F,x,y,h))
end

# loss = sum([sum((uobs[i][5:5:end,55] - u[i][5:5:end,55])^2) for i = 1:length(u)])
loss = sum([sum((uobs[i][1:end,55] - u[i][1:end,55])^2) for i = 1:length(u)])

init(sess)
@show run(sess, loss)

# lineview(sess, F, loss, f, ones(n+1, m+1))
# gradview(sess, pl, loss, ones((n+1)* (m+1)))


# meshview(sess, pl, loss, F0'[:])

BFGS!(sess, loss, 1000,var_to_bounds=Dict(F=>(0.5,100.0)))

pcolormesh(run(sess, F))
colorbar()