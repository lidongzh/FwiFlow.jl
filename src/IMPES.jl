using PyPlot
using LinearAlgebra
using PyTensorFlow
using PyCall
np = pyimport("numpy")
include("poisson_op.jl")

# Solver parameters
# m = 20
# n = 20
# h = 1.0 # 50 meters
# T = 25 # 100 days
# NT = 25
# Δt = 0.7/NT 
# x = (1:m)*h|>collect
# z = (1:n)*h|>collect
# X, Z = np.meshgrid(x, z)
m = 20
n = 20
h = 20.0
# T = 100 # 100 days
# NT = 100
# Δt = T/(NT+1)
NT = 20
Δt = 1
# T = NT() # 100 days
z = (1:m)*h|>collect
x = (1:n)*h|>collect
X, Z = np.meshgrid(x, z)

function Krw(Sw)
    return Sw ^ 2
end

function Kro(So)
    return So ^2
end
ρw = constant(996.9571)
ρo = constant(640.7385)
μw = constant(1e-3)
μo = constant(1e-3)
K = Variable(9.8692e-14*ones(m,n))
g = constant(0.0)
ϕ = Variable(0.25*ones(m,n))

function geto(o::Union{Array,PyObject}, i::Int64, j::Int64)
    if i==-1
        ii = 1:m-2
    elseif i==0
        ii = 2:m-1
    else
        ii = 3:m
    end
    if j==-1
        jj = 1:n-2
    elseif j==0
        jj = 2:n-1
    else
        jj = 3:n
    end
    return o[ii,jj]
end

function G(f, p)
    f1 = (geto(f, 0, 0) + geto(f, 1, 0))/2
    f2 = (geto(f, -1, 0) + geto(f, 0, 0))/2
    f3 = (geto(f,0,1) + geto(f,0,0))/2
    f4 = (geto(f,0,-1) + geto(f,0,0))/2
    rhs = -f1*(geto(p,1,0)-geto(p,0,0)) +
            f2*(geto(p,0,0)-geto(p,-1,0)) -
            f3*(geto(p,0,1)-geto(p,0,0)) +
            f4*(geto(p,0,0)-geto(p,0,-1))
    local q
    if isa(rhs, Array)
        q = zeros(m, n)
        q[2:m-1, 2:n-1] = rhs/h^2
    else
        q = constant(zeros(m, n))
        q = scatter_add(q, 2:m-1, 2:n-1, rhs/h^2)
    end
    q
end


function ave_normal(quantity, m, n)
    aa = 0.0
    for i = 1:m
        for j = 1:n
            aa = aa + quantity[i,j]
        end
    end
    return aa/(m*n)
end


# variables : sw, u, v, p
# (time dependent) parameters: qw, qo, ϕ
function onestep(sw, qw, qo)
    # step 1: update p
    λw = Krw(sw)/μw
    λo = Kro(1-sw)/μo
    λ = λw + λo
    f = λw/λ
    q = qw + qo
    Θ = G(K.*(λw*ρw+λo*ρo)*g, Z)
    load_normal = (Θ+q) - ave_normal(Θ+q,m,n)
    p = poisson_op(λ.*K, load_normal, constant(h))

    # step 2: update u, v
    rhs_u = -geto(K, 0, 0).*geto(λ, 0, 0)/h.*(geto(p, 1, 0) - geto(p, 0, 0))
    rhs_v = -geto(K, 0, 0).*geto(λ, 0, 0)/h.*(geto(p, 0, 1) - geto(p, 0, 0)) +
            geto(K, 0, 0).*geto(λw*ρw+λo*ρo, 0, 0)*g
    u = constant(zeros(m, n))
    v = constant(zeros(m, n))
    u = scatter_add(u, 2:m-1, 2:n-1, rhs_u)
    v = scatter_add(v, 2:m-1, 2:n-1, rhs_v)

    # step 3: update sw
    rhs = geto(qw, 0, 0) - (geto(f, 1, 0)-geto(f, 0, 0))/h.*geto(u, 0, 0) -
            (geto(f, 0, 1)-geto(f, 0, 0))/h.*geto(v, 0, 0) -
            geto(f, 0, 0) .* ( (geto(u, 0, 0)-geto(u, -1, 0))/h + (geto(v, 0, 0)-geto(v, 0, -1))/h) -
            geto(G(K.*f.*λo*(ρw-ρo)*g, Z), 0, 0)
    rhs = Δt*rhs/geto(ϕ, 0, 0)
    sw = scatter_add(sw, 2:m-1, 2:n-1, rhs)
    return sw, p, u, v, f
end



"""
solve(qw, qo, sw0, p0)
Solve the two phase flow equation. 
`qw` and `qo` -- `NT x m x n` numerical array, `qw[i,:,:]` the corresponding value of qw at i*Δt
`sw0` and `p0` -- initial value for `sw` and `p`. `m x n` numerical array.
"""
function solve(qw, qo, sw0)
    qw_arr = constant(qw) # qw: NT x m x n array
    qo_arr = constant(qo)
    function condition(i, tas...)
        i <= NT
    end
    function body(i, tas...)
        println(i)
        ta_sw, ta_p, ta_u, ta_v, ta_f = tas
        sw, p, u, v, f = onestep(read(ta_sw, i), qw_arr[i], qo_arr[i])
        ta_sw = write(ta_sw, i+1, sw)
        ta_p = write(ta_p, i+1, p)
        ta_u = write(ta_u, i+1, u)
        ta_v = write(ta_v, i+1, v)
        ta_f = write(ta_f, i+1, f)
        i+1, ta_sw, ta_p, ta_u, ta_v, ta_f
    end
    ta_sw, ta_p = TensorArray(NT+1), TensorArray(NT+1)
    ta_u, ta_v, ta_f = TensorArray(NT+1), TensorArray(NT+1), TensorArray(NT+1)
    ta_sw = write(ta_sw, 1, constant(sw0))
    i = constant(1, dtype=Int32)
    _, ta_sw, ta_p, ta_u, ta_v, ta_f = while_loop(condition, body, [i; ta_sw; ta_p; ta_u; ta_v; ta_f])
    out_sw, out_p, out_u, out_v, ta_f = stack(ta_sw), stack(ta_p), stack(ta_u), stack(ta_v), stack(ta_f)
end

function vis(val, args...;kwargs...)
    close("all")
    ns = Int64.(round.(LinRange(1,size(val,1),9)))
    for i = 1:9
        subplot(330+i)
        imshow(val[ns[i],:,:], args...;kwargs...)
        colorbar()
    end
end

qw = zeros(NT, m, n)
qw[:,5, 5] .= 0.0026
qo = zeros(NT, m, n)
qo[:,5, 15] .= -0.004
sw0 = zeros(m, n)
out_sw, out_p, out_u, out_v, out_f = solve(qw, qo, sw0)


sess = Session(); init(sess)
S, P, U, V, F = run(sess, [out_sw, out_p, out_u, out_v, out_f])
vis(S)
# vis(U)
# figure()
# vis(P)
error("stop")


#=
# Step 1: Assign numerical values to qw, qo, sw0, p0
# qw = 
# qo = 
# sw0 = 
# p0 = 
qw = zeros(NT, m, n)
qw[:,15,5] .= 0.0018
qo = zeros(NT, m, n)
sw0 = zeros(m, n)
p0 = 3.0337e+07*ones(m,n)

# # Step 2: Construct Graph
out_sw, out_p = solve(qw, qo, sw0, p0)

# # Step 3: Run
sess = Session()
init(sess)
sw, p = run(sess, [out_sw, out_p])

# # Step 4: Visualize
# vis(sw)
=#