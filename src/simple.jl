using PyPlot
using LinearAlgebra
using PyTensorFlow
using PyCall
np = pyimport("numpy")
include("poisson_op.jl")

# Solver parameters
m = 20
n = 20
h = 1.0 # 50 meters
T = 0.01 # 100 days
NT = 1000
Δt = T/NT
x = (1:m)*h|>collect
z = (1:n)*h|>collect
X, Z = np.meshgrid(x, z)


# todo 
#=
Krw, Kro -- function mxn tensor to mxn tensor
μw, μo -- known mxn
ρw, ρo -- known mxn
K -- scalar
g -- constant ≈ 9.8?
ϕ -- known mxn
=#
function Krw(Sw)
    return Sw ^ 2
end

function Kro(So)
    return So ^2
end
ρw = constant(1.)
ρo = constant(1.)
μw = constant(1.)
μo = constant(1.)
K = Variable(ones(m,n))
g = constant(1.0)
ϕ = Variable(0.25 .* ones(m,n))

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
        # q[2:m-1, 2:n-1] += rhs/h^2
    end
    q
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
    Θ = G((λw*ρw+λo*ρo)*g, Z)
    p = poisson_op(λ.*K, Θ+q, constant(h))


    # step 2: update u, v
    rhs_u = -(geto(K, 0, 0)+geto(K,1,0))/2.0 .* (geto(λ, 0, 0) + get(λ, 1, 0))/2h .* (geto(p, 1, 0) - geto(p, 0, 0))
    rhs_v = -(geto(K, 0, 0)+geto(K,0,1))/2.0 .* (geto(λ, 0, 0) + get(λ, 0, 1))/2h .* (geto(p, 0, 1) - geto(p, 0, 0)) +
            (geto(K, 0, 0)+geto(K,0,1))/2.0 .* (geto(λw*ρw+λo*ρo, 0, 0)+geto(λw*ρw+λo*ρo, 0, 1))/2 * g
    u = constant(zeros(m, n))
    v = constant(zeros(m, n))
    u = scatter_add(u, 2:m-1, 2:n-1, rhs_u)
    v = scatter_add(v, 2:m-1, 2:n-1, rhs_v)

    # step 3: update sw
    rhs = geto(qw, 0, 0) - (geto(f, 1, 0)-geto(f, 0, 0))/h.*geto(u, 0, 0) -
            (geto(f, 0, 1)-geto(f, 0, 0))/h.*geto(v, 0, 0) -
            geto(f, 0, 0) .* ( 
                (geto(u, 0, 0)-geto(u, -1, 0))/h + (geto(v, 0, 0)-geto(v, 0, -1))/h
            ) -
            geto(G(K.*f.*λo*(ρw-ρo)*g, Z), 0, 0)
    rhs = Δt*rhs/geto(ϕ, 0, 0)
    sw = scatter_add(sw, 2:m-1, 2:n-1, rhs)
    return sw, p
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
        ta_sw, ta_p = tas
        sw, p = onestep(read(ta_sw, i), qw_arr[i], qo_arr[i])
        ta_sw = write(ta_sw, i+1, sw)
        ta_p = write(ta_p, i+1, p)
        i+1, ta_sw, ta_p
    end
    ta_sw, ta_p = TensorArray(NT+1), TensorArray(NT+1)
    ta_sw = write(ta_sw, 1, constant(sw0))
    i = constant(1, dtype=Int32)
    _, ta_sw, ta_p = while_loop(condition, body, [i; ta_sw; ta_p])
    out_sw, out_p = stack(ta_sw), stack(ta_p)
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
qw[:,15,5] .= 1.0
qo = zeros(NT, m, n)
sw0 = zeros(m, n)
out_sw, out_p = solve(qw, qo, sw0)


sess = Session(); init(sess)
S, P = run(sess, [out_sw, out_p])
vis(S)
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