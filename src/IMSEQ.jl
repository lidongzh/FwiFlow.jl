using PyPlot
using LinearAlgebra
using PyTensorFlow
using PyCall
np = pyimport("numpy")
include("poisson_op.jl")
include("laplacian_op.jl")
include("sat_op.jl")


# Solver parameters
m = 15
n = 30
h = 110.0
# T = 100 # 100 days
# NT = 100
# Δt = T/(NT+1)
NT = 92
Δt = 20.
# T = NT() # 100 days
z = (1:m)*h|>collect
x = (1:n)*h|>collect
X, Z = np.meshgrid(x, z)

tf_Z = constant(Z)
tf_h = constant(h)
tf_t = constant(Δt)

const ALPHA = 0.006323996017182
# const ALPHA = 0.001127
const SRC_CONST = 5.6146
# const SRC_CONST = 1.0
const GRAV_CONST = 1.0/144.0
ρw = constant(62.238)
ρo = constant(40.0)
μw = constant(1.0)
μo = constant(3.0)
K_np = 80.0 .* ones(m,n)
# K_np[16,:] .= 5e-14
# K_np[12:14,:] .= 100.0
K = constant(K_np)
g = constant(9.8*GRAV_CONST)
ϕ = constant(0.25 .* ones(m,n))


function Krw(Sw)
    return Sw ^ 1.5
end

function Kro(So)
    return So ^1.5
end

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

function ave_normal(quantity, m, n)
    # aa = 0.0
    # for i = 1:m
    #     for j = 1:n
    #         aa = aa + quantity[i,j]
    #     end
    # end
    aa = sum(quantity)
    return aa/(m*n)
end


# variables : sw, u, v, p
# (time dependent) parameters: qw, qo, ϕ
function onestep(sw, p, qw, qo, Δt_dyn)
    # step 1: update p
    # λw = Krw(sw)/μw
    # λo = Kro(1-sw)/μo
    λw = sw.*sw/μw
    λo = (1-sw).*(1-sw)/μo
    λ = λw + λo
    f = λw/λ
    q = qw + qo + λw/λo.*qo
    potential_c = (ρw-ρo)*g .* tf_Z

    # Θ = laplacian_op(K.*λo, potential_c, tf_h, constant(0.0))
    Θ = upwlap_op(K, λo, potential_c, tf_h, constant(0.0))

    load_normal = (Θ+q) - ave_normal(Θ+q,m,n)

    # p = poisson_op(λ.*K, load_normal, tf_h, constant(0.0), constant(0)) # potential p = pw - ρw*g*h 
    p = upwps_op(K, λ, load_normal, p, tf_h, constant(0.0), constant(0)) # potential p = pw - ρw*g*h 

    # step 2: update u, v
    u = constant(zeros(m, n))
    v = constant(zeros(m, n))

    Δt_dyn = constant(0.0)

    # # step 3: update sw
    for j= 1:10
        λw = sw.*sw/μw
        λo = (1-sw).*(1-sw)/μo
        λ = λw + λo
        f = λw/λ
        rhs = qw + λw/λo.*qo + upwlap_op(K, f.*λ, p, tf_h, constant(0.0))
        rhs = Δt*rhs/ϕ
        sw = sw + rhs
    end

    return sw, p
end


# variables : sw, u, v, p
# (time dependent) parameters: qw, qo, ϕ
function onestep2(sw, p, qw, qo, sw_ref, Δt_dyn)
    # step 1: update p
    # λw = Krw(sw)/μw
    # λo = Kro(1-sw)/μo
    λw = sw.*sw/μw
    λo = (1-sw).*(1-sw)/μo
    λ = λw + λo
    f = λw/λ
    q = qw + qo + λw/λo.*qo
    potential_c = (ρw-ρo)*g .* tf_Z

    # Θ = laplacian_op(K.*λo, potential_c, tf_h, constant(0.0))
    Θ = upwlap_op(K, λo, potential_c, tf_h, constant(0.0))

    load_normal = (Θ+q/ALPHA) - ave_normal(Θ+q/ALPHA,m,n)

    # p = poisson_op(λ.*K, load_normal, tf_h, constant(0.0), constant(0)) # potential p = pw - ρw*g*h 
    p = upwps_op(K, λ, load_normal, p, tf_h, constant(0.0), constant(0)) # potential p = pw - ρw*g*h 

    # step 2: update u, v
    u = constant(zeros(m, n))
    v = constant(zeros(m, n))

    Δt_dyn = constant(0.0)

    # # step 3: update sw

    sw_new = sat_op(sw, p, K, ϕ, qw, qo, sw_ref, constant(Δt), tf_h)

    return sw_new, p, u, v, f, Δt_dyn
end


"""
solve(qw, qo, sw0, p0)
Solve the two phase flow equation. 
`qw` and `qo` -- `NT x m x n` numerical array, `qw[i,:,:]` the corresponding value of qw at i*Δt
`sw0` and `p0` -- initial value for `sw` and `p`. `m x n` numerical array.
"""
function solve(qw, qo, sw0, sw1)
    qw_arr = constant(qw) # qw: NT x m x n array
    qo_arr = constant(qo)
    ta_sw, ta_p, ta_sw_ref = TensorArray(NT+1), TensorArray(NT+1), TensorArray(NT+1)
    ta_u, ta_v, ta_f, Δt_array = TensorArray(NT+1), TensorArray(NT+1), TensorArray(NT+1), TensorArray(NT+1)
    ta_sw = write(ta_sw, 1, constant(sw0))
    ta_sw_ref = write(ta_sw_ref, 1, sw1)
    ta_p = write(ta_p, 1, constant(zeros(m,n)))
    i = constant(1, dtype=Int32)
    function condition(i, tas...)
        i <= NT
    end
    function body(i, tas...)
        ta_sw, ta_p, ta_u, ta_v, ta_f, ta_sw_ref, Δt_array = tas
        sw, p, u, v, f, Δt_dyn = onestep2(read(ta_sw, i), read(ta_p, i), qw_arr[i], qo_arr[i], read(ta_sw_ref, i), read(Δt_array, i))
        ta_sw = write(ta_sw, i+1, sw)
        ta_sw_ref = write(ta_sw_ref, i+1, sw)
        ta_p = write(ta_p, i+1, p)
        ta_u = write(ta_u, i+1, u)
        ta_v = write(ta_v, i+1, v)
        ta_f = write(ta_f, i+1, f)
        Δt_array = write(Δt_array, i, Δt_dyn)
        i+1, ta_sw, ta_p, ta_u, ta_v, ta_f, ta_sw_ref, Δt_array
    end

    _, ta_sw, ta_p, ta_u, ta_v, ta_f, ta_sw_ref, Δt_array = while_loop(condition, body, [i; ta_sw; ta_p; ta_u; ta_v; ta_f; ta_sw_ref; Δt_array])
    out_sw, out_p, out_u, out_v, out_f, out_Δt = stack(ta_sw), stack(ta_p), stack(ta_u), stack(ta_v), stack(ta_f), stack(Δt_array)
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

xx, yy = np.meshgrid(1:n, 1:m)
Gaussian1 = exp.(-1.0.*((xx.-5).^2+(yy.-7).^2))
Gaussian2 = exp.(-1.0.*((xx.-25).^2+(yy.-7).^2))
qw = zeros(NT, m, n)

# qw[:,12,5] .= (0.0026/h^3)
qw[:,7,5] .= 1400 * (1/h^2)/20 * SRC_CONST
qo = zeros(NT, m, n)

# qo[:,12,25] .= -(0.004/h^3)
qo[:,7,25] .= -2200 * (1/h^2)/20 * SRC_CONST
sw0 = zeros(m,n)
# sw0[10:12,16:18] .= 0.3

# sw1, p1 = onestep(constant(sw0), constant(zeros(m,n)), constant(qw[1,:,:]), constant(qo[1,:,:]), constant(0.0))
sw1 = constant(sw0)
out_sw, out_p, out_u, out_v, out_f, out_Δt = solve(qw, qo, sw0, sw1)


sess = Session(); init(sess)
S, P, U, V, F, T = run(sess, [out_sw, out_p, out_u, out_v, out_f, out_Δt])


vis(S)

# quiver(x, z, V[80,:,:],U[80,:,:])
# title("velocity field")

# vis(P)
