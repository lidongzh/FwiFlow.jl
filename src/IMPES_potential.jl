using PyPlot
using LinearAlgebra
using PyTensorFlow
using PyCall
np = pyimport("numpy")
include("poisson_op.jl")
include("laplacian_op.jl")

# Solver parameters

m = 15
n = 30
h = 20.0
# T = 100 # 100 days
# NT = 100
# Δt = T/(NT+1)
NT = 150
Δt = 8640
# T = NT() # 100 days
z = (1:m)*h|>collect
x = (1:n)*h|>collect
X, Z = np.meshgrid(x, z)

tf_Z = constant(Z)
tf_h = constant(h)

function Krw(Sw)
    return Sw ^ 1.5
end

function Kro(So)
    return So ^1.5
end

ρw = constant(996.9571)
ρo = constant(640.7385)
μw = constant(1e-3)
μo = constant(3e-3)
K_np = 9.8692e-14*ones(m,n)
# K_np[8:10,:] .= 5e-13
K = constant(K_np)
g = constant(9.8)
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
function onestep(sw, qw, qo, Δt_dyn)
    # step 1: update p
    # λw = Krw(sw)/μw
    # λo = Kro(1-sw)/μo
    λw = sw.*sw/μw
    λo = (1-sw).*(1-sw)/μo
    λ = λw + λo
    f = λw/λ
    q = qw + qo
    potential_c = -(ρo-ρw)*g .* tf_Z
    Θ = laplacian_op(K.*λo, potential_c, tf_h, -(ρo-ρw)*g)
    # Θ = upwlap_op(K, λo*(ρw-ρo)*g, tf_Z, tf_h, tf_h)
    load_normal = (Θ+q) - ave_normal(Θ+q,m,n)
    p = poisson_op(λ.*K, load_normal, tf_h, constant(0.0), constant(0))
    # p = constant(ones(m,n))

    # step 2: update u, v
    rhs_u = -geto(K, 0, 0).*geto(λ, 0, 0)/h.*(geto(p, 1, 0) - geto(p, 0, 0))
    rhs_v = -geto(K, 0, 0).*geto(λ, 0, 0)/h.*(geto(p, 0, 1) - geto(p, 0, 0)) +
            geto(K, 0, 0).*geto(λw*ρw+λo*ρo, 0, 0)*g
    u = constant(zeros(m, n))
    v = constant(zeros(m, n))
    u = scatter_add(u, 2:m-1, 2:n-1, rhs_u)
    v = scatter_add(v, 2:m-1, 2:n-1, rhs_v)

    # # step 3: update sw

    # rhs = qw + λw/λo.*qo + upwlap_op(K, f.*λ, p, constant(h), ρo*g) - laplacian_op(f.*K.*λ.*ρw.*g, tf_Z, constant(h), constant(h))
    rhs = qw + λw/λo.*qo + upwlap_op(K, f.*λ, p, tf_h, constant(0.0))
    max_rhs = maximum(abs(rhs/ϕ))
    Δt_dyn =  0.001/max_rhs
    # NT_local = Δt/Δt_dyn
    for i= 1:10
        λw = sw.*sw/μw
        λo = (1-sw).*(1-sw)/μo
        λ = λw + λo
        f = λw/λ
        rhs = qw + λw/λo.*qo + upwlap_op(K, f.*λ, p, tf_h, constant(0.0))
        rhs = Δt*rhs/ϕ
        sw = sw + rhs
    end

    # sw = clamp(sw, 1e-16, 1.0-1e-16)
    return sw, p, u, v, f, Δt_dyn
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
        ta_sw, ta_p, ta_u, ta_v, ta_f, Δt_array = tas
        sw, p, u, v, f, Δt_dyn = onestep(read(ta_sw, i), qw_arr[i], qo_arr[i], read(Δt_array, i))
        ta_sw = write(ta_sw, i+1, sw)
        ta_p = write(ta_p, i+1, p)
        ta_u = write(ta_u, i+1, u)
        ta_v = write(ta_v, i+1, v)
        ta_f = write(ta_f, i+1, f)
        Δt_array = write(Δt_array, i, Δt_dyn)
        i+1, ta_sw, ta_p, ta_u, ta_v, ta_f, Δt_array
    end
    ta_sw, ta_p = TensorArray(NT+1), TensorArray(NT+1)
    ta_u, ta_v, ta_f, Δt_array = TensorArray(NT+1), TensorArray(NT+1), TensorArray(NT+1), TensorArray(NT+1)
    ta_sw = write(ta_sw, 1, constant(sw0))
    i = constant(1, dtype=Int32)
    _, ta_sw, ta_p, ta_u, ta_v, ta_f, Δt_array = while_loop(condition, body, [i; ta_sw; ta_p; ta_u; ta_v; ta_f; Δt_array])
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

qw[:,7,5] .= (0.0026/h^2)/h
qo = zeros(NT, m, n)

qo[:,7,25] .= -(0.004/h^2)/h
sw0 = zeros(m, n)

out_sw, out_p, out_u, out_v, out_f, out_Δt = solve(qw, qo, sw0)


sess = Session(); init(sess)
S, P, U, V, F, T = run(sess, [out_sw, out_p, out_u, out_v, out_f, out_Δt])


vis(S)

# quiver(x, z, V[80,:,:],U[80,:,:])
# title("velocity field")

# vis(P)
