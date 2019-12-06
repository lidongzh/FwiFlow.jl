using PyPlot
using LinearAlgebra
using PyTensorFlow
using PyCall
np = pyimport("numpy")
include("poisson_op.jl")
include("laplacian_op.jl")

# Solver parameters

m = 20
n = 30
h = 20.0
# T = 100 # 100 days
# NT = 100
# Δt = T/(NT+1)
NT = 200
Δt = 864
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

# ρw = constant(62.238)
# ρo = constant(40.0)
# μw = constant(1.0)
# μo = constant(3.0)
# K_np = 1.0 .* ones(m,n)
# # K_np[16,:] .= 5e-14
# # K_np[8:10,:] .= 1.2
# # xx, yy = np.meshgrid(1:n, 1:m)
# # Gaussian1 = 0.1 .* exp.(-1.0.*((xx.-10).^2+(yy.-10).^2))
# # K_np += Gaussian1
# K = constant(K_np)
# g = constant(0.0)
# ϕ = Variable(0.25 .* ones(m,n))
# const ALPHA = 0.6323996017182

ρw = constant(996.9571)
ρo = constant(640.7385)
μw = constant(1e-3)
μo = constant(3e-3)
K_np = 9.8692e-14*ones(m,n)
# K_np[8:end,:] .= 5e-14
K = constant(K_np)
g = constant(9.8)
ϕ = Variable(0.25*ones(m,n))
const ALPHA = 1.0

# ρw = constant(1.0)
# ρo = constant(1.0)
# μw = constant(1.0)
# μo = constant(1.0)
# K_np = ones(m,n)
# # K_np[16,:] .= 5e-14
# K_np[8:10,:] .= 1.10
# K = constant(K_np)
# g = constant(1.0)
# ϕ = Variable(ones(m,n))
# const ALPHA = 1.0

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
    # aa = sum(quantity)
    return aa/(m*n)
end


# # variables : sw, u, v, p
# # (time dependent) parameters: qw, qo, ϕ
# function onestep(sw, qw, qo)
#     # step 1: update p
#     # λw = Krw(sw)/μw
#     # λo = Kro(1-sw)/μo

#     λw = sw.*sw/μw
#     λo = (1 - sw).*(1 - sw)/μo
#     # λw = sw.*sw
#     # λo = (1 - sw).*(1 - sw)
#     # λo = constant(ones(m,n)) / μo
#     λ = λw + λo
#     f = λw/λ

#     # λw = constant(zeros(m,n)) / μw
#     # λo = constant(ones(m,n)) / μo
#     # λ = λw + λo
#     # f = sw

#     q = qw + qo + λw/λo.*qo
#     # q = qw + qo
#     potential_c = (ρw-ρo)*g .* tf_Z

#     # Θ = laplacian_op(K.*λo, potential_c, tf_h, constant(0.0))
#     Θ = upwlap_op(K, λo, potential_c, tf_h, constant(0.0))

#     load_normal = (Θ+q/ALPHA) - ave_normal(Θ+q/ALPHA,m,n)
#     # load_normal = (Θ+q/ALPHA)

#     # p = poisson_op(λ.*K, load_normal, tf_h, constant(0.0), constant(0)) # potential p = pw - ρw*g*h 
#     p = upwps_op(K, λ, load_normal, constant(zeros(m,n)), tf_h, constant(0.0), constant(2)) # potential p = pw - ρw*g*h 


#     # # step 3: update sw
#     rhs = qw + λw/λo.*qo + upwlap_op(K, f.*λ, p, tf_h, constant(0.0))
#     max_rhs = maximum(abs(rhs/ϕ))
#     Δt_dyn =  0.001/max_rhs
#     NT_local = Δt/Δt_dyn
#     for i= 1:10
#         λw = sw.*sw/μw
#         λo = (1-sw).*(1-sw)/μo
#         λ = λw + λo
#         f = λw/λ
#         rhs = qw + λw/λo.*qo + upwlap_op(K, f.*λ, p, tf_h, constant(0.0))
#         rhs = Δt*rhs/ϕ
#         sw = sw + rhs
#     end

#     # sw = sub_steps(sw, K, tf_h, p, μw, μo, qw, qo, ϕ, Δt, 100)


#     # rhs = qw + λw/λo.*qo + ALPHA * upwlap_op(K, f.*λ, p, tf_h, constant(0.0))
#     # # rhs = qw + ALPHA * laplacian_op(K.*f.*λ, p, tf_h, constant(0.0))
#     # sw = sw + Δt*rhs/ϕ

#     # sw = clamp(sw, 1e-16, 1.0-1e-16)
#     return sw
# end

# variables : sw, u, v, p
# (time dependent) parameters: qw, qo, ϕ
function onestep2(sw, qw, qo)
    # step 1: update p
    # λw = Krw(sw)/μw
    # λo = Kro(1-sw)/μo
    λw = sw.*sw/μw
    λo = (1-sw).*(1-sw)/μo
    λ = λw + λo
    f = λw/λ
    q = qw + qo
    Θ = -laplacian_op(K.*(λw*ρw+λo*ρo)*g, tf_Z, tf_h, constant(1.0))
    # Θ = -upwlap_op(K, (λw*ρw+λo*ρo)*g, tf_Z, tf_h, tf_h)
    load_normal = (Θ+q) - ave_normal(Θ+q,m,n)
    p = poisson_op(λ.*K, load_normal, tf_h, ρo*g, constant(0))
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
    rhs = qw + λw/λo.*qo + upwlap_op(K, f.*λ, p, tf_h, ρo*g) - upwlap_op(K, f.*λ.*ρw.*g, tf_Z, tf_h, constant(1.0))
    max_rhs = maximum(abs(rhs/ϕ))
    Δt_dyn =  0.001/max_rhs
    # NT_local = Δt/Δt_dyn
    for i= 1:10
        λw = sw.*sw/μw
        λo = (1-sw).*(1-sw)/μo
        λ = λw + λo
        f = λw/λ
        rhs = qw + λw/λo.*qo + upwlap_op(K, f.*λ, p, tf_h, ρo*g) - upwlap_op(K, f.*λ.*ρw.*g, tf_Z, tf_h, constant(1.0))
        rhs = Δt*rhs/ϕ
        sw = sw + rhs
    end

    # sw = clamp(sw, 1e-16, 1.0-1e-16)
    return sw
end

# function sub_steps(sw, K, tf_h, p, μw, μo, qw, qo, ϕ, Δt, nsteps)
#     tf_sw_sub = TensorArray(nsteps+1)
#     function condition(i)
#         tf.less(i, nsteps+1)
#     end
#     function body(i, tf_sw_sub)
#         sw_local = read(tf_sw_sub, i-1)
#         λw = sw_local.*sw_local/μw
#         λo = (1-sw_local).*(1-sw_local)/μo
#         λ = λw + λo
#         f = λw/λ
#         rhs = qw + λw/λo.*qo + upwlap_op(K, f.*λ, p, tf_h, constant(0.0))
#         sw_new = sw_new + Δt*rhs/ϕ
#         i+1, write(tf_sw_sub, i+1, sw_new)
#     end
#     tf_sw_sub = write(tf_sw_sub, 1, sw)
#     i = constant(2, dtype=Int32)
#     _, tf_sw_sub = while_loop(condition, body, [i;tf_sw_sub])
#     read(tf_sw_sub, nsteps+1)
# end

# function evolve(sw, NT)
#     for i = 1:NT
#         sw = onestep(sw, tf_qw[i], tf_qw[i])
#     end 
#     return sw
# end

function evolve(sw, NT)
    # qw_arr = constant(qw) # qw: NT x m x n array
    # qo_arr = constant(qo)
    tf_sw = TensorArray(NT+1)
    function condition(i, ta)
        tf.less(i, NT+1)
    end
    function body(i, tf_sw)
        sw_local = onestep2(read(tf_sw, i),tf_qw[i], tf_qo[i])
        i+1, write(tf_sw, i+1, sw_local)
    end
    tf_sw = write(tf_sw, 1, sw)
    i = constant(1, dtype=Int32)
    _, out = while_loop(condition, body, [i;tf_sw])
    read(out, NT+1)
end


# qw = zeros(NT, m, n)
# qw[:,12,5] .= (1000.0/h^3)* 5.6146
# qo = zeros(NT, m, n)
# qo[:,12,25] .= -(1000.0/h^3)* 5.6146
# sw0 = zeros(m, n)

qw = zeros(NT, m, n)
qw[:,12,5] .= (0.0026/h^3)
qo = zeros(NT, m, n)
qo[:,12,25] .= -(0.004/h^3)
sw0 = zeros(m, n)

# qw = zeros(NT, m, n)
# qw[:,12,5] .= (1.0/h^3)
# qo = zeros(NT, m, n)
# qo[:,12,25] .= -(1.0/h^3)
# sw0 = zeros(m, n)

tf_qw = constant(qw)
tf_qo = constant(qo)
tf_sw0 = constant(sw0)

tf_out_sw = evolve(tf_sw0, NT)

nPhase=2


function objective_function(nPhase)
    tf_sw_array = Array{Any}(undef, nPhase)
    tf_sw_array[1] = tf_sw0
    for i = 2:nPhase
        @show i
        # 5 steps
        tf_sw_array[i] = evolve(tf_sw_array[i-1], NT)
    end

    # global J
    obj = 0.0
    for i = 2:nPhase
        # @show i
        # obj += 0.5*sum((v_pert_simu[i]-v_pert[i]).^2)
        # obj += norm(tf_sw_array[i])^2
        obj += sum(tanh(tf_sw_array[i]))
    end
    return obj
end

J = objective_function(nPhase)

tf_grad_K = gradients(J, K)



sess = Session(); init(sess)
S = run(sess, tf_out_sw)
imshow(S)
colorbar()

error("")

grad_K = run(sess, tf_grad_K)
imshow(grad_K)
colorbar()