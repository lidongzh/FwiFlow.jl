using PyTensorFlow
using PyCall
using LinearAlgebra
using PyPlot
using DelimitedFiles
using Random
Random.seed!(233)

np = pyimport("numpy")
include("poisson_op.jl")
include("laplacian_op.jl")

mutable struct Ctx
  m; n; h; NT; Δt; Z; X; ρw; ρo;
  μw; μo; K; g; ϕ; qw; qo; sw0
end

function tfCtxGen(m,n,h,NT,Δt,Z,X,ρw,ρo,μw,μo,K,g,ϕ,qw,qo,sw0)
  tf_h = constant(h)
  # tf_NT = constant(NT)
  # tf_Δt = constant(Δt)
  tf_Z = constant(Z)
  tf_X= constant(X)
  tf_ρw = constant(ρw)
  tf_ρo = constant(ρo)
  tf_μw = constant(μw)
  tf_μo = constant(μo)
  # tf_K = isa(K,Array) ? Variable(K) : K
  tf_K = Variable(K)
  tf_g = constant(g)
  tf_ϕ = Variable(ϕ)
  tf_qw = constant(qw)
  tf_qo = constant(qo)
  tf_sw0 = constant(sw0)
  return Ctx(m,n,tf_h,NT,Δt,tf_Z,tf_X,tf_ρw,tf_ρo,tf_μw,tf_μo,tf_K,tf_g,tf_ϕ,tf_qw,tf_qo, tf_sw0)
end

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
    aa = sum(quantity)
    return aa/(m*n)
end


# variables : sw, u, v, p
# (time dependent) parameters: qw, qo, ϕ
function onestep(sw, p, Δt_dyn, m,n,h,Δt,Z,ρw,ρo,μw,μo,K,g,ϕ,qw,qo)
    # step 1: update p
    # λw = Krw(sw)/μw
    # λo = Kro(1-sw)/μo
    λw = sw.*sw/μw
    λo = (1-sw).*(1-sw)/μo
    λ = λw + λo
    f = λw/λ
    q = qw + qo + λw/λo.*qo
    potential_c = (ρw - ρo)*g .* Z

    # Θ = laplacian_op(K.*λo, potential_c, h, constant(0.0))
    Θ = upwlap_op(K, λo, potential_c, h, constant(0.0))

    load_normal = (Θ+q) - ave_normal(Θ+q, m, n)

    # p = poisson_op(λ.*K, load_normal, h, constant(0.0), constant(0)) # potential p = pw - ρw*g*h 
    p = upwps_op(K, λ, load_normal, p, h, constant(0.0), constant(0)) # potential p = pw - ρw*g*h 

    # step 2: update u, v
    # rhs_u = -geto(K, 0, 0).*geto(λ, 0, 0)/h.*(geto(p, 1, 0) - geto(p, 0, 0))
    # rhs_v = -geto(K, 0, 0).*geto(λ, 0, 0)/h.*(geto(p, 0, 1) - geto(p, 0, 0)) +
    #         geto(K, 0, 0).*geto(λw*ρw+λo*ρo, 0, 0)*g
    u = constant(zeros(m, n))
    v = constant(zeros(m, n))
    # u = scatter_add(u, 2:m-1, 2:n-1, rhs_u)
    # v = scatter_add(v, 2:m-1, 2:n-1, rhs_v)

    # # step 3: update sw
    rhs = qw + λw/λo.*qo + upwlap_op(K, f.*λ, p, h, constant(0.0))
    max_rhs = maximum(abs(rhs/ϕ))
    Δt_dyn =  0.001/max_rhs
    # NT_local = Δt/Δt_dyn
    for i= 1:10
        λw = sw.*sw/μw
        λo = (1-sw).*(1-sw)/μo
        λ = λw + λo
        f = λw/λ
        rhs = qw + λw/λo.*qo + upwlap_op(K, f.*λ, p, h, constant(0.0))
        rhs = Δt*rhs/ϕ
        sw = sw + rhs
    end

    sw = clamp(sw, 1e-16, 1.0-1e-16)
    return sw, p, u, v, f, Δt_dyn
end



"""
impes(tf_ctx)
Solve the two phase flow equation. 
`qw` and `qo` -- `NT x m x n` numerical array, `qw[i,:,:]` the corresponding value of qw at i*Δt
`sw0` and `p0` -- initial value for `sw` and `p`. `m x n` numerical array.
"""
function impes(tf_ctx)
    ta_sw, ta_p = TensorArray(NT+1), TensorArray(NT+1)
    ta_u, ta_v, ta_f, Δt_array = TensorArray(NT+1), TensorArray(NT+1), TensorArray(NT+1), TensorArray(NT+1)
    ta_sw = write(ta_sw, 1, tf_ctx.sw0)
    ta_p = write(ta_p, 1, constant(zeros(tf_ctx.m, tf_ctx.n)))
    i = constant(1, dtype=Int32)
    # K = tf_ctx.K
    # ϕ = tf_ctx.ϕ
    function condition(i, tas...)
        i <= tf_ctx.NT
    end
    function body(i, tas...)
        ta_sw, ta_p, ta_u, ta_v, ta_f, Δt_array = tas
        sw, p, u, v, f, Δt_dyn = onestep(read(ta_sw, i), read(ta_p, i), read(Δt_array, i), 
          tf_ctx.m, tf_ctx.n, tf_ctx.h, tf_ctx.Δt, tf_ctx.Z, tf_ctx.ρw, tf_ctx.ρo,
          tf_ctx.μw, tf_ctx.μo, tf_ctx.K, tf_ctx.g, tf_ctx.ϕ, tf_ctx.qw[i], tf_ctx.qo[i])
        ta_sw = write(ta_sw, i+1, sw)
        ta_p = write(ta_p, i+1, p)
        ta_u = write(ta_u, i+1, u)
        ta_v = write(ta_v, i+1, v)
        ta_f = write(ta_f, i+1, f)
        Δt_array = write(Δt_array, i, Δt_dyn)
        i+1, ta_sw, ta_p, ta_u, ta_v, ta_f, Δt_array
    end

    _, ta_sw, ta_p, ta_u, ta_v, ta_f, Δt_array = while_loop(condition, body, [i; ta_sw; ta_p; ta_u; ta_v; ta_f; Δt_array;])
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