using Revise
using FwiFlow
using PyCall
using LinearAlgebra
using DelimitedFiles
using PyPlot
using MAT
matplotlib.use("agg")
np = pyimport("numpy")

# # mode 
# # 0 -- generate data 
# # 1 -- running inverse modeling
# mode = 1
# # mode 
# # 0 -- small data
# # 1 -- large data
# datamode = 0
# sparsity = 0.1

mode = parse(Int64, ARGS[1])
datamode = parse(Int64, ARGS[2])
sparsity = parse(Float64, ARGS[3])

FLDR = "datamode$(datamode)sparsity$sparsity"
if !isdir(FLDR)
    mkdir(FLDR)
end


# data structure for flow simulation
const K_CONST =  9.869232667160130e-16 * 86400 * 1e3
const ALPHA = 1.0
mutable struct Ctx
  m; n; h; NT; Δt; Z; X; ρw; ρo;
  μw; μo; K; g; ϕ; qw; qo; sw0
end

function tfCtxGen(m,n,h,NT,Δt,Z,X,ρw,ρo,μw,μo,K,g,ϕ,qw,qo,sw0,ifTrue)
  tf_h = constant(h)
  # tf_NT = constant(NT)
  tf_Δt = constant(Δt)
  tf_Z = constant(Z)
  tf_X= constant(X)
  tf_ρw = constant(ρw)
  tf_ρo = constant(ρo)
  tf_μw = constant(μw)
  tf_μo = constant(μo)
  # tf_K = isa(K,Array) ? Variable(K) : K
  if ifTrue
    tf_K = constant(K)
  else
    tf_K = Variable(K)
  end
  tf_g = constant(g)
  # tf_ϕ = Variable(ϕ)
  tf_ϕ = constant(ϕ)
  tf_qw = constant(qw)
  tf_qo = constant(qo)
  tf_sw0 = constant(sw0)
  return Ctx(m,n,tf_h,NT,tf_Δt,tf_Z,tf_X,tf_ρw,tf_ρo,tf_μw,tf_μo,tf_K,tf_g,tf_ϕ,tf_qw,tf_qo,tf_sw0)
end

if mode == 0
    global Krw, Kro
    # LET-type 
    Lw = 1.8
    Ew = 2.1
    Tw = 2.3
    Krwo = 0.6
    function Krw(Sw)
        return Krwo*Sw^Lw/(Sw^Lw + Ew*(1-Sw)^Tw)
    end

    function Kro(So)
        return So^Lw / (So^Lw + Ew*(1-So)^Tw)
    end
elseif mode == 1
    global Krw, Kro, θ1, θ2
    θ1 = Variable(ae_init([1,20,20,20,1]))
    θ2 = Variable(ae_init([1,20,20,20,1]))
    function Krw(Sw)
        Sw_ = tf.reshape(Sw, (-1,1))
        y = ae(Sw_, [20,20,20,1],θ1)
        tf.reshape((tanh(y)+1)/2, Sw.shape)
    end
    function Kro(So)
        So_ = tf.reshape(So, (-1,1))
        y = ae(So_, [20,20,20,1],θ2)
        tf.reshape(1-(tanh(y)+1)/2, So.shape)
    end
end


function krw_and_kro()
    sw = LinRange(0,1,100)|>collect|>constant
    Krw(sw), Kro(1-sw)
end

# IMPES for flow simulation
function ave_normal(quantity, m, n)
    aa = sum(quantity)
    return aa/(m*n)
end

function onestep(sw, p, m, n, h, Δt, Z, ρw, ρo, μw, μo, K, g, ϕ, qw, qo)
    # step 1: update p
    λw = Krw(sw)/μw
    λo = Kro(1-sw)/μo
    # λw = sw.*sw/μw
    # λo = (1-sw).*(1-sw)/μo
    λ = λw + λo
    q = qw + qo + λw/(λo+1e-16).*qo
    # q = qw + qo
    potential_c = (ρw - ρo)*g .* Z

    # Step 1: implicit potential
    Θ = upwlap_op(K * K_CONST, λo, potential_c, h, constant(0.0))

    load_normal = (Θ+q/ALPHA) - ave_normal(Θ+q/ALPHA, m, n)

    # p = poisson_op(λ.*K* K_CONST, load_normal, h, constant(0.0), constant(1))
    p = upwps_op(K * K_CONST, λ, load_normal, p, h, constant(0.0), constant(0)) # potential p = pw - ρw*g*h 

    # step 2: implicit transport
    sw = sat_op(sw, p, K * K_CONST, ϕ, qw, qo, μw, μo, sw, Δt, h)
    return sw, p
end


function imseq(tf_ctx)
    ta_sw, ta_p = TensorArray(NT+1), TensorArray(NT+1)
    ta_sw = write(ta_sw, 1, tf_ctx.sw0)
    ta_p = write(ta_p, 1, constant(zeros(tf_ctx.m, tf_ctx.n)))
    i = constant(1, dtype=Int32)
    function condition(i, tas...)
        i <= tf_ctx.NT
    end
    function body(i, tas...)
        ta_sw, ta_p = tas
        sw, p = onestep(read(ta_sw, i), read(ta_p, i), tf_ctx.m, tf_ctx.n, tf_ctx.h, tf_ctx.Δt, tf_ctx.Z, tf_ctx.ρw, tf_ctx.ρo, tf_ctx.μw, tf_ctx.μo, tf_ctx.K, tf_ctx.g, tf_ctx.ϕ, tf_ctx.qw[i], tf_ctx.qo[i])
        ta_sw = write(ta_sw, i+1, sw)
        ta_p = write(ta_p, i+1, p)
        i+1, ta_sw, ta_p
    end

    _, ta_sw, ta_p = while_loop(condition, body, [i, ta_sw, ta_p])
    out_sw, out_p = stack(ta_sw), stack(ta_p)
end

# visualization functions
function plot_saturation_series(S)

    z_inj = (9-1)*h + h/2.0
    x_inj = (3-1)*h + h/2.0
    z_prod = (9-1)*h + h/2.0
    x_prod = (28-1)*h + h/2.0
    fig2,axs = subplots(3,3, figsize=[30,15], sharex=true, sharey=true)
    ims = Array{Any}(undef, 9)
    for iPrj = 1:3
        for jPrj = 1:3
            @info iPrj, jPrj
            ims[(iPrj-1)*3+jPrj] = axs[iPrj,jPrj].imshow(S[survey_indices[(iPrj-1)*3+jPrj], :, :], extent=[0,n*h,m*h,0], vmin=0.0, vmax=0.6);
            axs[iPrj,jPrj].title.set_text("Snapshot $((iPrj-1)*3+jPrj)")
            if jPrj == 1 || jPrj == 1
                axs[iPrj,jPrj].set_ylabel("Depth (m)")
            end
            if iPrj == 3 || iPrj == 3
                axs[iPrj,jPrj].set_xlabel("Distance (m)")
            end
            # if iPrj ==2 && jPrj == 3
            # cb = fig2.colorbar(ims[(iPrj-1)*3+jPrj], ax=axs[iPrj,jPrj])
            # cb.set_label("Saturation")
            axs[iPrj,jPrj].scatter(x_inj, z_inj, c="r", marker=">", s=128)
            axs[iPrj,jPrj].scatter(x_prod, z_prod, c="r", marker="<", s=128)
        end
    end
    # fig2.subplots_adjust(wspace=0.04, hspace=0.042)
    fig2.subplots_adjust(wspace=0.02, hspace=0.18)
    cbar_ax = fig2.add_axes([0.91, 0.08, 0.01, 0.82])
    cb2 = fig2.colorbar(ims[1], cax=cbar_ax)
    cb2.set_label("Saturation")
    # savefig("figures_summary/Saturation_evo_patchy_init.pdf",bbox_inches="tight",pad_inches = 0);

end

function plot_saturation(S)

    z_inj = (Int(round(0.6*m))-1)*h + h/2.0
    x_inj = (Int(round(0.1*n))-1)*h + h/2.0
    z_prod = (Int(round(0.6*m))-1)*h + h/2.0
    x_prod = (Int(round(0.9*n))-1)*h + h/2.0
    imshow(S[end, :, :], extent=[0,n*h,m*h,0], vmin=0.0, vmax=0.6)        
    ylabel("Depth (m)")
    xlabel("Distance (m)")     
    colorbar()   
    scatter(x_inj, z_inj, c="r", marker=">", s=128)
    scatter(x_prod, z_prod, c="r", marker="<", s=128)

end


function plot_kr(krw, kro, w=missing, o=missing)
    close("all")
    plot(LinRange(0,1,100), krw, "b-", label="\$K_{rw}\$")
    plot(LinRange(0,1,100), kro, "r-", label="\$K_{ro}\$")
    if !ismissing(w)
        plot(LinRange(0,1,100), w, "g--", label="True \$K_{rw}\$")
        plot(LinRange(0,1,100), o, "c--", label="True \$K_{ro}\$")
    end
    xlabel("\$S_w\$")
    ylabel("\$K_r\$")
    legend()
end

# parameters for flow simulation
const SRC_CONST = 86400.0 #
const GRAV_CONST = 9.8    # gravity constant

if datamode==0
    # Hyperparameter for flow simulation
    global m = 15
    global n = 30
    global h = 30.0 # meter
    global NT  = 50
    global dt_survey = 5
    global Δt = 20.0 # day
else
    global m = 45
    global n = 90
    global h = 10.0 # meter
    global NT  = 100
    global dt_survey = 10
    global Δt = 10.0 # day
end

z = (1:m)*h|>collect
x = (1:n)*h|>collect
X, Z = np.meshgrid(x, z)

ρw = 501.9
ρo = 1053.0
μw = 0.1
μo = 1.0

K_init = 20.0 .* ones(m,n) # initial guess of permeability 

g = GRAV_CONST
ϕ = 0.25 .* ones(m,n)
qw = zeros(NT, m, n)
qw[:,Int(round(0.6*m)),Int(round(0.1*n))] .= 0.005 * (1/h^2)/10.0 * SRC_CONST
qo = zeros(NT, m, n)
qo[:,Int(round(0.6*m)),Int(round(0.9*n))] .= -0.005 * (1/h^2)/10.0 * SRC_CONST
sw0 = zeros(m, n)
survey_indices = collect(1:dt_survey:NT+1) # 10 stages
n_survey = length(survey_indices)



K = 20.0 .* ones(m,n) # millidarcy
K[Int(round(0.52*m)):Int(round(0.67*m)),:] .= 120.0
tfCtxTrue = tfCtxGen(m,n,h,NT,Δt,Z,X,ρw,ρo,μw,μo,K,g,ϕ,qw,qo, sw0, true)

out_sw_true, out_p_true = imseq(tfCtxTrue)
krw, kro = krw_and_kro()

using Random; Random.seed!(233)
obs_ids = rand(1:m*n, Int(round(m*n*sparsity)))
obs = tf.reshape(out_sw_true[1:NT+1], (NT+1,-1))[:, obs_ids]

# executing the simulation
if mode == 0
    # generate data
    sess = Session(); init(sess)
    S,obs_ = run(sess, [out_sw_true, obs])
    krw_, kro_ = run(sess, [krw, kro])
    matwrite("$FLDR/Data.mat", Dict("S"=>S, "krw"=>krw_, "kro"=>kro_, "obs"=>obs_))
    close("all");plot_kr(krw_, kro_); savefig("$FLDR/krwo.png")
    close("all");plot_saturation(S); savefig("$FLDR/sat.png")

else
    dat = Dict{String, Any}("loss" => Float64[])
    summary = i->begin
        global dat 
        krw_, kro_ = run(sess, [krw, kro])
        S = run(sess, out_sw_true)
        close("all");plot_kr(krw_, kro_, wref, oref); savefig("$FLDR/krwo$i.png")
        close("all");plot_saturation(S); savefig("$FLDR/sat$i.png")
        dat["S$i"] = S 
        dat["w$i"] = krw_
        dat["o$i"] = kro_
        dat["loss"] = [dat["loss"]; loss_]
        dat["theta1_$i"] = run(sess, θ1)
        dat["theta2_$i"] = run(sess, θ2)
        matwrite("$FLDR/invData.mat", dat)
    end
    d = matread("$FLDR/Data.mat")
    Sref, wref, oref, obsref = d["S"], d["krw"][:], d["kro"][:], d["obs"]
    # loss = sum((out_sw_true - Sref)^2)
    loss = sum((obsref-obs)^2)
    sess = Session(); init(sess)

    loss_ = Float64[];
    summary(0)
    for i = 1:100
        global loss_ = BFGS!(sess, loss, 100)
        summary(i)
    end
end




