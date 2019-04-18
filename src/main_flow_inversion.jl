include("ops_impes.jl")

# ========================= parameters ===========================
# m = 20
# n = 30
# h = 20.0
# # NT = 500
# NT = 20
# Δt = 8640
# z = (1:m)*h|>collect
# x = (1:n)*h|>collect
# X, Z = np.meshgrid(x, z)
# ρw = 996.9571
# ρo = 640.7385
# μw = 1e-3
# μo = 1e-3
# K = 9.8692e-14*ones(m,n) 
# # K_np[16,:] .= 5e-14
# K[8:10,:] .= 5e-14 
# g = 9.8
# ϕ = 0.25*ones(m,n)
# qw = zeros(NT, m, n)
# qw[:,12,5] .= (0.0026/h^3)
# qo = zeros(NT, m, n)
# qo[:,12,25] .= -(0.004/h^3)
# sw0 = zeros(m, n)

m = 20
n = 30
h = 1.0
# NT = 500
NT = 1000
Δt = 0.01
z = (1:m)*h|>collect
x = (1:n)*h|>collect
X, Z = np.meshgrid(x, z)
ρw = 1.0
ρo = 1.0
μw = 1.0
μo = 1.0
K = ones(m,n) 
# K[8:10,:] .= 1.2
# K[10:18, 14:18] .= 1.2
K[13:16,:] .= 5.0
K[8:13,14:18] .= 5.0
g = 1.0
ϕ = ones(m,n)
qw = zeros(NT, m, n)
qw[:,12,5] .= (1/h^3)
qo = zeros(NT, m, n)
qo[:,12,25] .= -(1/h^3)
sw0 = zeros(m, n)



tfCtxTrue = tfCtxGen(m,n,h,NT,Δt,Z,X,ρw,ρo,μw,μo,K,g,ϕ,qw,qo, sw0)
# K_init = K
K_init = ones(m,n)
tfCtxInit = tfCtxGen(m,n,h,NT,Δt,Z,X,ρw,ρo,μw,μo,K_init,g,ϕ,qw,qo,sw0)


# ========================= parameters ===========================
nPhase = NT

out_sw_true, out_p, out_u, out_v, out_f, out_Δt = impes(tfCtxTrue)
out_sw_syn, _, _, _, _, _ = impes(tfCtxInit)

function objective_function(out_sw_true, out_sw_syn, nPhase)
    tf.nn.l2_loss(out_sw_true-out_sw_syn)
end

J = objective_function(out_sw_true, out_sw_syn, nPhase)
# ========================= compute gradient ============================
sess = Session(); init(sess)
# S, P, U, V, F, T = run(sess, [out_sw_syn, out_p, out_u, out_v, out_f, out_Δt])
# vis(S)

# run(sess, J)
# error("")


# tf_grad_K = gradients(J, tfCtxInit.K)
# grad_K = run(sess, tf_grad_K)

# # @show maximum(abs.(run(sess, gradients(sum(out_sw_syn^2), K_init))))
# # close("all")
# imshow(grad_K)
# colorbar()


# ============================= inversion ===============================
__iter = 0
function print_iter(rk)
    global __iter
    if mod(__iter,1)==0
        println("\n************* ITER=$__iter *************\n")
    end
    __iter += 1
end

__cnt = 0
# invK = zeros(m,n)
function print_loss(l, invK)
    global __cnt
    if mod(__cnt,1)==0
        println("\niter=$__iter, eval=$__cnt, current loss=",l)
        # println("a=$a, b1=$b1, b2=$b2")
    end

    writedlm("./flow_fit_results/K_$(__iter).txt", invK)
    # writedlm("./flow_fit_results/b1_$(__iter).txt", b1)
    # writedlm("./flow_fit_results/b2_$(__iter).txt", b2)
    # writedlm("./flow_fit_results/l_$(__iter).txt", l)
    __cnt += 1
end

# config = tf.ConfigProto()
# config.intra_op_parallelism_threads = 24
# config.inter_op_parallelism_threads = 24
sess = Session(); init(sess);
opt = ScipyOptimizerInterface(J, method="L-BFGS-B",options=Dict("maxiter"=> 100, "ftol"=>1e-12, "gtol"=>1e-12))
ScipyOptimizerMinimize(sess, opt, loss_callback=print_loss, step_callback=print_iter, fetches=[J, tfCtxInit.K])

