include("ops_impes.jl")

# ========================= parameters ===========================

m = 20
n = 30
h = 100.0 # ft
# NT = 500
NT  = 50
Δt = 1.0 # day
z = (1:m)*h|>collect
x = (1:n)*h|>collect
X, Z = np.meshgrid(x, z)
ρw = 62.238 # lbm/scf (pound/ft^3)
ρo = 40.0 # lbm/scf (pound/ft^3)
μw = 1.0 # centi poise
μo = 3.0
K = 1.0 .* ones(m,n) # millidarcy
K[8:10,:] .= 2.0
K[10:18, 14:18] .= 2.0
# K[13:16,:] .= 80.0
# K[8:13,14:18] .= 80.0
# g = 9.8 * 7.0862e-04
g = 0.0
ϕ = 0.25 .* ones(m,n)
qw = zeros(NT, m, n)
qw[:,12,5] .= 1400 * (1/h^3) * 5.6146
qo = zeros(NT, m, n)
qo[:,12,25] .= -2200 * (1/h^3) * 5.6146

sw0 = zeros(m, n)
# const ALPHA = 0.006323996017182
const ALPHA = 0.6323996017182


tfCtxTrue = tfCtxGen(m,n,h,NT,Δt,Z,X,ρw,ρo,μw,μo,K,g,ϕ,qw,qo, sw0)
# K_init = K
K_init = 0.8 .* ones(m,n)
tfCtxInit = tfCtxGen(m,n,h,NT,Δt,Z,X,ρw,ρo,μw,μo,K_init,g,ϕ,qw,qo,sw0)


# ========================= parameters ===========================
nPhase = 10

out_sw_true, out_p, out_u, out_v, out_f, out_Δt = impes(tfCtxTrue)

out_sw_syn, _, _, _, _, _ = impes(tfCtxInit)

function objective_function(out_sw_true, out_sw_syn, nPhase)
    # tf.nn.l2_loss(out_sw_true-out_sw_syn)
    sum = 0.0
    for i in trunc.(Int64, range(2, NT, length=nPhase))
        sum += tf.nn.l2_loss(out_sw_true[i]-out_sw_syn[i])
        # sum += norm(out_sw_true[i]-out_sw_syn[i])^2
    end
    return sum
end

J = objective_function(out_sw_true, out_sw_syn, nPhase)
# ========================= compute gradient ============================
sess = Session(); init(sess)
S, P, U, V, F, T, Obj = run(sess, [out_sw_true, out_p, out_u, out_v, out_f, out_Δt, J])
vis(S)
# println("Obj = $Obj")
# error("")

# run(sess, J)
error("")


tf_grad_K = gradients(J, tfCtxInit.K)
grad_K = run(sess, tf_grad_K)
imshow(grad_K)
colorbar()

tf_grad_h = gradients(J, tfCtxInit.h)
grad_h = run(sess, tf_grad_h)

tf_grad_g = gradients(J, tfCtxInit.g)
grad_g = run(sess, tf_grad_g)

# ============================= inversion ===============================
# __iter = 0
# function print_iter(rk)
#     global __iter
#     if mod(__iter,1)==0
#         println("\n************* ITER=$__iter *************\n")
#     end
#     __iter += 1
# end

# __cnt = 0
# # invK = zeros(m,n)
# function print_loss(l, invK)
#     global __cnt
#     if mod(__cnt,1)==0
#         println("\niter=$__iter, eval=$__cnt, current loss=",l)
#         # println("a=$a, b1=$b1, b2=$b2")
#     end

#     writedlm("./flow_fit_results/K_$(__iter).txt", invK)
#     # writedlm("./flow_fit_results/b1_$(__iter).txt", b1)
#     # writedlm("./flow_fit_results/b2_$(__iter).txt", b2)
#     # writedlm("./flow_fit_results/l_$(__iter).txt", l)
#     __cnt += 1
# end

# # config = tf.ConfigProto()
# # config.intra_op_parallelism_threads = 24
# # config.inter_op_parallelism_threads = 24
# sess = Session(); init(sess);
# opt = ScipyOptimizerInterface(J, method="L-BFGS-B",options=Dict("maxiter"=> 100, "ftol"=>1e-12, "gtol"=>1e-12))
# ScipyOptimizerMinimize(sess, opt, loss_callback=print_loss, step_callback=print_iter, fetches=[J, tfCtxInit.K])

