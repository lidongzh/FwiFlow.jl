# include("ops_impes.jl")
include("ops_imseq.jl")

# ========================= parameters ===========================
const ALPHA = 0.006323996017182
const SRC_CONST = 5.6146
const GRAV_CONST = 1.0/144.0
m = 15
n = 30
h = 100.0 # ft
# NT = 500
NT  = 50
Δt = 20.0 # day
z = (1:m)*h|>collect
x = (1:n)*h|>collect
X, Z = np.meshgrid(x, z)
ρw = 62.238 # lbm/scf (pound/ft^3)
ρo = 40.0 # lbm/scf (pound/ft^3)
μw = 1.0 # centi poise
μo = 3.0
K = 20.0 .* ones(m,n) # millidarcy
K[8:10,:] .= 100.0
# K[10:18, 14:18] .= 100.0
# K[13:16,:] .= 80.0
# K[8:13,14:18] .= 80.0
# g = 9.8 * 7.0862e-04
g = 0.0
ϕ = 0.25 .* ones(m,n)
qw = zeros(NT, m, n)
qw[:,7,5] .= 1400 * (1/h^2)/20 * SRC_CONST
qo = zeros(NT, m, n)
qo[:,7,25] .= -2200 * (1/h^2)/20 * SRC_CONST

sw0 = zeros(m, n)


tfCtxTrue = tfCtxGen(m,n,h,NT,Δt,Z,X,ρw,ρo,μw,μo,K,g,ϕ,qw,qo, sw0, true)
# K_init = K
K_init = 20.0 .* ones(m,n)
tfCtxInit = tfCtxGen(m,n,h,NT,Δt,Z,X,ρw,ρo,μw,μo,K_init,g,ϕ,qw,qo,sw0, false)


# ========================= parameters ===========================
nPhase = 10

# out_sw_true, out_p, out_u, out_v, out_f, out_Δt = impes(tfCtxTrue)
# out_sw_syn, _, _, _, _, _ = impes(tfCtxInit)

out_sw_true, out_p_true = imseq(tfCtxTrue)
out_sw_syn, out_p_syn = imseq(tfCtxInit)


function objective_function(out_sw_true, out_sw_syn, nPhase)
    tf.nn.l2_loss(out_sw_true-out_sw_syn)
    # sum = 0.0
    # for i in trunc.(Int64, range(2, NT, length=nPhase))
    #     sum += tf.nn.l2_loss(out_sw_true[i]-out_sw_syn[i])
    #     # sum += norm(out_sw_true[i]-out_sw_syn[i])^2
    # end
    # return sum
end

J = objective_function(out_sw_true, out_sw_syn, nPhase)
gradK = gradients(J, tfCtxInit.K)
# ========================= compute gradient ============================
# sess = Session(); init(sess)
# # S, P, U, V, F, T, Obj = run(sess, [out_sw_true, out_p, out_u, out_v, out_f, out_Δt, J])
# S, P = run(sess, [out_sw_true, out_p_true])
# vis(S)


# # # run(sess, J)
# error("")


# tf_grad_K = gradients(J, tfCtxInit.K)
# grad_K = run(sess, tf_grad_K)
# imshow(grad_K)
# colorbar()

# ============================= inversion ===============================
if !isdir("flow_fit_results")
    mkdir("flow_fit_results")
end


__cnt = 0
# invK = zeros(m,n)
function print_loss(l, invK, grad)
    global __cnt, __l, __K, __grad
    if mod(__cnt,1)==0
        println("\niter=$__iter, eval=$__cnt, current loss=",l)
        # println("a=$a, b1=$b1, b2=$b2")
    end
    __l = l
    __K = invK
    __grad = grad
    __cnt += 1
end

__iter = 0
function print_iter(rk)
    global __iter, __l, __K, __grad
    if mod(__iter,1)==0
        println("\n************* ITER=$__iter *************\n")
    end
    __iter += 1
    open("./flow_fit_results/loss.txt", "a") do io 
        writedlm(io, Any[__iter __l])
    end
    writedlm("./flow_fit_results/K$__iter.txt", __K)
    writedlm("./flow_fit_results/gradK$__iter.txt", __grad)
end
# u = readdlm("...") --> plot(u[:,2])

# config = tf.ConfigProto()
# config.intra_op_parallelism_threads = 24
# config.inter_op_parallelism_threads = 24
sess = Session(); init(sess);
opt = ScipyOptimizerInterface(J, var_list=[tfCtxInit.K], var_to_bounds=Dict(tfCtxInit.K=> (20.0, 100.0)), method="L-BFGS-B", 
    options=Dict("maxiter"=> 100, "ftol"=>1e-12, "gtol"=>1e-12))
ScipyOptimizerMinimize(sess, opt, loss_callback=print_loss, step_callback=print_iter, fetches=[J, tfCtxInit.K, gradK])


# function fn(rk) rk --> variables
# end 

# Adam, GradientDescent
# LBFGS, run(sess, ..) -> f, df
# opt = ScipyOptimizerInterface(J, var_list=[...], var_to_bounds=[...], step_callback=fn, method="L-BFGS-B",options=Dict("maxiter"=> 100, "ftol"=>1e-12, "gtol"=>1e-12))

