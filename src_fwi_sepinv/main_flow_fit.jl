include("args.jl")

function sw_p_to_lambda_den(sw, p)
    sw = tf.reshape(sw, (1, m, n, 1))
    p = tf.reshape(p, (1, m, n, 1))
    sw = tf.image.resize_bilinear(sw, (nz, nx))
    p = tf.image.resize_bilinear(p, (nz, nx))
    sw = cast(sw, Float64)
    p = cast(p, Float64)
    sw = squeeze(sw)
    p = squeeze(p)
    # tran_lambda, tran_den = Gassman(sw)
    # tran_lambda, tran_den = RockLinear(sw) # test linear relationship
    tran_lambda, tran_den = Patchy(sw)
    # tran_lambda_pad =  tf.pad(tran_lambda, [nPml (nPml+nPad); nPml nPml], constant_values=3500.0^2*2200.0/3.0) /1e6
    # tran_den_pad = tf.pad(tran_den, [nPml (nPml+nPad); nPml nPml], constant_values=2200.0)
    return tran_lambda, tran_den
end

lambdasObs = Array{PyObject}(undef, n_survey-1)
densObs = Array{PyObject}(undef, n_survey-1)
for iSur = 2:n_survey
  lp = readdlm("./$(args["version"])/FWI_stage$(iSur)/loss.txt")
  Lp = Int64((lp[end,1]))
  lambdasObs[iSur-1] = constant(readdlm("./$(args["version"])/FWI_stage$(iSur)/Lambda$Lp.txt"))
  densObs[iSur-1] = constant(readdlm("./$(args["version"])/FWI_stage$(iSur)/Den$Lp.txt"))
end

tfCtxInit = tfCtxGen(m,n,h,NT,Δt,Z,X,ρw,ρo,μw,μo,K_init,g,ϕ,qw,qo, sw0, false)
out_sw_init, out_p_init = imseq(tfCtxInit)
lambdas = Array{PyObject}(undef, n_survey-1)
dens = Array{PyObject}(undef, n_survey-1)
for i = 2:n_survey
    sw = out_sw_init[survey_indices[i]]
    p = out_p_init[survey_indices[i]]
    lambdas[i-1], dens[i-1] = sw_p_to_lambda_den(sw, p)
end

function objective_function(lambdasObs, lambdas, densObs, dens)
    # tf.nn.l2_loss(lambdasObs - lambdas) + tf.nn.l2_loss(densObs - dens)
    tf.nn.l2_loss(lambdasObs - lambdas)
    # tf.nn.l2_loss(densObs - dens)
end

J = objective_function(lambdasObs, lambdas, densObs, dens)
gradK = gradients(J, tfCtxInit.K)

if !isdir("./$(args["version"])/flow_fit_results")
    mkdir("./$(args["version"])/flow_fit_results")
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
    open("./$(args["version"])/flow_fit_results/loss.txt", "a") do io 
        writedlm(io, Any[__iter __l])
    end
    writedlm("./$(args["version"])/flow_fit_results/K$__iter.txt", __K)
    writedlm("./$(args["version"])/flow_fit_results/gradK$__iter.txt", __grad)
end
# u = readdlm("...") --> plot(u[:,2])

# config = tf.ConfigProto()
# config.intra_op_parallelism_threads = 24
# config.inter_op_parallelism_threads = 24
sess = Session(); init(sess);
opt = ScipyOptimizerInterface(J, var_list=[tfCtxInit.K], var_to_bounds=Dict(tfCtxInit.K=> (10.0, 130.0)), method="L-BFGS-B", 
    options=Dict("maxiter"=> 100, "ftol"=>1e-12, "gtol"=>1e-12))
ScipyOptimizerMinimize(sess, opt, loss_callback=print_loss, step_callback=print_iter, fetches=[J, tfCtxInit.K, gradK])
