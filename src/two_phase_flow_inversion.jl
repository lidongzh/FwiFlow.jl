# Main program for two phase flow inversion
include("args.jl")

function sw_p_to_cp(sw, p)
    sw = tf.reshape(sw, (1, m, n, 1))
    p = tf.reshape(p, (1, m, n, 1))
    sw = tf.image.resize_bilinear(sw, (200,200))
    p = tf.image.resize_bilinear(p, (200,200))
    sw = cast(sw, Float64)
    p = cast(p, Float64)
    sw = squeeze(sw)
    p = squeeze(p)
    # sw = tf.pad(sw, [nPml nPml;nPml nPml] )
    # p = tf.pad(p, [nPml nPml;nPml nPml])
    # println(p, sw)
    return (1.0 +0.2*sw)*3000
end

# NOTE Generate Data
if args["generate_data"]
    println("Generate Test Data...")
    K = 20.0 .* ones(m,n) # millidarcy
    K[8:10,:] .= 100.0
    tfCtxTrue = tfCtxGen(m,n,h,NT,Δt,Z,X,ρw,ρo,μw,μo,K,g,ϕ,qw,qo, sw0, true)
    out_sw_true, out_p_true = imseq(tfCtxTrue)
    cps = Array{PyObject}(undef, n_survey)
    for i = 1:n_survey
        sw = out_sw_true[survey_indices[i]-1]
        p = out_p_true[survey_indices[i]-1]
        cps[i] = sw_p_to_cp(sw, p)
    end

    res = Array{PyObject}(undef, n_survey)
    for i = 1:n_survey
        if !isdir("./$(args["version"])/Data$i")
            mkdir("./$(args["version"])/Data$i")
        end
        para_fname = "./$(args["version"])/para_file$i.json"
        paraGen(nz, nx, dz, dx, nSteps, dt, f0, nPml, nPad, filter_para, isAc, para_fname, survey_fname, "./$(args["version"])/Data$i/")
        surveyGen(z_src, x_src, z_rec, x_rec, survey_fname)
        res[i] = fwi_obs_op(cps[i], tf_cs, tf_den, tf_stf, tf_gpu_id0, tf_shot_ids0, para_fname)
    end
    sess = Session(); init(sess)
    run(sess, res)
    error("Generate Data: Stop")
end

tfCtxInit = tfCtxGen(m,n,h,NT,Δt,Z,X,ρw,ρo,μw,μo,K_init,g,ϕ,qw,qo, sw0, false)
out_sw_init, out_p_init = imseq(tfCtxInit)
cps = Array{PyObject}(undef, n_survey)
for i = 1:n_survey
    sw = out_sw_init[survey_indices[i]-1]
    p = out_p_init[survey_indices[i]-1]
    cps[i] = sw_p_to_cp(sw, p)
end

# NOTE Compute FWI loss
loss = constant(0.0)
for i = 1:n_survey
    global loss
    para_fname = "./$(args["version"])/para_file$i.json"
    loss += fwi_op(cps[i], tf_cs, tf_den, tf_stf, tf_gpu_id0, tf_shot_ids0, para_fname)
end

sess = Session(); init(sess)
if args["verbose"]
    println("Initial loss = ", run(sess, loss))
    g = gradients(loss, tfCtxInit.K)
    G = run(sess, g)
    pcolormesh(G); savefig("test.png"); close("all")
end

# Optimization
__cnt = 0
# invK = zeros(m,n)
function print_loss(l, K)
    global __cnt, __l, __K
    if mod(__cnt,1)==0
        println("\niter=$__iter, eval=$__cnt, current loss=",l)
        # println("a=$a, b1=$b1, b2=$b2")
    end
    __cnt += 1
    __l = l
    __K = K
end

__iter = 0
function print_iter(rk)
    global __iter, __l
    if mod(__iter,1)==0
        println("\n************* ITER=$__iter *************\n")
    end
    __iter += 1
    open("./$(args["version"])/loss.txt", "a") do io 
        writedlm(io, Any[__iter __l])
    end
    open("./$(args["version"])/K$__iter.txt", "w") do io 
        writedlm(io, __K)
    end
end

sess = Session(); init(sess);
opt = ScipyOptimizerInterface(loss, var_list=[tfCtxInit.K], var_to_bounds=Dict(tfCtxInit.K=> (20.0, 100.0)), method="L-BFGS-B", 
    options=Dict("maxiter"=> 100, "ftol"=>1e-12, "gtol"=>1e-12))
@info "Optimization Starts..."
ScipyOptimizerMinimize(sess, opt, loss_callback=print_loss, step_callback=print_iter, fetches=[loss,tfCtxInit.K])

