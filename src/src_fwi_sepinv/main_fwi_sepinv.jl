function padding(lambda, den)
    tran_lambda = cast(lambda, Float64)
    tran_den = cast(den, Float64)
    lambda_pad = tf.pad(tran_lambda, [nPml (nPml+nPad); nPml nPml], constant_values=1.0/3.0*3500.0^2*2200.0/1e6)
    den_pad = tf.pad(tran_den, [nPml (nPml+nPad); nPml nPml], constant_values=2200.0)
    return lambda_pad, den_pad
end

# global loss

function fwi_sep(iSur)
  if !isdir("./$(args["version"])/FWI_stage$iSur/")
      mkdir("./$(args["version"])/FWI_stage$iSur/")
  end
  if iSur <= 2
    lambda_init = den .* (cp_nopad.^2 .- 2.0 .* cp_nopad.^2 ./3.0) /1e6
    den_init = den
  else
    lp = readdlm("./$(args["version"])/FWI_stage$(iSur-1)/loss.txt")
    Lp = Int64((lp[end,1]))
    lambda_init = readdlm("./$(args["version"])/FWI_stage$(iSur-1)/Lambda$Lp.txt")
    den_init = readdlm("./$(args["version"])/FWI_stage$(iSur-1)/Den$Lp.txt")
    # den_init = den
  end

  tf_lambda_inv = Variable(lambda_init, dtype=Float64)
  tf_den_inv = Variable(den_init, dtype=Float64)
  tf_lambda_inv_pad, tf_den_inv_pad = padding(tf_lambda_inv, tf_den_inv)

  shot_id_points = Int32.(trunc.(collect(LinRange(0, length(x_src)-1, nGpus+1))))
  shot_jump = 1
  # NOTE Compute FWI loss
  para_fname = "./$(args["version"])/para_file$iSur.json"

  loss = constant(0.0)
  for i = 1:nGpus
      # global loss
      tf_shot_ids = constant(collect(shot_id_points[i]:shot_jump:shot_id_points[i+1]), dtype=Int32)
      loss += fwi_op(tf_lambda_inv_pad, tf_shear_pad, tf_den_inv_pad, tf_stf, tf_gpu_id_array[i], tf_shot_ids0, para_fname)
  end
  gradLambda = gradients(loss, tf_lambda_inv)
  gradDen= gradients(loss, tf_den_inv)

  # loss = fwi_op(tf_lambda_inv_pad, tf_shear_pad, tf_den_inv_pad, tf_stf, tf_gpu_id_array[1], tf_shot_ids0, para_fname)



  # Optimization

  __cnt = 0
  __l = 0
  __Lambda = zeros(nz,nx)
  __gradLambda = zeros(nz,nx)
  __Den = zeros(nz,nx)
  __gradDen = zeros(nz,nx)
  # invK = zeros(m,n)
  function print_loss(l, Lambda, Den, gradLambda, gradDen)
      # global __l, __Lambda, __gradLambda, __Den, __gradDen
      if mod(__cnt,1)==0
          println("\niter=$__iter, eval=$__cnt, current loss=",l)
          # println("a=$a, b1=$b1, b2=$b2")
      end
      __cnt += 1
      __l = l
      __Lambda = Lambda
      __gradLambda = gradLambda
      __Den = Den
      __gradDen = gradDen
  end


  __iter = 0
  function print_iter(rk)
      # global __iter
      if mod(__iter,1)==0
          println("\n************* ITER=$__iter *************\n")
      end
      __iter += 1
      open("./$(args["version"])/FWI_stage$iSur/loss.txt", "a") do io
          println("\n outer_iter=$__iter, current loss=", __l)
          writedlm(io, Any[__iter __l])
      end
      open("./$(args["version"])/FWI_stage$iSur/Lambda$__iter.txt", "w") do io 
          writedlm(io, __Lambda)
      end
      open("./$(args["version"])/FWI_stage$iSur/gradLambda$__iter.txt", "w") do io
          writedlm(io, __gradLambda)
      end
      open("./$(args["version"])/FWI_stage$iSur/Den$__iter.txt", "w") do io 
          writedlm(io, __Den)
      end
      open("./$(args["version"])/FWI_stage$iSur/gradDen$__iter.txt", "w") do io
          writedlm(io, __gradDen)
      end
  end


  config = tf.ConfigProto()
  config.intra_op_parallelism_threads = 24
  config.inter_op_parallelism_threads = 24
  sess = Session(config=config); init(sess);

  lambda_lb = 5800.0
  lambda_ub = 9000.0
  opt = ScipyOptimizerInterface(loss, var_list=[tf_lambda_inv, tf_den_inv], var_to_bounds=Dict(tf_lambda_inv=> (lambda_lb, lambda_ub),tf_den_inv=> (2100.0, 2200.0)), method="L-BFGS-B", options=Dict("maxiter"=> 100, "ftol"=>1e-12, "gtol"=>1e-12))

# #   lambda_lb = (2500.0^2 - 2.0 * 3000.0^2/3.0) * 2500.0 /1e6
# #   lambda_ub = (4000.0^2 - 2.0 * 3000.0^2/3.0) * 2500.0 /1e6
#   lambda_lb = 5800.0
#   lambda_ub = 9000.0
#   opt = ScipyOptimizerInterface(loss, var_list=[tf_lambda_inv], var_to_bounds=Dict(tf_lambda_inv=> (lambda_lb, lambda_ub)), method="L-BFGS-B", options=Dict("maxiter"=> 100, "ftol"=>1e-12, "gtol"=>1e-12))
  @info "Optimization Starts..."
  ScipyOptimizerMinimize(sess, opt, loss_callback=print_loss, step_callback=print_iter, fetches=[loss,tf_lambda_inv,tf_den_inv,gradLambda,gradDen])
end