#=
Main program for two phase flow inversion
=# 


include("args.jl")
include("main_fwi_sepinv.jl")

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
    tran_lambda_pad =  tf.pad(tran_lambda, [nPml (nPml+nPad); nPml nPml], constant_values=3500.0^2*2200.0/3.0) /1e6
    tran_den_pad = tf.pad(tran_den, [nPml (nPml+nPad); nPml nPml], constant_values=2200.0)
    return tran_lambda_pad, tran_den_pad
end

# NOTE Generate Data
if args["generate_data"]
    println("Generate Test Data...")
    K = 20.0 .* ones(m,n) # millidarcy
    K[8:10,:] .= 120.0
    # K[17:21,:] .= 100.0
    # for i = 1:m
    #     for j = 1:n
    #         if i <= (14 - 24)/(30 - 1)*(j-1) + 24 && i >= (12 - 18)/(30 - 1)*(j-1) + 18
    #             K[i,j] = 100.0
    #         end
    #     end
    # end
    tfCtxTrue = tfCtxGen(m,n,h,NT,Δt,Z,X,ρw,ρo,μw,μo,K,g,ϕ,qw,qo, sw0, true)
    out_sw_true, out_p_true = imseq(tfCtxTrue)
    lambdas = Array{PyObject}(undef, n_survey)
    dens = Array{PyObject}(undef, n_survey)
    for i = 1:n_survey
        sw = out_sw_true[survey_indices[i]]
        p = out_p_true[survey_indices[i]]
        lambdas[i], dens[i] = sw_p_to_lambda_den(sw, p)
    end

    misfit = Array{PyObject}(undef, n_survey)
    for i = 1:n_survey
        if !isdir("./$(args["version"])/Data$i")
            mkdir("./$(args["version"])/Data$i")
        end
        para_fname = "./$(args["version"])/para_file$i.json"
        survey_fname = "./$(args["version"])/survey_file$i.json"
        paraGen(nz_pad, nx_pad, dz, dx, nSteps, dt, f0, nPml, nPad, para_fname, survey_fname, "./$(args["version"])/Data$i/")
        # shot_inds = collect(1:3:length(z_src)) .+ mod(i-1,3) # 5src rotation
        # shot_inds = i # 1src rotation
        shot_inds = collect(1:length(z_src)) # all sources
        surveyGen(z_src[shot_inds], x_src[shot_inds], z_rec, x_rec, survey_fname)
        tf_shot_ids0 = constant(collect(0:length(shot_inds)-1), dtype=Int32)
        misfit[i] = fwi_obs_op(lambdas[i], tf_shear_pad, dens[i], tf_stf, tf_gpu_id0, tf_shot_ids0, para_fname)
    end
    config = tf.ConfigProto()
    config.intra_op_parallelism_threads = 24
    config.inter_op_parallelism_threads = 24
    sess = Session(config=config); init(sess);
    run(sess, misfit)
    error("Generate Data: Stop")
end

for iSur = 2:n_survey
    fwi_sep(iSur)
end
