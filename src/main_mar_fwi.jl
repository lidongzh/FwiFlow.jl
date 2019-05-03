#=
Main program for FWI
=# 
using ArgParse 

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--generate_data"
            arg_type = Bool
            default = false
        "--version"
            arg_type = String
            default = "0000"
        "--verbose"
            arg_type = Bool
            default = false
    end

    return parse_args(s)
end

args = parse_commandline()
if !isdir("./$(args["version"])")
    mkdir("./$(args["version"])")
end

using PyTensorFlow
using PyCall
using LinearAlgebra
using PyPlot
using Random
Random.seed!(233)
include("ops_imseq.jl")
include("../Ops/FWI/fwi_util.jl")
include("fwi_util_op.jl")
np = pyimport("numpy")

nz = 134
nx = 384
dz = 24. # meters
dx = 24.
nSteps = 2001
dt = 0.0025
f0 = 4.5
filter_para = [0, 0.1, 100.0, 200.0]
isAc = true
nPml = 32
nPad = 32 - mod((nz+2*nPml), 32)
nz_pad = nz + 2*nPml + nPad
nx_pad = nx + 2*nPml

# reflection
x_src = collect(4:8:384)
z_src = 2ones(Int64, size(x_src))
x_rec = collect(3:381)
z_rec = 2ones(Int64, size(x_rec))

# xwell
# z_src = collect(5:10:nz-5) #14->11srcs 10->15srcs
# x_src = 5ones(Int64, size(z_src))
# z_rec = collect(5:1:nz-5)
# x_rec = (nx-5) .* ones(Int64, size(z_rec))

para_fname = "./$(args["version"])/para_file.json"
survey_fname = "./$(args["version"])/survey_file.json"
data_dir_name = "./$(args["version"])/Data"
paraGen(nz_pad, nx_pad, dz, dx, nSteps, dt, f0, nPml, nPad, filter_para, isAc, para_fname, survey_fname, data_dir_name)
surveyGen(z_src, x_src, z_rec, x_rec, survey_fname)

tf_cp = constant(reshape(reinterpret(Float32,read("Mar_models/Model_Cp_true.bin")),(nz_pad, nx_pad)), dtype=Float64)
cs = zeros(nz_pad, nx_pad)
den = 1000.0 .* ones(nz_pad, nx_pad)
cp_pad_value = 3000.0

# tf_cp = constant(cp)
tf_cs = constant(cs)
tf_den = constant(den)

src = Matrix{Float64}(undef, 1, 2001)
# # src[1,:] = Float64.(reinterpret(Float32, read("../Ops/FWI/Src/params/ricker_10Hz.bin")))
src[1,:] = Float64.(reinterpret(Float32, read("../Ops/FWI/Src/params/Mar_source_2001.bin")))
# src = sourceGene(f0, nSteps, dt)
tf_stf = constant(repeat(src, outer=length(z_src)))
# tf_para_fname = tf.strings.join([para_fname])
tf_gpu_id0 = constant(0, dtype=Int32)
tf_gpu_id1 = constant(1, dtype=Int32)
nGpus = 2
tf_gpu_id_array = constant(collect(0:nGpus-1), dtype=Int32)
tf_shot_ids0 = constant(collect(Int32, 0:length(x_src)-1), dtype=Int32)
shot_id_points = Int32.(trunc.(collect(LinRange(0, length(z_src)-1, nGpus+1))))


function pad_cp(cp)
    tran_cp = cast(cp, Float64)
    return tf.pad(tran_cp, [nPml (nPml+nPad); nPml nPml], constant_values=3000.0)
end

# NOTE Generate Data
if args["generate_data"]
    println("Generate Test Data...")

    if !isdir("./$(args["version"])/Data")
        mkdir("./$(args["version"])/Data")
    end

    res = fwi_obs_op(tf_cp, tf_cs, tf_den, tf_stf, tf_gpu_id0, tf_shot_ids0, para_fname)

    config = tf.ConfigProto()
    config.intra_op_parallelism_threads = 24
    config.inter_op_parallelism_threads = 24
    sess = Session(config=config); init(sess);
    run(sess, res)
    error("Generate Data: Stop")
end

cp_init = reshape(reinterpret(Float32,read("Mar_models/Model_Cp_init_1D.bin")),(nz_pad, nx_pad))
tf_cp_inv = Variable(cp_init, dtype=Float64)

Mask = ones(nz_pad, nx_pad)
Mask[nPml+1:nPml+10,:] .= 0.0
tf_cp_inv_msk = tf_cp_inv .* constant(Mask) + constant(cp_init[1,1] .* (1. .- Mask))

# NOTE Compute FWI loss
loss = constant(0.0)
for i = 1:nGpus
    global loss
    tf_shot_ids = constant(collect(shot_id_points[i] : shot_id_points[i+1]), dtype=Int32)
    loss += fwi_op(tf_cp_inv_msk, tf_cs, tf_den, tf_stf, tf_gpu_id_array[i], tf_shot_ids, para_fname)
end
gradCp = gradients(loss, tf_cp_inv)


if args["verbose"]
    sess = Session(); init(sess)
    println("Initial loss = ", run(sess, loss))
    g = gradients(loss, tfCtxInit.K)
    G = run(sess, g)
    pcolormesh(G); savefig("test.png"); close("all")
end

# Optimization
__cnt = 0
# invK = zeros(m,n)
function print_loss(l, Cp, gradCp)
    global __cnt, __l, __Cp, __gradCp
    if mod(__cnt,1)==0
        println("\niter=$__iter, eval=$__cnt, current loss=",l)
        # println("a=$a, b1=$b1, b2=$b2")
    end
    __cnt += 1
    __l = l
    __Cp = Cp
    __gradCp = gradCp
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
    open("./$(args["version"])/Cp$__iter.txt", "w") do io 
        writedlm(io, __Cp)
    end
    open("./$(args["version"])/gradCp$__iter.txt", "w") do io 
        writedlm(io, __gradCp)
    end
end

config = tf.ConfigProto()
config.intra_op_parallelism_threads = 24
config.inter_op_parallelism_threads = 24
sess = Session(config=config); init(sess);
# cp_low_bd = 1500. .* ones(nz_pad, nx_pad)
# cp_high_bd = 5500. .* ones(nz_pad, nx_pad)
# cp_high_bd[nPml+1:nPml+10,:] .= 1500.0
opt = ScipyOptimizerInterface(loss, var_list=[tf_cp_inv], var_to_bounds=Dict(tf_cp_inv=> (1500.0, 5500.0)), method="L-BFGS-B", 
    options=Dict("maxiter"=> 100, "ftol"=>1e-6, "gtol"=>1e-6))
@info "Optimization Starts..."
ScipyOptimizerMinimize(sess, opt, loss_callback=print_loss, step_callback=print_iter, fetches=[loss,tf_cp_inv,gradCp])

