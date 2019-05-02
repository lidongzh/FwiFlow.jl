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
# NOTE  Parameters
const ALPHA = 0.006323996017182
const SRC_CONST = 5.6146
const GRAV_CONST = 1.0/144.0
nPml = 32

m = 15
n = 30
scaling = (200-2nPml)/30
h = 100.0 # ft
# NT = 500
NT  = 50
dt_survey = 5
Δt = 20.0 # day
z = (1:m)*h|>collect
x = (1:n)*h|>collect
X, Z = np.meshgrid(x, z)
ρw = 62.238 # lbm/scf (pound/ft^3)
ρo = 40.0 # lbm/scf (pound/ft^3)
μw = 1.0 # centi poise
μo = 3.0
# K = 20.0 .* ones(m,n) # millidarcy
# K[8:10,:] .= 100.0
K_init = 20.0 .* ones(m,n)

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
survey_indices = collect(1:dt_survey:NT+1)
n_survey = length(survey_indices)

# NOTE Hyperparameter for fwi_op

# argsparse.jl
# ENV["CUDA_VISIBLE_DEVICES"] = 1
# ENV["PARAMDIR"] = "Src/params/"
# config = tf.ConfigProto(device_count = Dict("GPU"=>0))

# nz = 100
# nx = 200
dz = 3 # meters
dx = 3
nz = Int64(round(0.3048 * (m * h) / dz)) + 1
nx = Int64(round(0.3048 * (n * h) / dx)) + 1
nSteps = 2001
dt = 0.00025
f0 = 50.0
filter_para = [0, 0.1, 1000.0, 2000.0]
isAc = true
nPad = 32 - mod((nz+2*nPml), 32)
nz_pad = nz + 2*nPml + nPad
nx_pad = nx + 2*nPml

# reflection
# x_src = collect(5:20:nx-5)
# z_src = 5ones(Int64, size(x_src))
# x_rec = collect(5:1:nx-5)
# z_rec = 5 .* ones(Int64, size(x_rec))

# xwell
z_src = collect(5:20:nz-5)
x_src = 5ones(Int64, size(z_src))
z_rec = collect(5:1:nz-5)
x_rec = 5 .* ones(Int64, size(z_rec))

# para_fname = "./$(args["version"])/para_file.json"
survey_fname = "./$(args["version"])/survey_file.json"
# data_dir_name = "./$(args["version"])/Data"
# paraGen(nz, nx, dz, dx, nSteps, dt, f0, nPml, nPad, filter_para, isAc, para_fname, survey_fname, data_dir_name)
surveyGen(z_src, x_src, z_rec, x_rec, survey_fname)

cs = zeros(nz_pad, nx_pad)
den = 1000.0 .* ones(nz_pad, nx_pad)
cp_pad_value = 3000.0

# tf_cp = constant(cp)
tf_cs = constant(cs)
tf_den = constant(den)

# src = Matrix{Float64}(undef, 1, 2001)
# # src[1,:] = Float64.(reinterpret(Float32, read("../Ops/FWI/Src/params/ricker_10Hz.bin")))
# src[1,:] = Float64.(reinterpret(Float32, read("../Ops/FWI/Src/params/Mar_source_2001.bin")))
src = sourceGene(f0, nSteps, dt)
tf_stf = constant(repeat(src, outer=length(z_src)))
# tf_para_fname = tf.strings.join([para_fname])
tf_gpu_id0 = constant(0, dtype=Int32)
tf_gpu_id1 = constant(1, dtype=Int32)
nGpus = 2
tf_gpu_id_array = constant(collect(0:nGpus-1), dtype=Int32)
tf_shot_ids0 = constant(collect(Int32, 0:length(x_src)-1), dtype=Int32)
tf_shot_ids1 = constant(collect(Int32, 13:25), dtype=Int32)