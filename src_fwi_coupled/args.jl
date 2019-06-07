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
# const ALPHA = 0.006323996017182
# const SRC_CONST = 5.6146
# const GRAV_CONST = 1.0/144.0
const ALPHA = 1.0
const SRC_CONST = 86400.0
const GRAV_CONST = 1.0

# NOTE Hyperparameter for flow simulation
m = 15
n = 30
h = 30.0 # meter

NT  = 50
dt_survey = 5
Δt = 20.0 # day
z = (1:m)*h|>collect
x = (1:n)*h|>collect
X, Z = np.meshgrid(x, z)

# ρw = 996.9571
# ρo = 640.7385
# μw = 1.0
# μo = 3.0
ρw = 501.9
ρo = 1053.0
μw = 0.1
μo = 1.0

K_init = 20.0 .* ones(m,n)

g = 9.8*GRAV_CONST
ϕ = 0.25 .* ones(m,n)
qw = zeros(NT, m, n)
qw[:,9,3] .= 0.005 * (1/h^2)/10.0 * SRC_CONST
qo = zeros(NT, m, n)
qo[:,9,28] .= -0.005 * (1/h^2)/10.0 * SRC_CONST
sw0 = zeros(m, n)
survey_indices = collect(1:dt_survey:NT+1) # 10 stages
n_survey = length(survey_indices)

# NOTE Hyperparameter for fwi_op

# argsparse.jl
# ENV["CUDA_VISIBLE_DEVICES"] = 1
# ENV["PARAMDIR"] = "Src/params/"
# config = tf.ConfigProto(device_count = Dict("GPU"=>0))

dz = 3 # meters
dx = 3
nz = Int64(round((m * h) / dz)) + 1
nx = Int64(round((n * h) / dx)) + 1
nPml = 32
nSteps = 3001
dt = 0.00025
f0 = 50.0
nPad = 32 - mod((nz+2*nPml), 32)
nz_pad = nz + 2*nPml + nPad
nx_pad = nx + 2*nPml

# reflection
# x_src = collect(5:20:nx-5)
# z_src = 5ones(Int64, size(x_src))
# x_rec = collect(5:1:nx-5)
# z_rec = 5 .* ones(Int64, size(x_rec))

# xwell
# # z_src = collect(5:10:nz-5) #14->11srcs 10->15srcs
# # z_src = collect(5:10:nz-5)
z_src = collect(5:10:nz-5)
x_src = 5ones(Int64, size(z_src))
z_rec = collect(5:1:nz-5)
x_rec = (nx-5) .* ones(Int64, size(z_rec))

# para_fname = "./$(args["version"])/para_file.json"
# survey_fname = "./$(args["version"])/survey_file.json"
# data_dir_name = "./$(args["version"])/Data"
# paraGen(nz, nx, dz, dx, nSteps, dt, f0, nPml, nPad, filter_para, isAc, para_fname, survey_fname, data_dir_name)
# surveyGen(z_src, x_src, z_rec, x_rec, survey_fname)

cp_nopad = 3500.0 .* ones(nz, nx) # initial cp
cs = cp_nopad ./ sqrt(3.0)
den = 2200.0 .* ones(nz, nx)
cp_pad = 3500.0 .* ones(nz_pad, nx_pad) # initial cp
cs_pad = cp_pad ./ sqrt(3.0)
den_pad = 2200.0 .* ones(nz_pad, nx_pad)
cp_pad_value = 3500.0

# tf_cp = constant(cp)
tf_cs = constant(cs_pad)
tf_den = constant(den_pad)

# src = Matrix{Float64}(undef, 1, 2001)
# # src[1,:] = Float64.(reinterpret(Float32, read("../Ops/FWI/Src/params/ricker_10Hz.bin")))
# src[1,:] = Float64.(reinterpret(Float32, read("../Ops/FWI/Src/params/Mar_source_2001.bin")))
src = sourceGene(f0, nSteps, dt)
tf_stf = constant(repeat(src, outer=length(z_src)))
# tf_para_fname = tf.strings.join([para_fname])
tf_gpu_id0 = constant(0, dtype=Int32)
tf_gpu_id1 = constant(1, dtype=Int32)
nGpus = 3
# tf_gpu_id_array = constant(collect(0:nGpus-1), dtype=Int32)
tf_gpu_id_array = constant([0,1,2], dtype=Int32)
tf_shot_ids0 = constant(collect(Int32, 0:length(x_src)-1), dtype=Int32)
tf_shot_ids1 = constant(collect(Int32, 13:25), dtype=Int32)

# NOTE Hyperparameter for rock physics
tf_bulk_fl1 = constant(2.735e9)
tf_bulk_fl2 = constant(0.125e9) # to displace fl1
tf_bulk_sat1 = constant(den .* (cp_nopad.^2 .- 4.0/3.0 .* cp_nopad.^2 ./3.0)) # vp/vs ratio as sqrt(3)
tf_bulk_min = constant(36.6e9)
tf_shear_sat1 = constant(den .* cp_nopad.^2 ./3.0)
tf_ϕ_pad = tf.image.resize_bilinear(tf.reshape(constant(ϕ), (1, m, n, 1)), (nz, nx)) # upsample the porosity
tf_ϕ_pad = cast(tf_ϕ_pad, Float64)
tf_ϕ_pad = squeeze(tf_ϕ_pad)
tf_shear_pad = tf.pad(tf_shear_sat1, [nPml (nPml+nPad); nPml nPml], 
    constant_values=den[1,1] * cp_nopad[1,1]^2 /3.0) / 1e6

function Gassman(sw)
    tf_bulk_fl_mix = 1.0/( (1-sw)/tf_bulk_fl1 + sw/tf_bulk_fl2 )
    temp = tf_bulk_sat1/(tf_bulk_min - tf_bulk_sat1) - tf_bulk_fl1/tf_ϕ_pad /(tf_bulk_min - tf_bulk_fl1) + tf_bulk_fl_mix/tf_ϕ_pad /(tf_bulk_min - tf_bulk_fl_mix)

    tf_bulk_new = tf_bulk_min / (1.0/temp + 1.0)
    # tf_den_new = constant(den) + tf_ϕ_pad .* sw * (ρw - ρo) *16.018463373960138;
    tf_den_new = constant(den) + tf_ϕ_pad .* sw * (ρw - ρo)
    # tf_cp_new = sqrt((tf_bulk_new + 4.0/3.0 * tf_shear_sat1)/tf_den_new)
    tf_lambda_new = tf_bulk_new - 2.0/3.0 * tf_shear_sat1
    return tf_lambda_new, tf_den_new
end

function RockLinear(sw)
    # tf_lambda_new = constant(7500.0*1e6 .* ones(nz,nx)) + (17400.0-7500.0)*1e6 * sw
    tf_lambda_new = constant(7500.0*1e6 .* ones(nz,nx)) + (9200.0-7500.0)*1e6 * sw
    tf_den_new = constant(den) + tf_ϕ_pad .* sw * (ρw - ρo)
    return tf_lambda_new, tf_den_new
end

tf_patch_temp = tf_bulk_sat1/(tf_bulk_min - tf_bulk_sat1) - 
    tf_bulk_fl1/tf_ϕ_pad /(tf_bulk_min - tf_bulk_fl1) + 
    tf_bulk_fl2/tf_ϕ_pad /(tf_bulk_min - tf_bulk_fl2)
tf_bulk_sat2 = tf_bulk_min/(1.0/tf_patch_temp + 1.0)
function Patchy(sw)
    tf_bulk_new = 1/( (1-sw)/(tf_bulk_sat1+4.0/3.0*tf_shear_sat1) 
            + sw/(tf_bulk_sat2+4.0/3.0*tf_shear_sat1) ) - 4.0/3.0*tf_shear_sat1
    tf_lambda_new = tf_bulk_new - 2.0/3.0 * tf_shear_sat1
    tf_den_new = constant(den) + tf_ϕ_pad .* sw * (ρw - ρo)
    return tf_lambda_new, tf_den_new
end

