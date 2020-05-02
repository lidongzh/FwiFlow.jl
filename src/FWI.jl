export FWI, FWIExample, compute_observation
@with_kw mutable struct FWI
    nx::Int64 = 384
    nz::Int64 = 134 
    dx::Float64 = 24.
    dz::Float64 = 24.
    dt::Float64 = 0.0025
    nSteps::Int64 = 2000
    f0::Float64 = 4.5
    nPml::Int64 = 32
    nPad::Int64 = 32 - mod((nz+2*nPml), 32)
    nz_pad::Int64 = nz + 2*nPml + nPad
    nx_pad::Int64 = nx + 2*nPml
    para_fname::String = "para_file.json"
    survey_fname::String = "survey_file.json"
    data_dir_name::String = "Data"
    WORKSPACE::String = mktempdir()
    mask::Union{Missing, Array{Float64, 2}, PyObject} = missing
    neg_mask::Union{Missing, Array{Float64, 2}, PyObject} = missing
    ind_src_x::Union{Missing, Array{Int64, 1}} = missing
    ind_src_z::Union{Missing, Array{Int64, 1}} = missing
    ind_rec_x::Union{Missing, Array{Int64, 1}} = missing
    ind_rec_z::Union{Missing, Array{Int64, 1}} = missing
end

function FWI(nx::Int64, nz::Int64, dx::Float64, dz::Float64, nSteps::Int64, dt::Float64,
        stf_load::Union{Array{Float64}, PyObject}; kwargs...)
    fwi = FWI(nx = nx, nz = nz, dx = dx, dz = dx, nSteps = nSteps, dt = dt; kwargs...)
    Mask = zeros(fwi.nz_pad, fwi.nx_pad)
    Mask[fwi.nPml+1:fwi.nPml+nz, fwi.nPml+1:fwi.nPml+nx] .= 1.0
    Mask[fwi.nPml+1:fwi.nPml+10,:] .= 0.0
    fwi.mask = constant(Mask)
    fwi.neg_mask = 1 - constant(Mask)
    @assert size(stf_load) == (nSteps,)
    tf_stf_array = repeat(constant(stf_load)', length(fwi.ind_src_z), 1)
    paraGen(fwi.nz_pad, fwi.nx_pad, dz, dx, nSteps, dt, fwi.f0, fwi.nPml, fwi.nPad, 
            joinpath(fwi.WORKSPACE,fwi.para_fname), 
            joinpath(fwi.WORKSPACE,fwi.survey_fname), 
            joinpath(fwi.WORKSPACE,fwi.data_dir_name))
    surveyGen(fwi.ind_src_z, fwi.ind_src_x, fwi.ind_rec_z, fwi.ind_rec_x, joinpath(fwi.WORKSPACE,fwi.survey_fname))
    return fwi
end

function FWIExample()
    ind_src_x = collect(4:8:384)
    ind_src_z = 2ones(Int64, size(ind_src_x))
    ind_rec_x = collect(3:381)
    ind_rec_z = 2ones(Int64, size(ind_rec_x))
    data = matread("$(DATADIR)/sourceF_4p5_2_high.mat")["sourceF"][:]
    FWI(384, 134, 24., 24., 2000, 0.0025, data; 
        ind_src_x = ind_src_x,
        ind_src_z = ind_src_z,
        ind_rec_x = ind_rec_x,
        ind_rec_z = ind_rec_z)
end


function compute_observation(fwi::FWI, 
    λ_pad::Union{Array{Float64}, PyObject}, 
    μ_pad::Union{Array{Float64}, PyObject}, 
    ρ_pad::Union{Array{Float64}, PyObject}, 
    stf_array::Union{Array{Float64}, PyObject},
    shot_ids::Union{Array{Int64}, PyObject},
    gpu_id::Int64 = 0)
    stf_array = constant(stf_array)
    if length(size(stf_array))==1
        stf_array = repeat(stf_array', length(shot_ids), 1)
    end
    data = fwi_obs_op(λ_pad, μ_pad, ρ_pad, stf_array, gpu_id, shot_ids, fwi.para_fname) 
    data = reshape(data, (fwi.nSteps, length(ind_rec_z)))
end

function compute_misfit(fwi::FWI, 
    λ_pad::Union{Array{Float64}, PyObject}, 
    μ_pad::Union{Array{Float64}, PyObject}, 
    ρ_pad::Union{Array{Float64}, PyObject}, 
    stf_array::Union{Array{Float64}, PyObject},
    shot_ids::Union{Array{Int64}, PyObject},
    gpu_id::Int64 = 0)
    stf_array = constant(stf_array)
    if length(size(stf_array))==1
        stf_array = repeat(stf_array', length(tf_shot_ids), 1)
    end
    misfit = fwi_op(λ_pad, μ_pad, ρ_pad, stf_array, gpu_id, shot_ids, para_fname)
end

function padding(fwi::FWI)
end