export FWI, FWIExample, compute_observation, plot, compute_misfit
@with_kw mutable struct FWI
    nz::Int64 = 134 
    nx::Int64 = 384
    dz::Float64 = 24.
    dx::Float64 = 24.
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
    mask_neg::Union{Missing, Array{Float64, 2}, PyObject} = missing
    ind_src_x::Union{Missing, Array{Int64, 1}} = missing
    ind_src_z::Union{Missing, Array{Int64, 1}} = missing
    ind_rec_x::Union{Missing, Array{Int64, 1}} = missing
    ind_rec_z::Union{Missing, Array{Int64, 1}} = missing
end

function FWI(nz::Int64, nx::Int64, dz::Float64, dx::Float64, nSteps::Int64, dt::Float64;
        ind_src_z::Array{Int64, 1}, ind_src_x::Array{Int64, 1}, ind_rec_z::Array{Int64, 1}, ind_rec_x::Array{Int64, 1},
        kwargs...)
    fwi = FWI(nx = nx, nz = nz, dx = dx, dz = dz, nSteps = nSteps, dt = dt; 
            ind_src_x = ind_src_x,
            ind_src_z = ind_src_z,
            ind_rec_x = ind_rec_x,
            ind_rec_z = ind_rec_z,
            kwargs...)
    Mask = zeros(fwi.nz_pad, fwi.nx_pad)
    Mask[fwi.nPml+1:fwi.nPml+nz, fwi.nPml+1:fwi.nPml+nx] .= 1.0
    Mask[fwi.nPml+1:fwi.nPml+10,:] .= 0.0
    fwi.mask = constant(Mask)
    fwi.mask_neg = 1 - constant(Mask)
    @assert length(ind_rec_x)==length(ind_rec_z)
    @assert length(ind_src_x)==length(ind_src_z)
    paraGen(fwi.nz_pad, fwi.nx_pad, dz, dx, nSteps, dt, fwi.f0, fwi.nPml, fwi.nPad, 
            joinpath(fwi.WORKSPACE,fwi.para_fname), 
            joinpath(fwi.WORKSPACE,fwi.survey_fname), 
            joinpath(fwi.WORKSPACE,fwi.data_dir_name))
    surveyGen(ind_src_z, ind_src_x, ind_rec_z, ind_rec_x, joinpath(fwi.WORKSPACE,fwi.survey_fname))
    return fwi
end

function PyPlot.:plot(fwi::FWI)
    close("all")
    x1, y1, x2, y2 = 0.0, 0.0, fwi.nx_pad*fwi.dx, fwi.nz_pad*fwi.dz
    plot(LinRange(x1, x2, 100), y1*ones(100), "k")
    plot(LinRange(x1, x2, 100), y2*ones(100), "k")
    plot(x1*ones(100), LinRange(y1, y2, 100), "k")
    plot(x2*ones(100), LinRange(y1, y2, 100), "k")
    
    x1, y1, x2, y2 = fwi.nPml * fwi.dx, fwi.nPml * fwi.dz, (fwi.nx_pad - fwi.nPml) * fwi.dx, (fwi.nz_pad - fwi.nPml) * fwi.dz
    plot(LinRange(x1, x2, 100), y1*ones(100), "g", label="PML Boundary")
    plot(LinRange(x1, x2, 100), y2*ones(100), "g")
    plot(x1*ones(100), LinRange(y1, y2, 100), "g")
    plot(x2*ones(100), LinRange(y1, y2, 100), "g")

    plot( (fwi.nPml .+ fwi.ind_rec_x .- 1) * fwi.dx, (fwi.nPml + fwi.nPad .+ fwi.ind_rec_z .- 1) * fwi.dz, "r^", label="Receiver", markersize=1)
    plot( (fwi.nPml .+ fwi.ind_src_x .- 1) * fwi.dx, (fwi.nPml + fwi.nPad .+ fwi.ind_src_z .- 1) * fwi.dz, "bv", label="Source", markersize=1)
    gca().invert_yaxis()
    xlabel("Distance")
    ylabel("Depth")
    legend()
    axis("equal")
end

function FWIExample()
    ind_src_x = collect(4:8:384)
    ind_src_z = 2ones(Int64, size(ind_src_x))
    ind_rec_x = collect(3:381)
    ind_rec_z = 2ones(Int64, size(ind_rec_x))
    FWI(134,384, 24., 24., 2000, 0.0025; 
        ind_src_x = ind_src_x,
        ind_src_z = ind_src_z,
        ind_rec_x = ind_rec_x,
        ind_rec_z = ind_rec_z)
end


@doc raw"""
    compute_observation(sess::PyObject, fwi::FWI, 
        cp::Union{Array{Float64}, PyObject}, 
        cs::Union{Array{Float64}, PyObject}, 
        ρ::Union{Array{Float64}, PyObject}, 
        stf_array::Union{Array{Float64}, PyObject},
        shot_ids::Array{<:Integer};
        gpu_id::Int64 = 0, is_padded::Bool = false)

Computes the observations using given parameters. Note that `shot_ids` are 1-based.
"""
function compute_observation(sess::PyObject, fwi::FWI, 
    cp::Union{Array{Float64}, PyObject}, 
    cs::Union{Array{Float64}, PyObject}, 
    ρ::Union{Array{Float64}, PyObject}, 
    stf_array::Union{Array{Float64}, PyObject},
    shot_ids::Array{<:Integer};
    gpu_id::Int64 = 0, is_padded::Bool = false)
    cp_pad, cs_pad, ρ_pad = cp, cs, ρ
    if !is_padded
        cp_pad, cs_pad, ρ_pad = padding(fwi, cp, cs, ρ)
    end
    stf_array = constant(stf_array)
    if length(size(stf_array))==1
        stf_array = repeat(stf_array', length(shot_ids), 1)
    end
    λ_pad, μ_pad = velocity_to_moduli(cp_pad, cs_pad, ρ_pad)
    shot_ids = shot_ids .- 1
    shot_ids_ = constant(shot_ids, dtype=Int32)
    data = fwi_obs_op(λ_pad, μ_pad, ρ_pad, stf_array, gpu_id, shot_ids_, joinpath(fwi.WORKSPACE, fwi.para_fname) )
    run(sess, data)
    data = zeros(length(shot_ids), fwi.nSteps, length(fwi.ind_rec_z))
    for i = 1:length(shot_ids)
        A = read("$(fwi.WORKSPACE)/Data/Shot$(shot_ids[i]).bin")
        data[i,:,:] = reshape(reinterpret(Float32,A),(fwi.nSteps ,length(fwi.ind_rec_z)))
    end
    data
end


@doc raw"""
    compute_misfit(fwi::FWI, 
        cp::Union{Array{Float64}, PyObject}, 
        cs::Union{Array{Float64}, PyObject}, 
        ρ::Union{Array{Float64}, PyObject},
        stf_array::Union{Array{Float64}, PyObject},
        shot_ids::Union{Array{Int64}, PyObject};
        gpu_id::Int64 = 0, is_padded::Bool = false, is_masked::Bool = false, 
        cp_ref::Union{Array{Float64}, PyObject}, 
        cs_ref::Union{Array{Float64}, PyObject}, 
        ρ_ref::Union{Array{Float64}, PyObject})

Computes the misfit function for the simulation parameters $c_p$, $c_s$, $\rho$, and source time functions `stf_array`

- If `is_padded` is false, `compute_misfit` will pad the inputs automatically. 
- If `is_masked` is false, `compute_misfit` will add the mask `fwi.mask` to all variables. 
- `gpu_id` is an integer in {0,1,2,...,#gpus-1}
- `shot_ids` is 1-based.
"""
function compute_misfit(fwi::FWI, 
    cp::Union{Array{Float64}, PyObject}, 
    cs::Union{Array{Float64}, PyObject}, 
    ρ::Union{Array{Float64}, PyObject},
    stf_array::Union{Array{Float64}, PyObject},
    shot_ids::Union{Array{Int64}, PyObject};
    gpu_id::Int64 = 0, is_padded::Bool = false, is_masked::Bool = false, 
    cp_ref::Union{Array{Float64}, PyObject}, 
    cs_ref::Union{Array{Float64}, PyObject}, 
    ρ_ref::Union{Array{Float64}, PyObject})

    cp_pad, cs_pad, ρ_pad = cp, cs, ρ
    if !is_padded
        cp_pad, cs_pad, ρ_pad = padding(fwi, cp, cs, ρ)
    end

    cp_masked, cs_masked,ρ_masked = cp_pad, cs_pad, ρ_pad
    if !is_masked
        cp_masked = cp_pad .* fwi.mask + cp_ref .* fwi.mask_neg
        cs_masked = cs_pad .* fwi.mask + cs_ref .* fwi.mask_neg
        ρ_masked = ρ_pad .* fwi.mask + ρ_ref .* fwi.mask_neg
    end
    λ_masked, μ_masked = velocity_to_moduli(cp_masked, cs_masked,ρ_masked)


    stf_array = constant(stf_array)
    if length(size(stf_array))==1
        stf_array = repeat(stf_array', length(shot_ids), 1)'
    end
    shot_ids = constant(shot_ids, dtype=Int32) - 1
    misfit = fwi_op(λ_masked, μ_masked, ρ_masked, stf_array, gpu_id, shot_ids, para_fname)
end

function padding(fwi::FWI, cp::Union{PyObject, Array{Float64,2}})
    cp = constant(cp)
    nz, nx, nPml, nPad = fwi.nz, fwi.nx, fwi.nPml, fwi.nPad
    nz_orig, nx_orig = size(cp)
    tran_cp = tf.reshape(cp, (1, nz_orig, nx_orig, 1))
    if nz_orig!=nz || nx_orig!=nx 
        @info "resizee image to required size"
        tran_cp = squeeze(tf.image.resize_bilinear(tran_cp, (nz, nx)))
    end
	cp_pad = tf.pad(cp, [nPml (nPml+nPad); nPml nPml], "SYMMETRIC")
	cp_pad = cast(cp_pad, Float64)
	return cp_pad
end

function padding(fwi::FWI, cp::Union{PyObject, Array{Float64,2}}, cq...)
    [padding(fwi, cp);vcat([padding(fwi, c) for c in cq]...)]
end