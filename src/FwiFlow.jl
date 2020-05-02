module FwiFlow
    using Conda
    using PyCall
    using Reexport
    @reexport using ADCME
    using LinearAlgebra
    using PyPlot
    using Random
    using JSON
    using DataStructures
    using Dierckx
    using Parameters
    using MAT


    export DATADIR
    DATADIR = "$(@__DIR__)/../docs/data"

    function ADCME.:Session(args...;kwargs...)
        if has_gpu()
            @info "GPU found. Use GPU."
            config = tf.ConfigProto(
                device_count = Dict("GPU"=> 0)
            )
            sess = tf.Session(config=config)
        else 
            @info "No GPU found. Use only CPU."
            return ADCME.Session(args...; kwargs...)
        end
    end
    include("Core.jl")
    include("Utils.jl")
    include("FWI.jl")
    
end