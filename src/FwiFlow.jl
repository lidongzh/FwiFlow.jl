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

    function ADCME.:Session(args...;kwargs...)
        config = tf.ConfigProto(
            device_count = {'GPU': 0}
        )
        sess = Session(config=config)
    end
    include("$(@__DIR__)/Ops/ops.jl")
    include("utils.jl")
    
end