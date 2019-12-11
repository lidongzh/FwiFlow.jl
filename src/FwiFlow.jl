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

    include("$(@__DIR__)/Ops/ops.jl")
    include("utils.jl")
    
end