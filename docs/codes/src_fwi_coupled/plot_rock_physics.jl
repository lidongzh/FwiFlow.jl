using PyPlot
include("args.jl")

Sw = constant(collect(0:0.001:1))

lambda_brie_3 = Brie(Sw)