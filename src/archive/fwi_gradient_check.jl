include("ops.jl")

using PyTensorFlow
using PyCall
using PyPlot
using DelimitedFiles

function scalar_function(m)
    # print(m)
    v = fwi(m, 3)
    # return sum(convection_diffusion(m))
end

C = rand(20,20)
# m = constant(500*rand(134,384).+3000)
# v = constant(rand(134,384).*20)

v0 = zeros(134, 384)
# v0[50:80,150:180] .= 1.0
# v0[90:120,200:230] .= 1.0


nz = 134
nx = 384
v0[33:nz-33-1, 33:nx-33-1] .= 1.0
A = zeros(nz, nx)
B1 = zeros(nz, nx)
B2 = zeros(nz, nx)
u0 = zeros(nz, nx)
for i = 1:nz
    for j = 1:nx
        # A[i,j] = 1.0 + 0.1*exp(-0.001*((i-60)^2))
        # B1[i,j] = 1.0 + 0.1*sin(2π/m*i)
        # B2[i,j] = 1.0 + 0.1*cos(2π/n*j)
        # u0[i,j] = exp(-0.001*((j-200)^2+(i-60)^2))
        A[i,j] = ((i/nz)^2+(j/nx)^2)*0.2+1.0
        B1[i,j] = (2.0-(i/nz)^2)/2.0
        B2[i,j] = (2.0-(j/nx)^2)/2.0
    end
end
# u0[50:80, 50:80] .= 1.0
# u0[50:80, 200:230] .= 1.0
for i = 1:nz
    for j = nx
        u0[i,j] = 100 * sin(2π*(i-1)/(nz-1)) * sin(2π*(j-1)/(nx-1))
    end
end


# m = constant(3500*ones(134,384))
# m = constant(3500. .+ u0)
m = constant(3500*ones(134, 384))
# v = constant((1. .+ 0.1*rand(nz, nx)) * 500 .* v0)
v = constant((1. .+ 0.1*rand(nz, nx)) * 300 .* v0)

# v = constant(500*(v0))
y = scalar_function(m)
dy = gradients(y, m)
ms = Array{Any}(undef, 20)
ys = Array{Any}(undef,20)
s = Array{Any}(undef, 20)
w = Array{Any}(undef, 20)
gs =  @. (1/10)^(1:20)
# gs =  @. 1 / 20^(5)

for i = 1:20
    g = gs[i]
    ms[i] = m + g*v
    # ms[i] = m
    ys[i] = scalar_function(ms[i])
    s[i] = (ys[i] - y)
    w[i] = s[i] - g*sum(v.*dy)
end

sess = Session()
init(sess)
# run(sess, ys)
sval = run(sess, s)
wval = run(sess, w)
close("all")
loglog(gs, abs.(sval), "*-", label="finite difference")
loglog(gs, abs.(wval), "+-", label="automatic differentiation")
loglog(gs, gs.^2 * 0.5*abs(wval[1])/gs[1]^2, "--",label="\$\\mathcal{O}(\\gamma^2)\$")
loglog(gs, gs* 0.5*abs(sval[1])/gs[1], "--",label="\$\\mathcal{O}(\\gamma)\$")

# loglog(gs, abs.(sval), "*-", label="finite difference")
# loglog(gs, abs.(wval), "+-", label="automatic differentiation")
# loglog(gs, gs.^2 * 0.5*abs(wval[1])/gs[1]^2, "--",label="\$\\mathcal{O}(\\gamma^2)\$")
# loglog(gs, gs * 0.5*abs(sval[1])/gs[1], "--",label="\$\\mathcal{O}(\\gamma)\$")
# loglog(gs, abs.(sval-wval), "*-", label="sval-wval")

# loglog(gs, abs.(sval-wval) ./ abs.(sval), "--", label="sval/wval")

plt[:gca]()[:invert_xaxis]()
legend()
xlabel("\$\\gamma\$")
ylabel("Error")
cd("../../../src")
savefig("./gradient_checks/fwi_gradient_check.png")

println((sval-wval)./sval)


writedlm("./gradient_checks/fwi_gs.txt", gs)
writedlm("./gradient_checks/fwi_sval.txt", sval)
writedlm("./gradient_checks/fwi_wval.txt", wval)

@pyimport matplotlib2tikz as mpl
mpl.save("./gradient_checks/fwi_check.tex")