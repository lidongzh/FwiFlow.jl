include("ops.jl")

##############################

global a = 90*Variable(ones(m, n))
global b1 = 0.8*Variable(ones(m, n))
global b2 = -8.0*Variable(ones(m, n))
global u = Variable(u0)
u = u.*mask


function scalar_function(u)
    us = Array{Any}(undef, 3)
    us[1] = u
    for i = 2:3
        @show i
        us[i] = evolve(us[i-1], 5)
    end

    J = constant(0.0)
    for i = 1:3
        # global J
        J += fwi(us[i]+3500, constant(i, dtype=Int64))
        # J += sum(tanh(us[i])) #DL
        # J += sum(us[i].*us[i])
    end
    return J
end

# v = constant(rand(134,384)*500) ## DL scale it by 500
# v0 = zeros(m, n)
# v0[35:m-35-1, 35:n-35-1] .= 1.0
v = constant( rand(m, n)*100 )
v = v.*mask
y = scalar_function(u)
dy = gradients(y, u)
nn = 10
ms = Array{Any}(undef, nn)
ys = Array{Any}(undef, nn)
s = Array{Any}(undef, nn)
w = Array{Any}(undef, nn)
gs =  @. (1/10)^(1:nn)

for i = 1:nn
    g = gs[i]
    ms[i] = u + g*v
    ys[i] = scalar_function(ms[i])
    s[i] = ys[i] - y 
    w[i] = s[i] - g*sum(v.*dy)
end

sess = Session()
init(sess)
sval = run(sess, s)
wval = run(sess, w)
loglog(gs, abs.(sval), "*-", label="finite difference")
loglog(gs, abs.(wval), "+-", label="automatic differentiation")
loglog(gs, gs.^2 * 0.5*abs(wval[1])/gs[1]^2, "--",label="\$\\mathcal{O}(\\gamma^2)\$")
loglog(gs, gs * 0.5*abs(sval[1])/gs[1], "--",label="\$\\mathcal{O}(\\gamma)\$")

plt[:gca]()[:invert_xaxis]()
legend()
xlabel("\$\\gamma\$")
ylabel("Error")
cd("../../../src")
savefig("./gradient_checks/couple_gradient_check.png")

writedlm("./gradient_checks/coupled_gs.txt", gs)
writedlm("./gradient_checks/coupled_sval.txt", sval)
writedlm("./gradient_checks/coupled_wval.txt", wval)

@pyimport matplotlib2tikz as mpl
mpl.save("./gradient_checks/coupled_check.tex")