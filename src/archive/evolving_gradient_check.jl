include("ops.jl")

###### exact solution #######
m = 134
n = 384

A = zeros(m, n)
B1 = zeros(m, n)
B2 = zeros(m, n)
for i = 1:m 
    for j = 1:n 
        A[i,j] = ((i/m)^2+(j/n)^2)*0.2+1.0
        B1[i,j] = (2.0-(i/m)^2)/2.0
        B2[i,j] = (2.0-(j/n)^2)/2.0
    end
end
u0 = zeros(m, n)
u0[50:80,50:80] .= 1.0
u0[50:80,200:230] .= 1.0

for i = 1:m
    for j = n
        u0[i,j] = 100 * sin(2π*(i-1)/(m-1)) * sin(2π*(j-1)/(n-1))
    end
end
##############################

global a = 90*Variable(ones(m, n))^2
global b1 = 0.8*Variable(ones(m, n))^2
global b2 = -8.0*Variable(ones(m, n))^2
global u = 400*Variable(u0)

global u = Variable(u0)

# while loop 
function evolve(ah, NT)
    ta = TensorArray(NT)
    function condition(i, ta)
        tf.less(i, NT+1)
    end
    function body(i, ta)
        u = convection_diffusion(u, read(ta, i-1), b1, b2)
        i+1, write(ta, i, uh)
    end
    ta = write(ta, 1, ah)
    i = constant(2, dtype=Int32)
    _, out = while_loop(condition, body, [i;ta])
    read(out, NT)
end



function scalar_function(a)
    us = Array{Any}(undef, 11)
    as[1] = a
    for i = 2:11
        @show i
        us[i] = evolve(a, 5)
    end

    J = constant(0.0)
    for i = 1:11
        # global J
        # J += fwi(us[i]+3500, constant(i, dtype=Int64))
        J += sum(tanh(us[i])) #DL
    end
    return J
end

# v = constant(rand(134,384)*500) ## DL scale it by 500
v0 = zeros(m, n)
v0[32:m-32-1, 32:n-32-1] .= 1.0
v = constant((1. .+ 0.1*rand(m, n)) * 10 .* v0)
y = scalar_function(a)
dy = gradients(y, a)
as = Array{Any}(undef, 5)
ys = Array{Any}(undef, 5)
s = Array{Any}(undef, 5)
w = Array{Any}(undef, 5)
gs =  @. (1/10)^(1:5)

for i = 1:5
    g = gs[i]
    as[i] = u + g*v
    ys[i] = scalar_function(as[i])
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
println((sval-wval)./sval)
# writedlm("./gradient_checks/coupled_gs.txt", gs)
# writedlm("./gradient_checks/coupled_sval.txt", sval)
# writedlm("./gradient_checks/coupled_wval.txt", wval)

# @pyimport matplotlib2tikz as mpl
# mpl.save("./gradient_checks/coupled_check.tex")