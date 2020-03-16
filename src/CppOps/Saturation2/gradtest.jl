using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
using FwiFlow
# include("../ops.jl")
# Random.seed!(233)

function saturation(s0,dporodt,pt,perm,poro,qw,qo,muw,muo,sref,dt,h)
    saturation_ = load_op_and_grad("./build/libSaturation","saturation")
    s0,dporodt,pt,perm,poro,qw,qo,muw,muo,sref,dt,h = convert_to_tensor([s0,dporodt,pt,perm,poro,qw,qo,muw,muo,sref,dt,h], [Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64])
    saturation_(s0,dporodt,pt,perm,poro,qw,qo,muw,muo,sref,dt,h)
end

# TODO: specify your input parameters
m = 100
n = 10
h = 0.01
x = zeros(n, m)
y = zeros(n, m)
for i = 1:m 
    for j = 1:n 
        x[j, i] = h*i 
        y[j, i] = h*j
    end
end
t = 3.0 
s0 = @. (x^2 + y^2)/(1+x^2+y^2) * exp(-t)
dporodt = exp(t) * zeros(n, m)
pt = @. (x^2+y^2)
perm = rand(n, m)
poro = exp(t) * ones(n, m)
qw = ones(n, m)
qo = ones(n, m)
muw = 2.0 
muo = 3.0 
sref = s0 
dt = 0.01
h = 0.1 
u = sat_op2(s0,dporodt,pt,perm,poro,qw,qo,muw,muo,sref,dt,h)
u3 = sat_op(s0,pt,perm,poro,qw,qo,muw, muo,sref,dt,h)

# u3 = sat_op(s0,dporodt,pt,perm,poro,qw,qo,muw,muo,sref,dt,h)
sess = Session(); init(sess)
# @show run(sess, u)

@show run(sess, u3-u)

# uncomment it for testing gradients
# error() 


# TODO: change your test parameter to `m`
#       in the case of `multiple=true`, you also need to specify which component you are testings
# gradient check -- v
# s0 qw qo
function scalar_function(m)
    # return sum(saturation(s0,dporodt,pt,perm,poro,qw,qo,muw,muo,sref,dt,h)^2)
    return sum(sat_op2(m,dporodt,pt,perm,poro,qw,qo,muw,muo,sref,dt,h)^2)
    # return sum(sat_op(m,pt,perm,poro,qw,qo,muw,muo,sref,dt,h)^2)
end

# TODO: change `m_` and `v_` to appropriate values
m_ = constant(0.7*rand(n, m))
v_ = rand(n, m)

# m_ = constant(s0)
# v_ = rand(n, m)

y_ = scalar_function(m_)
dy_ = gradients(y_, m_)
ms_ = Array{Any}(undef, 5)
ys_ = Array{Any}(undef, 5)
s_ = Array{Any}(undef, 5)
w_ = Array{Any}(undef, 5)
gs_ =  @. 1 / 10^(1:5)

for i = 1:5
    g_ = gs_[i]
    ms_[i] = m_ + g_*v_
    ys_[i] = scalar_function(ms_[i])
    s_[i] = ys_[i] - y_
    w_[i] = s_[i] - g_*sum(v_.*dy_)
end

sess = Session(); init(sess)
sval_ = run(sess, s_)
wval_ = run(sess, w_)
close("all")
loglog(gs_, abs.(sval_), "*-", label="finite difference")
loglog(gs_, abs.(wval_), "+-", label="automatic differentiation")
loglog(gs_, gs_.^2 * 0.5*abs(wval_[1])/gs_[1]^2, "--",label="\$\\mathcal{O}(\\gamma^2)\$")
loglog(gs_, gs_ * 0.5*abs(sval_[1])/gs_[1], "--",label="\$\\mathcal{O}(\\gamma)\$")

plt.gca().invert_xaxis()
legend()
xlabel("\$\\gamma\$")
ylabel("Error")
