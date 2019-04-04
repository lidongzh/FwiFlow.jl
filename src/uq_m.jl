include("ops.jl")
using JLD2
@pyimport sklearn.linear_model as skl
@pyimport statsmodels.api as sm
cd(@__DIR__)
ms = Array{Any}(undef, 4)
ms[1] = readdlm("data/m1.txt")
ms[2]= readdlm("data/m2.txt")
ms[3] =  readdlm("data/m3.txt")
ms[4] =  readdlm("data/m4.txt")
show_field(ms[1])
error("")

U = Array{Any}(undef, 5)
U[1] = readdlm("data/U1.txt") .- 3500
U[2] = readdlm("data/U2.txt") .- 3500
U[3] = readdlm("data/U3.txt") .- 3500
U[4] = readdlm("data/U4.txt") .- 3500
U[5] = readdlm("data/U5.txt") .- 3500
# result at iteration 2222
global a_ = Variable(0.9952938707725836)
global b1_ = Variable(1.0022777690029745)
global b2_ = Variable(0.9996851265125094)
global a = 100*a_*ones(m,n)
global b1 = 1.0*b1_*ones(m,n)
global b2 = -10.0*b2_*ones(m,n)

u = Array{Any}(undef, 3)
hatu = Array{Any}(undef, 3)
e = Array{Any}(undef, 3)
da = Array{Any}(undef, 3)
db1 = Array{Any}(undef, 3)
db2 = Array{Any}(undef, 3)

include("uq_helper.jl")

for i = 1:3
    # @show 
    if i==1
        u[i] = one_step(constant(ms[1]),  NT, 100*a_, b1_, -10.0*b2_)
        # u[i] = evolve_explicit(constant(ms[1]), NT, a, b1, b2)
        u[i] = tf.multiply(u[i],mask)
    else
        u[i] = one_step(u[i-1], NT, 100*a_, b1_, -10.0*b2_)
        # u[i] = evolve_explicit(u[i-1], NT, a, b1, b2)
        u[i] =  tf.multiply(u[i],mask)
    end
    e[i] = u[i] - ms[i+1]
    da[i] = gradients(u[i], a_)
    db1[i] = gradients(u[i], b1_)
    db2[i] = gradients(u[i], b2_)
end


sess = Session()
init(sess)


da_ = run(sess, da)
db1_ = run(sess, db1)
db2_ = run(sess, db2)
e_ = run(sess, e)
u_ = run(sess, u)
Idx = [40;65;100]

i = 1
σ2 = sum(e_[1][Idx[1], :].^2 + e_[2][Idx[1],:].^2) / 
    (
        sum(da_[1][Idx[1],:].^2 + db1_[1][Idx[1],:].^2 + db2_[1][Idx[1],:].^2) + 
        sum(da_[2][Idx[1],:].^2 + db1_[2][Idx[1],:].^2 + db2_[2][Idx[1],:].^2) 
    )
σ = sqrt(σ2)
enew = (abs.(da_[3][Idx[1],:]) + abs.(db1_[3][Idx[1],:]) + abs.(db2_[3][Idx[1],:]))*σ

u0_ = u_[3][Idx[1],:]
plot((0:n-1)*24, u0_,"r--")
plot((0:n-1)*24, ms[4][Idx[1],:],"b")
fill_between((0:n-1)*24, u0_-enew, u0_+enew, alpha=0.5, color="orange")

#= 
Ids__ = patch_ids(20, 40)
function patch_fit(Y)
    
    predict = zeros(m*n)
    lower_bnd = zeros(m*n)
    upper_bnd = zeros(m*n)
    # println(res_ols[:summary]())
    

    for Ids in Ids__
        trainI = [Ids;Ids .+ m*n]
        J = Xtrain[trainI,:]
        sigma = sum(ytrain[trainI].^2)/length(trainI)
        s = sqrt.(LinearAlgebra.diag(sigma * inv(J'*J + 1e-10*LinearAlgebra.I)))
        @show s
        u_ = abs.(Y[Ids,:]) * s
        lower_bnd[Ids] = -u_ 
        upper_bnd[Ids] = u_
    end
    return lower_bnd, upper_bnd
end


@pyimport numpy as np
function visualize(m_, p, l, u)
    x = 24*collect(0:m-1)
    y = 24*collect(0:n-1)
    X, Y = np.meshgrid(y,x)
    l = reshape(l, m, n)
    u = reshape(u, m, n)
    p = reshape(p, m, n)
    figure(figsize=[12,4])
    subplot(131)
    plot(y, m_[40,:])
    plot(y, p[40,:],"--")
    fill_between(y, (p+l)[40,:], (p+u)[40,:],alpha=0.5, color="orange")
    ylim(-150,150)
    title("depth = 960 m")
    xlabel("distance (m)")
    ylabel("u")

    subplot(132)
    plot(y, m_[45,:])
    plot(y, p[65,:],"--")
    fill_between(y, (p+l)[65,:], (p+u)[65,:],alpha=0.5, color="orange")
    ylim(-150,150)
    title("depth = 1560 m")
    xlabel("distance (m)")
    gca()[:get_yaxis]()[:set_ticks]([])

    subplot(133)
    plot(y, m_[100,:])
    plot(y, p[100,:],"--")
    fill_between(y, (p+l)[100,:], (p+u)[100,:],alpha=0.5, color="orange")
    ylim(-150,150)
    title("depth = 3960 m")
    xlabel("distance (m)")
    gca()[:get_yaxis]()[:set_ticks]([])
end

l_, u_ = patch_fit(Xtrain[m*n+1:end,:])
visualize(ms[2], run(sess, u[1]), l_, u_)


error("")

for i = 1:2
figure(figsize=[12,4])
subplot(131);pcolormesh(run(sess, u[i]), vmin=-150, vmax=150); colorbar(); title("couple")
subplot(132);pcolormesh(U[i+1], vmin=-150, vmax=150); colorbar(); title("exact")
subplot(133);pcolormesh( ms[i+1], vmin=-150, vmax=150); colorbar(); title("direct FWI")
end


figure(figsize=[8,4])
subplot(121);pcolormesh(run(sess, u[3]), vmin=-150, vmax=150); colorbar(); title("couple")
subplot(122);pcolormesh(U[4], vmin=-150, vmax=150); colorbar(); title("exact")

=#