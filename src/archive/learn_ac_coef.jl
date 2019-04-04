include("ops.jl")


cd("../Ops/FWI/ops")
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
##############################

# a = 100*Variable(ones(m, n))^2
# b1 = 1.0*Variable(ones(m, n))^2
# b2 = -10.0*Variable(ones(m, n))^2
# u = 500*Variable(u0)
a = 90*Variable(ones(m, n))^2
b1 = 0.8*Variable(ones(m, n))^2
b2 = -8.0*Variable(ones(m, n))^2
u = 500*constant(u0)
# u = Variable(zeros(m, n))

# while loop 
function evolve(uh, NT)
    ta = TensorArray(NT)
    function condition(i, ta)
        tf.less(i, NT+1)
    end
    function body(i, ta)
        uh = convection_diffusion(read(ta, i-1), a, b1, b2)
        i+1, write(ta, i, uh)
    end
    ta = write(ta, 1, uh)
    i = constant(2, dtype=Int32)
    _, out = while_loop(condition, body, [i;ta])
    read(out, NT)
end

us = Array{Any}(undef, 3)
us[1] = u
for i = 2:3
    @show i
    us[i] = evolve(us[i-1], 5)
end

J = constant(0.0)
for i = 1:3
    global J
    J += fwi(us[i]+3500, constant(i, dtype=Int64))
end

__cnt = 0
function print_loss(l, a, b1, b2, u)

    global __cnt
    if mod(__cnt,1)==0
        println("\nevaluation $__cnt, current loss=",l,"\n")
    end

    writedlm("../../../src/data/a_$__cnt.txt", a)
    writedlm("../../../src/data/b1_$__cnt.txt", b1)
    writedlm("../../../src/data/b2_$__cnt.txt", b2)
    writedlm("../../../src/data/u_$__cnt.txt", u)
    writedlm("../../../src/data/l_$__cnt.txt", l)

    # close("all")
    # imshow(a, vmin=80, vmax=120)
    # xlabel("x"); ylabel("y")
    # savefig("../../../src/figures/a$__cnt.png")
    # savefig("../../../src/figures/a$__cnt.pdf")

    # close("all")
    # imshow(b1, vmin=0.5, vmax=1.5)
    # xlabel("x"); ylabel("y")
    # savefig("../../../src/figures/b1$__cnt.png")
    # savefig("../../../src/figures/b1$__cnt.pdf")

    # close("all")
    # imshow(b2, vmin=-20, vmax=0)
    # xlabel("x"); ylabel("y")
    # savefig("../../../src/figures/b2$__cnt.png")
    # savefig("../../../src/figures/b2$__cnt.pdf")

    # close("all")
    # imshow(u, vmin=300, vmax=600)
    # xlabel("x"); ylabel("y")
    # savefig("../../../src/figures/u$__cnt.png")
    # savefig("../../../src/figures/u$__cnt.pdf")

    __cnt += 1
end

__iter = 0
function print_iter(rk)
    global __iter
    if mod(__iter,1)==0
        println("************* ITER=$__iter *************")
    end
    __iter += 1
end

opt = ScipyOptimizerInterface(J, method="L-BFGS-B",options=Dict("maxiter"=> 30, "ftol"=>1e-12, "gtol"=>1e-12))
sess = Session()
init(sess)


ScipyOptimizerMinimize(sess, opt, loss_callback=print_loss, step_callback=print_iter, fetches=[J, a, b1, b2, u])




