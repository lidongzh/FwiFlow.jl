include("ops.jl")

do_test = false


if do_test
    u = Variable(u0)
    global a = 100*Variable(ones(m,n))
    global b1 = 1.0*Variable(ones(m, n))
    global b2 = -10.0*Variable(ones(m,n))
    global u = u.*mask
else
    global a_ = Variable(1.2)
    global b1_ = Variable(0.8)
    global b2_ = Variable(0.1)

    global a = 100*a_*ones(m,n)
    global b1 = 1.0*b1_*ones(m,n)
    global b2 = -10.0*b2_*ones(m,n)
    global u = Variable(zeros(m,n))
    u = u.*mask
end

nsrc = 5
us = Array{Any}(undef, nsrc)
us[1] = u
for i = 2:nsrc
    @show i
    us[i] = evolve(us[i-1], 5)
end


J = constant(0.0)
for i = 1:3
    global J
    J += fwi(us[i]+3500, constant(i, dtype=Int64))
end

if do_test
    sess = Session()
    init(sess)
    println(run(sess, J))
    ga = gradients(J, a)
    gb1 = gradients(J, b1)
    gb2 = gradients(J, b2)
    gu = gradients(J, u)
    println(abs.(run(sess, ga))|>maximum)
    println(abs.(run(sess, gb1))|>maximum)
    println(abs.(run(sess, gb2))|>maximum)
    println(abs.(run(sess, gu))|>maximum)
    error("Stop Test...")
end
J1 = fwi(us[1]+3500, constant(1, dtype=Int64))

__iter = 0
function print_iter(rk)
    global __iter
    if mod(__iter,1)==0
        println("\n************* ITER=$__iter *************\n")
    end
    __iter += 1
end

__cnt = 0
function print_loss(l, a, b1, b2, u)
    global __cnt
    if mod(__cnt,1)==0
        println("\niter=$__iter, eval=$__cnt, current loss=",l)
        println("a=$a, b1=$b1, b2=$b2")
    end
    writedlm("../../../src/data/param_$(__iter).txt", [a;b1;b2])
    writedlm("../../../src/data/u_$(__iter).txt", u)
    writedlm("../../../src/data/l_$(__iter).txt", l)
    __cnt += 1
end

__iter2 = 0
function print_iter2(rk)
    global __iter2
    if mod(__iter2,1)==0
        println("\n************* Phase I => ITER=$__iter2 *************\n")
    end
    __iter2 += 1
end

__cnt2 = 0
function print_loss2(l)
    global __cnt2
    if mod(__cnt2,1)==0
        println("\nPhase I: iter=$__iter2, eval=$__cnt2, current loss=",l,"\n")
    end
    writedlm("../../../src/data/pl_$(__iter2).txt", l)
    __cnt2 += 1
end

opt = ScipyOptimizerInterface(J, method="L-BFGS-B",options=Dict("maxiter"=> 30000, "ftol"=>1e-12, "gtol"=>1e-12))
opt1 = ScipyOptimizerInterface(J1, method="L-BFGS-B",options=Dict("maxiter"=> 100, "ftol"=>1e-12, "gtol"=>1e-12))

sess = Session()
init(sess)
ScipyOptimizerMinimize(sess, opt1, loss_callback=print_loss2, step_callback=print_iter2, fetches=[J1])
u__ = run(sess, u)
writedlm("../../../src/data/u_result.txt", u__)
ScipyOptimizerMinimize(sess, opt, loss_callback=print_loss, step_callback=print_iter, fetches=[J, a_, b1_, b2_, u])
