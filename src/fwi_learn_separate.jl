include("ops.jl")

do_test = false

# idx_phase = 2

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

# for i = 2:nsrc
#     @show i
#     us[i] = evolve(us[i-1], 5)
# end


J1 = fwi(u+3500, constant(3, dtype=Int64))


__iter2 = 0
function print_iter2(rk)
    global __iter2
    if mod(__iter2,1)==0
        println("\n************* Phase I => ITER=$__iter2 *************\n")
    end
    __iter2 += 1
end

__cnt2 = 0
function print_loss2(l, u)
    global __cnt2
    if mod(__cnt2,1)==0
        println("\nPhase I: iter=$__iter2, eval=$__cnt2, current loss=",l,"\n")
    end
    writedlm("../../../src/fwi_results_phase3/pl_$(__iter2).txt", l)
    writedlm("../../../src/fwi_results_phase3/u_$(__iter2).txt", u)
    __cnt2 += 1
end

# opt = ScipyOptimizerInterface(J, method="L-BFGS-B",options=Dict("maxiter"=> 30000, "ftol"=>1e-12, "gtol"=>1e-12))
opt1 = ScipyOptimizerInterface(J1, method="L-BFGS-B",options=Dict("maxiter"=> 100, "ftol"=>1e-12, "gtol"=>1e-12))

sess = Session()
init(sess)
ScipyOptimizerMinimize(sess, opt1, loss_callback=print_loss2, step_callback=print_iter2, fetches=[J1,u])
save(sess, "ata1.jld2")
