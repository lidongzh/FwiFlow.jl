function findindex()
    I1 = Int64[]
    I2 = Int64[]
    I3 = Int64[]
    I4 = Int64[]
    I5 = Int64[]
    I6 = Int64[]
    for i = 2:m-1
        for j = 2:n-1
            push!(I1, i+(j-1)*m)
            push!(I2, i+(j-1)*m)
            push!(I3, (i+1)+(j-1)*m)
            push!(I4, i-1+(j-1)*m)
            push!(I5, i+j*m)
            push!(I6, i+(j-2)*m)
        end
    end
    return I1, I2, I3, I4, I5, I6
end

I1, I2, I3, I4, I5, I6 = findindex()
function one_step(u, NT, a, b1, b2)
    dt = 1.0
    h = 24.0
    # @show u
    y = vec(u)
    lamb = dt/h/h
    mu = dt/h/2
    r1 = 1.0 - 4.0*a*lamb
    r2 = a*lamb+b1*mu
    r3 = a*lamb-b1*mu
    r4 = a*lamb+b2*mu
    r5 = a*lamb-b2*mu
    for i = 1:NT
        y = scatter_add(constant(zeros(m*n)), I1, r1*y[I2] + r2*y[I3] + r3*y[I4] + r4*y[I5] + r5*y[I6])
    end
    # @show m, n
    y = reshape(y, m, n)
    return y
end

