@testset "laplacian" begin
    h = 1.0
    rho = 1000.0
    G = 9.8
    len_z = 16
    len_x = 32
    nz = Int(len_z/h + 1)
    nx = Int(len_x/h + 1)
    tf_h=constant(1.0)
    coef = rand(nz, nx)
    func = rand(nz, nx)

    tf_coef = constant(coef)
    tf_func = constant(func)

    # gradient check -- v
    function scalar_function(m)
        # return sum(tanh(laplacian(m, tf_func, tf_h, constant(rho*G))))
        return sum(tanh(laplacian(tf_coef, m, tf_h, constant(rho*G))))
    end

    # m_  = tf_coef
    m_ = tf_func
    v_ = 0.01*rand(nz, nx)
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

    sess = Session()
    init(sess)
    sval_ = run(sess, s_)
    wval_ = run(sess, w_)
    close("all")
    loglog(gs_, abs.(sval_), "*-", label="finite difference")
    loglog(gs_, abs.(wval_), "+-", label="automatic differentiation")
    loglog(gs_, gs_.^2 * 0.5*abs(wval_[1])/gs_[1]^2, "--",label="\$\\mathcal{O}(\\gamma^2)\$")
    loglog(gs_, gs_ * 0.5*abs(sval_[1])/gs_[1], "--",label="\$\\mathcal{O}(\\gamma)\$")

    plt[:gca]()[:invert_xaxis]()
    legend()
    xlabel("\$\\gamma\$")
    ylabel("Error")
    savefig("laplacian.png"); close("all")
end

@testset "poisson" begin
    # gradient check -- v
    h = 1.0
    rho = 1000.0
    G = 9.8
    len_z = 16
    len_x = 32
    nz = Int(len_z/h + 1)
    nx = Int(len_x/h + 1)
    tf_h=constant(1.0)
    coef = zeros(nz, nx)
    rhs  = zeros(nz, nx)
    for i = 1:nz
        for j = 1:nx
            rhs[i,j] = -sin(2*pi/len_z*(i-1)*h) * sin(2*pi/len_x*(j-1)*h)
            coef[i,j] = 1.0 - cos(2*pi/len_z*(i-1)*h) * sin(2*pi/len_x*(j-1)*h) * len_z / (2*pi*rho*G)

            # rhs[i,j] = 2.0*(i-1)*h*exp(-(((i-1)*h)^2) -(((j-1)*h)^2)) * rho * G
            # coef[i,j] = 1.0 + exp(-(((i-1)*h)^2) -(((j-1)*h)^2))
        end
    end

    tf_coef = constant(coef)
    tf_rhs = constant(rhs)
    function scalar_function(m)
        return sum(tanh(poisson_op(tf_coef,m,tf_h,constant(rho*G), constant(0))))
        # return sum(tanh(poisson_op(m,tf_rhs,tf_h, constant(rho*G), constant(0))))
    end

    m_ = tf_rhs
    # m_  = tf_coef
    v_ = 0.01*rand(nz, nx)
    y_ = scalar_function(m_)
    dy_ = gradients(y_, m_)
    ms_ = Array{Any}(undef, 5)
    ys_ = Array{Any}(undef, 5)
    s_ = Array{Any}(undef, 5)
    w_ = Array{Any}(undef, 5)
    gs_ =  @. 1 / 20^(1:5)

    for i = 1:5
        g_ = gs_[i]
        ms_[i] = m_ + g_*v_
        ys_[i] = scalar_function(ms_[i])
        s_[i] = ys_[i] - y_
        w_[i] = s_[i] - g_*sum(v_.*dy_)
    end

    sess = Session()
    init(sess)
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
    savefig("poisson.png"); close("all")
end

@testset "sat_op" begin
    function ave_normal(quantity, m, n)
      aa = sum(quantity)
      return aa/(m*n)
    end


    # TODO: 
    # const ALPHA = 0.006323996017182
    const ALPHA = 1.0
    const SRC_CONST = 86400.0
    const K_CONST =  9.869232667160130e-16 * 86400
    nz=20
    nx=30
    sw = constant(zeros(nz, nx))
    swref = constant(zeros(nz,nx))
    μw = constant(0.001)
    μo = constant(0.003)
    K = constant(100.0 .* ones(nz, nx))
    ϕ = constant(0.25 .* ones(nz, nx))
    dt = constant(30.0)
    h = constant(100.0 * 0.3048)
    q1 = zeros(nz,nx)
    q2 = zeros(nz,nx)
    q1[10,5] = 0.002 * (1/(100.0 * 0.3048)^2)/20.0/0.3048 * SRC_CONST
    q2[10,25] = -0.002 * (1/(100.0 * 0.3048)^2)/20.0/0.3048 * SRC_CONST
    qw = constant(q1)
    qo = constant(q2)

    λw = sw.*sw/μw
    λo = (1-sw).*(1-sw)/μo
    λ = λw + λo
    f = λw/λ
    q = qw + qo + λw/(λo+1e-16).*qo

    # Θ = laplacian_op(K.*λo, potential_c, h, constant(0.0))
    Θ = upwlap_op(K*K_CONST, λo, constant(zeros(nz,nx)), h, constant(0.0))

    load_normal = (Θ+q/ALPHA) - ave_normal(Θ+q/ALPHA, nz, nx)

    tf_comp_p0 = upwps_op(K*K_CONST, λ, load_normal, constant(zeros(nz,nx)), h, constant(0.0), constant(2))
    sess = Session()
    init(sess)
    p0 = run(sess, tf_comp_p0)
    tf_p0 = constant(p0)

    # s = sat_op(sw,p0,K,ϕ,qw,qo,sw,dt,h)

    # function step(sw)
    #     λw = sw.*sw
    #     λo = (1-sw).*(1-sw)
    #     λ = λw + λo
    #     f = λw/λ
    #     q = qw + qo + λw/(λo+1e-16).*qo

    #     # Θ = laplacian_op(K.*λo, constant(zeros(nz,nx)), h, constant(0.0))
    #     Θ = upwlap_op(K, λo, constant(zeros(nz,nx)), h, constant(0.0))
    #     # Θ = constant(zeros(nz,nx))

    #     load_normal = (Θ+q/ALPHA) - ave_normal(Θ+q/ALPHA, nz, nx)

    #     p = poisson_op(λ.*K, load_normal, h, constant(0.0), constant(0)) # potential p = pw - ρw*g*h 
    #     # p = upwps_op(K, λ, load_normal, constant(zeros(nz,nx)), h, constant(0.0), constant(0))
        # sw = sat_op(sw,p,K,ϕ,qw,qo,μw,μo,sw,dt,h)
    #     return sw
    # end

    # NT=100
    # function evolve(sw, NT, qw, qo)
    #     # qw_arr = constant(qw) # qw: NT x m x n array
    #     # qo_arr = constant(qo)
    #     tf_sw = TensorArray(NT+1)
    #     function condition(i, ta)
    #         tf.less(i, NT+1)
    #     end
    #     function body(i, tf_sw)
    #         sw_local = step(read(tf_sw, i))
    #         i+1, write(tf_sw, i+1, sw_local)
    #     end
    #     tf_sw = write(tf_sw, 1, sw)
    #     i = constant(1, dtype=Int32)
    #     _, out = while_loop(condition, body, [i;tf_sw])
    #     read(out, NT+1)
    # end

    # s = evolve(sw, NT, qw, qo)



    # J = tf.nn.l2_loss(s)
    # tf_grad_K = gradients(J, K)
    # sess = Session()
    # init(sess)
    # # P = run(sess,p0)

    # # error("")
    # S=run(sess, s)
    # imshow(S);colorbar();

    # error("")

    # grad_K = run(sess, tf_grad_K)
    # imshow(grad_K);colorbar();
    # error("")
    # TODO: 

    # gradient check -- v
    function scalar_function(m)
        # return sum(tanh(sat_op(m,tf_p0,K*K_CONST,ϕ,qw,qo,μw,μo,constant(zeros(nz,nx)),dt,h)))
        return sum(tanh(sat_op(sw,m,K*K_CONST,ϕ,qw,qo,μw,μo,constant(zeros(nz,nx)),dt,h)))
        # return sum(tanh(sat_op(sw,tf_p0,m,ϕ,qw,qo,μw,μo,constant(zeros(nz,nx)),dt,h)))
        # return sum(tanh(sat_op(sw,tf_p0,K*K_CONST,m,qw,qo,μw,μo,constant(zeros(nz,nx)),dt,h)))
    end

    # m_ = sw
    # v_ = 0.1 * rand(nz,nx)

    m_ = tf_p0
    v_ = 5e5 .* rand(nz,nx)

    # m_ = K*K_CONST
    # v_ = 10 .* rand(nz,nx) *K_CONST

    # m_ = ϕ
    # v_ = 0.1 * rand(nz,nx)

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

    sess = Session()
    init(sess)
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
    savefig("sat_op.png"); close("all")
end

@testset "upwlap_op" begin
    h = 1.0
    rho = 1000.0
    G = 9.8
    len_z = 16
    len_x = 32
    nz = Int(len_z/h + 1)
    nx = Int(len_x/h + 1)
    tf_h=constant(1.0)


    perm = rand(nz, nx)
    mobi = rand(nz, nx)
    func = rand(nz, nx)

    tf_perm = constant(perm)
    tf_mobi = constant(mobi)
    tf_func = constant(func)



    # gradient check -- v
    function scalar_function(m)
        # return sum(tanh(upwlap_op(m, tf_mobi, tf_func, tf_h, constant(rho*G))))
        # return sum(tanh(upwlap_op(tf_perm, m, tf_func, tf_h, constant(rho*G))))
        return sum(tanh(upwlap_op(tf_perm, tf_mobi, m, tf_h, constant(rho*G))))
    end

    # m_ = constant(rand(10,20))
    # m_ = tf_perm
    # m_ = tf_mobi
    m_ = tf_func
    v_ = rand(nz, nx)
    y_ = scalar_function(m_)
    dy_ = gradients(y_, m_)
    ms_ = Array{Any}(undef, 5)
    ys_ = Array{Any}(undef, 5)
    s_ = Array{Any}(undef, 5)
    w_ = Array{Any}(undef, 5)
    gs_ =  @. 1 / 20^(1:5)

    for i = 1:5
        g_ = gs_[i]
        ms_[i] = m_ + g_*v_
        ys_[i] = scalar_function(ms_[i])
        s_[i] = ys_[i] - y_
        w_[i] = s_[i] - g_*sum(v_.*dy_)
    end

    sess = Session()
    init(sess)
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
end