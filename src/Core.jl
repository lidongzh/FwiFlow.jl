export laplacian_op, poisson_op, sat_op, upwlap_op, upwps_op, fwi_op, fwi_obs_op, sat_op2, eikonal

OPS_DIR = joinpath(@__DIR__, "../deps/CustomOps")

@doc raw"""
    fwi_op(lambda::Union{PyObject, Array{Float64}},mu::Union{PyObject, Array{Float64}},
        den::Union{PyObject, Array{Float64}},stf::Union{PyObject, Array{Float64}},
        gpu_id::Union{PyObject, Integer},shot_ids::Union{PyObject, Array{T}},para_fname::String) where T<:Integer

Computes the FWI loss function. 

- `lambda` : Lame's first parameter (unit: MPa)
- `mu` : Lame's second parameter (shear modulus, unit: MPa)
- `den` : Density 
- `stf` : Source time functions  
- `gpu_id` : The ID of GPU to run this FWI operator
- `shot_ids` : The source function IDs (determining the location of sources)
- `para_fname` : Parameter file location
"""
function fwi_op(lambda::Union{PyObject, Array{Float64}},mu::Union{PyObject, Array{Float64}},
        den::Union{PyObject, Array{Float64}},stf::Union{PyObject, Array{Float64}},
        gpu_id::Union{PyObject, Integer},shot_ids::Union{PyObject, Array{T}},para_fname::String) where T<:Integer
    lambda = convert_to_tensor(lambda, dtype=Float64)
    mu = convert_to_tensor(mu, dtype=Float64)
    den = convert_to_tensor(den, dtype=Float64)
    stf = convert_to_tensor(stf, dtype=Float64)
    gpu_id = convert_to_tensor(gpu_id, dtype=Int32)
    shot_ids = convert_to_tensor(shot_ids, dtype=Int32)
    fwi_op = load_op_and_grad("$OPS_DIR/FWI/build/libFwiOp", "fwi_op")
    fwi_op(lambda,mu,den,stf,gpu_id,shot_ids,para_fname)
end


@doc raw"""
    fwi_obs_op(lambda::Union{PyObject, Array{Float64}},mu::Union{PyObject, Array{Float64}},
        den::Union{PyObject, Array{Float64}},stf::Union{PyObject, Array{Float64}},
        gpu_id::Union{PyObject, Integer},shot_ids::Union{PyObject, Array{T}},para_fname::String) where T<:Integer

Generates the observation data and store them as files which will be used by [`fwi_op`](@ref)
For the meaning of parameters, see [`fwi_op`](@ref).
"""
function fwi_obs_op(lambda::Union{PyObject, Array{Float64}},mu::Union{PyObject, Array{Float64}},
    den::Union{PyObject, Array{Float64}},stf::Union{PyObject, Array{Float64}},
    gpu_id::Union{PyObject, Integer},shot_ids::Union{PyObject, Array{T}},para_fname::String) where T<:Integer
    lambda = convert_to_tensor(lambda, dtype=Float64)
    mu = convert_to_tensor(mu, dtype=Float64)
    den = convert_to_tensor(den, dtype=Float64)
    stf = convert_to_tensor(stf, dtype=Float64)
    gpu_id = convert_to_tensor(gpu_id, dtype=Int32)
    shot_ids = convert_to_tensor(shot_ids, dtype=Int32)
    fwi_obs_op = load_op("$OPS_DIR/FWI/build/libFwiOp", "fwi_obs_op")
    fwi_obs_op(lambda, mu, den, stf, gpu_id, shot_ids, para_fname)
end


@doc raw"""
    laplacian_op(coef::Union{PyObject, Array{Float64}}, f::Union{PyObject, Array{Float64}}, 
            h::Union{PyObject, Float64}, ρ::Union{PyObject, Float64})

Computes the Laplacian of function $f(\mathbf{x})$; here ($\mathbf{x}=[z\quad x]^T$)
```math 
-\nabla\cdot\left(c(\mathbf{x}) \nabla \left(u(\mathbf{x}) -\rho \begin{bmatrix}z \\ 0\end{bmatrix}  \right)\right)
``` 
"""
function laplacian_op(coef::Union{PyObject, Array{Float64}}, f::Union{PyObject, Array{Float64}}, 
    h::Union{PyObject, Float64}, ρ::Union{PyObject, Float64})
    coef = convert_to_tensor(coef, dtype=Float64)
    f = convert_to_tensor(f, dtype=Float64)
    h = convert_to_tensor(h, dtype=Float64)
    ρ = convert_to_tensor(ρ, dtype=Float64)
    laplacian = load_op_and_grad("$OPS_DIR/Laplacian/build/libLaplacian", "laplacian")
    laplacian_op(coef, f, h, ρ)
end

@doc raw"""
    poisson_op(c::Union{PyObject, Float64}, g::Union{PyObject, Float64}, 
        h::Union{PyObject, Float64}, ρ::Union{PyObject, Float64}, index::Union{Integer, PyObject}=0)

Solves the Poisson equation ($\mathbf{x}=[z\quad x]^T$)

$\begin{aligned}
-\nabla\cdot\left(c(\mathbf{x}) \nabla \left(u(\mathbf{x}) -\rho  \begin{bmatrix}z \\ 0\end{bmatrix}   \right)\right) &=  g(\mathbf{x}) & \mathbf{x}\in \Omega\\
\frac{\partial u(x)}{\partial n} &=  0 & \mathbf{x}\in \Omega\\
\end{aligned}$

Here $\Omega=[0,n_zh]\times [0, n_xh]$. The equation is solved using finite difference method, where the step size in each direction is $h$. Mathematically, the solution to the PDE is determined up to a constant. Numerically, we discretize the equation with the scheme
```math
(A+E_{11})\mathbf{u} = \mathbf{f}
```
where $A$ is the finite difference coefficient matrix,

```math
(E_{11})_{ij} = \left\{ \begin{matrix}1 & i=j=1 \\ 0 & \text{ otherwise }\end{matrix}\right.
```

- `index` : `Int32`, when `index=1`, `SparseLU` is used to solve the linear system; otherwise the function invokes algebraic multigrid method from `amgcl`. 
"""
function poisson_op(c::Union{PyObject, Array{Float64}}, g::Union{PyObject, Array{Float64}}, 
    h::Union{PyObject, Float64}, 
    ρ::Union{PyObject, Float64}, index::Union{PyObject, Integer}=0)
    c = convert_to_tensor(c, dtype=Float64)
    g = convert_to_tensor(g, dtype=Float64)
    h = convert_to_tensor(h, dtype=Float64)
    ρ = convert_to_tensor(ρ, dtype=Float64)
    index = convert_to_tensor(index, dtype=Int64)
    poisson_op = load_op_and_grad("$OPS_DIR/Poisson/build/libPoissonOp", "poisson_op")
    poisson_op(c, g, h, ρ, index)
end

@doc raw"""
    sat_op(s0::Union{PyObject, Array{Float64}},pt::Union{PyObject, Array{Float64}},
        permi::Union{PyObject, Array{Float64}},poro::Union{PyObject, Array{Float64}},
        qw::Union{PyObject, Array{Float64}},qo::Union{PyObject, Array{Float64}},
        muw::Union{PyObject, Float64},muo::Union{PyObject, Float64},
        sref::Union{PyObject, Array{Float64}},dt::Union{PyObject, Float64},h::Union{PyObject, Float64})

Solves the following discretized equation 
```math
\phi (S_2^{n + 1} - S_2^n) - \nabla  \cdot \left( {{m_2}(S_2^{n + 1})K\nabla \Psi _2^n} \right) \Delta t= \left(q_2^n + q_1^n \frac{m_2(S^{n+1}_2)}{m_1(S^{n+1}_2)}\right) \Delta t
```
where
```math
m_2(s) = \frac{s^2}{\mu_w}\qquad m_1(s) = \frac{(1-s)^2}{\mu_o}
```
This is a nonlinear equation and is solved with the Newton-Raphson method. 


- `s0` : $n_z\times n_x$, saturation of fluid, i.e., $S_2^n$
- `pt` : $n_z\times n_x$, potential of fluid, i.e., $\Psi_2^n$
- `permi` : $n_z\times n_x$, permeability, i.e., $K$ 
- `poro` : $n_z\times n_x$, porosity, i.e., $\phi$
- `qw` : $n_z\times n_x$, injection or production rate of ﬂuid 1, $q_2^n$
- `qo` : $n_z\times n_x$, injection or production rate of ﬂuid 2, $q_1^n$
- `muw` : viscosity of fluid 1, i.e., $\mu_w$
- `muo` : viscosity of fluid 2, i.e., $\mu_o$
- `sref` : $n_z\times n_x$, initial guess for $S_2^{n+1}$
- `dt` : Time step size  
- `h` : Spatial step size
"""
function sat_op(s0::Union{PyObject, Array{Float64}},pt::Union{PyObject, Array{Float64}},
    permi::Union{PyObject, Array{Float64}},poro::Union{PyObject, Array{Float64}},
    qw::Union{PyObject, Array{Float64}},qo::Union{PyObject, Array{Float64}},
    muw::Union{PyObject, Float64},muo::Union{PyObject, Float64},
    sref::Union{PyObject, Array{Float64}},dt::Union{PyObject, Float64},h::Union{PyObject, Float64})
    s0 = convert_to_tensor(s0, dtype=Float64)
    pt = convert_to_tensor(pt, dtype=Float64)
    permi = convert_to_tensor(permi, dtype=Float64)
    poro = convert_to_tensor(poro, dtype=Float64)
    qw = convert_to_tensor(qw, dtype=Float64)
    qo = convert_to_tensor(qo, dtype=Float64)
    muw = convert_to_tensor(muw, dtype=Float64)
    muo = convert_to_tensor(muo, dtype=Float64)
    sref = convert_to_tensor(sref, dtype=Float64)
    dt = convert_to_tensor(dt, dtype=Float64)
    h = convert_to_tensor(h, dtype=Float64)
    sat_op = load_op_and_grad("$OPS_DIR/Saturation/build/libSatOp", "sat_op")
    sat_op(s0,pt,permi,poro,qw,qo,muw,muo,sref,dt,h)
end

@doc raw"""
    sat_op2(s0::Union{PyObject, Array{Float64}},
    dporodt::Union{PyObject, Array{Float64}},
    pt::Union{PyObject, Array{Float64}},
    permi::Union{PyObject, Array{Float64}},
    poro::Union{PyObject, Array{Float64}},
    qw::Union{PyObject, Array{Float64}},
    qo::Union{PyObject, Array{Float64}},
    muw::Union{PyObject, Float64},
    muo::Union{PyObject, Float64},
    sref::Union{PyObject, Array{Float64}},
    dt::Union{PyObject, Float64},
    h::Union{PyObject, Float64})

Solves the following discretized equation 
```math
\phi (S_2^{n + 1} - S_2^n) + \Delta t \dot \phi S_2^{n+1} - \nabla  \cdot \left( {{m_2}(S_2^{n + 1})K\nabla \Psi _2^n} \right) \Delta t= \left(q_2^n + q_1^n \frac{m_2(S^{n+1}_2)}{m_1(S^{n+1}_2)}\right) \Delta t
```
where
```math
m_2(s) = \frac{s^2}{\mu_w}\qquad m_1(s) = \frac{(1-s)^2}{\mu_o}
```
This is a nonlinear equation and is solved with the Newton-Raphson method. 


- `s0` : $n_z\times n_x$, saturation of fluid, i.e., $S_2^n$
- `dporodt` : $n_z\times n_x$, rate of porosity, $\dot \phi$
- `pt` : $n_z\times n_x$, potential of fluid, i.e., $\Psi_2^n$
- `permi` : $n_z\times n_x$, permeability, i.e., $K$ 
- `poro` : $n_z\times n_x$, porosity, i.e., $\phi$
- `qw` : $n_z\times n_x$, injection or production rate of ﬂuid 1, $q_2^n$
- `qo` : $n_z\times n_x$, injection or production rate of ﬂuid 2, $q_1^n$
- `muw` : viscosity of fluid 1, i.e., $\mu_w$
- `muo` : viscosity of fluid 2, i.e., $\mu_o$
- `sref` : $n_z\times n_x$, initial guess for $S_2^{n+1}$
- `dt` : Time step size  
- `h` : Spatial step size
"""
function sat_op2(s0::Union{PyObject, Array{Float64}},
    dporodt::Union{PyObject, Array{Float64}},
    pt::Union{PyObject, Array{Float64}},
    permi::Union{PyObject, Array{Float64}},
    poro::Union{PyObject, Array{Float64}},
    qw::Union{PyObject, Array{Float64}},
    qo::Union{PyObject, Array{Float64}},
    muw::Union{PyObject, Float64},
    muo::Union{PyObject, Float64},
    sref::Union{PyObject, Array{Float64}},
    dt::Union{PyObject, Float64},
    h::Union{PyObject, Float64})
    s0 = convert_to_tensor(s0, dtype=Float64)
    dporodt = convert_to_tensor(dporodt, dtype=Float64)
    pt = convert_to_tensor(pt, dtype=Float64)
    permi = convert_to_tensor(permi, dtype=Float64)
    poro = convert_to_tensor(poro, dtype=Float64)
    qw = convert_to_tensor(qw, dtype=Float64)
    qo = convert_to_tensor(qo, dtype=Float64)
    muw = convert_to_tensor(muw, dtype=Float64)
    muo = convert_to_tensor(muo, dtype=Float64)
    sref = convert_to_tensor(sref, dtype=Float64)
    dt = convert_to_tensor(dt, dtype=Float64)
    h = convert_to_tensor(h, dtype=Float64)
    saturation_ = load_op_and_grad("$OPS_DIR/Saturation2/build/libSaturation","saturation")
    saturation_(s0,dporodt,pt,permi,poro,qw,qo,muw,muo,sref,dt,h)
end


@doc raw"""
    upwlap_op(perm::Union{PyObject, Array{Float64}},
        mobi::Union{PyObject, Array{Float64}},
        func::Union{PyObject, Array{Float64}},
        h::Union{PyObject, Float64},
        rhograv::Union{PyObject, Float64})

Computes the Laplacian of function $f(\mathbf{x})$; here $\mathbf{x}=[z\quad x]^T$.
```math 
\nabla\cdot\left(m(\mathbf{x})K(\mathbf{x}) \nabla \left(f(\mathbf{x}) -\rho \begin{bmatrix}z \\ 0\end{bmatrix}  \right)\right)
``` 
The permeability on the computational grid is computed with Harmonic mean; 
the mobility is computed with upwind scheme. 

- `perm` : $n_z\times n_x$, permeability of fluid, i.e., $K$
- `mobi` : $n_z\times n_x$, mobility of fluid, i.e., $m$
- `func` : $n_z\times n_x$, potential of fluid, i.e., $f$
- `h` : `Float64`, spatial step size 
- `rhograv` : `Float64`, i.e., $\rho$
"""
function upwlap_op(perm::Union{PyObject, Array{Float64}},
    mobi::Union{PyObject, Array{Float64}},
    func::Union{PyObject, Array{Float64}},
    h::Union{PyObject, Float64},rhograv::Union{PyObject, Float64})
    perm = convert_to_tensor(perm, dtype=Float64)
    mobi = convert_to_tensor(mobi, dtype=Float64)
    func = convert_to_tensor(func, dtype=Float64)
    h = convert_to_tensor(h, dtype=Float64)
    rhograv = convert_to_tensor(rhograv, dtype=Float64)
    upwlap_op = load_op_and_grad("$OPS_DIR/Upwlap/build/libUpwlapOp", "upwlap_op")
    upwlap_op(perm,mobi,func,h,rhograv)
end

@doc raw"""
    upwps_op(perm::Union{PyObject, Array{Float64}},mobi::Union{PyObject, Array{Float64}},
    src,funcref,h::Union{PyObject, Float64},rhograv::Union{PyObject, Float64},index::Union{PyObject, Integer})

Solves the Poisson equation 
```math
-\nabla\cdot\left(m(\mathbf{x})K(\mathbf{x}) \nabla \left(u(\mathbf{x}) -\rho  \begin{bmatrix}z \\ 0\end{bmatrix}   \right)\right) =  g(\mathbf{x}) 
```
See [`upwps_op`](@ref) for detailed description. 

- `perm` : $n_z\times n_x$, permeability of fluid, i.e., $K$
- `mobi` : $n_z\times n_x$, mobility of fluid, i.e., $m$
- `src` : $n_z\times n_x$, source function, i.e., $g(\mathbf{x})$
- `funcref` : $n_z\times n_x$, currently it is not not used
- `h` : `Float64`, spatial step size 
- `rhograv` : `Float64`, i.e., $\rho$
- `index` : `Int32`, when `index=1`, `SparseLU` is used to solve the linear system; otherwise the function invokes algebraic multigrid method from `amgcl`. 
"""
function upwps_op(perm::Union{PyObject, Array{Float64}},mobi::Union{PyObject, Array{Float64}},
    src,funcref,h::Union{PyObject, Float64},rhograv::Union{PyObject, Float64},index::Union{PyObject, Integer})
    perm = convert_to_tensor(perm, dtype=Float64)
    mobi = convert_to_tensor(mobi, dtype=Float64)
    funcref = convert_to_tensor(funcref, dtype=Float64)
    h = convert_to_tensor(h, dtype=Float64)
    rhograv = convert_to_tensor(rhograv, dtype=Float64)
    index = convert_to_tensor(index, dtype=Int64)
    upwps_op = load_op_and_grad("$OPS_DIR/Upwps/build/libUpwpsOp", "upwps_op")
    upwps_op(perm,mobi,src,funcref,h,rhograv,index)
end


export time_fractional_op, time_fractional_t_op

@doc raw"""
    time_fractional_t_op(i::Union{Integer, PyObject}, ta::PyObject, 
        α::Union{Float64, PyObject}, Δt::Union{Float64, PyObject})

Returns the coefficients for the time fractional derivative 
```math
{}_0^CD_t^\alpha f(t) = \frac{1}{\Gamma(1-\alpha)}\int_0^t \frac{f'(\tau)d\tau}{(t-\tau)^\alpha}
```

The discretization scheme used here is 

$\begin{aligned}
& {}_0^CD_\tau^\alpha u(\tau_n) \\
= &\frac{\Delta \tau^{-\alpha}}{\Gamma(2-\alpha)}\left[G_0 u_n - \sum_{k=1}^{n-1}(G_{n-k-1}-G_{n-k})u_k + G_n u_0 \right] + \mathcal{O}(\Delta \tau^{2-\alpha})	
\end{aligned}$

Here  
```math
G_m = (m+1)^{1-\alpha} - m^{1-\alpha}, \quad m\geq 0, \quad 0<\alpha<1
```

The function returns 
- `c` : $\frac{\Delta \tau^{-\alpha}}{\Gamma(2-\alpha)}$
- `cum` : $- \sum_{k=1}^{n-1}(G_{n-k-1}-G_{n-k})u_k + G_n u_0$
"""
function time_fractional_t_op(i::Union{Integer, PyObject}, ta::PyObject, 
         α::Union{Float64, PyObject}, Δt::Union{Float64, PyObject})
    α = convert_to_tensor(α, dtype=Float64)
    i = convert_to_tensor(i, dtype=Int32)

    function aki(k,i)
        a = cast(i-k, Float64)
        return (a+2.0)^(1-α)-2*(a+1.0)^(1-α)+a^(1-α)
    end
    
    function cond1(k, i, ta, cum)
        k<=i
    end
    
    function body1(k, i, ta, cum)
        u = read(ta, k)
        return k+1, i, ta, cum+aki(k,i)*u
    end

    cum = tf.zeros_like(read(ta, 1))
    k = constant(1, dtype=Int32)
    _, _, _, cum = while_loop(cond1, body1, [k, i, ta, cum])
    c = Δt^(-α)/exp(tf.math.lgamma(2-α))
    return c, cum 
end


@doc raw"""
    time_fractional_op(α::Union{Float64, PyObject}, f_fun::Function, 
        T::Float64, u0::Union{Float64, Array{Float64}, PyObject}, 
        NT::Int64, θ=missing)

Returns a $(NT+1)\times \texttt{size}(u_0)$ solution array. The function solves the following time fractional differential equation with explicit scheme 

```math
{}_0^CD_t^\alpha u(t) = f(t, u, \theta)
```
"""
function time_fractional_op(α::Union{Float64, PyObject}, f_fun::Function, 
        T::Float64, u0::Union{Float64, Array{Float64}, PyObject}, 
        NT::Int64, θ=missing)
        Δt = T/NT
        ta = TensorArray(NT+1)
        ta = write(ta, 1, convert_to_tensor(u0))
        function condition(i, ta)
            i<=NT+1
        end
        function body(i, ta)
            # time = (i-1)*Δt, 
            u = read(ta, i-1)
            k = cast(i,Float64)-1
            c, cum = time_fractional_t_op(k, ta, α, Δt)
            F = f_fun(k*Δt, u, θ)
            ta = write(ta, i, F/c - cum)
            i+1, ta 
        end
        i = constant(2, dtype=Int32)
        _, out = while_loop(condition, body, [i, ta])
        return stack(out)
end

@doc raw"""
    eikonal(f::Union{Array{Float64}, PyObject},
    srcx::Int64,srcy::Int64,h::Float64)

Solves the Eikonal equation 

$$|\nabla u(x)| = f(x)$$

where $f(x)$ is the reciprocal of speeds. 
"""
function eikonal(f::Union{Array{Float64}, PyObject},
    srcx::Int64,srcy::Int64,h::Float64)
    n_, m_ = size(f) # m width, n depth 
    n = n_-1
    m = m_-1
    eikonal_ = load_op_and_grad("$OPS_DIR/Eikonal/build/libEikonal","eikonal")
    f,srcx,srcy,m,n,h = convert_to_tensor([f,srcx,srcy,m,n,h], [Float64,Int64,Int64,Int64,Int64,Float64])
    f = reshape(f, (-1,))
    u = eikonal_(f,srcx,srcy,m,n,h)
    u = set_shape(u, (length(f),))
    reshape(u, (n_, m_))
end
