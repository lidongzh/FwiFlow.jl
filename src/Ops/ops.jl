export laplacian_op, poisson_op, sat_op, upwlap_op, upwps_op, fwi_op, fwi_obs_op

OPS_DIR = @__DIR__

"""
"""
function fwi_op(args...)
    fwi_op = load_op_and_grad("$OPS_DIR/FWI/build/libFwiOp", "fwi_op")
    fwi_op(args...)
end


function fwi_obs_op(args...)
    fwi_obs_op = load_op("$OPS_DIR/FWI/build/libFwiOp", "fwi_obs_op")
    fwi_obs_op(args...)
end

function laplacian(args...)
    laplacian = load_op_and_grad("$OPS_DIR/Laplacian/build/libLaplacian", "laplacian")
    laplacian(args...)
end


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


@doc raw"""
    poisson_op(c::Union{PyObject, Float64}, g::Union{PyObject, Float64}, 
        h::Union{PyObject, Float64}, ρ::Union{PyObject, Float64}, index::Union{Integer, PyObject}=0)

Solves the Poisson equation 

$\begin{aligned}
-\nabla\cdot(c(x) \nabla u(x)) &=  & x\in \Omega\\
\frac{\partial u(x)}{\partial n} &=  & x\in \Omega\\
\end{aligned}$
where $\Omega=[0,n_zh]\times [0, n_xh]$. The equation is solved using finite difference method, where the step size in each direction is $h$. Mathematically, the solution to the PDE is determined up to a constant. Numerically, we discretize the equation with the scheme
```math
(A+E_{11})\mathbf{u} = \mathbf{f}
```
where $A$ is the finite difference coefficient matrix,
```math
(E_{11})_{ij} = \begin{cases}1 & i=j=1 \\ 0 & \mbox{ otherwise }\end{cases}
```
![](../../docs/assets/doc_domain.png)
"""
function poisson_op(c::Union{PyObject, Float64}, g::Union{PyObject, Float64}, h::Union{PyObject, Float64}, ρ::Union{PyObject, Float64}, index::Union{Integer, PyObject}=0)
    c = convert_to_tensor(c, dtype=Float64)
    g = convert_to_tensor(g, dtype=Float64)
    h = convert_to_tensor(h, dtype=Float64)
    ρ = convert_to_tensor(ρ, dtype=Float64)
    index = convert_to_tensor(index, dtype=Int64)
    poisson_op = load_op_and_grad("$OPS_DIR/Poisson/build/libPoissonOp", "poisson_op")
    poisson_op(c, g, h, ρ, index)
end

function sat_op(args...)
    sat_op = load_op_and_grad("$OPS_DIR/Saturation/build/libSatOp", "sat_op")
    sat_op(args...)
end


function upwlap_op(args...)
    upwlap_op = load_op_and_grad("$OPS_DIR/Upwlap/build/libUpwlapOp", "upwlap_op")
    upwlap_op(args...)
end

function upwps_op(args...)
    upwps_op = load_op_and_grad("$OPS_DIR/Upwps/build/libUpwpsOp", "upwps_op")
    upwps_op(args...)
end