export laplacian_op, poisson_op, sat_op, upwlap_op, upwps_op, fwi_op, fwi_obs_op

OPS_DIR = @__DIR__

const PA = Union{PyObject, Array{Float64}}
const PD = Union{PyObject, Float64}
const PI = Union{PyObject, Integer}

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


@doc raw"""
    laplacian_op(coef::Union{PyObject, Array{Float64}}, f::Union{PyObject, Array{Float64}}, 
            h::Union{PyObject, Float64}, ρ::Union{PyObject, Float64})

Computes the Laplacian of function $f(\mathbf{x})$; here ($\mathbf{x}=[z x]^T$)
```math 
-\nabla\cdot\left(c(\mathbf{x}) \nabla \left(u(\mathbf{x}) -\rho\nabla \begin{bmatrix}z \\ 0\end{bmatrix}  \right)\right)
``` 
"""
function laplacian_op(coef::PA, f::PA, h::PD, ρ::PD)
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

Solves the Poisson equation ($\mathbf{x}=[z x]^T$)

$\begin{aligned}
-\nabla\cdot\left(c(\mathbf{x}) \nabla \left(u(\mathbf{x}) -\rho\nabla z  \right)\right) &=  g(\mathbf{x}) & \mathbf{x}\in \Omega\\
\frac{\partial u(x)}{\partial n} &=  0 & \mathbf{x}\in \Omega\\
\end{aligned}$

Here $\Omega=[0,n_zh]\times [0, n_xh]$. The equation is solved using finite difference method, where the step size in each direction is $h$. Mathematically, the solution to the PDE is determined up to a constant. Numerically, we discretize the equation with the scheme
```math
(A+E_{11})\mathbf{u} = \mathbf{f}
```
where $A$ is the finite difference coefficient matrix,
```math
(E_{11})_{ij} = \begin{cases}1 & i=j=1 \\ 0 & \mbox{ otherwise }\end{cases}
```

When `index=1`, the Eigen `SparseLU` is used to solve the linear system; otherwise the function invokes algebraic multigrid method from `amgcl`. 
"""
function poisson_op(c::PA, g::PA, h::PD, 
            ρ::PD, index::PI=0)
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