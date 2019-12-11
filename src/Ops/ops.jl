export laplacian_op, poisson_op, sat_op, upwlap_op, upwps_op, fwi_op, fwi_obs_op

OPS_DIR = @__DIR__

@doc raw"""
    fwi_op(cp::Union{PyObject, Array{Float64}},cs::Union{PyObject, Array{Float64}},
    den::Union{PyObject, Array{Float64}},stf::Union{PyObject, Array{Float64}},
    gpu_id::Union{PyObject, Integer},shot_ids::Union{PyObject, Array{T}},para_fname::String) where T<:Integer

Computes the FWI loss function. 

- `cp` : P-wave velocity
- `cs` : S-wave velocity
- `den` : Density 
- `stf` : Source time functions  
- `gpu_id` : The ID of GPU to run this FWI operator
- `shot_ids` : The source function IDs (determining the location of sources)
- `para_fname` : Parameter file location
"""
function fwi_op(cp::Union{PyObject, Array{Float64}},cs::Union{PyObject, Array{Float64}},
        den::Union{PyObject, Array{Float64}},stf::Union{PyObject, Array{Float64}},
        gpu_id::Union{PyObject, Integer},shot_ids::Union{PyObject, Array{T}},para_fname::String) where T<:Integer
    cp = convert_to_tensor(cp, dtype=Float64)
    cs = convert_to_tensor(cs, dtype=Float64)
    den = convert_to_tensor(den, dtype=Float64)
    stf = convert_to_tensor(stf, dtype=Float64)
    gpu_id = convert_to_tensor(gpu_id, dtype=Int32)
    shot_ids = convert_to_tensor(shot_ids, dtype=Int32)
    fwi_op = load_op_and_grad("$OPS_DIR/FWI/build/libFwiOp", "fwi_op")
    fwi_op(cp,cs,den,stf,gpu_id,shot_ids,para_fname)
end


@doc raw"""
    fwi_obs_op(cp::Union{PyObject, Array{Float64}},cs::Union{PyObject, Array{Float64}},
        den::Union{PyObject, Array{Float64}},stf::Union{PyObject, Array{Float64}},
        gpu_id::Union{PyObject, Integer},shot_ids::Union{PyObject, Array{T}},para_fname::String) where T<:Integer

Generates the observation data and store them as files which will be used by [`fwi_op`](@ref)
For the meaning of parameters, see [`fwi_op`](@ref).
"""
function fwi_obs_op(cp::Union{PyObject, Array{Float64}},cs::Union{PyObject, Array{Float64}},
    den::Union{PyObject, Array{Float64}},stf::Union{PyObject, Array{Float64}},
    gpu_id::Union{PyObject, Integer},shot_ids::Union{PyObject, Array{T}},para_fname::String) where T<:Integer
    fwi_obs_op = load_op("$OPS_DIR/FWI/build/libFwiOp", "fwi_obs_op")
    fwi_obs_op(args...)
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
(E_{11})_{ij} = \left\{ \begin{matrix}1 & i=j=1 \\ 0 & \mbox{ otherwise }\end{matrix}\right.
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