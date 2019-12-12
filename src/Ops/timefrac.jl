export time_fractional_op

@doc raw"""
    time_fractional_op(i::Union{Integer, PyObject}, ta::PyObject, nz::Integer, nx::Integer,
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
function time_fractional_op(i::Union{Integer, PyObject}, ta::PyObject, nz::Integer, nx::Integer,
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

    cum = constant(zeros(nz, nx))
    k = constant(1, dtype=Int32)
    _, _, _, cum = while_loop(cond1, body1, [k, i, ta, cum])
    c = Δt^(-α)/exp(tf.math.lgamma(2-α))
    return c, cum 
end
