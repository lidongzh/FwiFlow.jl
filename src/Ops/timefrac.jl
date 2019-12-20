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