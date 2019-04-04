# Two_Phase_Flow_FWI

This project consider the coupling of the wave equation and a two-phase incompressible immiscible flow equation, mainly for CO2 injection or water injection monitoring

u_tt = m(x) u_xx + f(x,t)

m_t = grad(a(x)grad(m)) + b(x)*grad(m)

The time scale T_2 for the second equation is much larger than that of the first one T_1

T_2 >> T_1

a(x), b(x) are unknown functions and will be calibrated using observation data d_i(x), which depends on u_i for each observation time i


# Instruction

1. Compile AdvectionDiffusion

```
cd Ops/AdvectionDiffusion/
mkdir build
cd build
cmake ..
make -j
```

2. Test AdvectionDiffusion and Generate Data (required)
```
julia> include("cdtest.jl")
julia> include("gradtest.jl")
```

3. Compile CUFA
```
cd Ops/FWI/CUFD/Src
make -j
```

4. Compile Wrapper
```
cd Ops/FWI/ops/build
cmake ..
make -j
```

5. Generate data
```
julia> include("generate_m.jl")
python main_calc_obs_data.py
python fwitest.py
```

6. Test Wrapper
```
cd src
```

```
julia> include("fwi_gradient_check.jl")
julia> include("coupled_gradient_check")
```

7. Run experiments
```
julia> include("learn_m.jl")
```
