# Getting Started

## Intelligent Automatic Differentiation (IAD): Philosophy

We treat physical simulations as a chain of multiple differentiable operators, such as discrete Laplacian evaluation, a Poisson solver and a single implicit time stepping for nonlinear PDEs. They are like building blocks that can be assembled to make simulation tools for new physical models. 

Those operators are differentiable and integrated in a computational graph so that the gradients can be computed automatically and efficiently via analyzing the dependency in the graph. Independent operators are executed in parallel. With the gradients we can perform gradient-based PDE-constrained optimization for inverse problems. 

FwiFlow is built on [ADCME](https://github.com/kailaix/ADCME.jl), a powerful static graph based automatic differentiation library for scientific computing (with TensorFlow backend). FwiFlow implements the idea of **Intelligent Automatic Differentiation**. 

![](docs/src/assets/op.png)

## Tutorials

Here are some examples to start with (`*` denotes advanced examples)

- [FWI](https://lidongzh.github.io/FwiFlow.jl/dev/tutorials/fwi/)
- [Two Phase Flow Inversion](https://lidongzh.github.io/FwiFlow.jl/dev/tutorials/flow/)
- *[Coupled Inversion](https://github.com/lidongzh/FwiFlow.jl/tree/master/docs/codes/src_fwi_coupled)
- *[Coupled Inversion: Channel Flow](https://github.com/lidongzh/FwiFlow.jl/tree/master/docs/codes/src_fwi_channel)


## FwiFlow: Application of IAD to FWI and Two Phase Flow Coupled Inversion

This framework uses waveform data to invert for intrinsic parameters (e.g., permeability and porosity) in subsurface problems, with coupled flow physics, rock physics, and wave physics models.

![](assets/diagram.png)

IAD provides three levels of user control with 

- built-in differentiable operators from modern deep-learning infrastructures (TensorFlow), and customized operators that can either 
- encapsulate analytic adjoint gradient computation or 
- handle the forward simulation and compute the corresponding gradient for a single time step. 

This intelligent strategy strikes a good balance between computational efficiency and programming efficiency and would serve as a paradigm for a wide range of PDE-constrained geophysical inverse problems.

### Physical Models

### Flow Physics
The flow physics component maps from intrinsic properties such as permeability to flow properties, such as fluid saturation. We use a model of two-phase flow in porous media as an example. The governing equations are convervation of mass, Darcy's law, and other relationships.

### Rock Physics
The rock physics model describes the relationship between fluid properties and rock elastic properties. As one fluid phase displaces the other, the bulk modulus and density of rocks vary. 

### Wave Physics
The elastic wave equation maps from elastic properties to wavefields, such as particle velocity and stress, which can be recorded by receiver arrays as seismic waveform data.

The elastic wave equation maps from elastic properties to wavefields, such as particle velocity and stress, which can be recorded by receiver arrays as seismic waveform data.


###	The Adjoint Method & Automatic Differentation

![](./assets/flow_comp_graph.png)


### Customized Operators

