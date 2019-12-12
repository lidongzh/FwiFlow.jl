# FwiFlow: Wave and Flow Inversion with Intelligent Automatic Differentiation

![](https://travis-ci.org/lidongzh/FwiFlow.jl.svg?branch=master)
![Coverage Status](https://coveralls.io/repos/github/lidongzh/FwiFlow.jl/badge.svg?branch=master)

| Documentation                                                |
| ------------------------------------------------------------ |
| [![](https://img.shields.io/badge/docs-dev-blue.svg)](https://lidongzh.github.io/FwiFlow.jl/dev) |



## Installation 

```julia
using Pkg
Pkg.add("FwiFlow")
```

## Philosophy

We treat physical simulations as a chain of multiple differentiable operators, such as discrete Laplacian evaluation, a Poisson solver and a single implicit time stepping for nonlinear PDEs. They are like building blocks that can be assembled to make simulation tools for new physical models. 

Those operators are differentiable and integrated in a computational graph so that the gradients can be computed automatically and efficiently via analyzing the dependency in the graph. Independent operators are parallelized executed. With the gradients we can perform gradient-based PDE-constrained optimization for inverse problems. 

FwiFlow is built on [ADCME](https://github.com/kailaix/ADCME.jl), a powerful static graph based automatic differentiation library for scientific computing (with TensorFlow backend). FwiFlow implements the idea of **Intelligent Automatic Differentiation**. 

<p align="center">
  <img src="docs/src/assets/op.png" width="50%">
</p>



## Applications

The following examples are for inversion 

| <img src="docs/src/assets/marmousi_inv.png" width="200">     | <img src="docs/src/assets/flow.png" width="200">             | <img src="docs/src/assets/diagram.png" width="200"> |
| ------------------------------------------------------------ | ------------------------------------------------------------ | --------------------------------------------------- |
| [Full-waveform Inversion](https://lidongzh.github.io/FwiFlow.jl/dev/tutorials/fwi/) | [Two Phase Flow](https://lidongzh.github.io/FwiFlow.jl/dev/tutorials/flow/) | FWI-Two Phase Flow Coupled Inversion                |





## Research Papers

1. **Kailai Xu**  (co-first author), **Dongzhuo Li**  (co-first author), Eric Darve, and Jerry M. Harris. *Learning Hidden Dynamics using Intelligent Automatic Differentiation*.
2. **Dongzhuo Li** (co-first author), **Kailai Xu** (co-first author), Jerry M. Harris, and Eric Darve. *Time-lapse Full-waveform Inversion for Subsurface Flow Problems with Intelligent Automatic Diﬀerentiation*.

## LICENSE

MIT License
Copyright (c) 2019 Dongzhuo Li and Kailai Xu


