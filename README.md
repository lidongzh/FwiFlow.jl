# FwiFlow: Wave and Flow Inversion with Intelligent Automatic Differentiation

![](https://travis-ci.org/lidongzh/FwiFlow.jl.svg?branch=master)
![Coverage Status](https://coveralls.io/repos/github/lidongzh/FwiFlow.jl/badge.svg?branch=master)

| Quick Install                   | Documentation                                                |
| ------------------------------- | ------------------------------------------------------------ |
| `using Pkg; Pkg.add("FwiFlow")` | [![](https://img.shields.io/badge/docs-dev-blue.svg)](https://lidongzh.github.io/FwiFlow.jl/dev) |

## Highlights

- GPU-accelearated FWI module with checkpointing schemes;
- AMG-accelerated implicit pressure-implicit saturation scheme;
- Time Fractional Partial Differential Equations

## Philosophy

We treat physical simulations as a chain of multiple differentiable operators, such as discrete Laplacian evaluation, a Poisson solver and a single implicit time stepping for nonlinear PDEs. They are like building blocks that can be assembled to make simulation tools for new physical models. 

Those operators are differentiable and integrated in a computational graph so that the gradients can be computed automatically and efficiently via analyzing the dependency in the graph. Independent operators are parallelized executed. With the gradients we can perform gradient-based PDE-constrained optimization for inverse problems. 

FwiFlow is built on [ADCME](https://github.com/kailaix/ADCME.jl), a powerful static graph based automatic differentiation library for scientific computing (with TensorFlow backend). FwiFlow implements the idea of **Intelligent Automatic Differentiation**. 

<p align="center">
  <img src="docs/src/assets/op.png" width="50%">
</p>



## Applications

The following examples are for inversion 

| <img src="docs/src/assets/marmousi_inv.png" width="270">     | <img src="docs/src/assets/flow.png" width="270">             | <img src="docs/src/assets/diagram.png" width="270"> |
| ------------------------------------------------------------ | ------------------------------------------------------------ | --------------------------------------------------- |
| [Full-waveform Inversion](https://lidongzh.github.io/FwiFlow.jl/dev/tutorials/fwi/) | [Two Phase Flow](https://lidongzh.github.io/FwiFlow.jl/dev/tutorials/flow/) | FWI-Two Phase Flow Coupled Inversion                |
| <img src="docs/src/assets/frac.png" width="270">             |                                                              |                                                     |
| [Time Fractional PDE](https://lidongzh.github.io/FwiFlow.jl/dev/tutorials/timefrac/)                                          |                                                              |                                                     |





## Research Papers

1. **Kailai Xu**  (co-first author), **Dongzhuo Li**  (co-first author), Eric Darve, and Jerry M. Harris. [*Learning Hidden Dynamics using Intelligent Automatic Differentiation*](http://arxiv.org/abs/1912.07547).
2. **Dongzhuo Li** (co-first author), **Kailai Xu** (co-first author), Jerry M. Harris, and Eric Darve. [*Time-lapse Full-waveform Inversion for Subsurface Flow Problems with Intelligent Automatic Diﬀerentiation*](https://arxiv.org/abs/1912.07552).

## LICENSE

MIT License
Copyright (c) 2019 Dongzhuo Li and Kailai Xu (Equal Contribution to FwiFlow.jl)


