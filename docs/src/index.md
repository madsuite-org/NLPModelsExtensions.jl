# NLPModelsExtensions.jl

Extension package for [NLPModels.jl](https://github.com/JuliaSmoothOptimizers/NLPModels.jl).

The extensions form one chain of abstract types, each adding to the one above:

```
AbstractNLPModel
└── AbstractBatchedNLPModel         a batch of NLPs sharing one sparsity pattern
    └── AbstractParametricNLPModel  + derivatives with respect to parameters
          = AbstractExtNLPModel     the type to subtype
```

| Layer | Adds | Default |
|---|---|---|
| [`AbstractBatchedNLPModel`](@ref) | `nbatch` instances, data as matrices with one column per instance | a single problem is `nbatch = 1` |
| [`AbstractParametricNLPModel`](@ref) | derivatives with respect to parameters `Θ` (`npar × nbatch`) | no parameters, `npar = 0` |

- **Models** subtype [`AbstractExtNLPModel`](@ref), the bottom of the chain. It costs nothing:
  a model without parameters implements only the batched API.
- **Code that consumes models** accepts the least derived type it needs, e.g. a batched solver
  takes `AbstractBatchedNLPModel` and a sensitivity method takes `AbstractParametricNLPModel`.
- **An ordinary `AbstractNLPModel`** becomes an `AbstractExtNLPModel` with [`ExtNLPModel`](@ref).
- **Other packages** extend the chain by subtyping `AbstractExtNLPModel`, giving their new
  functions neutral defaults.

`AbstractExtNLPModel` is an alias of the bottom of the chain, currently
`AbstractParametricNLPModel`, which is the name Julia prints.
