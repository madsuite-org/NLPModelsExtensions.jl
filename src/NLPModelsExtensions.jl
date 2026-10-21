"""
    NLPModelsExtensions

Extensions of the `NLPModels` API as one chain of abstract types:

    AbstractNLPModel
    └── AbstractBatchedNLPModel      a batch of NLPs sharing one sparsity pattern
        └── AbstractParametricNLPModel   + derivatives with respect to parameters
              = AbstractExtNLPModel      the type to subtype

A single problem is a batch of size 1 and a problem without parameters has `npar = 0`,
so every model subtypes [`AbstractExtNLPModel`](@ref). Code that consumes models accepts
the least derived type it needs. An ordinary `AbstractNLPModel` is wrapped with
[`ExtNLPModel`](@ref).
"""
module NLPModelsExtensions

using NLPModels
import NLPModels: obj, grad, grad!, cons, cons!, jac_coord, jac_coord!, jac_structure!,
    hess_coord, hess_coord!, hess_structure!

# `obj!` and `get_nbatch` belong to the batch API of NLPModels ≥ 0.21.12. Extend them when
# present so that `using NLPModels, NLPModelsExtensions` does not make the names ambiguous.
@static if isdefined(NLPModels, :obj!)
    import NLPModels: obj!
else
    function obj! end
end
@static if isdefined(NLPModels, :get_nbatch)
    import NLPModels: get_nbatch
else
    function get_nbatch end
end

const LAGRANGIAN_HESSIAN_PAR = raw"""
```math
∂²L/∂x∂θ(x,y) = σ ∂²f/∂x∂θ(x) + \sum_i yᵢ ∂²cᵢ/∂x∂θ(x),
```
with `σ` the objective weight of the instance"""

include("models.jl")
include("utils.jl")

include("batched/meta.jl")
include("batched/api.jl")

include("parametric/meta.jl")
include("parametric/api.jl")
include("parametric/bounds.jl")

include("wrappers/ext_nlp_model.jl")

end
