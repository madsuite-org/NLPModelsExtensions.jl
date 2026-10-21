export AbstractBatchedNLPModel, AbstractParametricNLPModel, AbstractExtNLPModel

"""
    AbstractBatchedNLPModel{T, M <: AbstractMatrix{T}} <: AbstractNLPModel{T, M}

A batch of `nbatch` NLPs that share dimensions and the sparsity patterns of the Jacobian and the
Hessian of the Lagrangian; only the numerical data and the bounds differ. Data are matrices of
type `M` with one column per instance, and `get_nvar`, `get_ncon`, `get_nnzj` and `get_nnzh`
are per instance. A single problem is a batch with `nbatch = 1`.

Concrete models store a [`BatchedNLPModelMeta`](@ref) in `nlp.meta` and a `Counters` in
`nlp.counters`, as in `NLPModels`.
"""
abstract type AbstractBatchedNLPModel{T, M <: AbstractMatrix{T}} <: AbstractNLPModel{T, M} end

"""
    AbstractParametricNLPModel{T, M <: AbstractMatrix{T}} <: AbstractBatchedNLPModel{T, M}

A batched NLP with derivatives with respect to parameters, a `npar × nbatch` matrix with one
column per instance. Dimensions come from [`get_par_meta`](@ref), which defaults to no parameters.
"""
abstract type AbstractParametricNLPModel{T, M <: AbstractMatrix{T}} <: AbstractBatchedNLPModel{T, M} end

"""
    AbstractExtNLPModel{T, M <: AbstractMatrix{T}}

The model type to subtype: batched (`nbatch = 1` for a single problem) and parametric
(`npar = 0` without parameters). Other packages extend it with their own subtypes, giving any
new function a neutral default so their models remain usable wherever an `AbstractExtNLPModel`
is accepted.

`AbstractExtNLPModel` is an alias of the bottom of the chain, currently
[`AbstractParametricNLPModel`](@ref), which is the name Julia prints.
"""
const AbstractExtNLPModel = AbstractParametricNLPModel
