export ExtNLPModel

"""
    ExtNLPModel(model)

Wrap an ordinary `AbstractNLPModel` as an [`AbstractExtNLPModel`](@ref): a batch of one
instance, with data as one-column matrices. Evaluations are forwarded to `model` on the first
column, and the counters are those of `model`.

Defaults to no parameters (`npar = 0`). The package that defines the type of `model` adds
parameters by extending [`get_par_meta`](@ref) and the parametric functions on
`ExtNLPModel{T, M, <:ModelType}`, e.g. ExaModels for an `ExaModel`.
"""
struct ExtNLPModel{T, M <: AbstractMatrix{T}, N <: AbstractNLPModel{T}} <: AbstractExtNLPModel{T, M}
    meta::BatchedNLPModelMeta{T, M}
    counters::Counters
    model::N
end

function ExtNLPModel(model::AbstractNLPModel)
    col(v) = reshape(copy(v), :, 1)
    meta = BatchedNLPModelMeta(
        get_nvar(model), 1;
        x0 = col(get_x0(model)), lvar = col(get_lvar(model)), uvar = col(get_uvar(model)),
        ncon = get_ncon(model), y0 = col(get_y0(model)), lcon = col(get_lcon(model)), ucon = col(get_ucon(model)),
        nnzj = get_nnzj(model), nnzh = get_nnzh(model), minimize = get_minimize(model), name = get_name(model),
    )
    # share the counters object, so evaluations counted by `model` show on the wrapper
    counters = isdefined(NLPModels, :get_counters) ? NLPModels.get_counters(model) : model.counters
    return ExtNLPModel(meta, counters, model)
end

obj!(nlp::ExtNLPModel, X::AbstractMatrix, f::AbstractVector) = fill!(f, obj(nlp.model, view(X, :, 1)))
grad!(nlp::ExtNLPModel, X::AbstractMatrix, G::AbstractMatrix) = (grad!(nlp.model, view(X, :, 1), view(G, :, 1)); G)
cons!(nlp::ExtNLPModel, X::AbstractMatrix, C::AbstractMatrix) = (cons!(nlp.model, view(X, :, 1), view(C, :, 1)); C)
jac_structure!(nlp::ExtNLPModel, rows::AbstractVector{T}, cols::AbstractVector{T}) where {T} = jac_structure!(nlp.model, rows, cols)
jac_coord!(nlp::ExtNLPModel, X::AbstractMatrix, V::AbstractMatrix) = (jac_coord!(nlp.model, view(X, :, 1), view(V, :, 1)); V)
hess_structure!(nlp::ExtNLPModel, rows::AbstractVector{T}, cols::AbstractVector{T}) where {T} = hess_structure!(nlp.model, rows, cols)
function hess_coord!(nlp::ExtNLPModel, X::AbstractMatrix, Y::AbstractMatrix, σ::AbstractVector, V::AbstractMatrix)
    hess_coord!(nlp.model, view(X, :, 1), view(Y, :, 1), view(V, :, 1); obj_weight = only(Array(σ)))
    return V
end
