export get_nbatch, obj!

# Implementers of an `AbstractBatchedNLPModel` provide the in-place methods below; the
# structure is shared by the instances, values have one column per instance.
#
#     obj!(nlp, X, f)                     f: nbatch
#     grad!(nlp, X, G)                    G: nvar × nbatch
#     cons!(nlp, X, C)                    C: ncon × nbatch
#     jac_structure!(nlp, rows, cols)     length nnzj
#     jac_coord!(nlp, X, V)               V: nnzj × nbatch
#     hess_structure!(nlp, rows, cols)    length nnzh
#     hess_coord!(nlp, X, Y, σ, V)        Y: ncon × nbatch, σ: nbatch, V: nnzh × nbatch
#
# `jac_structure(nlp)` and `hess_structure(nlp)` come from `NLPModels`.

"""
    f = obj!(nlp, X, f)

Evaluate the objective of each instance at the columns of `X`, overwriting `f` of length `nbatch`.
"""
obj!

"""
    f = obj(nlp, X)

Evaluate the objective of each instance at the columns of `X`, a vector of length `nbatch`.
"""
function obj(nlp::AbstractBatchedNLPModel, X::AbstractMatrix)
    @sizecheck get_nvar(nlp) get_nbatch(nlp) X
    return obj!(nlp, X, similar(X, get_nbatch(nlp)))
end

"""
    G = grad(nlp, X)

Evaluate the objective gradient of each instance at the columns of `X`, `nvar × nbatch`.
"""
function grad(nlp::AbstractBatchedNLPModel{T, M}, X::AbstractMatrix) where {T, M}
    @sizecheck get_nvar(nlp) get_nbatch(nlp) X
    return grad!(nlp, X, M(undef, get_nvar(nlp), get_nbatch(nlp)))
end

"""
    C = cons(nlp, X)

Evaluate the constraints of each instance at the columns of `X`, `ncon × nbatch`.
"""
function cons(nlp::AbstractBatchedNLPModel{T, M}, X::AbstractMatrix) where {T, M}
    @sizecheck get_nvar(nlp) get_nbatch(nlp) X
    return cons!(nlp, X, M(undef, get_ncon(nlp), get_nbatch(nlp)))
end

"""
    V = jac_coord(nlp, X)

Evaluate the constraints Jacobian of each instance at the columns of `X` in sparse coordinate
format, `nnzj × nbatch` with the structure of `jac_structure(nlp)`.
"""
function jac_coord(nlp::AbstractBatchedNLPModel{T, M}, X::AbstractMatrix) where {T, M}
    @sizecheck get_nvar(nlp) get_nbatch(nlp) X
    return jac_coord!(nlp, X, M(undef, get_nnzj(nlp), get_nbatch(nlp)))
end

"""
    V = hess_coord(nlp, X, Y, σ)

Evaluate the Lagrangian Hessian of each instance at the columns of `(X, Y)` with objective weights
`σ` of length `nbatch`, in sparse coordinate format, `nnzh × nbatch` with the structure of
`hess_structure(nlp)`. Only the lower triangle is returned.
"""
function hess_coord(nlp::AbstractBatchedNLPModel{T, M}, X::AbstractMatrix, Y::AbstractMatrix, σ::AbstractVector) where {T, M}
    @sizecheck get_nvar(nlp) get_nbatch(nlp) X
    @sizecheck get_ncon(nlp) get_nbatch(nlp) Y
    @lencheck get_nbatch(nlp) σ
    return hess_coord!(nlp, X, Y, σ, M(undef, get_nnzh(nlp), get_nbatch(nlp)))
end
