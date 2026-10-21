export get_par, set_par!, grad_par, grad_par!
export jac_par_structure!, jac_par_structure, jac_par_coord!, jac_par_coord
export jprod_par, jprod_par!, jtprod_par, jtprod_par!
export hess_par_structure!, hess_par_structure, hess_par_coord!, hess_par_coord
export hprod_par, hprod_par!, htprod_par, htprod_par!

"""
    Θ = get_par(nlp)

Return the parameters, a `npar × nbatch` matrix with one column per instance.
Defaults to an empty matrix when `npar = 0`.
"""
function get_par(nlp::AbstractParametricNLPModel{T, M}) where {T, M}
    get_npar(nlp) == 0 || _not_implemented(get_par, nlp)
    return M(undef, 0, get_nbatch(nlp))
end

"""
    set_par!(nlp, Θ)

Set the parameters to the `npar × nbatch` matrix `Θ`. Defaults to doing nothing when `npar = 0`.
"""
function set_par!(nlp::AbstractParametricNLPModel, Θ)
    get_npar(nlp) == 0 || _not_implemented(set_par!, nlp)
    @sizecheck 0 get_nbatch(nlp) Θ
    return nlp
end

"""
    G = grad_par!(nlp, X, G)

Evaluate ``∂f/∂θ(x)``, the gradient of the objective function with respect to the parameters, at the columns of `X`, overwriting `G` (`npar × nbatch`).
Defaults to doing nothing when `npar = 0`.
"""
function grad_par!(nlp::AbstractParametricNLPModel, X, G)
    get_npar(nlp) == 0 || _not_implemented(grad_par!, nlp)
    return G
end

"""
    G = grad_par(nlp, X)

Evaluate ``∂f/∂θ(x)``, the gradient of the objective function with respect to the parameters, at the columns of `X`, `npar × nbatch`.
"""
function grad_par(nlp::AbstractParametricNLPModel{T, M}, X::AbstractMatrix) where {T, M}
    @sizecheck get_nvar(nlp) get_nbatch(nlp) X
    G = M(undef, get_npar(nlp), get_nbatch(nlp))
    return grad_par!(nlp, X, G)
end

"""
    jac_par_structure!(nlp, rows, cols)

Return the structure of the constraints Jacobian with respect to the parameters ``∂c/∂θ`` in sparse coordinate format in place.
Defaults to an empty structure when `nnzj_par = 0`.
"""
function jac_par_structure!(nlp::AbstractParametricNLPModel, rows, cols)
    get_nnzj_par(nlp) == 0 || _not_implemented(jac_par_structure!, nlp)
    return rows, cols
end

"""
    (rows,cols) = jac_par_structure(nlp)

Return the structure of the constraints Jacobian with respect to the parameters ``∂c/∂θ`` in sparse coordinate format.
"""
function jac_par_structure(nlp::AbstractParametricNLPModel)
    nnzj = get_nnzj_par(nlp)
    rows = Vector{Int}(undef, nnzj)
    cols = Vector{Int}(undef, nnzj)
    return jac_par_structure!(nlp, rows, cols)
end

"""
    V = jac_par_coord!(nlp, X, V)

Evaluate ``∂c/∂θ(x)``, the constraints Jacobian with respect to the parameters, at the columns of `X` in sparse coordinate format, overwriting `V` (`nnzj_par × nbatch`).
Defaults to doing nothing when `nnzj_par = 0`.
"""
function jac_par_coord!(nlp::AbstractParametricNLPModel, X, V)
    get_nnzj_par(nlp) == 0 || _not_implemented(jac_par_coord!, nlp)
    return V
end

"""
    V = jac_par_coord(nlp, X)

Evaluate ``∂c/∂θ(x)``, the constraints Jacobian with respect to the parameters, at the columns of `X` in sparse coordinate format, `nnzj_par × nbatch`.
"""
function jac_par_coord(nlp::AbstractParametricNLPModel{T, M}, X::AbstractMatrix) where {T, M}
    @sizecheck get_nvar(nlp) get_nbatch(nlp) X
    V = M(undef, get_nnzj_par(nlp), get_nbatch(nlp))
    return jac_par_coord!(nlp, X, V)
end

"""
    JV = jprod_par(nlp, X, V)

Evaluate ``∂c/∂θ(x)v``, the Jacobian-vector product with respect to the parameters, at the columns of `X` and `V`, `ncon × nbatch`.
"""
function jprod_par(nlp::AbstractParametricNLPModel{T, M}, X::AbstractMatrix, V::AbstractMatrix) where {T, M}
    JV = M(undef, get_ncon(nlp), get_nbatch(nlp))
    return jprod_par!(nlp, X, V, JV)
end

"""
    JV = jprod_par!(nlp, X, V, JV)

Evaluate ``∂c/∂θ(x)v``, the Jacobian-vector product with respect to the parameters, at the columns of `X` and `V` in place.
This function allocates.
"""
function jprod_par!(nlp::AbstractParametricNLPModel, X::AbstractMatrix, V::AbstractMatrix, JV::AbstractMatrix)
    rows, cols = jac_par_structure(nlp)
    vals = jac_par_coord(nlp, X)
    return jprod_par!(nlp, rows, cols, vals, V, JV)
end

"""
    JV = jprod_par!(nlp, rows, cols, vals, V, JV)

Evaluate ``∂c/∂θ(x)v``, the Jacobian-vector product with respect to the parameters, where the Jacobian of each instance is given by `(rows, cols, vals[:, k])` in triplet format.
"""
function jprod_par!(
        nlp::AbstractParametricNLPModel,
        rows::AbstractVector{<:Integer},
        cols::AbstractVector{<:Integer},
        vals::AbstractMatrix,
        V::AbstractMatrix,
        JV::AbstractMatrix,
    )
    nbatch = get_nbatch(nlp)
    @lencheck get_nnzj_par(nlp) rows cols
    @sizecheck get_nnzj_par(nlp) nbatch vals
    @sizecheck get_npar(nlp) nbatch V
    @sizecheck get_ncon(nlp) nbatch JV
    return _coo_prod!(rows, cols, vals, V, JV)
end

"""
    JtV = jtprod_par(nlp, X, V)

Evaluate ``∂c/∂θ(x)^Tv``, the transposed-Jacobian-vector product with respect to the parameters, at the columns of `X` and `V`, `npar × nbatch`.
"""
function jtprod_par(nlp::AbstractParametricNLPModel{T, M}, X::AbstractMatrix, V::AbstractMatrix) where {T, M}
    JtV = M(undef, get_npar(nlp), get_nbatch(nlp))
    return jtprod_par!(nlp, X, V, JtV)
end

"""
    JtV = jtprod_par!(nlp, X, V, JtV)

Evaluate ``∂c/∂θ(x)^Tv``, the transposed-Jacobian-vector product with respect to the parameters, at the columns of `X` and `V` in place.
This function allocates.
"""
function jtprod_par!(nlp::AbstractParametricNLPModel, X::AbstractMatrix, V::AbstractMatrix, JtV::AbstractMatrix)
    rows, cols = jac_par_structure(nlp)
    vals = jac_par_coord(nlp, X)
    return jtprod_par!(nlp, rows, cols, vals, V, JtV)
end

"""
    JtV = jtprod_par!(nlp, rows, cols, vals, V, JtV)

Evaluate ``∂c/∂θ(x)^Tv``, the transposed-Jacobian-vector product with respect to the parameters, where the Jacobian of each instance is given by `(rows, cols, vals[:, k])` in triplet format.
"""
function jtprod_par!(
        nlp::AbstractParametricNLPModel,
        rows::AbstractVector{<:Integer},
        cols::AbstractVector{<:Integer},
        vals::AbstractMatrix,
        V::AbstractMatrix,
        JtV::AbstractMatrix,
    )
    nbatch = get_nbatch(nlp)
    @lencheck get_nnzj_par(nlp) rows cols
    @sizecheck get_nnzj_par(nlp) nbatch vals
    @sizecheck get_ncon(nlp) nbatch V
    @sizecheck get_npar(nlp) nbatch JtV
    return _coo_prod!(cols, rows, vals, V, JtV)
end

"""
    hess_par_structure!(nlp, rows, cols)

Return the structure of the Lagrangian Hessian with respect to the variables and the parameters ``∂²L/∂x∂θ`` in sparse coordinate format in place.
Defaults to an empty structure when `nnzh_par = 0`.
"""
function hess_par_structure!(nlp::AbstractParametricNLPModel, rows, cols)
    get_nnzh_par(nlp) == 0 || _not_implemented(hess_par_structure!, nlp)
    return rows, cols
end

"""
    (rows,cols) = hess_par_structure(nlp)

Return the structure of the Lagrangian Hessian with respect to the variables and the parameters ``∂²L/∂x∂θ`` in sparse coordinate format.
"""
function hess_par_structure(nlp::AbstractParametricNLPModel)
    nnzh = get_nnzh_par(nlp)
    rows = Vector{Int}(undef, nnzh)
    cols = Vector{Int}(undef, nnzh)
    return hess_par_structure!(nlp, rows, cols)
end

"""
    V = hess_par_coord!(nlp, X, Y, σ, V)

Evaluate the Lagrangian Hessian with respect to the variables and the parameters ``∂²L/∂x∂θ(x,y)`` at the columns of `(X, Y)` in sparse coordinate format,
with the objective of instance `k` scaled by `σ[k]`, i.e.,
$(LAGRANGIAN_HESSIAN_PAR), overwriting `V` (`nnzh_par × nbatch`).
Defaults to doing nothing when `nnzh_par = 0`.
"""
function hess_par_coord!(nlp::AbstractParametricNLPModel, X, Y, σ, V)
    get_nnzh_par(nlp) == 0 || _not_implemented(hess_par_coord!, nlp)
    return V
end

"""
    V = hess_par_coord(nlp, X, Y, σ)

Evaluate the Lagrangian Hessian with respect to the variables and the parameters ``∂²L/∂x∂θ(x,y)`` at the columns of `(X, Y)` in sparse coordinate format,
with the objective of instance `k` scaled by `σ[k]`, i.e.,
$(LAGRANGIAN_HESSIAN_PAR), `nnzh_par × nbatch`.
"""
function hess_par_coord(nlp::AbstractParametricNLPModel{T, M}, X::AbstractMatrix, Y::AbstractMatrix, σ::AbstractVector) where {T, M}
    @sizecheck get_nvar(nlp) get_nbatch(nlp) X
    @sizecheck get_ncon(nlp) get_nbatch(nlp) Y
    @lencheck get_nbatch(nlp) σ
    V = M(undef, get_nnzh_par(nlp), get_nbatch(nlp))
    return hess_par_coord!(nlp, X, Y, σ, V)
end

"""
    HV = hprod_par(nlp, X, Y, V, σ)

Evaluate the product of the Lagrangian Hessian with respect to the variables and the parameters ``∂²L/∂x∂θ(x,y)`` at the columns of `(X, Y)` with the columns of `V`,
with the objective of instance `k` scaled by `σ[k]`, where the Lagrangian Hessian ``∂²L/∂x∂θ(x,y)`` is
$(LAGRANGIAN_HESSIAN_PAR). Returns `nvar × nbatch`.
"""
function hprod_par(nlp::AbstractParametricNLPModel{T, M}, X::AbstractMatrix, Y::AbstractMatrix, V::AbstractMatrix, σ::AbstractVector) where {T, M}
    HV = M(undef, get_nvar(nlp), get_nbatch(nlp))
    return hprod_par!(nlp, X, Y, V, σ, HV)
end

"""
    HV = hprod_par!(nlp, X, Y, V, σ, HV)

Evaluate the product of the Lagrangian Hessian with respect to the variables and the parameters ``∂²L/∂x∂θ(x,y)`` at the columns of `(X, Y)` with the columns of `V` in place,
with the objective of instance `k` scaled by `σ[k]`, where the Lagrangian Hessian ``∂²L/∂x∂θ(x,y)`` is
$(LAGRANGIAN_HESSIAN_PAR).
This function allocates.
"""
function hprod_par!(
        nlp::AbstractParametricNLPModel, X::AbstractMatrix, Y::AbstractMatrix, V::AbstractMatrix, σ::AbstractVector, HV::AbstractMatrix,
    )
    rows, cols = hess_par_structure(nlp)
    vals = hess_par_coord(nlp, X, Y, σ)
    return hprod_par!(nlp, rows, cols, vals, V, HV)
end

"""
    HV = hprod_par!(nlp, rows, cols, vals, V, HV)

Evaluate the product of the Lagrangian Hessian with respect to the variables and the parameters ``∂²L/∂x∂θ``, given for each instance by `(rows, cols, vals[:, k])` in triplet format, with the columns of `V` in place.
"""
function hprod_par!(
        nlp::AbstractParametricNLPModel,
        rows::AbstractVector{<:Integer},
        cols::AbstractVector{<:Integer},
        vals::AbstractMatrix,
        V::AbstractMatrix,
        HV::AbstractMatrix,
    )
    nbatch = get_nbatch(nlp)
    @lencheck get_nnzh_par(nlp) rows cols
    @sizecheck get_nnzh_par(nlp) nbatch vals
    @sizecheck get_npar(nlp) nbatch V
    @sizecheck get_nvar(nlp) nbatch HV
    return _coo_prod!(rows, cols, vals, V, HV)
end

"""
    HtV = htprod_par(nlp, X, Y, V, σ)

Evaluate the product of the transposed Lagrangian Hessian with respect to the variables and the parameters ``∂²L/∂x∂θ(x,y)^T`` at the columns of `(X, Y)` with the columns of `V`,
with the objective of instance `k` scaled by `σ[k]`, where the Lagrangian Hessian ``∂²L/∂x∂θ(x,y)`` is
$(LAGRANGIAN_HESSIAN_PAR). Returns `npar × nbatch`.
"""
function htprod_par(nlp::AbstractParametricNLPModel{T, M}, X::AbstractMatrix, Y::AbstractMatrix, V::AbstractMatrix, σ::AbstractVector) where {T, M}
    HtV = M(undef, get_npar(nlp), get_nbatch(nlp))
    return htprod_par!(nlp, X, Y, V, σ, HtV)
end

"""
    HtV = htprod_par!(nlp, X, Y, V, σ, HtV)

Evaluate the product of the transposed Lagrangian Hessian with respect to the variables and the parameters ``∂²L/∂x∂θ(x,y)^T`` at the columns of `(X, Y)` with the columns of `V` in place,
with the objective of instance `k` scaled by `σ[k]`, where the Lagrangian Hessian ``∂²L/∂x∂θ(x,y)`` is
$(LAGRANGIAN_HESSIAN_PAR).
This function allocates.
"""
function htprod_par!(
        nlp::AbstractParametricNLPModel, X::AbstractMatrix, Y::AbstractMatrix, V::AbstractMatrix, σ::AbstractVector, HtV::AbstractMatrix,
    )
    rows, cols = hess_par_structure(nlp)
    vals = hess_par_coord(nlp, X, Y, σ)
    return htprod_par!(nlp, rows, cols, vals, V, HtV)
end

"""
    HtV = htprod_par!(nlp, rows, cols, vals, V, HtV)

Evaluate the product of the transposed Lagrangian Hessian with respect to the variables and the parameters ``∂²L/∂x∂θ^T``, given for each instance by `(rows, cols, vals[:, k])` in triplet format, with the columns of `V` in place.
"""
function htprod_par!(
        nlp::AbstractParametricNLPModel,
        rows::AbstractVector{<:Integer},
        cols::AbstractVector{<:Integer},
        vals::AbstractMatrix,
        V::AbstractMatrix,
        HtV::AbstractMatrix,
    )
    nbatch = get_nbatch(nlp)
    @lencheck get_nnzh_par(nlp) rows cols
    @sizecheck get_nnzh_par(nlp) nbatch vals
    @sizecheck get_nvar(nlp) nbatch V
    @sizecheck get_npar(nlp) nbatch HtV
    return _coo_prod!(cols, rows, vals, V, HtV)
end
