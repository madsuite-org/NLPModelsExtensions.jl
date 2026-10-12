export get_par, set_par!, grad_par, grad_par!
export jac_par_structure!, jac_par_structure, jac_par_coord!, jac_par_coord
export jac_par, jac_par_dense!, jprod_par, jprod_par!, jtprod_par, jtprod_par!
export hess_par_structure!, hess_par_structure, hess_par_coord!, hess_par_coord
export hess_par, hess_par_dense!, hprod_par, hprod_par!, htprod_par, htprod_par!

"""
    θ = get_par(nlp)

Return the parameter vector.
"""
function get_par end

"""
    set_par!(nlp, θ)

Set the parameter vector to `θ`.
"""
function set_par! end

"""
    g = grad_par!(nlp, x, g)

Evaluate ``∂f/∂θ(x)``, the gradient of the objective function with respect to the parameters at `x` in place.
"""
function grad_par! end

"""
    g = grad_par(nlp, x)

Evaluate ``∂f/∂θ(x)``, the gradient of the objective function with respect to the parameters at `x`.
"""
function grad_par(nlp::AbstractNLPModel{T, S}, x::AbstractVector) where {T, S}
    @lencheck get_nvar(nlp) x
    g = S(undef, get_npar(nlp))
    return grad_par!(nlp, x, g)
end

"""
    jac_par_structure!(nlp, rows, cols)

Return the structure of the constraints Jacobian with respect to the parameters ``∂c/∂θ`` in sparse coordinate format in place.
"""
function jac_par_structure! end

"""
    (rows,cols) = jac_par_structure(nlp)

Return the structure of the constraints Jacobian with respect to the parameters ``∂c/∂θ`` in sparse coordinate format.
"""
function jac_par_structure(nlp::AbstractNLPModel)
    nnzj = get_nnzj_par(nlp)
    rows = similar(get_x0(nlp), Int, nnzj)
    cols = similar(get_x0(nlp), Int, nnzj)
    return jac_par_structure!(nlp, rows, cols)
end

"""
    vals = jac_par_coord!(nlp, x, vals)

Evaluate ``∂c/∂θ(x)``, the constraints Jacobian with respect to the parameters at `x` in sparse coordinate format, overwriting `vals`.
"""
function jac_par_coord! end

"""
    vals = jac_par_coord(nlp, x)

Evaluate ``∂c/∂θ(x)``, the constraints Jacobian with respect to the parameters at `x` in sparse coordinate format.
"""
function jac_par_coord(nlp::AbstractNLPModel{T, S}, x::AbstractVector) where {T, S}
    @lencheck get_nvar(nlp) x
    vals = S(undef, get_nnzj_par(nlp))
    return jac_par_coord!(nlp, x, vals)
end

"""
    Jθ = jac_par(nlp, x)

Evaluate ``∂c/∂θ(x)``, the constraints Jacobian with respect to the parameters at `x` as a sparse matrix.
"""
function jac_par(nlp::AbstractNLPModel, x::AbstractVector)
    @lencheck get_nvar(nlp) x
    rows, cols = jac_par_structure(nlp)
    vals = jac_par_coord(nlp, x)
    return sparse(rows, cols, vals, get_ncon(nlp), get_npar(nlp))
end

"""
    Jθ = jac_par_dense!(nlp, x, Jθ)

Evaluate ``∂c/∂θ(x)``, the constraints Jacobian with respect to the parameters at `x` in dense format, overwriting `Jθ`.
"""
function jac_par_dense!(nlp::AbstractNLPModel, x::AbstractVector, Jθ::AbstractMatrix)
    @lencheck get_nvar(nlp) x
    rows, cols = jac_par_structure(nlp)
    vals = jac_par_coord(nlp, x)
    return _dense!(Jθ, rows, cols, vals)
end

"""
    Jv = jprod_par(nlp, x, v)

Evaluate ``∂c/∂θ(x)v``, the Jacobian-vector product with respect to the parameters at `x`.
"""
function jprod_par(nlp::AbstractNLPModel{T, S}, x::AbstractVector, v::AbstractVector) where {T, S}
    @lencheck get_nvar(nlp) x
    @lencheck get_npar(nlp) v
    Jv = S(undef, get_ncon(nlp))
    return jprod_par!(nlp, x, v, Jv)
end

"""
    Jv = jprod_par!(nlp, x, v, Jv)

Evaluate ``∂c/∂θ(x)v``, the Jacobian-vector product with respect to the parameters at `x` in place.
This function allocates.
"""
function jprod_par!(nlp::AbstractNLPModel, x::AbstractVector, v::AbstractVector, Jv::AbstractVector)
    @lencheck get_nvar(nlp) x
    rows, cols = jac_par_structure(nlp)
    vals = jac_par_coord(nlp, x)
    return jprod_par!(nlp, rows, cols, vals, v, Jv)
end

"""
    Jv = jprod_par!(nlp, rows, cols, vals, v, Jv)

Evaluate ``∂c/∂θ(x)v``, the Jacobian-vector product with respect to the parameters, where the Jacobian is given by `(rows, cols, vals)` in triplet format.
"""
function jprod_par!(
        nlp::AbstractNLPModel,
        rows::AbstractVector{<:Integer},
        cols::AbstractVector{<:Integer},
        vals::AbstractVector,
        v::AbstractVector,
        Jv::AbstractVector,
    )
    @lencheck get_nnzj_par(nlp) rows cols vals
    @lencheck get_npar(nlp) v
    @lencheck get_ncon(nlp) Jv
    return coo_prod!(rows, cols, vals, v, Jv)
end

"""
    Jtv = jtprod_par(nlp, x, v)

Evaluate ``∂c/∂θ(x)^Tv``, the transposed-Jacobian-vector product with respect to the parameters at `x`.
"""
function jtprod_par(nlp::AbstractNLPModel{T, S}, x::AbstractVector, v::AbstractVector) where {T, S}
    @lencheck get_nvar(nlp) x
    @lencheck get_ncon(nlp) v
    Jtv = S(undef, get_npar(nlp))
    return jtprod_par!(nlp, x, v, Jtv)
end

"""
    Jtv = jtprod_par!(nlp, x, v, Jtv)

Evaluate ``∂c/∂θ(x)^Tv``, the transposed-Jacobian-vector product with respect to the parameters at `x` in place.
This function allocates.
"""
function jtprod_par!(nlp::AbstractNLPModel, x::AbstractVector, v::AbstractVector, Jtv::AbstractVector)
    @lencheck get_nvar(nlp) x
    rows, cols = jac_par_structure(nlp)
    vals = jac_par_coord(nlp, x)
    return jtprod_par!(nlp, rows, cols, vals, v, Jtv)
end

"""
    Jtv = jtprod_par!(nlp, rows, cols, vals, v, Jtv)

Evaluate ``∂c/∂θ(x)^Tv``, the transposed-Jacobian-vector product with respect to the parameters, where the Jacobian is given by `(rows, cols, vals)` in triplet format.
"""
function jtprod_par!(
        nlp::AbstractNLPModel,
        rows::AbstractVector{<:Integer},
        cols::AbstractVector{<:Integer},
        vals::AbstractVector,
        v::AbstractVector,
        Jtv::AbstractVector,
    )
    @lencheck get_nnzj_par(nlp) rows cols vals
    @lencheck get_ncon(nlp) v
    @lencheck get_npar(nlp) Jtv
    return coo_prod!(cols, rows, vals, v, Jtv)
end

"""
    hess_par_structure!(nlp, rows, cols)

Return the structure of the Lagrangian Hessian with respect to the variables and the parameters ``∂²L/∂x∂θ`` in sparse coordinate format in place.
"""
function hess_par_structure! end

"""
    (rows,cols) = hess_par_structure(nlp)

Return the structure of the Lagrangian Hessian with respect to the variables and the parameters ``∂²L/∂x∂θ`` in sparse coordinate format.
"""
function hess_par_structure(nlp::AbstractNLPModel)
    nnzh = get_nnzh_par(nlp)
    rows = similar(get_x0(nlp), Int, nnzh)
    cols = similar(get_x0(nlp), Int, nnzh)
    return hess_par_structure!(nlp, rows, cols)
end

"""
    vals = hess_par_coord!(nlp, x, vals; obj_weight=1.0)

Evaluate the objective Hessian with respect to the variables and the parameters ``∂²f/∂x∂θ(x)`` at `x` in sparse coordinate format,
with objective function scaled by `obj_weight`, i.e.,
$(OBJECTIVE_HESSIAN_PAR), overwriting `vals`.
"""
function hess_par_coord!(nlp::AbstractNLPModel{T, S}, x::AbstractVector, vals::AbstractVector; obj_weight = one(eltype(x))) where {T, S}
    @lencheck get_nvar(nlp) x
    @lencheck get_nnzh_par(nlp) vals
    y = fill!(S(undef, get_ncon(nlp)), 0)
    return hess_par_coord!(nlp, x, y, vals; obj_weight)
end

"""
    vals = hess_par_coord!(nlp, x, y, vals; obj_weight=1.0)

Evaluate the Lagrangian Hessian with respect to the variables and the parameters ``∂²L/∂x∂θ(x,y)`` at `(x,y)` in sparse coordinate format,
with objective function scaled by `obj_weight`, i.e.,
$(LAGRANGIAN_HESSIAN_PAR), overwriting `vals`.
"""
function hess_par_coord! end

"""
    vals = hess_par_coord(nlp, x; obj_weight=1.0)

Evaluate the objective Hessian with respect to the variables and the parameters ``∂²f/∂x∂θ(x)`` at `x` in sparse coordinate format,
with objective function scaled by `obj_weight`, i.e.,
$(OBJECTIVE_HESSIAN_PAR).
"""
function hess_par_coord(nlp::AbstractNLPModel{T, S}, x::AbstractVector; obj_weight = one(eltype(x))) where {T, S}
    @lencheck get_nvar(nlp) x
    vals = S(undef, get_nnzh_par(nlp))
    return hess_par_coord!(nlp, x, vals; obj_weight)
end

"""
    vals = hess_par_coord(nlp, x, y; obj_weight=1.0)

Evaluate the Lagrangian Hessian with respect to the variables and the parameters ``∂²L/∂x∂θ(x,y)`` at `(x,y)` in sparse coordinate format,
with objective function scaled by `obj_weight`, i.e.,
$(LAGRANGIAN_HESSIAN_PAR).
"""
function hess_par_coord(nlp::AbstractNLPModel{T, S}, x::AbstractVector, y::AbstractVector; obj_weight = one(eltype(x))) where {T, S}
    @lencheck get_nvar(nlp) x
    @lencheck get_ncon(nlp) y
    vals = S(undef, get_nnzh_par(nlp))
    return hess_par_coord!(nlp, x, y, vals; obj_weight)
end

"""
    Hxθ = hess_par(nlp, x; obj_weight=1.0)

Evaluate the objective Hessian with respect to the variables and the parameters ``∂²f/∂x∂θ(x)`` at `x` as a sparse matrix,
with objective function scaled by `obj_weight`, i.e.,
$(OBJECTIVE_HESSIAN_PAR).
"""
function hess_par(nlp::AbstractNLPModel, x::AbstractVector; obj_weight = one(eltype(x)))
    @lencheck get_nvar(nlp) x
    rows, cols = hess_par_structure(nlp)
    vals = hess_par_coord(nlp, x; obj_weight)
    return sparse(rows, cols, vals, get_nvar(nlp), get_npar(nlp))
end

"""
    Hxθ = hess_par(nlp, x, y; obj_weight=1.0)

Evaluate the Lagrangian Hessian with respect to the variables and the parameters ``∂²L/∂x∂θ(x,y)`` at `(x,y)` as a sparse matrix,
with objective function scaled by `obj_weight`, i.e.,
$(LAGRANGIAN_HESSIAN_PAR).
"""
function hess_par(nlp::AbstractNLPModel, x::AbstractVector, y::AbstractVector; obj_weight = one(eltype(x)))
    @lencheck get_nvar(nlp) x
    @lencheck get_ncon(nlp) y
    rows, cols = hess_par_structure(nlp)
    vals = hess_par_coord(nlp, x, y; obj_weight)
    return sparse(rows, cols, vals, get_nvar(nlp), get_npar(nlp))
end

"""
    Hxθ = hess_par_dense!(nlp, x, Hxθ; obj_weight=1.0)

Evaluate the objective Hessian with respect to the variables and the parameters ``∂²f/∂x∂θ(x)`` at `x` in dense format, overwriting `Hxθ`,
with objective function scaled by `obj_weight`, i.e.,
$(OBJECTIVE_HESSIAN_PAR).
"""
function hess_par_dense!(nlp::AbstractNLPModel, x::AbstractVector, Hxθ::AbstractMatrix; obj_weight = one(eltype(x)))
    @lencheck get_nvar(nlp) x
    rows, cols = hess_par_structure(nlp)
    vals = hess_par_coord(nlp, x; obj_weight)
    return _dense!(Hxθ, rows, cols, vals)
end

"""
    Hxθ = hess_par_dense!(nlp, x, y, Hxθ; obj_weight=1.0)

Evaluate the Lagrangian Hessian with respect to the variables and the parameters ``∂²L/∂x∂θ(x,y)`` at `(x,y)` in dense format, overwriting `Hxθ`,
with objective function scaled by `obj_weight`, i.e.,
$(LAGRANGIAN_HESSIAN_PAR).
"""
function hess_par_dense!(
        nlp::AbstractNLPModel, x::AbstractVector, y::AbstractVector, Hxθ::AbstractMatrix; obj_weight = one(eltype(x))
    )
    @lencheck get_nvar(nlp) x
    @lencheck get_ncon(nlp) y
    rows, cols = hess_par_structure(nlp)
    vals = hess_par_coord(nlp, x, y; obj_weight)
    return _dense!(Hxθ, rows, cols, vals)
end

"""
    Hv = hprod_par(nlp, x, v; obj_weight=1.0)

Evaluate the product of the objective Hessian with respect to the variables and the parameters ``∂²f/∂x∂θ(x)`` at `x` with the vector `v`,
with objective function scaled by `obj_weight`, where the objective Hessian ``∂²f/∂x∂θ(x)`` is
$(OBJECTIVE_HESSIAN_PAR).
"""
function hprod_par(nlp::AbstractNLPModel{T, S}, x::AbstractVector, v::AbstractVector; obj_weight = one(eltype(x))) where {T, S}
    @lencheck get_nvar(nlp) x
    @lencheck get_npar(nlp) v
    Hv = S(undef, get_nvar(nlp))
    return hprod_par!(nlp, x, v, Hv; obj_weight)
end

"""
    Hv = hprod_par(nlp, x, y, v; obj_weight=1.0)

Evaluate the product of the Lagrangian Hessian with respect to the variables and the parameters ``∂²L/∂x∂θ(x,y)`` at `(x,y)` with the vector `v`,
with objective function scaled by `obj_weight`, where the Lagrangian Hessian ``∂²L/∂x∂θ(x,y)`` is
$(LAGRANGIAN_HESSIAN_PAR).
"""
function hprod_par(nlp::AbstractNLPModel{T, S}, x::AbstractVector, y::AbstractVector, v::AbstractVector; obj_weight = one(eltype(x))) where {T, S}
    @lencheck get_nvar(nlp) x
    @lencheck get_ncon(nlp) y
    @lencheck get_npar(nlp) v
    Hv = S(undef, get_nvar(nlp))
    return hprod_par!(nlp, x, y, v, Hv; obj_weight)
end

"""
    Hv = hprod_par!(nlp, x, v, Hv; obj_weight=1.0)

Evaluate the product of the objective Hessian with respect to the variables and the parameters ``∂²f/∂x∂θ(x)`` at `x` with the vector `v` in place,
with objective function scaled by `obj_weight`, where the objective Hessian ``∂²f/∂x∂θ(x)`` is
$(OBJECTIVE_HESSIAN_PAR).
This function allocates.
"""
function hprod_par!(nlp::AbstractNLPModel{T, S}, x::AbstractVector, v::AbstractVector, Hv::AbstractVector; obj_weight = one(eltype(x))) where {T, S}
    @lencheck get_nvar(nlp) x Hv
    @lencheck get_npar(nlp) v
    y = fill!(S(undef, get_ncon(nlp)), 0)
    return hprod_par!(nlp, x, y, v, Hv; obj_weight)
end

"""
    Hv = hprod_par!(nlp, x, y, v, Hv; obj_weight=1.0)

Evaluate the product of the Lagrangian Hessian with respect to the variables and the parameters ``∂²L/∂x∂θ(x,y)`` at `(x,y)` with the vector `v` in place,
with objective function scaled by `obj_weight`, where the Lagrangian Hessian ``∂²L/∂x∂θ(x,y)`` is
$(LAGRANGIAN_HESSIAN_PAR).
This function allocates.
"""
function hprod_par!(
        nlp::AbstractNLPModel, x::AbstractVector, y::AbstractVector, v::AbstractVector, Hv::AbstractVector; obj_weight = one(eltype(x))
    )
    @lencheck get_nvar(nlp) x
    @lencheck get_ncon(nlp) y
    rows, cols = hess_par_structure(nlp)
    vals = hess_par_coord(nlp, x, y; obj_weight)
    return hprod_par!(nlp, rows, cols, vals, v, Hv)
end

"""
    Hv = hprod_par!(nlp, rows, cols, vals, v, Hv)

Evaluate the product of the Lagrangian Hessian with respect to the variables and the parameters ``∂²L/∂x∂θ`` given by `(rows, cols, vals)` in triplet format with the vector `v` in place.
"""
function hprod_par!(
        nlp::AbstractNLPModel,
        rows::AbstractVector{<:Integer},
        cols::AbstractVector{<:Integer},
        vals::AbstractVector,
        v::AbstractVector,
        Hv::AbstractVector,
    )
    @lencheck get_nnzh_par(nlp) rows cols vals
    @lencheck get_npar(nlp) v
    @lencheck get_nvar(nlp) Hv
    return coo_prod!(rows, cols, vals, v, Hv)
end

"""
    Htv = htprod_par(nlp, x, v; obj_weight=1.0)

Evaluate the product of the transposed objective Hessian with respect to the variables and the parameters ``∂²f/∂x∂θ(x)^T`` at `x` with the vector `v`,
with objective function scaled by `obj_weight`, where the objective Hessian ``∂²f/∂x∂θ(x)`` is
$(OBJECTIVE_HESSIAN_PAR).
"""
function htprod_par(nlp::AbstractNLPModel{T, S}, x::AbstractVector, v::AbstractVector; obj_weight = one(eltype(x))) where {T, S}
    @lencheck get_nvar(nlp) x v
    Htv = S(undef, get_npar(nlp))
    return htprod_par!(nlp, x, v, Htv; obj_weight)
end

"""
    Htv = htprod_par(nlp, x, y, v; obj_weight=1.0)

Evaluate the product of the transposed Lagrangian Hessian with respect to the variables and the parameters ``∂²L/∂x∂θ(x,y)^T`` at `(x,y)` with the vector `v`,
with objective function scaled by `obj_weight`, where the Lagrangian Hessian ``∂²L/∂x∂θ(x,y)`` is
$(LAGRANGIAN_HESSIAN_PAR).
"""
function htprod_par(nlp::AbstractNLPModel{T, S}, x::AbstractVector, y::AbstractVector, v::AbstractVector; obj_weight = one(eltype(x))) where {T, S}
    @lencheck get_nvar(nlp) x v
    @lencheck get_ncon(nlp) y
    Htv = S(undef, get_npar(nlp))
    return htprod_par!(nlp, x, y, v, Htv; obj_weight)
end

"""
    Htv = htprod_par!(nlp, x, v, Htv; obj_weight=1.0)

Evaluate the product of the transposed objective Hessian with respect to the variables and the parameters ``∂²f/∂x∂θ(x)^T`` at `x` with the vector `v` in place,
with objective function scaled by `obj_weight`, where the objective Hessian ``∂²f/∂x∂θ(x)`` is
$(OBJECTIVE_HESSIAN_PAR).
This function allocates.
"""
function htprod_par!(nlp::AbstractNLPModel{T, S}, x::AbstractVector, v::AbstractVector, Htv::AbstractVector; obj_weight = one(eltype(x))) where {T, S}
    @lencheck get_nvar(nlp) x v
    @lencheck get_npar(nlp) Htv
    y = fill!(S(undef, get_ncon(nlp)), 0)
    return htprod_par!(nlp, x, y, v, Htv; obj_weight)
end

"""
    Htv = htprod_par!(nlp, x, y, v, Htv; obj_weight=1.0)

Evaluate the product of the transposed Lagrangian Hessian with respect to the variables and the parameters ``∂²L/∂x∂θ(x,y)^T`` at `(x,y)` with the vector `v` in place,
with objective function scaled by `obj_weight`, where the Lagrangian Hessian ``∂²L/∂x∂θ(x,y)`` is
$(LAGRANGIAN_HESSIAN_PAR).
This function allocates.
"""
function htprod_par!(
        nlp::AbstractNLPModel, x::AbstractVector, y::AbstractVector, v::AbstractVector, Htv::AbstractVector; obj_weight = one(eltype(x))
    )
    @lencheck get_nvar(nlp) x
    @lencheck get_ncon(nlp) y
    rows, cols = hess_par_structure(nlp)
    vals = hess_par_coord(nlp, x, y; obj_weight)
    return htprod_par!(nlp, rows, cols, vals, v, Htv)
end

"""
    Htv = htprod_par!(nlp, rows, cols, vals, v, Htv)

Evaluate the product of the transposed Lagrangian Hessian with respect to the variables and the parameters ``∂²L/∂x∂θ^T`` given by `(rows, cols, vals)` in triplet format with the vector `v` in place.
"""
function htprod_par!(
        nlp::AbstractNLPModel,
        rows::AbstractVector{<:Integer},
        cols::AbstractVector{<:Integer},
        vals::AbstractVector,
        v::AbstractVector,
        Htv::AbstractVector,
    )
    @lencheck get_nnzh_par(nlp) rows cols vals
    @lencheck get_nvar(nlp) v
    @lencheck get_npar(nlp) Htv
    return coo_prod!(cols, rows, vals, v, Htv)
end

function _dense!(A, rows, cols, vals)
    fill!(A, zero(eltype(A)))
    for k in eachindex(rows, cols, vals)
        A[rows[k], cols[k]] += vals[k]
    end
    return A
end
