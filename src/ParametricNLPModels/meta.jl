export ParametricNLPModelMeta, get_par_meta

"""
    ParametricNLPModelMeta(; kwargs...)

Metadata for the derivatives with respect to the parameters.

- `npar`: number of parameters
- `nnzo_par`: number of nonzeros in ``∂f/∂θ``
- `nnzj_par`: number of nonzeros in ``∂c/∂θ``
- `nnzh_par`: number of nonzeros in ``∂²L/∂x∂θ``
- `nnzj_lvar_par`, `nnzj_uvar_par`, `nnzj_lcon_par`, `nnzj_ucon_par`: number of nonzeros in ``∂x^ℓ/∂θ``, ``∂x^u/∂θ``, ``∂c^ℓ/∂θ``, ``∂c^u/∂θ``
- `grad_par_available`, `jac_par_available`, `hess_par_available`, `jprod_par_available`, `jtprod_par_available`, `hprod_par_available`, `htprod_par_available`: whether the function is implemented
"""
struct ParametricNLPModelMeta
    npar::Int
    nnzo_par::Int
    nnzj_par::Int
    nnzh_par::Int
    nnzj_lvar_par::Int
    nnzj_uvar_par::Int
    nnzj_lcon_par::Int
    nnzj_ucon_par::Int
    grad_par_available::Bool
    jac_par_available::Bool
    hess_par_available::Bool
    jprod_par_available::Bool
    jtprod_par_available::Bool
    hprod_par_available::Bool
    htprod_par_available::Bool
end

function ParametricNLPModelMeta(;
        npar = 0, nnzo_par = 0, nnzj_par = 0, nnzh_par = 0,
        nnzj_lvar_par = 0, nnzj_uvar_par = 0, nnzj_lcon_par = 0, nnzj_ucon_par = 0,
        grad_par_available = true, jac_par_available = true, hess_par_available = true,
        jprod_par_available = true, jtprod_par_available = true,
        hprod_par_available = true, htprod_par_available = true,
    )
    dims = (npar, nnzo_par, nnzj_par, nnzh_par, nnzj_lvar_par, nnzj_uvar_par, nnzj_lcon_par, nnzj_ucon_par)
    if any(<(0), dims)
        error("Nonsensical dimensions")
    end
    return ParametricNLPModelMeta(
        dims..., grad_par_available, jac_par_available, hess_par_available,
        jprod_par_available, jtprod_par_available, hprod_par_available, htprod_par_available,
    )
end

"""
    get_par_meta(nlp)

Return the `ParametricNLPModelMeta` of `nlp`. Defaults to no parameters.
"""
get_par_meta(nlp::AbstractNLPModel) = ParametricNLPModelMeta()

for field in fieldnames(ParametricNLPModelMeta)
    meth = Symbol("get_", field)
    @eval begin
        @doc """
            $($meth)(nlp)
            $($meth)(par_meta)

        Return the value `$($(QuoteNode(field)))` from `par_meta` or `get_par_meta(nlp)`.
        """
        $meth(par_meta::ParametricNLPModelMeta) = getproperty(par_meta, $(QuoteNode(field)))
        $meth(nlp::AbstractNLPModel) = $meth(get_par_meta(nlp))
        export $meth
    end
end
