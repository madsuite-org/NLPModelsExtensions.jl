# ParametricNLPModels.jl

Parametric sensitivity interface for [NLPModels.jl](https://github.com/JuliaSmoothOptimizers/NLPModels.jl).

```math
\begin{aligned}
  \min_{x} \; & f(x, θ) \\
  \text{subject to} \quad & c^ℓ(θ) \leq c(x, θ) \leq c^u(θ) \\
                          & x^ℓ(θ) \leq x \leq x^u(θ)
\end{aligned}
```

- ``∂f/∂θ``, the objective gradient with respect to the parameters, `npar`
- ``∂c/∂θ``, the constraints Jacobian with respect to the parameters, `ncon × npar`
- ``∂²L/∂x∂θ``, the Lagrangian Hessian with respect to the variables and the parameters, `nvar × npar`
- ``∂x^ℓ/∂θ``, ``∂x^u/∂θ``, ``∂c^ℓ/∂θ``, ``∂c^u/∂θ``, the bounds with respect to the parameters, `nvar × npar` and `ncon × npar`

Dimensions and availability come from [`ParametricNLPModelMeta`](@ref), returned by [`get_par_meta`](@ref). 

## Reference guide

The naming follows NLPModels.jl conventions:

- `!` means in place
- `_par` means with respect to the parameters
- `_coord` means coordinate format
- `_dense` means dense format
- `prod` means matrix-vector product, `tprod` the transposed product

| Function | ParametricNLPModels function |
|---|---|
| ``θ`` | [`get_npar`](@ref), [`get_par`](@ref), [`set_par!`](@ref) |
| ``∂f/∂θ`` | [`get_nnzo_par`](@ref), [`grad_par`](@ref), [`grad_par!`](@ref) |
| ``∂c/∂θ`` | [`get_nnzj_par`](@ref), [`jac_par`](@ref), [`jac_par_structure`](@ref), [`jac_par_structure!`](@ref), [`jac_par_coord`](@ref), [`jac_par_coord!`](@ref), [`jac_par_dense!`](@ref), [`jprod_par`](@ref), [`jprod_par!`](@ref), [`jtprod_par`](@ref), [`jtprod_par!`](@ref) |
| ``∂²L/∂x∂θ`` | [`get_nnzh_par`](@ref), [`hess_par`](@ref), [`hess_par_structure`](@ref), [`hess_par_structure!`](@ref), [`hess_par_coord`](@ref), [`hess_par_coord!`](@ref), [`hess_par_dense!`](@ref), [`hprod_par`](@ref), [`hprod_par!`](@ref), [`htprod_par`](@ref), [`htprod_par!`](@ref) |
| ``∂x^ℓ/∂θ`` | [`get_nnzj_lvar_par`](@ref), [`lvar_jac_par`](@ref), [`lvar_jac_par_structure`](@ref), [`lvar_jac_par_structure!`](@ref), [`lvar_jac_par_coord`](@ref), [`lvar_jac_par_coord!`](@ref), [`lvar_jac_par_dense!`](@ref), [`lvar_jprod_par`](@ref), [`lvar_jprod_par!`](@ref), [`lvar_jtprod_par`](@ref), [`lvar_jtprod_par!`](@ref) |
| ``∂x^u/∂θ`` | [`get_nnzj_uvar_par`](@ref), [`uvar_jac_par`](@ref), [`uvar_jac_par_structure`](@ref), [`uvar_jac_par_structure!`](@ref), [`uvar_jac_par_coord`](@ref), [`uvar_jac_par_coord!`](@ref), [`uvar_jac_par_dense!`](@ref), [`uvar_jprod_par`](@ref), [`uvar_jprod_par!`](@ref), [`uvar_jtprod_par`](@ref), [`uvar_jtprod_par!`](@ref) |
| ``∂c^ℓ/∂θ`` | [`get_nnzj_lcon_par`](@ref), [`lcon_jac_par`](@ref), [`lcon_jac_par_structure`](@ref), [`lcon_jac_par_structure!`](@ref), [`lcon_jac_par_coord`](@ref), [`lcon_jac_par_coord!`](@ref), [`lcon_jac_par_dense!`](@ref), [`lcon_jprod_par`](@ref), [`lcon_jprod_par!`](@ref), [`lcon_jtprod_par`](@ref), [`lcon_jtprod_par!`](@ref) |
| ``∂c^u/∂θ`` | [`get_nnzj_ucon_par`](@ref), [`ucon_jac_par`](@ref), [`ucon_jac_par_structure`](@ref), [`ucon_jac_par_structure!`](@ref), [`ucon_jac_par_coord`](@ref), [`ucon_jac_par_coord!`](@ref), [`ucon_jac_par_dense!`](@ref), [`ucon_jprod_par`](@ref), [`ucon_jprod_par!`](@ref), [`ucon_jtprod_par`](@ref), [`ucon_jtprod_par!`](@ref) |
