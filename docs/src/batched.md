# Batched models

An [`AbstractBatchedNLPModel`](@ref) is a batch of `nbatch` NLPs

```math
\begin{aligned}
  \min_{x} \; & f_k(x) \\
  \text{subject to} \quad & c^ℓ_k \leq c_k(x) \leq c^u_k \\
                          & x^ℓ_k \leq x \leq x^u_k
\end{aligned}
\qquad k = 1, …, \texttt{nbatch}
```

that share dimensions and the sparsity patterns of the Jacobian and of the Hessian of the
Lagrangian. It is an `AbstractNLPModel{T, M}` whose storage `M` is a matrix type: data have one
column per instance, and `get_nvar`, `get_ncon`, `get_nnzj` and `get_nnzh` are per instance.
Metadata is a [`BatchedNLPModelMeta`](@ref) in `nlp.meta`, counters a `Counters` in `nlp.counters`.

Because a batch model is an `AbstractNLPModel`, it is accepted by code typed on
`AbstractNLPModel`, but only batch-aware code can evaluate it: `NLPModels` solvers expect vector
storage.

## Implementing a batched model

Implement the in-place methods of `NLPModels` with matrix arguments, plus [`obj!`](@ref):

| Function | Arguments |
|---|---|
| [`obj!`](@ref) | `(nlp, X, f)`, `f` of length `nbatch` |
| `grad!` | `(nlp, X, G)`, `G` is `nvar × nbatch` |
| `cons!` | `(nlp, X, C)`, `C` is `ncon × nbatch` |
| `jac_structure!` | `(nlp, rows, cols)`, shared, length `nnzj` |
| `jac_coord!` | `(nlp, X, V)`, `V` is `nnzj × nbatch` |
| `hess_structure!` | `(nlp, rows, cols)`, shared, length `nnzh` |
| `hess_coord!` | `(nlp, X, Y, σ, V)`, `Y` is `ncon × nbatch`, `σ` the objective weights of length `nbatch`, `V` is `nnzh × nbatch` |

The allocating versions `obj`, `grad`, `cons`, `jac_coord` and `hess_coord` follow, and
`jac_structure` and `hess_structure` come from `NLPModels`. [`get_nbatch`](@ref) returns the
number of instances.

`obj!` and `get_nbatch` are the functions of the same name in `NLPModels` ≥ 0.21.12, extended
with methods for batched models.
