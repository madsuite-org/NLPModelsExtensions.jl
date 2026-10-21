# f(x) = θ₁ x₁² + x₂ and c(x) = θ₂ x₁ x₂ with x₁ ≥ -θ₁ and c(x) ≥ θ₂, one column of Θ per instance
struct TestModel{T} <: AbstractExtNLPModel{T, Matrix{T}}
    meta::BatchedNLPModelMeta{T, Matrix{T}}
    counters::Counters
    Θ::Matrix{T}
end
function TestModel(Θ::Matrix)
    nbatch = size(Θ, 2)
    lvar = [-Θ[1:1, :]; fill(-Inf, 1, nbatch)]
    meta = BatchedNLPModelMeta(2, nbatch; lvar, ncon = 1, lcon = Θ[2:2, :], nnzj = 2, nnzh = 2)
    return TestModel(meta, Counters(), copy(Θ))
end

function NLPModelsExtensions.obj!(nlp::TestModel, X, f)
    increment!(nlp, :neval_obj)
    f .= nlp.Θ[1, :] .* X[1, :] .^ 2 .+ X[2, :]
    return f
end
function NLPModels.grad!(nlp::TestModel, X::AbstractMatrix, G::AbstractMatrix)
    G[1, :] .= 2 .* nlp.Θ[1, :] .* X[1, :]
    G[2, :] .= 1
    return G
end
function NLPModels.cons!(nlp::TestModel, X::AbstractMatrix, C::AbstractMatrix)
    C[1, :] .= nlp.Θ[2, :] .* X[1, :] .* X[2, :]
    return C
end
function NLPModels.jac_structure!(nlp::TestModel, rows::AbstractVector{T}, cols::AbstractVector{T}) where {T}
    copyto!(rows, [1, 1])
    copyto!(cols, [1, 2])
    return rows, cols
end
function NLPModels.jac_coord!(nlp::TestModel, X::AbstractMatrix, V::AbstractMatrix)
    V[1, :] .= nlp.Θ[2, :] .* X[2, :]
    V[2, :] .= nlp.Θ[2, :] .* X[1, :]
    return V
end
function NLPModels.hess_structure!(nlp::TestModel, rows::AbstractVector{T}, cols::AbstractVector{T}) where {T}
    copyto!(rows, [1, 2])
    copyto!(cols, [1, 1])
    return rows, cols
end
function NLPModels.hess_coord!(nlp::TestModel, X::AbstractMatrix, Y::AbstractMatrix, σ::AbstractVector, V::AbstractMatrix)
    V[1, :] .= 2 .* σ .* nlp.Θ[1, :]
    V[2, :] .= Y[1, :] .* nlp.Θ[2, :]
    return V
end

function NLPModelsExtensions.get_par_meta(nlp::TestModel)
    return ParametricNLPModelMeta(; npar = 2, nnzo_par = 1, nnzj_par = 1, nnzh_par = 3, nnzj_lvar_par = 1, nnzj_lcon_par = 1)
end
NLPModelsExtensions.get_par(nlp::TestModel) = nlp.Θ
function NLPModelsExtensions.set_par!(nlp::TestModel, Θ)
    copyto!(nlp.Θ, Θ)
    nlp.meta.lvar[1, :] .= -Θ[1, :]
    nlp.meta.lcon[1, :] .= Θ[2, :]
    return nlp
end
function NLPModelsExtensions.grad_par!(nlp::TestModel, X, G)
    G[1, :] .= X[1, :] .^ 2
    G[2, :] .= 0
    return G
end
function NLPModelsExtensions.jac_par_structure!(nlp::TestModel, rows, cols)
    rows[1], cols[1] = 1, 2
    return rows, cols
end
function NLPModelsExtensions.jac_par_coord!(nlp::TestModel, X, V)
    V[1, :] .= X[1, :] .* X[2, :]
    return V
end
function NLPModelsExtensions.hess_par_structure!(nlp::TestModel, rows, cols)
    copyto!(rows, [1, 1, 2])
    copyto!(cols, [1, 2, 2])
    return rows, cols
end
function NLPModelsExtensions.hess_par_coord!(nlp::TestModel, X, Y, σ, V)
    V[1, :] .= 2 .* σ .* X[1, :]
    V[2, :] .= Y[1, :] .* X[2, :]
    V[3, :] .= Y[1, :] .* X[1, :]
    return V
end
function NLPModelsExtensions.lvar_jac_par_structure!(nlp::TestModel, rows, cols)
    rows[1], cols[1] = 1, 1
    return rows, cols
end
NLPModelsExtensions.lvar_jac_par_coord!(nlp::TestModel, V) = fill!(V, -1)
function NLPModelsExtensions.lcon_jac_par_structure!(nlp::TestModel, rows, cols)
    rows[1], cols[1] = 1, 2
    return rows, cols
end
NLPModelsExtensions.lcon_jac_par_coord!(nlp::TestModel, V) = fill!(V, 1)

# Dense references for one instance: ∂c/∂θ and ∂²L/∂x∂θ
Jθ(x) = [0 x[1] * x[2]]
Hxθ(x, y, σ) = [2σ * x[1] y[1] * x[2]; 0 y[1] * x[1]]

# Batched but not parametric
struct BatchedNLPModel <: AbstractBatchedNLPModel{Float64, Matrix{Float64}}
    meta::BatchedNLPModelMeta{Float64, Matrix{Float64}}
    counters::Counters
end

# Declares parameters without implementing them
struct Undeclared <: AbstractExtNLPModel{Float64, Matrix{Float64}}
    meta::BatchedNLPModelMeta{Float64, Matrix{Float64}}
    counters::Counters
end
NLPModelsExtensions.get_par_meta(::Undeclared) = ParametricNLPModelMeta(; npar = 1, nnzj_par = 1)

# f(x) = (x₁ - 1)² + x₂² and c(x) = x₁ x₂, an ordinary single-instance model
struct VectorModel <: AbstractNLPModel{Float64, Vector{Float64}}
    meta::NLPModelMeta{Float64, Vector{Float64}}
    counters::Counters
end
function VectorModel()
    meta = NLPModelMeta(2; x0 = [0.5, 2.0], ncon = 1, lcon = [1.0], ucon = [1.0], nnzj = 2, nnzh = 3, name = "vector")
    return VectorModel(meta, Counters())
end
function NLPModels.obj(nlp::VectorModel, x::AbstractVector)
    increment!(nlp, :neval_obj)
    return (x[1] - 1)^2 + x[2]^2
end
NLPModels.grad!(nlp::VectorModel, x::AbstractVector, g::AbstractVector) = copyto!(g, [2(x[1] - 1), 2x[2]])
NLPModels.cons_nln!(nlp::VectorModel, x::AbstractVector, c::AbstractVector) = copyto!(c, [x[1] * x[2]])
function NLPModels.jac_nln_structure!(nlp::VectorModel, rows::AbstractVector{T}, cols::AbstractVector{T}) where {T}
    copyto!(rows, [1, 1])
    copyto!(cols, [1, 2])
    return rows, cols
end
NLPModels.jac_nln_coord!(nlp::VectorModel, x::AbstractVector, vals::AbstractVector) = copyto!(vals, [x[2], x[1]])
function NLPModels.hess_structure!(nlp::VectorModel, rows::AbstractVector{T}, cols::AbstractVector{T}) where {T}
    copyto!(rows, [1, 2, 2])
    copyto!(cols, [1, 1, 2])
    return rows, cols
end
function NLPModels.hess_coord!(nlp::VectorModel, x::AbstractVector, y::AbstractVector, vals::AbstractVector; obj_weight = 1.0)
    return copyto!(vals, [2obj_weight, y[1], 2obj_weight])
end
