using Test
using NLPModels
using NLPModelsExtensions.ParametricNLPModels
using SparseArrays

# f(x) = θ₁ x₁² + x₂ and c(x) = θ₂ x₁ x₂
struct TestModel{T} <: AbstractNLPModel{T, Vector{T}}
    meta::NLPModelMeta{T, Vector{T}}
    counters::Counters
    θ::Vector{T}
end
TestModel(θ::Vector) = TestModel(NLPModelMeta(2; ncon = 1, nnzj = 2, nnzh = 1, lvar = [-θ[1], -Inf], lcon = [θ[2]]), Counters(), θ)

function ParametricNLPModels.get_par_meta(nlp::TestModel)
    return ParametricNLPModelMeta(; npar = 2, nnzo_par = 1, nnzj_par = 1, nnzh_par = 3, nnzj_lvar_par = 1, nnzj_lcon_par = 1)
end
ParametricNLPModels.get_par(nlp::TestModel) = nlp.θ
ParametricNLPModels.set_par!(nlp::TestModel, θ) = (copyto!(nlp.θ, θ); nlp)
function ParametricNLPModels.jac_par_structure!(nlp::TestModel, rows, cols)
    rows[1], cols[1] = 1, 2
    return rows, cols
end
function ParametricNLPModels.jac_par_coord!(nlp::TestModel, x, vals)
    vals[1] = x[1] * x[2]
    return vals
end
function ParametricNLPModels.hess_par_structure!(nlp::TestModel, rows, cols)
    copyto!(rows, [1, 1, 2])
    copyto!(cols, [1, 2, 2])
    return rows, cols
end
function ParametricNLPModels.hess_par_coord!(nlp::TestModel, x, y, vals; obj_weight = one(eltype(x)))
    copyto!(vals, [2obj_weight * x[1], y[1] * x[2], y[1] * x[1]])
    return vals
end
function ParametricNLPModels.grad_par!(nlp::TestModel, x, g)
    copyto!(g, [x[1]^2, 0.0])
    return g
end
function ParametricNLPModels.lvar_jac_par_structure!(nlp::TestModel, rows, cols)
    rows[1], cols[1] = 1, 1
    return rows, cols
end
function ParametricNLPModels.lvar_jac_par_coord!(nlp::TestModel, vals)
    vals[1] = -1.0
    return vals
end
function ParametricNLPModels.lcon_jac_par_structure!(nlp::TestModel, rows, cols)
    rows[1], cols[1] = 1, 2
    return rows, cols
end
function ParametricNLPModels.lcon_jac_par_coord!(nlp::TestModel, vals)
    vals[1] = 1.0
    return vals
end

struct BlankModel <: AbstractNLPModel{Float64, Vector{Float64}}
    meta::NLPModelMeta{Float64, Vector{Float64}}
    counters::Counters
end

@testset "ParametricNLPModels" begin
    nlp = TestModel([2.0, 3.0])
    x, y, w, v = [1.5, -2.0], [0.5], 0.3, [1.0, 2.0]
    J = [0.0 -3.0]
    H = [3.0 -1.0; 0.0 0.75]
    Hw = [2w * x[1] -1.0; 0.0 0.75]

    @test jac_par_structure(nlp) == ([1], [2])
    @test jac_par_coord(nlp, x) == [-3.0]
    @test jac_par(nlp, x) isa SparseMatrixCSC
    @test jac_par(nlp, x) == J
    @test jac_par_dense!(nlp, x, ones(1, 2)) == J
    @test jprod_par(nlp, x, v) == [-6.0]
    @test jprod_par!(nlp, [1], [2], [-3.0], v, zeros(1)) == [-6.0]
    @test jtprod_par(nlp, x, [2.0]) == [0.0, -6.0]
    @test jtprod_par!(nlp, [1], [2], [-3.0], [2.0], zeros(2)) == [0.0, -6.0]

    @test hess_par_structure(nlp) == ([1, 1, 2], [1, 2, 2])
    @test hess_par_coord(nlp, x, y) == [3.0, -1.0, 0.75]
    @test hess_par(nlp, x, y) isa SparseMatrixCSC
    @test hess_par(nlp, x, y) == H
    @test hess_par(nlp, x, y; obj_weight = w) == Hw
    @test hess_par_dense!(nlp, x, y, ones(2, 2)) == H
    @test hess_par_dense!(nlp, x, y, ones(2, 2); obj_weight = w) == Hw
    @test hprod_par(nlp, x, y, v) == [1.0, 1.5]
    @test hprod_par(nlp, x, y, v; obj_weight = w) == [-1.1, 1.5]
    @test hprod_par!(nlp, [1, 1, 2], [1, 2, 2], [3.0, -1.0, 0.75], v, zeros(2)) == [1.0, 1.5]
    @test htprod_par(nlp, x, y, v) == [3.0, 0.5]
    @test htprod_par!(nlp, [1, 1, 2], [1, 2, 2], [3.0, -1.0, 0.75], v, zeros(2)) == [3.0, 0.5]

    H0, H0w = [3.0 0.0; 0.0 0.0], [2w * x[1] 0.0; 0.0 0.0]
    @test hess_par_coord(nlp, x) == [3.0, 0.0, 0.0]
    @test hess_par_coord!(nlp, x, zeros(3); obj_weight = w) == [2w * x[1], 0.0, 0.0]
    @test hess_par(nlp, x) == H0
    @test hess_par(nlp, x; obj_weight = w) == H0w
    @test hess_par_dense!(nlp, x, ones(2, 2)) == H0
    @test hprod_par(nlp, x, v) == [3.0, 0.0]
    @test hprod_par!(nlp, x, v, zeros(2); obj_weight = w) == [2w * x[1], 0.0]
    @test htprod_par(nlp, x, v) == [3.0, 0.0]
    @test htprod_par!(nlp, x, v, zeros(2)) == [3.0, 0.0]

    @test grad_par(nlp, x) == [2.25, 0.0]
    @test grad_par!(nlp, x, zeros(2)) == [2.25, 0.0]

    @test lvar_jac_par_structure(nlp) == ([1], [1])
    @test lvar_jac_par_coord(nlp) == [-1.0]
    @test lvar_jac_par(nlp) isa SparseMatrixCSC
    @test lvar_jac_par(nlp) == [-1.0 0.0; 0.0 0.0]
    @test lvar_jac_par_dense!(nlp, ones(2, 2)) == [-1.0 0.0; 0.0 0.0]
    @test lvar_jprod_par(nlp, v) == [-1.0, 0.0]
    @test lvar_jprod_par!(nlp, v, zeros(2)) == [-1.0, 0.0]
    @test lvar_jtprod_par(nlp, [2.0, 3.0]) == [-2.0, 0.0]
    @test lvar_jtprod_par!(nlp, [2.0, 3.0], zeros(2)) == [-2.0, 0.0]
    @test lcon_jac_par(nlp) == [0.0 1.0]
    @test lcon_jprod_par(nlp, v) == [2.0]
    @test lcon_jtprod_par(nlp, [2.0]) == [0.0, 2.0]

    @test get_par_meta(nlp) isa ParametricNLPModelMeta
    @test get_npar(nlp) == 2
    @test get_nnzo_par(nlp) == 1
    @test get_nnzj_lvar_par(nlp) == 1
    @test get_nnzj_uvar_par(nlp) == 0
    @test get_jac_par_available(nlp)

    @test get_par(nlp) == [2.0, 3.0]
    set_par!(nlp, [4.0, 5.0])
    @test get_par(nlp) == [4.0, 5.0]

    @test ParametricNLPModels._dense!(zeros(1, 2), [1, 1], [2, 2], [1.0, 2.0]) == [0.0 3.0]

    @test_throws DimensionError jac_par_coord(nlp, [1.0])
    @test_throws DimensionError hess_par_coord(nlp, x, [1.0, 2.0])
    @test_throws DimensionError jprod_par(nlp, x, [1.0])
    @test_throws DimensionError htprod_par(nlp, x, y, [1.0])
    @test_throws DimensionError lvar_jprod_par(nlp, [1.0])
    @test_throws ErrorException ParametricNLPModelMeta(; npar = -1)

    blank = BlankModel(NLPModelMeta(2), Counters())
    @test get_par_meta(blank) == ParametricNLPModelMeta()
    @test get_npar(blank) == 0
    @test get_nnzj_lvar_par(blank) == 0
    @test_throws MethodError jac_par(blank, x)
end
