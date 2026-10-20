@testset "chain" begin
    @test AbstractExtNLPModel === AbstractParametricNLPModel
    @test AbstractParametricNLPModel <: AbstractBatchedNLPModel <: AbstractNLPModel
    @test TestModel([2.0; 3.0;;]) isa AbstractExtNLPModel

    b = BatchOnly(BatchedNLPModelMeta(2, 3), Counters())
    @test b isa AbstractNLPModel
    @test !(b isa AbstractExtNLPModel)
    @test get_nbatch(b) == 3
    @test_throws MethodError get_npar(b)
end

@testset "BatchedNLPModelMeta" begin
    meta = BatchedNLPModelMeta(2, 3; ncon = 1)
    @test get_nbatch(meta) == 3
    @test get_nvar(meta) == 2
    @test get_ncon(meta) == 1
    @test size(get_x0(meta)) == (2, 3)
    @test size(get_y0(meta)) == (1, 3)
    @test all(==(-Inf), get_lvar(meta)) && all(==(Inf), get_ucon(meta))
    @test get_nnzj(meta) == 2
    @test get_nnzh(meta) == 3
    @test get_minimize(meta)
    @test occursin("nbatch: 3", sprint(show, meta))

    @test_throws ErrorException BatchedNLPModelMeta(0, 3)
    @test_throws ErrorException BatchedNLPModelMeta(2, 0)
    @test_throws DimensionMismatch BatchedNLPModelMeta(2, 3; x0 = zeros(2, 2))
    @test_throws DimensionMismatch BatchedNLPModelMeta(2, 3; ncon = 1, lcon = zeros(2, 3))
end

@testset "AbstractBatchedNLPModel" begin
    Θ = [2.0 4.0 1.0; 3.0 5.0 -1.0]
    X = [1.5 0.5 -1.0; -2.0 1.0 2.0]
    Y = [0.5 -1.0 2.0]
    σ = [1.0, 0.3, 0.0]
    nlp = TestModel(Θ)

    @test get_nbatch(nlp) == 3
    @test get_nvar(nlp) == 2
    @test get_ncon(nlp) == 1
    @test get_lvar(nlp) == [-2.0 -4.0 -1.0; -Inf -Inf -Inf]
    @test get_lcon(nlp) == [3.0 5.0 -1.0]

    @test obj(nlp, X) ≈ [2.5, 2.0, 3.0]
    @test grad(nlp, X) ≈ [6.0 4.0 -2.0; 1.0 1.0 1.0]
    @test cons(nlp, X) ≈ [-9.0 2.5 2.0]
    @test jac_structure(nlp) == ([1, 1], [1, 2])
    @test jac_coord(nlp, X) ≈ [-6.0 5.0 -2.0; 4.5 2.5 1.0]
    @test hess_structure(nlp) == ([1, 2], [1, 1])
    @test hess_coord(nlp, X, Y, σ) ≈ [4.0 2.4 0.0; 1.5 -5.0 -2.0]

    reset!(nlp)
    obj(nlp, X)
    @test neval_obj(nlp) == 1
    @test occursin("nbatch: 3", sprint(show, nlp))

    @test_throws DimensionMismatch obj(nlp, X[:, 1:2])
    @test_throws DimensionMismatch grad(nlp, X')
    @test_throws DimensionMismatch hess_coord(nlp, X, [Y; Y], σ)
    @test_throws DimensionError hess_coord(nlp, X, Y, [1.0])
end
