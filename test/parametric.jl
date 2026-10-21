@testset "AbstractParametricNLPModel" begin
    nlp = TestModel([2.0 4.0; 3.0 5.0])
    X = [1.5 0.5; -2.0 1.0]
    Y = [0.5 -1.0]
    σ = [1.0, 0.3]
    V = [1.0 -1.0; 2.0 0.5]
    W = [2.0 -1.0]
    cols(A) = eachindex(axes(A, 2))

    @test jac_par_structure(nlp) == ([1], [2])
    @test jac_par_coord(nlp, X) == [-3.0 0.5]
    @test jac_par_coord!(nlp, X, zeros(1, 2)) == [-3.0 0.5]
    JV = jprod_par(nlp, X, V)
    @test all(JV[:, k] ≈ Jθ(X[:, k]) * V[:, k] for k in cols(X))
    @test jprod_par!(nlp, X, V, zeros(1, 2)) ≈ JV
    @test jprod_par!(nlp, [1], [2], [-3.0 0.5], V, zeros(1, 2)) ≈ JV
    JtW = jtprod_par(nlp, X, W)
    @test all(JtW[:, k] ≈ Jθ(X[:, k])' * W[:, k] for k in cols(X))
    @test jtprod_par!(nlp, X, W, zeros(2, 2)) ≈ JtW
    @test jtprod_par!(nlp, [1], [2], [-3.0 0.5], W, zeros(2, 2)) ≈ JtW

    @test hess_par_structure(nlp) == ([1, 1, 2], [1, 2, 2])
    @test hess_par_coord(nlp, X, Y, σ) ≈ [3.0 0.3; -1.0 -1.0; 0.75 -0.5]
    @test hess_par_coord!(nlp, X, Y, σ, zeros(3, 2)) ≈ [3.0 0.3; -1.0 -1.0; 0.75 -0.5]
    HV = hprod_par(nlp, X, Y, V, σ)
    @test all(HV[:, k] ≈ Hxθ(X[:, k], Y[:, k], σ[k]) * V[:, k] for k in cols(X))
    @test hprod_par!(nlp, X, Y, V, σ, zeros(2, 2)) ≈ HV
    @test hprod_par!(nlp, [1, 1, 2], [1, 2, 2], hess_par_coord(nlp, X, Y, σ), V, zeros(2, 2)) ≈ HV
    HtV = htprod_par(nlp, X, Y, V, σ)
    @test all(HtV[:, k] ≈ Hxθ(X[:, k], Y[:, k], σ[k])' * V[:, k] for k in cols(X))
    @test htprod_par!(nlp, X, Y, V, σ, zeros(2, 2)) ≈ HtV
    @test htprod_par!(nlp, [1, 1, 2], [1, 2, 2], hess_par_coord(nlp, X, Y, σ), V, zeros(2, 2)) ≈ HtV

    @test grad_par(nlp, X) == [2.25 0.25; 0.0 0.0]
    @test grad_par!(nlp, X, zeros(2, 2)) == [2.25 0.25; 0.0 0.0]

    @test lvar_jac_par_structure(nlp) == ([1], [1])
    @test lvar_jac_par_coord(nlp) == [-1.0 -1.0]
    @test lvar_jprod_par(nlp, V) == [-1.0 1.0; 0.0 0.0]
    @test lvar_jprod_par!(nlp, V, zeros(2, 2)) == [-1.0 1.0; 0.0 0.0]
    @test lvar_jtprod_par(nlp, [2.0 3.0; 4.0 5.0]) == [-2.0 -3.0; 0.0 0.0]
    @test lvar_jtprod_par!(nlp, [2.0 3.0; 4.0 5.0], zeros(2, 2)) == [-2.0 -3.0; 0.0 0.0]
    @test lcon_jac_par_coord(nlp) == [1.0 1.0]
    @test lcon_jprod_par(nlp, V) == [2.0 0.5]
    @test lcon_jtprod_par(nlp, W) == [0.0 0.0; 2.0 -1.0]

    # uvar and ucon have no nonzeros, so the defaults apply
    @test uvar_jac_par_structure(nlp) == (Int[], Int[])
    @test size(uvar_jac_par_coord(nlp)) == (0, 2)
    @test uvar_jprod_par(nlp, V) == zeros(2, 2)
    @test ucon_jtprod_par(nlp, W) == zeros(2, 2)

    @test get_par_meta(nlp) isa ParametricNLPModelMeta
    @test get_npar(nlp) == 2
    @test get_nnzo_par(nlp) == 1
    @test get_nnzj_lvar_par(nlp) == 1
    @test get_nnzj_uvar_par(nlp) == 0
    @test get_jac_par_available(nlp)

    @test get_par(nlp) == [2.0 4.0; 3.0 5.0]
    set_par!(nlp, [4.0 6.0; 5.0 7.0])
    @test get_par(nlp) == [4.0 6.0; 5.0 7.0]
    @test get_lvar(nlp)[1, :] == [-4.0, -6.0]
    @test get_lcon(nlp) == [5.0 7.0]

    @test_throws DimensionError jac_par_coord(nlp, X[:, 1:1])
    @test_throws DimensionError hess_par_coord(nlp, X, Y, [1.0])
    @test_throws DimensionError jprod_par(nlp, X, ones(1, 2))
    @test_throws DimensionError htprod_par(nlp, X, Y, ones(1, 2), σ)
    @test_throws DimensionError jprod_par!(nlp, [1, 1], [2, 2], [-3.0 0.5], V, zeros(1, 2))
    @test_throws DimensionError lvar_jprod_par(nlp, ones(1, 2))
    @test_throws ErrorException ParametricNLPModelMeta(; npar = -1)
end

@testset "no parameters" begin
    nlp = ExtNLPModel(VectorModel())
    X = reshape([1.5, -2.0], 2, 1)
    Y = reshape([0.5], 1, 1)

    @test get_par_meta(nlp) == ParametricNLPModelMeta()
    @test get_npar(nlp) == 0
    @test size(get_par(nlp)) == (0, 1)
    @test set_par!(nlp, zeros(0, 1)) === nlp
    @test size(grad_par(nlp, X)) == (0, 1)
    @test jac_par_structure(nlp) == (Int[], Int[])
    @test size(jac_par_coord(nlp, X)) == (0, 1)
    @test jprod_par(nlp, X, zeros(0, 1)) == zeros(1, 1)
    @test hess_par_structure(nlp) == (Int[], Int[])
    @test size(hess_par_coord(nlp, X, Y, [1.0])) == (0, 1)
    @test hprod_par(nlp, X, Y, zeros(0, 1), [1.0]) == zeros(2, 1)
    @test lcon_jac_par_structure(nlp) == (Int[], Int[])
    @test size(lcon_jac_par_coord(nlp)) == (0, 1)

    # a model that declares parameters must implement them
    u = Undeclared(BatchedNLPModelMeta(2, 1), Counters())
    @test_throws ErrorException get_par(u)
    @test_throws ErrorException jac_par_structure(u)
    @test lvar_jac_par_structure(u) == (Int[], Int[])
end
