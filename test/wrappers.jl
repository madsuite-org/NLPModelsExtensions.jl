@testset "ExtNLPModel" begin
    single = VectorModel()
    nlp = ExtNLPModel(single)
    x, y = [1.5, -2.0], [0.5]
    X, Y = reshape(x, 2, 1), reshape(y, 1, 1)

    @test nlp isa AbstractExtNLPModel
    @test get_nbatch(nlp) == 1
    @test get_nvar(nlp) == 2
    @test get_ncon(nlp) == 1
    @test get_name(nlp) == "vector"
    @test get_x0(nlp) == reshape(get_x0(single), 2, 1)
    @test get_lcon(nlp) == reshape(get_lcon(single), 1, 1)
    @test get_nnzj(nlp) == get_nnzj(single)
    @test get_nnzh(nlp) == get_nnzh(single)

    @test obj(nlp, X) == [obj(single, x)]
    @test grad(nlp, X) == reshape(grad(single, x), 2, 1)
    @test cons(nlp, X) == reshape(cons(single, x), 1, 1)
    @test jac_structure(nlp) == jac_structure(single)
    @test jac_coord(nlp, X) == reshape(jac_coord(single, x), :, 1)
    @test hess_structure(nlp) == hess_structure(single)
    @test hess_coord(nlp, X, Y, [0.3]) == reshape(hess_coord(single, x, y; obj_weight = 0.3), :, 1)

    reset!(single)
    obj(nlp, X)
    @test neval_obj(nlp) == neval_obj(single) == 1
    @test occursin("nbatch: 1", sprint(show, nlp))

    # the wrapper owns its data
    get_x0(nlp) .= 0
    @test get_x0(single) == [0.5, 2.0]
end
