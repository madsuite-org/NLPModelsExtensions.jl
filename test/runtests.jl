using Test
using NLPModelsExtensions

@testset "exports" begin
    for mod in (ParametricNLPModels,)
        for name in names(mod)
            Base.isexported(mod, name) && @test Base.isexported(NLPModelsExtensions, name)
        end
    end
end

include("ParametricNLPModels.jl")
