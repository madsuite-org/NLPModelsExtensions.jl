module NLPModelsExtensions

include("ParametricNLPModels/ParametricNLPModels.jl")

using .ParametricNLPModels

for mod in (ParametricNLPModels,)
    for name in names(mod)
        Base.isexported(mod, name) && @eval export $name
    end
end

end
