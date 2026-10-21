"""
    @sizecheck m n X Y …

Check that matrices `X`, `Y`, etc. have size `(m, n)`.
"""
macro sizecheck(m, n, vars...)
    exprs = Expr[]
    for var in vars
        varname = string(var)
        push!(exprs, :(
            if size($(esc(var)), 1) != $(esc(m))
                throw(DimensionError(string($varname, " rows"), $(esc(m)), size($(esc(var)), 1)))
            end
        ))
        push!(exprs, :(
            if size($(esc(var)), 2) != $(esc(n))
                throw(DimensionError(string($varname, " columns"), $(esc(n)), size($(esc(var)), 2)))
            end
        ))
    end
    return Expr(:block, exprs...)
end

# Batched sparse coordinate product: column `k` of `AV` is `A(vals[:, k]) * V[:, k]`.
function _coo_prod!(rows, cols, vals::AbstractMatrix, V::AbstractMatrix, AV::AbstractMatrix)
    for k in axes(AV, 2)
        coo_prod!(rows, cols, view(vals, :, k), view(V, :, k), view(AV, :, k))
    end
    return AV
end

_not_implemented(f, nlp) = error("$(typeof(nlp)) does not implement $f")
