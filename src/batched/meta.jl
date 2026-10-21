export BatchedNLPModelMeta

"""
    BatchedNLPModelMeta(nvar, nbatch; kwargs...)

Metadata of a batch of `nbatch` NLPs with `nvar` variables each. Data are matrices with one
column per instance.

- `x0`, `lvar`, `uvar`: initial point and variable bounds, `nvar × nbatch`
- `ncon`: number of constraints per instance
- `y0`, `lcon`, `ucon`: initial multipliers and constraint bounds, `ncon × nbatch`
- `nnzj`, `nnzh`: nonzeros per instance in the Jacobian and in the lower triangle of the Hessian of the Lagrangian
- `minimize`: whether the instances are minimized
- `name`: name of the batch

`get_nvar`, `get_x0` and the other `NLPModels` getters work on the meta and on the model.
"""
struct BatchedNLPModelMeta{T, M <: AbstractMatrix{T}}
    nvar::Int
    ncon::Int
    nbatch::Int
    x0::M
    lvar::M
    uvar::M
    y0::M
    lcon::M
    ucon::M
    nnzj::Int
    nnzh::Int
    minimize::Bool
    name::String
end

function BatchedNLPModelMeta(
        nvar::Int, nbatch::Int;
        x0::M = zeros(nvar, nbatch),
        lvar::M = fill!(similar(x0), eltype(x0)(-Inf)),
        uvar::M = fill!(similar(x0), eltype(x0)(Inf)),
        ncon::Int = 0,
        y0::M = fill!(similar(x0, ncon, nbatch), zero(eltype(x0))),
        lcon::M = fill!(similar(y0), eltype(x0)(-Inf)),
        ucon::M = fill!(similar(y0), eltype(x0)(Inf)),
        nnzj::Int = nvar * ncon,
        nnzh::Int = nvar * (nvar + 1) ÷ 2,
        minimize::Bool = true,
        name::String = "Batched NLP",
    ) where {M <: AbstractMatrix}
    if nvar < 1 || nbatch < 1 || ncon < 0 || nnzj < 0 || nnzh < 0
        error("Nonsensical dimensions")
    end
    @sizecheck nvar nbatch x0 lvar uvar
    @sizecheck ncon nbatch y0 lcon ucon
    return BatchedNLPModelMeta{eltype(M), M}(
        nvar, ncon, nbatch, x0, lvar, uvar, y0, lcon, ucon, nnzj, nnzh, minimize, name,
    )
end

for field in filter(!=(:nbatch), fieldnames(BatchedNLPModelMeta))
    meth = Symbol("get_", field)
    @eval NLPModels.$meth(meta::BatchedNLPModelMeta) = getproperty(meta, $(QuoteNode(field)))
end

"""
    get_nbatch(nlp)
    get_nbatch(meta)

Return the number of instances in the batch, the number of columns of the data matrices.
"""
get_nbatch(meta::BatchedNLPModelMeta) = meta.nbatch
get_nbatch(nlp::AbstractBatchedNLPModel) = get_nbatch(nlp.meta)

function Base.show(io::IO, meta::BatchedNLPModelMeta)
    println(io, "  Batched problem name: ", meta.name)
    println(io, "  nbatch: ", meta.nbatch, "  nvar: ", meta.nvar, "  ncon: ", meta.ncon,
        "  nnzj: ", meta.nnzj, "  nnzh: ", meta.nnzh)
    return println(io)
end
