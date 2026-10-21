for block in (:lvar, :uvar, :lcon, :ucon)
    nrow = block in (:lvar, :uvar) ? :get_nvar : :get_ncon
    nnz = Symbol("get_nnzj_", block, "_par")
    nnzname = "nnzj_$(block)_par"
    structure!, structure = Symbol(block, "_jac_par_structure!"), Symbol(block, "_jac_par_structure")
    coord!, coord = Symbol(block, "_jac_par_coord!"), Symbol(block, "_jac_par_coord")
    jprod!, jprod = Symbol(block, "_jprod_par!"), Symbol(block, "_jprod_par")
    jtprod!, jtprod = Symbol(block, "_jtprod_par!"), Symbol(block, "_jtprod_par")
    symbol = Dict(:lvar => "x^ℓ", :uvar => "x^u", :lcon => "c^ℓ", :ucon => "c^u")[block]
    name, B = string(block), "∂$symbol/∂θ"
    @eval begin
        export $structure!, $structure, $coord!, $coord, $jprod!, $jprod, $jtprod!, $jtprod

        @doc """
            $($structure!)(nlp, rows, cols)

        Return the structure of ``$($B)``, the `$($name)` bounds with respect to the parameters, in sparse coordinate format in place.
        Defaults to an empty structure when `$($nnzname) = 0`.
        """
        function $structure!(nlp::AbstractParametricNLPModel, rows, cols)
            $nnz(nlp) == 0 || _not_implemented($structure!, nlp)
            return rows, cols
        end

        @doc """
            (rows,cols) = $($structure)(nlp)

        Return the structure of ``$($B)``, the `$($name)` bounds with respect to the parameters, in sparse coordinate format.
        """
        function $structure(nlp::AbstractParametricNLPModel)
            n = $nnz(nlp)
            rows = Vector{Int}(undef, n)
            cols = Vector{Int}(undef, n)
            return $structure!(nlp, rows, cols)
        end

        @doc """
            V = $($coord!)(nlp, V)

        Evaluate ``$($B)``, the `$($name)` bounds with respect to the parameters, in sparse coordinate format, overwriting `V` (`$($nnzname) × nbatch`).
        Defaults to doing nothing when `$($nnzname) = 0`.
        """
        function $coord!(nlp::AbstractParametricNLPModel, V)
            $nnz(nlp) == 0 || _not_implemented($coord!, nlp)
            return V
        end

        @doc """
            V = $($coord)(nlp)

        Evaluate ``$($B)``, the `$($name)` bounds with respect to the parameters, in sparse coordinate format, `$($nnzname) × nbatch`.
        """
        function $coord(nlp::AbstractParametricNLPModel{T, M}) where {T, M}
            V = M(undef, $nnz(nlp), get_nbatch(nlp))
            return $coord!(nlp, V)
        end

        @doc """
            JV = $($jprod)(nlp, V)

        Evaluate ``($($B))v``, the Jacobian-vector product of the `$($name)` bounds with respect to the parameters, at the columns of `V`.
        """
        function $jprod(nlp::AbstractParametricNLPModel{T, M}, V::AbstractMatrix) where {T, M}
            JV = M(undef, $nrow(nlp), get_nbatch(nlp))
            return $jprod!(nlp, V, JV)
        end

        @doc """
            JV = $($jprod!)(nlp, V, JV)

        Evaluate ``($($B))v``, the Jacobian-vector product of the `$($name)` bounds with respect to the parameters, at the columns of `V` in place.
        This function allocates.
        """
        function $jprod!(nlp::AbstractParametricNLPModel, V::AbstractMatrix, JV::AbstractMatrix)
            @sizecheck get_npar(nlp) get_nbatch(nlp) V
            @sizecheck $nrow(nlp) get_nbatch(nlp) JV
            rows, cols = $structure(nlp)
            vals = $coord(nlp)
            return _coo_prod!(rows, cols, vals, V, JV)
        end

        @doc """
            JtV = $($jtprod)(nlp, V)

        Evaluate ``($($B))^Tv``, the transposed-Jacobian-vector product of the `$($name)` bounds with respect to the parameters, at the columns of `V`.
        """
        function $jtprod(nlp::AbstractParametricNLPModel{T, M}, V::AbstractMatrix) where {T, M}
            JtV = M(undef, get_npar(nlp), get_nbatch(nlp))
            return $jtprod!(nlp, V, JtV)
        end

        @doc """
            JtV = $($jtprod!)(nlp, V, JtV)

        Evaluate ``($($B))^Tv``, the transposed-Jacobian-vector product of the `$($name)` bounds with respect to the parameters, at the columns of `V` in place.
        This function allocates.
        """
        function $jtprod!(nlp::AbstractParametricNLPModel, V::AbstractMatrix, JtV::AbstractMatrix)
            @sizecheck $nrow(nlp) get_nbatch(nlp) V
            @sizecheck get_npar(nlp) get_nbatch(nlp) JtV
            rows, cols = $structure(nlp)
            vals = $coord(nlp)
            return _coo_prod!(cols, rows, vals, V, JtV)
        end
    end
end
