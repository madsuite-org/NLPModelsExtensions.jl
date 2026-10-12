for block in (:lvar, :uvar, :lcon, :ucon)
    nrow = block in (:lvar, :uvar) ? :get_nvar : :get_ncon
    nnz = Symbol("get_nnzj_", block, "_par")
    structure!, structure = Symbol(block, "_jac_par_structure!"), Symbol(block, "_jac_par_structure")
    coord!, coord = Symbol(block, "_jac_par_coord!"), Symbol(block, "_jac_par_coord")
    jac, dense! = Symbol(block, "_jac_par"), Symbol(block, "_jac_par_dense!")
    jprod!, jprod = Symbol(block, "_jprod_par!"), Symbol(block, "_jprod_par")
    jtprod!, jtprod = Symbol(block, "_jtprod_par!"), Symbol(block, "_jtprod_par")
    symbol = Dict(:lvar => "x^ℓ", :uvar => "x^u", :lcon => "c^ℓ", :ucon => "c^u")[block]
    name, B = string(block), "∂$symbol/∂θ"
    @eval begin
        export $structure!, $structure, $coord!, $coord, $jac, $dense!, $jprod!, $jprod, $jtprod!, $jtprod

        @doc """
            $($structure!)(nlp, rows, cols)

        Return the structure of ``$($B)``, the `$($name)` bounds with respect to the parameters, in sparse coordinate format in place.
        """
        function $structure! end

        @doc """
            (rows,cols) = $($structure)(nlp)

        Return the structure of ``$($B)``, the `$($name)` bounds with respect to the parameters, in sparse coordinate format.
        """
        function $structure(nlp::AbstractNLPModel)
            n = $nnz(nlp)
            rows = similar(get_x0(nlp), Int, n)
            cols = similar(get_x0(nlp), Int, n)
            return $structure!(nlp, rows, cols)
        end

        @doc """
            vals = $($coord!)(nlp, vals)

        Evaluate ``$($B)``, the `$($name)` bounds with respect to the parameters, in sparse coordinate format, overwriting `vals`.
        """
        function $coord! end

        @doc """
            vals = $($coord)(nlp)

        Evaluate ``$($B)``, the `$($name)` bounds with respect to the parameters, in sparse coordinate format.
        """
        function $coord(nlp::AbstractNLPModel{T, S}) where {T, S}
            vals = S(undef, $nnz(nlp))
            return $coord!(nlp, vals)
        end

        @doc """
            J = $($jac)(nlp)

        Evaluate ``$($B)``, the `$($name)` bounds with respect to the parameters, as a sparse matrix.
        """
        function $jac(nlp::AbstractNLPModel)
            rows, cols = $structure(nlp)
            vals = $coord(nlp)
            return sparse(rows, cols, vals, $nrow(nlp), get_npar(nlp))
        end

        @doc """
            J = $($dense!)(nlp, J)

        Evaluate ``$($B)``, the `$($name)` bounds with respect to the parameters, in dense format, overwriting `J`.
        """
        function $dense!(nlp::AbstractNLPModel, J::AbstractMatrix)
            rows, cols = $structure(nlp)
            vals = $coord(nlp)
            return _dense!(J, rows, cols, vals)
        end

        @doc """
            Jv = $($jprod)(nlp, v)

        Evaluate ``($($B))v``, the Jacobian-vector product of the `$($name)` bounds with respect to the parameters.
        """
        function $jprod(nlp::AbstractNLPModel{T, S}, v::AbstractVector) where {T, S}
            @lencheck get_npar(nlp) v
            Jv = S(undef, $nrow(nlp))
            return $jprod!(nlp, v, Jv)
        end

        @doc """
            Jv = $($jprod!)(nlp, v, Jv)

        Evaluate ``($($B))v``, the Jacobian-vector product of the `$($name)` bounds with respect to the parameters, in place.
        This function allocates.
        """
        function $jprod!(nlp::AbstractNLPModel, v::AbstractVector, Jv::AbstractVector)
            @lencheck get_npar(nlp) v
            @lencheck $nrow(nlp) Jv
            rows, cols = $structure(nlp)
            vals = $coord(nlp)
            return coo_prod!(rows, cols, vals, v, Jv)
        end

        @doc """
            Jtv = $($jtprod)(nlp, v)

        Evaluate ``($($B))^Tv``, the transposed-Jacobian-vector product of the `$($name)` bounds with respect to the parameters.
        """
        function $jtprod(nlp::AbstractNLPModel{T, S}, v::AbstractVector) where {T, S}
            @lencheck $nrow(nlp) v
            Jtv = S(undef, get_npar(nlp))
            return $jtprod!(nlp, v, Jtv)
        end

        @doc """
            Jtv = $($jtprod!)(nlp, v, Jtv)

        Evaluate ``($($B))^Tv``, the transposed-Jacobian-vector product of the `$($name)` bounds with respect to the parameters, in place.
        This function allocates.
        """
        function $jtprod!(nlp::AbstractNLPModel, v::AbstractVector, Jtv::AbstractVector)
            @lencheck $nrow(nlp) v
            @lencheck get_npar(nlp) Jtv
            rows, cols = $structure(nlp)
            vals = $coord(nlp)
            return coo_prod!(cols, rows, vals, v, Jtv)
        end
    end
end
