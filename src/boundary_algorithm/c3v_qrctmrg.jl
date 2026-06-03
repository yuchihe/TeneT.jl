# C3v QRCTMRG boundary algorithm
#
# QR-based CTMRG for honeycomb tensors with three virtual legs.
# The local tensor convention is TeneT-style `(D, D, D, d)`: three virtual
# legs followed by the physical leg. A length-1 unit cell uses the same tensor
# on both honeycomb sublattices; a length-2 unit cell uses distinct A/B tensors.

function _c3v_check_site(A, label)
    ndims(A) == 4 || throw(ArgumentError("C3vQRCTMRG expects a rank-4 tensor `(D,D,D,d)`, got rank $(ndims(A))."))
    D = size(A, 1)
    size(A, 2) == D && size(A, 3) == D ||
        throw(ArgumentError("C3vQRCTMRG expects equal virtual dimensions on the three honeycomb legs for $label."))
    return A
end

function _c3v_site_pair(M::StructArray)
    nsites = length(M.data)
    nsites in (1, 2) ||
        throw(ArgumentError("C3vQRCTMRG supports one- or two-site C3v honeycomb unit cells, got $nsites sites."))
    A = _c3v_check_site(M.data[1], "site 1")
    B = nsites == 1 ? A : _c3v_check_site(M.data[2], "site 2")
    size(A)[1:4] == size(B)[1:4] ||
        throw(DimensionMismatch("C3vQRCTMRG two-site tensors must have matching `(D,D,D,d)` dimensions, got $(size(A)) and $(size(B))."))
    return A, B
end

_c3v_downcast_sites(T::Type, Ms::Tuple) = (_downcast_eltype(T, Ms[1]), _downcast_eltype(T, Ms[2]))
_c3v_downcast_sites(::Nothing, Ms::Tuple) = Ms

_c3v_as_single_env(env::C3vTwoSiteCTMEnv, site::Int) =
    site == 1 ? C3vCTMEnv(env.CA, env.RA) :
    site == 2 ? C3vCTMEnv(env.CB, env.RB) :
    throw(ArgumentError("C3v two-site environment site index must be 1 or 2, got $site."))
_c3v_as_single_env(env::C3vCTMEnv, site::Int) = env

_c3v_two_site_env(envA::C3vCTMEnv, envB::C3vCTMEnv) =
    C3vTwoSiteCTMEnv(envA.C, envA.R, envB.C, envB.R)

_c3v_boundary_eltype(env::C3vCTMEnv) = eltype(env.R)
_c3v_boundary_eltype(env::C3vTwoSiteCTMEnv) = eltype(env.RA)

_c3v_downcast_env(T::Type, env::C3vCTMEnv) =
    C3vCTMEnv(_downcast_eltype(T, env.C), _downcast_eltype(T, env.R))
_c3v_downcast_env(T::Type, env::C3vTwoSiteCTMEnv) =
    C3vTwoSiteCTMEnv(_downcast_eltype(T, env.CA), _downcast_eltype(T, env.RA),
                     _downcast_eltype(T, env.CB), _downcast_eltype(T, env.RB))

function _c3v_corner_spectrum_error(Cnew, C)
    return ignore_derivatives() do
        snew = svdvals(Array(Cnew))
        sold = svdvals(Array(C))
        snew ./= norm(snew)
        sold ./= norm(sold)
        norm(snew - sold)
    end
end

_c3v_hermitianize(C) = (C + adjoint(C)) ./ 2
_c3v_reflect_edge(R) = (R + permutedims(conj.(R), (4, 3, 2, 1))) ./ 2

function _c3v_double_layer(M::AbstractArray)
    D = size(M, 1)
    size(M, 2) == D && size(M, 3) == D ||
        throw(ArgumentError("C3vQRCTMRG expects equal virtual dimensions on the three honeycomb legs."))
    @tensor M2[a,d,b,e,c,f] := M[a,b,c,p] * conj(M[d,e,f,p])
    M2layer = reshape(M2, D^2, D^2, D^2)
    @tensor C0[a,d,b,e] := M[a,b,c,p] * conj(M[d,e,c,p])
    C0 = _c3v_hermitianize(reshape(C0, D^2, D^2))
    @tensor R0[i,k,a] := C0[i,j] * M2layer[a,j,k]
    return C0, reshape(R0, D^2, D, D, D^2)
end

function _c3v_embed_boundary(C0, R0, chi::Int, M)
    D = size(R0, 2)
    chi0 = size(C0, 1)
    n = min(chi, chi0)

    C = fill!(similar(M, chi, chi), zero(eltype(M)))
    R = fill!(similar(M, chi, D, D, chi), zero(eltype(M)))
    C[1:n, 1:n] .= C0[1:n, 1:n]
    R[1:n, :, :, 1:n] .= R0[1:n, :, :, 1:n]

    C = _c3v_hermitianize(C)
    R = _c3v_reflect_edge(R)
    C /= ignore_derivatives(() -> norm(C))
    R /= ignore_derivatives(() -> norm(R))
    return C3vCTMEnv(C, R)
end

function init_env(M::StructArray, chi::Int, alg::C3vQRCTMRG)
    A, B = _c3v_site_pair(M)
    CA0, RA0 = _c3v_double_layer(A)
    envA = _c3v_embed_boundary(CA0, RA0, chi, A)
    if length(M.data) == 1
        return envA
    end

    CB0, RB0 = _c3v_double_layer(B)
    envB = _c3v_embed_boundary(CB0, RB0, chi, B)
    return _c3v_two_site_env(envA, envB)
end

function _c3v_qr(C, R)
    chi, D = size(R, 1), size(R, 2)
    @tensor CR[i,j,k,m] := C[i,q] * R[q,j,k,m]
    Q, Rmat = qr_for_ad(reshape(CR, chi * D * D, chi))
    return reshape(Q, chi, D, D, chi), Rmat
end

_c3v_update_R(V, R, M; inner_etype=nothing) =
    _c3v_update_R(V, R, M, M; inner_etype)

function _c3v_update_R(V, R, ML, MR; inner_etype=nothing)
    if inner_etype === nothing || inner_etype == real(eltype(R))
        @tensor Rnew[t,u,v,c] := conj(V[i,a,b,t]) * R[i,j,k,l] *
            ML[j,a,p,x] * conj(ML[k,b,q,x]) *
            MR[u,m,p,y] * conj(MR[v,n,q,y]) *
            V[l,m,n,c]
        return Rnew
    else
        T_out = eltype(R)
        V_t = _downcast_eltype(inner_etype, V)
        R_t = _downcast_eltype(inner_etype, R)
        ML_t = _downcast_eltype(inner_etype, ML)
        MR_t = _downcast_eltype(inner_etype, MR)
        @tensor Rnew_t[t,u,v,c] := conj(V_t[i,a,b,t]) * R_t[i,j,k,l] *
            ML_t[j,a,p,x] * conj(ML_t[k,b,q,x]) *
            MR_t[u,m,p,y] * conj(MR_t[v,n,q,y]) *
            V_t[l,m,n,c]
        return T_out.(Rnew_t)
    end
end

function _c3v_update_C(Rnew, Rmat, V)
    @tensor Cnew[c,r] := conj(Rnew[t,j,k,c]) * Rmat[t,b] * V[b,j,k,r]
    return Cnew
end

"""
    c3v_qrctmrg_step(env::C3vCTMEnv, M, alg::C3vQRCTMRG)

One C3v QRCTMRG step for three-leg honeycomb PEPS tensor(s). `M` may be a
single tensor or a tuple `(MA, MB)` for inequivalent A/B sublattices.
"""
function c3v_qrctmrg_step(env::C3vCTMEnv, M::AbstractArray, alg::C3vQRCTMRG)
    return c3v_qrctmrg_step(env, (M, M), alg)
end

function c3v_qrctmrg_step(env::C3vCTMEnv, M::Tuple, alg::C3vQRCTMRG)
    C = _c3v_hermitianize(env.C)
    R = _c3v_reflect_edge(env.R)
    ML, MR = M
    V, Rmat = _c3v_qr(C, R)
    Rnew = _c3v_update_R(V, R, ML, MR; inner_etype=alg.inner_etype)
    Cnew = _c3v_update_C(Rnew, Rmat, V)

    Cnew = _c3v_hermitianize(Cnew)
    Rnew = _c3v_reflect_edge(Rnew)
    Cnew /= ignore_derivatives(() -> norm(Cnew))
    Rnew /= ignore_derivatives(() -> norm(Rnew))
    err = ignore_derivatives(() -> norm(Cnew - C))

    return C3vCTMEnv(Cnew, Rnew), err
end

function c3v_qrctmrg_step(env::C3vTwoSiteCTMEnv, M::AbstractArray, alg::C3vQRCTMRG)
    return c3v_qrctmrg_step(env, (M, M), alg)
end

function c3v_qrctmrg_step(env::C3vTwoSiteCTMEnv, M::Tuple, alg::C3vQRCTMRG)
    MA, MB = M
    envA = _c3v_as_single_env(env, 1)
    envB = _c3v_as_single_env(env, 2)

    envBnew, _ = c3v_qrctmrg_step(envA, (MA, MB), alg)
    envAnew, _ = c3v_qrctmrg_step(envB, (MB, MA), alg)
    err = max(_c3v_corner_spectrum_error(envAnew.C, env.CA),
              _c3v_corner_spectrum_error(envBnew.C, env.CB))

    return _c3v_two_site_env(envAnew, envBnew), err
end

function _c3v_qrctmrg_ad_plain(env::Union{C3vCTMEnv,C3vTwoSiteCTMEnv}, M, alg::C3vQRCTMRG, t)
    local err
    for i in 1:alg.maxiter_ad
        env, err = checkpoint(alg.step_checkpoint, c3v_qrctmrg_step, env, M, alg)
        ignore_derivatives() do
            alg.verbosity >= 3 && i % alg.show_every == 0 &&
                @info @sprintf("C3vQRCTMRG@step: %4d\terr = %.3e\ttime = %.3f sec", i, err, time()-t)
            if i == alg.maxiter_ad && err >= alg.tol && alg.verbosity >= 2
                @warn @sprintf("C3vQRCTMRG cancel@step: %4d\terr = %.3e\ttime = %.3f sec", i, err, time()-t)
            end
        end
    end
    return env, err
end

function leading_boundary(env::Union{C3vCTMEnv,C3vTwoSiteCTMEnv}, M::StructArray, alg::C3vQRCTMRG)
    M = _c3v_site_pair(M)
    t = ignore_derivatives(() -> time())
    local err

    T_orig = _c3v_boundary_eltype(env)
    want_whole = alg.whole_vumps_etype !== nothing && alg.whole_vumps_etype != real(T_orig)
    if want_whole
        W = alg.whole_vumps_etype
        env = _c3v_downcast_env(W, env)
        M = _c3v_downcast_sites(W, M)
    end
    alg_wholemode = alg
    if want_whole
        alg_wholemode = deepcopy(alg)
        alg_wholemode.inner_etype = nothing
    end

    ignore_derivatives(() -> alg.verbosity >= 2 && @info "Start C3v QRCTMRG iteration without AD...")
    ignore_derivatives() do
        for i in 1:alg.maxiter
            env, err = c3v_qrctmrg_step(env, M, alg_wholemode)
            alg.verbosity >= 3 && i % alg.show_every == 0 &&
                ignore_derivatives(() -> @info @sprintf("C3vQRCTMRG@step: %4d\terr = %.3e\ttime = %.3f sec", i, err, time()-t))
            if err < alg.tol && i >= alg.miniter
                alg.verbosity >= 2 &&
                    ignore_derivatives(() -> @info @sprintf("C3vQRCTMRG conv@step: %4d\terr = %.3e\ttime = %.3f sec", i, err, time()-t))
                break
            end
            if i == alg.maxiter
                alg.verbosity >= 2 && ignore_derivatives(() -> @warn @sprintf("C3vQRCTMRG cancel@step: %4d\terr = %.3e\ttime = %.3f sec", i, err, time()-t))
            end
        end
    end

    ignore_derivatives(() -> alg.verbosity >= 2 && @info "Start C3v QRCTMRG iteration with AD...")
    if alg.maxiter_ad > 0 &&
       !want_whole &&
       alg.inner_etype === nothing &&
       alg.inner_etype_final_steps == 0
        env, err = _c3v_qrctmrg_ad_plain(env, M, alg, t)
        return env, err
    end

    alg_ad = alg_wholemode
    alg_ad_coarse = alg
    if want_whole || alg.inner_etype !== nothing
        alg_ad_coarse = deepcopy(alg_wholemode)
        alg_ad_coarse.inner_etype = nothing
    end
    mixed_active = (alg.inner_etype !== nothing) || want_whole
    for i in 1:alg.maxiter_ad
        alg_this_iter = alg_ad
        in_polish = alg.inner_etype_final_steps > 0 &&
                    i > alg.maxiter_ad - alg.inner_etype_final_steps
        if mixed_active && in_polish
            alg_this_iter = alg_ad_coarse
        end
        if want_whole && in_polish && _c3v_boundary_eltype(env) != T_orig
            env = _c3v_downcast_env(real(T_orig), env)
            M = _c3v_downcast_sites(real(T_orig), M)
        end
        env, err = checkpoint(alg.step_checkpoint, c3v_qrctmrg_step, env, M, alg_this_iter)
        alg.verbosity >= 3 && i % alg.show_every == 0 &&
            ignore_derivatives(() -> @info @sprintf("C3vQRCTMRG@step: %4d\terr = %.3e\ttime = %.3f sec", i, err, time()-t))
        if err < alg.tol && i >= alg.miniter_ad
            alg.verbosity >= 2 &&
                ignore_derivatives(() -> @info @sprintf("C3vQRCTMRG conv@step: %4d\terr = %.3e\ttime = %.3f sec", i, err, time()-t))
            break
        end
        if i == alg.maxiter_ad
            alg.verbosity >= 2 && ignore_derivatives(() -> @warn @sprintf("C3vQRCTMRG cancel@step: %4d\terr = %.3e\ttime = %.3f sec", i, err, time()-t))
        end
    end
    if want_whole && _c3v_boundary_eltype(env) != T_orig
        env = _c3v_downcast_env(real(T_orig), env)
    end
    return env, err
end

ObsEnv(env::C3vCTMEnv, M::StructArray, ::C3vQRCTMRG, model=nothing) = env
ObsEnv(env::C3vTwoSiteCTMEnv, M::StructArray, ::C3vQRCTMRG, model=nothing) = env

function update!(env::C3vCTMEnv, envp::C3vCTMEnv)
    env.C .= envp.C
    env.R .= envp.R
    return env
end

function update!(env::C3vTwoSiteCTMEnv, envp::C3vTwoSiteCTMEnv)
    env.CA .= envp.CA
    env.RA .= envp.RA
    env.CB .= envp.CB
    env.RB .= envp.RB
    return env
end

function _c3v_rc_crc(env::C3vCTMEnv)
    @unpack C, R = env
    @tensor RC[i,j,k,m] := R[i,j,k,l] * C[l,m]
    @tensor CRC[i,k,l,m] := C[i,j] * RC[j,k,l,m]
    return RC, CRC
end

function _c3v_L(RC, CRC, M, op)
    @tensor L[a,p,q,m] := conj(RC[i,b,c,a]) * CRC[i,j,k,m] *
        M[j,b,p,x] * op[x,y] * conj(M[k,c,q,y])
    return L
end

_c3v_contract(L1, L2) = sum(L1 .* permutedims(L2, (4, 2, 3, 1)))

function _c3v_one_site(env::C3vCTMEnv, M, op)
    RC, CRC = _c3v_rc_crc(env)
    Id = _arraytype(M)(Matrix{Float64}(I, size(M, 4), size(M, 4)))
    Li = _c3v_L(RC, CRC, M, Id)
    Lo = _c3v_L(RC, CRC, M, op)
    return _c3v_contract(Lo, Li)
end

function _c3v_two_site_bond(env::C3vCTMEnv, ML, MR, terms)
    RC, CRC = _c3v_rc_crc(env)
    IdL = _arraytype(ML)(Matrix{Float64}(I, size(ML, 4), size(ML, 4)))
    IdR = _arraytype(MR)(Matrix{Float64}(I, size(MR, 4), size(MR, 4)))
    LiL = _c3v_L(RC, CRC, ML, IdL)
    LiR = _c3v_L(RC, CRC, MR, IdR)
    n = _c3v_contract(LiL, LiR)
    e = sum(c * _c3v_contract(_c3v_L(RC, CRC, ML, OL),
                              _c3v_L(RC, CRC, MR, OR)) / n
            for (c, OL, OR) in terms)
    return e, n
end

function _c3v_two_site_bond(env::C3vTwoSiteCTMEnv, ML, MR, terms; site::Int=1)
    return _c3v_two_site_bond(_c3v_as_single_env(env, site), ML, MR, terms)
end

function magnetization_value(model, A, env::Union{C3vCTMEnv,C3vTwoSiteCTMEnv}, params)
    A1 = A.data[1]
    atype = _arraytype(A1)
    etype = eltype(A1)
    S = model.S
    Sx = atype(const_Sx(S))
    Sy = atype(const_Sy(S))
    Sz = atype(const_Sz(S))
    Id = atype(Matrix{Float64}(I, size(A1, 4), size(A1, 4)))

    m_dict = Dict{String, Any}()
    Mnorms = Float64[]
    for q in 1:length(A.data)
        Aq = A.data[q]
        envq = _c3v_as_single_env(env, q)
        pos = Tuple(findfirst(==(q), A.pattern))
        key = "$(pos[1]),$(pos[2])"
        n = _c3v_one_site(envq, Aq, Id)
        Mx = _c3v_one_site(envq, Aq, Sx) / n
        My = etype <: Real ? 0.0 : _c3v_one_site(envq, Aq, Sy) / n
        Mz = _c3v_one_site(envq, Aq, Sz) / n
        Mag = [Mx, My, Mz]
        Mnorm = norm(Mag)
        push!(Mnorms, real(Mnorm))
        params.verbosity >= 4 && println("M[$key] = $(Mag)\n|M| = $(Mnorm)")
        m_dict[key] = Dict("Mx" => Mx, "My" => My, "Mz" => Mz, "|M|" => Mnorm)
    end

    Mmean = sum(Mnorms) / length(Mnorms)
    params.verbosity >= 4 && println("|M|_mean = $(Mmean)")
    return Mmean, m_dict
end

function _c3v_edge_transfer(C, R)
    @tensor Cnew[c,r] := conj(R[t,j,k,c]) * C[t,b] * R[b,j,k,r]
    return Cnew
end

function _c3v_transfer_initial(C)
    x0 = similar(C)
    for I in CartesianIndices(x0)
        i, j = Tuple(I)
        x0[I] = convert(eltype(C), sin(0.37 * i + 0.91 * j) + cos(0.13 * i * j))
    end
    return x0
end

function cor_len_value(env::C3vCTMEnv, params, M; method::Symbol=:mps)
    method in (:mps, :channel) ||
        error("cor_len_value: unknown method=$(method) (use :mps or :channel).")
    @unpack C, R = env

    lambdas, _, info = eigsolve(Ci -> _c3v_edge_transfer(Ci, R), _c3v_transfer_initial(C), 8, :LM; maxiter=300, ishermitian=false)
    info.converged == 0 && @warn "cor_len not converged"
    order = sortperm(abs.(lambdas); rev=true)
    lambdas = lambdas[order]
    mu = abs.(lambdas ./ lambdas[1])
    lambda2_index = 2
    for i in 2:length(lambdas)
        if !isapprox(mu[i], 1; rtol=1e-8, atol=1e-12)
            lambda2_index = i
            break
        end
    end

    xi = -1 / log(mu[lambda2_index])
    params.verbosity >= 4 && println("xi = $(xi)")
    return xi
end

function cor_len_value(env::C3vTwoSiteCTMEnv, params, M; method::Symbol=:mps)
    xiA = cor_len_value(_c3v_as_single_env(env, 1), params, M; method)
    xiB = cor_len_value(_c3v_as_single_env(env, 2), params, M; method)
    xi = max(real(xiA), real(xiB))
    params.verbosity >= 4 && println("xi_AB = $(xi)")
    return xi
end

function imag_error(env::Union{C3vCTMEnv,C3vTwoSiteCTMEnv}, A, iSy, params::iPEPSOptimize)
    vals = Float64[]
    for q in 1:length(A.data)
        A1 = A.data[q]
        envq = _c3v_as_single_env(env, q)
        Id = _arraytype(A1)(Matrix{Float64}(I, size(A1, 4), size(A1, 4)))
        n = _c3v_one_site(envq, A1, Id)
        My = _c3v_one_site(envq, A1, iSy)
        push!(vals, real(abs(My / n)))
    end
    return sum(vals) / length(vals)
end

function _c3v_one_site_imag_error(env::C3vCTMEnv, A1, iSy)
    Id = _arraytype(A1)(Matrix{Float64}(I, size(A1, 4), size(A1, 4)))
    n = _c3v_one_site(env, A1, Id)
    My = _c3v_one_site(env, A1, iSy)
    return abs(My / n)
end
