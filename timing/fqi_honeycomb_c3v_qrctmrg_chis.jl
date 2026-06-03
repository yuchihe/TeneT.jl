using TeneT
using LinearAlgebra
using TensorOperations
using KrylovKit
using Printf

using TeneT: qr_for_ad

const FQI_SOURCE = get(ENV, "FQI_FEATURELESS_HONEYCOMB_PATH",
                       raw"C:\Users\yuche\Ising\featureless_honeycomb.jl")
include(FQI_SOURCE)

struct C3vDLEnv{CT<:AbstractMatrix,RT<:AbstractArray{<:Number,3}}
    C::CT
    R::RT
end

function hermitianize_corner(C)
    return (C + adjoint(C)) ./ 2
end

function symmetrize_edge(R)
    chi, Q, chir = size(R)
    D = round(Int, sqrt(Q))
    D^2 == Q || throw(DimensionMismatch("edge middle dimension $Q is not a square double-layer dimension"))
    R4 = reshape(R, chi, D, D, chir)
    return reshape((R4 + permutedims(conj.(R4), (4, 3, 2, 1))) ./ 2, size(R))
end

function parse_chis()
    s = get(ENV, "FQI_C3V_CHIS", "32,64,128")
    return parse.(Int, split(s, ","))
end

function c3v_site_from_fqi(T)
    # T is stored as (phys, a, b, c); C3vQRCTMRG uses (a, b, c, phys).
    return permutedims(T, (2, 3, 4, 1))
end

function fqi_c3v_sublattice_tensors(c1=0.9, c2=0.1; etype=Float64)
    Ts = fqi_site_tensor(c1, c2; etype)
    X = fqi_bond_half(; etype)
    Tu = c3v_site_from_fqi(absorb_bonds(Ts, X))
    Tv = c3v_site_from_fqi(absorb_bonds(Ts, collect(transpose(X))))
    return Tu, Tv
end

function double_layer3(A)
    D = size(A, 1)
    @tensor T[a,d,b,e,c,f] := A[a,b,c,p] * conj(A[d,e,f,p])
    return reshape(T, D^2, D^2, D^2)
end

function single_site_seed_env(A, chi::Int)
    D = size(A, 1)
    C0, R0 = TeneT._c3v_double_layer(A)
    Q = D^2
    n = min(chi, Q)
    C = zeros(eltype(A), chi, chi)
    R = zeros(eltype(A), chi, Q, chi)
    C[1:n, 1:n] .= C0[1:n, 1:n]
    R[1:n, :, 1:n] .= reshape(R0[1:n, :, :, 1:n], n, Q, n)
    C .= hermitianize_corner(C)
    R .= symmetrize_edge(R)
    C ./= norm(C)
    R ./= norm(R)
    return C3vDLEnv(C, R)
end

function c3v_dl_qr(C, R)
    chi, Q = size(R, 1), size(R, 2)
    @tensor CR[i,j,m] := C[i,q] * R[q,j,m]
    Qmat, Rmat = qr_for_ad(reshape(CR, chi * Q, chi))
    return reshape(Qmat, chi, Q, chi), Rmat
end

function align_phase!(Cnew, C)
    overlap = sum(conj.(C) .* Cnew)
    abs(overlap) == 0 && return Cnew
    Cnew ./= overlap / abs(overlap)
    return Cnew
end

function corner_spectrum(C)
    s = svdvals(C)
    n = norm(s)
    n == 0 && return s
    return s ./ n
end

function c3v_dl_step(env::C3vDLEnv, Tu, Tv; conjugate_boundary::Bool=false)
    C = hermitianize_corner(env.C)
    R = symmetrize_edge(env.R)
    V, Rmat = c3v_dl_qr(C, R)
    if conjugate_boundary
        @tensor Rnew[t,u,c] := conj(V[i,b,t]) * R[i,a,l] *
            Tu[a,b,p] * Tv[u,m,p] * V[l,m,c]
        @tensor Cnew[c,r] := conj(Rnew[t,j,c]) * Rmat[t,b] * V[b,j,r]
    else
        @tensor Rnew[t,u,c] := V[i,b,t] * R[i,a,l] *
            Tu[a,b,p] * Tv[u,m,p] * V[l,m,c]
        @tensor Cnew[c,r] := Rnew[t,j,c] * Rmat[t,b] * V[b,j,r]
    end
    Cnew = hermitianize_corner(Cnew)
    Rnew = symmetrize_edge(Rnew)
    Cnew ./= norm(Cnew)
    align_phase!(Cnew, C)
    Rnew ./= norm(Rnew)
    corner_delta = norm(Cnew - C)
    return C3vDLEnv(Cnew, Rnew), corner_delta
end

function leading_boundary_dl(env::C3vDLEnv, Tu, Tv;
                             tol=1e-10, maxiter=5000, miniter=1000,
                             show_every=100, verbosity=2,
                             conjugate_boundary::Bool=false,
                             xi_check_every::Int=100,
                             nev::Int=8,
                             stable_checks::Int=3)
    t0 = time()
    corner_delta = Inf
    xi = NaN
    xi_err = Inf
    ratio_err = Inf
    ratios = Float64[]
    prev_ratios = nothing
    stable_count = 0
    for iter in 1:maxiter
        env, corner_delta = c3v_dl_step(env, Tu, Tv; conjugate_boundary)
        checked_transfer = false
        if iter % xi_check_every == 0 || iter == maxiter
            checked_transfer = true
            spec = transfer_spectrum(env; nev, conjugate_boundary)
            xi = spec.xi
            ratios = spec.mu[2:min(end, nev)]
            if prev_ratios !== nothing
                n = min(length(ratios), length(prev_ratios))
                ratio_err = norm(ratios[1:n] .- prev_ratios[1:n])
                xi_prev = -1 / log(prev_ratios[1])
                xi_err = abs(xi - xi_prev)
            end
            prev_ratios = copy(ratios)
        end
        if verbosity >= 2 && iter % show_every == 0
            println(@sprintf("  step %5d  xi = %.12f  xi_err = %.6e  ratio_err = %.6e  corner_delta = %.6e  time = %.2f s",
                             iter, xi, xi_err, ratio_err, real(corner_delta), time() - t0))
            flush(stdout)
        end
        checked_transfer || continue
        isfinite(xi_err) && isfinite(ratio_err) || continue
        err = max(xi_err, ratio_err)
        stable_count = err < tol ? stable_count + 1 : 0
        if err < tol && iter >= miniter && stable_count >= stable_checks
            verbosity >= 1 && println(@sprintf("  converged at step %d, xi = %.12f, xi_err = %.6e, ratio_err = %.6e, corner_delta = %.6e, time = %.2f s",
                                                iter, xi, xi_err, ratio_err, real(corner_delta), time() - t0))
            return env, err, iter, xi, xi_err, ratio_err
        end
    end
    err = max(xi_err, ratio_err)
    verbosity >= 1 && println(@sprintf("  stopped at maxiter, xi = %.12f, xi_err = %.6e, ratio_err = %.6e, corner_delta = %.6e, time = %.2f s",
                                        xi, xi_err, ratio_err, real(corner_delta), time() - t0))
    return env, err, maxiter, xi, xi_err, ratio_err
end

function edge_transfer(C, R; conjugate_boundary::Bool=false)
    if conjugate_boundary
        @tensor Cnew[c,r] := conj(R[t,j,c]) * C[t,b] * R[b,j,r]
    else
        @tensor Cnew[c,r] := R[t,j,c] * C[t,b] * R[b,j,r]
    end
    return Cnew
end

function transfer_initial(C)
    x0 = similar(C)
    for I in CartesianIndices(x0)
        i, j = Tuple(I)
        x0[I] = convert(eltype(C), sin(0.37 * i + 0.91 * j) + cos(0.13 * i * j))
    end
    return x0
end

function transfer_spectrum(env::C3vDLEnv; nev=8, conjugate_boundary::Bool=false)
    vals, _, info = eigsolve(C -> edge_transfer(C, env.R; conjugate_boundary),
                             transfer_initial(env.C), nev, :LM;
                             maxiter=500, ishermitian=false)
    info.converged == 0 && @warn "edge transfer eigsolve did not converge"
    order = sortperm(abs.(vals); rev=true)
    vals = vals[order]
    lambda0 = vals[1]
    mu = abs.(vals ./ lambda0)
    gap = fill(Inf, length(mu))
    xi_modes = fill(0.0, length(mu))
    for i in eachindex(mu)
        if i == 1
            gap[i] = 0
            xi_modes[i] = Inf
        elseif mu[i] > 0
            gap[i] = -log(mu[i])
            xi_modes[i] = 1 / gap[i]
        end
    end
    lambda2_index = 2
    for k in 2:length(vals)
        if !isapprox(mu[k], 1; rtol=1e-8, atol=1e-12)
            lambda2_index = k
            break
        end
    end
    xi = xi_modes[lambda2_index]
    return (; xi=real(xi), vals, mu, gap, xi_modes, lambda2_index, info)
end

function correlation_length(env::C3vDLEnv; nev=8, conjugate_boundary::Bool=false)
    spec = transfer_spectrum(env; nev, conjugate_boundary)
    return spec.xi, spec.vals
end

function gap_extrapolation(chis, gaps)
    length(chis) >= 3 || return nothing
    best = nothing
    y = collect(Float64.(gaps))
    for b in range(0.1, 6.0; length=5901)
        x = Float64.(chis) .^ (-b)
        A = hcat(ones(length(x)), x)
        coeff = A \ y
        resid = norm(A * coeff - y)
        if best === nothing || resid < best.resid
            best = (; gap_inf=coeff[1], a=coeff[2], b, resid)
        end
    end
    return best
end

function run()
    blas_threads = parse(Int, get(ENV, "FQI_BLAS_THREADS",
                                  get(ENV, "JULIA_NUM_THREADS", "1")))
    BLAS.set_num_threads(blas_threads)

    chis = parse_chis()
    tol = parse(Float64, get(ENV, "FQI_C3V_TOL", "1e-10"))
    maxiter = parse(Int, get(ENV, "FQI_C3V_MAXITER", "5000"))
    miniter = parse(Int, get(ENV, "FQI_C3V_MINITER", "1000"))
    show_every = parse(Int, get(ENV, "FQI_C3V_SHOW_EVERY", "100"))
    xi_check_every = parse(Int, get(ENV, "FQI_C3V_XI_CHECK_EVERY", string(show_every)))
    stable_checks = parse(Int, get(ENV, "FQI_C3V_STABLE_CHECKS", "3"))
    conjugate_boundary = parse(Bool, get(ENV, "FQI_C3V_CONJ_BOUNDARY", "false"))

    Tu_single, Tv_single = fqi_c3v_sublattice_tensors(0.9, 0.1; etype=Float64)
    Tu_single ./= sqrt(norm(Tu_single))
    Tv_single ./= sqrt(norm(Tv_single))
    Tu = double_layer3(Tu_single)
    Tv = double_layer3(Tv_single)

    println("=== FQI honeycomb C3v QRCTMRG contraction ===")
    println("source = $FQI_SOURCE")
    println("D = $(size(Tu_single, 1)), d = $(size(Tu_single, 4)), Q = $(size(Tu, 1))")
    println("chis = $(chis), transfer_tol = $(tol), maxiter = $(maxiter), miniter = $(miniter)")
    println("conjugate_boundary = $(conjugate_boundary)")
    println("xi_check_every = $(xi_check_every), stable_checks = $(stable_checks)")
    println("correlation_length = Qi-style CTMRG transfer spectrum: mu_i=|lambda_i/lambda0|, xi_i=-1/log(mu_i)")
    println("julia_threads = $(Threads.nthreads()), blas_threads = $(BLAS.get_num_threads())")
    flush(stdout)

    result_chis = Int[]
    result_gaps = Float64[]
    result_xis = Float64[]
    for chi in chis
        println("\n--- chi = $chi ---")
        env = single_site_seed_env(Tu_single, chi)
        t0 = time()
        env, err, iter, xi, xi_err, ratio_err = leading_boundary_dl(env, Tu, Tv;
                                                                    tol, maxiter, miniter,
                                                                    show_every, verbosity=2,
                                                                    conjugate_boundary,
                                                                    xi_check_every,
                                                                    stable_checks=stable_checks)
        spec = transfer_spectrum(env; conjugate_boundary)
        cspec = corner_spectrum(env.C)
        rank12 = count(>(1e-12), cspec)
        rank14 = count(>(1e-14), cspec)
        elapsed = time() - t0
        gap = spec.gap[spec.lambda2_index]
        println(@sprintf("RESULT chi=%d iter=%d transfer_err=%.12e xi=%.12f gap=%.12e lambda2_index=%d xi_err=%.12e ratio_err=%.12e corner_rank_1e-12=%d corner_rank_1e-14=%d elapsed=%.3f",
                         chi, iter, real(err), spec.xi, gap, spec.lambda2_index,
                         xi_err, ratio_err, rank12, rank14, elapsed))
        println("corner_spectrum_head = ", join(map(x -> @sprintf("%.12e", x), cspec[1:min(end, 12)]), ", "))
        println("qiyang_lambda_abs = ", join(map(x -> @sprintf("%.12e", abs(x)), spec.vals[1:min(end, 6)]), ", "))
        println("qiyang_mu = ", join(map(x -> @sprintf("%.12e", x), spec.mu[1:min(end, 6)]), ", "))
        println("qiyang_gap = ", join(map(x -> @sprintf("%.12e", x), spec.gap[1:min(end, 6)]), ", "))
        println("qiyang_xi_modes = ", join(map(x -> isfinite(x) ? @sprintf("%.12e", x) : "Inf", spec.xi_modes[1:min(end, 6)]), ", "))
        flush(stdout)
        push!(result_chis, chi)
        push!(result_gaps, gap)
        push!(result_xis, spec.xi)
    end

    fit = gap_extrapolation(result_chis, result_gaps)
    if fit !== nothing
        xi_inf = fit.gap_inf > 0 ? 1 / fit.gap_inf : Inf
        println(@sprintf("GAP_FIT gap_chi = gap_inf + a*chi^(-b): gap_inf=%.12e xi_inf=%.12f a=%.12e b=%.6f resid=%.12e",
                         fit.gap_inf, xi_inf, fit.a, fit.b, fit.resid))
        flush(stdout)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    run()
end
